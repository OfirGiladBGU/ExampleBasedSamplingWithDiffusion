"""Multi-oracle dataset for descriptor-conditioned training -- Milestone M1 step 3.

One sample is an (image, oracle) PAIR: the same source image appears once per oracle, each time with
that oracle's own ground-truth offsets and its own measured descriptor field. Conditioning is the
MEASURED descriptor field, never a one-hot oracle label -- a one-hot is an arbitrary tag with no
semantics, nothing between [1,0,0] and [0,1,0] means anything, and the model could only ever emit the
trained styles. Measured descriptors make intermediate values meaningful and interpolation a real
request rather than a latent walk.

Composition, not reimplementation
---------------------------------
This wraps one `DynamicStippleDataset` per oracle rather than re-deriving the image-side tensors.
Everything about how rho / SDF / smart-init / offsets become tensors therefore stays byte-identical
to `control_v4`, and any future change there is inherited. This class only adds the descriptor
channel and the (image, oracle) indexing.

The split is BY SOURCE IMAGE, and reproduces control_v4's exactly
------------------------------------------------------------------
`train_control.py` splits with `torch.randperm(n, generator=Generator().manual_seed(42))` over
`sorted()` relative paths, taking the last `int(n * val_split)` as validation. `split_images()`
below performs the identical computation, so an image that is validation for a control_v4 run is
validation here too -- runs stay comparable.

Every oracle of a held-out image goes to validation together. That is the plan's non-negotiable:
descriptor generalisation (does the model respond to unseen descriptor values?) and image
generalisation (does it work on unseen images?) are two independent axes, and if the same image
appeared in train under one oracle and in val under another, the val set would silently be measuring
descriptor generalisation on memorised images.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def list_image_stems(source_dir, require_dir=None, require_ext=".npy"):
    """Sorted relative paths under source_dir, optionally filtered to those present in require_dir.

    Mirrors DynamicStippleDataset's own construction (walk, filter by extension, keep only stems
    that have a matching companion file, `sorted()`), because the split depends on this ordering.
    """
    source_stems = {}
    for root, _, files in os.walk(source_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in VALID_EXT:
                continue
            rel = os.path.relpath(os.path.join(root, f), source_dir)
            source_stems[os.path.splitext(rel)[0]] = rel

    if require_dir is None:
        return sorted(source_stems.values())

    have = set()
    for root, _, files in os.walk(require_dir):
        for f in files:
            if not f.endswith(require_ext):
                continue
            rel = os.path.relpath(os.path.join(root, f), require_dir)
            have.add(os.path.splitext(rel)[0])
    return sorted(source_stems[s] for s in source_stems if s in have)


def split_images(filenames, val_split=0.1, seed=42):
    """(train, val) relative paths, identical to train_control.py's split.

    Reproduced exactly rather than approximated: same randperm, same seed, same "val is the TAIL of
    the permutation" convention. Changing any of those would silently move images across the
    boundary and make a descriptor run non-comparable with the control_v4 baselines.
    """
    n = len(filenames)
    val_len = int(n * val_split)
    val_len = min(max(val_len, 0), max(n - 1, 0))
    train_len = n - val_len
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return ([filenames[i] for i in order[:train_len]],
            [filenames[i] for i in order[train_len:]])


# ---------------------------------------------------------------------------
# descriptor normalisation
# ---------------------------------------------------------------------------

class DescriptorNorm:
    """Applies the side-car normalisation written by precompute_descriptors.py --stage stats."""

    def __init__(self, stats_path):
        with open(stats_path) as fh:
            stats = json.load(fh)
        self.keys = list(stats["keys"])
        self.grid = int(stats.get("grid", 32))
        self.lo = np.array([stats["descriptors"][k]["lo"] for k in self.keys], dtype=np.float32)
        self.hi = np.array([stats["descriptors"][k]["hi"] for k in self.keys], dtype=np.float32)
        self.scale = 1.0 / np.maximum(self.hi - self.lo, 1e-9)

    @property
    def n_channels(self):
        return len(self.keys)

    def __call__(self, stack):
        """(K+1, G, G) raw stack -> (K, G, G) float32 in [0, 1].

        The trailing channel of the stored stack is the `valid` mask: windows holding too few points
        to estimate a descriptor. Those cells are NaN in the raw array. They are filled with 0.5
        (the midpoint of the normalised range) rather than 0, because 0 is a MEANINGFUL request --
        "make this region maximally regular" -- and writing it into cells where nothing was measured
        would train the model to obey a request that was never made. 0.5 is the least-committal
        filler; the mask is returned alongside so a loss can ignore those cells outright.
        """
        raw = np.asarray(stack, dtype=np.float32)
        valid = raw[-1] > 0.5
        d = (raw[: len(self.keys)] - self.lo[:, None, None]) * self.scale[:, None, None]
        d = np.clip(d, 0.0, 1.0)
        d[:, ~valid] = 0.5
        d[~np.isfinite(d)] = 0.5
        return d, valid.astype(np.float32)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class OracleStippleDataset(Dataset):
    """(image, oracle) samples: control_v4's tensors plus a normalised descriptor field.

    `oracle_dirs` maps an oracle name to its offsets directory. `descriptor_root` holds one
    `descriptors_<oracle>/` tree per oracle, as written by precompute_descriptors.py.
    """

    def __init__(self, source_dir, oracle_dirs, descriptor_root, stats_path, filenames,
                 grid_size=32, dataset_factory=None, drop_missing=True, **ds_kwargs):
        # Local import: torch-heavy, and this module is imported by torch-free tooling too.
        # Package path first -- train_control.py imports it as `control_v4.DynamicStippleDataset`,
        # and only that form resolves when the repo root (not control_v4/) is on sys.path.
        try:
            from control_v4.DynamicStippleDataset import DynamicStippleDataset
        except ImportError:
            from DynamicStippleDataset import DynamicStippleDataset

        factory = dataset_factory or DynamicStippleDataset
        self.norm = DescriptorNorm(stats_path)
        self.descriptor_root = descriptor_root
        self.grid_size = grid_size
        self.oracles = sorted(oracle_dirs)

        self.subsets = {}
        self.index = []          # (oracle, subset_idx)
        self.missing = {}
        for m in self.oracles:
            # Keep only images this oracle actually has BOTH offsets and descriptors for. BNOT is
            # short 36 icons from its solver crashes, and a partially generated oracle must not
            # silently shorten the others or shift the (image, oracle) alignment.
            keep = [f for f in filenames if self._has_all(f, oracle_dirs[m], m)] if drop_missing \
                else list(filenames)
            self.missing[m] = len(filenames) - len(keep)
            if not keep:
                continue
            self.subsets[m] = factory(source_dir, oracle_dirs[m], grid_size=grid_size,
                                      filenames=keep, **ds_kwargs)
            self.index.extend((m, i) for i in range(len(self.subsets[m])))

    def _has_all(self, rel_path, offsets_dir, oracle):
        stem = os.path.splitext(rel_path)[0]
        return (os.path.exists(os.path.join(offsets_dir, stem + ".npy")) and
                os.path.exists(os.path.join(self.descriptor_root,
                                            f"descriptors_{oracle}", stem + ".npy")))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        oracle, i = self.index[k]
        sample = self.subsets[oracle][i]
        stem = os.path.splitext(self.subsets[oracle].filenames[i])[0]
        raw = np.load(os.path.join(self.descriptor_root, f"descriptors_{oracle}", stem + ".npy"))
        d, valid = self.norm(raw)
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["descriptor"] = torch.from_numpy(d)
            sample["descriptor_valid"] = torch.from_numpy(valid)[None]
            sample["oracle"] = oracle
            return sample
        # Tuple-style samples: append rather than guess at positions.
        return tuple(sample) + (torch.from_numpy(d), torch.from_numpy(valid)[None], oracle)

    def summary(self):
        lines = [f"OracleStippleDataset: {len(self)} (image, oracle) samples over "
                 f"{len(self.subsets)} oracles, {self.norm.n_channels} descriptor channels "
                 f"{self.norm.keys}"]
        for m in self.oracles:
            n = len(self.subsets[m]) if m in self.subsets else 0
            miss = self.missing.get(m, 0)
            lines.append(f"    {m:10s} {n:6d} images" + (f"   ({miss} skipped: no offsets/descriptors)"
                                                         if miss else ""))
        return "\n".join(lines)
