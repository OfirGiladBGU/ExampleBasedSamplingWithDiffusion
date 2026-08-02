"""Dataset for the WVS<->GBN style-conditioned model (Phase 2).

Each sample is an (image, oracle) pair. WVS and GBN share the SAME source images (same rho), so
for one source we emit two samples: the WVS target offsets with its style value s, and the GBN
target offsets with its s. s is the per-icon normalized norm_nn_cv from precompute_style_s.py.

Returns (paper config: SDF / smart-init off):
    high_res        (1, H, W)   full-res grayscale source (== rho input)
    target_density  (1, G, G)   area-downsampled density
    offsets         (2, G, G)   this oracle's OT offsets (the training target)
    style_s         scalar      continuous style conditioning value
"""

import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class StyleStippleDataset(Dataset):
    def __init__(self, source_dir, oracle_offsets, style_s_json, grid_size=32, filenames=None):
        """
        Parameters
        ----------
        source_dir : str
            Directory of shared full-res grayscale source images.
        oracle_offsets : dict[str, str]
            {oracle_name: processed_offsets_dir}. e.g. {"WVS": ".../icons-50_512_WVS/processed_offsets",
            "GBN": ".../icons-50_512_GBN/processed_offsets"}.
        style_s_json : str
            Path to style_s.json from precompute_style_s.py (holds normalized s per stem/oracle).
        grid_size : int
        filenames : optional explicit list of source rel-paths (for train/val splits).
        """
        self.source_dir = source_dir
        self.grid_size = grid_size
        self.oracle_offsets = dict(oracle_offsets)

        with open(style_s_json) as fh:
            self.style = json.load(fh)["entries"]

        # source stem -> rel path
        src = {}
        if filenames is not None:
            for rel in filenames:
                src[os.path.splitext(rel)[0]] = rel
        else:
            for root, _, files in os.walk(source_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in VALID_EXT:
                        rel = os.path.relpath(os.path.join(root, f), source_dir)
                        src[os.path.splitext(rel)[0]] = rel

        # offsets availability per oracle
        self.off_maps = {}
        for name, odir in self.oracle_offsets.items():
            m = {}
            for root, _, files in os.walk(odir):
                for f in files:
                    if f.endswith(".npy"):
                        stem = os.path.splitext(os.path.relpath(os.path.join(root, f), odir))[0]
                        m[stem] = os.path.join(root, f)
            self.off_maps[name] = m

        # build (stem, oracle) sample list: source + offsets + a finite s must all be present
        self.samples = []
        for stem, rel in sorted(src.items()):
            se = self.style.get(stem)
            if se is None:
                continue
            for name in self.oracle_offsets:
                if stem in self.off_maps[name] and se.get(name) is not None:
                    self.samples.append((stem, rel, name))

        # Unique source rel-paths that produced at least one sample. Used by the training script
        # to split train/val by SOURCE icon (so both oracles of an icon stay in the same split).
        self.filenames = sorted({rel for _, rel, _ in self.samples})

    def __len__(self):
        return len(self.samples)

    def oracle_counts(self):
        counts = {}
        for _, _, name in self.samples:
            counts[name] = counts.get(name, 0) + 1
        return counts

    def __getitem__(self, idx):
        stem, rel, oracle = self.samples[idx]
        img = cv2.imread(os.path.join(self.source_dir, rel), cv2.IMREAD_GRAYSCALE)
        img_np = img.astype(np.float32) / 255.0

        high_res = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        target_density = F.interpolate(high_res, size=(self.grid_size, self.grid_size), mode="area")
        high_res = high_res.squeeze(0).contiguous()
        target_density = target_density.squeeze(0).contiguous()

        offsets = np.load(self.off_maps[oracle][stem])
        offsets = torch.from_numpy(offsets).to(torch.float32).clone().contiguous()

        s = float(self.style[stem][oracle])
        return {
            "high_res": high_res,
            "target_density": target_density,
            "offsets": offsets,
            "style_s": torch.tensor(s, dtype=torch.float32),
            "oracle": oracle,
        }
