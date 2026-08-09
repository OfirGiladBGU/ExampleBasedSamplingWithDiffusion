"""Shared filesystem / point-extraction helpers for the multi-oracle M0.

Kept separate from `m0_run.py` so `gen_oracles.py` and the M1 dataset builder can use the same
stem enumeration and the same point-extraction rules without importing the analysis code.
"""

import os

import numpy as np
from PIL import Image

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def stem_map(directory, exts=VALID_EXT):
    """stem -> full path, recursive. Stems are paths relative to `directory`, extension dropped,
    so the `Icons-50/<class>/<file>` nesting is preserved and stays comparable across oracles."""
    out = {}
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                rel = os.path.relpath(os.path.join(root, f), directory)
                out[os.path.splitext(rel)[0].replace("\\", "/")] = os.path.join(root, f)
    return out


def load_gray01(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def extract_centroids(path, n_points=None, min_area=1, rng_seed=42):
    """Dot centroids from a rendered stipple target, as (N, 2) in [0, 1], x then y.

    Mirrors `train_control.extract_points_from_target` with ONE deliberate difference: it never
    pads with uniform random points. Training pads to force a fixed tensor shape, which is fine
    for a diffusion target but would be fatal here -- injecting uniform noise into a point set
    whose regularity we are about to measure directly manufactures the descriptor difference we
    are testing for. Under-count is reported to the caller instead, and surplus is subsampled.
    """
    from scipy import ndimage

    img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    binary = ((255 - img) > 127).astype(np.uint8)
    labelled, n_labels = ndimage.label(binary)
    if n_labels == 0:
        return np.zeros((0, 2), dtype=np.float64)
    idx = np.arange(1, n_labels + 1)
    if min_area > 1:
        areas = ndimage.sum(binary, labelled, idx)
        idx = idx[areas >= min_area]
        if len(idx) == 0:
            return np.zeros((0, 2), dtype=np.float64)
    cents = ndimage.center_of_mass(binary, labelled, idx)
    h, w = img.shape
    pts = np.array([[cx / w, cy / h] for cy, cx in cents], dtype=np.float64)
    if n_points is not None and len(pts) > n_points:
        rng = np.random.RandomState(rng_seed)
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    return pts


def quantise(pts, res):
    """Snap points to the centres of a `res` x `res` lattice.

    Used by the M0 quantisation control: the dither oracles are lattice-bound by construction, so
    to show that any separation is not merely a lattice artefact we put a continuous oracle on the
    same lattice and re-measure.
    """
    q = (np.floor(np.asarray(pts) * res) + 0.5) / res
    return np.clip(q, 0.0, 1.0 - 1e-9)


def save_points(path, pts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, pts.astype(np.float32))
    os.replace(tmp, path)


def load_points(path):
    return np.load(path).astype(np.float64)


def read_stems_file(path):
    """One stem per line, '#' comments and blanks ignored."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line.replace("\\", "/"))
    return out


def stem_map_for(directory, stems, exts=VALID_EXT):
    """Build stem -> path for a KNOWN stem list without walking the tree.

    `stem_map` does an os.walk, which on the 10k-icon set means ~10k stat calls per directory. Over
    the SSHFS mount that alone took longer than the entire descriptor computation, and M0 needs
    four such directories. Given the stem list up front, the paths are constructible directly, so
    this does one existence probe per stem per extension instead.
    """
    out = {}
    for stem in stems:
        for ext in exts:
            p = os.path.join(directory, stem + ext)
            if os.path.exists(p):
                out[stem] = p
                break
    return out

DEFAULT_WHITE_THRESHOLD = 255.0


def drop_white_area_points(points, gray01, threshold=DEFAULT_WHITE_THRESHOLD, rng=None):
    """Remove points sitting on background and restore the count by duplicating survivors.

    Mirrors `control_v4/train_control.drop_white_area_points`. `threshold` is a SOURCE PIXEL VALUE
    in 0-255 -- a point is dropped when the pixel beneath it is at least that bright -- because that
    is the number you read off an image viewer when choosing it, unlike rho.

    Returns (points, n_dropped), with len() unchanged: the offsets tensor has exactly G*G cells and
    no empty-cell encoding, so the count cannot shrink.

    Duplication rather than resampling from rho: a duplicate is MARKED (nearest-neighbour distance
    exactly 0), so `descriptor_fields.drop_exact_duplicates` removes it and any measurement recovers
    the true statistics. An invented position cannot be undone.

    That last property is what makes offsets and descriptors agree without coordinating their RNG.
    The KEEP MASK is a pure function of (points, gray, threshold) -- no randomness -- and only the
    choice of which survivor to duplicate is random. Descriptors drop exact duplicates before
    measuring, so they see the kept set regardless of that choice, while the offsets carry the
    padding they need. Both therefore describe the same points even though only one has the padding.

    Refuses to act if it would empty the set, so an all-white or mis-thresholded image fails loudly
    downstream instead of silently producing G*G copies of one point.
    """
    rng = rng or np.random.RandomState(42)
    gray255 = np.clip(np.asarray(gray01, dtype=np.float64) * 255.0, 0.0, 255.0)
    h, w = gray255.shape
    pts = np.asarray(points, dtype=np.float64)
    ix = np.clip((pts[:, 0] * w).astype(np.int64), 0, w - 1)
    iy = np.clip((pts[:, 1] * h).astype(np.int64), 0, h - 1)
    keep = gray255[iy, ix] < threshold
    n_dropped = int((~keep).sum())
    if n_dropped == 0 or not keep.any():
        return pts, 0
    kept = pts[keep]
    dup = kept[rng.choice(len(kept), n_dropped, replace=True)]
    return np.vstack([kept, dup]), n_dropped

