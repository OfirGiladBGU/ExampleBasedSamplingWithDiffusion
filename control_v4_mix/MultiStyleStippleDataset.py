"""Dataset for the multi-oracle style-conditioned model (WVS / GBN / DITHER / ...).

Generalizes StyleStippleDataset from a scalar s to a K-dim ONE-HOT style vector, where K = number
of oracles. Each oracle shares the SAME source images (same rho); a sample from oracle k carries
the one-hot vector e_k (e.g. WVS=[1,0,0], GBN=[0,1,0], DITHER=[0,0,1]). Training only ever sees the
simplex VERTICES; interpolation to convex combinations like [0.5,0.5,0] is the (unsupervised) hope,
exactly analogous to the scalar s=0/1 -> s=0.5 case.

Returns (paper config: SDF / smart-init off):
    high_res        (1, H, W)   shared source (== rho input)
    target_density  (1, G, G)   area-downsampled density
    offsets         (2, G, G)   this oracle's OT offsets (the training target)
    style_vec       (K,)        one-hot over oracles
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class MultiStyleStippleDataset(Dataset):
    def __init__(self, source_dir, oracle_names, oracle_offsets, grid_size=32, filenames=None):
        """
        oracle_names   : ordered list, e.g. ["WVS", "GBN", "DITHER"] -> fixes the one-hot index.
        oracle_offsets : {name: processed_offsets_dir} for each oracle (shared source).
        """
        self.source_dir = source_dir
        self.grid_size = grid_size
        self.oracle_names = list(oracle_names)
        self.K = len(self.oracle_names)
        self.oracle_offsets = dict(oracle_offsets)

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

        # per-oracle offset availability
        self.off_maps = {}
        for name in self.oracle_names:
            m = {}
            odir = self.oracle_offsets[name]
            for root, _, files in os.walk(odir):
                for f in files:
                    if f.endswith(".npy"):
                        stem = os.path.splitext(os.path.relpath(os.path.join(root, f), odir))[0]
                        m[stem] = os.path.join(root, f)
            self.off_maps[name] = m

        # (stem, rel, oracle_name, oracle_index) samples
        self.samples = []
        for stem, rel in sorted(src.items()):
            for k, name in enumerate(self.oracle_names):
                if stem in self.off_maps[name]:
                    self.samples.append((stem, rel, name, k))

        self.filenames = sorted({rel for _, rel, _, _ in self.samples})

    def __len__(self):
        return len(self.samples)

    def oracle_counts(self):
        counts = {}
        for _, _, name, _ in self.samples:
            counts[name] = counts.get(name, 0) + 1
        return counts

    def __getitem__(self, idx):
        stem, rel, oracle, k = self.samples[idx]
        img = cv2.imread(os.path.join(self.source_dir, rel), cv2.IMREAD_GRAYSCALE)
        img_np = img.astype(np.float32) / 255.0

        high_res = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        target_density = F.interpolate(high_res, size=(self.grid_size, self.grid_size), mode="area")
        high_res = high_res.squeeze(0).contiguous()
        target_density = target_density.squeeze(0).contiguous()

        offsets = np.load(self.off_maps[oracle][stem])
        offsets = torch.from_numpy(offsets).to(torch.float32).clone().contiguous()

        style_vec = torch.zeros(self.K, dtype=torch.float32)
        style_vec[k] = 1.0
        return {
            "high_res": high_res,
            "target_density": target_density,
            "offsets": offsets,
            "style_vec": style_vec,
            "oracle": oracle,
        }
