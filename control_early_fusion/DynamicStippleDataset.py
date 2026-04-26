"""Simplified stipple dataset for control_early_fusion.

Returns ``(high_res_image, offsets)`` per sample — no SDF, no smart_init.
Heavy per-sample preprocessing (scipy distance transforms, rejection sampling)
has been removed so loading speed is comparable to the baseline HDF5 dataset.
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class StippleDataset(Dataset):
    """Dataset yielding ``(high_res_image, gt_offsets)`` pairs.

    Parameters
    ----------
    source_dir : str
        Directory of full-resolution grayscale source images.
    offsets_dir : str
        Directory of pre-computed ``.npy`` offset files (shape ``(2, G, G)``).
        File stems must match the corresponding source image stems.
    filenames : list[str] | None
        If provided, load only these relative paths (used for train/val splits).
    preload_ram : bool
        Load all data to RAM at init time to eliminate per-batch disk I/O.
    """

    VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self, source_dir, offsets_dir, filenames=None, preload_ram=False):
        self.source_dir = source_dir
        self.offsets_dir = offsets_dir
        self.preload_ram = preload_ram
        self._ram_cache = {}

        if filenames is not None:
            self.filenames = list(filenames)
        else:
            source_stems = {}
            for root, _, files in os.walk(source_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() not in self.VALID_EXT:
                        continue
                    rel = os.path.relpath(os.path.join(root, f), source_dir)
                    stem = os.path.splitext(rel)[0]
                    source_stems[stem] = rel

            offset_stems = set()
            for root, _, files in os.walk(offsets_dir):
                for f in files:
                    if not f.endswith(".npy"):
                        continue
                    rel = os.path.relpath(os.path.join(root, f), offsets_dir)
                    offset_stems.add(os.path.splitext(rel)[0])

            self.filenames = sorted(
                source_stems[s] for s in source_stems if s in offset_stems
            )

        if self.preload_ram:
            self._preload()

    # ------------------------------------------------------------------
    def _preload(self):
        from tqdm import tqdm
        print(f"[StippleDataset] Preloading {len(self.filenames)} samples to RAM...")
        for filename in tqdm(self.filenames, desc="Preloading"):
            stem = os.path.splitext(filename)[0]
            img = cv2.imread(
                os.path.join(self.source_dir, filename), cv2.IMREAD_GRAYSCALE
            )
            offsets = np.load(os.path.join(self.offsets_dir, stem + ".npy"))
            self._ram_cache[stem] = {
                "img": img.astype(np.float32) / 255.0,
                "offsets": offsets.astype(np.float32),
            }
        print(f"[StippleDataset] Preload complete — {len(self._ram_cache)} samples in RAM.")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem = os.path.splitext(filename)[0]

        if stem in self._ram_cache:
            entry = self._ram_cache[stem]
            img_np = entry["img"]
            offsets_np = entry["offsets"]
        else:
            img = cv2.imread(
                os.path.join(self.source_dir, filename), cv2.IMREAD_GRAYSCALE
            )
            img_np = img.astype(np.float32) / 255.0
            offsets_np = np.load(
                os.path.join(self.offsets_dir, stem + ".npy")
            ).astype(np.float32)

        high_res = torch.from_numpy(img_np).unsqueeze(0).contiguous()  # (1, H, W)
        offsets = torch.from_numpy(offsets_np).contiguous()             # (2, G, G)
        return high_res, offsets

