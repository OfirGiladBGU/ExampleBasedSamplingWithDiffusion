"""Dataset for control_t2i training.

Each sample returns three tensors:
  - high_res_tensor    (1, H, W)   full-resolution grayscale source image in [0, 1]
  - target_density_map (1, G, G)   area-downsampled density (avg pooling)
  - offset_tensor      (2, G, G)   ground-truth OT offsets from .npy files
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DynamicStippleDataset(Dataset):
    """Dataset yielding (high_res_image, target_density, offsets).

    Parameters
    ----------
    source_dir : str
        Directory of full-resolution grayscale source images.
    offsets_dir : str
        Directory of pre-computed ``.npy`` offset files (shape ``(2, G, G)``).
        Filenames must share the same stem as the source images.
    grid_size : int
        Spatial resolution of the offset grid (default 32).
    filenames : list[str] | None
        If provided, use exactly these relative filenames instead of scanning source_dir.
    """

    VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        source_dir,
        offsets_dir,
        grid_size=32,
        filenames=None,
        # Legacy kwargs silently ignored so callers don't need updates:
        sdf_truncate_px=None,
        smart_init_cache_dir=None,
        smart_init_seed=None,
        preload_ram=False,
    ):
        self.source_dir = source_dir
        self.offsets_dir = offsets_dir
        self.grid_size = grid_size

        if filenames is not None:
            self.filenames = list(filenames)
        else:
            source_stems = {}
            for root, _, files in os.walk(source_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() not in self.VALID_EXT:
                        continue
                    rel_path = os.path.relpath(os.path.join(root, f), source_dir)
                    stem = os.path.splitext(rel_path)[0]
                    source_stems[stem] = rel_path

            offset_stems = set()
            for root, _, files in os.walk(offsets_dir):
                for f in files:
                    if not f.endswith(".npy"):
                        continue
                    rel_path = os.path.relpath(os.path.join(root, f), offsets_dir)
                    offset_stems.add(os.path.splitext(rel_path)[0])

            self.filenames = sorted(
                source_stems[stem] for stem in source_stems if stem in offset_stems
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem = os.path.splitext(filename)[0]

        # ── full-resolution grayscale source ─────────────────────────
        img = cv2.imread(os.path.join(self.source_dir, filename), cv2.IMREAD_GRAYSCALE)
        img_np = img.astype(np.float32) / 255.0
        high_res = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W)

        # ── low-res density map (free — just avg pooling) ─────────────
        target_density = F.interpolate(
            high_res.unsqueeze(0), size=(self.grid_size, self.grid_size), mode="area"
        ).squeeze(0)  # (1, G, G)

        # ── ground-truth offsets ─────────────────────────────────────
        offsets = torch.from_numpy(
            np.load(os.path.join(self.offsets_dir, stem + ".npy"))
        ).float()  # (2, G, G)

        return (
            high_res.contiguous(),
            target_density.contiguous(),
            offsets.contiguous(),
        )

