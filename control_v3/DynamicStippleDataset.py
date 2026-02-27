"""Dataset for Dynamic ControlNet V3 training.

Each sample returns three tensors:
  - high_res_tensor    (1, H, W)   full-resolution grayscale source image
  - target_density_map (1, 32, 32) area-downsampled density (avg pooling)
  - offset_tensor      (2, 32, 32) ground-truth OT offsets from .npy files
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DynamicStippleDataset(Dataset):
    """Dataset that yields (high_res_image, target_density, offsets).

    Parameters
    ----------
    source_dir : str
        Directory of full-resolution grayscale source images.
    offsets_dir : str
        Directory of pre-computed ``.npy`` offset files (shape ``(2, G, G)``).
        Filenames must share the same stem as the source images.
    grid_size : int
        Spatial resolution of the offset grid (default 32).
    """

    VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self, source_dir, offsets_dir, grid_size=32):
        self.source_dir = source_dir
        self.offsets_dir = offsets_dir
        self.grid_size = grid_size

        self.filenames = sorted(
            f for f in os.listdir(source_dir)
            if os.path.splitext(f)[1].lower() in self.VALID_EXT
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem = os.path.splitext(filename)[0]

        # ── full-resolution grayscale source ─────────────────────────
        img = cv2.imread(
            os.path.join(self.source_dir, filename), cv2.IMREAD_GRAYSCALE
        )
        high_res = torch.from_numpy(img).float() / 255.0
        high_res = high_res.unsqueeze(0)  # (1, H, W)

        # ── target density map via area interpolation ────────────────
        target_density = F.interpolate(
            high_res.unsqueeze(0),
            size=(self.grid_size, self.grid_size),
            mode="area",
        ).squeeze(0)  # (1, grid_size, grid_size)

        # ── ground-truth offsets ─────────────────────────────────────
        offsets = np.load(os.path.join(self.offsets_dir, stem + ".npy"))
        offsets = torch.from_numpy(offsets).float()  # (2, grid_size, grid_size)

        return high_res, target_density, offsets
