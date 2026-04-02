"""Dataset for Dynamic ControlNet V4 training.

Each sample returns five tensors:
  - high_res_tensor    (1, H, W)   full-resolution grayscale source image
  - target_density_map (1, 32, 32) area-downsampled density (avg pooling)
    - high_res_sdf       (1, H, W)   normalized signed distance field
    - target_sdf_map     (1, 32, 32) downsampled SDF
  - offset_tensor      (2, 32, 32) ground-truth OT offsets from .npy files
    - smart_init_grid    (1, 32, 32) cached/derived Smart Init hint map
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from control_v4.conditioning import build_condition_tensors_from_image
from control_v4.smart_init import build_smart_init_from_image


class DynamicStippleDataset(Dataset):
    """Dataset that yields (high_res_image, target_density, high_res_sdf, target_sdf, offsets).

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

    def __init__(
        self,
        source_dir,
        offsets_dir,
        grid_size=32,
        sdf_truncate_px=0.0,
        smart_init_cache_dir=None,
        smart_init_seed=42,
    ):
        self.source_dir = source_dir
        self.offsets_dir = offsets_dir
        self.grid_size = grid_size
        self.sdf_truncate_px = sdf_truncate_px
        self.smart_init_cache_dir = smart_init_cache_dir
        self.smart_init_seed = int(smart_init_seed)
        if self.smart_init_cache_dir:
            os.makedirs(self.smart_init_cache_dir, exist_ok=True)

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

        # Keep only source files that have a matching offsets file.
        self.filenames = sorted(source_stems[stem] for stem in source_stems if stem in offset_stems)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem = os.path.splitext(filename)[0]

        # ── full-resolution grayscale source ─────────────────────────
        img = cv2.imread(
            os.path.join(self.source_dir, filename), cv2.IMREAD_GRAYSCALE
        )
        high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_image(
            img.astype(np.float32) / 255.0,
            self.grid_size,
            sdf_truncate_px=self.sdf_truncate_px,
        )
        high_res = high_res.squeeze(0).contiguous()
        target_density = target_density.squeeze(0).contiguous()
        high_res_sdf = high_res_sdf.squeeze(0).contiguous()
        target_sdf = target_sdf.squeeze(0).contiguous()

        # ── ground-truth offsets ─────────────────────────────────────
        offsets = np.load(os.path.join(self.offsets_dir, stem + ".npy"))
        offsets = torch.from_numpy(offsets).to(torch.float32).clone().contiguous()  # (2, grid_size, grid_size)

        smart_init_grid = None
        smart_init_offsets_np = None
        if self.smart_init_cache_dir:
            smart_path = os.path.join(self.smart_init_cache_dir, stem + ".npy")
            smart_offsets_path = os.path.join(self.smart_init_cache_dir, stem + "_offsets.npy")
            if os.path.exists(smart_path) and os.path.exists(smart_offsets_path):
                smart_init_grid = np.load(smart_path)
                smart_init_offsets_np = np.load(smart_offsets_path)
            else:
                _, smart_init_offsets_np, smart_init_grid = build_smart_init_from_image(
                    img.astype(np.float32) / 255.0,
                    grid_size=self.grid_size,
                    n_points=self.grid_size * self.grid_size,
                    seed=self.smart_init_seed,
                )
                os.makedirs(os.path.dirname(smart_path), exist_ok=True)
                np.save(smart_path, smart_init_grid)
                np.save(smart_offsets_path, smart_init_offsets_np)
        else:
            _, smart_init_offsets_np, smart_init_grid = build_smart_init_from_image(
                img.astype(np.float32) / 255.0,
                grid_size=self.grid_size,
                n_points=self.grid_size * self.grid_size,
                seed=self.smart_init_seed,
            )

        smart_init_grid = torch.from_numpy(smart_init_grid).to(torch.float32).clone().contiguous()
        smart_init_offsets = torch.from_numpy(smart_init_offsets_np).to(torch.float32).clone().contiguous()

        return high_res, target_density, high_res_sdf, target_sdf, offsets, smart_init_grid, smart_init_offsets
