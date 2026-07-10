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
import torch.nn.functional as F
from torch.utils.data import Dataset

from control_fm.conditioning import (
    build_condition_tensors_from_cached_sdf,
    compute_raw_sdf,
)
from control_fm.smart_init import (
    build_smart_init_from_image,
)


class DynamicStippleDataset(Dataset):
    """Dataset that yields a dict of tensors for the enabled feature set.

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
        cache_data_dir=None,
        smart_init_seed=42,
        smart_init_features=True,
        sdf_features=True,
        filenames=None,
        preload_ram=False,
        provide_smart_init_source=False,
    ):
        self.source_dir = source_dir
        self.offsets_dir = offsets_dir
        self.grid_size = grid_size
        self.sdf_truncate_px = sdf_truncate_px
        self.smart_init_features = bool(smart_init_features)
        self.provide_smart_init_source = bool(provide_smart_init_source)
        # Smart-init offsets/grid are loaded when used as a conditioning channel
        # (smart_init_features) OR as the Flow-Matching coupling source
        # (provide_smart_init_source). Decoupling lets the coupling use the smart-init
        # as the ODE source without also feeding it as a conditioning hint.
        self._load_smart_init = self.smart_init_features or self.provide_smart_init_source
        self.sdf_features = bool(sdf_features)
        self.cache_data_dir = cache_data_dir if (self._load_smart_init or self.sdf_features) else None
        self.smart_init_seed = int(smart_init_seed)
        self.preload_ram = preload_ram
        self._ram_cache = {}  # stem -> dict with cached numpy arrays

        if self.cache_data_dir:
            os.makedirs(self.cache_data_dir, exist_ok=True)

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

            # Keep only source files that have a matching offsets file.
            self.filenames = sorted(source_stems[stem] for stem in source_stems if stem in offset_stems)

        # Preload everything to RAM if requested (eliminates all disk I/O per batch)
        if self.preload_ram and self.cache_data_dir:
            self._preload_to_ram()

    def _smart_init_paths(self, stem):
        """Cache paths for smart-init grid/offsets.

        GT offsets are NOT here -- they live in offsets_dir/{stem}.npy. This cache only
        holds smart-init artifacts:
            grid    : {stem}_smartinit_grid.npy
            offsets : {stem}_smartinit_offsets.npy
        """
        d = self.cache_data_dir
        grid = os.path.join(d, stem + "_smartinit_grid.npy")
        offs = os.path.join(d, stem + "_smartinit_offsets.npy")
        return grid, offs

    def _preload_to_ram(self):
        """Load all cached data (optional SDF / smart init / offsets / images) into RAM."""
        from tqdm import tqdm
        print(f"[DynamicStippleDataset] Preloading {len(self.filenames)} samples to RAM...")
        for filename in tqdm(self.filenames, desc="Preloading"):
            stem = os.path.splitext(filename)[0]
            cache_entry = {}

            # Load source image
            img = cv2.imread(os.path.join(self.source_dir, filename), cv2.IMREAD_GRAYSCALE)
            cache_entry["img"] = img.astype(np.float32) / 255.0

            if self.sdf_features:
                # Load or compute raw SDF
                sdf_path = os.path.join(self.cache_data_dir, stem + "_sdf_raw.npy")
                if os.path.exists(sdf_path):
                    cache_entry["raw_sdf"] = np.load(sdf_path)
                else:
                    raw_sdf = compute_raw_sdf(cache_entry["img"])
                    os.makedirs(os.path.dirname(sdf_path), exist_ok=True)
                    np.save(sdf_path, raw_sdf)
                    cache_entry["raw_sdf"] = raw_sdf

            # Load offsets
            cache_entry["offsets"] = np.load(os.path.join(self.offsets_dir, stem + ".npy"))

            if self._load_smart_init:
                # Load or compute smart init
                smart_path, smart_offsets_path = self._smart_init_paths(stem)
                if os.path.exists(smart_path) and os.path.exists(smart_offsets_path):
                    cache_entry["smart_init_grid"] = np.load(smart_path)
                    cache_entry["smart_init_offsets"] = np.load(smart_offsets_path)
                else:
                    _, smart_offsets_np, smart_grid = build_smart_init_from_image(
                        cache_entry["img"],
                        grid_size=self.grid_size,
                        n_points=self.grid_size * self.grid_size,
                        seed=self.smart_init_seed,
                    )
                    os.makedirs(os.path.dirname(smart_path), exist_ok=True)
                    if not os.path.exists(smart_path):
                        np.save(smart_path, smart_grid)
                    if not os.path.exists(smart_offsets_path):
                        np.save(smart_offsets_path, smart_offsets_np)
                    cache_entry["smart_init_grid"] = smart_grid
                    cache_entry["smart_init_offsets"] = smart_offsets_np

            self._ram_cache[stem] = cache_entry
        print(f"[DynamicStippleDataset] Preload complete. RAM cache size: {len(self._ram_cache)} samples.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem = os.path.splitext(filename)[0]

        # ═══════════════════════════════════════════════════════════════
        # FAST PATH: RAM preload (zero disk I/O)
        # ═══════════════════════════════════════════════════════════════
        if stem in self._ram_cache:
            cache = self._ram_cache[stem]
            img_np = cache["img"]
            offsets_np = cache["offsets"]

            if self.sdf_features:
                raw_sdf = cache["raw_sdf"]
                high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_cached_sdf(
                    img_np, raw_sdf, self.grid_size, sdf_truncate_px=self.sdf_truncate_px
                )
                high_res_sdf = high_res_sdf.squeeze(0).contiguous()
                target_sdf = target_sdf.squeeze(0).contiguous()
            else:
                # Fast path when SDF features are disabled: avoid any distance-transform work.
                high_res = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
                target_density = F.interpolate(high_res, size=(self.grid_size, self.grid_size), mode="area")
                high_res_sdf = None
                target_sdf = None

            high_res = high_res.squeeze(0).contiguous()
            target_density = target_density.squeeze(0).contiguous()
            offsets = torch.from_numpy(offsets_np).to(torch.float32).clone().contiguous()

            sample = {
                "high_res": high_res,
                "target_density": target_density,
                "offsets": offsets,
            }
            if self.sdf_features:
                sample["high_res_sdf"] = high_res_sdf
                sample["target_sdf"] = target_sdf
            if self._load_smart_init:
                smart_init_grid_np = cache["smart_init_grid"]
                smart_init_offsets_np = cache["smart_init_offsets"]
                sample["smart_init_grid"] = torch.from_numpy(smart_init_grid_np).to(torch.float32).clone().contiguous()
                sample["smart_init_offsets"] = torch.from_numpy(smart_init_offsets_np).to(torch.float32).clone().contiguous()

            return sample

        # ═══════════════════════════════════════════════════════════════
        # DISK PATH: Load image, use cached SDF if available
        # ═══════════════════════════════════════════════════════════════

        # ── full-resolution grayscale source ─────────────────────────
        img = cv2.imread(os.path.join(self.source_dir, filename), cv2.IMREAD_GRAYSCALE)
        img_np = img.astype(np.float32) / 255.0

        if self.sdf_features:
            # ── SDF with disk caching (avoids scipy.ndimage per epoch) ───
            if self.cache_data_dir:
                sdf_path = os.path.join(self.cache_data_dir, stem + "_sdf_raw.npy")
                if os.path.exists(sdf_path):
                    # FAST: load pre-computed raw SDF
                    raw_sdf = np.load(sdf_path)
                else:
                    # SLOW: compute and cache raw SDF
                    raw_sdf = compute_raw_sdf(img_np)
                    os.makedirs(os.path.dirname(sdf_path), exist_ok=True)
                    np.save(sdf_path, raw_sdf)

                high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_cached_sdf(
                    img_np, raw_sdf, self.grid_size, sdf_truncate_px=self.sdf_truncate_px
                )
            else:
                # No cache dir: compute SDF every time (slow path)
                from control_fm.conditioning import build_condition_tensors_from_image
                high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_image(
                    img_np, self.grid_size, sdf_truncate_px=self.sdf_truncate_px
                )

            high_res_sdf = high_res_sdf.squeeze(0).contiguous()
            target_sdf = target_sdf.squeeze(0).contiguous()
        else:
            # Fast path when SDF features are disabled: avoid any distance-transform work.
            high_res = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
            target_density = F.interpolate(high_res, size=(self.grid_size, self.grid_size), mode="area")
            high_res_sdf = None
            target_sdf = None

        high_res = high_res.squeeze(0).contiguous()
        target_density = target_density.squeeze(0).contiguous()

        # ── ground-truth offsets ─────────────────────────────────────
        offsets = np.load(os.path.join(self.offsets_dir, stem + ".npy"))
        offsets = torch.from_numpy(offsets).to(torch.float32).clone().contiguous()

        sample = {
            "high_res": high_res,
            "target_density": target_density,
            "offsets": offsets,
        }

        if self.sdf_features:
            sample["high_res_sdf"] = high_res_sdf
            sample["target_sdf"] = target_sdf

        if self._load_smart_init:
            # ── Smart Init with disk caching ─────────────────────────────
            smart_init_grid = None
            smart_init_offsets_np = None
            if self.cache_data_dir:
                smart_path, smart_offsets_path = self._smart_init_paths(stem)
                if os.path.exists(smart_path) and os.path.exists(smart_offsets_path):
                    smart_init_grid = np.load(smart_path)
                    smart_init_offsets_np = np.load(smart_offsets_path)
                else:
                    _, smart_init_offsets_np, smart_init_grid = build_smart_init_from_image(
                        img_np,
                        grid_size=self.grid_size,
                        n_points=self.grid_size * self.grid_size,
                        seed=self.smart_init_seed,
                    )
                    os.makedirs(os.path.dirname(smart_path), exist_ok=True)
                    if not os.path.exists(smart_path):
                        np.save(smart_path, smart_init_grid)
                    if not os.path.exists(smart_offsets_path):
                        np.save(smart_offsets_path, smart_init_offsets_np)
            else:
                _, smart_init_offsets_np, smart_init_grid = build_smart_init_from_image(
                    img_np,
                    grid_size=self.grid_size,
                    n_points=self.grid_size * self.grid_size,
                    seed=self.smart_init_seed,
                )

            sample["smart_init_grid"] = torch.from_numpy(smart_init_grid).to(torch.float32).clone().contiguous()
            sample["smart_init_offsets"] = torch.from_numpy(smart_init_offsets_np).to(torch.float32).clone().contiguous()

        return sample
