"""Evaluate selected train/valid dataset samples and export a train-style panel.

This debug utility reproduces the exact train/valid split logic used in
``control_v4/train_control.py``:

    all_indices = torch.randperm(N, generator=torch.Generator().manual_seed(42))

You can pick rows from either split by their split-local indices, e.g.:

    --train-samples "[]" --valid-samples "[0,1,2,3]"

The output panel matches the 4-column format used during training:
Condition | GT | Predict | GT Offset Quiver
"""

import argparse
import ast
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.Config import ParseSampleConfig
from control_v4.DynamicControlNet import DynamicControlNet
from control_v4.DynamicStippleDataset import DynamicStippleDataset
from control_v4.train_control import (
    _grid_centers_flat,
    dynamic_collate,
    ensure_offsets_dir,
    offsets_to_coords_gpu,
    render_smart_init_gpu,
    sample_eval_batch,
    save_val_panel,
)


# Editable defaults (copied locally so this script is self-contained)
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"
CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_data_celeba_5K_1024_no_random/checkpoints/dynamic_controlnet_v4_ep2670.pt"

SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_celeba_5K_1024/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_celeba_5K_1024/target"
OFFSETS_DIR = ""

CACHE_DATA_DIR = ""
OUTPUT_DIR = "control_v4/eval_outputs"

GRID_SIZE = 32
VAL_SPLIT = 0.1
SPLIT_SEED = 42
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0

EVAL_TIMESTEPS = 1000
RESAMPLE_JUMPS = 2
TRUNCATION_RATIO = 0.30

DEVICE = "cuda"
ENABLE_GECCO = True
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False
SMART_INIT_SPLAT_SIGMA_PX = 0.5

# Split-local indices to visualize (your requested debug controls)
TRAIN_SAMPLES = []
VALID_SAMPLES = [0, 1, 2, 3, 4, 5, 6, 7]

PANEL_NAME = "eval_panel.png"
META_NAME = "eval_selection.json"


def _parse_index_list(raw_text, arg_name):
    """Parse a Python-list style argument into a list of non-negative ints."""
    try:
        parsed = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"{arg_name} must be a Python list literal, got: {raw_text}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"{arg_name} must be a list, got: {type(parsed).__name__}")

    out = []
    for i, value in enumerate(parsed):
        if not isinstance(value, int):
            raise ValueError(f"{arg_name}[{i}] must be int, got: {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{arg_name}[{i}] must be >= 0, got: {value}")
        out.append(value)
    return out


def _validate_indices(indices, split_size, split_name):
    """Ensure all requested split-local indices are in range."""
    bad = [idx for idx in indices if idx >= split_size]
    if bad:
        raise IndexError(
            f"{split_name} indices out of range for split size {split_size}: {bad}"
        )


def _load_models(args, device):
    """Load diffusion backbone and trained ControlNet checkpoint."""
    if not args.control_ckpt:
        raise ValueError("--control-ckpt is required")
    if not os.path.isfile(args.control_ckpt):
        raise FileNotFoundError(f"Control checkpoint was not found: {args.control_ckpt}")

    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    diffusion.eval()
    denoiser = diffusion.model

    control_net = DynamicControlNet(
        denoiser,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        batch_coords_features=args.batch_coords_features,
    ).to(device)

    state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.load_state_dict(state["control_net"], strict=False)
    control_net.eval()
    return diffusion, denoiser, control_net


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--control-ckpt", default=CONTROL_CKPT, help="Trained control_v4 checkpoint path")

    parser.add_argument("--source", default=SOURCE_DIR)
    parser.add_argument("--target", default=TARGET_DIR)
    parser.add_argument("--offsets", default=OFFSETS_DIR)
    parser.add_argument("--cache-data-dir", default=CACHE_DATA_DIR)
    parser.add_argument("--out", default=OUTPUT_DIR)

    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)

    parser.add_argument("--eval-timesteps", type=int, default=EVAL_TIMESTEPS)
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO)

    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--enable-gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    parser.add_argument(
        "--smart-init-features",
        action=argparse.BooleanOptionalAction,
        default=SMART_INIT_FEATURES,
    )
    parser.add_argument(
        "--sdf-features",
        action=argparse.BooleanOptionalAction,
        default=SDF_FEATURES,
    )
    parser.add_argument(
        "--batch-coords-features",
        action=argparse.BooleanOptionalAction,
        default=BATCH_COORDS_FEATURES,
    )
    parser.add_argument(
        "--enable-smart-init-splat-sigma",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_SMART_INIT_SPLAT_SIGMA,
        help="Re-render smart init grid from smart_init_offsets with Gaussian splat before sampling",
    )
    parser.add_argument(
        "--smart-init-splat-sigma-px",
        type=float,
        default=SMART_INIT_SPLAT_SIGMA_PX,
        help="Gaussian sigma in grid-pixel units for optional smart init soft splatting",
    )

    parser.add_argument("--train-samples", default=str(TRAIN_SAMPLES), help="Python list literal, e.g. [] or [0,2,4]")
    parser.add_argument("--valid-samples", default=str(VALID_SAMPLES), help="Python list literal")
    parser.add_argument("--panel-name", default=PANEL_NAME)
    parser.add_argument("--meta-name", default=META_NAME)
    args = parser.parse_args()

    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("--val-split must be in [0, 1)")
    if not (0.0 < args.truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0, 1]")

    train_samples = _parse_index_list(args.train_samples, "--train-samples")
    valid_samples = _parse_index_list(args.valid_samples, "--valid-samples")
    if len(train_samples) + len(valid_samples) == 0:
        raise ValueError("At least one sample is required across --train-samples and --valid-samples")

    args.offsets = ensure_offsets_dir(args.source, args.target, args.offsets, args.grid_size)
    os.makedirs(args.out, exist_ok=True)

    cache_data_dir = args.cache_data_dir
    if not (args.smart_init_features or args.sdf_features):
        cache_data_dir = None
    elif not cache_data_dir:
        cache_data_dir = os.path.join(args.out, "cache_data")

    base_dataset = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        preload_ram=False,
    )
    if len(base_dataset) == 0:
        raise RuntimeError("Dataset is empty after matching source images and offsets")

    val_len = int(len(base_dataset) * args.val_split)
    val_len = min(max(val_len, 0), max(len(base_dataset) - 1, 0))
    train_len = len(base_dataset) - val_len

    all_indices = torch.randperm(
        len(base_dataset),
        generator=torch.Generator().manual_seed(args.split_seed),
    ).tolist()
    train_indices = all_indices[:train_len]
    val_indices = all_indices[train_len:]

    train_filenames = [base_dataset.filenames[i] for i in train_indices]
    val_filenames = [base_dataset.filenames[i] for i in val_indices]

    train_dataset = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        filenames=train_filenames,
        preload_ram=False,
    )
    val_dataset = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        filenames=val_filenames,
        preload_ram=False,
    )

    _validate_indices(train_samples, len(train_dataset), "train")
    _validate_indices(valid_samples, len(val_dataset), "valid")

    selected = []
    meta_rows = []
    for idx in train_samples:
        selected.append(train_dataset[idx])
        meta_rows.append(
            {
                "split": "train",
                "split_index": idx,
                "filename": train_dataset.filenames[idx],
            }
        )
    for idx in valid_samples:
        selected.append(val_dataset[idx])
        meta_rows.append(
            {
                "split": "valid",
                "split_index": idx,
                "filename": val_dataset.filenames[idx],
            }
        )

    batch = dynamic_collate(selected)
    device = torch.device(args.device)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    diffusion, denoiser, control_net = _load_models(args, device)

    if (
        args.smart_init_features
        and args.enable_smart_init_splat_sigma
        and ("smart_init_offsets" in batch)
    ):
        grid_centers_flat = _grid_centers_flat(args.grid_size, device=device, dtype=torch.float32)
        smart_coords = offsets_to_coords_gpu(batch["smart_init_offsets"], args.grid_size, grid_centers_flat)
        batch["smart_init_grid"] = render_smart_init_gpu(
            smart_coords,
            grid_size=args.grid_size,
            sigma_px=args.smart_init_splat_sigma_px,
            grid_centers_flat=grid_centers_flat,
        )

    pred_raw = sample_eval_batch(
        diffusion,
        denoiser,
        control_net,
        batch,
        device,
        n_samples=batch["high_res"].shape[0],
        timesteps=args.eval_timesteps,
        resample_jumps=args.resample_jumps,
        show_tqdm=True,
        tqdm_desc="eval_dataset sampling",
        truncation_ratio=args.truncation_ratio,
    )

    panel_path = os.path.join(args.out, args.panel_name)
    saved = save_val_panel(
        panel_path,
        batch["high_res"].detach().cpu().numpy(),
        batch["offsets"].detach().cpu().numpy(),
        pred_raw.detach().cpu().numpy(),
        max_samples=len(selected),
    )
    if not saved:
        raise RuntimeError("Panel export failed (matplotlib unavailable or no samples)")

    meta = {
        "source": args.source,
        "offsets": args.offsets,
        "grid_size": args.grid_size,
        "val_split": args.val_split,
        "split_seed": args.split_seed,
        "train_size": len(train_dataset),
        "valid_size": len(val_dataset),
        "selected_rows": meta_rows,
        "panel_path": panel_path,
        "control_ckpt": args.control_ckpt,
    }
    meta_path = os.path.join(args.out, args.meta_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved panel: {panel_path}")
    print(f"Saved metadata: {meta_path}")
    print("Selection order:")
    for i, row in enumerate(meta_rows):
        print(f"  row {i}: {row['split']}[{row['split_index']}] -> {row['filename']}")


if __name__ == "__main__":
    main()
