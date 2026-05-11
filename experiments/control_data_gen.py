#!/usr/bin/env python3
"""ControlNet V4 Dataset Generator using the sample_control API.

Reads source images from a data folder and runs ControlNet V4 inference to
generate stipple point distributions. Outputs target images with timing info.

Usage:
    python experiments/control_data_gen.py \\
        --data_path /path/to/data \\
        --control_ckpt control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt \\
        --timesteps 1000 \\
        --track_time
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_v4.sample_control import run_inference_on_directory

# ── Editable Defaults ───────────────────────────────────────────────────

CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT = "config/GBN/model.ckpt"
CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"

DATA_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/quantitative_advance_metrics/source"

# Inference parameters
TIMESTEPS = 1000
GRID_SIZE = 32
TRUNCATION_RATIO = 0.30
RESAMPLE_JUMPS = 2

# Feature flags
ENABLE_GECCO = True
ENABLE_SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SPLAT_SIGMA_PX = 0.5

# I/O
DEVICE = "cuda"
TRACK_TIME = True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate stipple targets using ControlNet V4 inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, default=DATA_PATH,
                        help="Root data folder containing 'source/' subfolder")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                        help="Path to diffusion model config")
    parser.add_argument("--base_ckpt", type=str, default=BASE_CKPT,
                        help="Path to base diffusion model checkpoint")
    parser.add_argument("--control_ckpt", type=str, default=CONTROL_CKPT,
                        help="Path to ControlNet checkpoint")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    parser.add_argument("--truncation_ratio", type=float, default=TRUNCATION_RATIO,
                        help="Fraction of full schedule to use (< 1.0 uses Smart Init)")
    parser.add_argument("--enable_gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    parser.add_argument("--enable_smart_init_features", action=argparse.BooleanOptionalAction,
                        default=ENABLE_SMART_INIT_FEATURES)
    parser.add_argument("--sdf_features", action=argparse.BooleanOptionalAction, default=SDF_FEATURES)
    parser.add_argument("--batch_coords_features", action=argparse.BooleanOptionalAction,
                        default=BATCH_COORDS_FEATURES)
    parser.add_argument("--enable_smart_init_splat_sigma", action=argparse.BooleanOptionalAction,
                        default=ENABLE_SMART_INIT_SPLAT_SIGMA)
    parser.add_argument("--smart_init_seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--smart_init_splat_sigma_px", type=float, default=SMART_INIT_SPLAT_SIGMA_PX)
    parser.add_argument("--sdf_truncate_px", type=float, default=SDF_TRUNCATE_PX)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--resample_jumps", type=int, default=RESAMPLE_JUMPS,
                        help="Number of resample jumps to use during sampling")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False,
                        help="Overwrite existing outputs; otherwise skip already processed images")
    parser.add_argument("--track_time", action=argparse.BooleanOptionalAction, default=TRACK_TIME,
                        help="Enable time tracking; saves elapsed time per image to timestamps/ subfolder")

    args = parser.parse_args()

    # NOTE: Build paths
    data_path = args.data_path
    SOURCE_PATH = os.path.join(args.data_path, "source")
    TARGET_PATH = os.path.join(args.data_path, "target")
    JSON_PATH = os.path.join(args.data_path, "prompt.json")
    TIMESTAMPS_PATH = os.path.join(args.data_path, "timestamps") if args.track_time else None

    print(f"Data path: {data_path}")
    print(f"Source path: {SOURCE_PATH}")
    print(f"Target path: {TARGET_PATH}")
    print(f"JSON path: {JSON_PATH}")
    print(f"Timesteps: {args.timesteps}, Grid: {args.grid_size}x{args.grid_size}")
    print(f"GECCO: {args.enable_gecco}, Smart Init: {args.enable_smart_init_features}")
    print(f"Resample jumps: {args.resample_jumps}")

    os.makedirs(SOURCE_PATH, exist_ok=True)
    os.makedirs(TARGET_PATH, exist_ok=True)
    if TIMESTAMPS_PATH is not None:
        os.makedirs(TIMESTAMPS_PATH, exist_ok=True)

    # Call the public API from sample_control
    run_inference_on_directory(
        input_dir=SOURCE_PATH,
        config_path=args.config,
        base_ckpt=args.base_ckpt,
        control_ckpt=args.control_ckpt,
        grid_size=args.grid_size,
        timesteps=args.timesteps,
        enable_gecco=args.enable_gecco,
        smart_init_features=args.enable_smart_init_features,
        sdf_features=args.sdf_features,
        batch_coords_features=args.batch_coords_features,
        truncation_ratio=args.truncation_ratio,
        t_start_step=-1,
        smart_init_seed=args.smart_init_seed,
        sdf_truncate_px=args.sdf_truncate_px,
        resample_jumps=args.resample_jumps,
        enable_smart_init_splat_sigma=args.enable_smart_init_splat_sigma,
        smart_init_splat_sigma_px=args.smart_init_splat_sigma_px,
        no_ot=False,
        export_png=True,
        export_npy=False,
        export_conditions=False,
        track_time=args.track_time,
        device=args.device,
        target_dir=TARGET_PATH,
        timestamps_dir=TIMESTAMPS_PATH,
        json_path=JSON_PATH,
        overwrite=args.overwrite,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
