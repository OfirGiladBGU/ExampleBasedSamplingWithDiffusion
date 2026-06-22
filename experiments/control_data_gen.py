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


def main() -> int:
    # Defaults #
    BASE_CONFIG_PATH = "config/GBN/config.json"
    BASE_CKPT_PATH = "config/GBN/model.ckpt"

    # Model parameters
    ENABLE_GECCO = True
    ENABLE_ADAPTIVE_GATE_INJECTION = True
    EVAL_TIMESTEPS = 1000
    TRUNCATION_RATIO = 0.30
    RESAMPLE_JUMPS = 2
    SMART_INIT_FEATURES = False
    SDF_FEATURES = False
    BATCH_COORDS_FEATURES = False
    # ENABLE_SMART_INIT_JITTER = False
    ENABLE_SMART_INIT_SPLAT_SIGMA = False

    FREEZE_DENOISER = True  # TODO: Add in the other scripts
    # GRID_SIZE = 32
    SMART_INIT_SEED = 42
    SDF_TRUNCATE_PX = 8.0
    # SMART_INIT_JITTER_PX = 0.5
    SMART_INIT_SPLAT_SIGMA_PX = 0.5

    # I/O
    DEVICE = "cuda"
    OVERWRITE = True
    EXPORT_PNG = True
    EXPORT_NPY = True

    TRACK_TIME = False
    TRACK_TIME_FULL = False
    TRACK_PROFILE_TRACE = False

    ############################
    # CONFIGURATION PARAMETERS #
    ############################
    
    # ICONS
    # CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
    # DATA_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/quantitative_advance_metrics"
    # GRID_SIZE = 32

    # Faces Set Sample
    # CONTROL_CKPT = "control_v4/train_outputs_data_celeba_5K_1024_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
    # DATA_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/faces_results_compare"
    # GRID_SIZE = 32

    # ICONS - TIMES - V1
    # CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
    # DATA_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/icons_results_runtimes"
    # GRID_SIZE = 24
    # GRID_SIZE = 32
    # GRID_SIZE = 48


    # ICONS - TIMES - V2
    CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
    DATA_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/icons_results_runtimes"
    GRID_SIZE = 16
    # GRID_SIZE = 24 
    # GRID_SIZE = 32
    # GRID_SIZE = 40
    # GRID_SIZE = 48
    # GRID_SIZE = 56
    # GRID_SIZE = 64
    # GRID_SIZE = 72
    # GRID_SIZE = 80
    # GRID_SIZE = 88
    # GRID_SIZE = 96
    # GRID_SIZE = 104
    # GRID_SIZE = 112
    TRACK_TIME = True
    # TRACK_TIME_FULL = True
    # TRACK_PROFILE_TRACE = True

    parser = argparse.ArgumentParser(
        description="Generate stipple targets using ControlNet V4 inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=Path, default=DATA_PATH,
                        help="Root data folder containing 'source/' subfolder")
    parser.add_argument("--base_config_path", type=str, default=BASE_CONFIG_PATH,
                        help="Path to base diffusion model config")
    parser.add_argument("--base_ckpt_path", type=str, default=BASE_CKPT_PATH,
                        help="Path to base diffusion model checkpoint")
    parser.add_argument("--control_ckpt_path", type=str, default=CONTROL_CKPT,
                        help="Path to ControlNet checkpoint")
    parser.add_argument("--eval_timesteps", type=int, default=EVAL_TIMESTEPS)
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    parser.add_argument("--truncation_ratio", type=float, default=TRUNCATION_RATIO,
                        help="Fraction of full schedule to use (< 1.0 uses Smart Init)")
    parser.add_argument("--enable_gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    parser.add_argument("--enable_adaptive_gate_injection", action=argparse.BooleanOptionalAction, default=ENABLE_ADAPTIVE_GATE_INJECTION)
    parser.add_argument("--smart_init_features", action=argparse.BooleanOptionalAction,
                        default=SMART_INIT_FEATURES)
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
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE,
                        help="Overwrite existing outputs; otherwise skip already processed images")
    parser.add_argument("--export_png", action=argparse.BooleanOptionalAction, default=EXPORT_PNG,
                        help="Export target stipple images as PNGs to target/ subfolder")
    parser.add_argument("--export_npy", action=argparse.BooleanOptionalAction, default=EXPORT_NPY,
                        help="Export target stipple coordinates as NPYs to target_npy/ subfolder")
    parser.add_argument("--track_time", action=argparse.BooleanOptionalAction, default=TRACK_TIME,
                        help="Enable time tracking; saves elapsed time per image to timestamps/ subfolder")
    parser.add_argument("--track_time_full", action=argparse.BooleanOptionalAction, default=TRACK_TIME_FULL,
                        help="Enable full time tracking; saves detailed timing info per image to timestamps/ subfolder")
    parser.add_argument("--profile_trace", action=argparse.BooleanOptionalAction, default=TRACK_PROFILE_TRACE,
                        help="Enable profiling trace; saves Chrome trace JSON to timestamps/ subfolder")
    args = parser.parse_args()

    # NOTE: Build paths
    data_path = args.data_path
    SOURCE_PATH = os.path.join(args.data_path, "source")
    TARGET_PATH = os.path.join(args.data_path, "target")
    JSON_PATH = os.path.join(args.data_path, "prompt.json")
    TARGET_NPY_PATH = os.path.join(args.data_path, "target_npy")
    TIMESTAMPS_PATH = os.path.join(args.data_path, "timestamps")

    print(f"Data path: {data_path}")
    print(f"Source path: {SOURCE_PATH}")
    print(f"Target path: {TARGET_PATH}")
    print(f"JSON path: {JSON_PATH}")
    print(f"Target NPY path: {TARGET_NPY_PATH}")
    print(f"Timesteps path: {TIMESTAMPS_PATH}")

    # Model flags
    truncation_cutoff = max(1, int(args.eval_timesteps * args.truncation_ratio))
    print(f"Grid size (px)                        : {args.grid_size}")
    print(f"GECCO dynamic features enabled        : {args.enable_gecco}")
    print(f"Adaptive gate injection enabled       : {args.enable_adaptive_gate_injection}")
    print(f"Truncation ratio                      : {args.truncation_ratio:.3f}")
    print(f"Truncation cutoff timesteps           : {truncation_cutoff}/{args.eval_timesteps}")
    print(f"Eval resample-jumps                   : {args.resample_jumps}")
    print(f"Smart Init features enabled           : {args.smart_init_features}")
    print(f"SDF features enabled                  : {args.sdf_features}")
    print(f"Batch coords features enabled         : {args.batch_coords_features}")
    # print(f"Min-SNR gamma                         : {args.min_snr_gamma}")
    print(f"SDF truncation (px)                   : {args.sdf_truncate_px}")
    # print(f"Smart Init micro-jitter (train, px)  : {args.smart_init_jitter_px}")
    print(f"Smart Init soft-splat sigma (px)     : {args.smart_init_splat_sigma_px}")
    # print(f"Smart Init jitter enabled            : {args.enable_smart_init_jitter}")
    print(f"Smart Init splat-sigma enabled       : {args.enable_smart_init_splat_sigma}")

    os.makedirs(SOURCE_PATH, exist_ok=True)
    if args.export_png: os.makedirs(TARGET_PATH, exist_ok=True)
    if args.export_npy: os.makedirs(TARGET_NPY_PATH, exist_ok=True)
    if args.track_time: os.makedirs(TIMESTAMPS_PATH, exist_ok=True)

    # Call the public API from sample_control
    # Dataset generation uses OT and writes PNGs to target/ by default.
    run_inference_on_directory(
        base_config_path=args.base_config_path,
        base_ckpt_path=args.base_ckpt_path,
        control_ckpt_path=args.control_ckpt_path,
        grid_size=args.grid_size,
        eval_timesteps=args.eval_timesteps,
        enable_gecco=args.enable_gecco,
        enable_adaptive_gate_injection=args.enable_adaptive_gate_injection,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        batch_coords_features=args.batch_coords_features,
        truncation_ratio=args.truncation_ratio,
        t_start_step=-1,
        smart_init_seed=args.smart_init_seed,
        sdf_truncate_px=args.sdf_truncate_px,
        resample_jumps=args.resample_jumps,
        enable_smart_init_splat_sigma=args.enable_smart_init_splat_sigma,
        smart_init_splat_sigma_px=args.smart_init_splat_sigma_px,
        device=args.device,
        # Paths
        export_png=args.export_png,
        export_npy=args.export_npy,
        track_time=args.track_time,
        track_time_full=args.track_time_full,
        profile_trace=args.profile_trace,
        source_path=Path(SOURCE_PATH),
        target_path=Path(TARGET_PATH),
        target_npy_path=Path(TARGET_NPY_PATH),
        timestamps_path=Path(TIMESTAMPS_PATH),
        json_path=Path(JSON_PATH),
        overwrite=args.overwrite,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
