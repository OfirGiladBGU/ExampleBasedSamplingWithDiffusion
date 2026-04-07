#!/usr/bin/env python3
"""CLI entry point for train_free_v4 OT warp sampling."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.Config import ParseSampleConfig
from train_free_v2.utils_guidance import (
    compute_density_statistics,
    save_point_cloud,
    visualize_density_guidance,
    visualize_point_cloud_grid,
)
from train_free_v4.ot_warp_pipeline import run_unconditional_ot_warp_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="train_free_v4: unconditional blue-noise sampling followed by OT warp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to target density image")
    parser.add_argument("--base_ckpt", default="config/GBN/model.ckpt", help="Base diffusion checkpoint")
    parser.add_argument("--config", default="config/GBN/config.json", help="Model config path")
    parser.add_argument("--backend", choices=["none", "cdf"], default="cdf", help="Warp backend")
    parser.add_argument("--n_samples", type=int, default=1, help="Batch size")
    parser.add_argument("--timesteps", type=int, default=1000, help="Reverse diffusion steps")
    parser.add_argument("--model_grid_size", type=int, default=32, help="Model output grid size")
    parser.add_argument("--warp_grid_size", type=int, default=64, help="Warp solver grid size")
    parser.add_argument("--density_mode", choices=["dark", "light"], default="dark", help="Which image intensities attract points")
    parser.add_argument("--density_gamma", type=float, default=1.0, help="Density shaping gamma")
    parser.add_argument("--cdf_eps", type=float, default=1e-8, help="Numerical epsilon for CDF inversion")
    parser.add_argument("--disable_cdf_interpolation", action="store_true", help="Disable subpixel interpolation for debugging")
    parser.add_argument("--lloyd_relax_steps", type=int, default=15, help="Lloyd relaxation steps after CDF warp (0 disables)")
    parser.add_argument("--lloyd_tau", type=float, default=0.005, help="Softmax temperature for Lloyd step (lower = sharper boundaries)")
    parser.add_argument("--lloyd_grid_size", type=int, default=64, help="Lloyd solver grid size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Execution device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="train_free_v4/sample_outputs", help="Output directory")
    parser.add_argument("--no_visualization", action="store_true", help="Disable PNG output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("train_free_v4: Unconditional OT Warp Pipeline")
    print("=" * 80)
    print(f"Image: {args.image}")
    print(f"Backend: {args.backend}")
    print(f"Samples: {args.n_samples}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Model grid: {args.model_grid_size}")
    print(f"Warp grid: {args.warp_grid_size}")
    print(f"Density mode: {args.density_mode}")
    print(f"Density gamma: {args.density_gamma}")
    if args.lloyd_relax_steps > 0:
        print(f"Lloyd relax steps: {args.lloyd_relax_steps}")
        print(f"Lloyd tau: {args.lloyd_tau}")
        print(f"Lloyd grid size: {args.lloyd_grid_size}")
    print(f"Device: {args.device}")
    print()

    try:
        model = ParseSampleConfig(args.config)
        model.load_state_dict(torch.load(args.base_ckpt, map_location=args.device)["diffu"])
        model.to(args.device)
        model.set_num_timesteps(args.timesteps)
        model.eval()
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return 1

    try:
        outputs = run_unconditional_ot_warp_pipeline(
            model=model,
            image_path=args.image,
            batch_size=args.n_samples,
            model_grid_size=args.model_grid_size,
            warp_grid_size=args.warp_grid_size,
            backend=args.backend,
            density_mode=args.density_mode,
            density_gamma=args.density_gamma,
            cdf_eps=args.cdf_eps,
            interpolation=not args.disable_cdf_interpolation,
            lloyd_relax_steps=args.lloyd_relax_steps,
            lloyd_tau=args.lloyd_tau,
            lloyd_grid_size=args.lloyd_grid_size,
            device=args.device,
            with_tqdm=True,
        )
    except Exception as exc:
        print(f"Sampling pipeline failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    uniform_points = outputs["uniform_points"]
    warped_points = outputs["warped_points"]
    density_image = outputs["density_image"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.output_dir, f"{args.backend}_{timestamp}")

    save_point_cloud(uniform_points, base + "_uniform.npy")
    save_point_cloud(warped_points, base + "_warped.npy")

    if not args.no_visualization:
        visualize_point_cloud_grid(uniform_points, output_path=base + "_uniform")
        visualize_point_cloud_grid(warped_points, output_path=base + "_warped")
        visualize_density_guidance(warped_points, density_image, output_path=base + "_comparison")

    uniform_stats = compute_density_statistics(uniform_points, grid_size=args.model_grid_size)
    warped_stats = compute_density_statistics(warped_points, grid_size=args.model_grid_size)

    print("Uniform stats:")
    for key, value in uniform_stats.items():
        print(f"  {key}: {value}")
    print("Warped stats:")
    for key, value in warped_stats.items():
        print(f"  {key}: {value}")

    print("\nSaved outputs with prefix:")
    print(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
