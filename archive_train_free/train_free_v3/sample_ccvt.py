#!/usr/bin/env python3
"""CLI for train_free_v3 CCVT-guided diffusion sampling."""

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
    inverse_ot_transform,
    load_target_image,
    save_point_cloud,
    visualize_density_guidance,
    visualize_point_cloud_grid,
)
from train_free_v3.guided_sample_ccvt import sample_with_ccvt_guidance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="train_free_v3: in-loop CCVT/Lloyd guidance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True)
    p.add_argument("--base_ckpt", default="config/GBN/model.ckpt")
    p.add_argument("--config", default="config/GBN/config.json")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--lambda_scale", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.0,
                   help="Optional safety clip for guidance vector (0 disables)")

    p.add_argument("--ccvt_mode", choices=["lloyd", "ccvt", "repulsion"], default="ccvt")
    p.add_argument("--ccvt_grid_size", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--num_steps", type=int, default=1)
    p.add_argument("--repulsion_strength", type=float, default=0.2)
    p.add_argument("--repulsion_radius", type=float, default=0.03)
    p.add_argument("--resample-jumps", type=int, default=0,
                   help="RePaint-style micro-loops per timestep (0 disables)")
    p.add_argument("--jump-length", type=int, default=10,
                   help="Apply resample jumps every N timesteps")

    p.add_argument("--debug_guidance", action="store_true")
    p.add_argument("--debug_every", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="train_free_v3/sample_outputs")
    p.add_argument("--no_visualization", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("train_free_v3: CCVT-guided diffusion")
    print("=" * 80)

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
        high_res, target_density = load_target_image(args.image, grid_size=args.grid_size, device=args.device)
        if args.n_samples > 1:
            target_density = target_density.repeat(args.n_samples, 1, 1, 1)
            high_res = high_res.repeat(args.n_samples, 1, 1, 1)
    except Exception as exc:
        print(f"Failed to load image: {exc}")
        return 1

    try:
        offsets = sample_with_ccvt_guidance(
            model=model,
            target_density=target_density,
            shape=(args.n_samples, 2, args.grid_size, args.grid_size),
            timesteps=args.timesteps,
            lambda_0=args.lambda_scale,
            grad_clip=args.grad_clip,
            device=args.device,
            with_tqdm=True,
            debug_guidance=args.debug_guidance,
            debug_every=args.debug_every,
            ccvt_mode=args.ccvt_mode,
            ccvt_grid_size=args.ccvt_grid_size,
            tau=args.tau,
            num_steps=args.num_steps,
            repulsion_strength=args.repulsion_strength,
            repulsion_radius=args.repulsion_radius,
            resample_jumps=args.resample_jumps,
            jump_length=args.jump_length,
        )
    except Exception as exc:
        print(f"Sampling failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    points = inverse_ot_transform(offsets, grid_size=args.grid_size)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(args.output_dir, f"{args.ccvt_mode}_{ts}")

    save_point_cloud(points, prefix + ".npy")

    if not args.no_visualization:
        visualize_point_cloud_grid(points, output_path=prefix + "_points")
        visualize_density_guidance(points, high_res, output_path=prefix + "_comparison")

    stats = compute_density_statistics(points, grid_size=args.grid_size)
    print("Generation stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("Saved outputs with prefix:")
    print(prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
