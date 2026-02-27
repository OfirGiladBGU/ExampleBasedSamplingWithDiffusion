"""Training-free stipple generation via quadtree + guided diffusion.

Produces density-adaptive stipple points from a grayscale image without
any ControlNet training.  The image is split into quadtree cells, each
cell runs the frozen unconditional diffusion model, and boundary
repulsion guidance stitches the cells together.

Usage (from project root):
    python train_free_v1/sample_trainfree.py --image path/to/image.png
    python train_free_v1/sample_trainfree.py \\
        --image path/to/image.png \\
        --budget 1024 --timesteps 200 --lambda-scale 50 --seed 42
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False

from utils.Config import ParseSampleConfig
from train_free_v1.quadtree import build_and_normalize, visualize_quadtree
from train_free_v1.guided_sampling import guided_sample

CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"


def load_image(path: str) -> np.ndarray:
    """Load and return a grayscale image as uint8 HxW array."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def visualize_result(image: np.ndarray, points: np.ndarray,
                     save_path: str, title: str = "") -> None:
    """Plot stipple points over the source image."""
    if not HAS_MPL:
        return

    H, W = image.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Source image")
    axes[0].axis("off")

    axes[1].scatter(points[:, 0] * W, points[:, 1] * H,
                    s=0.3, c="black", edgecolors="none")
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Stipple points ({len(points)})")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray", vmin=0, vmax=255, alpha=0.3)
    axes[2].scatter(points[:, 0] * W, points[:, 1] * H,
                    s=0.3, c="black", edgecolors="none")
    axes[2].set_xlim(0, W)
    axes[2].set_ylim(H, 0)
    axes[2].set_aspect("equal")
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=str, required=True,
                        help="Path to grayscale source image")
    parser.add_argument("--budget", type=int, default=1024,
                        help="Total number of stipple points")
    parser.add_argument("--timesteps", type=int, default=200,
                        help="Diffusion timesteps for sampling")
    parser.add_argument("--lambda-scale", type=float, default=50.0,
                        help="Guidance strength multiplier")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Hard clamp on guidance gradient")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Max cells per U-Net forward pass")
    parser.add_argument("--max-points", type=int, default=64,
                        help="Max points per quadtree cell (grid_size^2)")
    parser.add_argument("--min-cell-pixels", type=int, default=4,
                        help="Minimum cell size in pixels before stopping split")
    parser.add_argument("--skip-threshold", type=float, default=0.5,
                        help="Skip cells with budget below this")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: train_free_v1/outputs/<stem>)")

    args = parser.parse_args()
    device = torch.device(args.device)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = args.output or os.path.join("train_free_v1", "outputs", stem)
    os.makedirs(out_dir, exist_ok=True)

    # ── print run config ─────────────────────────────────────────────
    print("=" * 60)
    print("  Train-Free Stipple Generation V1")
    print("=" * 60)
    print(f"  Image:           {args.image}")
    print(f"  Budget:          {args.budget} points")
    print(f"  Timesteps:       {args.timesteps}")
    print(f"  Lambda scale:    {args.lambda_scale}")
    print(f"  Grad clip:       {args.grad_clip}")
    print(f"  Chunk size:      {args.chunk_size}")
    print(f"  Max pts/cell:    {args.max_points}")
    print(f"  Min cell pixels: {args.min_cell_pixels}")
    print(f"  Skip threshold:  {args.skip_threshold}")
    print(f"  Seed:            {args.seed}")
    print(f"  Device:          {args.device}")
    print(f"  Output:          {out_dir}")
    print("=" * 60)

    # ── load image ───────────────────────────────────────────────────
    image = load_image(args.image)
    H, W = image.shape
    print(f"\nImage loaded: {H}x{W}, "
          f"mean={image.mean():.1f}, min={image.min()}, max={image.max()}")

    # ── build quadtree ───────────────────────────────────────────────
    print("\nBuilding quadtree...")
    cells = build_and_normalize(
        image,
        total_budget=args.budget,
        max_points=args.max_points,
        min_cell_pixels=args.min_cell_pixels,
        skip_threshold=args.skip_threshold,
    )
    K = len(cells)
    total_pts = sum(c.budget for c in cells)
    n_pairs = sum(len(c.neighbors) for c in cells) // 2
    budgets = [c.budget for c in cells]
    widths = [c.width for c in cells]
    print(f"  Cells:           {K}")
    print(f"  Total points:    {total_pts}")
    print(f"  Neighbour pairs: {n_pairs}")
    if K > 0:
        print(f"  Cell widths:     [{min(widths):.4f} .. {max(widths):.4f}]")
        print(f"  Budgets:         [{min(budgets)} .. {max(budgets)}]")
        print(f"  Avg neighbours:  {sum(len(c.neighbors) for c in cells) / K:.1f}")

    if K == 0:
        print("No cells generated (image may be entirely white). Exiting.")
        return

    visualize_quadtree(image, cells,
                       os.path.join(out_dir, "quadtree.png"))
    print(f"  Saved: {out_dir}/quadtree.png")

    # ── load frozen diffusion model ──────────────────────────────────
    print("\nLoading frozen diffusion model...")
    diffusion = ParseSampleConfig(CONFIG_PATH)
    diffusion.load_state_dict(
        torch.load(CKPT_PATH, map_location="cpu")["diffu"])
    diffusion.to(device)
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    denoiser = diffusion.model
    for p in denoiser.parameters():
        p.requires_grad = False
    denoiser.eval()

    print(f"  Model loaded ({sum(p.numel() for p in denoiser.parameters()):,} params, frozen)")
    print(f"  Timesteps: {diffusion.num_timesteps}")

    # ── extract schedule tensors ─────────────────────────────────────
    schedule = dict(
        betas=diffusion.betas,
        sqrt_alphas_cumprod=diffusion.sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=diffusion.sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas_cumprod=diffusion.sqrt_recip_alphas_cumprod,
        sqrt_recipm1_alphas_cumprod=diffusion.sqrt_recipm1_alphas_cumprod,
        posterior_mean_coef1=diffusion.posterior_mean_coef1,
        posterior_mean_coef2=diffusion.posterior_mean_coef2,
        posterior_variance=diffusion.posterior_variance,
        posterior_log_variance_clipped=diffusion.posterior_log_variance_clipped,
    )

    # ── run guided sampling ──────────────────────────────────────────
    print(f"\nRunning guided sampling ({K} cells x {args.timesteps} steps)...")
    t0 = time.time()

    points = guided_sample(
        model=denoiser,
        cells=cells,
        **schedule,
        num_timesteps=diffusion.num_timesteps,
        device=device,
        lambda_scale=args.lambda_scale,
        grad_clip=args.grad_clip,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    elapsed = time.time() - t0
    print(f"\n  Sampling complete: {len(points)} points in {elapsed:.1f}s")
    print(f"  Points range: x=[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]  "
          f"y=[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
    in_bounds = ((points >= 0) & (points <= 1)).all(axis=1).sum()
    print(f"  In-bounds:    {in_bounds}/{len(points)} "
          f"({100 * in_bounds / len(points):.1f}%)")

    # ── save results ─────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "points.npy"), points)
    print(f"\n  Saved points: {out_dir}/points.npy")

    visualize_result(
        image, points,
        os.path.join(out_dir, "result.png"),
        title=f"Train-Free V1 | {len(points)} pts | "
              f"λ={args.lambda_scale} | T={args.timesteps}",
    )
    print(f"  Saved visualisation: {out_dir}/result.png")

    # ── save config ──────────────────────────────────────────────────
    config = {
        "image": args.image,
        "budget": args.budget,
        "timesteps": args.timesteps,
        "lambda_scale": args.lambda_scale,
        "grad_clip": args.grad_clip,
        "max_points": args.max_points,
        "min_cell_pixels": args.min_cell_pixels,
        "skip_threshold": args.skip_threshold,
        "seed": args.seed,
        "num_cells": K,
        "total_points": int(len(points)),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Results in: {out_dir}/")


if __name__ == "__main__":
    main()
