"""Test the training-free method on a single (source, target) pair.

Analogous to control_v3/test_overfit.py but with no training loop --
a single guided-diffusion call produces the result immediately.
The output is compared against the ground-truth target using the same
metrics visualisation (grid capacity, spacing quality).

Usage (from project root):
    python train_free_v1/test_single.py --sample-index 0
    python train_free_v1/test_single.py --sample-index 0 --budget 1024 --timesteps 200
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
except Exception:
    plt = None
    HAS_MPL = False

from utils.Config import ParseSampleConfig
from utils.stippling_metrics import visualize_overfit_metrics
from train_free_v1.quadtree import build_and_normalize, visualize_quadtree
from train_free_v1.guided_sampling import guided_sample

# ── paths ────────────────────────────────────────────────────────────
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_taksim"
SOURCE_DIR = os.path.join(DATA_ROOT, "source")
TARGET_DIR = os.path.join(DATA_ROOT, "target")
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"


# ── helpers ──────────────────────────────────────────────────────────

def extract_points_from_image(img_path):
    """Detect all dot centroids in a stippled image -> (N, 2) in [0, 1]."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    inv = 255 - img_np
    binary = (inv > 127).astype(np.uint8)

    from scipy import ndimage
    labelled, n_labels = ndimage.label(binary)
    centroids = ndimage.center_of_mass(binary, labelled, range(1, n_labels + 1))

    h, w = img_np.shape
    pts = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)
    return pts


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1024,
                        help="Total stipple point budget")
    parser.add_argument("--timesteps", type=int, default=200,
                        help="Diffusion reverse steps")
    parser.add_argument("--lambda-scale", type=float, default=50.0,
                        help="Guidance strength multiplier")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Hard clamp on guidance gradient")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Max cells per U-Net forward pass")
    parser.add_argument("--max-points", type=int, default=64,
                        help="Max points per quadtree cell")
    parser.add_argument("--min-cell-pixels", type=int, default=4)
    parser.add_argument("--skip-threshold", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=2,
                        help="Number of independent samples to generate")
    parser.add_argument("--point-size", type=float, default=1.0,
                        help="Scatter point size in output plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── pick the single example ──────────────────────────────────────
    source_files = sorted(os.listdir(SOURCE_DIR))
    if args.sample_index >= len(source_files):
        sys.exit(f"sample-index {args.sample_index} out of range "
                 f"(dataset has {len(source_files)} files)")

    fname = source_files[args.sample_index]
    stem = os.path.splitext(fname)[0]
    source_path = os.path.join(SOURCE_DIR, fname)
    target_path = os.path.join(TARGET_DIR, fname)

    if not os.path.exists(target_path):
        sys.exit(f"Target not found: {target_path}")

    out_dir = os.path.join("train_free_v1", "test_outputs", stem)
    os.makedirs(out_dir, exist_ok=True)

    # ── load images ──────────────────────────────────────────────────
    print("=" * 60)
    print("  Train-Free V1 -- Single Example Test")
    print("=" * 60)
    print(f"  Example:  {fname}")
    print(f"  Source:   {source_path}")
    print(f"  Target:   {target_path}")
    print(f"  Budget:   {args.budget}")
    print(f"  Timesteps:{args.timesteps}")
    print(f"  Lambda:   {args.lambda_scale}")
    print(f"  Samples:  {args.n_samples}")
    print("=" * 60)

    source_np = np.array(Image.open(source_path).convert("L"))
    target_np = np.array(Image.open(target_path).convert("L"))

    Image.fromarray(source_np).save(os.path.join(out_dir, "source.png"))
    Image.fromarray(target_np).save(os.path.join(out_dir, "target.png"))

    # ── extract GT points from target ────────────────────────────────
    gt_points = extract_points_from_image(target_path)
    print(f"\n  GT points detected: {len(gt_points)} "
          f"(budget: {args.budget})")

    # ── build quadtree from source image ─────────────────────────────
    print("\nBuilding quadtree...")
    cells = build_and_normalize(
        source_np,
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

    if K == 0:
        print("No cells generated. Exiting.")
        return

    visualize_quadtree(source_np, cells,
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

    frozen_params = sum(p.numel() for p in denoiser.parameters())
    print(f"  Model: {frozen_params:,} params (frozen)")
    print(f"  Timesteps: {diffusion.num_timesteps}")

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

    # ── generate samples ─────────────────────────────────────────────
    pred_pointsets = []
    for s_idx in range(args.n_samples):
        sample_seed = args.seed + s_idx
        print(f"\n--- Sample {s_idx} (seed={sample_seed}) ---")
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
            seed=sample_seed,
        )

        elapsed = time.time() - t0
        print(f"  {len(points)} points in {elapsed:.1f}s")
        print(f"  Range: x=[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]  "
              f"y=[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")

        pred_pointsets.append(points)
        np.save(os.path.join(out_dir, f"points_sample{s_idx}.npy"), points)

    # ── visualise with the same metrics as control_v3/test_overfit ────
    print("\nGenerating comparison visualisation...")
    vis_path = os.path.join(out_dir, "comparison.png")
    saved = visualize_overfit_metrics(
        source_np, target_np, gt_points,
        pred_pointsets, vis_path, step=None,
        point_size=args.point_size,
    )
    print(f"  Saved: {vis_path}")

    # ── save metrics ─────────────────────────────────────────────────
    metrics = {
        "example": fname,
        "method": "train_free_v1",
        "budget": args.budget,
        "timesteps": args.timesteps,
        "lambda_scale": args.lambda_scale,
        "grad_clip": args.grad_clip,
        "num_cells": K,
        "total_points": [int(len(p)) for p in pred_pointsets],
        "n_samples": args.n_samples,
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
