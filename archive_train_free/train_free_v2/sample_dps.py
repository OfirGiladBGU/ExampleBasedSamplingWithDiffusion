#!/usr/bin/env python3
"""
train_free_v2: Sinkhorn-Guided Diffusion Posterior Sampling

Zero-shot point cloud generation guided by target density images.
No training required—pure inference-time guidance using optimal transport.

Usage:
    python sample_dps.py \
        --image path/to/target.png \
        --base_ckpt path/to/model.ckpt \
        --config path/to/config.json \
        --output_dir ./outputs

Example (from repo root):
    cd ExampleBasedSamplingWithDiffusion
    python train_free_v2/sample_dps.py \
        --image train_free_v2/examples/star.png \
        --base_ckpt config/GBN/model.ckpt \
        --config config/GBN/config.json \
        --lambda_scale 1.0 \
        --n_samples 1 \
        --timesteps 1000 \
        --output_dir train_free_v2/sample_outputs
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.Config import ParseSampleConfig
from train_free_v2.guided_sample_dps import sample_with_sinkhorn_guidance
from train_free_v2.sinkhorn_loss import SinkhornDensityLoss
from train_free_v2.utils_guidance import (
    load_target_image,
    inverse_ot_transform,
    save_point_cloud,
    visualize_point_cloud_grid,
    visualize_density_guidance,
    compute_density_statistics
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sinkhorn-guided diffusion sampling for point clouds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to target density image (PNG/JPG, grayscale)'
    )
    
    # Model arguments
    parser.add_argument(
        '--base_ckpt',
        type=str,
        default='config/GBN/model.ckpt',
        help='Path to base diffusion model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/GBN/config.json',
        help='Path to model config file'
    )
    
    # Sampling arguments
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help='Number of point clouds to generate (batch size)'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000,
        help='Number of reverse diffusion steps (1000 recommended, <1000 faster but less quality)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Guidance arguments
    parser.add_argument(
        '--lambda_scale',
        type=float,
        default=1.0,
        help='Guidance strength scaling factor (λ₀). Higher = stronger density matching'
    )
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=1.0,
        help='Gradient clipping magnitude (prevent explosions)'
    )
    parser.add_argument(
        '--debug_guidance',
        action='store_true',
        help='Print guidance diagnostics during sampling (nudge vs U-Net step)'
    )
    parser.add_argument(
        '--debug_every',
        type=int,
        default=100,
        help='Print diagnostics every N timesteps when --debug_guidance is set'
    )
    parser.add_argument(
        '--blur',
        type=float,
        default=0.05,
        help='Sinkhorn entropy regularization (blur parameter)'
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        default=32,
        help='Internal grid resolution for Sinkhorn (must match model output)'
    )
    parser.add_argument(
        '--sinkhorn_grid_size',
        type=int,
        default=64,
        help='Sinkhorn density grid resolution (can be higher than model grid for richer guidance)'
    )
    parser.add_argument(
        '--density_mode',
        type=str,
        default='dark',
        choices=['dark', 'light'],
        help="Density polarity: 'dark' means darker pixels attract points, 'light' means brighter attract"
    )
    parser.add_argument(
        '--density_gamma',
        type=float,
        default=2.0,
        help='Density gamma shaping (>1 sharper hotspots, <1 smoother)'
    )
    parser.add_argument(
        '--no_contrast_stretch',
        action='store_true',
        help='Disable per-image contrast stretch in Sinkhorn density preprocessing'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='train_free_v2/sample_outputs',
        help='Directory to save results'
    )
    parser.add_argument(
        '--save_npy',
        action='store_true',
        default=True,
        help='Save point clouds as .npy files'
    )
    parser.add_argument(
        '--save_visualization',
        action='store_true',
        default=True,
        help='Generate and save visualization images'
    )
    parser.add_argument(
        '--save_comparison',
        action='store_true',
        default=True,
        help='Save side-by-side comparison: target image vs generated points'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to run on ('cuda' or 'cpu')"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("train_free_v2: Sinkhorn-Guided Diffusion Posterior Sampling")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Image: {args.image}")
    print(f"  Base model: {args.base_ckpt}")
    print(f"  Config: {args.config}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Lambda scale: {args.lambda_scale}")
    print(f"  Device: {args.device}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")
    
    # ===== LOAD IMAGE =====
    print("Loading target density image...")
    try:
        high_res_image, target_density = load_target_image(
            args.image,
            grid_size=args.grid_size,
            device=args.device
        )
        print(f"✓ Image loaded: {high_res_image.shape} (high-res), {target_density.shape} (downsampled)")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return 1
    
    # ===== LOAD MODEL =====
    print("\nLoading base diffusion model...")
    try:
        model = ParseSampleConfig(args.config)
        model.load_state_dict(torch.load(args.base_ckpt, map_location=args.device)["diffu"])
        model.to(args.device)
        model.set_num_timesteps(args.timesteps)
        model.eval()
        print(f"✓ Model loaded: {args.timesteps} timesteps")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return 1
    
    # ===== CREATE SINKHORN LOSS =====
    print("\nInitializing Sinkhorn loss function...")
    try:
        sinkhorn_loss_fn = SinkhornDensityLoss(
            blur=args.blur,
            grid_size=args.sinkhorn_grid_size,
            density_mode=args.density_mode,
            density_gamma=args.density_gamma,
            contrast_stretch=not args.no_contrast_stretch
        ).to(args.device)
        print(
            f"✓ Sinkhorn loss ready (blur={args.blur}, grid={args.sinkhorn_grid_size}x{args.sinkhorn_grid_size}, "
            f"mode={args.density_mode}, gamma={args.density_gamma}, "
            f"contrast_stretch={not args.no_contrast_stretch})"
        )
    except Exception as e:
        print(f"✗ Failed to initialize Sinkhorn loss: {e}")
        return 1

    print(
        f"Target density stats: min={target_density.min().item():.4f}, "
        f"max={target_density.max().item():.4f}, mean={target_density.mean().item():.4f}"
    )
    
    # ===== SAMPLE WITH GUIDANCE =====
    print("\nGenerating point clouds with Sinkhorn guidance...")
    try:
        if target_density.shape[0] == 1 and args.n_samples > 1:
            target_density = target_density.repeat(args.n_samples, 1, 1, 1)

        offsets = sample_with_sinkhorn_guidance(
            model,
            target_image=target_density,
            shape=(args.n_samples, 2, args.grid_size, args.grid_size),
            timesteps=args.timesteps,
            lambda_0=args.lambda_scale,
            grad_clip=args.grad_clip,
            device=args.device,
            cond=None,
            sinkhorn_loss_fn=sinkhorn_loss_fn,
            with_tqdm=True,
            debug_guidance=args.debug_guidance,
            debug_every=args.debug_every
        )
        print(f"✓ Generated offsets: {offsets.shape}")
    except Exception as e:
        print(f"✗ Failed during sampling: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== INVERSE OT TRANSFORM =====
    print("\nApplying inverse optimal transport transform...")
    try:
        points = inverse_ot_transform(offsets, grid_size=args.grid_size)
        print(f"✓ Transformed to points: {points.shape}")
    except Exception as e:
        print(f"✗ Failed to transform: {e}")
        return 1
    
    # ===== SAVE RESULTS =====
    print("\nSaving results...")
    
    # Base filename (timestamp-based)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"points_{timestamp}"
    
    # Save point clouds
    if args.save_npy:
        try:
            npy_path = os.path.join(args.output_dir, basename + '.npy')
            save_point_cloud(points, npy_path, format='npy')
        except Exception as e:
            print(f"  Warning: Failed to save .npy: {e}")
    
    # Save visualization
    if args.save_visualization:
        try:
            viz_path = os.path.join(args.output_dir, basename + '_pointcloud')
            visualize_point_cloud_grid(points, output_path=viz_path, show=False)
        except Exception as e:
            print(f"  Warning: Failed to save visualization: {e}")
    
    # Save comparison (points over target density)
    if args.save_comparison:
        try:
            comp_path = os.path.join(args.output_dir, basename + '_comparison')
            visualize_density_guidance(points, target_density, output_path=comp_path, show=False)
        except Exception as e:
            print(f"  Warning: Failed to save comparison: {e}")
    
    # ===== PRINT STATISTICS =====
    print("\nGeneration Statistics:")
    try:
        stats = compute_density_statistics(points, grid_size=args.grid_size)
        for key, val in stats.items():
            if isinstance(val, tuple):
                print(f"  {key}: {val[0]:.4f} to {val[1]:.4f}")
            else:
                print(f"  {key}: {val:.4f}")
    except Exception as e:
        print(f"  Warning: Failed to compute statistics: {e}")
    
    print("\n" + "=" * 80)
    print("✓ Sampling complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
