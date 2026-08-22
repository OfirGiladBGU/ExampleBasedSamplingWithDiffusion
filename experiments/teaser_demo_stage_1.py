"""teaser_demo_stage_1.py -- Part 1: run the truncated denoising on the demo image.

Our method uses TRUNCATED (SDEdit) denoising: it starts from the rejection prior at
t_start = eval_timesteps * truncation_ratio (= 500 for ratio 0.5) and denoises to the
end. This script runs that on a single condition image and snapshots the point state
at t = 500 (rejection prior), t = 750 (mid) and t = 1000 (final), saving each as:

    OUTPUT_DIR/steps/<t_label>.npy   (N,2) points -- used for the vectorized figure
    OUTPUT_DIR/steps/<t_label>.png   scatter preview

The condition image (source/Demo.png) is left in place for stage 2 to load.

Snapshot mapping (with truncation 0.5): figure-t = t_start + elapsed_denoise_steps, so
    t=500  -> elapsed 0   (denoising step _0000)
    t=750  -> elapsed 250 (denoising step _0250)
    t=1000 -> the final result
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Transforms import to_pointset_optimal_transport
from control_v4.sample_control import load_pipeline, process_single_image

# ── Config: OUR method = full weights + SDEdit truncation 0.5 ──────────────────
BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = ""   # full weights are from-scratch: load_pipeline restores the denoiser from the control ckpt
CONTROL_CKPT = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt"

INPUT_IMAGE = "experiments/outputs/teaser_demo_results/source/Demo.png"
OUTPUT_DIR = "experiments/outputs/teaser_demo_results"

GRID_SIZE = 32
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
INFER_TRUNCATION_RATIO = 0.5
RESAMPLE_JUMPS = 0
EVAL_TIMESTEPS = 1000
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SPLAT_SIGMA_PX = 0.5
ENABLE_SMART_INIT_SPLAT_SIGMA = False
DEVICE = "cuda"

STEP_INTERVAL = 250   # saves elapsed 0 (t=500) and 250 (t=750); t=1000 comes from the final export
# (figure-t label, denoising 'elapsed' step or "final")
SNAPSHOTS = [("t500", 0), ("t750", 250), ("t1000", "final")]

DOT_SIZE = 4.0


def offsets_to_points(off_2gg):
    """(2,G,G) offset field -> (N,2) point set, matching sample_control's final export."""
    pts = to_pointset_optimal_transport(np.asarray(off_2gg, dtype=np.float64))
    return pts.reshape(pts.shape[0], -1).T


def save_scatter(pts, path, dot_size):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=dot_size, c="black", linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="Run truncated denoising on the demo image, export t=500/750/1000.")
    ap.add_argument("--image", default=INPUT_IMAGE)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--control-ckpt", default=CONTROL_CKPT)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    stem = Path(args.image).stem
    out = Path(args.output)
    final_npy_dir = out / "_final_npy"
    final_png_dir = out / "_final_png"
    denoise_dir = out / "_denoising"
    steps_dir = out / "steps"

    t_start = int(EVAL_TIMESTEPS * INFER_TRUNCATION_RATIO)
    print(f"Demo '{stem}': full weights, trunc={INFER_TRUNCATION_RATIO} -> t_start={t_start}")
    print(f"  snapshots {[ (lbl, w) for lbl, w in SNAPSHOTS ]}  interval={STEP_INTERVAL}")
    if args.dry_run:
        print(f"DRY -> {steps_dir}")
        return 0

    for d in (final_npy_dir, final_png_dir, denoise_dir, steps_dir):
        d.mkdir(parents=True, exist_ok=True)

    diffusion, control_net = load_pipeline(
        base_config_path=BASE_CONFIG_PATH, base_ckpt_path=BASE_CKPT_PATH, control_ckpt_path=args.control_ckpt,
        grid_size=GRID_SIZE, enable_gecco=ENABLE_GECCO,
        enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        batch_coords_features=BATCH_COORDS_FEATURES, device=DEVICE,
    )

    process_single_image(
        image_path=Path(args.image), diffusion=diffusion, control_net=control_net,
        grid_size=GRID_SIZE, truncation_ratio=INFER_TRUNCATION_RATIO, eval_timesteps=EVAL_TIMESTEPS,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        resample_jumps=RESAMPLE_JUMPS, sdf_truncate_px=SDF_TRUNCATE_PX, t_start_step=-1,
        smart_init_seed=SMART_INIT_SEED, smart_init_splat_sigma_px=SMART_INIT_SPLAT_SIGMA_PX,
        enable_smart_init_splat_sigma=ENABLE_SMART_INIT_SPLAT_SIGMA,
        show_denoising_interval=STEP_INTERVAL, device=DEVICE,
        export_conditions=False, export_png=True, export_npy=True,
        track_time=False, show_denoising=True,   # <- intermediate frames
        conditions_dir=None, png_dir=final_png_dir, npy_dir=final_npy_dir,
        timestamps_dir=None, denoising_dir=denoise_dir,
    )

    # Assemble clean point snapshots (all as (N,2) points).
    for label, which in SNAPSHOTS:
        if which == "final":
            pts = np.load(final_npy_dir / f"{stem}.npy").astype(np.float64)   # already (N,2)
        else:
            step_npy = denoise_dir / "npy" / f"{stem}_step_{int(which):04d}.npy"
            if not step_npy.exists():
                print(f"  WARN: missing {step_npy} (check STEP_INTERVAL / truncation)"); continue
            off = np.load(step_npy).astype(np.float64)      # (n_samples, 2, G, G)
            pts = offsets_to_points(off[0])
        np.save(steps_dir / f"{label}.npy", pts)
        save_scatter(pts, steps_dir / f"{label}.png", args.dot_size)
        print(f"  saved {label}: {pts.shape[0]} pts -> {steps_dir / f'{label}.npy'}")

    print(f"\nStage 1 done -> {steps_dir}  (condition: {Path(args.image)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
