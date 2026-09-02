"""flow_images_stage_2.py -- Part 2: run the model and export every pipeline representation.

Reproduces, for a single image, the full method pipeline (both TRAIN and INFER paths) so the
figure can show how a stipple is processed. Exports into OUTPUT_DIR/<stem>/:

    03_rejection_prior.png / .npy      the rejection-sampling init points (INFER seed)
    04_gt_offsets_quiver.png / .npy    OT offsets of the GT stipple (the TRAIN target x0)
    05_noised_offsets_quiver.png / .npy   x_t = q_sample(x0, t, noise)  (TRAIN "T Noise" input)
    06_noise_quiver.png / .npy         the Gaussian noise added at t (the training target)
    07_pred_noise_quiver.png / .npy    the noise the model predicts from x_t (compared to 06)  [TRAIN Predict]
    08_final_result.png / .npy         the final points after full truncated (SDEdit) denoising  [INFER Denoised]
    09_rejection_offsets_quiver.png / .npy  OT offsets of the rejection prior  [INFER OT Offsets]
    10_infer_noised_offsets_quiver.png / .npy  rejection offsets noised to t_start (SDEdit init)  [INFER +T=500]
    11_infer_pred_noise_quiver.png / .npy   the noise the model predicts at t_start  [INFER Predict]

Offset fields (2,G,G) are saved as a quiver PNG + raw .npy; point sets (N,2) as a scatter
PNG + raw .npy. The single forward at t mirrors train_control's training step exactly:
    noise = randn ; offsets_t = q_sample(x0, t, noise) ; noise_pred = denoiser(offsets_t, t, controls).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Transforms import to_pointset_optimal_transport, to_image_optimal_transport
from control_v4.sample_control import load_pipeline, load_condition, process_single_image
from control_v4.smart_init import (
    generate_smart_init_points_from_density,
    smart_init_points_to_offsets,
    add_noise_at_t,
)

# ── Config: OUR method = full weights + SDEdit truncation 0.5 ──────────────────
DATA_PATH = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN"
IMAGE_NAME = "emoji-one_4_monkey.png"
OUTPUT_DIR = "experiments/outputs/flow_images"

BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = ""   # full weights are from-scratch: load_pipeline restores the denoiser from the ckpt
CONTROL_CKPT = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt"

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

# Timestep for the single TRAIN-style noise step. Default = the SDEdit start (t_start = 500).
NOISE_T = int(EVAL_TIMESTEPS * INFER_TRUNCATION_RATIO)
DOT_SIZE = 4.0


def save_quiver(offsets, png_path):
    """offsets: (2,G,G) -> magnitude-coloured quiver PNG with a colorbar."""
    n = offsets.shape[-1]
    yy, xx = np.mgrid[0:n, 0:n]
    dx, dy = offsets[0], offsets[1]
    mag = np.sqrt(dx * dx + dy * dy)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    q = ax.quiver(xx, yy, dx, dy, mag, angles="xy", scale_units="xy", scale=1.0,
                  cmap="viridis", width=0.004)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    fig.colorbar(q, ax=ax, shrink=0.8)
    fig.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_scatter(pts, png_path, dot_size=DOT_SIZE):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=dot_size, c="black", linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_offset_pair(offsets, out_dir, name):
    """Save a (2,G,G) offset field as <name>_quiver.png + <name>.npy."""
    offsets = np.asarray(offsets, dtype=np.float64)
    np.save(out_dir / f"{name}.npy", offsets)
    save_quiver(offsets, out_dir / f"{name}_quiver.png")


def save_point_pair(pts, out_dir, name, dot_size=DOT_SIZE):
    """Save an (N,2) point set as <name>.png (scatter) + <name>.npy."""
    pts = np.asarray(pts, dtype=np.float64)
    np.save(out_dir / f"{name}.npy", pts)
    save_scatter(pts, out_dir / f"{name}.png", dot_size)


def offsets_to_points(off_2gg):
    pts = to_pointset_optimal_transport(np.asarray(off_2gg, dtype=np.float64))
    return pts.reshape(pts.shape[0], -1).T


def find_source(data, image_name):
    """Find <image_name> anywhere under data/source/ (nested by category);
    return (full_path, relative_path_under_source) or (None, None)."""
    src_root = Path(data) / "source"
    matches = sorted(src_root.rglob(image_name))
    if not matches:
        return None, None
    return matches[0], matches[0].relative_to(src_root)


def parse_args():
    ap = argparse.ArgumentParser(description="Flow figure stage 2: model + all pipeline representations.")
    ap.add_argument("--data-path", default=DATA_PATH)
    ap.add_argument("--image", default=IMAGE_NAME)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--control-ckpt", default=CONTROL_CKPT)
    ap.add_argument("--t", type=int, default=NOISE_T, help="Timestep for the single TRAIN-style noise step.")
    ap.add_argument("--trunc", type=float, default=INFER_TRUNCATION_RATIO)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--seed", type=int, default=0, help="Seed for the added Gaussian noise (reproducible).")
    return ap.parse_args()


def main():
    args = parse_args()
    data = Path(args.data_path)
    stem = Path(args.image).stem
    out = Path(args.output) / stem
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    source_path, rel = find_source(data, args.image)
    if source_path is None:
        print(f"Image '{args.image}' not found under {data / 'source'}"); return 2
    target_npy = data / "target" / rel.with_suffix(".npy")
    if not target_npy.exists():
        print(f"Missing target npy (needed for GT offsets): {target_npy}"); return 2
    print(f"  found source: {source_path}")

    # Condition tensors + grayscale image (same load the model uses).
    image_01, high_res, target_density, high_res_sdf, target_sdf = load_condition(
        source_path, GRID_SIZE, device, sdf_features=SDF_FEATURES, sdf_truncate_px=SDF_TRUNCATE_PX)

    # (03) Rejection-sampling prior -- the INFER seed (points, no noise).
    smart_points = generate_smart_init_points_from_density(
        image_01, n_points=GRID_SIZE * GRID_SIZE, seed=SMART_INIT_SEED)
    save_point_pair(smart_points, out, "03_rejection_prior", args.dot_size)

    # (04) GT stipple -> OT offsets = the TRAIN target x0 (2,G,G).
    gt_points = np.load(target_npy).astype(np.float64)
    gt_offsets = np.asarray(to_image_optimal_transport(gt_points.astype(np.float32)), dtype=np.float64)
    save_offset_pair(gt_offsets, out, "04_gt_offsets")

    # ── Model ──────────────────────────────────────────────────────────────────
    diffusion, control_net = load_pipeline(
        base_config_path=BASE_CONFIG_PATH, base_ckpt_path=BASE_CKPT_PATH, control_ckpt_path=args.control_ckpt,
        grid_size=GRID_SIZE, enable_gecco=ENABLE_GECCO,
        enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        batch_coords_features=BATCH_COORDS_FEATURES, device=device)
    denoiser = diffusion.model
    denoiser.eval(); control_net.eval()

    # ── One TRAIN-style forward at t (mirrors train_control's training step) ────
    x_0 = torch.from_numpy(gt_offsets).unsqueeze(0).float().to(device)   # (1,2,G,G)
    t_tensor = torch.full((1,), int(args.t), dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    noise = torch.randn(x_0.shape, generator=gen, device=device, dtype=x_0.dtype)  # (06) eps
    with torch.no_grad():
        offsets_t = diffusion.q_sample(x_0, t_tensor, noise)            # (05) x_t
        controls = control_net(offsets_t, t_tensor, high_res, target_density,
                               high_res_sdf=high_res_sdf, target_sdf_map=target_sdf,
                               target_smart_init_map=None)
        noise_pred = denoiser(offsets_t, t_tensor, controls=controls)   # (07) eps_pred

    save_offset_pair(offsets_t[0].detach().cpu().numpy(), out, f"05_noised_offsets_t{int(args.t)}")
    save_offset_pair(noise[0].detach().cpu().numpy(), out, "06_noise")
    save_offset_pair(noise_pred[0].detach().cpu().numpy(), out, "07_pred_noise")

    # ── INFER-path forward: rejection prior's OT offsets, its noised (SDEdit) init at
    #    t_start, and the model's predicted noise on it (the INFER "Predict" panel). ──
    rej_offsets = np.asarray(smart_init_points_to_offsets(smart_points), dtype=np.float64)  # (09)
    save_offset_pair(rej_offsets, out, "09_rejection_offsets")

    t_start = int(EVAL_TIMESTEPS * args.trunc)
    x_init = torch.from_numpy(rej_offsets).unsqueeze(0).float().to(device)
    noise_i = torch.randn(x_init.shape, generator=gen, device=device, dtype=x_init.dtype)
    with torch.no_grad():
        # SDEdit init: rejection offsets forward-noised to t_start (exactly sample_control's seed).
        infer_noised = add_noise_at_t(x_init, diffusion.alphas_cumprod[t_start], noise=noise_i)  # (10)
        t_infer = torch.full((1,), t_start, dtype=torch.long, device=device)
        controls_i = control_net(infer_noised, t_infer, high_res, target_density,
                                 high_res_sdf=high_res_sdf, target_sdf_map=target_sdf,
                                 target_smart_init_map=None)
        pred_infer = denoiser(infer_noised, t_infer, controls=controls_i)  # (11)
    save_offset_pair(infer_noised[0].detach().cpu().numpy(), out, f"10_infer_noised_offsets_t{t_start}")
    save_offset_pair(pred_infer[0].detach().cpu().numpy(), out, "11_infer_pred_noise")

    # ── (08) Full truncated (SDEdit) inference -> final points ──────────────────
    tmp_npy_dir = out / "_final_tmp"
    tmp_npy_dir.mkdir(parents=True, exist_ok=True)
    process_single_image(
        image_path=source_path, diffusion=diffusion, control_net=control_net,
        grid_size=GRID_SIZE, truncation_ratio=args.trunc, eval_timesteps=EVAL_TIMESTEPS,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        resample_jumps=RESAMPLE_JUMPS, sdf_truncate_px=SDF_TRUNCATE_PX, t_start_step=-1,
        smart_init_seed=SMART_INIT_SEED, smart_init_splat_sigma_px=SMART_INIT_SPLAT_SIGMA_PX,
        enable_smart_init_splat_sigma=ENABLE_SMART_INIT_SPLAT_SIGMA,
        show_denoising_interval=50, device=device,
        export_conditions=False, export_png=False, export_npy=True,
        track_time=False, show_denoising=False,
        conditions_dir=None, png_dir=None, npy_dir=tmp_npy_dir,
        timestamps_dir=None, denoising_dir=None)
    final_npy = tmp_npy_dir / f"{stem}.npy"
    if final_npy.exists():
        final_pts = np.load(final_npy).astype(np.float64)
        save_point_pair(final_pts, out, "08_final_result", args.dot_size)
    else:
        print(f"  [warn] final inference npy not found: {final_npy}")

    print(f"\nStage 2 done -> {out}  (noise step t={int(args.t)}, trunc={args.trunc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
