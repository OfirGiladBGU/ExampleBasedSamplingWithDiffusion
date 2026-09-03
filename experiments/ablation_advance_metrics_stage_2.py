"""Stage 2 ablation scorer.

This script reads the point-set outputs produced by stage 1 from
`OUTPUT_DIR/{RESULTS_DIR}/epoch_*_npy/`, computes the advanced metrics per
example, and writes JSON files to `OUTPUT_DIR/{RESULTS_DIR}/epoch_*_json/`.

The per-example JSON contains:
    M1_v1_cvt_energy / M1_v2_power_cvt_energy
    M2_v1_capacity_delta_c / M2_v2_power_displacement
    M3_emd_distance
    M4_sinkhorn_ot_cost
    M5_spatial_measure_rho_mean
    M6_minsnr_loss
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.Config import ParseSampleConfig
from utils.stippling_metrics_advance import compute_all_advanced_metrics
from control_v4.sample_control import load_pipeline, load_condition
from control_v4.train_control import density_match_loss, _grid_centers_flat
from data.Transforms import to_image_optimal_transport


CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"
FREEZE_DENOISER = True  # TODO: Add in the other scripts
GRID_SIZE = 32
ENABLE_GECCO = False
ENABLE_ADAPTIVE_GATE_INJECTION = False
EVAL_TIMESTEPS = 1000
INFER_TRUNCATION_RATIO = 1.0
RESAMPLE_JUMPS = 0
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False


# Vanilla
WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_vanilla/checkpoints"
RESULTS_DIR = "vanilla"


# Unfrozen
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_unfrozen/checkpoints"
# RESULTS_DIR = "unfrozen"
# BASE_CKPT_PATH = ""  # NOTE
# FREEZE_DENOISER = False  # NOTE


# GECCO
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_gecco/checkpoints"
# RESULTS_DIR = "gecco"
# ENABLE_GECCO = True  # NOTE


# Adaptive gate injection
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_agi/checkpoints"
# RESULTS_DIR = "agi"
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE


# Full
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints"
# RESULTS_DIR = "full"
# BASE_CKPT_PATH = ""  # NOTE
# FREEZE_DENOISER = False  # NOTE
# ENABLE_GECCO = True  # NOTE
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE


# Full + SDEdit
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints"
# RESULTS_DIR = "sdedit"
# BASE_CKPT_PATH = ""  # NOTE
# FREEZE_DENOISER = False  # NOTE
# ENABLE_GECCO = True  # NOTE
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE
# INFER_TRUNCATION_RATIO = 0.5  # NOTE


# Default folders
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e500_b50_1024"
# GT oracle folder under resources/ -- read ONLY by the M6 losses.
TARGET_NAME = "target_GBN_1024"

MIN_SNR_GAMMA = 5.0
# Drawn by ablation_advance_metrics_stage_0.py; stage 2 only reads it.
TIMESTEPS_NAME = "minsnr_timesteps.json"
# NUM_SAMPLES = -1
NUM_SAMPLES = 50
# EVERY_EPOCH = -1  # Keep all checkpoints by default
EVERY_EPOCH = 500

# Component-2 (density-match KDE) loss constants -- mirror control_v4/train_control.py.
DENSITY_LOSS_WEIGHT = 0.8
DENSITY_KDE_GRID = 32
DENSITY_KDE_SIGMA_PX = 1.0
DENSITY_LOSS_T_FRAC = 0.4      # mirrors train_control.DENSITY_LOSS_T_FRAC
DENSITY_LOSS_T_SOFT = 0.0      # mirrors train_control.DENSITY_LOSS_T_SOFT

# --- How the M6 losses are computed ------------------------------------------
# LOSS_APPROX = True   Fast proxy, no model: MinSNR-weighted density MSE + KDE loss, both
#                      computed on POINT SETS (prediction vs GT .npy). Scores the stage-1
#                      EXPORTED points. Pure CPU/numpy, ~55 ms per image.
# LOSS_APPROX = False  The TRUE training loss: re-noises the GT offsets to t, runs the
#                      variant's ControlNet + denoiser, and measures the noise-prediction
#                      error -- exactly reproducing train_control.py's
#                          loss = denoise_loss + density_w * d_loss
#                      This scores the MODEL at that checkpoint (the same quantity training
#                      logs as train/val loss), NOT the exported point set. Needs the
#                      checkpoints in WEIGHTS_DIR and one forward pass per image.
LOSS_APPROX = False
DEVICE = "cpu"                 # "cpu" or "cuda"; used only when LOSS_APPROX is False
LOSS_SEED = 42                 # deterministic per-image noise draw for the true loss

# --- The two parts of this stage, independently runnable ---------------------
# Part 1: the per-image geometry metrics (M1-M5) -> {RESULTS_DIR}/epoch_*_json/*.json
# Part 2: the M6 training losses, evaluated as ONE BATCH over the whole validation set
#         per epoch -> {RESULTS_DIR}/loss_avg.json
# They share nothing at runtime, so a failed part 2 can be re-run on its own with
# RUN_METRICS_P1 = False (and vice versa) without recomputing the other.
RUN_METRICS_P1 = True
RUN_METRICS_P2 = True
LOSS_FILE = "loss_avg.json"
# Rows per forward pass in part 2. A row is one (image, timestep) pair, so the batch is
# len(val_names) * samples_per_image rows in total. Lower this if memory is tight.
LOSS_BATCH_SIZE = 10

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# M6_minsnr_loss        = Component-1 proxy only (MinSNR-weighted density MSE).
# M6_v2_minsnr_kde_loss = full training-loss proxy: Component 1 + w * Component-2 KDE loss.
METRIC_ORDER = [
    "M1_v1_cvt_energy",
    "M1_v2_power_cvt_energy",
    "M2_v1_capacity_delta_c",
    "M2_v2_power_displacement",
    "M2_v3_power_cell_cap_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
]

# Part 2 losses. Not per-image: they are a single set-level mean per epoch, written to
# LOSS_FILE. See compute_epoch_losses for why the KDE term cannot be averaged per image.
LOSS_ORDER = [
    "M6_minsnr_loss",
    "M6_v2_minsnr_kde_loss",
]
MC_APPROX = False


def parse_args():
    parser = argparse.ArgumentParser(description="Score stage 1 point outputs with advanced metrics")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Base output folder for stage 1 / stage 2 artifacts")
    parser.add_argument("--target-name", default=TARGET_NAME,
                        help="GT target folder under resources/ (used by the M6 losses)")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Number of validation samples to score in order; -1 means use all")
    parser.add_argument("--epochs", default="all", help="'all' or comma-separated substrings matching epoch dir names")
    parser.add_argument("--every-epoch", type=int, default=EVERY_EPOCH,
                        help="Keep only checkpoints whose epoch (from '*ep{N}') is a multiple of this; 0 = all")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be processed")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--base-ckpt", default=CKPT_PATH)
    parser.add_argument("--timesteps", type=int, default=EVAL_TIMESTEPS)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--min-snr-gamma", type=float, default=MIN_SNR_GAMMA)
    parser.add_argument("--density-loss-weight", type=float, default=DENSITY_LOSS_WEIGHT,
                        help="Component-2 weight in the M6_v2 composite (mirrors training DENSITY_LOSS_WEIGHT)")
    parser.add_argument("--density-kde-grid", type=int, default=DENSITY_KDE_GRID,
                        help="KDE map resolution for the M6_v2 density-match term")
    parser.add_argument("--density-kde-sigma-px", type=float, default=DENSITY_KDE_SIGMA_PX,
                        help="Gaussian sigma (in KDE-grid pixels) for the M6_v2 density-match term")
    parser.add_argument("--density-loss-t-frac", type=float, default=DENSITY_LOSS_T_FRAC,
                        help="Component 2 applies only where t < frac * timesteps (true-loss mode)")
    parser.add_argument("--density-loss-t-soft", type=float, default=DENSITY_LOSS_T_SOFT,
                        help="Soft-ramp width at the Component 2 t cutoff; 0 = hard mask (true-loss mode)")
    parser.add_argument("--loss-approx", action=argparse.BooleanOptionalAction, default=LOSS_APPROX,
                        help="Use the fast point-set proxy for M6. --no-loss-approx runs the real "
                             "ControlNet + denoiser forward pass for the exact training loss.")
    parser.add_argument("--device", default=DEVICE, choices=["cpu", "cuda"],
                        help="Device for the true-loss forward pass (LOSS_APPROX = False only)")
    parser.add_argument("--run-metrics-p1", action=argparse.BooleanOptionalAction,
                        default=RUN_METRICS_P1,
                        help="Part 1: per-image geometry metrics (M1-M5) into epoch_*_json/")
    parser.add_argument("--run-metrics-p2", action=argparse.BooleanOptionalAction,
                        default=RUN_METRICS_P2,
                        help="Part 2: the batched M6 losses into loss_avg.json")
    parser.add_argument("--loss-file", default=LOSS_FILE,
                        help="Part 2 output, written in the model root")
    parser.add_argument("--loss-batch-size", type=int, default=LOSS_BATCH_SIZE,
                        help="Rows (image x timestep pairs) per forward pass in part 2")
    parser.add_argument("--loss-seed", type=int, default=LOSS_SEED,
                        help="Seed for the per-image noise draw in true-loss mode (kept identical "
                             "across variants/epochs so the comparison is paired)")
    parser.add_argument("--mc-approx", action=argparse.BooleanOptionalAction, default=MC_APPROX,
                        help="Use the legacy Monte Carlo approximation for the advanced metrics. Default is now exact/deterministic quadrature; pass --mc-approx only to reproduce previously published approximate numbers.")
    parser.add_argument("--timestep-file", type=str, default=None,
                        help="Override the per-image MinSNR timesteps JSON. Defaults to "
                             "output/resources/minsnr_timesteps.json, drawn by stage 0.")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def limit_validation_images(val_names, num_samples):
    if int(num_samples) < 0:
        return val_names
    return val_names[: int(num_samples)]


def filter_by_every_epoch(epoch_dirs, every):
    """Keep only epoch dirs whose epoch (parsed from 'epoch_{N}_npy') is a multiple of `every`.

    every <= 0 disables the filter (keeps all). NOTE: stage_2 filters epoch DIRECTORIES
    (epoch_500_npy, ...), so the epoch is parsed with the 'epoch_' prefix -- not the '*ep' prefix
    used on checkpoint files in stage_1.
    """
    if int(every) <= 0:
        return epoch_dirs
    kept = []
    for d in epoch_dirs:
        m = re.search(r"epoch_(\d+)", os.path.basename(str(d)))
        if m and int(m.group(1)) % int(every) == 0:
            kept.append(d)
    return kept


def load_validation_data(out_base, target_name):
    """Read the shared resources staged by ablation_advance_metrics_stage_0.py:

        OUTPUT_DIR/resources/validation_manifest.json
        OUTPUT_DIR/resources/source/
        OUTPUT_DIR/resources/<target_name>/     (GT, used only by the M6 losses)
    """
    res = Path(out_base) / "resources"
    manifest_path = res / "validation_manifest.json"
    source_dir = res / "source"
    target_dir = res / target_name
    missing = [str(p) for p in (manifest_path, source_dir, target_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing staged resources: " + ", ".join(missing)
            + "  -- run experiments/ablation_advance_metrics_stage_0.py first.")
    return json.loads(manifest_path.read_text()), source_dir, target_dir


def find_epoch_dirs(model_root, pattern_filter=None):
    root = Path(model_root)
    candidates = [p for p in root.glob("epoch_*_npy") if p.is_dir()]

    if pattern_filter and pattern_filter != "all":
        wanted = [s.strip() for s in pattern_filter.split(",") if s.strip()]
        candidates = [c for c in candidates if any(w in c.name for w in wanted)]

    def epoch_of_name(name):
        m = re.search(r"epoch_(\d{2,6})_npy", name)
        return int(m.group(1)) if m else None

    def sort_key(path):
        ep = epoch_of_name(path.name)
        if ep is not None:
            return (0, ep)
        return (1, int(path.stat().st_mtime))

    candidates.sort(key=sort_key)
    return candidates


def epoch_id_from_name(name):
    m = re.search(r"epoch_(\d{2,6})", name)
    if m:
        return m.group(1)
    m = re.search(r"(\d{2,6})", name)
    if m:
        return m.group(1)
    return str(int(time.time()))


def load_pointset(np_path):
    pts = np.load(np_path)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected Nx2 point array in {np_path}, got shape {pts.shape}")
    return pts


def render_points_to_density(points, grid_size):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float64)
    pts = np.clip(points, 0.0, 1.0 - 1e-12)
    px = (pts[:, 0] * grid_size).astype(np.int64)
    py = (pts[:, 1] * grid_size).astype(np.int64)
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    np.add.at(grid, (py, px), 1.0)
    mass = grid.sum()
    if mass <= 1e-12:
        return np.zeros_like(grid)
    return grid / mass


def build_diffusion_schedule(args):
    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"])
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()
    return diffusion


def load_timesteps(args, out_base: Path):
    """Load the per-image MinSNR timesteps drawn by ablation_advance_metrics_stage_0.py.

    Returns {image name -> [timestep indices]}. Load-only on purpose: the draw happens
    once, in stage 0, so re-running stage 2 -- or running it for another variant, or on
    another device -- always scores against exactly the same timesteps.
    """
    tfp = Path(args.timestep_file) if args.timestep_file else out_base / "resources" / TIMESTEPS_NAME
    if not tfp.exists():
        raise FileNotFoundError(
            f"MinSNR timesteps not found: {tfp}  -- run "
            f"experiments/ablation_advance_metrics_stage_0.py first.")
    loaded = json.loads(tfp.read_text())
    print(f"Loaded MinSNR timesteps for {len(loaded)} images from {tfp}")
    return {k: [int(x) for x in v] for k, v in loaded.items()}


def compute_minsnr_proxy(pred_points, gt_points, diffusion, args, timesteps=None):
    """
    Compute MinSNR-weighted loss. If `timesteps` is None, use args.min_snr_timestep
    as a single-element list. If `timesteps` is a list, compute weighted MSE per
    timestep and return the average across samples.

    Both sides use the exact point sets through the same estimator: the teacher is the
    target .npy, NOT the rasterized .png (which would bin coincident dots into one cell).
    """
    pred_density = render_points_to_density(pred_points, args.grid_size)
    target_density = render_points_to_density(gt_points, args.grid_size)
    mse = float(np.mean((pred_density - target_density) ** 2))

    if args.min_snr_gamma <= 0:
        return mse

    total = 0.0
    count = 0
    L = len(diffusion.alphas_cumprod)
    for t in timesteps:
        t_idx = int(np.clip(int(t), 0, L - 1))
        alpha_t = float(diffusion.alphas_cumprod[t_idx].item())
        snr = alpha_t / max(1.0 - alpha_t, 1e-8)
        weight = min(snr, float(args.min_snr_gamma)) / max(snr, 1e-8)
        total += mse * weight
        count += 1

    return float(total / max(1, count))


def gaussian_kde_map_np(points, kde_grid, sigma_px):
    """Numpy port of train_control.gaussian_kde_map for a single point set.

    Splats (N,2) normalized [x,y] coords onto a (K,K) map indexed [y,x] with a
    Gaussian of sigma = sigma_px / K, summed over points, then normalised to
    *mean 1* (sum = K*K) -- exactly the convention the training KDE loss uses.
    """
    K = int(kde_grid)
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.ones((K, K), dtype=np.float64)
    lin = (np.arange(K, dtype=np.float64) + 0.5) / float(K)
    cx, cy = np.meshgrid(lin, lin)                 # cx[r,c]=lin[c] (x), cy[r,c]=lin[r] (y)
    centers = np.stack([cx.ravel(), cy.ravel()], axis=-1)      # (K*K, 2), row-major [y,x]
    sigma = max(float(sigma_px), 1e-4) / float(K)
    d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)   # (N, K*K)
    density = np.exp(-d2 / (2.0 * sigma * sigma)).sum(axis=0)          # (K*K,)
    s = density.sum()
    if s <= 1e-8:
        return np.ones((K, K), dtype=np.float64)
    density = density * (density.size / s)
    return density.reshape(K, K)


def compute_kde_proxy(pred_points, gt_points, args):
    """Component-2 (density-match) proxy, already scaled by density_loss_weight:
    density_loss_weight * MSE between the mean-1 KDE of the predicted points and the
    mean-1 KDE of the teacher points. A faithful mirror of the  density_w * d_loss  term
    of train_control.density_match_loss: same kernel, same sigma, same mean-1 normalisation,
    and -- crucially -- the same estimator on BOTH sides.

    Note: an earlier version used the box-averaged target image as the teacher field. That
    compared a Gaussian splat against a box downsample, so the kernel mismatch (not any real
    prediction error) dominated the MSE and inflated this term by ~35x.

    M6_v2_minsnr_kde_loss = M6_minsnr_loss + compute_kde_proxy(...).
    """
    K = int(args.density_kde_grid)
    kde_pred = gaussian_kde_map_np(pred_points, K, args.density_kde_sigma_px)   # mean 1
    kde_true = gaussian_kde_map_np(gt_points, K, args.density_kde_sigma_px)     # mean 1
    kde_mse = float(np.mean((kde_pred - kde_true) ** 2))
    return float(args.density_loss_weight) * kde_mse


def find_epoch_checkpoints(weights_dir):
    """Map epoch id (as a string, e.g. '5000') -> checkpoint path in WEIGHTS_DIR.

    Mirrors ablation_advance_metrics_stage_1.find_checkpoints: skips best_* snapshots and
    parses the epoch out of the file name, so stage 2 can reload the exact weights stage 1
    used to produce each epoch_*_npy directory.
    """
    p = Path(weights_dir)
    out = {}
    for pat in ("*.pt", "*.pth", "*.ckpt"):
        for c in p.glob(pat):
            if not c.is_file() or c.name.startswith("best_"):
                continue
            m = re.search(r"(\d{2,6})", c.name)
            if m:
                out[m.group(1)] = str(c)
    return out


def build_true_loss_models(ckpt_path, args):
    """Load the variant's pipeline for one epoch and bundle what the true loss needs."""
    if SMART_INIT_FEATURES or SDF_FEATURES:
        raise NotImplementedError(
            "The true-loss path does not build the SDF / smart-init conditioning maps this "
            "variant was trained with, so its loss would be wrong. Use --loss-approx for "
            "SMART_INIT_FEATURES / SDF_FEATURES variants."
        )
    device = torch.device(args.device)
    diffusion, control_net = load_pipeline(
        base_config_path=args.config,
        base_ckpt_path=args.base_ckpt,
        control_ckpt_path=ckpt_path,
        grid_size=args.grid_size,
        enable_gecco=ENABLE_GECCO,
        enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION,
        smart_init_features=SMART_INIT_FEATURES,
        sdf_features=SDF_FEATURES,
        batch_coords_features=BATCH_COORDS_FEATURES,
        device=device,
    )
    diffusion.set_num_timesteps(args.timesteps)
    denoiser = diffusion.model
    denoiser.eval()
    control_net.eval()
    # Grid centres for the Component-2 loss, built once (they never change per image).
    # density_match_loss -> offsets_to_coords_gpu dereferences point_centers WITHOUT a
    # None guard, so these are required, not optional: mirrors train_control.py, which
    # builds them at their own grid size in float32.
    return {
        "diffusion": diffusion,
        "control_net": control_net,
        "denoiser": denoiser,
        "device": device,
        "point_centers": _grid_centers_flat(args.grid_size, device=device, dtype=torch.float32),
        "kde_centers": _grid_centers_flat(args.density_kde_grid, device=device, dtype=torch.float32),
    }


def build_loss_inputs(val_names, source_dir, target_dir, timesteps_map, args, device):
    """Build everything part 2 needs, once, and reuse it for every epoch.

    The GT offsets, the conditioning tensors and the noise depend only on the dataset --
    not on the checkpoint -- so they are built a single time rather than re-read for each
    of the ~10 epochs. Returns a flat list of ROWS, one per (image, timestep) pair, which
    is exactly the batch the training loss consumes.

    Noise is drawn per image on the CPU RNG seeded by (loss_seed, stem), so a row's noise
    is identical across variants, epochs and devices.
    """
    G = int(args.grid_size)
    rows = []
    for name in val_names:
        stem = Path(name).stem
        source_path = source_dir / name
        target_npy = target_dir / f"{stem}.npy"
        if not source_path.exists():
            print(f"  skipping {name}: missing source image {source_path}")
            continue
        if not target_npy.exists():
            print(f"  skipping {name}: missing target point set {target_npy}")
            continue
        ts = timesteps_map.get(name)
        if not ts:
            print(f"  skipping {name}: no MinSNR timesteps")
            continue

        _, high_res, target_density, _, _ = load_condition(
            source_path, G, device, sdf_features=SDF_FEATURES)
        x0_np = np.asarray(to_image_optimal_transport(
            load_pointset(target_npy).astype(np.float64)), dtype=np.float32)
        x_0 = torch.from_numpy(x0_np).unsqueeze(0)                 # (1, 2, G, G), CPU

        gen = torch.Generator()      # CPU generator: cuda has a separate RNG stream
        gen.manual_seed((int(args.loss_seed) + zlib.crc32(stem.encode("utf-8"))) % (2 ** 31))
        for t_raw in ts:
            noise = torch.randn(x_0.shape, generator=gen, dtype=x_0.dtype)
            rows.append({
                "name": name, "t": int(t_raw), "x_0": x_0,
                "high_res": high_res, "target_density": target_density, "noise": noise,
            })
    return rows


def compute_epoch_losses_true(rows, models, args):
    """The training loss over the whole validation set for one checkpoint, in batches.

    Aggregation matches control_v4/train_control.py exactly, and this is why part 2 is a
    batch rather than a per-image loop:

      * denoise term -- training takes  (per_sample_mse * min_snr_weight).mean()  over the
        batch, so we sum the weighted MSEs and divide by the total row count.
      * density term -- training takes  (per_sample * w).sum() / w.sum(), normalising by
        the ACTIVE rows (those with t < t_cutoff), NOT by the batch size. Averaging
        per-image values would instead divide by the row count and silently dilute the
        result by the inactive fraction (~54% at one sample per image).

    density_match_loss returns the already-divided loss plus the active FRACTION, so the
    numerator is recovered as loss * active_fraction * batch_size.

    Returns (minsnr_loss, kde_loss, active_fraction).
    """
    diffusion = models["diffusion"]
    control_net = models["control_net"]
    denoiser = models["denoiser"]
    device = models["device"]
    G = int(args.grid_size)
    t_cutoff = float(args.density_loss_t_frac) * float(args.timesteps)
    n_steps = len(diffusion.alphas_cumprod)

    mse_sum = 0.0        # sum of MinSNR-weighted per-sample MSEs
    n_rows = 0
    kde_num = 0.0        # sum of (per-sample KDE loss * active weight)
    kde_den = 0.0        # sum of active weights
    bs = max(1, int(args.loss_batch_size))

    with torch.no_grad():
        for start in range(0, len(rows), bs):
            chunk = rows[start:start + bs]
            B = len(chunk)
            x_0 = torch.cat([r["x_0"] for r in chunk], dim=0).to(device)
            noise = torch.cat([r["noise"] for r in chunk], dim=0).to(device)
            high_res = torch.cat([r["high_res"] for r in chunk], dim=0).to(device)
            target_density = torch.cat([r["target_density"] for r in chunk], dim=0).to(device)
            t = torch.tensor([int(np.clip(r["t"], 0, n_steps - 1)) for r in chunk],
                             dtype=torch.long, device=device)

            offsets_t = diffusion.q_sample(x_0, t, noise)
            controls = control_net(
                offsets_t, t, high_res, target_density,
                high_res_sdf=None, target_sdf_map=None, target_smart_init_map=None,
            )
            noise_pred = denoiser(offsets_t, t, controls=controls)

            per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2, 3))
            if args.min_snr_gamma > 0:
                alphas_cumprod_t = diffusion.alphas_cumprod.gather(0, t)
                snr = alphas_cumprod_t / torch.clamp(1.0 - alphas_cumprod_t, min=1e-8)
                min_snr_weight = (torch.clamp(snr, max=float(args.min_snr_gamma))
                                  / torch.clamp(snr, min=1e-8))
                mse_sum += float((per_sample_mse * min_snr_weight).sum())
            else:
                mse_sum += float(per_sample_mse.sum())
            n_rows += B

            x0_pred = diffusion.predict_xstart_from_noise(offsets_t, t, noise_pred)
            d_loss, d_active = density_match_loss(
                x0_pred, x_0, t,
                grid_size=G,
                kde_grid=int(args.density_kde_grid),
                sigma_px=float(args.density_kde_sigma_px),
                t_cutoff=t_cutoff,
                t_soft=float(args.density_loss_t_soft),
                kde_centers=models["kde_centers"],
                point_centers=models["point_centers"],
            )
            active = float(d_active) * B
            kde_num += float(d_loss) * active
            kde_den += active
            print(f"    rows {start + 1}-{start + B}/{len(rows)}", flush=True)

    minsnr_loss = mse_sum / max(1, n_rows)
    kde_loss = float(args.density_loss_weight) * (kde_num / kde_den if kde_den > 1e-8 else 0.0)
    return minsnr_loss, kde_loss, (kde_den / max(1, n_rows))


def compute_epoch_losses_approx(epoch_dir, val_names, target_dir, diffusion, args, timesteps_map):
    """The LOSS_APPROX = True path: the point-set proxy, averaged over the validation set.

    No model and no t-mask here, so a plain mean over images is the right aggregation.
    Scores the stage-1 EXPORTED points rather than the model.
    """
    epoch_dir = Path(epoch_dir)
    minsnr_sum = 0.0
    kde_sum = 0.0
    n = 0
    for name in val_names:
        stem = Path(name).stem
        pred_path = epoch_dir / f"{stem}.npy"
        target_npy = target_dir / f"{stem}.npy"
        if not pred_path.exists() or not target_npy.exists():
            print(f"  skipping {name}: missing prediction or target point set")
            continue
        pred_points = load_pointset(pred_path)
        gt_points = load_pointset(target_npy)
        ts = timesteps_map.get(name)
        minsnr_sum += compute_minsnr_proxy(pred_points, gt_points, diffusion, args, timesteps=ts)
        kde_sum += compute_kde_proxy(pred_points, gt_points, args)
        n += 1
    n = max(1, n)
    return minsnr_sum / n, kde_sum / n, 1.0


def update_loss_file(path: Path, epoch_id, values):
    """Merge one epoch's losses into LOSS_FILE, keeping the epochs numerically ordered.

    Merging rather than overwriting means part 2 can be re-run for a single epoch (or
    resumed after a failure) without discarding the epochs already scored.
    """
    data = json.loads(path.read_text()) if path.exists() else {}
    for key in LOSS_ORDER:
        series = data.get(key, {})
        series[str(epoch_id)] = float(values[key])
        data[key] = {str(e): series[str(e)] for e in sorted(int(x) for x in series)}
    path.write_text(json.dumps(data, indent=2))


def serialize_metrics(metrics):
    ordered = {}
    for key in METRIC_ORDER:
        if key in metrics:
            ordered[key] = float(metrics[key])
    return ordered


def score_epoch_dir(epoch_dir, output_epoch_dir, val_names, source_backup_dir, args, epoch_id):
    """Part 1: the per-image geometry metrics (M1-M5) for one epoch.

    Reads only the stage-1 predictions and the source condition -- the M6 losses moved to
    part 2, which evaluates them as one batch per epoch.
    """
    output_epoch_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir = Path(epoch_dir)

    for idx, name in enumerate(val_names, start=1):
        stem = Path(name).stem
        pred_path = epoch_dir / f"{stem}.npy"
        if not pred_path.exists():
            print(f"epoch {epoch_id} image {idx}: Missing prediction file: {pred_path}")
            continue

        source_path = source_backup_dir / name
        if not source_path.exists():
            print(f"epoch {epoch_id} image {idx}: Missing source backup image: {source_path}")
            continue

        source_img = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if source_img is None:
            print(f"epoch {epoch_id} image {idx}: Could not read source image: {source_path}")
            continue

        print(f"epoch {epoch_id} image {idx}: {name}", flush=True)
        # Scoring one image must never abort the sweep: a run covers every variant x epoch x
        # image, so a single degenerate sample would otherwise throw away hours of work.
        try:
            pred_points = load_pointset(pred_path)
            advanced_metrics = compute_all_advanced_metrics(
                pred_points,
                source_img.astype(np.float64) / 255.0,
                mc_approx=args.mc_approx,
            )
            payload = serialize_metrics(advanced_metrics)
        except Exception as exc:
            print(f"epoch {epoch_id} image {idx}: FAILED ({type(exc).__name__}: {exc}) -- skipping",
                  flush=True)
            continue

        out_json = output_epoch_dir / f"{stem}.json"
        out_json.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)

    if not args.run_metrics_p1 and not args.run_metrics_p2:
        print("Nothing to do: both --no-run-metrics-p1 and --no-run-metrics-p2 were given")
        return 2

    val_names, source_backup_dir, target_backup_dir = load_validation_data(out_base, args.target_name)
    val_names = limit_validation_images(val_names, args.num_samples)
    print(f"Using {len(val_names)} validation images for scoring")

    model_root = out_base / RESULTS_DIR
    if not model_root.exists():
        print(f"Model results directory does not exist: {model_root}")
        return 2

    epoch_dirs = find_epoch_dirs(model_root, pattern_filter=args.epochs)
    if len(epoch_dirs) == 0:
        print(f"No epoch_*_npy directories found in {model_root}")
        return 2

    # Keep only epoch dirs at multiples of --every-epoch (0/negative => all)
    epoch_dirs = filter_by_every_epoch(epoch_dirs, args.every_epoch)

    print(f"Model {RESULTS_DIR}: found {len(epoch_dirs)} epoch directories")
    print(f"Part 1 (metrics): {'ON' if args.run_metrics_p1 else 'OFF'}   "
          f"Part 2 (losses): {'ON' if args.run_metrics_p2 else 'OFF'}")
    if args.dry_run:
        for epoch_dir in epoch_dirs:
            eid = epoch_id_from_name(epoch_dir.name)
            print(f"DRY epoch {eid} -> {model_root / f'epoch_{eid}_json'}")
        return 0

    # Part 2 needs the variant's weights per epoch; fail fast before any work is done.
    ckpt_map = {}
    if args.run_metrics_p2 and not args.loss_approx:
        ckpt_map = find_epoch_checkpoints(WEIGHTS_DIR)
        missing = [epoch_id_from_name(d.name) for d in epoch_dirs
                   if epoch_id_from_name(d.name) not in ckpt_map]
        if missing:
            print(f"LOSS_APPROX is False but {WEIGHTS_DIR} has no checkpoint for epochs: "
                  f"{', '.join(missing)}")
            return 2

    # ---- Part 1: per-image geometry metrics -------------------------------
    if args.run_metrics_p1:
        print("\n=== Part 1: per-image metrics ===")
        for epoch_dir in epoch_dirs:
            eid = epoch_id_from_name(epoch_dir.name)
            output_epoch_dir = model_root / f"epoch_{eid}_json"
            print(f"Scoring epoch {eid} from {epoch_dir} -> {output_epoch_dir}", flush=True)
            score_epoch_dir(epoch_dir, output_epoch_dir, val_names,
                            source_backup_dir, args, eid)

    # ---- Part 2: the batched M6 losses ------------------------------------
    if args.run_metrics_p2:
        print("\n=== Part 2: losses ===")
        timesteps_map = load_timesteps(args, out_base)
        loss_path = model_root / args.loss_file
        diffusion = build_diffusion_schedule(args)

        rows = None
        if args.loss_approx:
            print("Approximate-loss mode: M6 scored on point sets (no model forward pass)")
        else:
            print(f"True-loss mode on {args.device}: matched {len(ckpt_map)} checkpoints "
                  f"in {WEIGHTS_DIR}")
            # Dataset-side tensors do not depend on the checkpoint: build them once.
            device = torch.device(args.device)
            rows = build_loss_inputs(val_names, source_backup_dir, target_backup_dir,
                                     timesteps_map, args, device)
            if not rows:
                print("No usable validation rows for part 2")
                return 2
            print(f"Built {len(rows)} rows "
                  f"({len(val_names)} images x timesteps), batch size {args.loss_batch_size}")

        for epoch_dir in epoch_dirs:
            eid = epoch_id_from_name(epoch_dir.name)
            print(f"Losses for epoch {eid}", flush=True)
            if args.loss_approx:
                minsnr_loss, kde_loss, active = compute_epoch_losses_approx(
                    epoch_dir, val_names, target_backup_dir, diffusion, args, timesteps_map)
            else:
                print(f"  loading weights: {ckpt_map[eid]}", flush=True)
                models = build_true_loss_models(ckpt_map[eid], args)
                minsnr_loss, kde_loss, active = compute_epoch_losses_true(rows, models, args)

            values = {
                "M6_minsnr_loss": minsnr_loss,
                "M6_v2_minsnr_kde_loss": minsnr_loss + kde_loss,
            }
            update_loss_file(loss_path, eid, values)
            print(f"  M6_minsnr_loss={minsnr_loss:.6e}  kde={kde_loss:.6e}  "
                  f"M6_v2={values['M6_v2_minsnr_kde_loss']:.6e}  "
                  f"(density active on {100.0 * active:.0f}% of rows)", flush=True)

        print(f"Wrote losses to {loss_path}")

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
