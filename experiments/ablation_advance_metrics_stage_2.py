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
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.Config import ParseSampleConfig
from utils.stippling_metrics_advance import compute_all_advanced_metrics


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
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/target"
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e500_b50_1024"

MIN_SNR_GAMMA = 5.0
SPLIT_SEED = 42
VAL_SPLIT = 0.1
# NUM_SAMPLES = -1
NUM_SAMPLES = 50
# EVERY_EPOCH = -1  # Keep all checkpoints by default
EVERY_EPOCH = 500

# Component-2 (density-match KDE) loss constants -- mirror control_v4/train_control.py.
DENSITY_LOSS_WEIGHT = 0.8
DENSITY_KDE_GRID = 32
DENSITY_KDE_SIGMA_PX = 1.0

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# M6_minsnr_loss        = Component-1 proxy only (MinSNR-weighted density MSE).
# M6_v2_minsnr_kde_loss = full training-loss proxy: Component 1 + w * Component-2 KDE loss.
METRIC_ORDER = [
    "M1_v1_cvt_energy",
    "M1_v2_power_cvt_energy",
    "M2_v0_voronoi_mass_cv",
    "M2_v1_capacity_delta_c",
    "M2_v2_power_displacement",
    "M2_v3_power_cell_cap_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
    "M6_minsnr_loss",
    "M6_v2_minsnr_kde_loss",
]
MC_APPROX = False


def parse_args():
    parser = argparse.ArgumentParser(description="Score stage 1 point outputs with advanced metrics")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Base output folder for stage 1 / stage 2 artifacts")
    parser.add_argument("--source", default=SOURCE_DIR, help="Original source images folder")
    parser.add_argument("--target", default=TARGET_DIR, help="Original target images folder")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Fraction for validation split")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Number of validation samples to score in order; -1 means use all")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="Deterministic seed for split")
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
    parser.add_argument("--mc-approx", action=argparse.BooleanOptionalAction, default=MC_APPROX,
                        help="Use the legacy Monte Carlo approximation for the advanced metrics. Default is now exact/deterministic quadrature; pass --mc-approx only to reproduce previously published approximate numbers.")
    parser.add_argument("--samples-per-image", type=int, default=1,
                        help="Number of MinSNR timestep samples per validation image (MC approximation)")
    parser.add_argument("--timestep-file", type=str, default=None,
                        help="Optional path to load/save the per-image timesteps JSON. If not set, will write to output/validation_data/minsnr_timesteps.json")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def list_images(folder):
    p = Path(folder)
    return [str(f) for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in IMAGE_EXTS]


def select_validation_images(all_images, val_frac, seed):
    imgs = sorted(all_images)
    n_total = len(imgs)
    val_len = int(n_total * float(val_frac))
    val_len = min(max(val_len, 0), max(n_total - 1, 0))
    train_len = n_total - val_len

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(int(seed))).tolist()
    val_indices = indices[train_len:]
    return [imgs[i] for i in val_indices]


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


def _build_name_map(root_dir):
    root = Path(root_dir)
    name_map = {}
    if not root.exists():
        return name_map
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            name_map[path.name] = path
    return name_map


def backup_validation_images(val_names, out_base, source_dir, target_dir):
    out_base_p = Path(out_base)
    val_data_dir = out_base_p / "validation_data"
    source_backup_dir = val_data_dir / "source"
    target_backup_dir = val_data_dir / "target"
    manifest_path = out_base_p / "validation_manifest.json"

    source_backup_dir.mkdir(parents=True, exist_ok=True)
    target_backup_dir.mkdir(parents=True, exist_ok=True)

    source_map = _build_name_map(source_dir)
    target_map = _build_name_map(target_dir)

    for name in val_names:
        source_src = source_map.get(name)
        if source_src is not None:
            source_dst = source_backup_dir / name
            if not source_dst.exists():
                shutil.copy2(source_src, source_dst)

        target_src = target_map.get(name)
        if target_src is not None:
            target_dst = target_backup_dir / name
            if not target_dst.exists():
                shutil.copy2(target_src, target_dst)

    manifest_path.write_text(json.dumps(val_names, indent=2))
    return source_backup_dir, target_backup_dir


def ensure_validation_data(out_base, source_dir, target_dir, val_frac, seed):
    out_base_p = Path(out_base)
    val_data_dir = out_base_p / "validation_data"
    source_backup_dir = val_data_dir / "source"
    target_backup_dir = val_data_dir / "target"
    manifest_path = out_base_p / "validation_manifest.json"

    if manifest_path.exists():
        val_names = json.loads(manifest_path.read_text())
        backup_validation_images(val_names, out_base, source_dir, target_dir)
        return val_names, source_backup_dir, target_backup_dir

    all_images = list_images(source_dir)
    if len(all_images) == 0:
        raise FileNotFoundError(f"No source images found in {source_dir}")

    val_images = select_validation_images(all_images, val_frac, seed)
    val_names = [Path(img).name for img in val_images]
    backup_validation_images(val_names, out_base, source_dir, target_dir)
    return val_names, source_backup_dir, target_backup_dir


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


def image_to_density(image_u8, grid_size):
    image_01 = np.asarray(image_u8, dtype=np.float64) / 255.0
    target = 1.0 - np.clip(image_01, 0.0, 1.0)
    if target.shape[0] != grid_size or target.shape[1] != grid_size:
        target = cv2.resize(target, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    mass = target.sum()
    if mass <= 1e-12:
        return np.zeros((grid_size, grid_size), dtype=np.float64)
    return target / mass


def build_diffusion_schedule(args):
    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"])
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()
    return diffusion


def generate_or_load_timesteps(val_names, args, out_base: Path):
    """
    Returns a dict mapping filename -> list of integer timestep indices.
    If args.timestep_file is provided and exists, load it. Otherwise use
    out_base/minsnr_timesteps.json (next to validation_manifest.json). If it
    does not exist, generate timesteps with a deterministic RNG seeded by
    args.seed and save to that path.
    """
    if args.timestep_file:
        tfp = Path(args.timestep_file)
    else:
        tfp = out_base / "minsnr_timesteps.json"

    if tfp.exists():
        loaded = json.loads(tfp.read_text())
        return {k: [int(x) for x in v] for k, v in loaded.items()}

    # generate
    rng = np.random.RandomState(int(args.seed))
    max_t = max(1, int(args.timesteps) - 1)
    samples = max(1, int(args.samples_per_image))
    mapping = {}
    for name in val_names:
        if max_t <= 0:
            ts = [0] * samples
        else:
            ts = [int(rng.randint(0, max_t + 1)) for _ in range(samples)]
        mapping[name] = ts

    tfp.parent.mkdir(parents=True, exist_ok=True)
    tfp.write_text(json.dumps(mapping, indent=2))
    print(f"Wrote timestep map to {tfp}")
    return mapping


def compute_minsnr_proxy(pred_points, target_image_u8, diffusion, args, timesteps=None):
    """
    Compute MinSNR-weighted loss. If `timesteps` is None, use args.min_snr_timestep
    as a single-element list. If `timesteps` is a list, compute weighted MSE per
    timestep and return the average across samples.
    """
    pred_density = render_points_to_density(pred_points, args.grid_size)
    target_density = image_to_density(target_image_u8, args.grid_size)
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


def compute_kde_proxy(pred_points, target_image_u8, args):
    """Component-2 (density-match) proxy, already scaled by density_loss_weight:
    density_loss_weight * MSE between the mean-1 KDE of the predicted points and the
    mean-1 target density from the target image. Mirrors the  density_w * d_loss  term
    of train_control.density_match_loss (same kernel + mean-1 normalisation), with the
    teacher point set replaced by the target-image density field (no GT points here).

    M6_v2_minsnr_kde_loss = M6_minsnr_loss + compute_kde_proxy(...).
    """
    K = int(args.density_kde_grid)
    kde_pred = gaussian_kde_map_np(pred_points, K, args.density_kde_sigma_px)   # mean 1
    tgt = image_to_density(target_image_u8, K)          # sum = 1 (or zeros)
    s = tgt.sum()
    kde_true = tgt * (tgt.size / s) if s > 1e-8 else np.ones((K, K), dtype=np.float64)
    kde_mse = float(np.mean((kde_pred - kde_true) ** 2))
    return float(args.density_loss_weight) * kde_mse


def serialize_metrics(metrics):
    ordered = {}
    for key in METRIC_ORDER:
        if key in metrics:
            ordered[key] = float(metrics[key])
    return ordered


def score_epoch_dir(epoch_dir, output_epoch_dir, val_names, source_backup_dir, target_backup_dir, diffusion, args, timesteps_map, epoch_id):
    output_epoch_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir = Path(epoch_dir)
    total = len(val_names)

    for idx, name in enumerate(val_names, start=1):
        stem = Path(name).stem
        pred_path = epoch_dir / f"{stem}.npy"
        if not pred_path.exists():
            print(f"epoch {epoch_id} image {idx}: Missing prediction file: {pred_path}")
            continue

        source_path = source_backup_dir / name
        target_path = target_backup_dir / name
        if not source_path.exists():
            print(f"epoch {epoch_id} image {idx}: Missing source backup image: {source_path}")
            continue
        if not target_path.exists():
            print(f"epoch {epoch_id} image {idx}: Missing target backup image: {target_path}")
            continue

        source_img = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if source_img is None:
            print(f"epoch {epoch_id} image {idx}: Could not read source image: {source_path}")
            continue
        if target_img is None:
            print(f"epoch {epoch_id} image {idx}: Could not read target image: {target_path}")
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
            ts = None
            if timesteps_map is not None:
                ts = timesteps_map.get(name)
            minsnr_loss = compute_minsnr_proxy(pred_points, target_img, diffusion, args, timesteps=ts)
            kde_loss = compute_kde_proxy(pred_points, target_img, args)
            payload["M6_minsnr_loss"] = minsnr_loss
            payload["M6_v2_minsnr_kde_loss"] = minsnr_loss + kde_loss
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

    val_names, source_backup_dir, target_backup_dir = ensure_validation_data(
        out_base,
        args.source,
        args.target,
        args.val_split,
        args.seed,
    )
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
    if args.dry_run:
        for epoch_dir in epoch_dirs:
            eid = epoch_id_from_name(epoch_dir.name)
            print(f"DRY epoch {eid} -> {model_root / f'epoch_{eid}_json'}")
        return 0

    diffusion = build_diffusion_schedule(args)

    # generate or load per-image timesteps mapping for MinSNR MC approximation
    timesteps_map = generate_or_load_timesteps(val_names, args, out_base)

    for epoch_dir in epoch_dirs:
        eid = epoch_id_from_name(epoch_dir.name)
        output_epoch_dir = model_root / f"epoch_{eid}_json"
        print(f"Scoring epoch {eid} from {epoch_dir} -> {output_epoch_dir}", flush=True)
        score_epoch_dir(
            epoch_dir,
            output_epoch_dir,
            val_names,
            source_backup_dir,
            target_backup_dir,
            diffusion,
            args,
            timesteps_map,
            eid,
        )

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
