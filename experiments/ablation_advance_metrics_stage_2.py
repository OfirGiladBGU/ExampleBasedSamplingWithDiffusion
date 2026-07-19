"""Stage 2 ablation scorer.

This script reads the point-set outputs produced by stage 1 from
`OUTPUT_DIR/{RESULTS_DIR}/epoch_*_npy/`, computes the advanced metrics per
example, and writes JSON files to `OUTPUT_DIR/{RESULTS_DIR}/epoch_*_json/`.

The per-example JSON contains:
    M1_cvt_energy
    M2_voronoi_mass_cv
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

# Vanilla
WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_vanilla/checkpoints"
RESULTS_DIR = "vanilla"
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

# GECCO
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_gecco/checkpoints"
# RESULTS_DIR = "gecco"
# GRID_SIZE = 32
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = False
# EVAL_TIMESTEPS = 1000
# INFER_TRUNCATION_RATIO = 1.0
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Adaptive gate injection
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_agi/checkpoints"
# RESULTS_DIR = "agi"
# GRID_SIZE = 32
# ENABLE_GECCO = False
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# INFER_TRUNCATION_RATIO = 1.0
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Full
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints"
# RESULTS_DIR = "full"
# GRID_SIZE = 32
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# INFER_TRUNCATION_RATIO = 1.0
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Full + SDEdit
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints"
# RESULTS_DIR = "sdedit"
# GRID_SIZE = 32
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# INFER_TRUNCATION_RATIO = 0.3
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Full + SDEdit + resample jumps
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints"
# RESULTS_DIR = "sdedit_resample"
# GRID_SIZE = 32
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# INFER_TRUNCATION_RATIO = 0.3
# RESAMPLE_JUMPS = 2
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# Default folders
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e400_b50_1024"

MIN_SNR_GAMMA = 5.0
SPLIT_SEED = 42
VAL_SPLIT = 0.1
# NUM_SAMPLES = -1
NUM_SAMPLES = 50
# NUM_EPOCHS = -1
NUM_EPOCHS = 400

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

METRIC_ORDER = [
    "M1_cvt_energy",
    "M2_voronoi_mass_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
    "M6_minsnr_loss",
]


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
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of epoch directories to process in sorted order; -1 means use all")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be processed")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--base-ckpt", default=CKPT_PATH)
    parser.add_argument("--timesteps", type=int, default=EVAL_TIMESTEPS)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--min-snr-gamma", type=float, default=MIN_SNR_GAMMA)
    parser.add_argument("--mc-approx", action=argparse.BooleanOptionalAction, default=True)
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


def limit_checkpoints(ckpts, num_epochs):
    if int(num_epochs) < 0:
        return ckpts
    return ckpts[: int(num_epochs)]


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
        payload["M6_minsnr_loss"] = compute_minsnr_proxy(pred_points, target_img, diffusion, args, timesteps=ts)

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

    # Apply numeric epoch limiting if requested (NUM_EPOCHS < 0 => use all)
    epoch_dirs = limit_checkpoints(epoch_dirs, args.num_epochs)

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
