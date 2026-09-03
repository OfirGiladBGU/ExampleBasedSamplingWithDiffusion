"""Stage 2 ablation scorer.

This script reads the point-set outputs produced by stage 1 from
`OUTPUT_DIR/{RESULTS_DIR}/epoch_*_npy/`, computes the advanced geometry metrics per
example, and writes JSON files to `OUTPUT_DIR/{RESULTS_DIR}/epoch_*_json/`.

The per-example JSON contains:
    M1_v1_cvt_energy / M1_v2_power_cvt_energy
    M2_v1_capacity_delta_c / M2_v2_power_displacement / M2_v3_power_cell_cap_cv
    M3_emd_distance
    M4_sinkhorn_ot_cost
    M5_spatial_measure_rho_mean

The M6 training losses used to be computed here too, but they measure the MODEL (a
noise-prediction / density-match loss against the training objective), not this
exported point set, and are reported as a single set-level number per epoch rather
than per image -- both a poor fit for this stage. They now live in their own pipeline:
    ablation_loss_results_stage_0.py / _1.py / _2.py
which reads control_v4/train_outputs_*/losses_log.json (written live during training)
directly, with no re-inference needed.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stippling_metrics_advance import compute_all_advanced_metrics


# Vanilla
RESULTS_DIR = "vanilla"

# Unfrozen
# RESULTS_DIR = "unfrozen"

# GECCO
# RESULTS_DIR = "gecco"

# Adaptive gate injection
# RESULTS_DIR = "agi"

# Full
# RESULTS_DIR = "full"

# Full + SDEdit (shares full's exported points and weights; only inference truncation differs)
# RESULTS_DIR = "sdedit"


# Default folders
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e500_b50_1024"

# NUM_SAMPLES = -1
NUM_SAMPLES = 50
# EVERY_EPOCH = -1  # Keep all checkpoints by default
EVERY_EPOCH = 500

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
MC_APPROX = False


def parse_args():
    parser = argparse.ArgumentParser(description="Score stage 1 point outputs with advanced metrics")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Base output folder for stage 1 / stage 2 artifacts")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Number of validation samples to score in order; -1 means use all")
    parser.add_argument("--epochs", default="all", help="'all' or comma-separated substrings matching epoch dir names")
    parser.add_argument("--every-epoch", type=int, default=EVERY_EPOCH,
                        help="Keep only checkpoints whose epoch (from '*ep{N}') is a multiple of this; 0 = all")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be processed")
    parser.add_argument("--mc-approx", action=argparse.BooleanOptionalAction, default=MC_APPROX,
                        help="Use the legacy Monte Carlo approximation for the advanced metrics. Default is now exact/deterministic quadrature; pass --mc-approx only to reproduce previously published approximate numbers.")
    return parser.parse_args()


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


def load_validation_data(out_base):
    """Read the shared resources staged by ablation_advance_metrics_stage_0.py:

        OUTPUT_DIR/resources/validation_manifest.json
        OUTPUT_DIR/resources/source/
    """
    res = Path(out_base) / "resources"
    manifest_path = res / "validation_manifest.json"
    source_dir = res / "source"
    missing = [str(p) for p in (manifest_path, source_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing staged resources: " + ", ".join(missing)
            + "  -- run experiments/ablation_advance_metrics_stage_0.py first.")
    return json.loads(manifest_path.read_text()), source_dir


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


def serialize_metrics(metrics):
    ordered = {}
    for key in METRIC_ORDER:
        if key in metrics:
            ordered[key] = float(metrics[key])
    return ordered


def score_epoch_dir(epoch_dir, output_epoch_dir, val_names, source_backup_dir, args, epoch_id):
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

    val_names, source_backup_dir = load_validation_data(out_base)
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

    for epoch_dir in epoch_dirs:
        eid = epoch_id_from_name(epoch_dir.name)
        output_epoch_dir = model_root / f"epoch_{eid}_json"
        print(f"Scoring epoch {eid} from {epoch_dir} -> {output_epoch_dir}", flush=True)
        score_epoch_dir(epoch_dir, output_epoch_dir, val_names, source_backup_dir, args, eid)

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
