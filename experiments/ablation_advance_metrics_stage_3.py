"""Stage 3: aggregate per-example JSONs into per-epoch averages and a single metrics_avg.json

This script scans `OUTPUT_DIR/{RESULTS_DIR}/epoch_*_json/`, computes the mean
for each metric across examples in each epoch, writes `epoch_{id}_metrics.json`
next to each epoch folder, and produces `metrics_avg.json` in the model root
that maps metric -> { epoch_id: avg }. This layout is convenient for plotting
metric curves over epochs.

The math is simple mean; per-epoch files are written to avoid re-reading many
small files when re-aggregating.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RESULTS_DIR = "vanilla"
# RESULTS_DIR = "unfrozen"
# RESULTS_DIR = "gecco"
# RESULTS_DIR = "agi"
# RESULTS_DIR = "full"
# RESULTS_DIR = "sdedit"


# Default folders
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e400_b50_1024"

# NUM_SAMPLES = -1
NUM_SAMPLES = 50
# NUM_EPOCHS = -1
NUM_EPOCHS = 400

METRIC_ORDER = [
    "M1_cvt_energy",
    "M2_voronoi_mass_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
    "M6_minsnr_loss",
]


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate per-example JSON metrics into per-epoch averages")
    p.add_argument("--output", default=OUTPUT_DIR, help="Base output folder used by stages 1/2")
    p.add_argument("--results-dir", default=RESULTS_DIR, help="Subfolder for model results (RESULTS_DIR)")
    p.add_argument("--epochs", default="all", help="'all' or comma-separated substrings to match epoch dirs")
    p.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                   help=f"Number of epoch directories to process in sorted order; -1 means use all (default: {NUM_EPOCHS})")
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                   help=f"Limit number of examples per epoch to aggregate; -1 means use all (default: {NUM_SAMPLES})")
    p.add_argument("--write-per-epoch", action="store_true", help="Also write per-epoch epoch_{id}_metrics.json files")
    p.add_argument("--metrics-file", default="metrics_avg.json", help="Name of the aggregated metrics file to write in model root")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def limit_validation_images(val_images, num_samples):
    if int(num_samples) < 0:
        return val_images
    return val_images[: int(num_samples)]


def limit_checkpoints(ckpts, num_epochs):
    if int(num_epochs) < 0:
        return ckpts
    return ckpts[: int(num_epochs)]


def find_epoch_json_dirs(model_root, pattern_filter=None):
    root = Path(model_root)
    candidates = [p for p in root.glob("epoch_*_json") if p.is_dir()]

    if pattern_filter and pattern_filter != "all":
        wanted = [s.strip() for s in pattern_filter.split(",") if s.strip()]
        candidates = [c for c in candidates if any(w in c.name for w in wanted)]

    def epoch_of_name(name):
        m = re.search(r"epoch_(\d{2,6})_json", name)
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


def load_manifest_order(out_base):
    """Load validation_manifest.json to get the canonical file order."""
    manifest_path = Path(out_base) / "validation_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return []


def aggregate_epoch(epoch_dir: Path, manifest_names=None):
    """Read all JSONs under epoch_dir and return dict metric->mean.
    
    If manifest_names is provided, process only those files in that order.
    Otherwise, process all files in sorted order.
    """
    all_files = {p.stem: p for p in epoch_dir.glob("*.json") if p.is_file()}
    
    # If manifest provided, use it to enforce order and filter
    if manifest_names:
        files = []
        for name in manifest_names:
            stem = Path(name).stem
            if stem in all_files:
                files.append(all_files[stem])
    else:
        files = sorted(all_files.values())
    
    if len(files) == 0:
        return {}

    sums = {}
    counts = {}
    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        for k, v in data.items():
            try:
                fv = float(v)
            except Exception:
                continue
            sums[k] = sums.get(k, 0.0) + fv
            counts[k] = counts.get(k, 0) + 1

    avgs = {}
    for k in sums:
        avgs[k] = float(sums[k] / max(1, counts.get(k, 1)))
    return avgs


def main():
    args = parse_args()
    out_base = Path(args.output)
    model_root = out_base / args.results_dir
    if not model_root.exists():
        print(f"Model results directory does not exist: {model_root}")
        return 2

    epoch_dirs = find_epoch_json_dirs(model_root, pattern_filter=args.epochs)
    if len(epoch_dirs) == 0:
        print(f"No epoch_*_json directories found in {model_root}")
        return 2

    print(f"Found {len(epoch_dirs)} epoch json directories in {model_root}")
    if args.dry_run:
        for p in epoch_dirs:
            print(p)
        return 0

    # Apply numeric epoch limiting if requested
    epoch_dirs = limit_checkpoints(epoch_dirs, getattr(args, "num_epochs", -1))
    
    # Load manifest to enforce correct ordering of examples
    manifest = load_manifest_order(out_base)
    manifest_limited = limit_validation_images(manifest, getattr(args, "num_samples", -1))
    print(f"Using {len(manifest_limited)} examples from manifest (in order)")

    # Build metric -> { epoch: avg }
    metric_time_series = {m: {} for m in METRIC_ORDER}

    for idx, ed in enumerate(epoch_dirs, start=1):
        eid = epoch_id_from_name(ed.name)
        print(f"[{idx}/{len(epoch_dirs)}] Aggregating epoch {eid}")
        # Aggregate using manifest order to limit and order samples
        avgs = aggregate_epoch(ed, manifest_names=manifest_limited)
        if len(avgs) == 0:
            avgs = {}
        if args.write_per_epoch:
            target_path = model_root / f"epoch_{eid}_metrics.json"
            target_path.write_text(json.dumps({k: avgs.get(k, None) for k in METRIC_ORDER}, indent=2))

        for k in METRIC_ORDER:
            if k in avgs:
                metric_time_series[k][eid] = float(avgs[k])

    metrics_file = model_root / args.metrics_file
    metrics_file.write_text(json.dumps(metric_time_series, indent=2))
    print(f"Wrote aggregated metrics to {metrics_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
