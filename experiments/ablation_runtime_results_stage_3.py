"""ablation_runtime_results_stage_3.py -- Part 3: merge the per-method means.

Reads each method's runtime_avg.json (written by stage 2) and merges them into one file:

    OUTPUT_DIR/runtime_avg_merged.json = {method: {timing_label: mean_seconds, ...}, ...}

Also prints a method x timing table. Methods are ordered by RESULT_DIR_LIST (any extra
discovered methods are appended); DIR_TO_LABEL_MAP only prettifies the printed/merged names.
"""

import argparse
import json
from pathlib import Path

OUTPUT_DIR = "experiments/outputs/ablation_runtime_results"
AVG_NAME = "runtime_avg.json"

# Preferred order for the merged table (extras are appended in discovery order).
RESULT_DIR_LIST = ["vanilla", "unfrozen", "gecco", "agi", "full", "sdedit"]
DIR_TO_LABEL_MAP = {
    "vanilla": "Vanilla", "unfrozen": "Unfrozen", "gecco": "GECCO",
    "agi": "Gated", "full": "Full", "sdedit": "SDEdit",
}

# Timing rows to print (in order); any others found are appended.
TIMING_ORDER = [
    "Smart Init Time",
    "Denoising Time",
    "Total Inference Time",
    "Optimal Transport & Saving Time",
    "Total Execution Time",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Merge per-method runtime means into one file + table.")
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--avg-name", default=AVG_NAME)
    ap.add_argument("--out-name", default="runtime_avg_merged.json")
    return ap.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output)

    # discover methods that have a runtime_avg.json; keep RESULT_DIR_LIST order first
    have = {d.name for d in out_base.iterdir() if d.is_dir() and (d / args.avg_name).exists()}
    ordered = [m for m in RESULT_DIR_LIST if m in have] + sorted(have - set(RESULT_DIR_LIST))
    if not ordered:
        print(f"No {args.avg_name} found under {out_base} (run stage 2 first)"); return 2

    merged = {}
    timing_labels = list(TIMING_ORDER)
    for m in ordered:
        data = json.loads((out_base / m / args.avg_name).read_text())
        label = DIR_TO_LABEL_MAP.get(m, m)
        merged[label] = {lbl: vals["mean"] for lbl, vals in data.items()}
        for lbl in data:
            if lbl not in timing_labels:
                timing_labels.append(lbl)

    merged_path = out_base / args.out_name
    merged_path.write_text(json.dumps(merged, indent=2))

    # printed table: rows = timing labels, cols = methods
    names = [DIR_TO_LABEL_MAP.get(m, m) for m in ordered]
    w = max(28, *(len(t) for t in timing_labels)) + 1
    print(f"\n{'timing (s)':<{w}}" + "".join(f"{n:>12s}" for n in names))
    for lbl in timing_labels:
        row = "".join(f"{merged[n].get(lbl, float('nan')):>12.4f}" for n in names)
        print(f"{lbl:<{w}}{row}")
    print(f"\nSaved merged runtimes to {merged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
