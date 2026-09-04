#!/usr/bin/env python
"""Stage 1: run sample_control.py over the staged images at several grid sizes.

Same shape as profiling_test.py, with two differences: it sweeps BOTH staged images
(not one), and it passes NO time-tracking flags -- this stage produces point sets for a
qualitative figure, it does not measure speed. Use profiling_test.py for timing.

Reads   OUTPUT_DIR/source/<image>            (staged by grid_size_results_stage_0.py)
Writes  OUTPUT_DIR/sample_outputs_<grid>/<stem>/npy/<stem>.npy   (+ png/)

which is the layout grid_size_results_stage_2.py reads to build the figure.

Grid size G yields G*G points: 16->256, 24->576, 32->1024, 48->2304, 64->4096.

    python experiments/grid_size_results_stage_1.py
    python experiments/grid_size_results_stage_1.py --dry-run
    python experiments/grid_size_results_stage_1.py --device cpu
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = EXPERIMENTS_DIR.parent / "control_v4" / "sample_control.py"
OUTPUT_DIR = "experiments/outputs/grid_size_results"

# Grid sizes to sweep (G*G points each).
GRID_SIZES = [16, 24, 32, 48, 64]

# Per-grid output folder under OUTPUT_DIR.
OUTPUT_SUBDIR_FORMAT = "sample_outputs_{}"

DEVICE = "cuda"

# No --track_time / --track_time_full / --profile_trace here: this stage is not a
# timing run, and those flags only add profiling overhead and stray _time.txt files.
ADDITIONAL_ARGS = []


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Sweep grid sizes over the staged images.")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Grid-size results folder.")
    ap.add_argument("--grid-sizes", type=int, nargs="+", default=GRID_SIZES)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--dry-run", action="store_true", help="Print the commands; run nothing.")
    return ap.parse_args()


def main():
    args = parse_args()

    if not SCRIPT_PATH.exists():
        print(f"Error: sample_control.py not found at {SCRIPT_PATH}")
        return 1

    source_dir = Path(args.output) / "source"
    if not source_dir.is_dir():
        print(f"Error: {source_dir} not found -- run experiments/grid_size_results_stage_0.py first.")
        return 1

    images = sorted(p for p in source_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not images:
        print(f"Error: no images under {source_dir}")
        return 1

    print(f"script    : {SCRIPT_PATH}")
    print(f"images    : {[p.name for p in images]}")
    print(f"grid sizes: {args.grid_sizes}  (points: {[g * g for g in args.grid_sizes]})")
    print(f"device    : {args.device}\n")

    failures = []
    for grid_size in args.grid_sizes:
        out_dir = str(Path(args.output) / OUTPUT_SUBDIR_FORMAT.format(grid_size))
        for img in images:
            print(f"\n{'=' * 70}")
            print(f"grid_size={grid_size}  ({grid_size * grid_size} pts)  image={img.name}")
            print(f"{'=' * 70}")

            cmd = [
                sys.executable, str(SCRIPT_PATH),
                "--grid_size", str(grid_size),
                "--image_path", str(img),
                "--output_dir", out_dir,
                "--device", args.device,
            ] + ADDITIONAL_ARGS

            print(f"Command: {' '.join(cmd)}\n")
            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\nWarning: grid_size={grid_size} {img.name} exited with code {result.returncode}")
                failures.append((grid_size, img.name))
            else:
                print(f"\nCompleted: grid_size={grid_size} {img.name}")

    print(f"\n{'=' * 70}")
    if args.dry_run:
        print(f"DRY RUN: {len(args.grid_sizes) * len(images)} run(s) would be executed.")
    elif failures:
        print(f"Done with {len(failures)} failure(s): {failures}")
    else:
        print("All grid sizes processed!")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
