#!/usr/bin/env python3
"""Stage 0: stage the two input images for the grid-size comparison.

Copies into OUTPUT_DIR/source/:

    z_validation_data/Icons-50_1024/source/emoji-one_4_monkey.png   a normal icon
    z_validation_data/maps/stress_test_density.png                  a synthetic density map

Stage 1 then runs sample_control.py over these at each grid size; stage 2 merges the
results into one figure. The two images are listed explicitly (not a manifest prefix):
this comparison is a qualitative figure with hand-picked subjects, not a metric over a
validation split.

    python experiments/grid_size_results_stage_0.py
    python experiments/grid_size_results_stage_0.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR = BASE / "grid_size_results"

# (source path relative to outputs/, destination filename)
IMAGES = [
    "z_validation_data/Icons-50_1024/source/emoji-one_4_monkey.png",
    "z_validation_data/maps/stress_test_density.png",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Copy the grid-size comparison inputs into OUTPUT_DIR/source/.")
    ap.add_argument("--output", default=str(OUTPUT_DIR), help="Grid-size results folder.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing.")
    return ap.parse_args()


def main():
    args = parse_args()
    dst_dir = Path(args.output) / "source"

    srcs = [BASE / rel for rel in IMAGES]
    missing = [str(p) for p in srcs if not p.exists()]
    if missing:
        print("Error: input image(s) not found:")
        for m in missing:
            print(f"    {m}")
        return 1

    print(f"dst: {dst_dir}\n")
    for p in srcs:
        print(f"  {p.name}  <- {p}")

    if args.dry_run:
        print("\nDRY RUN: nothing copied.")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in srcs:
        shutil.copy2(p, dst_dir / p.name)

    print(f"\nCopied {len(srcs)} images -> {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
