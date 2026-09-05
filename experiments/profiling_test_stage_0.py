#!/usr/bin/env python3
"""profiling_test_stage_0.py -- Part 0: stage the profiling input image.

Copies the single image the timing runs are profiled on out of the shared validation
folder:

    z_validation_data/Icons-50_1024/source/<image>.png
        -> OUTPUT_DIR/source/<image>.png

That destination is exactly INPUT_IMAGE_PATH in profiling_test_stage_1.py, so running this
first is what makes stage 1 runnable from a clean checkout.

There is no selection to make here and so no validation manifest: profiling measures how
runtime scales with the grid size, not how output quality varies across images, so stage 1
sweeps GRID_SIZES over this ONE fixed image. It is staged rather than read in place only so
a profiling run is self-contained and cannot be perturbed by edits to the shared folder.

    python experiments/profiling_test_stage_0.py
    python experiments/profiling_test_stage_0.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR = BASE / "profiling_test"

# (source path relative to outputs/, destination filename)
IMAGES = [
    "z_validation_data/Icons-50_1024/source/emoji-one_4_monkey.png",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Copy the profiling test inputs into OUTPUT_DIR/source/.")
    ap.add_argument("--output", default=str(OUTPUT_DIR), help="Profiling test results folder.")
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
