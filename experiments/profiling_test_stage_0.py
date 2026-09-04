# COPY only the monkey image from the "z_validation_data/Icons-50_1024/source" folder to the "experiments/outputs/teaser_icons_stage_0/source" folder.
# and build the dir needed for stage 1
# later fix stage 1 to point to this image in the "experiments/outputs/profiling_test"


#!/usr/bin/env python3
"""Stage 0: stage the two input images for the profiling test.

Copies into OUTPUT_DIR/source/:

    z_validation_data/Icons-50_1024/source/emoji-one_4_monkey.png   a normal icon
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
