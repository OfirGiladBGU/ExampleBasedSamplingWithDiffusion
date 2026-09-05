"""compare_advance_metrics_stage_0.py -- Part 0: stage the conditioning map.

Copies the density map that stage 1 measures against out of the shared maps folder:

    z_validation_data/maps/<map>.png
        -> experiments/outputs/images_results_metrics/<example>/source/<map>.png

That destination is exactly DEFAULT_INPUT_IMAGE in compare_advance_metrics_stage_1.py, so
running this first is what makes stage 1 runnable from a clean checkout. Only the source
image is staged -- the per-method point sets (target_*_1024/<map>.npy) are produced by the
samplers, not by this script.

The maps are synthetic analytic densities rather than dataset images, so there is no
validation manifest here and nothing to select: the example folder names the single map it
uses. To stage a different example, switch the active MAPS block below (they mirror the
DEFAULT_INPUT_IMAGE blocks in stage 1).

    python experiments/compare_advance_metrics_stage_0.py
    python experiments/compare_advance_metrics_stage_0.py --dry-run
    python experiments/compare_advance_metrics_stage_0.py --overwrite
"""

import argparse
import shutil
from pathlib import Path

SRC_DIR = "experiments/outputs/z_validation_data/maps"
OUTPUT_DIR = "experiments/outputs/images_results_metrics"
SOURCE_NAME = "source"

# {example folder: map file in SRC_DIR}. Every entry listed here is staged.
MAPS = {
    "quadratic": "quadratic_density_gradient.png",
}
# MAPS = {
#     "quadratic": "quadratic_density_gradient.png",
#     "stress": "stress_test_density.png",
# }


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stage the conditioning density map(s) for compare_advance_metrics_stage_1.")
    ap.add_argument("--src", default=SRC_DIR, help="Shared maps folder.")
    ap.add_argument("--output", default=OUTPUT_DIR, help="images_results_metrics base folder.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-copy maps that are already staged (default: leave them alone).")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing.")
    return ap.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src)
    out_base = Path(args.output)

    if not src_dir.is_dir():
        print(f"Error: maps folder not found: {src_dir}")
        return 1

    plan, missing = [], []
    for example, name in MAPS.items():
        src = src_dir / name
        dst = out_base / example / SOURCE_NAME / name
        (plan if src.exists() else missing).append((src, dst))

    for src, dst in missing:
        print(f"Error: map not found: {src}")
    if missing:
        return 1

    copied = skipped = 0
    for src, dst in plan:
        # An already-staged map is left in place: stage 1 and the samplers both read this
        # exact file, so silently replacing it mid-experiment would decouple the staged
        # point sets from the image they were generated for.
        if dst.exists() and not args.overwrite:
            print(f"skip (exists): {dst}")
            skipped += 1
            continue
        print(f"{src}  ->  {dst}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    print("")
    if args.dry_run:
        print(f"DRY RUN: {copied} map(s) would be copied, {skipped} left in place.")
    else:
        print(f"Copied {copied} map(s); {skipped} already staged (use --overwrite to replace).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
