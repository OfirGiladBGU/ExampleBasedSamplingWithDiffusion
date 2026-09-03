"""quantitative_advance_metrics_stage_0.py -- Part 0: stage the validation data.

Copies the WHOLE content of the shared validation folder into the quantitative-metrics
output folder:

    experiments/outputs/z_validation_data/Icons-50_1024/
      -> experiments/outputs/quantitative_advance_metrics/

That shared folder is built ONCE by experiments/z_copy_validatation_data.py, which does
all of the validation-split selection and the searching/flattening. It holds:

    source/                     the validation condition images
    target_<METHOD>_<points>/   per-method stipple results (png + npy)
    validation_manifest.json    the selected filenames, in order

so this stage is a plain recursive copy -- no manifest reading, no tree walking. Files
already in the destination are overwritten; anything else already there (the model
result folders and the *_json metric folders) is left untouched.

    python experiments/quantitative_advance_metrics_stage_0.py
    python experiments/quantitative_advance_metrics_stage_0.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

SRC_DIR = "experiments/outputs/z_validation_data/Icons-50_1024"
DST_DIR = "experiments/outputs/quantitative_advance_metrics"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Copy the shared validation folder into the quantitative metrics folder.")
    ap.add_argument("--src", default=SRC_DIR,
                    help="Shared validation folder (built by z_copy_validatation_data.py).")
    ap.add_argument("--dst", default=DST_DIR, help="Destination metrics folder.")
    ap.add_argument("--dry-run", action="store_true", help="List what would be copied; copy nothing.")
    return ap.parse_args()


def main():
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)

    if not src.is_dir():
        print(f"Error: source folder not found: {src}")
        return 1

    items = sorted(src.iterdir())
    if not items:
        print(f"Error: source folder is empty: {src}")
        return 1

    print(f"src: {src}")
    print(f"dst: {dst}\n")
    for item in items:
        if item.is_dir():
            print(f"  {item.name}/  ({sum(1 for _ in item.iterdir())} files)")
        else:
            print(f"  {item.name}")

    if args.dry_run:
        print(f"\nDRY RUN: would copy {len(items)} item(s) into {dst}")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)

    print(f"\nCopied {len(items)} item(s) -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
