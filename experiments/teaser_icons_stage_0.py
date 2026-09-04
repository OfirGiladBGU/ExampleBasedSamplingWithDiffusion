"""teaser_icons_stage_0.py -- Part 0: stage the validation manifest.

Copies ONLY the manifest out of the shared validation folder (built once by
experiments/z_copy_validatation_data.py):

    z_validation_data/<dataset>/validation_manifest.json
        -> OUTPUT_DIR/validation_manifest.json

Unlike the other stage-0 scripts this does NOT copy source images. Stage 1 needs the GBN
OFFSETS as well as the images, and those are built from the full training set
(ensure_offsets_dir) and are far too large to stage here -- so stage 1 still reads the real
source/offsets dirs. The manifest's only job is to tell it WHICH files are the validation
set, in order, so it no longer has to re-derive the seed-42 split to find them.

The manifest is stored in SELECTION order, so manifest[i] is exactly the image the old
seed-42 split_index i resolved to -- the VALID_SAMPLES indices keep their meaning.

    python experiments/teaser_icons_stage_0.py
    python experiments/teaser_icons_stage_0.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

SRC_DIR = "experiments/outputs/z_validation_data/Icons-50_1024"
OUTPUT_DIR = "experiments/outputs/teaser_icons_results"
MANIFEST_NAME = "validation_manifest.json"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Copy the validation manifest into OUTPUT_DIR/.")
    ap.add_argument("--src", default=SRC_DIR,
                    help="Shared validation folder (built by z_copy_validatation_data.py).")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Teaser output folder.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing.")
    return ap.parse_args()


def main():
    args = parse_args()
    src_manifest = Path(args.src) / MANIFEST_NAME
    out = Path(args.output)

    if not src_manifest.exists():
        print(f"Error: manifest not found: {src_manifest}")
        return 1

    print(f"src: {src_manifest}")
    print(f"dst: {out / MANIFEST_NAME}")

    if args.dry_run:
        print("\nDRY RUN: nothing copied.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_manifest, out / MANIFEST_NAME)

    import json
    n = len(json.loads((out / MANIFEST_NAME).read_text()))
    print(f"\nCopied {MANIFEST_NAME} ({n} entries) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
