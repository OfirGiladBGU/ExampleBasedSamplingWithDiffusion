"""ablation_runtime_results_stage_0.py -- Part 0: stage the shared runtime resources.

Copies what the runtime ablation needs out of the shared validation folder (built once by
experiments/z_copy_validatation_data.py) into a single RESOURCES folder:

    z_validation_data/<dataset>/source/                   -> OUTPUT_DIR/resources/source/
    z_validation_data/<dataset>/validation_manifest.json  -> OUTPUT_DIR/resources/validation_manifest.json

Stage 1 then times exactly the images the manifest lists, in manifest order, instead of
re-deriving the split from the full training set with a seed. The manifest is stored in
SELECTION order (not alphabetical), so manifest[:NUM_SAMPLES] reproduces the seed-42 split
the earlier runs used -- verified identical to the already-measured timestamps, so this
staging changes which images are timed not at all.

Only source/ is needed: runtime measurement never reads a GT target.

Layout: everything shared lives under OUTPUT_DIR/resources/, so every OTHER folder
directly under OUTPUT_DIR is a method folder (vanilla, gecco, ...).

    python experiments/ablation_runtime_results_stage_0.py
    python experiments/ablation_runtime_results_stage_0.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

SRC_DIR = "experiments/outputs/z_validation_data/Icons-50_1024"
OUTPUT_DIR = "experiments/outputs/ablation_runtime_results"
RESOURCES_NAME = "resources"
MANIFEST_NAME = "validation_manifest.json"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Copy source/ and the manifest into OUTPUT_DIR/resources/.")
    ap.add_argument("--src", default=SRC_DIR,
                    help="Shared validation folder (built by z_copy_validatation_data.py).")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Runtime ablation output folder.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing.")
    return ap.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    res = Path(args.output) / RESOURCES_NAME

    src_source = src / "source"
    src_manifest = src / MANIFEST_NAME
    for p, what in ((src_source, "source folder"), (src_manifest, "manifest")):
        if not p.exists():
            print(f"Error: {what} not found: {p}")
            return 1

    n_src = sum(1 for _ in src_source.iterdir())
    print(f"src: {src}")
    print(f"dst: {res}\n")
    print(f"  source/ ({n_src} files)          -> {res / 'source'}")
    print(f"  {MANIFEST_NAME}             -> {res / MANIFEST_NAME}")

    if args.dry_run:
        print("\nDRY RUN: nothing copied.")
        return 0

    res.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_source, res / "source", dirs_exist_ok=True)
    shutil.copy2(src_manifest, res / MANIFEST_NAME)

    print(f"\nCopied source/ ({n_src}) + {MANIFEST_NAME} -> {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
