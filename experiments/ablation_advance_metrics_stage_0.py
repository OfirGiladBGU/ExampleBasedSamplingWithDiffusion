"""ablation_advance_metrics_stage_0.py -- Part 0: stage the shared ablation resources.

Copies what the ablation needs out of the shared validation folder (built once by
experiments/z_copy_validatation_data.py) into a single RESOURCES folder:

    z_validation_data/<dataset>/source/                   -> OUTPUT_DIR/resources/source/
    z_validation_data/<dataset>/validation_manifest.json  -> OUTPUT_DIR/resources/validation_manifest.json

No GT target folder is staged. The M6 losses were removed from this pipeline -- they
measure the model rather than the exported point set, and now live in
ablation_loss_results_stage_0/1/2.py, driven by control_v4/train_outputs_*/losses_log.json.
They were the only consumer of both the per-image MinSNR timestep draw and the GT target
copy, so neither happens here any more: stages 1-4 need only source/ and the manifest.
(The originals still live in z_validation_data/<dataset>/ if a target is ever needed again.)

Layout: everything shared lives under OUTPUT_DIR/resources/ (source, the manifest, and the
plots stage 4 writes), so every OTHER folder directly under OUTPUT_DIR is an
ablation-variant folder.

    python experiments/ablation_advance_metrics_stage_0.py
    python experiments/ablation_advance_metrics_stage_0.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

SRC_DIR = "experiments/outputs/z_validation_data/Icons-50_1024"
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e500_b50_1024"
RESOURCES_NAME = "resources"
MANIFEST_NAME = "validation_manifest.json"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Copy source/ and the manifest into OUTPUT_DIR/resources/.")
    ap.add_argument("--src", default=SRC_DIR,
                    help="Shared validation folder (built by z_copy_validatation_data.py).")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Ablation output folder.")
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
