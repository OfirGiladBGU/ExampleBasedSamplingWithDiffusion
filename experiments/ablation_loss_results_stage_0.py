"""ablation_loss_results_stage_0.py -- Part 0: stage the raw training loss logs.

Copies losses_log.json (the true per-epoch train/valid loss recorded live during
training, see control_v4/train_control.py) out of each variant's train_outputs_*
folder into a per-variant subfolder here:

    control_v4/train_outputs_Icons-50_1024_GBN_<variant>/losses_log.json
        -> OUTPUT_DIR/<variant>/losses_log.json

mirroring the layout of experiments/outputs/ablation_advance_metrics_e500_b50_1024/
(one subfolder per RESULT_DIR_LIST entry), so stage 1 can read them the same way
stage 3 of that pipeline reads per-epoch JSONs.

`sdedit` has no training run of its own -- it reuses the `full` checkpoints and only
changes the inference truncation ratio at sampling time, so its training loss is not a
meaningful separate quantity. It is skipped here entirely (no folder is created for it);
stage 1 records it in losses_avg.json as `null`, explicitly marking it not applicable
rather than silently omitting it.

    python experiments/ablation_loss_results_stage_0.py
    python experiments/ablation_loss_results_stage_0.py --dry-run
"""

import argparse
import shutil
from pathlib import Path

CONTROL_DIR = "control_v4"
OUTPUT_DIR = "experiments/outputs/ablation_loss_results_e500_b50_1024"

RESULT_DIR_LIST = [
    "vanilla",
    "unfrozen",
    "gecco",
    "agi",
    "full",
    "sdedit",
]

# sdedit is inference-only (shares full's training run, only INFER_TRUNCATION_RATIO
# differs at sampling time) -- its training loss is not computed. Skipped in stage 0/1/2.
NO_TRAINING_RUN = {"sdedit"}

LOSSES_LOG_NAME = "losses_log.json"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Copy losses_log.json from each train_outputs_* folder into OUTPUT_DIR/<variant>/.")
    ap.add_argument("--control-dir", default=CONTROL_DIR, help="Folder holding train_outputs_Icons-50_1024_GBN_*.")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Ablation loss-results output folder.")
    ap.add_argument("--result-dirs", default=",".join(RESULT_DIR_LIST),
                    help="Comma-separated variant names to stage.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing.")
    return ap.parse_args()


def main():
    args = parse_args()
    control_dir = Path(args.control_dir)
    out_base = Path(args.output)
    variants = [v.strip() for v in args.result_dirs.split(",") if v.strip()]

    print(f"control dir: {control_dir}")
    print(f"output dir : {out_base}\n")

    ok, missing, skipped = [], [], []
    for variant in variants:
        if variant in NO_TRAINING_RUN:
            print(f"  [SKIP]    {variant:10s}: no training run of its own (inference-only) "
                  f"-- losses_avg.json will report it as null")
            skipped.append(variant)
            continue

        src = control_dir / f"train_outputs_Icons-50_1024_GBN_{variant}" / LOSSES_LOG_NAME
        dst = out_base / variant / LOSSES_LOG_NAME

        if not src.exists():
            print(f"  [MISSING] {variant:10s} <- {src}")
            missing.append(variant)
            continue

        print(f"  {variant:10s} <- {src}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        ok.append(variant)

    if args.dry_run:
        print("\nDRY RUN: nothing copied.")
        return 0

    print(f"\nCopied {len(ok)}/{len(variants)} variant(s) -> {out_base}"
          f" ({len(skipped)} skipped as inference-only)")
    if missing:
        print(f"Missing losses_log.json for: {', '.join(missing)} "
              f"-- run at least one epoch of training with the updated train_control.py, "
              f"or (for older runs) regenerate it from the crash logs first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
