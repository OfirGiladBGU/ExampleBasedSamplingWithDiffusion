"""ablation_loss_results_stage_1.py -- Part 1: aggregate the staged loss logs.

Reads OUTPUT_DIR/<variant>/losses_log.json (staged by stage 0) for every variant and
writes ONE combined file at the ablation root:

    OUTPUT_DIR/losses_avg.json

shaped exactly like the combined metrics_avg.json the ablation_advance_metrics_stage_4.py
pipeline writes -- variant -> metric -> { epoch_label: value } -- so the two ablations
overlay on the same epoch axis and any existing tooling that reads that shape works here
unchanged:

    { "vanilla": {"M6_train": {"500": ..., "1000": ...}, "M6_valid": {...}},
      "agi":     {...},
      ... }

There is no per-component breakdown here (unlike the old M6_minsnr_loss / M6_v2 split):
this is the raw total training loss straight from train_control.py, so it is reported as
one metric, M6_train (plus M6_valid for reference -- not plotted by stage 2).

`sdedit` has no training run of its own (see stage 0) -- it still appears as a key here,
with M6_train/M6_valid set to `null`, so downstream readers see it was considered and
found not applicable rather than silently missing.

--- The epoch off-by-one ---
losses_log.json is keyed by the 0-indexed loop variable `epoch` (0..args.epochs-1) at the
moment its loss was computed. A checkpoint named dynamic_ep{N}.ckpt is saved right after
that SAME iteration, but under `WEIGTHS_FILENAME_FORMAT.format(epoch=epoch+1)` -- i.e.
checkpoint label N corresponds to losses_log key (N-1), not N. metrics_avg.json's epoch
labels come from those checkpoint names, so this stage subtracts 1 to line up with them:
label 500 in the output <- losses_log["train"]["499"].
"""

import argparse
import json
from pathlib import Path

OUTPUT_DIR = "experiments/outputs/ablation_loss_results_e500_b50_1024"

RESULT_DIR_LIST = [
    "vanilla",
    "unfrozen",
    "gecco",
    "agi",
    "full",
    "sdedit",
]

LOSSES_LOG_NAME = "losses_log.json"
LOSSES_AVG_NAME = "losses_avg.json"

# Kept in sync with ablation_loss_results_stage_0.py: sdedit is inference-only.
NO_TRAINING_RUN = {"sdedit"}

# The checkpoint/epoch grid the rest of the ablation pipeline reports on.
EVERY_EPOCH = 500

# See the epoch off-by-one note above: checkpoint label N <- losses_log key (N-1).
EPOCH_LABEL_OFFSET = 1


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate per-variant losses_log.json into one losses_avg.json")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Ablation loss-results output folder (stage 0's --output)")
    ap.add_argument("--result-dirs", default=",".join(RESULT_DIR_LIST), help="Comma-separated variant names")
    ap.add_argument("--every-epoch", type=int, default=EVERY_EPOCH,
                    help="Epoch grid to report on; must match the metrics ablation's --every-epoch "
                         "for the two to overlay on the same x-axis")
    ap.add_argument("--losses-avg-file", default=LOSSES_AVG_NAME)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def epoch_labels(available_keys, every):
    """Checkpoint-style labels (every, 2*every, ...) that are actually coverable, i.e.
    label - EPOCH_LABEL_OFFSET exists as a key in the source log."""
    if not available_keys:
        return []
    max_key = max(int(k) for k in available_keys)
    labels = []
    n = every
    while n - EPOCH_LABEL_OFFSET <= max_key:
        if str(n - EPOCH_LABEL_OFFSET) in available_keys:
            labels.append(n)
        n += every
    return labels


def aggregate_variant(variant_dir: Path, every: int):
    log_path = variant_dir / LOSSES_LOG_NAME
    if not log_path.exists():
        return None
    data = json.loads(log_path.read_text())
    train_src = data.get("train", {})
    valid_src = data.get("valid", {})

    labels = epoch_labels(train_src, every)
    m6_train = {str(lbl): float(train_src[str(lbl - EPOCH_LABEL_OFFSET)]) for lbl in labels}
    m6_valid = {str(lbl): float(valid_src[str(lbl - EPOCH_LABEL_OFFSET)])
                for lbl in labels if str(lbl - EPOCH_LABEL_OFFSET) in valid_src}

    return {"M6_train": m6_train, "M6_valid": m6_valid}


def main():
    args = parse_args()
    out_base = Path(args.output)
    variants = [v.strip() for v in args.result_dirs.split(",") if v.strip()]

    combined = {}
    for variant in variants:
        if variant in NO_TRAINING_RUN:
            print(f"  {variant:10s}: not applicable (inference-only) -- recorded as null")
            combined[variant] = {"M6_train": None, "M6_valid": None}
            continue

        result = aggregate_variant(out_base / variant, args.every_epoch)
        if result is None:
            print(f"  [MISSING] {variant:10s}: no {LOSSES_LOG_NAME} under {out_base / variant} "
                  f"-- run stage 0 first")
            continue
        n_train = len(result["M6_train"])
        n_valid = len(result["M6_valid"])
        print(f"  {variant:10s}: M6_train={n_train} points, M6_valid={n_valid} points "
              f"(epoch grid: every {args.every_epoch})")
        combined[variant] = result

    if not combined:
        print("\nNo variants aggregated -- nothing to write.")
        return 2

    out_path = out_base / args.losses_avg_file
    if args.dry_run:
        print(f"\nDRY RUN: would write {out_path}")
        return 0

    # sort_keys=True would alphabetize the nested epoch-string keys too ('1000' before
    # '500'), scrambling the on-disk order even though epoch_labels() already inserts
    # them numerically ascending -- so this preserves insertion order instead.
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"\nWrote {out_path} ({len(combined)} variant(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
