"""ablation_loss_results_stage_2.py -- Part 2: plot the training and validation loss curves.

Reads OUTPUT_DIR/<variant>/losses_log.json directly (staged by stage 0) -- NOT the
checkpoint-grid losses_avg.json stage 1 writes -- so the plots use every recorded epoch
(all 5000, dense) rather than the 10-point checkpoint sample. Plots both loss splits, one
curve per variant per plot, into:

    OUTPUT_DIR/resources/plots/m6_train.png (+ .pdf)   "M6 (train): MinSNR + KDE Loss"
    OUTPUT_DIR/resources/plots/m6_test.png  (+ .pdf)   "M6 (test): MinSNR + KDE Loss"

"test" here is losses_log.json's "valid" split -- this project reports its held-out split
as "test", not "valid"; the file itself keeps the "valid" key from train_control.py, only
the plot title/filename use "test". Colors are assigned the same way
ablation_advance_metrics_stage_4.py does (indexed into tab10 by position in
RESULT_DIR_LIST), so a variant keeps the same color across both ablations' plots.

`sdedit` has no training run of its own (see stage 0) -- no losses_log.json was staged
for it, so it is naturally skipped. EXCLUDE_FROM_PLOT is kept as an explicit, documented
guard rather than relying only on the missing-file check, mirroring stage_4's own
METRIC_LABELS exclude_result_dirs=["sdedit"] for this same metric.

Epoch axis: losses_log.json keys are the 0-indexed loop variable at the moment the loss
was computed (see train_control.py); +1 is added for the x-axis so epoch 5000 here means
"after completing the 5000th epoch", matching the checkpoint-name convention
(dynamic_ep5000.ckpt) used everywhere else in the ablation pipeline.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "experiments/outputs/ablation_loss_results_e500_b50_1024"

RESULT_DIR_LIST = [
    "vanilla",
    "unfrozen",
    "gecco",
    "agi",
    "full",
    "sdedit",
]
DIR_TO_LABEL_MAP = {
    "vanilla": "Vanilla",
    "unfrozen": "Unfrozen",
    "gecco": "GECCO",
    "agi": "Gated",
    "full": "Full",
    "sdedit": "SDEdit",
}
# sdedit has no training run of its own -- there is nothing to plot for it.
EXCLUDE_FROM_PLOT = ["sdedit"]

LOSSES_LOG_NAME = "losses_log.json"
YLABEL = "Loss"

# One entry per plot: (losses_log.json split key, output file stem, plot title).
# "valid" in the file -> "test" in the title/filename, matching this project's naming.
PLOTS = [
    ("train", "m6_train", "M6 (train): MinSNR + KDE Loss"),
    ("valid", "m6_test", "M6 (test): MinSNR + KDE Loss"),
]

FIG_WIDTH = 1000
FIG_HEIGHT = 600
FIG_DPI = 100


def parse_args():
    ap = argparse.ArgumentParser(description="Plot the full-resolution M6 train/test loss curves across variants")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Ablation loss-results output folder")
    ap.add_argument("--result-dirs", default=",".join(RESULT_DIR_LIST))
    ap.add_argument("--exclude", default=",".join(EXCLUDE_FROM_PLOT),
                    help="Comma-separated variants to compute-but-hide (sdedit has no training run)")
    ap.add_argument("--fig-width", type=int, default=FIG_WIDTH)
    ap.add_argument("--fig-height", type=int, default=FIG_HEIGHT)
    ap.add_argument("--dpi", type=int, default=FIG_DPI)
    return ap.parse_args()


def load_series(variant_dir: Path, split: str):
    """(epochs, values) for one variant/split, straight from its staged losses_log.json,
    numerically sorted -- NOT lexicographically, which silently scrambles the line
    (e.g. "500" sorts after "4500" as a string) and draws a spurious jump in the plot."""
    log_path = variant_dir / LOSSES_LOG_NAME
    if not log_path.exists():
        return None
    data = json.loads(log_path.read_text())
    series = data.get(split, {})
    if not series:
        return None
    epochs = sorted((int(k) for k in series), key=int)
    values = [float(series[str(e)]) for e in epochs]
    # +1: losses_log.json is 0-indexed at the loop variable; the rest of the pipeline
    # labels epochs by the 1-indexed checkpoint name (dynamic_ep{epoch+1}.ckpt).
    return np.array(epochs) + 1, np.array(values)


def make_plot(out_base, variants, excluded, color_map, split, stem, title, args):
    plots_dir = out_base / "resources" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_size_inches = (args.fig_width / float(args.dpi), args.fig_height / float(args.dpi))
    fig, ax = plt.subplots(figsize=fig_size_inches)

    plotted = 0
    for name in variants:
        if name in excluded:
            continue
        result = load_series(out_base / name, split)
        if result is None:
            print(f"  [skip] {name}: no '{split}' series in {out_base / name / LOSSES_LOG_NAME}")
            continue
        epochs, values = result
        label = DIR_TO_LABEL_MAP.get(name, name)
        ax.plot(epochs, values, label=label, color=color_map.get(name))
        print(f"  {name:10s}: {len(epochs)} points [{epochs[0]}..{epochs[-1]}]")
        plotted += 1

    if plotted == 0:
        print(f"  No variant had a '{split}' series to plot -- skipping {stem}")
        plt.close(fig)
        return 0

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel(YLABEL, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(
        fontsize=13,
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
        framealpha=0.9,
        handlelength=2.0,
        borderpad=0.6,
    )
    fig.tight_layout()

    outp = plots_dir / f"{stem}.png"
    fig.savefig(outp, dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
    try:
        fig.savefig(str(outp.with_suffix('.pdf')), dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
    except Exception:
        pass
    plt.close(fig)

    print(f"  Wrote {outp} ({outp.with_suffix('.pdf')}) with {plotted} variant curve(s)")
    return plotted


def main():
    args = parse_args()
    out_base = Path(args.output)
    variants = [v.strip() for v in args.result_dirs.split(",") if v.strip()]
    excluded = {v.strip() for v in args.exclude.split(",") if v.strip()}

    colors = plt.cm.tab10.colors
    color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(RESULT_DIR_LIST)}

    total_plotted = 0
    for split, stem, title in PLOTS:
        print(f"=== {title} ===")
        total_plotted += make_plot(out_base, variants, excluded, color_map, split, stem, title, args)

    if total_plotted == 0:
        print("\nNo plots were written")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
