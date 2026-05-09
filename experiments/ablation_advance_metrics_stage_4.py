"""Stage 4: plot aggregated metrics across model variants

This script scans a base output folder for model subfolders (e.g.
`outputs/ablation_advance_metrics/vanilla`, `.../gecco`, ...). For each model
it expects `metrics_avg.json` under the model folder with the structure:

    { "M1_cvt_energy": { "25": 0.34, "50": 0.28, ... },
      "M2_voronoi_mass_cv": { ... },
      ... }

The script builds a 2x3 subplot (one subplot per metric in METRIC_ORDER)
containing a curve per model (epoch -> mean metric). Output is saved to
`<out_base>/plots/metrics_compare.png` and `<out_base>/plots/metrics_compare.pdf`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRIC_ORDER = [
    "M1_cvt_energy",
    "M2_voronoi_mass_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
    "M6_minsnr_loss",
]

RESULT_DIR_LIST = [
    "vanilla",
    "gecco",
    "agi",
    "full",
    "sdedit",
    "sdedit_resample",
]


def parse_args():
    p = argparse.ArgumentParser(description="Plot aggregated metrics across model variants")
    p.add_argument("--output", default="/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/ablation_advance_metrics", help="Base output folder containing model result subfolders")
    p.add_argument("--result-dirs", default=",".join(RESULT_DIR_LIST), help="Comma-separated list of model subfolders to include (default: RESULT_DIR_LIST in script)")
    p.add_argument("--metrics-file", default="metrics_avg.json", help="Name of aggregated metrics file in each model folder")
    p.add_argument("--out-plot", default="plots/metrics_compare.png", help="Relative path (from output) to write the combined plot")
    return p.parse_args()


def load_metrics_for_model(model_dir: Path, metrics_file: str):
    path = model_dir / metrics_file
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # convert epoch keys to ints
    converted = {}
    for metric, series in data.items():
        conv = {int(k): float(v) for k, v in series.items()}
        converted[metric] = conv
    return converted


def make_plots(out_base: Path, model_names, metrics_file, out_plot_rel):
    model_metrics = {}
    for m in model_names:
        md = out_base / m
        metrics = load_metrics_for_model(md, metrics_file)
        if metrics is None:
            print(f"Warning: skipping {m} (no {metrics_file})")
            continue
        model_metrics[m] = metrics

    if len(model_metrics) == 0:
        raise RuntimeError("No model metrics found to plot")

    # Create figure 2x3
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.reshape(-1)
    colors = plt.cm.tab10.colors

    for i, metric in enumerate(METRIC_ORDER):
        ax = axs[i]
        for j, (mname, metrics) in enumerate(model_metrics.items()):
            series = metrics.get(metric, {})
            if not series:
                continue
            epochs = np.array(sorted(series.keys()), dtype=int)
            values = np.array([series[e] for e in epochs], dtype=float)
            ax.plot(epochs, values, label=mname, color=colors[j % len(colors)])

        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize="small")

    fig.tight_layout()
    outp = out_base / out_plot_rel
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=200)
    # also save PDF
    try:
        fig.savefig(str(outp.with_suffix('.pdf')))
    except Exception:
        pass
    plt.close(fig)
    print(f"Wrote plots to {outp}")


def main():
    args = parse_args()
    out_base = Path(args.output)
    model_names = [s.strip() for s in args.result_dirs.split(",") if s.strip()]

    make_plots(out_base, model_names, args.metrics_file, args.out_plot)


if __name__ == '__main__':
    main()
