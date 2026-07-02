"""Stage 4: plot aggregated metrics across model variants

This script scans a base output folder for model subfolders (e.g.
`outputs/ablation_advance_metrics/vanilla`, `.../gecco`, ...). For each model
it expects `metrics_avg.json` under the model folder with the structure:

    { "M1_cvt_energy": { "25": 0.34, "50": 0.28, ... },
      "M2_voronoi_mass_cv": { ... },
      ... }

The script builds a 2x3 subplot (one subplot per metric in METRIC_ORDER)
containing a curve per model (epoch -> mean metric). Output is saved to
`<out_base>/metrics_compare.png` and `<out_base>/metrics_compare.pdf`.
"""

# TODO:
# - make plots wider 4:3 ratio (keep them inside a "plots" subfolder and with nameing "m1.png", "m2.png", etc.)
# - rename to header to not be with underscore (e.g. "M1: CVT Energy") for better plot titles
# - put legends on the y axis
# - split to 6 seperate plots instead of 2x3 grid (easier to read)
# - keep color consistency across all plots (e.g. vanilla always blue, gecco always orange, etc.)
# 

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULT_DIR_LIST = [
    "vanilla",
    "gecco",
    "agi",
    "full",
    "sdedit",
    "sdedit_resample",
]


# Default folders
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e400_b50_1024"

# Default metrics file name
METRICS_FILE = "metrics_avg.json"

# Default figure settings (pixels and DPI) — change these at top of script
FIG_WIDTH = 1000
FIG_HEIGHT = 600
FIG_DPI = 100


METRIC_ORDER = [
    "M1_cvt_energy",
    "M2_voronoi_mass_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
    "M6_minsnr_loss",
]

# Human-friendly metric titles for plot headers
METRIC_LABELS = {
    # Each metric can map to either a string (used for both title and ylabel)
    # or a dict with explicit `title` and `ylabel` fields.
    "M1_cvt_energy": {"title": "M1: CVT Energy", "ylabel": "Energy", "exclude_result_dirs": []},
    "M2_voronoi_mass_cv": {"title": "M2: Capacity Constraint Fulfillment", "ylabel": "Voronoi Mass", "exclude_result_dirs": []},
    "M3_emd_distance": {"title": "M3: EMD Distance", "ylabel": "Distance", "exclude_result_dirs": []},
    "M4_sinkhorn_ot_cost": {"title": "M4: Sinkhorn OT Cost", "ylabel": "Cost", "exclude_result_dirs": []},
    "M5_spatial_measure_rho_mean": {"title": "M5: Spatial Measure ρ Mean", "ylabel": "ρ Mean", "exclude_result_dirs": []},
    "M6_minsnr_loss": {
        "title": "M6: MinSNR Loss",
        "ylabel": "Loss",
        "exclude_result_dirs": ["sdedit", "sdedit_resample"],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot aggregated metrics across model variants")
    p.add_argument("--output", default=OUTPUT_DIR, help="Base output folder containing model result subfolders")
    p.add_argument("--result-dirs", default=",".join(RESULT_DIR_LIST), help="Comma-separated list of model subfolders to include (default: RESULT_DIR_LIST in script)")
    p.add_argument("--metrics-file", default=METRICS_FILE, help="Name of aggregated metrics file in each model folder")
    # The script writes six separate plots into the `plots/` subfolder under the output directory.
    p.add_argument("--fig-width", type=int, default=FIG_WIDTH, help=f"Figure width in pixels (default: {FIG_WIDTH})")
    p.add_argument("--fig-height", type=int, default=FIG_HEIGHT, help=f"Figure height in pixels (default: {FIG_HEIGHT})")
    p.add_argument("--dpi", type=int, default=FIG_DPI, help=f"DPI to use when saving figures (default: {FIG_DPI})")
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


def write_combined_metrics(out_base: Path, model_metrics: dict):
    combined_path = out_base / METRICS_FILE
    combined = {
        model_name: metrics
        for model_name, metrics in model_metrics.items()
    }
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True))
    print(f"Wrote combined metrics to {combined_path}")


def make_plots(out_base: Path, model_names, metrics_file, fig_width_px: int, fig_height_px: int, dpi: int):
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

    write_combined_metrics(out_base, model_metrics)

    # Create consistent color mapping across runs (keep mapping stable)
    colors = plt.cm.tab10.colors
    color_map = {}
    for idx, name in enumerate(RESULT_DIR_LIST):
        color_map[name] = colors[idx % len(colors)]
    # assign colors for any additional models not in RESULT_DIR_LIST
    next_idx = len(RESULT_DIR_LIST)
    for m in model_metrics.keys():
        if m not in color_map:
            color_map[m] = colors[next_idx % len(colors)]
            next_idx += 1

    plots_dir = out_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create one separate plot per metric and save as m1.png, m2.png, ...
    fig_size_inches = (fig_width_px / float(dpi), fig_height_px / float(dpi))
    for i, metric in enumerate(METRIC_ORDER):
        fig, ax = plt.subplots(figsize=fig_size_inches)

        metric_cfg = METRIC_LABELS.get(metric, {})
        metric_include = metric_cfg.get("include_result_dirs", []) if isinstance(metric_cfg, dict) else []
        metric_exclude = metric_cfg.get("exclude_result_dirs", []) if isinstance(metric_cfg, dict) else []

        for mname in model_names:
            if mname not in model_metrics:
                continue
            if metric_include and mname not in metric_include:
                continue
            if metric_exclude and mname in metric_exclude:
                continue
            metrics = model_metrics[mname]
            series = metrics.get(metric, {})
            if not series:
                continue
            epochs = np.array(sorted(series.keys()), dtype=int)
            values = np.array([series[e] for e in epochs], dtype=float)
            ax.plot(epochs, values, label=mname, color=color_map.get(mname))

        lab = METRIC_LABELS.get(metric, None)
        if isinstance(lab, dict):
            title_text = lab.get('title', metric.replace('_', ' '))
            ylabel_text = lab.get('ylabel', title_text)
        elif isinstance(lab, str):
            title_text = lab
            ylabel_text = lab
        else:
            title_text = metric.replace('_', ' ')
            ylabel_text = title_text

        ax.set_title(title_text, fontsize=16)
        ax.set_xlabel("epoch", fontsize=14)
        ax.set_ylabel(ylabel_text, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.2)

        # Place larger legend inside the plot (top-left) for readability.
        ax.legend(
            fontsize=13,
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            bbox_transform=ax.transAxes,
            framealpha=0.9,
            handlelength=2.0,
            borderpad=0.6,
        )

        # Tighten layout and save with bbox_inches to ensure the entire plot (including legend)
        fig.tight_layout()

        outp = plots_dir / f"m{i+1}.png"
        fig.savefig(outp, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        try:
            fig.savefig(str(outp.with_suffix('.pdf')), dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        except Exception:
            pass
        plt.close(fig)

    print(f"Wrote individual metric plots to {plots_dir}")


def main():
    args = parse_args()
    out_base = Path(args.output)
    model_names = [s.strip() for s in args.result_dirs.split(",") if s.strip()]

    make_plots(
        out_base,
        model_names,
        args.metrics_file,
        args.fig_width,
        args.fig_height,
        args.dpi,
    )


if __name__ == '__main__':
    main()
