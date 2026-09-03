"""Stage 4: plot aggregated geometry metrics across model variants

This script scans a base output folder for model subfolders (e.g.
`outputs/ablation_advance_metrics_e500_b50_1024/vanilla`, `.../gecco`, ...). For each
model it expects `metrics_avg.json` under the model folder with the structure:

    { "M1_v1_cvt_energy": { "500": 0.34, "1000": 0.28, ... },
      "M2_v1_capacity_delta_c": { ... },
      ... }

(written by ablation_advance_metrics_stage_3.py). The M6 training losses are a separate
pipeline now -- see ablation_loss_results_stage_0/1/2.py -- and are not read here.

The script builds one plot per metric in METRIC_ORDER (a curve per model, epoch -> mean
metric), saved as `<out_base>/resources/plots/m{i}.png` (+ `.pdf`).
"""

# TODO:
# - put legends on the y axis

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


# Default folders
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e500_b50_1024"

METRICS_FILE = "metrics_avg.json"

# Default figure settings (pixels and DPI) — change these at top of script
FIG_WIDTH = 1000
FIG_HEIGHT = 600
FIG_DPI = 100


# M1 and M2 each have a Voronoi-cell version and a power-cell version; M2_v3 (the
# per-point kNN-local capacity check, a weak discriminator -- see its docstring in
# stippling_metrics_advance.py) is still computed and stored in metrics_avg.json by
# stage 2/3, just not plotted here by default. M3-M5 have one version each.
METRIC_ORDER = [
    "M1_v1_cvt_energy",
    "M1_v2_power_cvt_energy",
    "M2_v1_capacity_delta_c",
    "M2_v2_power_displacement",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
]

# Output filename stem per metric (plots_dir/<stem>.png/.pdf). M1/M2 spell out their
# version (v1=Voronoi, v2=Power) since both are plotted; M3-M5 have one version each, and
# M6 is reserved for the separate loss pipeline (ablation_loss_results_stage_2.py).
PLOT_STEM = {
    "M1_v1_cvt_energy": "m1_v1",
    "M1_v2_power_cvt_energy": "m1_v2",
    "M2_v1_capacity_delta_c": "m2_v1",
    "M2_v2_power_displacement": "m2_v2",
    "M3_emd_distance": "m3",
    "M4_sinkhorn_ot_cost": "m4",
    "M5_spatial_measure_rho_mean": "m5",
}

# Human-friendly metric titles for plot headers.
#
# M1_v1/M1_v2 apply the SAME integrand -- sum of rho(x)*||x - s_owner||^2 -- to the two
# different partitions; only the cell assignment differs. They are named for each
# partition's own literature rather than sharing one name: on Voronoi cells this is the
# classical CVT (Centroidal Voronoi Tessellation) energy that Lloyd's method and WVS
# minimise, while on capacity-optimal power cells it is the \star_0-HOT_{2,2} (Higher-Order
# Optimal Transport) energy that BNOT minimises (de Goes et al., Sec. 2.3). "CVT energy"
# would be a misnomer on the power side, since a power/Laguerre diagram is not a Voronoi
# tessellation. M2_v1/M2_v2 do NOT: M2_v1 is capacity deviation on
# (zero-weight, i.e. ordinary) Voronoi cells; M2_v2 is centroid displacement on SOLVED
# power cells, not capacity deviation -- capacity deviation collapses to ~0 for every
# method once weights are solved (see compute_m2_v2_power_displacement's docstring), so
# displacement is the metric that actually discriminates on the power side. Their titles
# are named for what each one actually measures rather than forced to match.
METRIC_LABELS = {
    "M1_v1_cvt_energy": {"title": "M1 (Voronoi): CVT Energy", "ylabel": "Energy", "exclude_result_dirs": []},
    "M1_v2_power_cvt_energy": {"title": "M1 (Power): HOT Energy", "ylabel": "Energy", "exclude_result_dirs": []},
    "M2_v1_capacity_delta_c": {"title": r"M2 (Voronoi): Capacity Deviation $\delta_c$", "ylabel": r"$\delta_c$", "exclude_result_dirs": []},
    "M2_v2_power_displacement": {"title": "M2 (Power): Centroid Displacement", "ylabel": "displacement", "exclude_result_dirs": []},
    "M3_emd_distance": {"title": "M3: EMD Distance", "ylabel": "Distance", "exclude_result_dirs": []},
    "M4_sinkhorn_ot_cost": {"title": "M4: Sinkhorn OT Cost", "ylabel": "Cost", "exclude_result_dirs": []},
    "M5_spatial_measure_rho_mean": {"title": "M5: Spatial Measure ρ Mean", "ylabel": "ρ Mean", "exclude_result_dirs": []},
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot aggregated geometry metrics across model variants")
    p.add_argument("--output", default=OUTPUT_DIR, help="Base output folder containing model result subfolders")
    p.add_argument("--result-dirs", default=",".join(RESULT_DIR_LIST), help="Comma-separated list of model subfolders to include (default: RESULT_DIR_LIST in script)")
    p.add_argument("--metrics-file", default=METRICS_FILE, help="Name of aggregated metrics file in each model folder")
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
    # Data dump keyed by DIRECTORY name. DIR_TO_LABEL_MAP is a display-only map
    # (plot legends), so it is intentionally NOT applied here. sort_keys is NOT used:
    # it would alphabetize the nested epoch-string keys too ("1000" before "500"),
    # scrambling the on-disk order even though the values (already int epoch -> float)
    # are correct; variant-name order and each metric's epoch order are left as inserted.
    combined_path = out_base / METRICS_FILE
    combined_path.write_text(json.dumps(model_metrics, indent=2))
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

    # Consistent color PER MODEL, keyed by directory name and ordered by RESULT_DIR_LIST,
    # so e.g. vanilla keeps the same color across every plot. Labels never affect color.
    colors = plt.cm.tab10.colors
    color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(RESULT_DIR_LIST)}
    next_idx = len(RESULT_DIR_LIST)
    for name in model_metrics:
        if name not in color_map:
            color_map[name] = colors[next_idx % len(colors)]
            next_idx += 1

    plots_dir = out_base / "resources" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create one separate plot per metric and save as m1.png, m2.png, ...
    fig_size_inches = (fig_width_px / float(dpi), fig_height_px / float(dpi))
    for i, metric in enumerate(METRIC_ORDER):
        fig, ax = plt.subplots(figsize=fig_size_inches)

        metric_cfg = METRIC_LABELS.get(metric, {})
        metric_include = metric_cfg.get("include_result_dirs", []) if isinstance(metric_cfg, dict) else []
        metric_exclude = metric_cfg.get("exclude_result_dirs", []) if isinstance(metric_cfg, dict) else []

        for name in model_names:                       # name = the directory (data lookup key)
            if name not in model_metrics:
                continue
            if metric_include and name not in metric_include:
                continue
            if metric_exclude and name in metric_exclude:
                continue
            series = model_metrics[name].get(metric, {})
            if not series:
                continue
            # key=int: these are epoch-string keys ("500","1000",...,"5000"). A plain
            # sorted() sorts them LEXICOGRAPHICALLY ("500" lands after "4500"), which
            # silently draws a spurious jump backward near the end of every curve.
            epochs = np.array(sorted(series.keys(), key=int), dtype=int)
            values = np.array([series[e] for e in epochs], dtype=float)
            label = DIR_TO_LABEL_MAP.get(name, name)   # DIR_TO_LABEL_MAP: legend text ONLY
            ax.plot(epochs, values, label=label, color=color_map.get(name))

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

        stem = PLOT_STEM.get(metric, f"m{i+1}")
        outp = plots_dir / f"{stem}.png"
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
