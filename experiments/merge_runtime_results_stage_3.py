#!/usr/bin/env python3
"""Stage 3: plot the aggregated runtimes of merge_runtime_results_stage_2.py.

Reads ``runtimes_avg.json`` ::

    {"WVS": {"256": {"avg": ..., "std": ..., "count": N}, ...}, "BNOT": ..., "GBN": ..., "CN": ...}

and writes, into a ``plots/`` folder next to it, one line per method of the average total
runtime per image against the point budget (PNG at 300 dpi + vector PDF). Styled after
profiling_test_stage_2_plot_json.py: the same figure size, the same "grid" / "points" x-axis
modes and the same "32x32 (1024 pts)" tick labels.

Only the budgets listed in POINT_BUDGETS are drawn (edit the list, or pass --budgets). A
budget missing for some method is simply skipped for that method's line.

Run from the project root:
    python experiments/merge_runtime_results_stage_3.py
    python experiments/merge_runtime_results_stage_3.py --x_axis points
    python experiments/merge_runtime_results_stage_3.py --budgets 256 1024 4096 --y_scale linear
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Input / output ──────────────────────────────────────────────────────────
RUNTIMES_JSON = "experiments/outputs/icons_results_runtimes/runtimes_avg.json"
PLOTS_DIRNAME = "plots"                    # created next to RUNTIMES_JSON
PLOT_BASENAME = "runtime_scaling"          # -> plots/runtime_scaling_by_<x_axis>.png/.pdf

# ── Which point budgets to plot (grid**2) ───────────────────────────────────
POINT_BUDGETS = [
    256,     # 16 x 16
    576,     # 24 x 24
    1024,    # 32 x 32
    1600,    # 40 x 40
    2304,    # 48 x 48
    3136,    # 56 x 56
    4096,    # 64 x 64
    5184,    # 72 x 72
    6400,    # 80 x 80
    # 7744,    # 88 x 88
    # 9216,    # 96 x 96
    # 10816,   # 104 x 104
    # 12544,   # 112 x 112
]

# ── Methods (plotting order, legend names) ──────────────────────────────────
METHODS = ["WVS", "BNOT", "GBN", "CN"]
METHOD_LABELS = {"WVS": "WVS", "BNOT": "BNOT", "GBN": "GBN", "CN": "Ours"}
OURS_METHOD = "CN"                          # drawn thicker so it stands out

# ── Look (matches profiling_test_stage_2_plot_json.py) ──────────────────────
FIG_WIDTH = 11.5
FIG_HEIGHT = 4.5
DPI = 300
DEFAULT_X_AXIS_MODE = "grid"                # "grid" (categorical spacing) or "points" (proportional)
DEFAULT_Y_SCALE = "log"                     # "log" or "linear"; runtimes span ~3 orders of magnitude
SHOW_STD = False                            # error bars = std over the images of each budget
SHOW_TITLE = False                          # the file name already says what the plot is
LEGEND_LOC = "upper left"                   # inset inside the axes (matplotlib loc string)
X_LABEL = "Points Budget"
Y_LABEL = "Average Runtime (Seconds)"


def load_runtimes(path):
    with open(path, "r") as fh:
        return json.load(fh)


def budget_label(points):
    g = math.isqrt(points)
    return f"{g}x{g}\n({points} pts)" if g * g == points else f"{points} pts"


def plot_runtimes(runtimes, budgets, out_path, x_mode, y_scale, show_std):
    if x_mode not in ("grid", "points"):
        raise ValueError("x_mode must be 'grid' or 'points'")
    budgets = sorted(budgets)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=150)

    x_of = (lambda b: math.isqrt(b)) if x_mode == "grid" else (lambda b: b)
    for method in METHODS:
        if method not in runtimes:
            print(f"[WARN] method {method} not in {RUNTIMES_JSON}; skipped")
            continue
        stats = runtimes[method]
        have = [b for b in budgets if str(b) in stats]
        missing = [b for b in budgets if str(b) not in stats]
        if missing:
            print(f"[WARN] {method}: no runtime for budgets {missing}")
        if not have:
            continue
        x = [x_of(b) for b in have]
        y = [stats[str(b)]["avg"] for b in have]
        err = [stats[str(b)]["std"] for b in have] if show_std else None
        ours = method == OURS_METHOD
        ax.errorbar(x, y, yerr=err, marker="o", markersize=6 if ours else 5,
                    linewidth=3.0 if ours else 1.8, capsize=3 if show_std else 0,
                    label=METHOD_LABELS.get(method, method), zorder=3 if ours else 2)

    xt = [x_of(b) for b in budgets]
    ax.set_xticks(xt)
    ax.set_xticklabels([budget_label(b) for b in budgets], fontsize=9,
                       rotation=30 if x_mode == "points" else 0,
                       ha="right" if x_mode == "points" else "center")
    ax.set_xlabel(X_LABEL, fontsize=12, fontweight="bold")
    ax.set_ylabel(Y_LABEL, fontsize=12, fontweight="bold")
    ax.set_yscale(y_scale)
    if SHOW_TITLE:
        ax.set_title(f"Runtime Scaling by {'Grid Size' if x_mode == 'grid' else 'Point Count'}",
                     fontsize=15, fontweight="bold")

    ax.legend(loc=LEGEND_LOC, fontsize=10, ncol=1, framealpha=0.9)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved plot to {out_path}")
    print(f"Saved plot to {out_path.with_suffix('.pdf')}")


def parse_args():
    ap = argparse.ArgumentParser(description="Plot per-method runtimes from runtimes_avg.json.")
    ap.add_argument("--runtimes_json", default=RUNTIMES_JSON, help="Output of merge_runtime_results_stage_2.py.")
    ap.add_argument("--budgets", type=int, nargs="+", default=POINT_BUDGETS,
                    help="Point budgets to plot (replaces POINT_BUDGETS).")
    ap.add_argument("--x_axis", choices=["grid", "points"], default=DEFAULT_X_AXIS_MODE,
                    help="'grid' for categorical spacing or 'points' for proportional point-count spacing.")
    ap.add_argument("--y_scale", choices=["log", "linear"], default=DEFAULT_Y_SCALE)
    ap.add_argument("--show_std", action=argparse.BooleanOptionalAction, default=SHOW_STD,
                    help="Draw std error bars.")
    return ap.parse_args()


def main():
    args = parse_args()
    src = Path(args.runtimes_json)
    if not src.is_file():
        print(f"Error: {src} not found (run merge_runtime_results_stage_2.py first)")
        return 1
    out = src.parent / PLOTS_DIRNAME / f"{PLOT_BASENAME}_by_{args.x_axis}.png"
    plot_runtimes(load_runtimes(src), args.budgets, out, args.x_axis, args.y_scale, args.show_std)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
