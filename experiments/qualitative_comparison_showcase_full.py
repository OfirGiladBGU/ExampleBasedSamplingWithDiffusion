"""qualitative_comparison_showcase_full.py

The comparison showcase of qualitative_comparison_showcase.py with the quadratic capacity test of
compare_advance_metrics_stage_1.py added as its FIRST row.

Columns (and headers) are the showcase's COLUMNS, including its "||" black separators
(default Target || BNOT || WVS | GBN | Ours-WVS | Ours-GBN); --columns overrides them.
    Row 1      -> capacity test: the quadratic density map under Target, each method's points under
                  its own column, dashed quarter guides, and the per-quarter capacity (%) printed
                  below every cell (target mass for Target, point share for each method).
    Rows 2..N  -> the showcase rows, exactly as qualitative_comparison_showcase.py selects them
                  (DIR_MAP / SAMPLES_MAP / OUT_NAME there).

Nothing is duplicated: the dataset maps, row selection and cell rendering come from
qualitative_comparison_showcase.py, and the quadratic input, per-method point files, capacity
maths and guide styling from compare_advance_metrics_stage_1.py. Change either script's
configuration and this figure follows.

Writes: OUT_DIR/<out-name>.pdf (+ .png)
Run from the project root:  python experiments/qualitative_comparison_showcase_full.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualitative_comparison_showcase as showcase  # noqa: E402
import compare_advance_metrics_stage_1 as capacity  # noqa: E402

# ── Output ──────────────────────────────────────────────────────────────────
OUT_DIR = showcase.OUT_DIR
OUT_NAME = f"{showcase.OUT_NAME}_full"

# ── Capacity row ────────────────────────────────────────────────────────────
CAPACITY_INPUT = capacity.DEFAULT_INPUT_IMAGE
CAPACITY_COMPARE_LIST = capacity.DEFAULT_COMPARE_LIST   # [{column label: .npy/.png path}, ...]
CAPACITY_N_POINTS = 32 * 32        # every method fit to one budget, as in the capacity script
QUARTERS = 4
PCT_FONT_SIZE = 7.5                  # capacity percentages under the first-row cells
PCT_OFFSET = -0.04                 # y of the percentages, in axes fraction below the cell
CAPACITY_GAP = 0.20                # extra space under row 1 for the percentages (fraction of a row)
MARK_BEST = capacity.MARK_BEST     # red box on the capacity closest to the target, per quarter

# ── Showcase rows / shared look ─────────────────────────────────────────────
CELL = showcase.CELL
DOT_SIZE = showcase.DOT_SIZE
HEADER_FONT_SIZE = 13
ROW_GAP = 0.03                     # space between the other rows (fraction of a row)


def draw_quarter_guides(ax):
    for x in np.linspace(0.0, 1.0, QUARTERS + 1)[1:-1]:
        ax.plot([x, x], [0, 1], transform=ax.transAxes, color=capacity.CAPACITY_GUIDE_COLOR,
                linestyle=capacity.CAPACITY_GUIDE_LINESTYLE,
                linewidth=capacity.CAPACITY_GUIDE_LINEWIDTH, zorder=0)


def draw_percentages(ax, values, best=None):
    centres = (np.arange(QUARTERS) + 0.5) / QUARTERS
    for q, (x, v) in enumerate(zip(centres, values)):
        box = None
        if best is not None and best[q]:
            box = dict(boxstyle="square,pad=0.2", edgecolor="red", facecolor="none", linewidth=1.2)
        ax.text(x, PCT_OFFSET, f"{v:.1f}%", transform=ax.transAxes, ha="center", va="top",
                fontsize=PCT_FONT_SIZE, bbox=box)


def closest_to_target(target, emps):
    """Per method, per quarter: True where it is (one of) the closest to the target, as displayed."""
    diff = {k: [round(abs(round(e[q], 1) - round(target[q], 1)), 1) for q in range(QUARTERS)]
            for k, e in emps.items()}
    best = [min(d[q] for d in diff.values()) for q in range(QUARTERS)]
    return {k: [d[q] <= best[q] + 1e-5 for q in range(QUARTERS)] for k, d in diff.items()}


def render_capacity_row(axes_row, columns, input_path, compare_list, dot_size, mark_best):
    """Row 1: Target = density map, every other column = that method's points, with capacities."""
    img_u8 = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"capacity input not found: {input_path}")
    image_01 = img_u8.astype(np.float64) / 255.0
    target_caps = capacity.calculate_target_capacities(image_01, quarters=QUARTERS)

    paths = {label: path for item in compare_list for label, path in item.items()}
    points, emps = {}, {}
    for col in columns:
        if col != "Target" and col in paths:
            points[col] = capacity.extract_points_from_target(paths[col], CAPACITY_N_POINTS)
            emps[col] = capacity.calculate_empirical_capacities(points[col], quarters=QUARTERS)
    best = closest_to_target(target_caps, emps) if mark_best and emps else {}

    for ax, col in zip(axes_row, columns):
        if col == showcase.SEPARATOR:
            ax.axis("off")
            continue
        if col == "Target":
            ax.imshow(image_01, cmap="gray", vmin=0.0, vmax=1.0)
            draw_quarter_guides(ax)
            draw_percentages(ax, target_caps)
        elif col in points:
            pts = points[col]
            ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=dot_size, c="black", linewidths=0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            draw_quarter_guides(ax)
            draw_percentages(ax, emps[col], best.get(col))
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="red")
        ax.axis("off")
    print(f"capacity row: target {', '.join(f'{v:.1f}%' for v in target_caps)}")


def parse_args():
    ap = argparse.ArgumentParser(description="Comparison showcase with the quadratic capacity test as its first row.")
    ap.add_argument("--output", default=OUT_DIR, help="Folder to write the panel into.")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--columns", default=",".join(showcase.COLUMNS),
                    help=f"Comma-separated columns, left to right; '{showcase.SEPARATOR}' draws a black separator.")
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--capacity-input", default=CAPACITY_INPUT, help="Density map of the capacity test.")
    ap.add_argument("--dirs", default=None,
                    help="JSON object {name: path} merged into the showcase DIR_MAP.")
    ap.add_argument("--samples", default=None,
                    help="JSON object {name: [indices]} replacing the showcase SAMPLES_MAP.")
    ap.add_argument("--mark-best", action=argparse.BooleanOptionalAction, default=MARK_BEST,
                    help="Box the capacity closest to the target in each quarter.")
    ap.add_argument("--no-headers", action="store_true", help="Hide the column labels.")
    return ap.parse_args()


def main():
    args = parse_args()
    dirs, samples = showcase.resolve_maps(args.dirs, args.samples)

    # Showcase rows, stacked in DIR_MAP order (same selection logic as the showcase script).
    rows, used, col_dirs = [], [], {}
    for name in dirs:
        idx = samples.get(name) or []
        if not idx:
            continue
        rows += showcase.resolve_rows(name, showcase.list_stems(dirs[name]), idx)
        col_dirs[name] = showcase.resolve_col_dirs(dirs[name])
        used.append(f"{len(idx)} {name}")

    columns = showcase.parse_columns(args.columns)
    widths = showcase.column_widths(columns)
    n_cols = len(columns)
    show_headers = showcase.SHOW_HEADERS and not args.no_headers
    print(f"full panel: 1 capacity row + {len(rows)} showcase rows x "
          f"{sum(c != showcase.SEPARATOR for c in columns)} cols"
          + (f" ({' + '.join(used)})" if used else "") + f"; layout {' '.join(columns)}")

    # Row 1, then a spacer holding the percentages, then the showcase rows.
    ratios = [1.0, CAPACITY_GAP] + [1.0] * len(rows)
    fig = plt.figure(figsize=(CELL * sum(widths), CELL * sum(ratios)), dpi=140)
    gs = gridspec.GridSpec(len(ratios), n_cols, figure=fig, height_ratios=ratios,
                           width_ratios=widths, wspace=0.03, hspace=ROW_GAP)

    cap_axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    render_capacity_row(cap_axes, columns, args.capacity_input, CAPACITY_COMPARE_LIST,
                        args.dot_size, args.mark_best)
    if show_headers:
        for ax, col in zip(cap_axes, columns):
            if col != showcase.SEPARATOR:
                ax.set_title(col, fontsize=HEADER_FONT_SIZE)

    last_axes = cap_axes
    for r, (dataset, stem, idx) in enumerate(rows):
        last_axes = []
        for c, col in enumerate(columns):
            ax = fig.add_subplot(gs[2 + r, c])
            last_axes.append(ax)
            if col == showcase.SEPARATOR:
                ax.axis("off")
                continue
            showcase.render_cell(ax, col_dirs[dataset], col, stem, args.dot_size)
    # one continuous line per separator, from the capacity row down to the last showcase row
    showcase.draw_separators(fig, columns, cap_axes, last_axes)

    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)
    pdf = out_base / f"{args.out_name}.pdf"
    png = out_base / f"{args.out_name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {pdf}\n       {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
