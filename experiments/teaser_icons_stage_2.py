"""teaser_icons_stage_2.py

Stage 2 of the teaser figure. Reads the per-sample artifacts written by
teaser_icons_stage_1.py and lays them out like the paper teaser: one or more
BIG columns side by side, each big column being a [ Target | Result | OT Map ]
triple, with the selected samples as rows.

Set the layout with VALID_SAMPLES as a list-of-lists -- one inner list per big
column, its entries the split_index values (from selection.json) shown as rows:

    VALID_SAMPLES = [[0, 1, 2, 3], [4, 5, 6, 7]]   # two big columns, 4 rows each

A flat list is treated as a single big column. Big columns are separated by a
vertical line. Result cells are VECTOR scatter and OT-Map cells are VECTOR quiver
(each with its own colorbar), so the PDF stays crisp at any zoom.

Reads ONLY from the stage-1 folder (no model):
    OUTPUT_DIR/selection.json
    OUTPUT_DIR/target/<stem>.png     (Target condition image)
    OUTPUT_DIR/result/<stem>.npy     (predicted points, N x 2)
    OUTPUT_DIR/ot_map/<stem>.npy     (GT GBN offset field, 2 x G x G)

Writes:
    OUTPUT_DIR/teaser_icons_panel.pdf   (+ .png)
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "experiments/outputs/teaser_icons_results"

# Layout: list-of-lists -> one inner list per big column (entries are split_index
# values shown as rows). A flat list is a single big column.
VALID_SAMPLES = [[1, 2, 4, 5], [6, 7, 8, 9]]

# The three sub-columns of every big column, left to right.
SUBCOLS = ["Target", "Result", "OT Map"]

DOT_SIZE = 1.0        # vector scatter marker size (pt^2) for the Result column
QUIVER_CMAP = "viridis"
QUIVER_WIDTH = 0.004
SEP_RATIO = 0.12      # separator column width relative to a content column
CELL = 2.0            # inches per content column / row
SHOW_COLORBAR = True  # per-cell colorbar on the OT Map column


def normalize_big_cols(valid_samples):
    """Return VALID_SAMPLES as a list of big columns (each a list of split_index)."""
    if valid_samples and isinstance(valid_samples[0], (list, tuple)):
        return [[int(i) for i in group] for group in valid_samples]
    return [[int(i) for i in valid_samples]]


def build_column_spec(n_big_cols):
    """Flatten big columns into (kind, big_col, subcol) columns with 'sep' between big cols."""
    spec = []
    for bc in range(n_big_cols):
        for sub in SUBCOLS:
            spec.append(("content", bc, sub))
        if bc < n_big_cols - 1:
            spec.append(("sep", None, None))
    return spec


def draw_quiver(ax, offsets):
    """Draw the GT offset field as a magnitude-coloured quiver; return the handle."""
    n_grid = offsets.shape[-1]
    yy, xx = np.mgrid[0:n_grid, 0:n_grid]
    dx, dy = offsets[0], offsets[1]
    mag = np.sqrt(dx * dx + dy * dy)
    q = ax.quiver(xx, yy, dx, dy, mag, angles="xy", scale_units="xy", scale=1.0,
                  cmap=QUIVER_CMAP, width=QUIVER_WIDTH)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    return q


def parse_args():
    ap = argparse.ArgumentParser(description="Teaser stage 2: assemble the big-column panel.")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Stage-1 folder.")
    ap.add_argument("--valid-samples", default=json.dumps(VALID_SAMPLES),
                    help="JSON list-of-lists (big columns) or a flat list (one big column).")
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--out-name", default="teaser_icons_panel")
    ap.add_argument("--no-headers", action="store_true", help="Hide the Target/Result/OT Map labels.")
    ap.add_argument("--no-colorbar", action="store_true", help="Hide the OT-Map colorbars.")
    return ap.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output)

    sel_rows = json.loads((out_base / "selection.json").read_text())["selected_rows"]
    by_index = {int(r["split_index"]): r for r in sel_rows}

    try:
        raw = json.loads(args.valid_samples)
    except (ValueError, TypeError):
        raw = ast.literal_eval(args.valid_samples)
    big_cols = normalize_big_cols(raw)

    missing = [i for group in big_cols for i in group if i not in by_index]
    if missing:
        raise KeyError(f"VALID_SAMPLES {sorted(set(missing))} not in selection.json "
                       f"(available split_index: {sorted(by_index)})")

    col_spec = build_column_spec(len(big_cols))
    n_rows = max((len(group) for group in big_cols), default=0)
    if n_rows == 0:
        raise ValueError("VALID_SAMPLES has no samples")
    n_cols = len(col_spec)
    width_ratios = [1.0 if kind == "content" else SEP_RATIO for kind, _, _ in col_spec]

    n_content = sum(1 for k, _, _ in col_spec if k == "content")
    print(f"panel: {n_rows} rows x {n_content} content cols "
          f"(+{n_cols - n_content} separators); big columns {big_cols}")

    show_cbar = SHOW_COLORBAR and not args.no_colorbar
    fig_w = CELL * sum(width_ratios) * (1.15 if show_cbar else 1.0)
    fig_h = CELL * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=140, squeeze=False,
                             gridspec_kw={"width_ratios": width_ratios})

    for r in range(n_rows):
        for c, (kind, bc, sub) in enumerate(col_spec):
            ax = axes[r][c]

            if kind == "sep":
                ax.axvline(0.5, color="black", lw=1.4)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
                continue

            group = big_cols[bc]
            if r == 0 and not args.no_headers:
                ax.set_title(sub, fontsize=12)

            if r >= len(group):          # ragged big column -> blank cell
                ax.axis("off")
                continue

            sample = by_index[group[r]]
            stem = sample["stem"]

            if sub == "Target":
                p = out_base / "target" / f"{stem}.png"
                if p.exists():
                    im = plt.imread(str(p))
                    # Pin the grayscale range to [0,1] so imshow does NOT auto-contrast-
                    # stretch (which made light icons render far darker than the file).
                    if im.ndim == 2:
                        ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
                    else:
                        ax.imshow(im)
                else:
                    ax.text(0.5, 0.5, "no target", ha="center", va="center", fontsize=6, color="red")
                ax.set_aspect("equal")
                ax.axis("off")

            elif sub == "Result":
                p = out_base / "result" / f"{stem}.npy"
                if p.exists():
                    pts = np.load(p).astype(np.float64)
                    ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=args.dot_size, c="black", linewidths=0)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="red")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

            else:  # OT Map
                p = out_base / "ot_map" / f"{stem}.npy"
                if p.exists():
                    offsets = np.load(p).astype(np.float64)
                    q = draw_quiver(ax, offsets)
                    if show_cbar:
                        fig.colorbar(q, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="red")
                    ax.axis("off")

    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    pdf = out_base / f"{args.out_name}.pdf"
    png = out_base / f"{args.out_name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {pdf}\n       {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
