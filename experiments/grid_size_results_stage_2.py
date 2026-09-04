#!/usr/bin/env python3
"""
Merge sampled point-count results into a single labelled comparison figure.

Point-count cells are rendered from the raw `.npy` point sets (vector scatter)
so the output PDF is high quality / zoomable.  "Target" cells are raster
images shown with imshow.  Cell type is auto-detected by file extension:
`.npy` -> scatter, anything else -> image.

Reads what grid_size_results_stage_0.py staged and grid_size_results_stage_1.py produced;
to change which images / grid sizes are shown, edit ROW_IMAGES / GRID_SIZES below (keeping
GRID_SIZES in step with stage 1).

    python experiments/grid_size_results_stage_2.py            # -> pointcount_generalization.pdf
    python experiments/grid_size_results_stage_2.py -o fig.png --dpi 300
    python experiments/grid_size_results_stage_2.py --marker 1.5 --auto-marker

Point conventions (matching sample_control.py):
  npy is (N, 2) float in [0, 1]; plotted as scatter(x, 1 - y) with the y axis
  flipped so orientation matches the produced pngs.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ========================================================================== #
# Paths are derived from the stage 0/1 layout, so they cannot drift out of sync:
#   stage 0 -> BASE_DIR/source/<image>
#   stage 1 -> BASE_DIR/sample_outputs_<grid>/<stem>/npy/<stem>.npy
# Each row is one image: the Target condition followed by one cell per grid size,
# labelled with its point count (grid*grid).
# ========================================================================== #
BASE_DIR = "experiments/outputs/grid_size_results"

# Must match grid_size_results_stage_1.GRID_SIZES.
GRID_SIZES = [16, 24, 32, 48, 64]

# Which images to show, and how to lay them out. Each image contributes a group of cells
# (its Target followed by one cell per grid size); the nesting controls how the groups are
# arranged into rows:
#
#   ROW_IMAGES = ["a.png", "b.png"]          -> ONE row: a's group then b's group, side by side
#   ROW_IMAGES = [["a.png"], ["b.png"]]      -> TWO rows, one image each (stacked)
#   ROW_IMAGES = [["a.png", "b.png"], ["c.png"]]  -> row 1 has a+b side by side, row 2 has c
#
# Names are the files stage 0 stages into source/.
ROW_IMAGES = [
    "emoji-one_4_monkey.png",
    "stress_test_density.png",
]


def _normalize_rows(row_images):
    """Accept a flat list (one row with every image) or a list of lists (one row each)."""
    if all(isinstance(x, str) for x in row_images):
        return [list(row_images)]
    return [[x] if isinstance(x, str) else list(x) for x in row_images]


def _image_group(image_name, base_dir, grid_sizes):
    """One image's cells: its Target, then one cell per grid size labelled grid*grid."""
    stem = os.path.splitext(image_name)[0]
    group = [{"Target": os.path.join(base_dir, "source", image_name)}]
    for g in grid_sizes:
        npy = os.path.join(base_dir, "sample_outputs_%d" % g, stem, "npy", "%s.npy" % stem)
        group.append({str(g * g): npy})
    return group


def _build_image_paths(base_dir=BASE_DIR, grid_sizes=GRID_SIZES, row_images=ROW_IMAGES):
    """[[{label: path}, ...], ...] -- see ROW_IMAGES for how rows are formed."""
    rows = []
    for row in _normalize_rows(row_images):
        cells = []
        for image_name in row:
            cells.extend(_image_group(image_name, base_dir, grid_sizes))
        rows.append(cells)
    return rows


IMAGE_PATHS = _build_image_paths()
OUTPUT_PATH = "experiments/outputs/grid_size_results/pointcount_generalization.pdf"

# Labels (the dict keys) rendered in bold.
BOLD_LABELS = ["Target", "1024"]
# ========================================================================== #


def display_label(key):
    return "%s pts" % key if key.isdigit() else key


# --------------------------------------------------------------------------- #
# Cell renderers
# --------------------------------------------------------------------------- #
def _placeholder(ax, text="missing"):
    ax.text(0.5, 0.5, text, ha="center", va="center", color="0.6",
            fontsize=9, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.8")


def draw_points(ax, path, marker, alpha):
    pts = np.asarray(np.load(path), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        _placeholder(ax, "bad npy")
        return
    # y flipped to match the produced pngs (see sample_control.save_sample_image)
    ax.scatter(pts[:, 0], 1.0 - pts[:, 1], c="black", s=marker, alpha=alpha,
               linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_image(ax, path):
    img = mpimg.imread(path)
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_cell(ax, path, marker, alpha):
    if not path or not os.path.isfile(path):
        if path:
            print("  ! missing: %s" % path)
        _placeholder(ax)
        ax.axis("off")
        return
    try:
        if path.lower().endswith(".npy"):
            draw_points(ax, path, marker, alpha)
        else:
            draw_image(ax, path)
    except Exception as exc:  # noqa: BLE001
        print("  ! failed on %s: %s" % (path, exc))
        _placeholder(ax, "load error")
        ax.axis("off")


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #
def render(out_path, cell=2.2, marker=2.0, alpha=0.85, dpi=300,
           auto_marker=False):
    rows = IMAGE_PATHS
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell, n_rows * cell),
        squeeze=False,
    )

    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(row):
                ax.axis("off")
                continue
            (key, path), = row[c].items()   # single-entry dict -> (label, path)
            m = marker
            if auto_marker and path and path.lower().endswith(".npy") and key.isdigit():
                m = max(0.3, min(8.0, marker * 1024.0 / int(key)))
            draw_cell(ax, path, m, alpha)
            ax.set_title(display_label(key), fontsize=12,
                         fontweight="bold" if key in BOLD_LABELS else "normal")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005,
                        wspace=0.06, hspace=0.18)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s  (%d rows x %d cols, dpi=%d)" % (out_path, n_rows, n_cols, dpi))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=OUTPUT_PATH,
                    help="output path; .pdf gives vector output (default)")
    ap.add_argument("--cell", type=float, default=2.2,
                    help="cell size in inches")
    ap.add_argument("--marker", type=float, default=2.0,
                    help="scatter marker size (points^2)")
    ap.add_argument("--auto-marker", action="store_true",
                    help="scale marker size down as point count grows")
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="scatter point opacity")
    ap.add_argument("--dpi", type=int, default=300,
                    help="dpi (affects raster outputs / embedded images)")
    args = ap.parse_args()

    render(args.out, cell=args.cell, marker=args.marker, alpha=args.alpha,
           dpi=args.dpi, auto_marker=args.auto_marker)


if __name__ == "__main__":
    main()
