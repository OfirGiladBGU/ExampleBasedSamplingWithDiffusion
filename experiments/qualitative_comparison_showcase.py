"""qualitative_comparison_showcase.py

Comparison showcase: the fixed 5 columns [Target, WVS, BNOT, GBN, Ours], one row
per sample. Icons rows come first, faces rows are ALWAYS placed below them. Control
each block with its own flat VALID_SAMPLES list; an empty list [] drops that block.

Images come straight from the *_results_compare folders (no model, no rendering):
    Target -> source/<stem>.png              (shown as an image)
    WVS    -> target_WVS_1024/<stem>.npy      (shown as vector point scatter)
    BNOT   -> target_BNOT_1024/<stem>.npy
    GBN    -> target_GBN_1024/<stem>.npy
    Ours   -> target_CN_1024/<stem>.npy       (CN = our ControlNet model)

The stipple columns are drawn from the .npy points (crisp at any size, vector in
the PDF) rather than the rendered .png (which fades when downsampled).

A "sample index" is a position into the sorted list of source/*.png stems of that
dataset, so VALID_SAMPLES_ICONS = [0, 5, 12] picks the 1st, 6th and 13th icon.

Writes: OUT_DIR/<out-name>.pdf (+ .png)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Data locations ────────────────────────────────────────────────────────────
ICONS_DIR = "experiments/outputs/icons_results_compare"
FACES_DIR = "experiments/outputs/faces_results_compare"
OUT_DIR = "experiments/outputs/qualitative_showcase"

# ── Row selection (edit these to find the panel you want) ─────────────────────
# Indices into the sorted source stems of each dataset. Faces are ALWAYS below
# icons. Use [] to show only icons or only faces.

# MAIN left
VALID_SAMPLES_ICONS = [0, 1, 2, 3]
VALID_SAMPLES_FACES = [1, 2]
OUT_NAME = "qualitative_comparison_left"

# MAIN right
# VALID_SAMPLES_ICONS = [4, 7, 8, 14]
# VALID_SAMPLES_FACES = [8, 12]
# OUT_NAME = "qualitative_comparison_right"

# APPENDIX left
# VALID_SAMPLES_ICONS = [150, 151, 200, 201, 202, 205, 156, 157, 158, 159, 203, 161]
# VALID_SAMPLES_FACES = []
# OUT_NAME = "qualitative_comparison_appendix_left"

# APPENDIX right
# VALID_SAMPLES_ICONS = [301, 311, 164, 165, 166, 305, 168, 315, 170, 171, 172, 307]
# VALID_SAMPLES_FACES = []
# OUT_NAME = "qualitative_comparison_appendix_right"

# ── Fixed columns -> subfolder ────────────────────────────────────────────────
COLUMNS = ["Target", "WVS", "BNOT", "GBN", "Ours"]
COL_TO_DIR = {
    "Target": "source",
    "WVS": "target_WVS_1024",
    "BNOT": "target_BNOT_1024",
    "GBN": "target_GBN_1024",
    "Ours": "target_CN_1024",
}

CELL = 2.0            # inches per cell
DOT_SIZE = 2.0        # scatter marker size (pt^2) for the stipple columns
SHOW_HEADERS = True   # column labels on the top row
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_stems(dataset_dir):
    """Sorted stems of the Target (source) images of a dataset."""
    src = Path(dataset_dir) / "source"
    if not src.is_dir():
        return []
    return sorted(p.stem for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)


def resolve_rows(dataset, stems, indices):
    """Turn a flat index list into (dataset, stem, index) rows; validate range."""
    bad = [i for i in indices if not (0 <= i < len(stems))]
    if bad:
        raise IndexError(f"{dataset} VALID_SAMPLES {bad} out of range (0..{len(stems) - 1})")
    return [(dataset, stems[i], i) for i in indices]


def render_cell(ax, root, column, stem, dot_size):
    """Target -> image; every other column -> vector point scatter from .npy."""
    if column == "Target":
        p = root / COL_TO_DIR[column] / f"{stem}.png"
        if p.exists():
            im = plt.imread(str(p))
            if im.ndim == 2:                     # pin grayscale, no auto-contrast
                ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(im)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="red")
    else:
        p = root / COL_TO_DIR[column] / f"{stem}.npy"
        if p.exists():
            pts = np.load(p).astype(np.float64)
            ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=dot_size, c="black", linewidths=0)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="red")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    ax.axis("off")


def parse_args():
    ap = argparse.ArgumentParser(description="Comparison showcase (Target/WVS/BNOT/GBN/Ours).")
    ap.add_argument("--icons-dir", default=ICONS_DIR)
    ap.add_argument("--faces-dir", default=FACES_DIR)
    ap.add_argument("--output", default=OUT_DIR, help="Folder to write the panel into.")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--icons-samples", default=json.dumps(VALID_SAMPLES_ICONS),
                    help="JSON flat list of icon row indices (overrides the in-file list).")
    ap.add_argument("--faces-samples", default=json.dumps(VALID_SAMPLES_FACES),
                    help="JSON flat list of face row indices (overrides the in-file list).")
    ap.add_argument("--no-headers", action="store_true", help="Hide the column labels.")
    return ap.parse_args()


def main():
    args = parse_args()
    icon_idx = json.loads(args.icons_samples)
    face_idx = json.loads(args.faces_samples)

    roots = {"icons": Path(args.icons_dir), "faces": Path(args.faces_dir)}
    icons_stems = list_stems(args.icons_dir) if icon_idx else []
    faces_stems = list_stems(args.faces_dir) if face_idx else []

    rows = resolve_rows("icons", icons_stems, icon_idx) + resolve_rows("faces", faces_stems, face_idx)
    if not rows:
        raise ValueError("Both VALID_SAMPLES lists are empty; nothing to show.")

    n_rows, n_cols = len(rows), len(COLUMNS)
    show_headers = SHOW_HEADERS and not args.no_headers
    print(f"comparison panel: {n_rows} rows x {n_cols} cols "
          f"({len(icon_idx)} icons + {len(face_idx)} faces)")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(CELL * n_cols, CELL * n_rows),
                             dpi=140, squeeze=False)
    for r, (dataset, stem, idx) in enumerate(rows):
        for c, col in enumerate(COLUMNS):
            ax = axes[r][c]
            render_cell(ax, roots[dataset], col, stem, args.dot_size)
            if r == 0 and show_headers:
                ax.set_title(col, fontsize=13)

    fig.subplots_adjust(wspace=0.03, hspace=0.03)
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
