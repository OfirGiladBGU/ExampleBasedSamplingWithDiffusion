"""qualitative_comparison_showcase.py

Comparison showcase: the fixed columns [Target, WVS, BNOT, GBN, Ours-WVS, Ours-GBN], one row
per sample. Icons rows come first, faces rows are ALWAYS placed below them. Control
each block with its own flat VALID_SAMPLES list; an empty list [] drops that block.

Images come straight from the validation folders (no model, no rendering):
    Target   -> source/<stem>.png            (shown as an image)
    WVS      -> target_WVS_<N>/<stem>.npy    (shown as vector point scatter)
    BNOT     -> target_BNOT_<N>/<stem>.npy
    GBN      -> target_GBN_<N>/<stem>.npy
    Ours-WVS -> target_CN-WVS_<N>/<stem>.npy   (our ControlNet, trained on WVS targets)
    Ours-GBN -> target_CN-GBN_<N>/<stem>.npy   (our ControlNet, trained on GBN targets)

<N> is the dataset's point budget and is NOT hard-coded: icons/faces use 1024 and
ShapeNet uses 1600, so each target folder is resolved by glob per dataset.

The stipple columns are drawn from the .npy points (crisp at any size, vector in
the PDF) rather than the rendered .png (which fades when downsampled).

A "sample index" is a position into that dataset's validation_manifest.json order
(the order the validation split was drawn in). If a dataset has no manifest, the
alphabetical order of source/*.png is used instead. NOTE: manifest order is NOT
alphabetical, so an index selects a different image than it did when this script sorted
the source folder.

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
ICONS_DIR = "experiments/outputs/z_validation_data/Icons-50_1024"
FACES_DIR = "experiments/outputs/z_validation_data/CelebA-5K_1024"
SHAPENET_DIR = "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K_1600"
OUT_DIR = "experiments/outputs/qualitative_showcase"

# ── Row selection (edit these to find the panel you want) ─────────────────────
# Indices into the sorted source stems of each dataset. Faces are ALWAYS below
# icons. Use [] to show only icons or only faces.

# MAIN left
VALID_SAMPLES_ICONS = [0, 1, 2, 3]
VALID_SAMPLES_FACES = [1, 2]
VALID_SAMPLES_SHAPENET = []
OUT_NAME = "qualitative_comparison_left"

# MAIN right
# VALID_SAMPLES_ICONS = [4, 7, 8, 14]
# VALID_SAMPLES_FACES = [8, 12]
# VALID_SAMPLES_SHAPENET = []
# OUT_NAME = "qualitative_comparison_right"

# APPENDIX left
# VALID_SAMPLES_ICONS = [150, 151, 200, 201, 202, 205, 156, 157, 158, 159, 203, 161]
# VALID_SAMPLES_FACES = []
# VALID_SAMPLES_SHAPENET = []
# OUT_NAME = "qualitative_comparison_appendix_left"

# APPENDIX right
# VALID_SAMPLES_ICONS = [301, 311, 164, 165, 166, 305, 168, 315, 170, 171, 172, 307]
# VALID_SAMPLES_FACES = []
# VALID_SAMPLES_SHAPENET = []
# OUT_NAME = "qualitative_comparison_appendix_right"

# ── Fixed columns -> subfolder ────────────────────────────────────────────────
COLUMNS = ["Target", "WVS", "BNOT", "GBN", "Ours-WVS", "Ours-GBN"]
COL_TO_DIR = {
    "Target": "source",
    "WVS": "target_WVS_*",
    "BNOT": "target_BNOT_*",
    "GBN": "target_GBN_*",
    "Ours-WVS": "target_CN-WVS_*",
    "Ours-GBN": "target_CN-GBN_*",
}

CELL = 2.0            # inches per cell
DOT_SIZE = 2.0        # scatter marker size (pt^2) for the stipple columns
SHOW_HEADERS = True   # column labels on the top row
IMG_EXTS = {".png", ".jpg", ".jpeg"}
MANIFEST_NAME = "validation_manifest.json"


def list_stems(dataset_dir):
    """Stems in validation_manifest.json order, else alphabetical source order.

    The manifest records the validation split in the order it was drawn, which is what
    every other stage indexes by -- so a sample index means the same image here as it
    does elsewhere. Any source image not named in the manifest is appended afterwards
    (alphabetically) so it stays reachable rather than silently unselectable.
    """
    root = Path(dataset_dir)
    src = root / "source"
    if not src.is_dir():
        return []
    alpha = sorted(p.stem for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)

    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        return alpha

    have = set(alpha)
    ordered, seen = [], set()
    for name in json.loads(manifest.read_text()):
        stem = Path(name).stem
        if stem in have and stem not in seen:
            ordered.append(stem)
            seen.add(stem)
    extra = [s for s in alpha if s not in seen]
    if extra:
        print(f"  [note] {root.name}: {len(extra)} source image(s) not in the manifest, "
              f"appended after it (e.g. {extra[:2]})")
    return ordered + extra


def resolve_col_dirs(root):
    """{column: real subfolder path or None}, resolving the point-budget glob once.

    COL_TO_DIR holds patterns like target_WVS_* because the budget differs per dataset
    (1024 for icons/faces, 1600 for ShapeNet). Resolved here rather than per cell.
    """
    root = Path(root)
    out = {}
    for col, pattern in COL_TO_DIR.items():
        if "*" not in pattern:
            out[col] = root / pattern
            continue
        matches = sorted(p for p in root.glob(pattern) if p.is_dir())
        if not matches:
            out[col] = None
        else:
            if len(matches) > 1:
                print(f"  [warn] {root.name}: {pattern} matched {len(matches)} folders "
                      f"{[m.name for m in matches]}; using {matches[0].name}")
            out[col] = matches[0]
    return out


def resolve_rows(dataset, stems, indices):
    """Turn a flat index list into (dataset, stem, index) rows; validate range."""
    bad = [i for i in indices if not (0 <= i < len(stems))]
    if bad:
        raise IndexError(f"{dataset} VALID_SAMPLES {bad} out of range (0..{len(stems) - 1})")
    return [(dataset, stems[i], i) for i in indices]


def render_cell(ax, col_dirs, column, stem, dot_size):
    """Target -> image; every other column -> vector point scatter from .npy."""
    d = col_dirs.get(column)
    if column == "Target":
        p = (d / f"{stem}.png") if d else None
        if p is not None and p.exists():
            im = plt.imread(str(p))
            if im.ndim == 2:                     # pin grayscale, no auto-contrast
                ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(im)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="red")
    else:
        p = (d / f"{stem}.npy") if d else None
        if p is not None and p.exists():
            pts = np.load(p).astype(np.float64)
            ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=dot_size, c="black", linewidths=0)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="red")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    ax.axis("off")


def parse_args():
    ap = argparse.ArgumentParser(description="Comparison showcase (Target/WVS/BNOT/GBN/Ours-WVS/Ours-GBN).")
    ap.add_argument("--icons-dir", default=ICONS_DIR)
    ap.add_argument("--faces-dir", default=FACES_DIR)
    ap.add_argument("--shapenet-dir", default=SHAPENET_DIR)
    ap.add_argument("--output", default=OUT_DIR, help="Folder to write the panel into.")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--icons-samples", default=json.dumps(VALID_SAMPLES_ICONS),
                    help="JSON flat list of icon row indices (overrides the in-file list).")
    ap.add_argument("--faces-samples", default=json.dumps(VALID_SAMPLES_FACES),
                    help="JSON flat list of face row indices (overrides the in-file list).")
    ap.add_argument("--shapenet-samples", default=json.dumps(VALID_SAMPLES_SHAPENET),
                    help="JSON flat list of ShapeNet row indices (overrides the in-file list).")
    ap.add_argument("--no-headers", action="store_true", help="Hide the column labels.")
    return ap.parse_args()


def main():
    args = parse_args()
    icon_idx = json.loads(args.icons_samples)
    face_idx = json.loads(args.faces_samples)
    shape_idx = json.loads(args.shapenet_samples)

    roots = {"icons": Path(args.icons_dir), "faces": Path(args.faces_dir),
             "shapenet": Path(args.shapenet_dir)}
    icons_stems = list_stems(args.icons_dir) if icon_idx else []
    faces_stems = list_stems(args.faces_dir) if face_idx else []
    shape_stems = list_stems(args.shapenet_dir) if shape_idx else []

    # Rows are stacked icons -> faces -> shapenet.
    rows = (resolve_rows("icons", icons_stems, icon_idx)
            + resolve_rows("faces", faces_stems, face_idx)
            + resolve_rows("shapenet", shape_stems, shape_idx))
    if not rows:
        raise ValueError("All VALID_SAMPLES lists are empty; nothing to show.")

    # Resolve each dataset's point-budget folders once (icons/faces 1024, shapenet 1600).
    col_dirs = {name: resolve_col_dirs(root) for name, root in roots.items()}

    n_rows, n_cols = len(rows), len(COLUMNS)
    show_headers = SHOW_HEADERS and not args.no_headers
    print(f"comparison panel: {n_rows} rows x {n_cols} cols "
          f"({len(icon_idx)} icons + {len(face_idx)} faces + {len(shape_idx)} shapenet)")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(CELL * n_cols, CELL * n_rows),
                             dpi=140, squeeze=False)
    for r, (dataset, stem, idx) in enumerate(rows):
        for c, col in enumerate(COLUMNS):
            ax = axes[r][c]
            render_cell(ax, col_dirs[dataset], col, stem, args.dot_size)
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
