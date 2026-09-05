"""qualitative_ours_showcase.py

Our-results showcase: a free grid of [Target, Ours] pairs (like the teaser, but
each cell is a sample shown as Target + our stipple result).

Arrange it with nested lists -- COLUMN-major, like the teaser: one inner list per
COLUMN, its entries the sample indices read TOP-TO-BOTTOM down that column. So

    VALID_SAMPLES_ICONS = [[0, 1, 2], [3, 4, 5]]

is 2 pair-columns x 3 rows:

    [0] [3]
    [1] [4]
    [2] [5]

Each index expands to the OURS_SUBCOLS cells (default: Target then Ours). Icons
grid on top, faces grid ALWAYS below it. A flat list is treated as a single
column. Columns may be ragged (short ones are blank-padded). Use [] to drop a block.
Blocks stack icons -> faces -> shapenet.

Images come straight from the validation folders (no model, no rendering):
    Target   -> source/<stem>.png              (shown as an image)
    Ours-WVS -> target_CN-WVS_<N>/<stem>.npy   (vector scatter; ControlNet on WVS targets)
    Ours-GBN -> target_CN-GBN_<N>/<stem>.npy   (vector scatter; ControlNet on GBN targets)

<N> is the dataset's point budget and is NOT hard-coded: icons/faces use 1024 and
ShapeNet uses 1600, so each target folder is resolved by glob per dataset.

The stipple result is drawn from the .npy points (crisp at any size, vector in the
PDF) rather than the rendered .png (which fades when downsampled).

A "sample index" is a position into that dataset's validation_manifest.json order
(the order the validation split was drawn in); if a dataset has no manifest, the
alphabetical order of source/*.png is used instead. NOTE: manifest order is NOT
alphabetical, so an index selects a different image than it did when this script
sorted the source folder.

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

# ── Grid selection (edit these to find the panel you want) ────────────────────
# COLUMN-major: one inner list per column, entries read top-to-bottom. Faces are
# ALWAYS below icons. A flat list = a single column. Use [] to drop a block.

# MAIN
VALID_SAMPLES_ICONS = [
    [33, 16, 18], 
    [19, 20, 22], 
    [23, 24, 26], 
    [28, 29, 32]
]
VALID_SAMPLES_FACES = [[7], [11], [14], [10]]
VALID_SAMPLES_SHAPENET = []
OUT_NAME = "qualitative_ours"

# APPENDIX
# VALID_SAMPLES_ICONS = [
#     [35, 36, 97, 38, 39, 40, 41, 42, 43, 44, 45], 
#     [51, 52, 100, 54, 55, 101, 57, 58, 59, 60, 61], 
#     [67, 68, 69, 104, 106, 72, 73, 74, 75, 76, 109], 
#     [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92],
#     [127, 47, 98, 125, 120, 63, 64, 65, 108, 79, 80]
# ]
# VALID_SAMPLES_FACES = [[15], [16], [19], [18], [3]]
# VALID_SAMPLES_SHAPENET = []
# OUT_NAME = "qualitative_ours_appendix"

# What each sample cell shows, left to right. Edit to ["Ours"] for result-only.
OURS_SUBCOLS = ["Target", "Ours-WVS", "Ours-GBN"]

COL_TO_DIR = {
    "Target": "source",
    "Ours-WVS": "target_CN-WVS_*",
    "Ours-GBN": "target_CN-GBN_*",
}

CELL = 2.0             # inches per cell
DOT_SIZE = 2.0         # scatter marker size (pt^2) for the "Ours" cells
SHOW_HEADERS = False   # off by default (matches the tiled look); True repeats labels on row 0
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
    """{sub-column: real subfolder path or None}, resolving the point-budget glob once.

    COL_TO_DIR holds patterns like target_CN-WVS_* because the budget differs per dataset
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


def normalize_cols(valid_samples):
    """Return VALID_SAMPLES as a list of COLUMNS (each a list of indices). Flat -> one column."""
    if not valid_samples:
        return []
    if isinstance(valid_samples[0], (list, tuple)):
        return [[int(i) for i in col] for col in valid_samples]
    return [[int(i) for i in valid_samples]]


def build_block(dataset, stems, columns):
    """Column-major -> list of grid-rows. Each grid-row is a list of cells, where a
    cell is (dataset, stem, sub) or None (blank pad for a ragged/short column).
    """
    if not columns:
        return []
    for c, col in enumerate(columns):
        bad = [i for i in col if not (0 <= i < len(stems))]
        if bad:
            raise IndexError(f"{dataset} column {c} indices {bad} out of range (0..{len(stems) - 1})")
    n_rows = max(len(col) for col in columns)
    grid = []
    for r in range(n_rows):
        row_cells = []
        for col in columns:
            if r < len(col):
                stem = stems[col[r]]
                for sub in OURS_SUBCOLS:
                    row_cells.append((dataset, stem, sub))
            else:
                row_cells.extend([None] * len(OURS_SUBCOLS))   # blank pad this pair
        grid.append(row_cells)
    return grid


def render_cell(ax, col_dirs, sub, stem, dot_size):
    """Target -> image; Ours -> vector point scatter from .npy."""
    d = col_dirs.get(sub)
    if sub == "Target":
        p = (d / f"{stem}.png") if d else None
        if p is not None and p.exists():
            im = plt.imread(str(p))
            if im.ndim == 2:
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
    ap = argparse.ArgumentParser(description="Our-results showcase (Target|Ours pairs, column-major grid).")
    ap.add_argument("--icons-dir", default=ICONS_DIR)
    ap.add_argument("--faces-dir", default=FACES_DIR)
    ap.add_argument("--shapenet-dir", default=SHAPENET_DIR)
    ap.add_argument("--output", default=OUT_DIR, help="Folder to write the panel into.")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--icons-samples", default=json.dumps(VALID_SAMPLES_ICONS),
                    help="JSON column-major nested list (or flat = one column) of icon indices.")
    ap.add_argument("--faces-samples", default=json.dumps(VALID_SAMPLES_FACES),
                    help="JSON column-major nested list (or flat = one column) of face indices.")
    ap.add_argument("--shapenet-samples", default=json.dumps(VALID_SAMPLES_SHAPENET),
                    help="JSON column-major nested list (or flat = one column) of ShapeNet indices.")
    ap.add_argument("--headers", action="store_true", help="Show Target/Ours labels on the top row.")
    return ap.parse_args()


def main():
    args = parse_args()
    icons_cols = normalize_cols(json.loads(args.icons_samples))
    faces_cols = normalize_cols(json.loads(args.faces_samples))
    shape_cols = normalize_cols(json.loads(args.shapenet_samples))

    roots = {"icons": Path(args.icons_dir), "faces": Path(args.faces_dir),
             "shapenet": Path(args.shapenet_dir)}
    icons_stems = list_stems(args.icons_dir) if icons_cols else []
    faces_stems = list_stems(args.faces_dir) if faces_cols else []
    shape_stems = list_stems(args.shapenet_dir) if shape_cols else []

    # Blocks stacked as extra rows: icons -> faces -> shapenet.
    all_rows = (build_block("icons", icons_stems, icons_cols)
                + build_block("faces", faces_stems, faces_cols)
                + build_block("shapenet", shape_stems, shape_cols))
    if not all_rows:
        raise ValueError("All VALID_SAMPLES lists are empty; nothing to show.")

    # Resolve each dataset's point-budget folders once (icons/faces 1024, shapenet 1600).
    col_dirs = {name: resolve_col_dirs(root) for name, root in roots.items()}

    n_rows = len(all_rows)
    n_cols = max(len(row) for row in all_rows)
    show_headers = SHOW_HEADERS or args.headers
    print(f"ours panel: {n_rows} rows x {n_cols} cols (max), pairs of {OURS_SUBCOLS} (column-major)")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(CELL * n_cols, CELL * n_rows),
                             dpi=140, squeeze=False)
    for r, row in enumerate(all_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            cell = row[c] if c < len(row) else None
            if cell is None:
                ax.axis("off")               # blank pad (ragged column or short row)
                continue
            dataset, stem, sub = cell
            render_cell(ax, col_dirs[dataset], sub, stem, args.dot_size)
            if r == 0 and show_headers:
                ax.set_title(sub, fontsize=12)

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
