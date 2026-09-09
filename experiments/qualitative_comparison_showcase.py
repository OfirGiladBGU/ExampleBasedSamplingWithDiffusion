"""qualitative_comparison_showcase.py

Comparison showcase: the fixed columns [Target, WVS, BNOT, GBN, Ours-WVS, Ours-GBN], one row
per sample.

Datasets are configured by two maps keyed by the same names, so adding one is a single line
in each rather than a new set of module-level variables:

    DIR_MAP     = {"icons": <path>, "faces": <path>, ...}    where each dataset lives
    SAMPLES_MAP = {"icons": [9, 18], "faces": [2], ...}      which rows to take from it

Blocks are stacked in DIR_MAP order, so that map alone decides which dataset appears above
which. A name missing from SAMPLES_MAP, or mapped to [], contributes no rows.

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

# ── Datasets: where each one lives ────────────────────────────────
# Add a dataset with one entry here and one in SAMPLES_MAP under the same key.
# Blocks are stacked in THIS order (first key = top rows of the panel).
DIR_MAP = {
    "icons": "experiments/outputs/z_validation_data/Icons-50_1024",
    "faces": "experiments/outputs/z_validation_data/CelebA-5K_1024",
    # The combined 3K set, superseded by the per-category sets below. Kept so old index
    # lists stay readable; nothing selects from it any more.
    "shapenet": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K_1600",
    "airplanes": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Airplanes_1600",
    "cars": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Cars_1600",
    "watercrafts": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Watercrafts_1600",
}
OUT_DIR = "experiments/outputs/qualitative_showcase"

# ── Row selection (edit these to find the panel you want) ───────────────────
# {dataset name: flat list of sample indices}. A key absent here, or mapped to [],
# contributes no rows -- there is no separate on/off switch to keep in sync.

# NOTE: Use the following script to remap the ShapeNet indices:
# qualitative_shapenet_index_remap.py

# MAIN left - 8 rows
SAMPLES_MAP = {
    "icons": [9, 18, 22, 25], 
    "faces": [2, 4],
    # (SHAPENET weights)
    "shapenet": [29, 33],
    # (ICONS weights)
    # # "shapenet": [29, 33],
    # "airplanes": [],
    # "cars": [757, 637],
    # "watercrafts": []
}
OUT_NAME = "qualitative_comparison_left"

# MAIN right - 8 rows
# SAMPLES_MAP = {
#     "icons": [30, 33, 34, 40], 
#     "faces": [6, 14], 
#     # (SHAPENET weights)
#     "shapenet": [46],
#     "airplanes": [60],
#     # (ICONS weights)
#     # # "shapenet": [8],
#     # "airplanes": [60],
#     # "cars": [],
#     # "watercrafts": [328],
# }
# OUT_NAME = "qualitative_comparison_right"

# APPENDIX left - 12 rows
# SAMPLES_MAP = {
#     "icons": [102, 104, 106, 108, 109, 110, 114, 116], 
#     "faces": [102, 106],
#     # (SHAPENET weights)
#     "shapenet": [100, 138],
#     # (ICONS weights)
#     # # "shapenet": [100, 138],
#     # "airplanes": [],
#     # "cars": [605],
#     # "watercrafts": [951]
# }
# OUT_NAME = "qualitative_comparison_appendix_left"

# APPENDIX right - 12 rows
# SAMPLES_MAP = {
#     "icons": [201, 202, 204, 205, 206, 207, 213, 217], 
#     "faces": [204, 210],
#     # (SHAPENET weights)
#     "shapenet": [200, 74],
#     # (ICONS weights)
#     # # "shapenet": [200, 242],
#     # "airplanes": [],
#     # "cars": [236],
#     # "watercrafts": [780]
# }
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


def resolve_maps(dirs_override, samples_override):
    """(dirs, samples) after the CLI overrides, with the two key sets cross-checked.

    --dirs MERGES, so one new dataset can be pointed at without restating the others.
    --samples REPLACES, because the selection is what varies per figure and merging would
    silently keep blocks the caller meant to drop.
    """
    dirs = dict(DIR_MAP)
    if dirs_override:
        dirs.update(json.loads(dirs_override))
    samples = json.loads(samples_override) if samples_override else dict(SAMPLES_MAP)

    unknown = [k for k in samples if k not in dirs]
    if unknown:
        raise KeyError(f"SAMPLES_MAP names {unknown} have no DIR_MAP entry "
                       f"(known: {sorted(dirs)})")
    return dirs, samples


def resolve_rows(dataset, stems, indices):
    """Turn a flat index list into (dataset, stem, index) rows; validate range."""
    bad = [i for i in indices if not (0 <= i < len(stems))]
    if bad:
        raise IndexError(f"{dataset} sample indices {bad} out of range (0..{len(stems) - 1})")
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
    ap.add_argument("--output", default=OUT_DIR, help="Folder to write the panel into.")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--dirs", default=None,
                    help="JSON object {name: path} merged into DIR_MAP; new names are added.")
    ap.add_argument("--samples", default=None,
                    help="JSON object {name: [indices]} replacing SAMPLES_MAP; "
                         "every name must exist in DIR_MAP.")
    ap.add_argument("--no-headers", action="store_true", help="Hide the column labels.")
    return ap.parse_args()


def main():
    args = parse_args()
    dirs, samples = resolve_maps(args.dirs, args.samples)

    # Iterating DIR_MAP (not SAMPLES_MAP) fixes the stacking order: which dataset sits above
    # which is a property of the panel, not of the order the selections happen to be typed.
    # Only datasets that contribute rows are touched, so an unused DIR_MAP entry costs
    # nothing and need not even exist on disk.
    rows, used, col_dirs = [], [], {}
    for name in dirs:
        idx = samples.get(name) or []
        if not idx:
            continue
        rows += resolve_rows(name, list_stems(dirs[name]), idx)
        col_dirs[name] = resolve_col_dirs(dirs[name])
        used.append(f"{len(idx)} {name}")
    if not rows:
        raise ValueError("SAMPLES_MAP selects no rows; nothing to show.")

    n_rows, n_cols = len(rows), len(COLUMNS)
    show_headers = SHOW_HEADERS and not args.no_headers
    print(f"comparison panel: {n_rows} rows x {n_cols} cols ({' + '.join(used)})")

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
