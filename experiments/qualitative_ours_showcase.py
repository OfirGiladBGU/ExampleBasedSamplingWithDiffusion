"""qualitative_ours_showcase.py

Our-results showcase: a free grid of [Target, Ours] pairs (like the teaser, but
each cell is a sample shown as Target + our stipple result).

Arrange it with nested lists -- COLUMN-major, like the teaser: one inner list per
COLUMN, its entries the sample indices read TOP-TO-BOTTOM down that column. So

    SAMPLES_MAP = {"icons": [[0, 1, 2], [3, 4, 5]]}

is 2 pair-columns x 3 rows:

    [0] [3]
    [1] [4]
    [2] [5]

Each index expands to the OURS_SUBCOLS cells (default: Target then Ours). A flat list is
treated as a single column. Columns may be ragged (short ones are blank-padded).

Datasets are configured by two maps keyed by the same names, so adding one is a single line
in each rather than a new set of module-level variables:

    DIR_MAP     = {"icons": <path>, "faces": <path>, ...}       where each dataset lives
    SAMPLES_MAP = {"icons": [[0, 1], [2, 3]], ...}              which cells to take from it

Datasets are stacked in DIR_MAP order, so that map alone decides which appears above
which. A name missing from SAMPLES_MAP, or mapped to [], contributes no cells.

Use -1 as a HOLE to leave a dataset out of one column without leaving a blank cell: the
entries below it shift up. In

    SAMPLES_MAP = {
        "icons":    [[158, 151], [163, 160], [161, 155], [162, -1]],
        "faces":    [[-1], [159], [-1], [157]],
        "shapenet": [[42], [-1], [160], [161]],
    }

every column ends up 3 entries long -- e.g. column 0 is icons 158, icons 151, shapenet 42 --
so the panel is 3 rows, not the 4 it would be if each dataset kept its own band. Columns
may still end up ragged; short ones are blank-padded at the bottom.

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

# ── Datasets: where each one lives ────────────────────────────────
# Add a dataset with one entry here and one in SAMPLES_MAP under the same key.
# Blocks are stacked in THIS order (first key = top rows of the panel).
DIR_MAP = {
    "icons": "experiments/outputs/z_validation_data/Icons-50_1024",
    "faces": "experiments/outputs/z_validation_data/CelebA-5K_1024",
    # The combined 3K set, superseded by the per-category sets below. Kept so old index
    # lists stay readable; nothing selects from it any more.
    "shapenet": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K_1600",
    # Fixed stacking order. HOLE (-1) can drop a dataset out of a column, but it cannot
    # reorder datasets within one -- that is always DIR_MAP order.
    "airplanes": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Airplanes_1600",
    "cars": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Cars_1600",
    "watercrafts": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Watercrafts_1600",
}
OUT_DIR = "experiments/outputs/qualitative_showcase"

# ── Grid selection (edit these to find the panel you want) ────────────────
# {dataset name: COLUMN-major lists} -- one inner list per column, entries read
# top-to-bottom. A flat list is a single column. A key absent here, or mapped to [],
# contributes no rows.

# NOTE: Use the following script to remap the ShapeNet indices:
# qualitative_shapenet_index_remap.py

# MAIN - 4 rows
SAMPLES_MAP = {
    "icons": [[158, 151], [163, 160], [161, 155], [162, 157]],
    "faces": [[161], [159], [171], [157]],
    # "shapenet": [[-1], [174], [184], [-1]],
    "airplanes": [[1], [-1], [-1], [12]],
    "cars":        [[-1], [-1], [787], [-1]],
    "watercrafts": [[-1], [380], [-1], [-1]],
}
OUT_NAME = "qualitative_ours"


# APPENDIX - 12 rows
# SAMPLES_MAP = {
#     "icons": [
#         [300, 301, 302, 303, 304, 305, 306, 307],
#         [400, 408, 402, 403, 404, 405, 406, 410],
#         [508, 501, 502, 503, 504, 505, 506, 509],
#         [600, 601, 602, 603, 604, 605, 608, 607],
#         [712, 701, 709, 703, 704, 705, 706, 710],
#     ],
#     "faces": [[300, 301], [319, 303], [304, 305], [314, 307], [318, 309]],
#     # "shapenet": [[250, 260], [252, 274], [261, -1], [276, 93], [225, 266]],
#     "airplanes": [[-1], [-1], [33], [-1], [-1]],
#     "cars":        [[129, 482], [994], [972], [600], [536, 826]],
#     "watercrafts": [[-1], [342], [-1], [524], [-1]],
# }
# OUT_NAME = "qualitative_ours_appendix"

# What each sample cell shows, left to right. Edit to ["Ours"] for result-only.
OURS_SUBCOLS = ["Target", "Ours-WVS", "Ours-GBN"]

# Sentinel index meaning "this dataset contributes nothing to this column". Entries from
# datasets further down DIR_MAP shift up into it, so a hole costs no row.
HOLE = -1

COL_TO_DIR = {
    "Target": "source",
    "Ours-WVS": "target_CN-WVS_*",
    "Ours-GBN": "target_CN-GBN_*",
}

CELL = 2.0             # inches per cell
DOT_SIZE = 2.0         # scatter marker size (pt^2) for the "Ours" cells
SHOW_HEADERS = True    # column labels (Target | Ours-WVS | Ours-GBN) on the top row
HEADER_FONTSIZE = 12   # size of those labels, in points
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


def normalize_cols(valid_samples):
    """Return a SAMPLES_MAP entry as a list of COLUMNS (each a list of indices). Flat -> one column."""
    if not valid_samples:
        return []
    if isinstance(valid_samples[0], (list, tuple)):
        return [[int(i) for i in col] for col in valid_samples]
    return [[int(i) for i in valid_samples]]


def build_grid(dirs, samples):
    """(rows, col_dirs) for the whole panel, merging every dataset column by column.

    Columns are assembled across datasets rather than per dataset: for grid column c, each
    dataset in DIR_MAP order contributes its own column c entries, in order, and any HOLE is
    skipped. Because the skip removes the slot rather than blanking it, a dataset lower in
    DIR_MAP fills the space -- which is the point of HOLE.

    Only datasets that actually contribute are read from disk, so an unused DIR_MAP entry
    need not exist.
    """
    per_ds, n_cols = {}, 0
    for name in dirs:                       # DIR_MAP order == stacking order
        cols = normalize_cols(samples.get(name) or [])
        if not cols:
            continue
        per_ds[name] = cols
        n_cols = max(n_cols, len(cols))
    if not per_ds:
        return [], {}

    stems = {name: list_stems(dirs[name]) for name in per_ds}
    col_dirs = {name: resolve_col_dirs(dirs[name]) for name in per_ds}

    for name, cols in per_ds.items():
        n = len(stems[name])
        for c, col in enumerate(cols):
            bad = [i for i in col if i != HOLE and not (0 <= i < n)]
            if bad:
                raise IndexError(f"{name} column {c} indices {bad} out of range "
                                 f"(0..{n - 1}, or {HOLE} for a hole)")

    # Per column: the samples that will be drawn down it, holes already removed.
    columns = []
    for c in range(n_cols):
        entries = []
        for name, cols in per_ds.items():
            if c < len(cols):
                entries += [(name, stems[name][i]) for i in cols[c] if i != HOLE]
        columns.append(entries)

    n_rows = max(len(e) for e in columns)
    grid = []
    for r in range(n_rows):
        row = []
        for entries in columns:
            if r < len(entries):
                dataset, stem = entries[r]
                row += [(dataset, stem, sub) for sub in OURS_SUBCOLS]
            else:
                row += [None] * len(OURS_SUBCOLS)   # blank pad a short column
        grid.append(row)
    return grid, col_dirs


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
    ap.add_argument("--output", default=OUT_DIR, help="Folder to write the panel into.")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--dirs", default=None,
                    help="JSON object {name: path} merged into DIR_MAP; new names are added.")
    ap.add_argument("--samples", default=None,
                    help="JSON object {name: column-major indices} replacing SAMPLES_MAP; "
                         "every name must exist in DIR_MAP.")
    ap.add_argument("--headers", action="store_true", help="Show Target/Ours labels on the top row.")
    return ap.parse_args()


def main():
    args = parse_args()
    dirs, samples = resolve_maps(args.dirs, args.samples)
    all_rows, col_dirs = build_grid(dirs, samples)
    if not all_rows:
        raise ValueError("SAMPLES_MAP selects no rows; nothing to show.")

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
                ax.set_title(sub, fontsize=HEADER_FONTSIZE)

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
