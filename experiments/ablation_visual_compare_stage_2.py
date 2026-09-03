"""ablation_visual_compare_stage_2.py

Stage 2 of the visual ablation comparison. Reads the shared point .npy produced by
ablation_visual_compare_stage_1.py, selects samples by split_index (VALID_SAMPLES, the same
index space stage 1 used -- indices into the staged manifest), and renders ONE comparison panel:

    rows    = the selected samples
    columns = [ Target condition image ] | [ train-based methods ] | [ inference-based methods ]

with a vertical line separating the column groups (default split 1, 5, 1). Method cells are
VECTOR scatter so the PDF stays crisp at any zoom; the Target cell is the source condition image.

Reads ONLY from the common folder (no model, no eval_dataset):
    OUTPUT_DIR/resources/validation_manifest.json      (staged by stage 0)
    OUTPUT_DIR/resources/source/<filename>             (Target condition image)
    OUTPUT_DIR/<method>/<stem>.npy                     (method prediction points)

Writes:
    OUTPUT_DIR/ablation_visual_panel.pdf   (+ .png)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "experiments/outputs/ablation_visual_results"
MANIFEST_NAME = "validation_manifest.json"

# Column groups, left to right. First group = the Target condition image (special "Target"
# column). Remaining groups = method result columns (each name -> <name>/<stem>.npy). A vertical
# separator line is drawn between consecutive groups. Default split = 1, 5, 1.
GROUPS = [
    ["Target"],
    ["vanilla", "unfrozen", "gecco", "agi", "full"],   # train-based
    ["sdedit"],                                          # inference-based
]
COL_LABELS = {
    "Target": "Target", "vanilla": "Vanilla", "unfrozen": "Unfrozen",
    "gecco": "GECCO", "agi": "Gated", "full": "Full", "sdedit": "SDEdit",
}

# Which samples to show, as indices into the staged manifest. None -> every sample that has
# a prediction in one of the method folders (i.e. whatever stage 1 actually ran).
VALID_SAMPLES = [1, 2, 4, 7]

DOT_SIZE = 1.2       # vector scatter marker size (pt^2)
SEP_RATIO = 0.10     # width of a separator column relative to a content column
CELL = 2.0           # inches per content column / row


def build_column_spec(groups):
    """Flatten groups into (kind, name) columns with 'sep' columns between groups."""
    spec = []
    for gi, g in enumerate(groups):
        for name in g:
            spec.append(("content", name))
        if gi < len(groups) - 1:
            spec.append(("sep", None))
    return spec


def parse_args():
    ap = argparse.ArgumentParser(description="Merge stage-1 NPYs into a grouped vector panel.")
    ap.add_argument("--output", default=OUTPUT_DIR, help="Shared folder from stage 1.")
    ap.add_argument("--valid-samples", default=("null" if VALID_SAMPLES is None else json.dumps(VALID_SAMPLES)),
                    help="JSON list of manifest indices (null/omit = every sample stage 1 ran).")
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--out-name", default="ablation_visual_panel")
    ap.add_argument("--no-headers", action="store_true", help="Hide the column labels row.")
    return ap.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output)
    res = out_base / "resources"
    src_dir = res / "source"
    manifest_path = res / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path} -- run experiments/ablation_visual_compare_stage_0.py first.")

    names = json.loads(manifest_path.read_text())

    sel = json.loads(args.valid_samples)
    if sel is None:
        # No explicit list: show whatever stage 1 actually produced, by matching the .npy
        # stems in the method folders back to their manifest indices.
        stem_to_index = {Path(n).stem: i for i, n in enumerate(names)}
        have = set()
        for group in GROUPS:
            for name in group:
                d = out_base / name
                if d.is_dir():
                    have |= {p.stem for p in d.glob("*.npy")}
        chosen = sorted(stem_to_index[st] for st in have if st in stem_to_index)
        if not chosen:
            raise FileNotFoundError(f"No method .npy predictions found under {out_base}")
    else:
        bad = [i for i in sel if not (0 <= i < len(names))]
        if bad:
            raise IndexError(f"VALID_SAMPLES out of range for {len(names)} manifest entries: {bad}")
        chosen = list(sel)
    samples = [{"split_index": i, "filename": names[i], "stem": Path(names[i]).stem}
               for i in chosen]

    col_spec = build_column_spec(GROUPS)
    width_ratios = [1.0 if kind == "content" else SEP_RATIO for kind, _ in col_spec]
    n_r, n_c = len(samples), len(col_spec)
    print(f"panel: {n_r} rows x {sum(1 for k,_ in col_spec if k=='content')} content cols "
          f"(+{n_c - sum(1 for k,_ in col_spec if k=='content')} separators); samples {chosen}")

    fig_w = CELL * sum(width_ratios)
    fig_h = CELL * n_r
    fig, axes = plt.subplots(n_r, n_c, figsize=(fig_w, fig_h), dpi=140, squeeze=False,
                             gridspec_kw={"width_ratios": width_ratios})

    for r, sample in enumerate(samples):
        for c, (kind, name) in enumerate(col_spec):
            ax = axes[r][c]
            if kind == "sep":
                ax.axvline(0.5, color="black", lw=1.4)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
                continue

            if name == "Target":
                sp = src_dir / sample["filename"]
                if sp.exists():
                    im = plt.imread(str(sp))
                    ax.imshow(im, cmap=("gray" if im.ndim == 2 else None))
                else:
                    ax.text(0.5, 0.5, "no source", ha="center", va="center", fontsize=6, color="red")
                ax.set_aspect("equal")
            else:
                p = out_base / name / f"{sample['stem']}.npy"
                if p.exists():
                    pts = np.load(p).astype(np.float64)
                    ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=args.dot_size, c="black", linewidths=0)
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="red")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.axis("off")
            if r == 0 and not args.no_headers:
                ax.set_title(COL_LABELS.get(name, name), fontsize=11)

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
