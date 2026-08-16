"""ablation_visual_compare_stage_2.py

Stage 2 of the visual ablation comparison. Reads the shared point .npy produced by
ablation_visual_compare_stage_1.py, selects samples by split_index (VALID_SAMPLES, the same
index space stage 1 used, via selection.json), and renders ONE comparison panel:

    rows    = the selected samples
    columns = [ Target condition image ] | [ train-based methods ] | [ inference-based methods ]

with a vertical line separating the column groups (default split 1, 5, 1). Method cells are
VECTOR scatter so the PDF stays crisp at any zoom; the Target cell is the source condition image.

Reads ONLY from the common folder (no model, no eval_dataset):
    OUTPUT_DIR/selection.json
    OUTPUT_DIR/validation_data/source/<filename>       (Target condition image)
    OUTPUT_DIR/<method>/<stem>.npy                      (method prediction points)

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

# Column groups, left to right. First group = the Target condition image (special "Target"
# column). Remaining groups = method result columns (each name -> <name>/<stem>.npy). A vertical
# separator line is drawn between consecutive groups. Default split = 1, 5, 1.
GROUPS = [
    ["Target"],
    ["vanilla", "unfrozen", "gecco", "agi", "full"],   # train-based
    ["sdedit"],                                          # inference-based
]
COL_LABELS = {
    "Target": "Target", "vanilla": "VANILLA", "unfrozen": "UNFROZEN",
    "gecco": "GECCO", "agi": "GATED", "full": "FULL", "sdedit": "SDEdit",
}

# Which samples to show, as split_index values from selection.json. None -> all stage-1 samples.
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
                    help="JSON list of split_index values from selection.json (null/omit = all).")
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--out-name", default="ablation_visual_panel")
    ap.add_argument("--no-headers", action="store_true", help="Hide the column labels row.")
    return ap.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output)
    src_dir = out_base / "validation_data" / "source"

    sel_rows = json.loads((out_base / "selection.json").read_text())["selected_rows"]
    by_index = {int(r["split_index"]): r for r in sel_rows}

    sel = json.loads(args.valid_samples)
    if sel is None:
        chosen = [r["split_index"] for r in sel_rows]           # all, in stage-1 order
    else:
        missing = [i for i in sel if i not in by_index]
        if missing:
            raise KeyError(f"VALID_SAMPLES {missing} not in selection.json "
                           f"(available split_index: {sorted(by_index)})")
        chosen = list(sel)
    samples = [by_index[i] for i in chosen]                     # each: {split_index, filename, stem}

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
