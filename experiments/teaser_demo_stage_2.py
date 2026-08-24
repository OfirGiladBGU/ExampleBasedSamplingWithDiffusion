"""teaser_demo_stage_2.py -- Part 2: assemble the demo teaser figure.

Reads the trajectory snapshots written by stage 1 and lays them out like the
reference figure: the condition image top-left ("Target Density"), a
"Conditioned Denoising Process" arrow spanning the top, and three scatter columns
below titled with their timesteps. Result columns are VECTOR scatter from the .npy
points, so the PDF stays crisp.

Reads:
    OUTPUT_DIR/source/<CONDITION_NAME>   (condition image, top-left)
    OUTPUT_DIR/steps/<label>.npy         (N,2) points per snapshot

Writes:
    OUTPUT_DIR/teaser_demo_panel.pdf (+ .png)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "experiments/outputs/teaser_demo_results"
CONDITION = "source/Demo.png"

# (snapshot .npy label, column title), left to right.
COLUMNS = [
    ("t500", "t = 500 (rejection prior)"),
    ("t750", "t = 750"),
    ("t1000", "t = 1000"),
]

DOT_SIZE = 4.0
DOT_COLOR = "black"       # colours are not required; single colour
TITLE_FS = 20
HEADER_FS = 22
OUT_NAME = "teaser_demo_panel"


def load_points(path):
    return np.load(path).astype(np.float64)


def parse_args():
    ap = argparse.ArgumentParser(description="Assemble the demo denoising-trajectory teaser.")
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--condition", default=CONDITION, help="Condition image path relative to --output.")
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--out-name", default=OUT_NAME)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    steps_dir = out / "steps"

    n = len(COLUMNS)
    fig = plt.figure(figsize=(4.6 * n, 8.0), dpi=150)

    # ── Condition image (top-left), framed ────────────────────────────────────
    ax_cond = fig.add_axes([0.015, 0.66, 0.16, 0.30])
    cond_path = out / args.condition
    if cond_path.exists():
        im = plt.imread(str(cond_path))
        if im.ndim == 2:
            ax_cond.imshow(im, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
        else:
            ax_cond.imshow(im, aspect="equal")
    else:
        ax_cond.text(0.5, 0.5, "no condition", ha="center", va="center", color="red", fontsize=8)
    ax_cond.set_xticks([]); ax_cond.set_yticks([])
    for sp in ax_cond.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(2.5)

    fig.text(0.20, 0.80, "Target\nDensity", fontsize=HEADER_FS, fontweight="bold",
             va="center", ha="left")

    # ── "Conditioned Denoising Process" arrow, spanning over the columns ───────
    ax_hdr = fig.add_axes([0.34, 0.63, 0.64, 0.33]); ax_hdr.axis("off")
    ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1)
    ax_hdr.text(0.5, 0.62, "Conditioned Denoising Process", fontsize=HEADER_FS,
                fontweight="bold", ha="center", va="center")
    ax_hdr.annotate("", xy=(1.0, 0.18), xytext=(0.0, 0.18),
                    arrowprops=dict(arrowstyle="-|>", lw=4.0, color="black",
                                    mutation_scale=32))

    # ── Three scatter columns ─────────────────────────────────────────────────
    left, right, bottom, top = 0.015, 0.985, 0.03, 0.55
    gap = 0.02
    col_w = (right - left - gap * (n - 1)) / n
    for k, (label, title) in enumerate(COLUMNS):
        x0 = left + k * (col_w + gap)
        ax = fig.add_axes([x0, bottom, col_w, top - bottom])
        p = steps_dir / f"{label}.npy"
        if p.exists():
            pts = load_points(p)
            ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=args.dot_size, c=DOT_COLOR, linewidths=0)
        else:
            ax.text(0.5, 0.5, f"missing {label}.npy", ha="center", va="center", color="red", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=10)

    pdf = out / f"{args.out_name}.pdf"
    png = out / f"{args.out_name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {pdf}\n       {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
