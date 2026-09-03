"""flow_images_stage_1.py -- Part 1: copy the dataset images for the pipeline figure.

Given a stippling dataset (with source/ and target/) and an image name, copies into
OUTPUT_DIR/<stem>/ the images that DON'T need the model:

    01_source.png                 the input condition image (source/<name>)
    02_target_stipple.png         the GBN/WVS stipple render (target/<stem>.png)
    02_target_stipple.npy         the exact stipple points (target/<stem>.npy)
    02_target_stipple_scatter.png a clean vector scatter of those points (for the figure)

Stage 2 (flow_images_stage_2.py) then runs the model and exports the rejection prior,
the offset quivers, the noised offsets, the (added + predicted) noise and the final result.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOTE: requires access to the original dataset to get:
#  The source image, target images and later the target offsets.
DATA_PATH = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN"
IMAGE_NAME = "emoji-one_4_monkey.png"
OUTPUT_DIR = "experiments/outputs/flow_images"
DOT_SIZE = 4.0


def save_scatter(pts, png_path, dot_size=DOT_SIZE):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=dot_size, c="black", linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def find_source(data, image_name):
    """Find <image_name> anywhere under data/source/ (the dataset is nested by category);
    return (full_path, relative_path_under_source) or (None, None)."""
    src_root = Path(data) / "source"
    matches = sorted(src_root.rglob(image_name))
    if not matches:
        return None, None
    return matches[0], matches[0].relative_to(src_root)


def parse_args():
    ap = argparse.ArgumentParser(description="Copy dataset images for the flow figure.")
    ap.add_argument("--data-path", default=DATA_PATH, help="Dataset root with source/ and target/.")
    ap.add_argument("--image", default=IMAGE_NAME, help="Image file name (in source/).")
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    return ap.parse_args()


def main():
    args = parse_args()
    data = Path(args.data_path)
    stem = Path(args.image).stem
    out = Path(args.output) / stem
    out.mkdir(parents=True, exist_ok=True)

    src_png, rel = find_source(data, args.image)
    if src_png is None:
        print(f"Image '{args.image}' not found under {data / 'source'}"); return 2
    tgt_png = data / "target" / rel.with_suffix(".png")
    tgt_npy = data / "target" / rel.with_suffix(".npy")
    print(f"  found source: {src_png}")

    shutil.copy2(src_png, out / "01_source.png")
    print(f"  01_source.png       <- {src_png}")

    if tgt_png.exists():
        shutil.copy2(tgt_png, out / "02_target_stipple.png")
        print(f"  02_target_stipple.png <- {tgt_png}")
    else:
        print(f"  [warn] no target png: {tgt_png}")

    if tgt_npy.exists():
        shutil.copy2(tgt_npy, out / "02_target_stipple.npy")
        pts = np.load(tgt_npy).astype(np.float64)
        save_scatter(pts, out / "02_target_stipple_scatter.png", args.dot_size)
        print(f"  02_target_stipple.npy <- {tgt_npy}  ({len(pts)} pts) + scatter")
    else:
        print(f"  [warn] no target npy: {tgt_npy}")

    print(f"\nStage 1 done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
