"""stress_test_stage_2.py

Stress test, stage 2 of 2: plot the results exported by stress_test_stage_1.py from
OUTPUT_ROOT_DIR/<dataset>/ (no model, no GPU):

    comparison_panel_paper_style.png/.pdf   Target + GT / Baseline / Ours rows, one column per sample
    metrics/<i>_<stem>_metrics.png          one overfit-style metrics panel per sample (CALCULATE_METRICS)

and print the spacing quality (CV / clumped % / spacing score) of every sample.

The source/target image of each sample is read from stress_manifest.json. Results exported before
the split (no manifest) still work: the pairs are re-picked from DATA_ROOT_DIR exactly as stage 1
picks them.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from utils.stippling_metrics import compute_spacing_quality, visualize_overfit_metrics


# Stress 1:
DATA_ROOT_DIR = r"/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1"
OUTPUT_ROOT_DIR = os.path.join("experiments", "outputs", "stress1_results")


# Stress 2 - SKIP:
# DATA_ROOT_DIR = r"/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2"
# OUTPUT_ROOT_DIR = os.path.join("experiments", "outputs", "stress2_results")


# Stress V2:
# DATA_ROOT_DIR = r"/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2"
# OUTPUT_ROOT_DIR = os.path.join("experiments", "outputs", "stress2_V2_results")


# Common settings
CALCULATE_METRICS = True
CAPACITY_GRID_SIZE = 32
MANIFEST_NAME = "stress_manifest.json"
PANEL_NAME = "comparison_panel_paper_style.png"


def load_pairs(out_dir, data_root, n_examples):
    """(source_paths, target_paths, condition_source_kind) of the exported samples.

    From stage 1's manifest when present; otherwise (results exported before the split)
    re-picked from data_root with stage 1's own picking rule, so the pairs are the same ones.
    """
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        return m["source_paths"], m["target_paths"], m["condition_source_kind"]

    print(f"  [note] no {MANIFEST_NAME} in {out_dir}; re-picking the pairs from {data_root}")
    from stress_test_stage_1 import _pick_matched_source_target_examples
    return _pick_matched_source_target_examples(
        os.path.join(data_root, "source"),
        os.path.join(data_root, "original"),
        os.path.join(data_root, "target"),
        n_examples,
    )


def print_metrics(name, points):
    spacing = compute_spacing_quality(points)
    print(
        f"{name:<12} CV={spacing['nn_cv']:.4f} | "
        f"Clumped={spacing['clumped_pct']:.2f}% | "
        f"SpacingScore={spacing['spacing_score']:.4f}"
    )
    return spacing


def save_panel(save_path, condition_image_01, gt_points_batch, baseline_points_batch, control_points_batch):
    if not HAS_MPL:
        print("matplotlib unavailable; skipping panel save")
        return False

    n_cols = len(gt_points_batch) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 8.6), dpi=180)

    axes[0, 0].imshow(condition_image_01, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Target", fontsize=15, fontweight="bold")
    axes[0, 0].axis("off")

    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Base Model", ha="center", va="center", fontsize=20, fontweight="bold")
    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, "Ours", ha="center", va="center", fontsize=20, fontweight="bold")

    for i, gt_points in enumerate(gt_points_batch):
        col = i + 1
        axes[0, col].scatter(gt_points[:, 0], 1.0 - gt_points[:, 1], s=0.5, c="black")
        axes[0, col].set_xlim(0, 1)
        axes[0, col].set_ylim(0, 1)
        axes[0, col].set_aspect("equal")
        axes[0, col].set_title(f"Sample {i+1}", fontsize=15)
        axes[0, col].axis("off")

        baseline_points = baseline_points_batch[i]
        axes[1, col].scatter(baseline_points[:, 0], 1.0 - baseline_points[:, 1], s=0.5, c="black")
        axes[1, col].set_xlim(0, 1)
        axes[1, col].set_ylim(0, 1)
        axes[1, col].set_aspect("equal")
        axes[1, col].axis("off")

        control_points = control_points_batch[i]
        axes[2, col].scatter(control_points[:, 0], 1.0 - control_points[:, 1], s=0.5, c="black")
        axes[2, col].set_xlim(0, 1)
        axes[2, col].set_ylim(0, 1)
        axes[2, col].set_aspect("equal")
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    # Also export a vector PDF of the same panel for high-detail / print quality
    # (the point scatters stay crisp at any zoom, unlike the rasterised PNG).
    plt.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stress test stage 2: plot the results exported by stress_test_stage_1.py."
    )
    parser.add_argument("--data-root", "--dataset-root", dest="data_root", default=DATA_ROOT_DIR)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", default=OUTPUT_ROOT_DIR)
    parser.add_argument(
        "--calculate-metrics",
        action=argparse.BooleanOptionalAction,
        default=CALCULATE_METRICS,
        help="Save one overfit-style metrics panel per sample into output_dir/<dataset>/metrics",
    )
    parser.add_argument(
        "--capacity-grid-size",
        type=int,
        default=CAPACITY_GRID_SIZE,
        help="Capacity grid size: >0 uses KxK, -1 uses full input image resolution",
    )
    args = parser.parse_args()

    if args.capacity_grid_size == 0 or args.capacity_grid_size < -1:
        raise ValueError("--capacity-grid-size must be > 0, or -1 for full input resolution")

    data_stem = os.path.basename(os.path.normpath(args.data_root))
    out_dir = os.path.join(args.output_dir, data_stem)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"No stage 1 results in {out_dir}; run stress_test_stage_1.py first")

    gt_points_batch = list(np.load(os.path.join(out_dir, "gt_points.npy"), allow_pickle=True))
    baseline_points_batch = np.load(os.path.join(out_dir, "baseline_points.npy"))
    control_points_batch = np.load(os.path.join(out_dir, "control_v4_points.npy"))
    n_examples = len(gt_points_batch)

    source_image_paths, target_image_paths, condition_source_kind = load_pairs(out_dir, args.data_root, n_examples)
    if len(target_image_paths) != n_examples:
        raise ValueError(f"{len(target_image_paths)} target paths for {n_examples} exported samples")
    condition_images_u8 = [np.array(Image.open(p).convert("L"), dtype=np.uint8) for p in source_image_paths]

    print(f"Results: {out_dir}")
    print(f"Condition source kind: {condition_source_kind}")
    for i in range(n_examples):
        print_metrics(f"Baseline[{i}]", baseline_points_batch[i])
    for i in range(n_examples):
        print_metrics(f"Control[{i}]", control_points_batch[i])

    panel_path = os.path.join(out_dir, PANEL_NAME)
    saved = save_panel(panel_path, condition_images_u8[0] / 255.0, gt_points_batch,
                       baseline_points_batch, control_points_batch)
    if saved:
        print(f"Saved panel to: {panel_path}")

    if args.calculate_metrics:
        metrics_dir = os.path.join(out_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metric_saved_count = 0

        for i, target_path in enumerate(target_image_paths):
            target_image_u8 = np.array(Image.open(target_path).convert("L"), dtype=np.uint8)
            sample_stem = os.path.splitext(os.path.basename(target_path))[0]
            metrics_path = os.path.join(metrics_dir, f"{i:03d}_{sample_stem}_metrics.png")
            saved_metrics = visualize_overfit_metrics(
                condition_images_u8[i],
                target_image_u8,
                gt_points_batch[i],
                [baseline_points_batch[i], control_points_batch[i]],
                metrics_path,
                step=None,
                gt_offsets=None,
                capacity_grid_size=args.capacity_grid_size,
                pred_labels=["Baseline", "Control V4"],
            )
            if saved_metrics:
                metric_saved_count += 1

        print(f"Saved {metric_saved_count}/{n_examples} metrics panels to: {metrics_dir}")


if __name__ == "__main__":
    main()
