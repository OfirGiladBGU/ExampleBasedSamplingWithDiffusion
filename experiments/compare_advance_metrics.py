"""Export-only helper to collect advanced metrics outputs.

This script does NOT run any model inference. It reads precomputed outputs
from `control_v4/sample_outputs_advance/<input_stem>/` and copies the
relevant files into `experiments/outputs/advance_metrics/` for easier inspection.

Defaults are configured for the `quadratic_V2` example used by the user.
"""
from pathlib import Path
import shutil
import os
import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import plotting/metric helpers from utils
try:
    from utils.stippling_metrics import (
        compute_grid_capacity,
        compute_spacing_quality,
        resolve_capacity_grid_size,
    )
    from utils.stippling_metrics_advance import (
        compute_all_advanced_metrics,
        visualize_adaptive_sampling_density_map,
        _render_advanced_metrics_row,
        _ADV_ROW_FONTSIZE,
        _ADV_ROW_TITLE_FONTSIZE,
        _ADV_ROW_HEIGHT_RATIO,
        _ADV_ROW_EXTRA_HEIGHT,
        plot_visual_m1_cvt_vectors,
        plot_visual_m2_capacity_constraint,
        plot_visual_m5_spatial_measure,
        _format_advanced_text,
    )
    HAS_UTILS = True
    UTILS_IMPORT_ERROR = None
except Exception as exc:
    HAS_UTILS = False
    UTILS_IMPORT_ERROR = exc

# display flag defaults (compatible with sample_control_advance.py)
SHOW_COLORBAR = True
CAPACITY_IGNORE_WHITE = True
HAS_MPL = True
ENABLE_ADAPTIVE_SAMPLING_DENSITY_MAP = False
MC_APPROX = True


# DEFAULT_INPUT_IMAGE = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/monkey/source/emoji-one_4_monkey.png"
# DEFAULT_COMPARE_LIST = [
#     {"WVS": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/monkey/target_WVS_1024/emoji-one_4_monkey.png"},
#     {"GBN": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/monkey/target_GBN_1024/emoji-one_4_monkey.png"},
#     {"ControlNet": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/monkey/target_CN_1024/emoji-one_4_monkey.png"},
# ]
# CLIP_TO_DOMAIN = True


DEFAULT_INPUT_IMAGE = (
    "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic_V2/source/quadratic_density_gradient.png"
)
DEFAULT_COMPARE_LIST = [
    {"WVS": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic_V2/target_WVS_1024/quadratic_density_gradient.png"},
    {"GBN": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic_V2/target_GBN_1024/quadratic_density_gradient.png"},
    {"ControlNet": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic_V2/target_CN_1024/quadratic_density_gradient.png"},
]
CLIP_TO_DOMAIN = False


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def collect_outputs(input_image_path: str, compare_list, src_base: str, out_base: str, mc_approx: bool = True):
    input_path = Path(input_image_path)
    stem = input_path.stem
    # New behavior: rebuild visuals and metrics from the provided prediction images.
    if not HAS_UTILS:
        print("Required utils not available (utils.stippling_metrics[_advance]). Aborting.")
        if UTILS_IMPORT_ERROR is not None:
            print(f"Import error: {UTILS_IMPORT_ERROR}")
        return 2

    # Load source condition image (grayscale u8)
    src_img_u8 = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if src_img_u8 is None:
        print(f"Could not read input image: {input_path}")
        return 3

    # determine n_points from grid_size flag if passed via src_base (legacy), else default to 32x32
    grid_size = 32
    n_points = grid_size * grid_size

    # prepare payloads by extracting points from provided prediction images
    compare_entries = []
    compare_payloads = []
    labels = []
    for item in compare_list:
        if not isinstance(item, dict) or len(item) != 1:
            print(f"Invalid compare entry, expected single-item dict: {item}")
            continue
        label, path = list(item.items())[0]
        labels.append(label)
        compare_entries.append((label, path))
        pts = extract_points_from_target(path, n_points)
        compare_payloads.append({"kind": "image", "points": pts})

    out_base_p = Path(out_base)
    out_dir = out_base_p / stem
    # create requested output subfolders
    metrics_dir = out_dir / "metrics"
    metrics_adv_vis_dir = out_dir / "metrics_advance_visual"
    metrics_adv_json_dir = out_dir / "metrics_advance"
    for d in (metrics_dir, metrics_adv_vis_dir, metrics_adv_json_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Compute advanced metrics for each prediction
    advanced_metrics_list = []
    for payload in compare_payloads:
        pts = payload["points"]
        metrics = compute_all_advanced_metrics(pts, src_img_u8.astype(np.float64) / 255.0, mc_approx=mc_approx)
        advanced_metrics_list.append(metrics)

    # Save metrics JSONs
    for label, metrics in zip(labels, advanced_metrics_list):
        j = _serialize_advanced_metrics(metrics)
        (metrics_adv_json_dir / f"{sanitize_name(label)}.json").write_text(json.dumps(j, indent=2))

    # Render visual panels: legacy and new table-free
    visualize_compare_panel_legacy(src_img_u8, compare_entries, compare_payloads, str(metrics_dir / "results_panel.png"), compute_advanced=True, advanced_metrics=advanced_metrics_list, mc_approx=mc_approx)
    visualize_compare_panel(src_img_u8, compare_entries, compare_payloads, str(metrics_adv_vis_dir / "results_panel.png"), compute_advanced=True, advanced_metrics=advanced_metrics_list, mc_approx=mc_approx)

    # AKDE density map is optional and disabled by default for this script.
    if ENABLE_ADAPTIVE_SAMPLING_DENSITY_MAP:
        density_dir = out_dir / "adaptive_sampling_density_map"
        density_dir.mkdir(parents=True, exist_ok=True)
        pred_pointsets = [p["points"] for p in compare_payloads]
        visualize_adaptive_sampling_density_map(
            src_img_u8,
            pred_pointsets,
            str(density_dir / "density_map.png"),
            pred_labels=labels,
        )

    manifest = {
        "input_image": str(input_path),
        "compare_entries": compare_entries,
        "output_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote outputs to: {out_dir}")
    return 0


def _serialize_advanced_metrics(metrics_dict):
    used_keys = {
        "M1_cvt_energy",
        "M2_voronoi_mass_cv",
        "M3_emd_distance",
        "M4_sinkhorn_ot_cost",
        "M5_spatial_measure_rho_mean",
    }
    return {key if key in used_keys else f"_{key}": value for key, value in metrics_dict.items()}


def save_sample_image(image_path, pts, out_png_path):
    cond_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cond_img is None:
        # If original image not found, create an empty canvas from points extents
        # default to 512x512
        h, w = 512, 512
    else:
        h, w = cond_img.shape

    out_img = np.full((h, w), 255, dtype=np.uint8)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        cv2.imwrite(out_png_path, out_img)
        return

    px = np.rint(pts[:, 0] * (w - 1)).astype(np.int32)
    py = np.rint(pts[:, 1] * (h - 1)).astype(np.int32)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    out_img[py, px] = 0
    cv2.imwrite(out_png_path, out_img)


def extract_points_from_target(img_path, n_points):
    """Detect dot centroids in a stippled target and return (N, 2) in [0, 1]."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    inv = 255 - img_np
    binary = (inv > 127).astype(np.uint8)

    from scipy import ndimage

    labelled, n_labels = ndimage.label(binary)
    if n_labels == 0:
        # fallback: treat non-white pixels as points
        ys, xs = np.nonzero(binary)
        centroids = [(x, y) for x, y in zip(xs, ys)]
        centroids = [(c[0], c[1]) for c in centroids]
    else:
        centroids = ndimage.center_of_mass(binary, labelled, range(1, n_labels + 1))

    h, w = img_np.shape
    points = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)

    rng = np.random.RandomState(42)
    if len(points) > n_points:
        points = points[rng.choice(len(points), n_points, replace=False)]
    elif len(points) < n_points:
        deficit = n_points - len(points)
        points = np.vstack([points, rng.rand(deficit, 2)])

    return points


def visualize_compare_panel(
    source_img_u8,
    compare_entries,
    compare_payloads,
    save_path,
    point_size=0.5,
    capacity_grid_size=16,
    compute_advanced=False,
    advanced_metrics=None,
    mc_approx=True,
):
    """Create a comparison panel where each entry is an image path prediction."""
    if not HAS_MPL:
        return None
    if len(compare_entries) == 0:
        return None

    n_cols = 1 + len(compare_entries)

    # Metric functions and titles (we want M1, M2, M5 rows beneath the top points row)
    plot_fns = [plot_visual_m1_cvt_vectors, plot_visual_m2_capacity_constraint, plot_visual_m5_spatial_measure]
    n_metric_rows = len(plot_fns)

    # Total rows = top points row + metric rows only (no advanced table here)
    n_rows = 1 + n_metric_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))

    if n_cols == 1:
        axes = axes[:, np.newaxis]

    image_01 = source_img_u8.astype(np.float64) / 255.0

    ax = axes[0, 0]
    ax.imshow(source_img_u8, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Input)")
    ax.axis("off")

    compare_points = []
    compare_labels = []

    for i, (label, value) in enumerate(compare_entries):
        ax = axes[0, 1 + i]
        payload = compare_payloads[i]
        compare_labels.append(label)

        pts = payload.get("points")
        compare_points.append(pts)
        ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=point_size, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        if payload["kind"] == "image":
            ax.set_title(f"{label} [image]")
        else:
            ax.set_title(f"{label} [sample {value}]")
        ax.axis("off")

    # Draw metric rows (M1, M2, M5) under each column
    metric_row_start = 1

    # Hide left column for metrics (condition column)
    for r in range(metric_row_start, metric_row_start + n_metric_rows):
        axes[r, 0].axis("off")

    for col_i, pts in enumerate(compare_points):
        for r_idx, fn in enumerate(plot_fns):
            ax = axes[metric_row_start + r_idx, 1 + col_i]
            try:
                if fn is plot_visual_m2_capacity_constraint:
                    fn(pts, image_01, ax, clip_to_domain=CLIP_TO_DOMAIN, show_colorbar=SHOW_COLORBAR, rng=None, mc_approx=mc_approx)
                else:
                    fn(pts, image_01, ax)
            except Exception:
                ax.axis("off")
                ax.set_title("(error)", fontsize=8)
            if r_idx == 0:
                current = ax.get_title()
                ax.set_title(f"[{compare_labels[col_i]}]  {current}", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def visualize_compare_panel_legacy(
    source_img_u8,
    compare_entries,
    compare_payloads,
    save_path,
    point_size=0.5,
    capacity_grid_size=16,
    compute_advanced=False,
    advanced_metrics=None,
    mc_approx=True,
):
    """Legacy compare panel: points row, optional advanced table, then capacity and spacing rows."""
    if not HAS_MPL:
        return None
    if len(compare_entries) == 0:
        return None

    n_cols = 1 + len(compare_entries)
    n_rows = 3 + (1 if compute_advanced else 0)

    if compute_advanced:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5 * n_cols, 4.5 * 3 + _ADV_ROW_EXTRA_HEIGHT),
            gridspec_kw={"height_ratios": [3, _ADV_ROW_HEIGHT_RATIO, 3, 3]},
        )
    else:
        fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 4.5 * 3))

    if n_cols == 1:
        axes = axes[:, np.newaxis]

    image_01 = source_img_u8.astype(np.float64) / 255.0
    ax = axes[0, 0]
    ax.imshow(source_img_u8, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Input)")
    ax.axis("off")

    compare_points = []
    compare_labels = []
    for i, (label, value) in enumerate(compare_entries):
        ax = axes[0, 1 + i]
        payload = compare_payloads[i]
        compare_labels.append(label)
        pts = payload["points"]
        compare_points.append(pts)
        ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=point_size, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(f"{label} [image]" if payload["kind"] == "image" else f"{label} [sample {value}]")
        ax.axis("off")

    if compute_advanced:
        _render_advanced_metrics_row(
            axes[1, :],
            compare_points,
            compare_labels,
            image_01,
            metrics_list=advanced_metrics,
            mc_approx=mc_approx,
        )

    cap_row = 2 if compute_advanced else 1
    spa_row = 3 if compute_advanced else 2

    axes[cap_row, 0].axis("off")
    axes[spa_row, 0].axis("off")

    cap_grid_shape = resolve_capacity_grid_size(image_01, capacity_grid_size)
    pred_caps = [
        compute_grid_capacity(compare_points[i], image_01, grid_size=cap_grid_shape, ignore_white=CAPACITY_IGNORE_WHITE)
        for i in range(len(compare_points))
    ]
    pred_spa = [compute_spacing_quality(compare_points[i]) for i in range(len(compare_points))]

    for i in range(len(compare_points)):
        cap = pred_caps[i]
        ax = axes[cap_row, 1 + i]
        status = cap["grid_status"]
        h_grid, w_grid = status.shape
        rgb = np.zeros((h_grid, w_grid, 3), dtype=np.float32)
        rgb[status == 0, :] = [0.0, 1.0, 0.0]
        rgb[status == -1, :] = [1.0, 0.0, 0.0]
        rgb[status == 1, :] = [0.0, 0.0, 1.0]
        ax.imshow(rgb, origin="upper", aspect="equal")
        ok_pct = 100.0 - cap["underfilled_pct"] - cap["overfilled_pct"]
        ax.set_title(
            f"{compare_labels[i]} Capacity\n"
            f"Grid:{cap_grid_shape[0]}x{cap_grid_shape[1]} | OK:{ok_pct:.0f}% Under:{cap['underfilled_pct']:.0f}% Over:{cap['overfilled_pct']:.0f}%\n"
            f"Score: {cap['score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    all_nn = [s["nn_distances"] for s in pred_spa]
    vmin = min(d.min() for d in all_nn)
    vmax = max(d.max() for d in all_nn)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin, vmax = 0.0, 1.0

    for i in range(len(compare_points)):
        spa = pred_spa[i]
        pts = compare_points[i]
        ax = axes[spa_row, 1 + i]
        sc = ax.scatter(pts[:, 0], 1 - pts[:, 1], c=spa["nn_distances"], cmap="RdYlBu", s=point_size * 3, alpha=0.8, vmin=vmin, vmax=vmax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(
            f"{compare_labels[i]} Spacing\n"
            f"CV:{spa['nn_cv']:.3f}  Clumped:{spa['clumped_pct']:.1f}%\n"
            f"Score: {spa['spacing_score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")
        plt.colorbar(sc, ax=ax, shrink=0.7, label="NN dist")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def parse_args():
    p = argparse.ArgumentParser(description="Collect precomputed advanced-metrics outputs")
    p.add_argument("--input", default=DEFAULT_INPUT_IMAGE, help="INPUT_IMAGE_PATH")
    p.add_argument(
        "--compare-list",
        default=None,
        help=("JSON list literal of compare entries; if omitted, uses built-in example."),
    )
    p.add_argument(
        "--src-base",
        default="control_v4/sample_outputs_advance",
        help="Base folder where sample outputs were written",
    )
    p.add_argument(
        "--out-base",
        default="experiments/outputs/advance_metrics",
        help="Destination base folder to collect outputs into",
    )
    p.add_argument(
        "--adaptive-sampling-density-map",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_ADAPTIVE_SAMPLING_DENSITY_MAP,
        help="Enable optional AKDE density map export to adaptive_sampling_density_map/",
    )
    p.add_argument(
        "--mc-approx",
        action=argparse.BooleanOptionalAction,
        default=MC_APPROX,
        help="Use Monte Carlo approximation for advanced metrics and M2 visualisation.",
    )
    return p.parse_args()


def main():
    global ENABLE_ADAPTIVE_SAMPLING_DENSITY_MAP
    args = parse_args()
    ENABLE_ADAPTIVE_SAMPLING_DENSITY_MAP = args.adaptive_sampling_density_map
    if args.compare_list is None:
        compare_list = DEFAULT_COMPARE_LIST
    else:
        try:
            compare_list = json.loads(args.compare_list)
        except Exception:
            print("--compare-list must be valid JSON list literal. Example: '[{\"WVS\":\"/path/to.png\"}]'")
            return 2

    rc = collect_outputs(args.input, compare_list, args.src_base, args.out_base, mc_approx=args.mc_approx)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
