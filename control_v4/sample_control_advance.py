"""Generate stipple point sets using Dynamic ControlNet V4 (Truncated Control)."""

import argparse
import ast
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from control_v4.conditioning import build_condition_tensors_from_image
from control_v4.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_v4.smart_init import (
    add_noise_at_t,
    generate_smart_init_points_from_density,
    render_smart_init_grid,
    save_smart_init_debug,
    smart_init_points_to_offsets,
)
from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from utils.stippling_metrics import (
    compute_grid_capacity,
    compute_spacing_quality,
    geometric_validation_score,
    resolve_capacity_grid_size,
)
from utils.stippling_metrics_advance import (
    compute_all_advanced_metrics,
    visualize_adaptive_sampling_density_map,
    _render_advanced_metrics_row,
    _ADV_ROW_HEIGHT_RATIO,
    _ADV_ROW_EXTRA_HEIGHT,
    plot_visual_m1_cvt_vectors,
    plot_visual_m2_capacity_constraint,
    plot_visual_m5_spatial_measure,
)


# Editable defaults 
CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT = "config/GBN/model.ckpt"

# CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep1900.pt"
CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"

INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/sample_with_GT_WVS/source/emoji-one_4_monkey.png"
COMPARE_IMAGE_LIST = [
    {"WVS": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/sample_with_GT_WVS/target/emoji-one_4_monkey.png"},
    {"GBN": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/sample_with_GT_GBN/target/emoji-one_4_monkey.png"},
    {"ControlNet": 0},
]
CLIP_TO_DOMAIN = True
MC_APPROX = True

# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/sample_with_GT_GBN/original/gradient0deg.png"
# COMPARE_IMAGE_LIST = [
#     {"WVS": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/sample_with_GT_WVS/target/gradient0deg.png"},
#     {"GBN": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/sample_with_GT_GBN/target/gradient0deg.png"},
#     {"ControlNet": 0},
# ]
# CLIP_TO_DOMAIN = False
# MC_APPROX = True

# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic_V2/source/quadratic_density_gradient.png"
# COMPARE_IMAGE_LIST = [
#     {"WVS": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic_V2/target_WVS_1024/quadratic_density_gradient.png"},
#     {"GBN": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic_V2/target_GBN_1024/quadratic_density_gradient.png"},
#     {"ControlNet": "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic_V2/target_CN_1024/quadratic_density_gradient.png"},
# ]
# CLIP_TO_DOMAIN = False
# MC_APPROX = True

# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic/source/quadratic_density_gradient.png"
# COMPARE_IMAGE_LIST = [
#     {"ControlNet": 0},
# ]
# CLIP_TO_DOMAIN = False

OUTPUT_DIR = "control_v4/sample_outputs_advance"
N_SAMPLES = 1
TIMESTEPS = 1000
GRID_SIZE = 32
RESAMPLE_JUMPS = 2
ENABLE_GECCO = True
DEVICE = "cuda"
SDF_TRUNCATE_PX = 8.0
SDF_FEATURES = False
SMART_INIT_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False
SHOW_DENOISING = False
SHOW_DENOISING_INTERVAL = 50
INFER_TRUNCATION_RATIO = 0.30
T_START_STEP = -1
SMART_INIT_SEED = 42
SMART_INIT_SPLAT_SIGMA_PX = 0.5
SHOW_COLORBAR = True
CAPACITY_GRID_SIZE = 32
# CAPACITY_GRID_SIZE = -1  # -1 for full input resolution
CAPACITY_IGNORE_WHITE = True   # exclude near-white background cells from capacity scoring
METRICS_ADVANCE = True
ADAPTIVE_SAMPLING_DENSITY_MAP = True
# CLIP_TO_DOMAIN = True  # Whether to clip predicted points to [0,1]² before metrics and visualisation (recommended for truncated control)


def _parse_compare_image_list(raw_text, arg_name):
    """Parse a Python list literal of single-item dicts into ordered compare entries."""
    try:
        parsed = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"{arg_name} must be a Python list literal, got: {raw_text}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"{arg_name} must be a list, got: {type(parsed).__name__}")

    entries = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"{arg_name}[{idx}] must be a dict, got: {type(item).__name__}")
        if len(item) != 1:
            raise ValueError(f"{arg_name}[{idx}] must contain exactly one key/value pair")
        label, value = next(iter(item.items()))
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{arg_name}[{idx}] key must be a non-empty string")
        if not (isinstance(value, str) or isinstance(value, int)):
            raise ValueError(
                f"{arg_name}[{idx}] value must be a string image path or an integer sample index"
            )
        entries.append((label, value))
    return entries


def _load_grayscale_image_u8(image_path):
    img_u8 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img_u8


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
    """Create a comparison panel where each entry is either an image path or a predicted sample index."""
    if not HAS_MPL:
        return None
    if len(compare_entries) == 0:
        return None

    n_cols = 1 + len(compare_entries)

    # Metric functions and titles (we want M1, M2, M5 rows beneath the top points row)
    plot_fns = [plot_visual_m1_cvt_vectors, plot_visual_m2_capacity_constraint, plot_visual_m5_spatial_measure]
    row_titles = ["[M1] Spatial Relaxation (CVT Vectors)", "[M2] Capacity Constraint (Voronoi Mass Deviation)", "[M5] Spatial Measure (ρ) Hotspots"]
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def extract_points_from_target(img_path, n_points):
    """Detect dot centroids in a stippled target and return (N, 2) in [0, 1]."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    inv = 255 - img_np
    binary = (inv > 127).astype(np.uint8)

    from scipy import ndimage
    labelled, n_labels = ndimage.label(binary)
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


def visualize_sample_metrics_no_gt(
    source_img_u8,
    pred_pointsets,
    save_path,
    point_size=0.5,
    capacity_grid_size=16,
    compute_advanced=False,
    advanced_metrics=None,
    mc_approx=True,
):
    """Create overfit-style metrics panel without GT column."""
    if not HAS_MPL:
        return None
    if len(pred_pointsets) == 0:
        return None

    n_preds = min(len(pred_pointsets), 4)
    n_cols = 1 + n_preds  # INPUT + predictions

    image_01 = source_img_u8.astype(np.float64) / 255.0

    ax = axes[0, 0]
    ax.imshow(source_img_u8, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Input)")
    ax.axis("off")

    for i in range(n_preds):
        ax = axes[0, 1 + i]
        pts = pred_pointsets[i]
        ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=point_size, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(f"Predict {i}")
        ax.axis("off")

    # Metric functions (M1, M2, M5) to show beneath the top points row
    plot_fns = [plot_visual_m1_cvt_vectors, plot_visual_m2_capacity_constraint, plot_visual_m5_spatial_measure]
    n_metric_rows = len(plot_fns)

    # Total rows = top points row + metric rows only (no advanced table here)
    n_rows = 1 + n_metric_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    metric_row_start = 1

    # Hide left column for metric rows
    for r in range(metric_row_start, metric_row_start + n_metric_rows):
        axes[r, 0].axis("off")

    # Draw per-column M1/M2/M5 visuals
    for col_i in range(n_preds):
        pts = pred_pointsets[col_i]
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
                ax.set_title(f"[Predict {col_i}]  {current}", fontsize=9)

    plt.tight_layout()
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def visualize_sample_metrics_no_gt_legacy(
    source_img_u8,
    pred_pointsets,
    save_path,
    point_size=0.5,
    capacity_grid_size=16,
    compute_advanced=False,
    advanced_metrics=None,
    mc_approx=True,
):
    """Legacy no-GT panel: points row, optional advanced table, then capacity and spacing rows."""
    if not HAS_MPL:
        return None
    if len(pred_pointsets) == 0:
        return None

    n_preds = min(len(pred_pointsets), 4)
    n_cols = 1 + n_preds
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

    for i in range(n_preds):
        ax = axes[0, 1 + i]
        pts = pred_pointsets[i]
        ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=point_size, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(f"Predict {i}")
        ax.axis("off")

    if compute_advanced:
        pred_labels = [f"Predict {i}" for i in range(n_preds)]
        _render_advanced_metrics_row(
            axes[1, :],
            pred_pointsets[:n_preds],
            pred_labels,
            image_01,
            metrics_list=advanced_metrics,
            mc_approx=mc_approx,
        )

    cap_row = 2 if compute_advanced else 1
    spa_row = 3 if compute_advanced else 2

    axes[cap_row, 0].axis("off")
    axes[spa_row, 0].axis("off")

    cap_grid_shape = resolve_capacity_grid_size(image_01, capacity_grid_size)
    pred_caps = [compute_grid_capacity(pred_pointsets[i], image_01, grid_size=cap_grid_shape, ignore_white=CAPACITY_IGNORE_WHITE) for i in range(n_preds)]
    pred_spa = [compute_spacing_quality(pred_pointsets[i]) for i in range(n_preds)]

    for i in range(n_preds):
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
            f"Predict {i} Capacity\n"
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

    for i in range(n_preds):
        spa = pred_spa[i]
        pts = pred_pointsets[i]
        ax = axes[spa_row, 1 + i]
        sc = ax.scatter(pts[:, 0], 1 - pts[:, 1], c=spa["nn_distances"], cmap="RdYlBu", s=point_size * 3, alpha=0.8, vmin=vmin, vmax=vmax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(
            f"Predict {i} Spacing\n"
            f"CV:{spa['nn_cv']:.3f}  Clumped:{spa['clumped_pct']:.1f}%\n"
            f"Score: {spa['spacing_score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")
        plt.colorbar(sc, ax=ax, shrink=0.7, label="NN dist")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_sample_image(image_path, pts, out_png_path):
    cond_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cond_img is None:
        return
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


def _serialize_advanced_metrics(metrics_dict):
    used_keys = {
        "M1_cvt_energy",
        "M2_voronoi_mass_cv",
        "M3_emd_distance",
        "M4_sinkhorn_ot_cost",
        "M5_spatial_measure_rho_mean",
    }
    return {
        key if key in used_keys else f"_{key}": value
        for key, value in metrics_dict.items()
    }


def _save_denoise_step(img_tensor, timestep_i, t_start, out_path):
    if not HAS_MPL:
        return
    offsets = img_tensor[0].detach().cpu().float().numpy()
    h, w = offsets.shape[1], offsets.shape[2]
    cx = (np.arange(w) + 0.5) / w
    cy = (np.arange(h) + 0.5) / h
    gx, gy = np.meshgrid(cx, cy)
    px = np.clip(gx + offsets[0] / w, 0.0, 1.0).flatten()
    py = np.clip(gy + offsets[1] / h, 0.0, 1.0).flatten()

    elapsed = t_start - 1 - timestep_i
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=110)
    ax.scatter(px, 1.0 - py, c="black", s=0.5, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"step {elapsed}/{max(t_start - 1, 1)} (t={timestep_i})", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def _save_condition_debug_tensors(
    high_res,
    high_res_sdf,
    target_density,
    target_sdf,
    smart_init_grid_raw,
    smart_init_grid_model,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)
    cond_map = {
        "high_res": high_res,
        "high_res_sdf": high_res_sdf,
        "target_density": target_density,
        "target_sdf": target_sdf,
        "smart_init_grid_raw": smart_init_grid_raw,
        "smart_init_grid_model_input": smart_init_grid_model,
    }
    for name, tensor in cond_map.items():
        if tensor is None:
            continue
        arr = tensor.detach().cpu().float().numpy().squeeze()
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    if HAS_MPL:
        ordered_names = [
            "high_res",
            "high_res_sdf",
            "target_density",
            "target_sdf",
            "smart_init_grid_raw",
            "smart_init_grid_model_input",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=140)
        for ax, name in zip(axes.flat, ordered_names):
            tensor = cond_map[name]
            if tensor is None:
                ax.axis("off")
                ax.set_title(f"{name} (disabled)")
                continue
            arr = tensor.detach().cpu().float().numpy().squeeze()
            if arr.ndim != 2:
                ax.axis("off")
                ax.set_title(f"{name} (invalid)")
                continue
            if "sdf" in name:
                vis = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
            else:
                vis = np.clip(arr, 0.0, 1.0)
            ax.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            ax.set_title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "conditions_collage.png"), dpi=140, bbox_inches="tight")
        plt.close()


def _grid_centers_flat(grid_size, device, dtype):
    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    return torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)


def _offsets_to_coords_gpu(offsets, grid_size, grid_centers_flat):
    bsz = offsets.shape[0]
    offs = offsets.permute(0, 2, 3, 1).reshape(bsz, grid_size * grid_size, 2)
    coords = grid_centers_flat.expand(bsz, -1, -1) + offs / float(grid_size)
    return coords.clamp(0.0, 1.0)


def _render_smart_init_gpu(coords, grid_size, sigma_px, device):
    """Gaussian soft splatting of (1, N, 2) coords to (1, 1, G, G) -- matches training."""
    lin = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    pixel_centers = torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)
    sigma = max(float(sigma_px), 1e-4) / float(grid_size)
    dist = torch.cdist(pixel_centers, coords, p=2)
    gauss = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
    return gauss.amax(dim=2).reshape(1, 1, grid_size, grid_size).clamp(0.0, 1.0)


def load_condition(image_path, grid_size, device, sdf_features=True, sdf_truncate_px=0.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_01 = img.astype(np.float32) / 255.0
    if not sdf_features:
        high_res = torch.from_numpy(image_01).unsqueeze(0).unsqueeze(0).to(device)
        target_density = torch.nn.functional.interpolate(high_res, size=(grid_size, grid_size), mode="area")
        return image_01, high_res, target_density, None, None
    high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_image(
        image_01,
        grid_size,
        device,
        sdf_truncate_px=sdf_truncate_px,
    )
    return image_01, high_res, target_density, high_res_sdf, target_sdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--base_ckpt", default=BASE_CKPT)
    parser.add_argument("--control_ckpt", default=CONTROL_CKPT)
    parser.add_argument("--input-image", "--image", dest="input_image", default=INPUT_IMAGE_PATH)
    parser.add_argument(
        "--compare-image-list",
        default=str(COMPARE_IMAGE_LIST),
        help="Python list literal of single-item dicts. Each value is either an image path or a predicted sample index.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--no_ot", action="store_true")
    parser.add_argument("--enable-gecco", default=ENABLE_GECCO, action=argparse.BooleanOptionalAction)
    parser.add_argument("--smart-init-features", action=argparse.BooleanOptionalAction, default=SMART_INIT_FEATURES)
    parser.add_argument("--batch-coords-features", action=argparse.BooleanOptionalAction, default=BATCH_COORDS_FEATURES)
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)
    parser.add_argument("--sdf-features", action=argparse.BooleanOptionalAction, default=SDF_FEATURES)
    parser.add_argument("--show-denoising", action=argparse.BooleanOptionalAction, default=SHOW_DENOISING)
    parser.add_argument(
        "--show-denoising-interval",
        "--denoise-interval",
        dest="show_denoising_interval",
        type=int,
        default=SHOW_DENOISING_INTERVAL,
    )
    parser.add_argument("--infer-truncation-ratio", type=float, default=INFER_TRUNCATION_RATIO,
                        help="INFERENCE truncation: SDEdit start level for sampling")
    parser.add_argument("--t-start-step", type=int, default=T_START_STEP,
                        help="If >=0, overrides truncation-ratio derived start step")
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--smart-init-splat-sigma-px", type=float, default=SMART_INIT_SPLAT_SIGMA_PX,
                        help="Gaussian sigma in grid-pixel units for Smart Init soft splatting (match training default)")
    parser.add_argument(
        "--enable-smart-init-splat-sigma",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_SMART_INIT_SPLAT_SIGMA,
        help="Enable Gaussian soft-splat rendering for Smart Init model input grid.",
    )
    parser.add_argument(
        "--capacity-grid-size",
        type=int,
        default=CAPACITY_GRID_SIZE,
        help="Capacity grid size: >0 uses KxK, -1 uses full input image resolution",
    )
    parser.add_argument(
        "--metrics-advance",
        action="store_true",
        default=METRICS_ADVANCE,
        help="Enable advanced M1-M5 metrics (Voronoi, Sinkhorn, EMD, Adaptive-NND, CVT)",
    )
    parser.add_argument(
        "--adaptive-sampling-density-map",
        action=argparse.BooleanOptionalAction,
        default=ADAPTIVE_SAMPLING_DENSITY_MAP,
        help="Enable GBN-style AKDE density map visualisation (saved to adaptive_sampling_density_map/)",
    )
    parser.add_argument(
        "--mc-approx",
        action=argparse.BooleanOptionalAction,
        default=MC_APPROX,
        help="Use Monte Carlo approximation for advanced metrics; disable for deterministic dense-grid evaluation",
    )
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    if not (0.0 < args.infer_truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0,1]")
    if args.capacity_grid_size == 0 or args.capacity_grid_size < -1:
        raise ValueError("--capacity-grid-size must be > 0, or -1 for full input resolution")

    compare_entries = _parse_compare_image_list(args.compare_image_list, "--compare-image-list")

    device = torch.device(args.device)

    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"], strict=False)
    diffusion.to(device)
    denoiser = diffusion.model
    denoiser.eval()

    control_net = DynamicControlNet(
        denoiser,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        batch_coords_features=args.batch_coords_features,
    ).to(device)
    state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.safe_load_state_dict(state, strict=False)
    # If trained with the base unfrozen, restore the trained base denoiser
    # (stored under "denoiser"; None/absent -> keep the original GBN base).
    if isinstance(state, dict) and state.get("denoiser") is not None:
        denoiser.load_state_dict(state["denoiser"], strict=False)
        print("Loaded trained (unfrozen) base denoiser from control checkpoint")
    control_net.eval()

    img_stem = os.path.splitext(os.path.basename(args.input_image))[0]
    sample_base_dir = os.path.join(args.out_dir, img_stem)
    os.makedirs(sample_base_dir, exist_ok=True)

    image_01, high_res, target_density, high_res_sdf, target_sdf = load_condition(
        args.input_image,
        args.grid_size,
        device,
        sdf_features=args.sdf_features,
        sdf_truncate_px=args.sdf_truncate_px,
    )

    smart_points = generate_smart_init_points_from_density(
        image_01,
        n_points=args.grid_size * args.grid_size,
        seed=args.smart_init_seed,
    )
    smart_offsets_np = smart_init_points_to_offsets(smart_points)
    smart_init_offsets = torch.from_numpy(smart_offsets_np).unsqueeze(0).to(device)

    if args.smart_init_features:
        smart_grid_np = render_smart_init_grid(smart_points, grid_size=args.grid_size)
        smart_init_grid_raw = torch.from_numpy(smart_grid_np).unsqueeze(0).to(device)
        if args.enable_smart_init_splat_sigma:
            grid_centers_flat = _grid_centers_flat(args.grid_size, device, smart_init_offsets.dtype)
            smart_coords = _offsets_to_coords_gpu(smart_init_offsets, args.grid_size, grid_centers_flat)
            smart_init_grid = _render_smart_init_gpu(
                smart_coords,
                args.grid_size,
                args.smart_init_splat_sigma_px,
                device,
            )
        else:
            smart_init_grid = smart_init_grid_raw
    else:
        smart_grid_np = None
        smart_init_grid_raw = None
        smart_init_grid = None

    conditions_dir = os.path.join(sample_base_dir, "conditions")
    _save_condition_debug_tensors(
        high_res,
        high_res_sdf,
        target_density,
        target_sdf,
        smart_init_grid_raw,
        smart_init_grid,
        conditions_dir,
    )

    if args.smart_init_features:
        smart_dir = os.path.join(sample_base_dir, "smart_init")
        save_smart_init_debug(
            smart_dir,
            smart_points,
            smart_offsets_np,
            smart_grid_np,
            model_input_grid=smart_init_grid.detach().cpu().numpy(),
        )

    controlled = DynamicControlledDenoiser(denoiser, control_net)
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
    diffusion.model = controlled
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    t_start = args.t_start_step if args.t_start_step >= 0 else int(args.timesteps * args.infer_truncation_ratio)
    t_start = int(np.clip(t_start, 1, max(args.timesteps - 1, 1)))

    n_samples = args.n_samples
    x_init = smart_init_offsets
    if x_init.shape[0] != n_samples:
        x_init = x_init.expand(n_samples, -1, -1, -1).contiguous()

    alpha_t = diffusion.alphas_cumprod[t_start]
    img = add_noise_at_t(x_init, alpha_t)

    print(f"Loaded checkpoint : {args.control_ckpt}")
    print(f"GECCO enabled     : {args.enable_gecco}")
    print(f"Smart Init enabled: {args.smart_init_features}")
    print(f"Smart Init splat-sigma enabled: {args.enable_smart_init_splat_sigma}")
    if args.enable_smart_init_splat_sigma:
        print(f"Smart Init splat sigma (px): {args.smart_init_splat_sigma_px}")
    print(f"Batch coords      : {args.batch_coords_features}")
    print(f"SDF enabled       : {args.sdf_features}")
    print(f"Timesteps         : {args.timesteps}")
    print(f"t_start           : {t_start}")
    print(f"Resample jumps    : {args.resample_jumps}")
    if args.capacity_grid_size == -1:
        print("Capacity grid     : full input resolution")
    else:
        print(f"Capacity grid     : {args.capacity_grid_size}x{args.capacity_grid_size}")

    steps_dir = None
    if args.show_denoising:
        steps_dir = os.path.join(sample_base_dir, "denoising_steps")
        os.makedirs(steps_dir, exist_ok=True)

    from tqdm import tqdm
    with torch.no_grad() if args.resample_jumps == 0 else torch.enable_grad():
        for i in tqdm(reversed(range(t_start)), total=t_start, desc="sampling_v4"):
            t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
            for u in range(args.resample_jumps + 1):
                with torch.no_grad():
                    img = diffusion.p_sample(
                        img,
                        cond=None,
                        t=t_tensor,
                        clip_denoised=diffusion.sample_clip,
                        with_sampling=True,
                    )

                if u == args.resample_jumps or i == 0:
                    break
                beta_i = diffusion.betas[i]
                noise = torch.randn_like(img)
                img = (1.0 - beta_i).sqrt() * img + beta_i.sqrt() * noise

            if steps_dir is not None:
                elapsed = t_start - 1 - i
                if elapsed % args.show_denoising_interval == 0:
                    step_path = os.path.join(steps_dir, f"step_{elapsed:04d}.png")
                    _save_denoise_step(img, i, t_start, step_path)

    samples_raw = img.detach().cpu().numpy()

    npy_dir = os.path.join(sample_base_dir, "npy")
    png_dir = os.path.join(sample_base_dir, "png")
    metrics_dir = os.path.join(sample_base_dir, "metrics")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    pred_pointsets = []

    for idx, s in enumerate(samples_raw):
        suffix = f"_{idx + 1}"
        npy_path = os.path.join(npy_dir, f"{img_stem}{suffix}.npy")
        png_path = os.path.join(png_dir, f"{img_stem}{suffix}.png")

        if not args.no_ot:
            pts = to_pointset_optimal_transport(s)
            pts = pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T
            pred_pointsets.append(pts)
            np.save(npy_path, pts)
            save_sample_image(args.input_image, pts, png_path)
        else:
            np.save(npy_path, s)

    if args.no_ot:
        print("Skipped metrics panel: --no_ot was enabled.")
    else:
        input_img_u8 = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
        panel_path = os.path.join(metrics_dir, "results_panel.png")
        panel_saved = None
        compare_payloads = []
        compare_pointsets = []
        compare_labels = []
        advanced_metrics_list = []
        gt_points = None
        if input_img_u8 is None:
            print(f"Skipped metrics panel: failed to read input image: {args.input_image}")
        elif len(pred_pointsets) == 0:
            print("Skipped metrics panel: no predicted point sets were generated.")
        elif len(compare_entries) > 0:
            n_points = pred_pointsets[0].shape[0]
            for label, value in compare_entries:
                compare_labels.append(label)
                if isinstance(value, int):
                    if value < 0 or value >= len(pred_pointsets):
                        raise IndexError(
                            f"Compare sample index {value} for '{label}' is out of range for {len(pred_pointsets)} predicted samples"
                        )
                    compare_pointsets.append(pred_pointsets[value])
                    compare_payloads.append({
                        "kind": "sample",
                        "label": label,
                        "index": value,
                        "points": pred_pointsets[value],
                    })
                else:
                    compare_img_u8 = _load_grayscale_image_u8(value)
                    compare_pts = extract_points_from_target(value, n_points)
                    compare_pointsets.append(compare_pts)
                    compare_payloads.append({
                        "kind": "image",
                        "label": label,
                        "path": value,
                        "image_u8": compare_img_u8,
                        "points": compare_pts,
                    })

                if args.metrics_advance:
                    advanced_metrics_list.append(
                        compute_all_advanced_metrics(compare_pointsets[-1], image_01, mc_approx=args.mc_approx)
                    )

            panel_saved = visualize_compare_panel_legacy(
                input_img_u8,
                compare_entries,
                compare_payloads,
                panel_path,
                capacity_grid_size=args.capacity_grid_size,
                compute_advanced=args.metrics_advance,
                advanced_metrics=advanced_metrics_list if args.metrics_advance else None,
                mc_approx=args.mc_approx,
            )
        else:
            if args.metrics_advance:
                advanced_metrics_list = [compute_all_advanced_metrics(pts, image_01, mc_approx=args.mc_approx) for pts in pred_pointsets]
            panel_saved = visualize_sample_metrics_no_gt_legacy(
                input_img_u8,
                pred_pointsets,
                panel_path,
                capacity_grid_size=args.capacity_grid_size,
                compute_advanced=args.metrics_advance,
                advanced_metrics=advanced_metrics_list if args.metrics_advance else None,
                mc_approx=args.mc_approx,
            )

        if panel_saved is not None:
            print(f"Saved metrics panel: {panel_saved}")
        else:
            print("Skipped metrics panel: matplotlib is not available.")

        geom = geometric_validation_score(pred_pointsets)
        print(
            "Geometry summary | "
            f"CV={geom['cv']:.4f} | "
            f"Clumped={geom['clumped_pct']:.2f}% | "
            f"Score={geom['score']:.4f}"
        )

        # Advanced metrics M1-M5 (optional) — text row already embedded in panel above
        if args.metrics_advance and not args.no_ot:
            try:
                metrics_advance_dir = os.path.join(sample_base_dir, "metrics_advance")
                os.makedirs(metrics_advance_dir, exist_ok=True)

                if compare_entries:
                    for idx, (label, value) in enumerate(compare_entries):
                        metrics_dict = advanced_metrics_list[idx]
                        source_desc = f"pred[{value}]" if isinstance(value, int) else value
                        metrics_json_path = os.path.join(metrics_advance_dir, f"metrics_{idx + 1}_{label}.json")
                        with open(metrics_json_path, "w") as f:
                            json.dump(_serialize_advanced_metrics(metrics_dict), f, indent=2)

                        print(
                            f"{label} advanced metrics ({source_desc}) | "
                            f"M1_CV={metrics_dict.get('M1_voronoi_mass_cv', 0.0):.4f} | "
                            f"M2_OT={metrics_dict.get('M2_sinkhorn_ot_cost', 0.0):.4f} | "
                            f"M3_EMD={metrics_dict.get('M3_emd_distance', 0.0):.4f} | "
                            f"M4_NND={metrics_dict.get('M4_adaptive_nnd_cv', 0.0):.4f}"
                        )

                if not compare_entries:
                    # Compute and save detailed M1-M5 metrics for each prediction
                    for idx, pts in enumerate(pred_pointsets):
                        metrics_dict = advanced_metrics_list[idx]
                        metrics_json_path = os.path.join(metrics_advance_dir, f"metrics_pred_{idx + 1}.json")

                        with open(metrics_json_path, "w") as f:
                            json.dump(_serialize_advanced_metrics(metrics_dict), f, indent=2)

                        print(f"Pred {idx + 1} advanced metrics | "
                              f"M1_CV={metrics_dict.get('M1_voronoi_mass_cv', 0.0):.4f} | "
                              f"M2_OT={metrics_dict.get('M2_sinkhorn_ot_cost', 0.0):.4f} | "
                              f"M3_EMD={metrics_dict.get('M3_emd_distance', 0.0):.4f} | "
                              f"M4_NND={metrics_dict.get('M4_adaptive_nnd_cv', 0.0):.4f}")
            except Exception as e:
                print(f"Warning: advanced metrics computation failed: {e}")

        # ── AKDE density map (GBN-style) ─────────────────────────────────
        if args.adaptive_sampling_density_map and input_img_u8 is not None and len(pred_pointsets) > 0:
            try:
                akde_dir = os.path.join(sample_base_dir, "adaptive_sampling_density_map")
                os.makedirs(akde_dir, exist_ok=True)
                akde_path = os.path.join(akde_dir, "density_map.png")
                akde_saved = visualize_adaptive_sampling_density_map(
                    input_img_u8,
                    compare_pointsets if compare_entries else pred_pointsets,
                    akde_path,
                    gt_points=None,
                    device=str(device),
                    pred_labels=compare_labels if compare_entries else None,
                )
                if akde_saved:
                    print(f"Saved AKDE density map : {akde_saved}")
            except Exception as e:
                print(f"Warning: AKDE density map visualisation failed: {e}")

        # ── Advanced visual metrics panel (compare-style layout) ──────────
        if args.metrics_advance and input_img_u8 is not None and len(pred_pointsets) > 0:
            try:
                metrics_advance_visual_dir = os.path.join(sample_base_dir, "metrics_advance_visual")
                os.makedirs(metrics_advance_visual_dir, exist_ok=True)
                metrics_advance_visual_path = os.path.join(metrics_advance_visual_dir, "results_panel.png")
                if compare_entries:
                    metrics_advance_visual_saved = visualize_compare_panel(
                        input_img_u8,
                        compare_entries,
                        compare_payloads,
                        metrics_advance_visual_path,
                        capacity_grid_size=args.capacity_grid_size,
                        compute_advanced=args.metrics_advance,
                        advanced_metrics=advanced_metrics_list if args.metrics_advance else None,
                        mc_approx=args.mc_approx,
                    )
                else:
                    metrics_advance_visual_saved = visualize_sample_metrics_no_gt(
                        input_img_u8,
                        pred_pointsets,
                        metrics_advance_visual_path,
                        capacity_grid_size=args.capacity_grid_size,
                        compute_advanced=args.metrics_advance,
                        advanced_metrics=advanced_metrics_list if args.metrics_advance else None,
                        mc_approx=args.mc_approx,
                    )
                if metrics_advance_visual_saved:
                    print(f"Saved metrics_advance_visual panel: {metrics_advance_visual_saved}")
            except Exception as e:
                print(f"Warning: metrics_advance_visual panel failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
