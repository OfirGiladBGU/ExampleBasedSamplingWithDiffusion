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
from tqdm import tqdm

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

# ---------------------------------------------------------------------------
# Layout flags for visualize_compare_panel
# ---------------------------------------------------------------------------
# Per-gap spacing between consecutive content rows, expressed as fractions of
# a content-row height. Index 0 controls the gap between the points row and
# the M1 row, index 1 between M1 and M2, and index 2 between M2 and M5. Tweak
# any individual entry to give a row more or less breathing room (e.g. raise
# ROW_SPACINGS[0] to leave space below the percentages text under the inputs).
ROW_SPACINGS = [0.2, 0.12, 0.2]

# When False, hide the per-axes plot titles in the comparison panel. This
# only affects the row/column header titles drawn on the metric axes; it does
# NOT remove the percentage text rendered below the images, nor any colorbar
# labels or tick labels.
SHOW_LABELS = False

# When True, every metric row that has colorbars displays only ONE colorbar
# placed to the LEFT of column 1 (the first data column), reusing the empty
# column 0 area below the input image. The remaining per-column colorbars in
# that row are hidden.
SINGLE_SCALE = True  # TODO: False case is broken - need to repair in the future

# Fraction of the grid cell width to use for the main plot area in metric rows.
# The remainder (1 - fraction) is reserved as a right-side gutter for colorbars.
# This prevents rows with colorbars from shrinking smaller than rows without.
PLOT_AREA_FRACTION = 0.88

# --- GLOBAL TEXT SIZE VARIABLE ---
TEXT_SIZE = 15


DEFAULT_INPUT_IMAGE = "experiments/outputs/images_results_metrics/quadratic_V2/source/quadratic_density_gradient.png"
DEFAULT_OUT_BASE = "experiments/outputs/images_results_metrics/quadratic_V2/results"
DEFAULT_COMPARE_LIST = [
    {"WVS": "experiments/outputs/images_results_metrics/quadratic_V2/target_WVS_1024/quadratic_density_gradient.npy"},
    {"BNOT": "experiments/outputs/images_results_metrics/quadratic_V2/target_BNOT_1024/quadratic_density_gradient.npy"},
    {"GBN": "experiments/outputs/images_results_metrics/quadratic_V2/target_GBN_1024/quadratic_density_gradient.npy"},
    {"Ours": "experiments/outputs/images_results_metrics/quadratic_V2/target_CN_1024/quadratic_density_gradient.npy"},
]
CLIP_TO_DOMAIN = False
CAPACITY_TEST = True
MARK_BEST = False  # Draws a red box around the capacity percentage closest to ground truth


# DEFAULT_INPUT_IMAGE = "experiments/outputs/images_results_metrics/monkey/source/emoji-one_4_monkey.png"
# DEFAULT_OUT_BASE = "experiments/outputs/images_results_metrics/monkey/results"
# DEFAULT_COMPARE_LIST = [
#     {"WVS": "experiments/outputs/images_results_metrics/monkey/target_WVS_1024/emoji-one_4_monkey.npy"},
#     {"BNOT": "experiments/outputs/images_results_metrics/monkey/target_BNOT_1024/emoji-one_4_monkey.npy"},
#     {"GBN": "experiments/outputs/images_results_metrics/monkey/target_GBN_1024/emoji-one_4_monkey.npy"},
#     {"Ours": "experiments/outputs/images_results_metrics/monkey/target_CN_1024/emoji-one_4_monkey.npy"},
# ]
# CLIP_TO_DOMAIN = True
# CAPACITY_TEST = False
# MARK_BEST = False 


# DEFAULT_INPUT_IMAGE = "experiments/outputs/images_results_metrics/plant2/source/plant2_400x400.png"
# DEFAULT_OUT_BASE = "experiments/outputs/images_results_metrics/plant2/results"
# DEFAULT_COMPARE_LIST = [
#     {"WVS": "experiments/outputs/images_results_metrics/plant2/target_WVS_1024/plant2_400x400.npy"},
#     {"BNOT": "experiments/outputs/images_results_metrics/plant2/target_BNOT_1024/plant2_400x400.npy"},
#     {"GBN": "experiments/outputs/images_results_metrics/plant2/target_GBN_1024/plant2_400x400.npy"},
#     {"Ours": "experiments/outputs/images_results_metrics/plant2/target_CN_1024/plant2_400x400.npy"},
# ]
# CLIP_TO_DOMAIN = True
# CAPACITY_TEST = False
# MARK_BEST = False 


# Styling for CAPACITY_TEST vertical quarter guides in the points row.
CAPACITY_GUIDE_COLOR = "deepskyblue"  # A bright, static light blue
CAPACITY_GUIDE_LINESTYLE = (0, (6, 4)) # Keeps the dashed pattern
CAPACITY_GUIDE_LINEWIDTH = 1.5


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def collect_outputs(input_image_path: str, compare_list, out_base: str, mc_approx: bool = True):
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

    # points-per-image budget (32x32 grid); each method is fit to this count for a fair compare
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
    scores_dir = out_dir / "scores"
    for d in (metrics_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Compute advanced metrics for each prediction
    advanced_metrics_list = []
    print(f"Computing advanced metrics for {len(compare_payloads)} images (this may take a moment)...")
    for payload in tqdm(compare_payloads, desc="Metrics Progress"):
        pts = payload["points"]
        metrics = compute_all_advanced_metrics(pts, src_img_u8.astype(np.float64) / 255.0, mc_approx=mc_approx)
        advanced_metrics_list.append(metrics)

    # Save metrics JSONs
    for label, metrics in zip(labels, advanced_metrics_list):
        j = _serialize_advanced_metrics(metrics)
        (scores_dir / f"{sanitize_name(label)}.json").write_text(json.dumps(j, indent=2))

    # Render visual panels: legacy and new table-free
    visualize_compare_panel_legacy(
        src_img_u8,
        compare_entries,
        compare_payloads,
        str(metrics_dir / "legacy_metrics.png"),
        compute_advanced=True,
        advanced_metrics=advanced_metrics_list,
        mc_approx=mc_approx,
    )
    visualize_compare_panel(
        src_img_u8,
        compare_entries,
        compare_payloads,
        str(metrics_dir / "advance_metrics.png"),
        compute_advanced=True,
        advanced_metrics=advanced_metrics_list,
        mc_approx=mc_approx,
        capacity_test=CAPACITY_TEST,
        mark_best=MARK_BEST,
    )

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
        "M2_v2_power_cell_cap_cv",
        "M2_v3_power_cell_cap_cv_masked",
        "M3_emd_distance",
        "M4_sinkhorn_ot_cost",
        "M5_spatial_measure_rho_mean",
    }
    return {key if key in used_keys else f"_{key}": value for key, value in metrics_dict.items()}


def calculate_empirical_capacities(points, quarters=4):
    x_coords = np.asarray(points, dtype=np.float64)[:, 0]
    bins = np.linspace(0.0, 1.0, quarters + 1, dtype=np.float64)
    counts, _ = np.histogram(x_coords, bins=bins)
    total_points = max(float(len(x_coords)), 1.0)
    return counts / total_points * 100.0


def calculate_target_capacities(reference_image, quarters=4):
    image = np.asarray(reference_image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image for target capacities, got shape {image.shape}")

    density = 1.0 - np.clip(image, 0.0, 1.0)
    column_mass = density.mean(axis=0)
    width = column_mass.shape[0]

    x_coords = np.linspace(0.0, 1.0, width, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, quarters + 1, dtype=np.float64)
    counts, _ = np.histogram(x_coords, bins=bins, weights=column_mass)

    total_mass = max(float(column_mass.sum()), 1.0e-12)
    return counts / total_mass * 100.0


def save_sample_image(image_path, pts, out_png_path):
    cond_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cond_img is None:
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


def _detect_centroids_from_png(img_path):
    """PNG fallback: detect dot centroids and return (N, 2) in [0, 1]."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    inv = 255 - img_np
    binary = (inv > 127).astype(np.uint8)

    from scipy import ndimage

    labelled, n_labels = ndimage.label(binary)
    if n_labels == 0:
        ys, xs = np.nonzero(binary)
        centroids = [(x, y) for x, y in zip(xs, ys)]
        centroids = [(c[0], c[1]) for c in centroids]
    else:
        centroids = ndimage.center_of_mass(binary, labelled, range(1, n_labels + 1))

    h, w = img_np.shape
    return np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)


def extract_points_from_target(path, n_points):
    """Return (N, 2) target points in [0, 1], read according to the PATH's extension.

    A ``.npy`` path loads exact coordinates directly; any other (image) path detects
    dot centroids from the PNG. Both share the [x/w, y/h], y-down convention, and the
    result is fit to n_points so every method shares one budget.
    """
    if os.path.splitext(str(path))[1].lower() == ".npy":
        points = np.load(path).astype(np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"expected (N, 2) points in {path}, got {points.shape}")
    else:
        points = _detect_centroids_from_png(path)

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
    capacity_test=False,
    mark_best=True,
):
    if not HAS_MPL:
        return None
    if len(compare_entries) == 0:
        return None

    # Apply global text size parameters
    plt.rcParams.update({
        'font.size': TEXT_SIZE,
        'axes.titlesize': TEXT_SIZE,
        'axes.labelsize': TEXT_SIZE,
        'xtick.labelsize': TEXT_SIZE,
        'ytick.labelsize': TEXT_SIZE
    })

    import matplotlib.gridspec as _gridspec

    n_cols = 1 + len(compare_entries)

    plot_fns = [plot_visual_m1_cvt_vectors, plot_visual_m2_capacity_constraint, plot_visual_m5_spatial_measure]
    n_metric_rows = len(plot_fns)

    n_content_rows = 1 + n_metric_rows
    n_gaps = n_content_rows - 1

    spacings = list(ROW_SPACINGS) if isinstance(ROW_SPACINGS, (list, tuple)) else [ROW_SPACINGS] * n_gaps
    if len(spacings) < n_gaps:
        fill = spacings[-1] if spacings else 0.30
        spacings = spacings + [fill] * (n_gaps - len(spacings))
    spacings = spacings[:n_gaps]
    spacings = [max(0.0, float(s)) for s in spacings]

    height_ratios = []
    for r in range(n_content_rows):
        height_ratios.append(1.0)
        if r < n_gaps:
            height_ratios.append(spacings[r])
    total_height_units = sum(height_ratios)

    fig_w = 4.5 * n_cols
    fig_h = 4.5 * total_height_units
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = _gridspec.GridSpec(
        nrows=len(height_ratios),
        ncols=n_cols,
        figure=fig,
        height_ratios=height_ratios,
        wspace=0.05,
        hspace=0.0,
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
    )

    def _gs_row(content_row_idx):
        return content_row_idx * 2

    axes = np.empty((n_content_rows, n_cols), dtype=object)
    for r in range(n_content_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[_gs_row(r), c])

    image_01 = source_img_u8.astype(np.float64) / 255.0

    # ----- Row 0: condition + per-prediction scatter -----
    ax = axes[0, 0]
    ax.imshow(source_img_u8, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Input)", fontsize=TEXT_SIZE)
    ax.axis("off")

    quarter_positions = [0.125, 0.375, 0.625, 0.875]
    quarter_lines = [0.25, 0.5, 0.75]
    
    target_caps = None
    if capacity_test:
        try:
            target_caps = calculate_target_capacities(image_01)
            for x in quarter_lines:
                ax.plot(
                    [x, x],
                    [0, 1],
                    transform=ax.transAxes,
                    color=CAPACITY_GUIDE_COLOR,
                    linestyle=CAPACITY_GUIDE_LINESTYLE,
                    linewidth=CAPACITY_GUIDE_LINEWIDTH,
                    zorder=0,
                )
            for xpos, capacity in zip(quarter_positions, target_caps):
                ax.text(xpos, -0.06, f"{capacity:.1f}%", va="top", ha="center", fontsize=TEXT_SIZE, transform=ax.transAxes)
        except Exception:
            pass

    compare_points = []
    compare_labels = []

    # Pre-calculate empirical capacities to find the best (closest to target) matches per quarter
    all_emps = []
    best_diffs = None
    if capacity_test:
        if target_caps is not None and mark_best:
            best_diffs = [float('inf')] * len(target_caps)
            
        for payload in compare_payloads:
            try:
                emp = calculate_empirical_capacities(payload.get("points"))
                all_emps.append(emp)
                
                # Check for closest match across all items
                if target_caps is not None and mark_best and len(emp) == len(target_caps):
                    for q in range(len(target_caps)):
                        # FIX: Round to 1 decimal place to fix floating-point tie-breaker bugs
                        diff = round(abs(round(emp[q], 1) - round(target_caps[q], 1)), 1)
                        if diff < best_diffs[q]:
                            best_diffs[q] = diff
            except Exception:
                all_emps.append(None)


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
        ax.set_title(f"{label}", fontsize=TEXT_SIZE)
        ax.axis("off")

        if capacity_test:
            for x in quarter_lines:
                ax.plot(
                    [x, x],
                    [0, 1],
                    transform=ax.transAxes,
                    color=CAPACITY_GUIDE_COLOR,
                    linestyle=CAPACITY_GUIDE_LINESTYLE,
                    linewidth=CAPACITY_GUIDE_LINEWIDTH,
                    zorder=0,
                )
            try:
                emp = all_emps[i]
                if emp is not None:
                    for q, (xpos, capacity) in enumerate(zip(quarter_positions, emp)):
                        is_best = False
                        
                        # Determine if this capacity is the closest to the target capacity
                        if target_caps is not None and mark_best and best_diffs is not None and len(emp) == len(target_caps):
                            # FIX: Evaluate the tie exactly as it will be displayed on screen
                            diff = round(abs(round(capacity, 1) - round(target_caps[q], 1)), 1)
                            if diff <= best_diffs[q] + 1e-5:
                                is_best = True
                        
                        # Apply red bounding box if it's the closest prediction
                        bbox_kwargs = None
                        if is_best:
                            bbox_kwargs = dict(boxstyle="square,pad=0.2", edgecolor="red", facecolor="none", linewidth=1.5)
                            
                        ax.text(
                            xpos, -0.06, f"{capacity:.1f}%", 
                            va="top", ha="center", fontsize=TEXT_SIZE, 
                            transform=ax.transAxes,
                            bbox=bbox_kwargs
                        )
            except Exception:
                pass

    # ----- Metric rows (M1, M2, M5) -----
    metric_row_start = 1

    # Export first-row (condition + per-prediction) images individually when
    # capacity_test is enabled. Files are written to a subfolder named "png"
    # next to the main save_path so they include the dashed guide lines and
    # the percentage text beneath the images for external inspection/diffing.
    # If SHOW_LABELS is False, titles are hidden in the exported PNGs as well.
    if capacity_test:
        try:
            png_dir = Path(save_path).resolve().parent.parent / "png"
            png_dir.mkdir(parents=True, exist_ok=True)
            pdf_dir = Path(save_path).resolve().parent.parent / "pdf"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            # Ensure renderer has drawn text so get_tightbbox works
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # Save original title visibility state if SHOW_LABELS is False
            saved_titles = {}
            if not SHOW_LABELS:
                for col_idx in range(n_cols):
                    ax = axes[0, col_idx]
                    saved_titles[col_idx] = ax.get_title()
                    ax.set_title("")

            # Condition image (column 0)
            ax0 = axes[0, 0]
            bbox = ax0.get_tightbbox(renderer)
            bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
            outp = png_dir / "condition.png"
            fig.savefig(outp, bbox_inches=bbox_inches, pad_inches=0.02)
            fig.savefig(pdf_dir / "condition.pdf", bbox_inches=bbox_inches, pad_inches=0.02)

            # Per-comparison columns
            for i, (label, _) in enumerate(compare_entries):
                axc = axes[0, 1 + i]
                bbox = axc.get_tightbbox(renderer)
                bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
                outp = png_dir / f"{sanitize_name(label)}.png"
                fig.savefig(outp, bbox_inches=bbox_inches, pad_inches=0.02)
                fig.savefig(pdf_dir / f"{sanitize_name(label)}.pdf", bbox_inches=bbox_inches, pad_inches=0.02)

            # Restore titles if they were hidden
            if not SHOW_LABELS:
                for col_idx, title_text in saved_titles.items():
                    axes[0, col_idx].set_title(title_text)
        except Exception as exc:
            print(f"Warning: failed exporting first-row PNGs: {exc}")

    for r in range(metric_row_start, metric_row_start + n_metric_rows):
        axes[r, 0].axis("off")

    header_title_axes = [axes[0, c] for c in range(n_cols)]

    cbars_per_metric_row = [[] for _ in range(n_metric_rows)]
    main_axes_set = set(id(a) for row in axes for a in row)

    for r_idx, fn in enumerate(plot_fns):
        before_ids = set(id(a) for a in fig.axes)
        for col_i, pts in enumerate(compare_points):
            ax = axes[metric_row_start + r_idx, 1 + col_i]
            try:
                if fn is plot_visual_m2_capacity_constraint:
                    fn(pts, image_01, ax, clip_to_domain=CLIP_TO_DOMAIN, show_colorbar=SHOW_COLORBAR, rng=None, mc_approx=mc_approx)
                else:
                    fn(pts, image_01, ax)
            except Exception:
                ax.axis("off")
                ax.set_title("(error)", fontsize=TEXT_SIZE)
            header_title_axes.append(ax)

            current = ax.get_title()
            ax.set_title(f"{current}", fontsize=TEXT_SIZE)

        after_ids = set(id(a) for a in fig.axes)
        new_axes = [a for a in fig.axes if id(a) in (after_ids - before_ids) and id(a) not in main_axes_set]
        cbars_per_metric_row[r_idx] = new_axes

    # ----- Post-Processing: Uniform Plot Widths & Single Scale -----
    for r_idx, cbar_axes in enumerate(cbars_per_metric_row):
        
        # FIX 1: Grab the perfect reference size from the M1 row (which didn't shrink)
        ref_ax = axes[metric_row_start, 1]
        ref_bbox = ref_ax.get_position()
        
        # Standardize plot widths for ALL metric rows (M1, M2, M5)
        for col_i in range(1, n_cols):
            main_ax = axes[metric_row_start + r_idx, col_i]
            
            # Completely sever any invisible colorbar constraints
            main_ax.set_axes_locator(None)
            
            try:
                cell_bbox = main_ax.get_subplotspec().get_position(fig)
                width_frac = 1.0 if SINGLE_SCALE else PLOT_AREA_FRACTION
                
                # FIX 2: Force every plot to exactly match M1's width and height!
                # We keep cell_bbox.y0 so it stays in its correct row.
                main_ax.set_position([
                    cell_bbox.x0, 
                    cell_bbox.y0, 
                    ref_bbox.width * width_frac, 
                    ref_bbox.height
                ])
            except Exception:
                pass

        if SINGLE_SCALE:
            if len(cbar_axes) > 0:
                keep = cbar_axes[0]
                
                # FIX 3: Physically remove the extra colorbars instead of just hiding them,
                # ensuring they don't take up "ghost" space in the layout.
                for extra in cbar_axes[1:]:
                    extra.remove()
                
                # Move ticks and labels to the left side
                keep.yaxis.set_ticks_position('left')
                keep.yaxis.set_label_position('left')
                
                # Reposition the kept colorbar into column-0 area
                try:
                    target_main = axes[metric_row_start + r_idx, 1]
                    col0_main = axes[metric_row_start + r_idx, 0]
                    tgt_pos = target_main.get_position()
                    col0_pos = col0_main.get_subplotspec().get_position(fig)
                    
                    cbar_width = min(0.025, col0_pos.width * 0.10)
                    cbar_height = tgt_pos.height * 0.70
                    cbar_y = tgt_pos.y0 + (tgt_pos.height - cbar_height) / 2.0
                    cbar_x = col0_pos.x1 - cbar_width - 0.012
                    
                    keep.set_position([cbar_x, cbar_y, cbar_width, cbar_height])
                except Exception:
                    pass
        else:
            # Not SINGLE_SCALE: tuck each colorbar into its respective right-side gutter
            for i, cb_ax in enumerate(cbar_axes):
                try:
                    col_i = i + 1
                    main_ax = axes[metric_row_start + r_idx, col_i]
                    cell_bbox = main_ax.get_subplotspec().get_position(fig)
                    
                    gutter_x = cell_bbox.x0 + cell_bbox.width * PLOT_AREA_FRACTION + 0.005
                    gutter_w = cell_bbox.width * (1.0 - PLOT_AREA_FRACTION)
                    
                    cb_width = min(0.025, gutter_w * 0.5)
                    cb_height = cell_bbox.height * 0.75
                    cb_y = cell_bbox.y0 + (cell_bbox.height - cb_height) / 2.0
                    
                    cb_ax.set_position([gutter_x, cb_y, cb_width, cb_height])
                except Exception:
                    pass

    # ----- SHOW_LABELS: hide axis titles only -----
    if not SHOW_LABELS:
        seen = set()
        for ax in header_title_axes:
            if id(ax) in seen:
                continue
            seen.add(id(ax))
            try:
                ax.set_title("")
            except Exception:
                pass

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")  # vector, high-quality
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
    if not HAS_MPL:
        return None
    if len(compare_entries) == 0:
        return None

    plt.rcParams.update({
        'font.size': TEXT_SIZE,
        'axes.titlesize': TEXT_SIZE,
        'axes.labelsize': TEXT_SIZE,
        'xtick.labelsize': TEXT_SIZE,
        'ytick.labelsize': TEXT_SIZE
    })

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
    ax.set_title("Condition (Input)", fontsize=TEXT_SIZE)
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
        ax.set_title(f"{label}", fontsize=TEXT_SIZE)
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
            fontsize=TEXT_SIZE,
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
            fontsize=TEXT_SIZE,
        )
        ax.axis("off")
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("NN dist", fontsize=TEXT_SIZE)
        cbar.ax.tick_params(labelsize=TEXT_SIZE)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")  # vector, high-quality
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
        "--out-base",
        default=DEFAULT_OUT_BASE,
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

    rc = collect_outputs(args.input, compare_list, args.out_base, mc_approx=args.mc_approx)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
