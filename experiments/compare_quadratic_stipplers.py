from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REFERENCE_IMAGE = Path("experiments") / "quadratic" / "source" / "quadratic_density_gradient.png"
DEFAULT_OUTPUT = Path("experiments") / "outputs" / "quadratic_comparison" / "comparison_panel.png"

# Edit this list to control the order and contents of the comparison.
# For raster inputs (png/jpg), you can add an optional "threshold" value per entry.
RESULT_SPECS = [
    {"label": "WVS", "path": Path("/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic/target_WVS_1024/quadratic_density_gradient.png"), "threshold": 127},
    {"label": "GBN", "path": Path("/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic/target_GBN_1024/quadratic_density_gradient.png"), "threshold": 127},
    {"label": "ControlNet", "path": Path("/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/quadratic/target_CN_1024/quadratic_density_gradient.png"), "threshold": 127},
]


def load_reference_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read reference image: {image_path}")
    return image.astype(np.float32) / 255.0


def extract_points_from_image(image_path, threshold=127):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Assume stipple points are dark on a bright background.
    _, binary = cv2.threshold(image, int(threshold), 255, cv2.THRESH_BINARY_INV)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    points = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area <= 0:
            continue
        cx, cy = centroids[label_id]
        points.append((cx, cy))

    if not points:
        return np.zeros((0, 2), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)
    h, w = image.shape
    denom_x = max(float(w - 1), 1.0)
    denom_y = max(float(h - 1), 1.0)
    points[:, 0] /= denom_x
    points[:, 1] /= denom_y
    return points


def load_points(points_path, threshold=127):
    path = Path(points_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        loaded = np.load(path)
        if len(loaded.files) == 1:
            points = loaded[loaded.files[0]]
        elif "points" in loaded.files:
            points = loaded["points"]
        else:
            raise ValueError(f"Ambiguous npz file format for {path}; expected a single array or a 'points' array.")
    elif suffix in {".csv", ".txt"}:
        points = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return extract_points_from_image(path, threshold=threshold)
    else:
        raise ValueError(f"Unsupported point file type: {path.suffix}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected an Nx2 point array in {path}, got shape {points.shape}")
    return points[:, :2]


def calculate_empirical_capacities(points, quarters=4):
    x_coords = np.asarray(points, dtype=np.float32)[:, 0]
    bins = np.linspace(0.0, 1.0, quarters + 1, dtype=np.float32)
    counts, _ = np.histogram(x_coords, bins=bins)
    total_points = max(float(len(x_coords)), 1.0)
    return counts / total_points * 100.0


def calculate_target_capacities(reference_image, quarters=4):
    image = np.asarray(reference_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image for target capacities, got shape {image.shape}")

    # In this setup, darker pixels represent higher target density.
    density = 1.0 - np.clip(image, 0.0, 1.0)
    column_mass = density.mean(axis=0)
    width = column_mass.shape[0]

    x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, quarters + 1, dtype=np.float32)
    counts, _ = np.histogram(x_coords, bins=bins, weights=column_mass)

    total_mass = max(float(column_mass.sum()), 1.0e-12)
    return counts / total_mass * 100.0


def load_result_sets(result_specs):
    result_sets = []
    for spec in result_specs:
        label = spec["label"]
        path = spec["path"]
        threshold = spec.get("threshold", 127)
        result_sets.append((label, load_points(path, threshold=threshold)))
    return result_sets


def plot_comparison(reference_image, result_sets, output_path):
    num_methods = len(result_sets)
    fig_height = 2.15 * (num_methods + 1)
    fig, axes = plt.subplots(num_methods + 1, 1, figsize=(9.0, fig_height), dpi=220)
    ref_h, ref_w = reference_image.shape[:2]
    target_box_aspect = float(ref_h) / max(float(ref_w), 1.0)

    if num_methods == 1:
        axes = np.array([axes[0], axes[1]])

    ax_ref = axes[0]
    ax_ref.imshow(reference_image, cmap="gray", aspect="auto", extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0)
    ax_ref.set_box_aspect(target_box_aspect)
    ax_ref.set_title("Target Density Function", fontsize=10)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    for spine in ax_ref.spines.values():
        spine.set_linewidth(0.8)

    quarter_positions = [0.125, 0.375, 0.625, 0.875]
    quarter_lines = [0.25, 0.5, 0.75]

    target_capacities = calculate_target_capacities(reference_image)
    for xpos, capacity in zip(quarter_positions, target_capacities):
        ax_ref.text(xpos, -0.14, f"{capacity:.1f}%", va="top", ha="center", fontsize=9, transform=ax_ref.transAxes)

    for index, (method_name, points) in enumerate(result_sets, start=1):
        ax = axes[index]
        ax.scatter(points[:, 0], 1.0 - points[:, 1], s=0.5, c="black", marker=".")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")
        ax.set_box_aspect(target_box_aspect)

        for x in quarter_lines:
            ax.axvline(x=x, color="0.75", linestyle="--", linewidth=0.5, zorder=0)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.2")

        ax.text(-0.03, 0.5, method_name, va="center", ha="right", fontsize=10, transform=ax.transAxes)

        capacities = calculate_empirical_capacities(points)
        for xpos, capacity in zip(quarter_positions, capacities):
            ax.text(xpos, -0.14, f"{capacity:.1f}%", va="top", ha="center", fontsize=9, transform=ax.transAxes)

        ax.text(0.5, 1.02, f"n={len(points)}", va="bottom", ha="center", fontsize=8, transform=ax.transAxes, color="0.35")

    plt.subplots_adjust(hspace=0.45, left=0.13, right=0.98, top=0.95, bottom=0.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    reference_image = load_reference_image(DEFAULT_REFERENCE_IMAGE)
    result_sets = load_result_sets(RESULT_SPECS)

    if not result_sets:
        raise ValueError("RESULT_SPECS is empty; add at least one {label, path} entry.")

    for method_name, points in result_sets:
        capacities = calculate_empirical_capacities(points)
        capacities_text = " | ".join(f"Q{i + 1}={value:.1f}%" for i, value in enumerate(capacities))
        print(f"{method_name}: n={len(points)} | {capacities_text}")

    plot_comparison(reference_image, result_sets, DEFAULT_OUTPUT)
    print(f"Saved comparison panel to: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()