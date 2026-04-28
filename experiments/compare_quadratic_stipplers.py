from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REFERENCE_IMAGE = Path("experiments") / "quadratic_density_gradient_25x100.png"
DEFAULT_OUTPUT = Path("experiments") / "outputs" / "quadratic_comparison" / "comparison_panel.png"

# Edit this list to control the order and contents of the comparison.
RESULT_SPECS = [
    {"label": "Lloyd", "path": Path("path/to/lloyd_points.npy")},
    {"label": "Balzer et al. [2009]", "path": Path("path/to/balzer_points.npy")},
    {"label": "Our Regression Task", "path": Path("path/to/our_points.npy")},
]


def load_reference_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read reference image: {image_path}")
    return image.astype(np.float32) / 255.0


def load_points(points_path):
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


def load_result_sets(result_specs):
    result_sets = []
    for spec in result_specs:
        label = spec["label"]
        path = spec["path"]
        result_sets.append((label, load_points(path)))
    return result_sets


def plot_comparison(reference_image, result_sets, output_path):
    num_methods = len(result_sets)
    fig_height = 2.15 * (num_methods + 1)
    fig, axes = plt.subplots(num_methods + 1, 1, figsize=(9.0, fig_height), dpi=220)

    if num_methods == 1:
        axes = np.array([axes[0], axes[1]])

    ax_ref = axes[0]
    ax_ref.imshow(reference_image, cmap="gray", aspect="auto", extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0)
    ax_ref.set_title("Target Density Function", fontsize=10)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    for spine in ax_ref.spines.values():
        spine.set_linewidth(0.8)

    quarter_positions = [0.125, 0.375, 0.625, 0.875]
    quarter_lines = [0.25, 0.5, 0.75]

    for index, (method_name, points) in enumerate(result_sets, start=1):
        ax = axes[index]
        ax.scatter(points[:, 0], 1.0 - points[:, 1], s=0.5, c="black", marker=".")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

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