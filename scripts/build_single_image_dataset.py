#!/usr/bin/env python
"""Build a dataset by repeating a single image's pointset."""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

from data.Transforms import to_image_optimal_transport
from build_gradient_dataset import extract_points_from_image


def normalize_points(points, target_sizes):
    num_points = points.shape[0]
    closest = min(target_sizes, key=lambda x: abs(x - num_points))
    ratio = closest / float(num_points)

    if ratio < 1.0:
        sample_count = int(ratio * num_points)
        indices = np.random.choice(num_points, sample_count, replace=False)
        points = points[indices]
    elif ratio > 1.0:
        sample_count = int(round(ratio * num_points))
        indices = np.random.choice(num_points, sample_count, replace=True)
        points = points[indices]

    return points.astype(np.float32)


def build_single_image_dataset(image_path, output_file, repeats, threshold, group_name, target_sizes):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    points = extract_points_from_image(str(image_path), threshold=threshold)
    if points.shape[0] == 0:
        raise ValueError("No points found in the image. Check threshold or input image.")

    points = normalize_points(points, target_sizes)
    num_points = points.shape[0]
    grid_size = int(math.sqrt(num_points))

    if grid_size * grid_size != num_points:
        raise ValueError("Point count must be a perfect square for OT transform.")

    data_t = to_image_optimal_transport(points)

    data = np.repeat(points[None, ...], repeats, axis=0)
    data_t = np.repeat(data_t[None, ...], repeats, axis=0)

    # Single-image test is unconditional; prop is unused but required by dataset format.
    prop = np.zeros((repeats, 1), dtype=np.float32)

    output_dir = output_file
    if isinstance(output_dir, str):
        output_dir = Path(output_dir).parent
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as f:
        group = f.create_group(group_name)
        scale_group = group.create_group(f"scale_{num_points}")
        scale_group.create_dataset("data", data=data)
        scale_group.create_dataset("data_t", data=data_t)
        scale_group.create_dataset("prop", data=prop)

    print("✅ Single-image dataset built")
    print(f"  Image: {image_path}")
    print(f"  Output: {output_file}")
    print(f"  Repeats: {repeats}")
    print(f"  Points: {num_points} ({grid_size}x{grid_size})")


def main():
    parser = argparse.ArgumentParser(description="Build a dataset from a single image")
    parser.add_argument("--image", required=True, help="Path to source PNG image")
    parser.add_argument("--output", default="data/datasets/single_image_dataset.hdf5", help="Output HDF5 file")
    parser.add_argument("--repeats", type=int, default=100, help="Number of repeated samples")
    parser.add_argument("--threshold", type=int, default=128, help="Pixel threshold for points")
    parser.add_argument("--group-name", default="single_image", help="HDF5 group name")
    parser.add_argument(
        "--point-sizes",
        nargs="+",
        type=int,
        default=[1024],
        help="Target point sizes (must be perfect squares)",
    )

    args = parser.parse_args()
    build_single_image_dataset(
        args.image,
        args.output,
        args.repeats,
        args.threshold,
        args.group_name,
        args.point_sizes,
    )


if __name__ == "__main__":
    main()
