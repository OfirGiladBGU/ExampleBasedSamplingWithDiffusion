"""Quantitative metrics stage 1: compute M1-M5 metrics for stipple results.

For each target image (the stipple-dot render in TARGET_DIR) this script:
    * extracts the stipple POINTS from the dot image, and
    * loads the matching grayscale CONDITION image from SOURCE_DIR as the
      target density rho.

All metrics M1-M5 are then computed comparing the point placement against the
GRAYSCALE source density (not against the dot image itself). Results are
written as one JSON per image to a parallel folder with a "_json" suffix.

Usage:
    python quantitative_advance_metrics_stage_1.py \\
        /path/to/target_WVS_1024 --source /path/to/source

Or with all defaults:
    python quantitative_advance_metrics_stage_1.py

Output folder will be created as {target}_json with one JSON per image.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stippling_metrics_advance import compute_all_advanced_metrics

# Default input folders
SOURCE_DIR = r"experiments/outputs/quantitative_advance_metrics/source"
TARGET_DIR = r"experiments/outputs/quantitative_advance_metrics/target_WVS_1024"
# TARGET_DIR = r"experiments/outputs/quantitative_advance_metrics/target_BNOT_1024"
# TARGET_DIR = r"experiments/outputs/quantitative_advance_metrics/target_GBN_1024"
# TARGET_DIR = r"experiments/outputs/quantitative_advance_metrics/target_CN_1024"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
POINTS_SOURCE = "npy"

METRIC_ORDER = [
    "M1_cvt_energy",
    "M2_voronoi_mass_cv",
    "M2_v2_power_cell_cap_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute M1-M5 metrics (points vs grayscale source density) and save as JSON"
    )
    p.add_argument(
        "--source",
        default=SOURCE_DIR,
        help=f"Folder with grayscale condition images used as density (default: {SOURCE_DIR})",
    )
    p.add_argument(
        "target",
        nargs="?",
        default=TARGET_DIR,
        help=f"Folder with stipple-dot target images (default: {TARGET_DIR})",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output folder for JSON files (default: {target}_json)",
    )
    p.add_argument(
        "--mc-approx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Monte Carlo approximation for advanced metrics (default: True)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be processed, don't compute",
    )
    p.add_argument(
        "--points-source",
        choices=("npy", "png", "auto"),
        default=POINTS_SOURCE,
        help="Where target points come from: 'npy' loads the exact <stem>.npy coords "
             "(default), 'png' detects dot centroids in the PNG, 'auto' uses the npy when "
             "present and falls back to the PNG.",
    )
    return p.parse_args()


def list_images(folder):
    """List all image files in folder, sorted."""
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(
        [str(f) for f in p.glob("*") 
         if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    )


def image_to_grayscale_01(image_path):
    """Load image and convert to grayscale [0, 1] normalized."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img.astype(np.float64) / 255.0


def extract_points_from_target(image_path, threshold=128, invert=False):
    """Extract stipple points and normalize to [0, 1] coordinates.

    Assumes DARK dots on a WHITE background: pixels strictly darker than
    `threshold` are taken as points. Set invert=True for the opposite
    convention (LIGHT dots on a DARK background).
    """
    img_u8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img_u8.shape

    if invert:
        mask = img_u8 > threshold      # light dots on dark background
    else:
        mask = img_u8 < threshold      # dark dots on white background

    ys, xs = np.where(mask)

    if xs.size == 0:
        raise ValueError(
            f"No stipple points found with threshold={threshold}, invert={invert}: "
            f"{image_path}"
        )

    points = np.column_stack([(xs + 0.5) / float(w), (ys + 0.5) / float(h)]).astype(np.float64)
    return points


def load_points_from_npy(npy_path):
    """Load exact point coords: (N, 2) float, [cx/w, cy/h], y-down, in [0, 1]."""
    pts = np.load(npy_path).astype(np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points in {npy_path}, got {pts.shape}")
    return pts


def get_target_points(image_path, points_source):
    """Return (N, 2) target points from exact .npy coords, PNG centroids, or auto.

    npy and PNG-centroid coordinates share the same convention ([x/w, y/h], y-down,
    [0, 1]), so the two are interchangeable for the metrics -- 'npy' just uses the
    solver's exact coordinates instead of re-detecting them from the rasterised dots.
    """
    npy_path = os.path.splitext(str(image_path))[0] + ".npy"
    if points_source == "png":
        return extract_points_from_target(image_path)
    if points_source == "npy":
        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"--points-source npy but no sibling .npy for {image_path} "
                f"(expected {npy_path}); use --points-source auto or png)."
            )
        return load_points_from_npy(npy_path)
    # auto: prefer exact coords, fall back to PNG centroid detection
    if os.path.exists(npy_path):
        return load_points_from_npy(npy_path)
    return extract_points_from_target(image_path)


def find_source_image(target_image_path, source_folder):
    """Find the grayscale condition image in source_folder matching a target.

    Matches by exact filename first, then by stem with any known extension.
    Returns a Path or raises FileNotFoundError.
    """
    target_name = Path(target_image_path).name
    target_stem = Path(target_image_path).stem
    src_dir = Path(source_folder)

    exact = src_dir / target_name
    if exact.exists():
        return exact

    candidates = [
        p for p in src_dir.glob(target_stem + ".*")
        if p.suffix.lower() in IMAGE_EXTS
    ]
    if candidates:
        return sorted(candidates)[0]

    raise FileNotFoundError(
        f"No matching source image for '{target_name}' in {source_folder}"
    )


def serialize_metrics(metrics):
    """Keep only M1-M5 in canonical order."""
    ordered = {}
    for key in METRIC_ORDER:
        if key in metrics:
            ordered[key] = float(metrics[key])
    return ordered


def process_images(target_folder, source_folder, output_folder, mc_approx=True, dry_run=False, points_source="npy"):
    """Score every target dot image against its matching grayscale source density."""
    target_path = Path(target_folder)
    source_path_dir = Path(source_folder)
    output_path = Path(output_folder)

    if not target_path.is_dir():
        print(f"Error: target folder does not exist: {target_path}")
        return 2
    if not source_path_dir.is_dir():
        print(f"Error: source folder does not exist: {source_path_dir}")
        return 2

    images = list_images(target_folder)
    if len(images) == 0:
        print(f"Error: no images found in {target_folder}")
        return 2

    print(f"Found {len(images)} target images in {target_folder}")
    print(f"Using grayscale density from source folder: {source_path_dir}")
    print(f"Target points source: {points_source}")

    if dry_run:
        print(f"DRY RUN: would process to {output_path}")
        for img in images:
            try:
                src = find_source_image(img, source_folder)
                print(f"  {Path(img).name}  <-  {src.name}")
            except FileNotFoundError as e:
                print(f"  {Path(img).name}  <-  MISSING SOURCE ({e})")
        return 0

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_path}")

    success_count = 0
    for idx, image_path in enumerate(images, start=1):
        image_name = Path(image_path).name
        stem = Path(image_path).stem
        json_name = f"{stem}.json"
        json_path = output_path / json_name

        try:
            # Target points: exact .npy coords, PNG centroids, or auto (per --points-source).
            points = get_target_points(image_path, points_source)

            # Density comes from the matching GRAYSCALE SOURCE condition image
            # (the field the stipples are meant to reproduce).
            source_image_path = find_source_image(image_path, source_folder)
            image_01 = image_to_grayscale_01(source_image_path)

            # compute_all_advanced_metrics expects points with shape (N, 2).
            metrics = compute_all_advanced_metrics(
                points,
                image_01,
                mc_approx=mc_approx,
            )

            # Keep only M1-M5 in canonical order
            payload = serialize_metrics(metrics)

            # Write JSON
            json_path.write_text(json.dumps(payload, indent=2))
            print(f"[{idx}/{len(images)}] {image_name} -> {json_name}")
            success_count += 1

        except Exception as e:
            print(f"[{idx}/{len(images)}] {image_name}: ERROR - {e}")
            continue

    print(f"Processed {success_count}/{len(images)} images successfully")
    return 0 if success_count == len(images) else 1


def main():
    args = parse_args()

    target_folder = args.target
    source_folder = args.source
    if args.output is None:
        output_folder = f"{target_folder}_json"
    else:
        output_folder = args.output

    print(f"Source: {source_folder}")
    print(f"Target: {target_folder}")
    print(f"Output: {output_folder}")

    rc = process_images(
        target_folder,
        source_folder,
        output_folder,
        mc_approx=args.mc_approx,
        dry_run=args.dry_run,
        points_source=args.points_source,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
