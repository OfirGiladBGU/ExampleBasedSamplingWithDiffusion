"""Quantitative metrics stage 1: compute M1-M5 metrics for target images.

This script reads all PNG/JPG images from a given target folder and computes
advanced metrics M1-M5, writing JSON results to a parallel folder with "_json" suffix.

Usage:
    python quantitative_advance_metrics_stage_1.py \\
        /path/to/target_BNOT_1024

Or with all defaults:
    python quantitative_advance_metrics_stage_1.py

Output folder will be created as {input_path}_json with one JSON per image.
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

# Default input folder (one of the quantitative metric folders)
DEFAULT_INPUT_FOLDER = r"experiments/outputs/quantitative_advance_metrics/target_WVS_1024"
# DEFAULT_INPUT_FOLDER = r"experiments/outputs/quantitative_advance_metrics/target_BNOT_1024"
# DEFAULT_INPUT_FOLDER = r"experiments/outputs/quantitative_advance_metrics/target_GBN_1024"
# DEFAULT_INPUT_FOLDER = r"experiments/outputs/quantitative_advance_metrics/target_CN_1024"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

METRIC_ORDER = [
    "M1_cvt_energy",
    "M2_voronoi_mass_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute M1-M5 metrics for target images and save as JSON"
    )
    p.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT_FOLDER,
        help=f"Input folder containing target images (default: {DEFAULT_INPUT_FOLDER})",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output folder for JSON files (default: {input}_json)",
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


def extract_points_from_target(image_path):
    """Extract stipple points from dark pixels and normalize to [0, 1] coordinates."""
    img_u8 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img_u8.shape

    # Use Otsu threshold and keep the darker class as stipple points.
    otsu_thr, _ = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(img_u8 < otsu_thr)

    # Fallbacks for edge cases (near-uniform or anti-aliased images).
    if xs.size == 0:
        ys, xs = np.where(img_u8 < 128)
    if xs.size == 0:
        ys, xs = np.where(img_u8 < 250)

    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    points = np.column_stack([(xs + 0.5) / float(w), (ys + 0.5) / float(h)]).astype(np.float64)
    return points


def serialize_metrics(metrics):
    """Keep only M1-M5 in canonical order."""
    ordered = {}
    for key in METRIC_ORDER:
        if key in metrics:
            ordered[key] = float(metrics[key])
    return ordered


def process_images(input_folder, output_folder, mc_approx=True, dry_run=False):
    """Process all images in input_folder and save metrics to output_folder."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.is_dir():
        print(f"Error: input folder does not exist: {input_path}")
        return 2

    images = list_images(input_folder)
    if len(images) == 0:
        print(f"Error: no images found in {input_folder}")
        return 2

    print(f"Found {len(images)} images in {input_folder}")

    if dry_run:
        print(f"DRY RUN: would process to {output_path}")
        for img in images:
            print(f"  {Path(img).name}")
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
            # Load image density and extract stipple points from target dots.
            image_01 = image_to_grayscale_01(image_path)
            points = extract_points_from_target(image_path)

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

    input_folder = args.input
    if args.output is None:
        output_folder = f"{input_folder}_json"
    else:
        output_folder = args.output

    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")

    rc = process_images(
        input_folder,
        output_folder,
        mc_approx=args.mc_approx,
        dry_run=args.dry_run,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
