"""Convert stippled dot images into offset tensors for V3 training.

Same OT pipeline as V1/V2 -- only the training architecture changed,
not the ground-truth representation.

For each stippled image (black dots on white background):
  1. Extract dot centroids via contour detection
  2. Map to the 32x32 offset grid using the repo's OT transform
  3. Save as a .npy file of shape (2, 32, 32)

Usage (from project root):
    python control_v3/prepare_data.py \
        --gt_images /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/target \
        --output    /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets \
        --grid_size 32
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Transforms import to_image_optimal_transport

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def extract_points_from_image(img_path, n_points):
    """Detect dot centroids in a stippled image and return (N, 2) in [0, 1]."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    inv = cv2.bitwise_not(img)
    _, thresh = cv2.threshold(inv, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = img.shape
    points = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        points.append([cx / w, cy / h])

    points = np.array(points, dtype=np.float64)

    if len(points) > n_points:
        indices = np.random.RandomState(42).choice(
            len(points), n_points, replace=False
        )
        points = points[indices]
    elif len(points) < n_points:
        deficit = n_points - len(points)
        pad = np.random.RandomState(42).rand(deficit, 2)
        points = np.vstack([points, pad])
        print(
            f"  WARNING: only {len(points) - deficit} dots detected, "
            f"padded {deficit} random points"
        )

    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt_images",
        default="/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/target",
        help="Directory of stippled (dot) ground-truth images",
    )
    parser.add_argument(
        "--output",
        default="/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets",
        help="Directory to write .npy offset files",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=32,
        help="Grid resolution (default 32 -> 1024 points)",
    )
    args = parser.parse_args()

    n_points = args.grid_size ** 2
    os.makedirs(args.output, exist_ok=True)

    files = sorted(
        f for f in os.listdir(args.gt_images)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    )
    print(f"Processing {len(files)} images  (grid={args.grid_size}, "
          f"N={n_points} points)")

    for fname in tqdm(files):
        pts = extract_points_from_image(
            os.path.join(args.gt_images, fname), n_points
        )
        offsets = to_image_optimal_transport(pts)

        stem = os.path.splitext(fname)[0]
        np.save(os.path.join(args.output, stem + ".npy"), offsets)

    print(f"Done. Offsets written to {args.output}")


if __name__ == "__main__":
    main()
