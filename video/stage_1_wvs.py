"""Stage 1 (local, CPU): Weighted Voronoi Stippling on one image, keeping every iteration.

The opening shot shows the classical per-target optimization at work, so this runs
Secord's weighted Lloyd relaxation on a pixel grid: each pixel is assigned to its nearest
point, and each point moves to the density-weighted centroid of its pixels. Density is
1 - intensity, the same convention as our sampler, and the initial points are drawn by
density-weighted rejection sampling.

    video/assets/trajectories/wvs_<name>.npz
        image    (H,W) the image in [0,1]
        points   (K,N,2) positions at iterations 0..ITERATIONS, unit square, y down
        steps    (K,) iteration index of each row

    python video/stage_1_wvs.py
    python video/stage_1_wvs.py --points 1024 --iterations 50
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRAJ_DIR  # noqa: E402

IMAGE = "video/assets/sources/z_validation_data/Icons-50_1024/source/emoji-one_4_monkey.png"
NAME = "monkey"
OUTPUT_DIR = str(TRAJ_DIR)
POINTS = 1024
ITERATIONS = 50              # the WVS default used for the paper's baseline
RESOLUTION = 512             # pixel grid the relaxation integrates over
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--name", default=NAME, help="Output is wvs_<name>.npz")
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--points", type=int, default=POINTS)
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--resolution", type=int, default=RESOLUTION)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def rejection_sample(density, n, rng):
    """n points with probability proportional to density, jittered inside their pixel."""
    h, w = density.shape
    p = density.ravel() / density.sum()
    idx = rng.choice(p.size, size=n, replace=True, p=p)
    ys, xs = np.divmod(idx, w)
    return np.stack([(xs + rng.random(n)) / w, (ys + rng.random(n)) / h], axis=1)


def lloyd_step(points, pix, weight):
    """One weighted Lloyd iteration: move each point to the weighted centroid of its cell."""
    _, owner = cKDTree(points).query(pix, k=1)
    n = len(points)
    mass = np.bincount(owner, weights=weight, minlength=n)
    cx = np.bincount(owner, weights=weight * pix[:, 0], minlength=n)
    cy = np.bincount(owner, weights=weight * pix[:, 1], minlength=n)
    moved = points.copy()
    ok = mass > 1e-12                      # a cell with no ink keeps its point where it is
    moved[ok, 0] = cx[ok] / mass[ok]
    moved[ok, 1] = cy[ok] / mass[ok]
    return moved


def main():
    a = parse_args()
    gray = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"cannot read {a.image}")
    image_01 = gray.astype(np.float32) / 255.0
    small = cv2.resize(image_01, (a.resolution, a.resolution), interpolation=cv2.INTER_AREA)
    density = np.clip(1.0 - small, 0.0, 1.0).astype(np.float64)

    r = a.resolution
    ys, xs = np.mgrid[0:r, 0:r]
    pix = np.stack([(xs.ravel() + 0.5) / r, (ys.ravel() + 0.5) / r], axis=1)
    keep = density.ravel() > 0
    pix, weight = pix[keep], density.ravel()[keep]

    rng = np.random.default_rng(a.seed)
    pts = rejection_sample(density, a.points, rng)
    frames = [pts.copy()]
    for it in range(1, a.iterations + 1):
        pts = lloyd_step(pts, pix, weight)
        frames.append(pts.copy())
        if it % 10 == 0 or it == a.iterations:
            shift = np.abs(frames[-1] - frames[-2]).max() * r
            print(f"  iteration {it:3d}: max move {shift:.2f} px")

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"wvs_{a.name}.npz"
    np.savez_compressed(dst, image=image_01, points=np.stack(frames).astype(np.float32),
                        steps=np.arange(len(frames), dtype=np.int32))
    print(f"{len(frames)} frames x {a.points} points -> {dst}")


if __name__ == "__main__":
    main()
