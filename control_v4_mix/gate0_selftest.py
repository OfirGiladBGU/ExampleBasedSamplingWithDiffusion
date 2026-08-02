"""Gate 0 self-test -- validates the edge-alignment descriptor math WITHOUT cluster data.

The real WVS/GBN icon data lives on the cluster, so this synthetic test lets us confirm the
descriptor behaves as designed on this dev machine before running the real verdict.

Construction: a filled disk on white (dark = ink). Under the SAME rho (the disk), we build two
point sets:
  * "GBN-like": points hugging the disk CONTOUR (a thin annulus at the boundary).
  * "WVS-like": points filling the disk INTERIOR uniformly (drawn proportional to rho).
The descriptor must report the GBN-like set as strongly edge-aligned (s_raw >> 1, small
distance-to-edge, high grad-at-points) and the WVS-like set as roughly rho-baseline
(s_raw ~= 1). If this fails, the measure is wrong and nothing downstream is trustworthy.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_alignment as ea  # noqa: E402


def make_disk_image(size=512, radius_frac=0.35):
    """Filled dark disk (ink) on white background, float32 [0,1]. dark = ink."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy)
    R = radius_frac * size
    img = np.ones((size, size), dtype=np.float32)
    img[r <= R] = 0.0  # ink
    return img, (cx, cy, R)


def sample_interior(cx, cy, R, size, n, seed=0):
    """WVS-like: uniform over the disk interior (== proportional to the flat rho on the disk)."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < n:
        xs = rng.uniform(0, size, size=n * 3)
        ys = rng.uniform(0, size, size=n * 3)
        keep = np.hypot(xs - cx, ys - cy) <= R
        for x, y in zip(xs[keep], ys[keep]):
            pts.append([x / size, y / size])
            if len(pts) >= n:
                break
    return np.array(pts, dtype=np.float64)


def sample_contour(cx, cy, R, size, n, band_px=3.0, seed=1):
    """GBN-like: a thin annulus hugging the disk boundary."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    rr = R - rng.uniform(0, band_px, size=n)  # just inside the edge
    xs = cx + rr * np.cos(theta)
    ys = cy + rr * np.sin(theta)
    return np.stack([xs / size, ys / size], axis=1).astype(np.float64)


def main():
    size = 512
    img, (cx, cy, R) = make_disk_image(size=size)
    n = 1024

    pts_wvs = sample_interior(cx, cy, R, size, n)
    pts_gbn = sample_contour(cx, cy, R, size, n)

    b_wvs = ea.compute_descriptor_bundle(img, pts_wvs)
    b_gbn = ea.compute_descriptor_bundle(img, pts_gbn)

    print("=" * 72)
    print("Gate 0 SELF-TEST (synthetic disk; no cluster data)")
    print("=" * 72)
    hdr = f"{'descriptor':<18}{'WVS-like':>14}{'GBN-like':>14}   sep?"
    print(hdr)
    print("-" * len(hdr))
    checks = []
    for key in ("s_raw", "observed_frac", "expected_frac", "grad_at_points",
                "dist_mean", "dist_median", "frac_within_2px", "frac_within_4px"):
        vw, vg = b_wvs[key], b_gbn[key]
        if key in ea.GBN_HIGHER_KEYS:
            ok = vg > vw
        elif key in ea.GBN_LOWER_KEYS:
            ok = vg < vw
        else:
            ok = None
        mark = "" if ok is None else ("  OK" if ok else "  <-- FAIL")
        print(f"{key:<18}{vw:>14.4f}{vg:>14.4f}{mark}")
        if ok is not None:
            checks.append((key, ok))

    # Core assertions.
    passed = True
    if not (b_gbn["s_raw"] > 3.0 * max(b_wvs["s_raw"], ea.EPS)):
        print("\nFAIL: GBN-like s_raw is not >> WVS-like s_raw.")
        passed = False
    if not (b_wvs["s_raw"] < 1.6):
        print(f"\nFAIL: WVS-like s_raw ({b_wvs['s_raw']:.3f}) should be ~rho-baseline (<1.6).")
        passed = False
    if not (b_gbn["dist_mean"] < b_wvs["dist_mean"]):
        print("\nFAIL: GBN-like points are not closer to edges than WVS-like.")
        passed = False
    n_fail = sum(1 for _, ok in checks if not ok)
    if n_fail > 0:
        print(f"\nFAIL: {n_fail} directional check(s) went the wrong way.")
        passed = False

    print("\n" + ("SELF-TEST PASSED" if passed else "SELF-TEST FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
