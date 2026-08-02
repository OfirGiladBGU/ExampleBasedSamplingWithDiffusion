"""Self-test for spacing_regularity -- validates the descriptor AND the density control, no data.

Four synthetic point sets on the unit square (all N ~= 1024):
  * regular          -- jittered grid            -> LOW norm_nn_cv, LOW clumped
  * poisson          -- uniform random           -> MID norm_nn_cv
  * clumped          -- Gaussian clusters        -> HIGH norm_nn_cv, HIGH clumped_pct_local
  * gradient_regular -- locally-regular lattice with a strong left-right DENSITY gradient

The decisive check is 'gradient_regular': it is locally regular but its density varies a lot, so
a naive GLOBAL nn_cv reads it as disordered while the density-normalized norm_nn_cv must stay
low. If that separation of global vs normalized does not hold, the density control is broken and
the whole attempt-2 verdict would be untrustworthy.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spacing_regularity as sr  # noqa: E402


def regular(n_side=32, jitter=0.15, seed=0):
    rng = np.random.RandomState(seed)
    g = (np.arange(n_side) + 0.5) / n_side
    x, y = np.meshgrid(g, g)
    p = np.stack([x.ravel(), y.ravel()], 1)
    p = p + (rng.rand(*p.shape) - 0.5) * jitter / n_side
    return np.clip(p, 0, 1)


def poisson(n=1024, seed=1):
    return np.random.RandomState(seed).rand(n, 2)


def clumped(n_clusters=64, per=16, spread=0.008, seed=2):
    rng = np.random.RandomState(seed)
    centers = rng.rand(n_clusters, 2)
    pts = np.repeat(centers, per, 0) + rng.randn(n_clusters * per, 2) * spread
    return np.clip(pts, 0, 1)


def gradient_regular(a=3.0, h0=0.014, jitter=0.15, seed=3):
    """Locally-regular ISOTROPIC lattice with a strong left->right density gradient.

    Crucial that the gradient is isotropic: local spacing h(x) = h0*(1+a*x) scales BOTH the
    column step and the within-column row step, so cells stay ~square (no shear). A naive warp
    that compresses only x makes cells elongated, and norm_nn_cv would (correctly) flag that
    anisotropy as disorder -- which is NOT what this case is meant to test. Here spacing varies
    by a factor (1+a) across the field (density by (1+a)^2), yet each neighbourhood is a regular
    isotropic lattice, so the density-normalized u = d1/s is ~constant and norm_nn_cv stays low
    while the global nn_cv is large.
    """
    rng = np.random.RandomState(seed)
    pts = []
    x = h0 * 0.5
    while x < 1.0:
        h = h0 * (1.0 + a * x)
        y = h * 0.5
        while y < 1.0:
            pts.append((x + (rng.rand() - 0.5) * jitter * h,
                        y + (rng.rand() - 0.5) * jitter * h))
            y += h
        x += h
    return np.clip(np.array(pts, dtype=np.float64), 0, 1)


def main():
    sets = {
        "regular": regular(),
        "poisson": poisson(),
        "clumped": clumped(),
        "gradient_regular": gradient_regular(),
    }
    b = {k: sr.compute_spacing_bundle(v, k=8) for k, v in sets.items()}

    print("=" * 90)
    print("spacing_regularity SELF-TEST (synthetic; no cluster data)")
    print("=" * 90)
    keys = ["n_points", "norm_nn_cv", "clumped_pct_local", "voronoi_area_cv_norm",
            "local_gap_ratio_mean", "nn_cv_global", "clumped_pct_global"]
    hdr = f"{'descriptor':<22}" + "".join(f"{name:>18}" for name in sets)
    print(hdr); print("-" * len(hdr))
    for key in keys:
        vals = "".join(f"{b[name][key]:>18.4f}" for name in sets)
        print(f"{key:<22}{vals}")

    passed = True

    def check(cond, msg):
        nonlocal passed
        if not cond:
            print(f"  FAIL: {msg}"); passed = False

    print()
    # Ordering of disorder.
    check(b["regular"]["norm_nn_cv"] < b["poisson"]["norm_nn_cv"],
          "regular should be more regular (lower norm_nn_cv) than poisson")
    check(b["poisson"]["norm_nn_cv"] < b["clumped"]["norm_nn_cv"],
          "poisson should be more regular than clumped")
    check(b["clumped"]["clumped_pct_local"] > 5.0 * max(b["regular"]["clumped_pct_local"], 0.1),
          "clumped set should have far higher local clumping than regular")

    # THE density-control check.
    gr = b["gradient_regular"]
    check(gr["norm_nn_cv"] < 2.0 * b["regular"]["norm_nn_cv"] + 0.06,
          "gradient_regular must stay LOW on the density-normalized norm_nn_cv "
          "(near the regular value, NOT near poisson)")
    check(gr["nn_cv_global"] > 1.8 * gr["norm_nn_cv"],
          "for a density GRADIENT, global nn_cv must be much larger than the normalized one "
          "(this is the confound the control removes)")
    check(gr["norm_nn_cv"] < 0.5 * b["poisson"]["norm_nn_cv"],
          "gradient_regular must be clearly MORE regular than poisson on the normalized measure")

    print("\n" + ("SELF-TEST PASSED" if passed else "SELF-TEST FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
