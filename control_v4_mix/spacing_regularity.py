"""Spacing / clumping regularity descriptors for the WVS <-> GBN axis (Gate 0, attempt 2).

Motivation
----------
The edge-alignment attempt failed: the WVS/GBN difference was a tone-transfer effect
(rho-decomposable). The next candidate the plan names is spacing/clumping REGULARITY -- how
uniformly each oracle spaces points at a given local density. This is a genuinely
non-decomposable point-process property (rho says HOW MANY points go somewhere, not whether they
are placed regularly or clumped once there), and both oracles are spacing-optimized samplers
(GBN = Gaussian Blue Noise; WVS = Weighted Voronoi / Lloyd relaxation), so a difference is
plausible.

The density confound (the lesson from attempt 1)
------------------------------------------------
Raw nearest-neighbour spacing varies simply because DENSITY varies with tone: points are closer
in dark regions. A global NN-distance CV (e.g. utils/stippling_metrics.compute_spacing_quality)
therefore mixes the density gradient with actual disorder -- the same class of confound that
tone was for edge alignment. So the PRIMARY measures here are DENSITY-NORMALIZED: each point's
spacing is divided by the local spacing implied by its own neighbourhood density, isolating
disorder from the density field. The global (confounded) measures are reported too, but only as
a cross-check.

Torch-free (numpy / scipy only). Points are (N, 2) in [0, 1].
"""

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-8

# Orientation: for these keys a SMALLER value means MORE regular (better blue-noise).
REGULAR_LOWER_KEYS = (
    "norm_nn_cv", "clumped_pct_local", "voronoi_area_cv_norm",
    "nn_cv_global", "clumped_pct_global",
)
# For these a LARGER value means MORE regular.
REGULAR_HIGHER_KEYS = ("local_gap_ratio_mean",)
# The density-CONTROLLED descriptors (verdict must rest on these, not the confounded globals).
DENSITY_CONTROLLED_KEYS = (
    "norm_nn_cv", "clumped_pct_local", "voronoi_area_cv_norm", "local_gap_ratio_mean",
)
PRIMARY_KEY = "norm_nn_cv"


def _polygon_area(poly):
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def local_density_and_spacing(pts, k=8):
    """Per-point local density (points per unit area) and implied local NN spacing.

    Density is estimated from the distance to the k-th neighbour: rho_i = k / (pi * r_k^2).
    Implied local spacing s_i = 1 / sqrt(rho_i). Returns (d1, rho, s, dk).
    """
    n = len(pts)
    k_eff = min(k, n - 1)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_eff + 1)  # col 0 is self (distance 0)
    d1 = dists[:, 1]
    dk = dists[:, k_eff]
    rho = k_eff / (np.pi * np.maximum(dk, EPS) ** 2)
    s = 1.0 / np.sqrt(np.maximum(rho, EPS))
    return d1, rho, s, dk


def voronoi_area_cv_norm(pts, rho, bbox_pad=0.02):
    """CV of DENSITY-NORMALIZED Voronoi cell areas (area_i * rho_i).

    For a locally-uniform (regular) set area_i * rho_i is nearly constant -> low CV. Border cells
    (touching infinity or reaching outside the padded unit box) are excluded. Returns nan if the
    Voronoi diagram cannot be built.
    """
    try:
        from scipy.spatial import Voronoi
        vor = Voronoi(pts)
    except Exception:
        return float("nan")
    lo, hi = -bbox_pad, 1.0 + bbox_pad
    vals = []
    for i, reg_idx in enumerate(vor.point_region):
        region = vor.regions[reg_idx]
        if len(region) == 0 or -1 in region:
            continue
        poly = vor.vertices[region]
        if poly.min() < lo or poly.max() > hi:
            continue
        vals.append(_polygon_area(poly) * rho[i])
    if len(vals) < 8:
        return float("nan")
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.std(vals) / (np.mean(vals) + EPS))


def compute_spacing_bundle(pts, k=8):
    """Full spacing-regularity bundle for one point set.

    Density-controlled (primary):
      * norm_nn_cv           -- CV of u_i = d1_i / s_i (density-invariant disorder; lower=regular)
      * clumped_pct_local    -- %% of points with d1_i < 0.5 * local spacing (density-adaptive)
      * voronoi_area_cv_norm -- CV of density-normalized Voronoi areas (lower = regular)
      * local_gap_ratio_mean -- mean d1_i / d_k_i (higher = more regular)
    Global (density-CONFOUNDED cross-check, mirrors compute_spacing_quality):
      * nn_cv_global, clumped_pct_global
      * n_points
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < k + 2:
        keys = list(REGULAR_LOWER_KEYS) + list(REGULAR_HIGHER_KEYS)
        out = {kk: float("nan") for kk in keys}
        out["n_points"] = int(n)
        return out

    d1, rho, s, dk = local_density_and_spacing(pts, k=k)
    u = d1 / np.maximum(s, EPS)  # density-invariant normalized nearest-neighbour distance

    out = {
        "n_points": int(n),
        "norm_nn_cv": float(np.std(u) / (np.mean(u) + EPS)),
        "clumped_pct_local": float(100.0 * np.mean(u < 0.5)),
        "local_gap_ratio_mean": float(np.mean(d1 / np.maximum(dk, EPS))),
        "voronoi_area_cv_norm": voronoi_area_cv_norm(pts, rho),
    }

    # Global (confounded) baseline -- mirrors utils/stippling_metrics.compute_spacing_quality.
    nn_mean = float(d1.mean()); nn_std = float(d1.std())
    out["nn_cv_global"] = nn_std / (nn_mean + EPS)
    expected_nn = 0.5 * (1.0 / n) ** 0.5
    out["clumped_pct_global"] = float(100.0 * np.mean(d1 < 0.3 * expected_nn))
    return out
