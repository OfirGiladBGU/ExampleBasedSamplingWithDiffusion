"""Advanced stippling metrics and GBN-style density visualization.

Extends stippling_metrics.py with:
    - M1-M5 quality metrics (Voronoi, Sinkhorn, EMD, Adaptive-NND, CVT)
    - GBN Algorithm 2 adaptive density reconstruction
    - 4-row overfit panel with M1-M5 text row
    - AKDE density-map visualization panel (GBN paper style)
"""

import os
import random
import numpy as np

# Blanket global seeds to ensure 100% determinism for 3rd party libraries
np.random.seed(42)
random.seed(42)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from utils.stippling_metrics import (
    compute_grid_capacity,
    compute_spacing_quality,
    resolve_capacity_grid_size,
)


# ============================================================================
# MODULE CONSTANTS
# ============================================================================

# Working resolution for the iterative power-diagram solve.
POWER_RESOLUTION = 256

# ── constants for the shared advanced-metrics row ──────────────────
_ADV_ROW_HEIGHT_RATIO   = 2.0
_ADV_ROW_EXTRA_HEIGHT   = 3.5  # inches added to base figure height
# Masked M2_v3: pixel value (0..255). The mask keeps pixels strictly BELOW this, so 255
# ignores only PURE-white background when restricting the power-cell probes to the stipple support.
MASK_THRESHOLD = 255


# ============================================================================
# SHARED HELPERS
# ============================================================================

# Dynamically fetch the current matplotlib global font size so it 
# perfectly syncs with compare_advance_metrics.py
def _fs():
    return plt.rcParams.get('font.size', 12) if plt else 12


def _to_density(image_01):
    """Convert a raw grayscale image (white=1, dark=0) into a target density
    rho(x) where dark regions carry HIGH mass (more stipples).

    All numeric metric functions weight by rho, so this single inversion keeps
    them consistent with the target density used by the visual functions
    (which compute ``1.0 - image_01`` inline). Returns float64 in [0, 1].
    """
    rho = 1.0 - np.asarray(image_01, dtype=np.float64)
    # Guard against tiny negative values from float round-off.
    return np.clip(rho, 0.0, 1.0)


def _warn_metric_failure(metric, exc):
    """Announce a swallowed metric failure instead of silently reporting 0.0.

    The handlers below return zeros so long-running callers keep going, but 0.0 is a PERFECT
    score for every lower-is-better metric here, so a silent failure would bias a batch mean
    downwards invisibly. `compute_all_advanced_metrics` omits keys instead (see `_collect`);
    these warnings cover the direct callers that still rely on the zero fallback.
    """
    import warnings
    warnings.warn(f"{metric} failed ({type(exc).__name__}: {exc}); reporting 0.0 for this "
                  f"sample -- treat it as MISSING, not as a perfect score",
                  RuntimeWarning, stacklevel=3)


def _collect(result, prefix, fn, *args, **kwargs):
    """Run one metric and merge its keys under `prefix`, isolating failures.

    M2_v2 and M2_v3 raise on a misconfigured input (no image, or a target carrying no mass)
    rather than silently reporting a perfect 0.0. That is the right behaviour for a single
    call, but a batch scorer must not lose an entire sample -- or an entire multi-hour run --
    because one image is degenerate. On failure we emit a warning and OMIT this metric's keys.

    Omitting rather than substituting 0.0 or NaN is deliberate: downstream aggregation skips
    absent keys, so a bad sample lowers that metric's `count` (visible in the summary) instead
    of quietly biasing its mean.
    """
    import warnings
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:
        warnings.warn(
            f"{fn.__name__} failed ({type(exc).__name__}: {exc}); omitting {prefix}* keys "
            f"for this sample",
            RuntimeWarning, stacklevel=2,
        )
        return

    # Drop non-finite values for the same reason failures are dropped. NaN/inf reach here
    # when the scored point set itself contains them -- an undertrained checkpoint can emit
    # NaN coordinates -- and a NaN written to JSON silently turns that epoch's mean into NaN
    # during aggregation. Omitting instead lowers the metric's `count`, which is visible.
    clean, dropped = {}, []
    for k, v in out.items():
        try:
            finite = np.isfinite(v)
        except TypeError:
            finite = True                     # non-numeric payloads pass through untouched
        if finite:
            clean[f"{prefix}{k}"] = v
        else:
            dropped.append(f"{prefix}{k}")
    if dropped:
        warnings.warn(
            f"{fn.__name__} returned non-finite values for {', '.join(dropped)}; omitting them "
            f"for this sample (usually a NaN/inf in the input point set)",
            RuntimeWarning, stacklevel=2,
        )
    result.update(clean)


def _coarsen_density(rho, budget=8000):
    """Block-sum `rho` down to at most `budget` non-empty cells.

    Returns (centres, masses) with centres in [0,1]^2 and masses summing to the total mass
    of `rho` times the pixel area, i.e. \\int rho dx. Empty cells are dropped.
    """
    rho = np.asarray(rho, dtype=np.float64)
    H, W = rho.shape
    pixel_area = 1.0 / float(H * W)
    for b in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        Hc = int(np.ceil(H / b))
        Wc = int(np.ceil(W / b))
        pad = np.zeros((Hc * b, Wc * b), dtype=np.float64)
        pad[:H, :W] = rho
        blocks = pad.reshape(Hc, b, Wc, b).sum(axis=(1, 3)) * pixel_area
        nz = blocks > 0
        if int(nz.sum()) <= budget or b == 64:
            iy, ix = np.nonzero(nz)
            # centre of each block, in image coordinates, clipped to the true image extent
            cx = np.minimum((ix + 0.5) * b, W) / float(W)
            cy = np.minimum((iy + 0.5) * b, H) / float(H)
            return np.stack([cx, cy], axis=1), blocks[iy, ix]
    raise AssertionError("unreachable")


def _polygon_area(poly):
    """Shoelace area of a polygon given as a list of (x, y) vertices."""
    n = len(poly)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x1, y1 = poly[i - 1]
        x2, y2 = poly[i]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _polygon_sample_points(vertices_clip, image_shape, mc_approx=True, rng=None):
    """Return points and pixel indices inside a polygon.

    When mc_approx is True, samples points randomly inside the polygon bbox.
    When mc_approx is False, uses all pixel centers inside the bbox.
    """
    from matplotlib.path import Path

    H, W = image_shape
    x_min, x_max = vertices_clip[:, 0].min(), vertices_clip[:, 0].max()
    y_min, y_max = vertices_clip[:, 1].min(), vertices_clip[:, 1].max()
    if x_max - x_min < 1e-8 or y_max - y_min < 1e-8:
        return None

    poly_path = Path(vertices_clip)

    if mc_approx:
        if rng is None:
            rng = np.random.default_rng(42)
        n_samples = max(10, min(50, int((x_max - x_min) * (y_max - y_min) * W * H * 0.5)))
        xs = rng.uniform(x_min, x_max, n_samples)
        ys = rng.uniform(y_min, y_max, n_samples)
        points_test = np.column_stack([xs, ys])
    else:
        x0 = max(0, int(np.floor(x_min * W)))
        x1 = min(W - 1, int(np.ceil(x_max * W)))
        y0 = max(0, int(np.floor(y_min * H)))
        y1 = min(H - 1, int(np.ceil(y_max * H)))
        if x1 < x0 or y1 < y0:
            return None
        xs = (np.arange(x0, x1 + 1, dtype=np.float64) + 0.5) / float(W)
        ys = (np.arange(y0, y1 + 1, dtype=np.float64) + 0.5) / float(H)
        gx, gy = np.meshgrid(xs, ys)
        points_test = np.column_stack([gx.ravel(), gy.ravel()])

    inside = poly_path.contains_points(points_test)
    if inside.sum() == 0:
        return None

    points_inside = points_test[inside]
    px = np.clip((points_inside[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip((points_inside[:, 1] * H).astype(int), 0, H - 1)
    return points_inside, px, py


def _clip_polygon_halfplane(poly, a0, a1, b):
    """Sutherland-Hodgman clip of polygon `poly` (list of (x, y)) by the half-plane
    a0*x + a1*y <= b. Returns the clipped vertex list (possibly empty).
    """
    out = []
    n = len(poly)
    if n == 0:
        return out
    for i in range(n):
        S = poly[i - 1]
        E = poly[i]
        sd = a0 * S[0] + a1 * S[1] - b
        ed = a0 * E[0] + a1 * E[1] - b
        s_in = sd <= 0.0
        e_in = ed <= 0.0
        if e_in:
            if not s_in:
                t = sd / (sd - ed)
                out.append((S[0] + t * (E[0] - S[0]), S[1] + t * (E[1] - S[1])))
            out.append(E)
        elif s_in:
            t = sd / (sd - ed)
            out.append((S[0] + t * (E[0] - S[0]), S[1] + t * (E[1] - S[1])))
    return out


def _clip_polygon_to_bbox(vertices, bbox=(0.0, 1.0, 0.0, 1.0)):
    xmin, xmax, ymin, ymax = bbox

    def _clip_edge(poly, ax0, ay0, ax1, ay1):
        if not poly:
            return []

        def _inside(p):
            return (ax1 - ax0) * (p[1] - ay0) - (ay1 - ay0) * (p[0] - ax0) >= 0

        def _intersect(p1, p2):
            dx, dy = ax1 - ax0, ay1 - ay0
            dpx, dpy = p2[0] - p1[0], p2[1] - p1[1]
            denom = dx * dpy - dy * dpx
            if abs(denom) < 1e-12:
                return p1
            t = (dy * (p1[0] - ax0) - dx * (p1[1] - ay0)) / denom
            return (p1[0] + t * dpx, p1[1] + t * dpy)

        out = []
        for i, curr in enumerate(poly):
            prev = poly[i - 1]
            if _inside(curr):
                if not _inside(prev):
                    out.append(_intersect(prev, curr))
                out.append(curr)
            elif _inside(prev):
                out.append(_intersect(prev, curr))
        return out

    poly = [(v[0], v[1]) for v in vertices]
    poly = _clip_edge(poly, xmin, ymin, xmax, ymin)   
    poly = _clip_edge(poly, xmax, ymin, xmax, ymax)   
    poly = _clip_edge(poly, xmax, ymax, xmin, ymax)   
    poly = _clip_edge(poly, xmin, ymax, xmin, ymin)   
    return np.array(poly) if poly else np.zeros((0, 2))


def _voronoi_regions_clipped(vor, bbox=(0.0, 1.0, 0.0, 1.0)):
    xmin, xmax, ymin, ymax = bbox
    center = vor.points.mean(axis=0)
    radius = max(xmax - xmin, ymax - ymin) * 4.0

    ridge_map = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_map.setdefault(p1, []).append((p2, v1, v2))
        ridge_map.setdefault(p2, []).append((p1, v1, v2))

    ext_verts = vor.vertices.tolist()

    result = []
    for pt_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        is_finite = all(v >= 0 for v in region)

        if is_finite:
            raw = vor.vertices[region]
        else:
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridge_map.get(pt_idx, []):
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue  

                tangent = vor.points[p2] - vor.points[pt_idx]
                norm = np.linalg.norm(tangent)
                if norm < 1e-12:
                    continue
                tangent /= norm
                normal = np.array([-tangent[1], tangent[0]])
                midpoint = vor.points[[pt_idx, p2]].mean(axis=0)
                sign = np.sign(np.dot(midpoint - center, normal))
                far_pt = vor.vertices[v2] + sign * normal * radius
                ext_verts.append(far_pt.tolist())
                new_region.append(len(ext_verts) - 1)

            if len(new_region) < 3:
                result.append((np.zeros((0, 2)), is_finite))
                continue

            ext_arr = np.array(ext_verts)
            vs = ext_arr[new_region]
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            raw = vs[np.argsort(angles)]

        clipped = _clip_polygon_to_bbox(raw, bbox)
        result.append((clipped, is_finite))

    return result


# ============================================================================
# POWER DIAGRAM CORE
# ============================================================================

def _knn_power_weights(pts, k=8):
    """Power weights from the local point spacing: w_i = r_k^2 / k, mean-centred.

    Derived from the kNN intensity estimate lambda_i = k / (pi r_k^2) as w_i = (1/lambda_i)/pi,
    i.e. the weight rule used by the earlier M2_v2 / M2_v3 code path. Weights carry units of
    length^2, as the power distance requires. Mean-centring is free: the power diagram is
    invariant to adding a constant to every weight.

    NOT the default -- this rule carries a boundary bias. A point on the edge of the support
    has neighbours on one side only, so its r_k is inflated and it receives a spuriously large
    weight. On a perfect lattice over a uniform density (where the true answer is exactly 0)
    this rule reports delta_c = 5.4e-3, with border points contributing 55% of the error while
    making up 12% of the set; interior weights have std exactly 0, border weights do not. On
    icon silhouettes every contour pixel is such a boundary. See `weight_mode="none"`, which
    is exact on that configuration, and tests/test_m2_capacity.py.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    k_eff = int(min(k, n - 1))
    if k_eff < 1:
        return np.zeros(n, dtype=np.float64)
    dk = cKDTree(pts).query(pts, k=k_eff + 1)[0][:, k_eff]
    w = np.maximum(dk, 1e-12) ** 2 / float(k_eff)
    return w - w.mean()


def _power_cell_masses(pts, rho, weights, shortlist=32):
    """m_i = \\int_{V_i^w} rho dx for every power cell, by exact pixel enumeration.

    Every pixel centre of the domain is assigned to its power-cell owner (the site
    minimising ||x - x_i||^2 - w_i, exactly the definition in BNOT Sec. 2.3) and contributes
    rho(pixel) * pixel_area. Deterministic: no Monte-Carlo, which matters because the old
    Voronoi M2 drew at most 50 bbox-rejection probes per cell and that sampling noise alone
    inflated delta_c.

    Owner lookup re-ranks a Euclidean k-NN shortlist by power distance (O(m log n)). That is
    exact for equal weights and for the small weight spreads produced by `_knn_power_weights`;
    `shortlist` bounds the error for larger spreads.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    n = len(pts)
    H, W = rho.shape
    ys, xs = np.mgrid[0:H, 0:W]
    ys = ys.ravel()
    xs = xs.ravel()
    probe = np.stack([(xs + 0.5) / float(W), (ys + 0.5) / float(H)], axis=1)
    kq = int(min(n, shortlist))
    dd, ii = cKDTree(pts).query(probe, k=kq)
    if kq == 1:
        dd = dd[:, None]
        ii = ii[:, None]
    pw = dd ** 2 - np.asarray(weights, dtype=np.float64)[ii]
    owners = ii[np.arange(len(probe)), np.argmin(pw, axis=1)]
    pixel_area = 1.0 / float(H * W)
    return np.bincount(owners, weights=rho[ys, xs], minlength=n).astype(np.float64) * pixel_area


def power_cell_areas(pts, weights=None, n_probe=48, rng=None, mask=None):
    """Monte-Carlo POWER (Laguerre) cell areas (they sum to 1). Ported from
    control_v4_mix_metrics/descriptor_fields.py. Power distance is |x - p|^2 - w; each probe is
    assigned to its min-power owner via a Euclidean shortlist re-ranked by power distance, which
    stays O(m log n) with bounded weights.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    rng = np.random.RandomState(0) if rng is None else rng
    if weights is None:
        weights = np.zeros(n)
    weights = np.asarray(weights, dtype=np.float64)
    m = int(n_probe * n)
    if mask is None:
        probe = rng.rand(m, 2)
    else:
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return np.zeros(n, dtype=np.float64)
        H, W = mask.shape
        sel = rng.randint(0, len(ys), size=m)
        probe = np.stack([(xs[sel] + rng.rand(m)) / float(W),
                          (ys[sel] + rng.rand(m)) / float(H)], axis=1)
    tree = cKDTree(pts)
    kq = min(n, 16)
    dd, ii = tree.query(probe, k=kq)
    if kq == 1:
        dd = dd[:, None]
        ii = ii[:, None]
    pw = dd ** 2 - weights[ii]
    owners = ii[np.arange(m), np.argmin(pw, axis=1)]
    counts = np.bincount(owners, minlength=n).astype(np.float64)
    return counts / max(m, 1)


def power_cell_areas_exact(pts, weights=None, domain=(0.0, 1.0, 0.0, 1.0)):
    """Exact POWER (Laguerre) cell areas on a rectangular domain, normalised to sum 1.

    The exact counterpart of `power_cell_areas` (used when mc_approx=False). Each cell
        V_i = { x : ||x - p_i||^2 - w_i <= ||x - p_j||^2 - w_j  for all j }
    is a convex polygon, because every pairwise power inequality is linear in x:
        2 (p_j - p_i) . x  <=  |p_j|^2 - |p_i|^2 + w_i - w_j.
    So the cell is the domain rectangle clipped by all those half-planes -- no Monte-
    Carlo. Sites are visited nearest-first and half-planes that do not cut the current
    polygon are skipped (exact prune); worst case is O(n^2). A point with a large enough
    weight can legitimately own an empty cell (area 0), as in the power-diagram paper.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if weights is None:
        weights = np.zeros(n)
    weights = np.asarray(weights, dtype=np.float64)
    x0, x1, y0, y1 = domain
    sq = (pts ** 2).sum(axis=1)                       # |p_i|^2
    rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    areas = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # nearest-first: the polygon shrinks fast so most far half-planes get pruned
        order = np.argsort(((pts - pts[i]) ** 2).sum(axis=1))
        a0 = 2.0 * (pts[:, 0] - pts[i, 0])
        a1 = 2.0 * (pts[:, 1] - pts[i, 1])
        bb = sq - sq[i] + weights[i] - weights
        poly = list(rect)
        for j in order:
            if j == i:
                continue
            aj0 = a0[j]; aj1 = a1[j]; bj = bb[j]
            # skip planes that leave the whole current polygon on site i's side
            if all(aj0 * vx + aj1 * vy - bj <= 1e-15 for (vx, vy) in poly):
                continue
            poly = _clip_polygon_halfplane(poly, aj0, aj1, bj)
            if len(poly) < 3:
                poly = []
                break
        areas[i] = _polygon_area(poly)
    total = areas.sum()
    if total > 1e-15:
        areas = areas / total
    return areas


def power_cell_areas_masked_full(pts, weights=None, mask=None):
    """Deterministic masked power-cell areas: assign EVERY non-white pixel centre to its power
    owner (no Monte-Carlo), so it uses the full masked support with no sampling noise. This is the
    exact (mc_approx=False) counterpart of the masked MC path -- the mask *is* the domain, so
    enumerating its pixels needs no polygon clipping. Areas are fractions of the mask (sum to 1);
    owner lookup uses the same 16-NN power-distance shortlist as power_cell_areas.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if weights is None:
        weights = np.zeros(n)
    weights = np.asarray(weights, dtype=np.float64)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros(n, dtype=np.float64)
    H, W = mask.shape
    probe = np.stack([(xs + 0.5) / float(W), (ys + 0.5) / float(H)], axis=1)
    tree = cKDTree(pts)
    kq = min(n, 16)
    dd, ii = tree.query(probe, k=kq)
    if kq == 1:
        dd = dd[:, None]
        ii = ii[:, None]
    pw = dd ** 2 - weights[ii]
    owners = ii[np.arange(len(probe)), np.argmin(pw, axis=1)]
    counts = np.bincount(owners, minlength=n).astype(np.float64)
    return counts / max(len(probe), 1)




class PowerDiagram:
    """Power (Laguerre) cells of a weighted point set, evaluated by exact pixel enumeration.

    Only pixels carrying density are kept as probes: mass and centroids are both rho-weighted,
    so a zero-density pixel contributes nothing to either, and the only thing dropping it
    changes is which cell nominally owns empty background. On icons this removes ~60% of the
    probes. The k-NN shortlist is built ONCE -- point positions never move while weights are
    solved, so each subsequent evaluation is an argmin over a fixed array, which is what makes
    an iterative weight solve affordable.
    """

    def __init__(self, pts, rho, shortlist=48):
        from scipy.spatial import cKDTree
        self.pts = np.asarray(pts, dtype=np.float64)
        self.n = len(self.pts)
        rho = np.asarray(rho, dtype=np.float64)
        H, W = rho.shape
        self.shape = (H, W)

        ys, xs = np.nonzero(rho > 0.0)
        if len(ys) == 0:
            gy, gx = np.mgrid[0:H, 0:W]
            ys, xs = gy.ravel(), gx.ravel()
        self.ys, self.xs = ys, xs
        self.probe = np.stack([(xs + 0.5) / W, (ys + 0.5) / H], axis=1)
        self.rho_w = rho[ys, xs] / float(H * W)          # includes the pixel area factor
        self.total_mass = float(self.rho_w.sum())

        kq = int(min(self.n, shortlist))
        dd, ii = cKDTree(self.pts).query(self.probe, k=kq)
        if kq == 1:
            dd, ii = dd[:, None], ii[:, None]
        self.d2 = dd ** 2
        self.ii = ii
        self.rows = np.arange(len(self.probe))

    def owners(self, w):
        """Index of the site minimising the power distance ||x - p_i||^2 - w_i."""
        return self.ii[self.rows, np.argmin(self.d2 - w[self.ii], axis=1)]

    def masses(self, w):
        return np.bincount(self.owners(w), weights=self.rho_w, minlength=self.n)

    def cell_stats(self, w):
        """(mass, centroid, CVT energy) per cell, all rho-weighted."""
        o = self.owners(w)
        m = np.bincount(o, weights=self.rho_w, minlength=self.n)
        cx = np.bincount(o, weights=self.rho_w * self.probe[:, 0], minlength=self.n)
        cy = np.bincount(o, weights=self.rho_w * self.probe[:, 1], minlength=self.n)
        safe = np.maximum(m, 1e-15)
        centroid = np.stack([cx / safe, cy / safe], axis=1)
        d2 = ((self.probe - self.pts[o]) ** 2).sum(axis=1)
        return m, centroid, float((self.rho_w * d2).sum())

    def owners_exact(self, w, chunk=16384):
        """Owner over ALL sites with no shortlist -- ground truth for validating the shortlist."""
        out = np.empty(len(self.probe), dtype=np.int64)
        sq = (self.pts ** 2).sum(axis=1)
        for a in range(0, len(self.probe), chunk):
            b = min(a + chunk, len(self.probe))
            pw = (-2.0 * (self.probe[a:b] @ self.pts.T)) + (sq - w)[None, :]
            out[a:b] = np.argmin(pw, axis=1)
        return out

    def shortlist_is_adequate(self, w):
        return float((self.owners(w) == self.owners_exact(w)).mean())

    def solve_weights(self, maxiter=200, tol=1e-16, bound_scale=64.0):
        """Weights that equalise capacity, by maximising the concave semi-discrete OT dual.

            Phi(w) = m* sum_i w_i + integral min_j( ||x-x_j||^2 - w_j ) rho dx
            dPhi/dw_i = m* - m_i(w)

        This is the problem BNOT solves (it uses Newton on the weighted Laplacian and reports a
        residual of ~1e-12 in 3-5 iterations); L-BFGS on the same dual is far simpler and, as
        tests/test_power_metrics.py shows, converges tightly enough that the metrics built on it
        are insensitive to the remaining residual.

        The weights are BOUNDED, which is not optional. Without a bound the fixed shortlist and
        the solve fight each other: a growing w_i should expand cell i onto distant pixels, but
        those pixels do not carry i in their shortlist, so the mass never arrives, the gradient
        keeps demanding a larger w_i, and it diverges -- measured at |w| ~ 1e51, with capacity
        getting WORSE rather than better. The default 64/n was picked from a sweep on real
        stipples (no weights pinned at the bound, shortlist agreeing with the exact assignment
        >99%); a badly density-mismatched point set may need more, which the returned
        `at_bound` fraction makes visible rather than silent.
        """
        from scipy.optimize import minimize
        zeros = np.zeros(self.n)
        if self.total_mass <= 1e-15:
            return zeros, {"converged": False, "reason": "no mass", "residual_delta_c": 0.0,
                           "residual_at_w0": 0.0, "iterations": 0, "at_bound": 0.0}
        m_star = self.total_mass / self.n
        lim = bound_scale / max(self.n, 1)

        def neg_phi_grad(w):
            pw = self.d2 - w[self.ii]
            j = np.argmin(pw, axis=1)
            phi = m_star * w.sum() + float((pw[self.rows, j] * self.rho_w).sum())
            m = np.bincount(self.ii[self.rows, j], weights=self.rho_w, minlength=self.n)
            return -phi, -(m_star - m)

        res = minimize(neg_phi_grad, zeros, jac=True, method="L-BFGS-B",
                       bounds=[(-lim, lim)] * self.n,
                       options={"maxiter": maxiter, "maxcor": 20, "ftol": tol, "gtol": tol})
        w = res.x - res.x.mean()                     # power diagram is shift-invariant

        def dc(m):
            return float(np.mean((m / max(m.mean(), 1e-15) - 1.0) ** 2))

        resid, resid0 = dc(self.masses(w)), dc(self.masses(zeros))
        if resid > resid0:
            # No weighting we can find improves capacity. This happens when points are stranded
            # on zero density -- no finite weight can hand them mass. Falling back to w = 0 and
            # saying so is the honest outcome.
            return zeros, {"converged": False, "reason": "no improvement",
                           "residual_delta_c": resid0, "residual_at_w0": resid0,
                           "iterations": int(res.nit), "at_bound": 0.0}
        return w, {"converged": bool(res.success), "residual_delta_c": resid,
                   "residual_at_w0": resid0, "iterations": int(res.nit),
                   "at_bound": float(np.mean(np.abs(res.x) >= lim * 0.999))}


def stranded_points(points, image_01, radius_px=6):
    """Mask of points whose entire neighbourhood carries no density.

    Such a point can never receive its share of mass however it is weighted, so the capacity
    problem is genuinely infeasible for it. That is a property of the point set worth reporting
    on its own rather than burying inside a failed weight solve.
    """
    rho = _to_density(image_01)
    H, W = rho.shape
    pts = np.asarray(points, dtype=np.float64)
    px = np.clip((pts[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip((pts[:, 1] * H).astype(int), 0, H - 1)
    r = int(radius_px)
    out = np.zeros(len(px), dtype=bool)
    for i, (x, y) in enumerate(zip(px, py)):
        patch = rho[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        out[i] = patch.size > 0 and patch.max() < 1e-3
    return out


def solve_power_diagram(points, image_01, resolution=POWER_RESOLUTION, shortlist=48,
                        maxiter=200, bound_scale=64.0):
    """Build the capacity-optimal power diagram once, for M1_v2 and M2_v2 to share.

    The source is downsampled to `resolution` first: the solve is iterative, and at 512x512
    each evaluation touches 262k probes for cells that are ~1000 pixels across. 256 keeps ~64
    probes per cell, which bounds the centroid error near 1e-4.

    Accuracy limit, measured: for displacements of ~5e-3 and above (WVS, GBN, ours) the value
    agrees with a 512 solve to better than ~2%. For a point set already very close to the
    optimum the discretisation error becomes a visible share of a very small number -- BNOT
    measures ~1e-3 here and shifts ~10% between 256 and 512. Rankings are unaffected, since the
    gaps between methods are several-fold, but do not read fine differences among near-optimal
    point sets at this resolution; raise it if that case matters.
    """
    import cv2
    rho = _to_density(image_01)
    if resolution and max(rho.shape) > resolution:
        rho = cv2.resize(rho, (resolution, resolution), interpolation=cv2.INTER_AREA)
    pts = np.clip(np.asarray(points, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    pd = PowerDiagram(pts, rho, shortlist=shortlist)
    w, info = pd.solve_weights(maxiter=maxiter, bound_scale=bound_scale)
    return pd, w, info


# ============================================================================
# M1  -  CVT ENERGY
# ============================================================================

def _cvt_energy_mc(points, image_01, rng=None):
    """Legacy Monte-Carlo CVT energy: scipy Voronoi, polygons clipped to the unit
    square, <=50 bbox-rejection probes per cell. Retained so the previously published
    approximate numbers can be reproduced; `compute_m1_v1_cvt_energy` is exact by default.
    """
    try:
        from scipy.spatial import Voronoi
        if rng is None:
            rng = np.random.default_rng(42)
        N = len(points)
        if N < 3:
            return {"cvt_energy": 0.0}
        rho = _to_density(image_01)
        pts_clip = np.clip(points, 1e-6, 1 - 1e-6)
        try:
            vor = Voronoi(pts_clip)
        except Exception:
            return {"cvt_energy": 0.0}
        H, W = rho.shape
        # Clip cells to the unit square so boundary cells contribute their true
        # (bounded) energy instead of being silently dropped.
        clipped_regions = _voronoi_regions_clipped(vor, bbox=(0.0, 1.0, 0.0, 1.0))
        total_energy = 0.0
        for pt_idx, (poly, _is_finite) in enumerate(clipped_regions):
            try:
                if poly.shape[0] < 3:
                    continue
                point = points[pt_idx]
                sample = _polygon_sample_points(poly, (H, W), mc_approx=True, rng=rng)
                if sample is None:
                    continue
                points_inside, px, py = sample
                area = _polygon_area(poly)
                n_in = len(points_inside)
                dist_sq = (points_inside[:, 0] - point[0]) ** 2 + (points_inside[:, 1] - point[1]) ** 2
                if True:
                    # MC estimate of  integral_cell rho * ||x - s||^2 dx
                    # = area * mean( rho(x) * ||x - s||^2 ) over uniform samples.
                    contrib = area * float(np.mean(rho[py, px] * dist_sq)) if n_in > 0 else 0.0
                else:
                    # Pixel-grid quadrature: each pixel has area (1/(W*H)).
                    contrib = float(np.sum(rho[py, px] * dist_sq)) / float(W * H)
                total_energy += contrib
            except Exception:
                continue
        return {"cvt_energy": float(np.clip(total_energy, 0, 1000))}
    except Exception as exc:
        _warn_metric_failure("M1 compute_m1_cvt_energy", exc)
        return {"cvt_energy": 0.0}


def compute_m1_v1_cvt_energy(points, image_01, rng=None, mc_approx=False, **mc_args):
    """M1 -- density-weighted CVT energy, E = sum_i \\int_{V_i} rho(x) ||x - s_i||^2 dx.

    Computed exactly by pixel quadrature: a Voronoi cell is by definition the set of points
    closest to its site, so assigning every pixel to its nearest site and accumulating
    rho * ||x - s||^2 * pixel_area evaluates the integral directly. This replaces the former
    scipy-Voronoi + polygon-clipping + per-cell Monte-Carlo path, which drew at most 50
    bbox-rejection probes per cell and could silently drop degenerate cells.

    Set mc_approx=True to fall back to `_cvt_energy_mc`, the legacy Monte-Carlo estimator,
    for reproducing previously published approximate numbers. The default is exact and
    deterministic, and `rng` then has no effect.
    """
    if mc_approx:
        return _cvt_energy_mc(points, image_01, rng=rng)
    try:
        from scipy.spatial import cKDTree
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 3:
            return {"cvt_energy": 0.0}
        rho = _to_density(image_01)
        H, W = rho.shape
        ys, xs = np.mgrid[0:H, 0:W]
        ys, xs = ys.ravel(), xs.ravel()
        probe = np.stack([(xs + 0.5) / float(W), (ys + 0.5) / float(H)], axis=1)
        d, _ = cKDTree(pts).query(probe, k=1)
        energy = float(np.sum(rho[ys, xs] * d ** 2) / float(H * W))
        return {"cvt_energy": float(np.clip(energy, 0, 1000))}
    except Exception as exc:
        _warn_metric_failure("M1 compute_m1_cvt_energy", exc)
        return {"cvt_energy": 0.0}


# ============================================================================
# M2  -  CAPACITY
# ============================================================================

def compute_m2_capacity_constraint(points, image_01, rng=None, mc_approx=True):
    """M2_v1 (legacy) -- capacity deviation over VORONOI cells, kept as the original metric.

    Retained deliberately as a backup/continuity column. Note two things when reading it:

    * With mc_approx=True (this function's own default, so standalone callers keep their
      historical behaviour) it reproduces the previously published numbers exactly, using
      <=50 bbox-rejection probes per cell. That estimator is noisy and biased high -- on the
      real Icons-50 outputs it overstates the deviation by roughly 3-8x.
    * With mc_approx=False it measures the same quantity as
      `compute_m2_v3_cap_capacity_deviation` with weight_mode="none" -- the two agree to
      corr = 0.99988 (mean relative difference 2.9%, pure quadrature). Prefer M2_v3 there:
      it enumerates pixels directly instead of clipping polygons, so it is both exact and
      faster.

    `compute_all_advanced_metrics` propagates its own `mc_approx` here, so the reported
    battery is internally consistent.
    """
    try:
        from scipy.spatial import Voronoi

        # NOTE: this used to call warnings.filterwarnings("ignore") with no scope, which
        # permanently silenced EVERY warning in the process as a side effect of computing one
        # metric -- including numpy/scipy numerical warnings from unrelated code later in the
        # run. Qhull's chatter is suppressed locally instead.
        if rng is None:
            rng = np.random.default_rng(42)

        zero = {"voronoi_mass_cv": 0.0, "voronoi_mass_std": 0.0, "voronoi_mass_mean": 0.0}

        N = len(points)
        if N < 3:
            return dict(zero)

        rho = _to_density(image_01)

        pts_clip = np.clip(points, 1e-6, 1 - 1e-6)
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")          # scoped: Qhull chatter only
                vor = Voronoi(pts_clip)
        except Exception:
            return dict(zero)

        # Clip every Voronoi cell (including unbounded boundary cells) to the
        # unit square so the cells form a true partition of Omega. This is
        # required for the c* = mean(c(s_i)) identity in the document to hold.
        clipped_regions = _voronoi_regions_clipped(vor, bbox=(0.0, 1.0, 0.0, 1.0))

        H_img, W_img = rho.shape
        capacities = np.full(N, np.nan)

        for pt_idx, (poly, _is_finite) in enumerate(clipped_regions):
            try:
                if poly.shape[0] < 3:
                    continue
                sample = _polygon_sample_points(poly, (H_img, W_img), mc_approx=mc_approx, rng=rng)
                if sample is None:
                    continue
                _, px, py = sample
                # capacity c(s_i) = integral of rho over the cell.
                # mean(rho over cell) * cell_area is an unbiased MC estimate of
                # that integral; using the mean (not sum) makes capacity
                # independent of how many sample pixels happen to fall in the cell.
                mean_rho = float(rho[py, px].mean())
                area = _polygon_area(poly)
                capacities[pt_idx] = mean_rho * area
            except Exception:
                continue

        valid = np.isfinite(capacities)
        if valid.sum() < 2:
            return dict(zero)

        c = capacities[valid]
        c = np.maximum(c, 0.0)
        c_star = c.mean()  # c* = (1/n) sum c(s_i) over the partition
        if c_star < 1e-12:
            return dict(zero)

        # delta_c = (1/n) sum (c(s_i)/c* - 1)^2  == squared coefficient of variation
        delta_c = float(np.mean((c / c_star - 1.0) ** 2))

        return {
            "voronoi_mass_cv": float(np.clip(delta_c, 0, 10)),  # squared CV (delta_c)
            "voronoi_mass_std": float(c.std()),
            "voronoi_mass_mean": float(c_star),
        }
    except Exception as exc:
        _warn_metric_failure("M2_v1 compute_m2_capacity_constraint", exc)
        return {"voronoi_mass_cv": 0.0, "voronoi_mass_std": 0.0, "voronoi_mass_mean": 0.0}


def compute_m2_v1_capacity(points, image_01, weight_mode="none", k=8):
    """M2_v3 -- capacity deviation delta_c on power cells, per BNOT Eq. (1) and Sec. 4.4.

    weight_mode:
      "none"  (default) w = 0. All weights equal, so the power diagram coincides with the
              ordinary Voronoi diagram (BNOT Sec. 2.3) and delta_c reduces to Balzer et al.'s
              capacity deviation verbatim. Exact on the analytic ground truth (a regular
              lattice over a uniform density scores exactly 0), free of tuning parameters,
              and free of the boundary bias described in `_knn_power_weights`.
      "knn"   power weights w_i = r_k^2/k from the local point spacing -- the rule the
              superseded M2_v2/M2_v3 path used. Tracks "none" within ~15% but is not exact
              on ground truth; kept for reproducing the earlier cell model.

    A note for interpreting the comparison: BNOT SOLVES for weights that equalise its own
    capacities, reaching a residual of ~1e-12 by construction (BNOT Sec. 4.2). Those weights
    are part of BNOT's output, and WVS / GBN / a learned sampler produce none. Scoring each
    method under its own optimal weights would therefore compare a different quantity per
    column, and measurably washes out the differences the metric exists to show. This function
    applies one disclosed rule identically to every method; BNOT's number must be read with
    the caveat that it is not the partition BNOT optimises.

    Returns delta_c, the raw std of the cell masses, and the global mean capacity m.
    """
    zero = {"capacity_delta_c": 0.0, "capacity_std": 0.0, "capacity_mean": 0.0}
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return dict(zero)
    rho = _to_density(image_01)
    if rho.ndim != 2:
        raise ValueError(f"compute_m2_v1_capacity: expected a 2-D density, got {rho.shape}")
    if rho.sum() <= 1e-12:
        raise ValueError("compute_m2_v1_capacity: target density carries no mass "
                         "(pure-white image); there is nothing to distribute.")

    if weight_mode == "knn":
        w = _knn_power_weights(pts, k=k)
    elif weight_mode == "none":
        w = np.zeros(n, dtype=np.float64)
    else:
        raise ValueError(f"weight_mode must be 'knn' or 'none'; got {weight_mode!r}")

    m = _power_cell_masses(pts, rho, w)
    # The cells partition the domain, so mean(m_i) == (1/n) * total mass == BNOT's m (Sec. 4.4).
    m_bar = m.mean()
    if m_bar <= 1e-15:
        return dict(zero)
    delta_c = float(np.mean((m / m_bar - 1.0) ** 2))
    return {
        "capacity_delta_c": float(np.clip(delta_c, 0, 10)),
        "capacity_std": float(m.std()),
        "capacity_mean": float(m_bar),
    }


def _power_cell_cap_cv(points, rho, k=8, shortlist=32):
    """Shared core for M2_v2. Returns (cap_cv, ratio_std).

    Power (Laguerre) cells with weights from the local spacing (w_i = r_k^2/k); each cell's
    integrated MASS is compared against the mass that the local density predicts, and cap_cv
    is the squared CV of that ratio.

    Previously this compared cell AREA rather than mass, which never read rho at all: a black,
    mid-grey and ramped interior all produced byte-identical scores. It also needed a support
    mask, because empty background inflated the outline cells' areas (3.26 -> 0.16). Both
    problems disappear once mass is integrated -- white background carries rho = 0 and so
    contributes nothing -- which is why the masked variant is gone (see module note on M2_v3).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < k + 2:
        return 0.0, 0.0
    from scipy.spatial import cKDTree
    eps = 1e-12
    k_eff = min(k, n - 1)
    dk = cKDTree(pts).query(pts, k=k_eff + 1)[0][:, k_eff]
    lam = k_eff / (np.pi * np.maximum(dk, eps) ** 2)     # local intensity
    w = (1.0 / np.maximum(lam, eps)) / np.pi             # r^2-scale weight
    w = w - w.mean()

    masses = _power_cell_masses(pts, rho, w, shortlist=shortlist)
    total = masses.sum()
    if total <= eps:
        return 0.0, 0.0
    expect = 1.0 / np.maximum(lam, eps)
    expect_n = expect / max(expect.sum(), eps)
    ratio = (masses / total) / np.maximum(expect_n, eps)
    r_star = ratio.mean()
    if r_star < eps:
        return 0.0, 0.0
    delta = float(np.mean((ratio / r_star - 1.0) ** 2))
    return float(np.clip(delta, 0, 10)), float(ratio.std())


def compute_m2_v3_knn_power_cell(points, image_01=None, k=8, **mc_args):
    """M2_v2 -- power-cell mass against the LOCALLY predicted mass.

    Merges the former M2_v2 (unmasked) and M2_v3 (masked) paths. They are provably the same
    measurement once mass replaces area: across 100 real Icons-50 images x 4 methods, the
    masked and unmasked values differed by exactly 0.0 (max |diff| = 0.000e+00), because the
    mask only ever removed pixels whose density was already zero.

    Interpretation, and why this is NOT the capacity metric: the yardstick here is per-point
    (each cell is judged against its own kNN density estimate), so this quantity is largely a
    measure of local partition regularity and is a weak discriminator between samplers --
    across the same 100 images its between-method spread is ~0.23x its within-method scatter.
    Use `compute_m2_v3_cap_capacity_deviation` for capacity adherence proper (separation 3.2x).

    `**mc_args` swallows the retired `rng` / `n_probe` / `mc_approx` arguments so old call
    sites keep working; the computation is now always exact.
    """
    if image_01 is None:
        raise ValueError("compute_m2_v3_knn_power_cell requires image_01 (grayscale in [0,1]); "
                         "the metric integrates the target density and cannot be computed "
                         "from points alone.")
    rho = _to_density(image_01)
    if rho.sum() <= 1e-12:
        # A blank target is a misconfiguration, not a perfect score. Raise rather than
        # return 0.0, matching compute_m2_v3_cap_capacity_deviation; batch callers isolate
        # this via _collect and simply omit the key.
        raise ValueError("compute_m2_v3_knn_power_cell: target density carries no mass "
                         "(pure-white image); there is nothing to distribute.")
    cv, std = _power_cell_cap_cv(points, rho, k=k)
    return {"power_cell_cap_cv": cv, "power_cell_cap_std": std}




def compute_m1_v2_power_cvt_energy(points, image_01, solution=None, **kw):
    """M1_v2 -- CVT energy under the capacity-optimal POWER partition.

    M1_v1 measures the CVT energy on Voronoi cells, which is precisely the objective Weighted
    Voronoi Stippling minimises, so WVS leads that column by construction. The power-diagram
    counterpart is the \\star_0-HOT_{2,2} energy that BNOT minimises (de Goes et al., Sec. 2.3),
    so reporting both gives each family its own ground rather than privileging one.

    Pass `solution` (from solve_power_diagram) to avoid re-solving the weights when M2_v2 has
    already computed them for the same input.
    """
    zero = {"power_cvt_energy": 0.0}
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return dict(zero)
    pd, w, info = solution if solution is not None else solve_power_diagram(points, image_01, **kw)
    if pd.total_mass <= 1e-15:
        raise ValueError("compute_m1_v2_power_cvt_energy: target density carries no mass.")
    _, _, energy = pd.cell_stats(w)
    return {"power_cvt_energy": float(np.clip(energy, 0, 1000))}


def compute_m2_v2_power_displacement(points, image_01, solution=None, **kw):
    """M2_v2 -- mean displacement from each point to its power-cell centroid.

    Why this and not "capacity under power cells": once the weights are solved, capacity
    deviation is ~0 for EVERY method by construction -- semi-discrete OT admits a solution for
    almost any configuration, so "does some weighting make this capacity-constrained?" is
    answered yes for all of them and the column ranks nothing (measured: ~4e-5 for all four
    samplers). Displacement does not collapse, because the weights equalise MASS while the
    positions are still not at the barycenters.

    What it means: at a critical point of BNOT's functional every point sits at the barycenter
    of its own power cell -- the position gradient is a natural extension of the CVT gradient
    and vanishes exactly there. So this measures how far a point set is from satisfying BNOT's
    own optimality condition, under the partition BNOT actually uses. That is the quantity
    reviewers asked for when they objected to judging BNOT on Voronoi cells.

    Reported alongside: the solver's residual, the fraction of weights pinned at the bound, and
    the count of stranded points -- all three are failure indicators that must stay visible,
    since a silently unconverged solve would otherwise look like a good score.
    """
    zero = {"power_displacement": 0.0, "power_displacement_norm": 0.0,
            "power_solver_residual": 0.0, "power_at_bound": 0.0, "power_stranded": 0.0}
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return dict(zero)
    pd, w, info = solution if solution is not None else solve_power_diagram(points, image_01, **kw)
    if pd.total_mass <= 1e-15:
        raise ValueError("compute_m2_v2_power_displacement: target density carries no mass.")

    _, centroid, _ = pd.cell_stats(w)
    disp = np.linalg.norm(centroid - pd.pts, axis=1)
    # Express displacement in units of the mean point spacing (1/sqrt(n)) so the value is
    # comparable across point budgets.
    return {
        "power_displacement": float(disp.mean()),
        "power_displacement_norm": float(disp.mean() * np.sqrt(n)),
        "power_solver_residual": float(info.get("residual_delta_c", float("nan"))),
        "power_at_bound": float(info.get("at_bound", 0.0)),
        "power_stranded": float(stranded_points(points, image_01).sum()),
    }


# ============================================================================
# M3  -  EARTH MOVER'S DISTANCE
# ============================================================================

def compute_m3_emd(points, target_points=None, image_01=None, rng=None, mc_approx=False, **mc_args):
    """M3 -- 2-Wasserstein distance from the points to the target density.

    Exact and deterministic by default: the target is built by mass-preserving block
    coarsening (`_coarsen_density`) and carries real rho weights, so `rng` has no effect.

    Set mc_approx=True for the legacy estimator -- 5000 grid points drawn with probability
    proportional to rho, then weighted uniformly -- which reproduces the previously
    published numbers and does depend on `rng`.

    The old non-MC branch is deliberately not reproducible: it took a regular stride
    subsample but kept uniform weights, discarding rho entirely and measuring transport to
    a uniform rectangle (~14x error on real icons). That was a bug, not an approximation.
    """
    try:
        import ot
        target_mass = None
        N = len(points)
        if N < 2:
            return {"emd_distance": 0.0}
        if target_points is None:
            if image_01 is None:
                target_points = points.copy()
            else:
                rho = _to_density(image_01)
                H, W = rho.shape
                yy, xx = np.mgrid[0:H, 0:W]
                grid_points = np.column_stack([(xx.ravel() + 0.5) / W, (yy.ravel() + 0.5) / H])
                density_weights = rho.ravel().astype(np.float64)
                density_sum = density_weights.sum()
                if density_sum <= 1e-12:
                    return {"emd_distance": 0.0}
                density_weights = density_weights / density_sum
                if mc_approx:
                    # Legacy estimator: draw 5000 grid points with probability proportional
                    # to rho, then weight them uniformly. Unbiased but seed-dependent.
                    if rng is None:
                        rng = np.random.default_rng(42)
                    if len(grid_points) > 5000:
                        idx = rng.choice(len(grid_points), 5000, replace=False, p=density_weights)
                        target_points = grid_points[idx]
                    else:
                        target_points = grid_points
                else:
                    # Mass-preserving deterministic quadrature. NOTE: the old non-MC branch
                    # took a regular stride subsample but kept UNIFORM weights, silently
                    # discarding rho and measuring transport to a uniform rectangle (~14x
                    # error on real icons). It is not reproducible here because it was a bug,
                    # not a valid approximation of this quantity.
                    target_points, target_mass = _coarsen_density(rho, budget=8000)
        if len(target_points) == 0:
            return {"emd_distance": 0.0}
        # W_2 uses squared-Euclidean ground cost; the distance is the sqrt of
        # the optimal transport cost.  (metric="euclidean" would give W_1.)
        M = ot.dist(points, target_points, metric="sqeuclidean")
        source_w = np.ones(len(points)) / len(points)
        if target_mass is None:
            target_w = np.ones(len(target_points)) / len(target_points)
        else:
            target_w = target_mass / target_mass.sum()
        try:
            # POT's default numItermax (100k) is enough for well-formed stipple outputs but
            # not for point sets far from the target -- exactly what an early-epoch ablation
            # checkpoint produces. There the network simplex stops early and overstates the
            # distance by ~0.5% (0.199177 vs the converged 0.198118) while warning. 1e6
            # converges those cases for ~0.5s more and leaves normal inputs untouched.
            cost = ot.emd2(source_w, target_w, M, numItermax=1_000_000)
            emd = float(np.sqrt(max(cost, 0.0)))
        except Exception:
            emd = 0.0
        return {"emd_distance": float(np.clip(emd, 0, 100))}
    except Exception as exc:
        _warn_metric_failure("M3 compute_m3_emd", exc)
        return {"emd_distance": 0.0}


# ============================================================================
# M4  -  SINKHORN DISTANCE
# ============================================================================

def compute_m4_sinkhorn(points, image_01, target_density=None, rng=None, reg=0.01,
                        mc_approx=False, **mc_args):
    """M4 -- entropy-regularised Sinkhorn distance to the target density.

    Exact and deterministic by default: the target is built by mass-preserving block
    coarsening. Set mc_approx=True for the legacy estimator (10k random draws with
    probability proportional to rho, uniform weights), which depends on `rng`.
    """
    try:
        import ot
        if rng is None:
            rng = np.random.default_rng(42)
        # Use the same target density convention as every other metric: dark = high mass.
        if target_density is None:
            target_density = _to_density(image_01)
        else:
            target_density = np.asarray(target_density, dtype=np.float64)
        N = len(points)
        H, W = image_01.shape
        if N == 0:
            return {"sinkhorn_ot_cost": 0.0, "sinkhorn_transport_cost": 0.0}
        if mc_approx:
            # Legacy estimator: 10k random draws with p ~ rho, then uniform weights.
            yy, xx = np.mgrid[0:H, 0:W]
            grid_points = np.column_stack([(xx.ravel() + 0.5) / W, (yy.ravel() + 0.5) / H])
            density_weights = target_density.ravel() / (target_density.sum() + 1e-8)
            density_weights = np.maximum(density_weights, 1e-10)
            density_weights /= density_weights.sum()
            if len(grid_points) > 10000:
                idx = rng.choice(len(grid_points), 10000, p=density_weights)
                target_pts = grid_points[idx]
                target_w = np.ones(len(target_pts)) / len(target_pts)
            else:
                target_pts = grid_points
                target_w = density_weights
        else:
            # Deterministic mass-preserving quadrature.
            target_pts, target_mass = _coarsen_density(target_density, budget=8000)
            target_w = target_mass / target_mass.sum()
        source_w = np.ones(N) / N
        # W_2-consistent ground cost: squared Euclidean.
        M = ot.dist(points, target_pts, metric="sqeuclidean")
        try:
            # numItermax was 200, which tripped POT's "did not converge" warning on ~73% of
            # real samples. That warning was previously invisible because M2_v1 globally
            # disabled warnings as a side effect. It is not an accuracy problem: at 200
            # iterations the plan already matches its marginals to ~2e-16 (source) and ~1e-7
            # (target), and the reported cost is identical to 6 decimals at 200, 2000 and
            # 50000 iterations, and under log-domain stabilisation. 2000 clears POT's
            # iteration-delta threshold for ~0.1s more per call, so the run log stays clean
            # and a genuine warning is not lost in the noise.
            P = ot.sinkhorn(source_w, target_w, M, reg=reg, numItermax=2000)
            transport_cost = float(np.sum(P * M))
            # d_lambda = <P, M> - lambda * H(P), with H(P) = -sum P log P
            # (Cuturi 2013). lambda is the regularization weight `reg`.
            nz = P > 1e-300
            entropy = float(-np.sum(P[nz] * np.log(P[nz])))
            d_lambda = transport_cost - reg * entropy
            # Report sqrt of the (non-negative) transport cost as the W_2-style
            # distance for interpretability; keep raw d_lambda too.
            sinkhorn_distance = float(np.sqrt(max(transport_cost, 0.0)))
        except Exception:
            transport_cost = 0.0
            d_lambda = 0.0
            sinkhorn_distance = 0.0
        return {
            "sinkhorn_ot_cost": float(np.clip(sinkhorn_distance, 0, 100)),
            "sinkhorn_transport_cost": float(np.clip(transport_cost, 0, 100)),
            "sinkhorn_d_lambda": float(np.clip(d_lambda, -100, 100)),
        }
    except Exception as exc:
        _warn_metric_failure("M4 compute_m4_sinkhorn", exc)
        return {"sinkhorn_ot_cost": 0.0, "sinkhorn_transport_cost": 0.0}


# ============================================================================
# M5  -  SPATIAL MEASURE
# ============================================================================

def compute_m5_spatial_measure(points, image_01):
    try:
        from scipy.spatial import cKDTree
        N = len(points)
        if N < 2:
            return {
                "spatial_measure_rho_mean": 0.0,
                "spatial_measure_rho_cv": 0.0
            }

        points = np.asarray(points, dtype=np.float64)
        rho_img = _to_density(image_01)  # dark = high target density
        H, W = rho_img.shape

        # Local target density at each sample (vectorized).
        gx = np.clip((points[:, 0] * (W - 1)).astype(int), 0, W - 1)
        gy = np.clip((points[:, 1] * (H - 1)).astype(int), 0, H - 1)
        local_density = rho_img[gy, gx]

        mean_density = local_density.mean()
        if mean_density < 1e-10:
            mean_density = 1.0
        # Expected number of points per unit area at each location.
        D_x = N * (local_density / (mean_density + 1e-10))
        D_x_safe = np.maximum(D_x, 1e-8)

        # Density-transformed differential-domain spacing (Wei & Wang 2011):
        # the raw Euclidean NN distance is scaled by the local distance field,
        # which is inversely proportional to sqrt(local density). This makes the
        # measure scale-invariant under non-uniform target densities.
        tree = cKDTree(points)
        nn_dists, _ = tree.query(points, k=2)
        raw_rmin = nn_dists[:, 1]
        # local distance field ~ 1 / sqrt(D_x); warp by dividing raw distance by it
        # => warped r_min = raw_rmin * sqrt(D_x).
        r_min = raw_rmin * np.sqrt(D_x_safe)

        # Maximal hexagonal packing distance for unit-density domain (D normalized out
        # by the warp above, so r_max is the constant packing distance at density 1).
        r_max = np.sqrt(2.0 / np.sqrt(3.0))

        rho_values = np.clip(r_min / (r_max + 1e-10), 0.0, 2.0)
        rho_mean = float(rho_values.mean())
        rho_std = float(rho_values.std())
        rho_cv = rho_std / (rho_mean + 1e-8)

        return {
            "spatial_measure_rho_mean": rho_mean,
            "spatial_measure_rho_cv": float(np.clip(rho_cv, 0, 10))
        }
    except Exception:
        return {
            "spatial_measure_rho_mean": 0.0,
            "spatial_measure_rho_cv": 0.0
        }


# ============================================================================
# AGGREGATE
# ============================================================================

def _format_advanced_text(metrics):
    """Format M1-M5 metrics as a compact monospace string for text axes."""
    return (
        f"M1v1 CVT (Voronoi)    : {metrics.get('M1_v1_cvt_energy', 0.0):.6f}\n"
        f"M1v2 CVT (power)      : {metrics.get('M1_v2_power_cvt_energy', 0.0):.6f}\n"
        f"M2v1 Capacity delta_c : {metrics.get('M2_v1_capacity_delta_c', 0.0):.4f}\n"
        f"M2v2 Power displace.  : {metrics.get('M2_v2_power_displacement', 0.0):.4f}\n"
        f"M2v3 kNN power cell   : {metrics.get('M2_v3_power_cell_cap_cv', 0.0):.4f}\n"
        f"M3 EMD                : {metrics.get('M3_emd_distance', 0.0):.4f}\n"
        f"M4 Sinkhorn Distance  : {metrics.get('M4_sinkhorn_ot_cost', 0.0):.4f}\n"
        f"M5 Spatial Measure    : {metrics.get('M5_spatial_measure_rho_mean', 0.0):.4f}"
    )


def compute_all_advanced_metrics(points, image_01, image_input_u8=None, mc_approx=False,
                                 power_metrics=True, power_resolution=POWER_RESOLUTION):
    """Run the full M1-M5 battery. Exact and deterministic unless mc_approx=True.

    Individual metric failures are isolated (see `_collect`): the remaining metrics are still
    reported, and the caller can detect the gap because the key is absent.
    """
    result = {}

    _collect(result, "M1_v1_", compute_m1_v1_cvt_energy, points, image_01,
             rng=np.random.default_rng(41), mc_approx=mc_approx)
    _collect(result, "M2_v0_", compute_m2_capacity_constraint, points, image_01,
             rng=np.random.default_rng(42), mc_approx=mc_approx)
    _collect(result, "M2_v1_", compute_m2_v1_capacity, points, image_01)
    _collect(result, "M2_v3_", compute_m2_v3_knn_power_cell, points, image_01)

    # M1_v2 and M2_v2 share one capacity-optimal power diagram: the weight solve is the
    # expensive part and solving it twice for the same input would double the cost.
    if power_metrics:
        try:
            solution = solve_power_diagram(points, image_01, resolution=power_resolution)
        except Exception as exc:
            import warnings
            warnings.warn(f"power-diagram solve failed ({type(exc).__name__}: {exc}); "
                          f"omitting M1_v2_* and M2_v2_* for this sample",
                          RuntimeWarning, stacklevel=2)
        else:
            _collect(result, "M1_v2_", compute_m1_v2_power_cvt_energy, points, image_01,
                     solution=solution)
            _collect(result, "M2_v2_", compute_m2_v2_power_displacement, points, image_01,
                     solution=solution)
    _collect(result, "M3_", compute_m3_emd, points, None, image_01,
             rng=np.random.default_rng(43), mc_approx=mc_approx)
    _collect(result, "M4_", compute_m4_sinkhorn, points, image_01,
             rng=np.random.default_rng(44), mc_approx=mc_approx)
    _collect(result, "M5_", compute_m5_spatial_measure, points, image_01)
    return result


# ============================================================================
# GBN DENSITY RECONSTRUCTION
# ============================================================================

if HAS_TORCH:
    @torch.no_grad()
    def gbn_algorithm_2(points, true_sigma, iterations=10):
        import torch as _torch
        N = points.shape[0]
        device = points.device
        dist_sq = _torch.cdist(points, points) ** 2
        self_mask = _torch.eye(N, device=device, dtype=_torch.bool)
        a = _torch.ones(N, device=device, dtype=_torch.float32)
        for _ in range(iterations):
            a_l = a.unsqueeze(0)
            exponent = -a_l * dist_sq / (2 * true_sigma ** 2)
            exponent = exponent.masked_fill(self_mask, -float("inf"))
            d_k = (a_l * _torch.exp(exponent)).sum(dim=1)
            a = 0.5 * a + 0.5 * d_k
            mean_sq = (a ** 2).mean()
            a = a / _torch.sqrt(mean_sq + 1e-8)
        return a

    @torch.no_grad()
    def reconstruct_density_map(samples, grid_size=(512, 512), alpha=1.0, iterations=10, batch_size=8192, **kwargs):
        import torch as _torch
        device = samples.device
        H, W = grid_size
        N = samples.shape[0]
        true_sigma = alpha / (N ** 0.5)
        a_k = gbn_algorithm_2(samples, true_sigma, iterations=iterations)
        y = _torch.linspace(0, 1, H, device=device)
        x = _torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = _torch.meshgrid(y, x, indexing="ij")
        grid_coords = _torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        density_map = _torch.zeros(H * W, device=device)
        a_k_row = a_k.unsqueeze(0)
        for i in range(0, H * W, batch_size):
            batch_coords = grid_coords[i : i + batch_size]
            dist_sq = _torch.cdist(batch_coords, samples) ** 2
            weights = a_k_row * _torch.exp(-a_k_row * dist_sq / (2 * true_sigma ** 2))
            density_map[i : i + batch_size] = weights.sum(dim=1)
        density_map = density_map.view(H, W)
        d_min, d_max = density_map.min(), density_map.max()
        if d_max - d_min > 1e-4:
            density_map = (density_map - d_min) / (d_max - d_min)
        else:
            density_map = _torch.zeros_like(density_map)
        return density_map
else:
    def gbn_algorithm_2(*args, **kwargs):
        raise ImportError("gbn_algorithm_2 requires PyTorch; torch not available in this environment")

    def reconstruct_density_map(*args, **kwargs):
        raise ImportError("reconstruct_density_map requires PyTorch; torch not available in this environment")


def process_target_density(source_img_u8, device):
    import torch as _torch
    import torchvision.transforms.functional as TF
    if source_img_u8.ndim == 3:
        img_float = _torch.tensor(source_img_u8, dtype=_torch.float32).mean(dim=-1) / 255.0
    else:
        img_float = _torch.tensor(source_img_u8, dtype=_torch.float32) / 255.0
    img_float = img_float.to(device)
    density_map = 1.0 - img_float
    density_map = density_map.unsqueeze(0).unsqueeze(0)
    smoothed = TF.gaussian_blur(density_map, kernel_size=[5, 5], sigma=[1.0, 1.0])
    smoothed = smoothed.squeeze()
    d_min, d_max = smoothed.min(), smoothed.max()
    if d_max > d_min:
        smoothed = (smoothed - d_min) / (d_max - d_min)
    return smoothed


# ============================================================================
# VISUALIZATION
# ============================================================================

def _render_advanced_metrics_row(axes_row, points_list, labels, image_01, metrics_list=None, mc_approx=True):
    """Fill one matplotlib axes row with M1-M5 metric text boxes."""
    axes_row[0].axis("off")
    for j, (pts, label) in enumerate(zip(points_list, labels)):
        ax = axes_row[1 + j]
        ax.axis("off")
        try:
            if metrics_list is not None and j < len(metrics_list) and metrics_list[j] is not None:
                m = metrics_list[j]
            else:
                m = compute_all_advanced_metrics(pts, image_01, mc_approx=mc_approx)
            text = _format_advanced_text(m)
        except Exception:
            text = "(metrics unavailable)"
        ax.text(
            0.5, 0.95, text,
            ha="center", va="top",
            fontsize=_fs(),
            fontfamily="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.85),
        )
        ax.set_title(f"{label} Adv. Metrics", fontsize=_fs())


def visualize_adaptive_sampling_density_map(
    source_img_u8,
    pred_pointsets,
    save_path,
    gt_points=None,
    grid_size=(512, 512),
    k=4,
    alpha=1.0,
    device=None,
    pred_labels=None,
):
    if not HAS_MPL:
        return None

    try:
        import torch as _torch
    except ImportError:
        print("Warning: torch not available, skipping AKDE visualisation")
        return None

    _device = _torch.device(device) if device else _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    target_density = process_target_density(source_img_u8, _device).cpu().numpy()

    try:
        from PIL import Image as _PILImage
        source_resized = np.array(
            _PILImage.fromarray(source_img_u8).resize((grid_size[1], grid_size[0]), _PILImage.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    except Exception:
        from scipy.ndimage import zoom as _zoom
        h, w = source_img_u8.shape
        source_resized = _zoom(source_img_u8.astype(np.float32) / 255.0,
                               (grid_size[0] / h, grid_size[1] / w), order=1)

    columns = [("Input", None, source_resized)]

    if gt_points is not None:
        gt_np = np.asarray(gt_points, dtype=np.float32)
        gt_map = reconstruct_density_map(_torch.from_numpy(gt_np).to(_device),
                                         grid_size=grid_size, alpha=alpha)
        columns.append(("GT", gt_np, gt_map.cpu().numpy()))

    for i, pts in enumerate(pred_pointsets):
        label = pred_labels[i] if (pred_labels and i < len(pred_labels)) else f"Pred {i + 1}"
        pts_np = np.asarray(pts, dtype=np.float32)
        pred_map = reconstruct_density_map(_torch.from_numpy(pts_np).to(_device),
                                           grid_size=grid_size, alpha=alpha)
        columns.append((label, pts_np, pred_map.cpu().numpy()))

    n_cols = len(columns)
    if n_cols == 0:
        return None

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9.0))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, (title, points, density) in enumerate(columns):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        if title == "Input":
            ax_top.imshow(source_resized, cmap="gray", vmin=0, vmax=1, origin="upper", aspect="equal")
            ax_top.set_title("Input Image", fontsize=_fs(), fontweight="bold", pad=6)
            ax_top.axis("off")
            ax_bot.imshow(target_density, cmap="gray_r", vmin=0, vmax=1, origin="upper", aspect="equal")
            ax_bot.set_title("Target Density Field", fontsize=_fs(), fontweight="bold", pad=6)
            ax_bot.axis("off")
            continue

        ax_top.scatter(points[:, 0], points[:, 1], s=0.5, c="black", marker=".")
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.invert_yaxis()
        ax_top.set_aspect("equal")
        ax_top.set_title(f"{title} Points", fontsize=_fs(), fontweight="bold", pad=6)
        ax_top.axis("off")

        ax_bot.imshow(density, cmap="gray_r", vmin=0, vmax=1, origin="upper", aspect="equal")
        ax_bot.set_title(f"{title} Density", fontsize=_fs(), fontweight="bold", pad=6)
        ax_bot.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    return save_path


def visualize_overfit_metrics(
    source_img,
    target_img,
    gt_points,
    pred_pointsets,
    save_path,
    step=None,
    point_size=0.5,
    gt_offsets=None,
    capacity_grid_size=16,
    pred_labels=None,
    compute_advanced=False,
    mc_approx=True,
):
    if not HAS_MPL:
        return None

    plt.rcParams.update({
        'font.size': _fs(),
        'axes.titlesize': _fs(),
        'axes.labelsize': _fs(),
        'xtick.labelsize': _fs(),
        'ytick.labelsize': _fs()
    })

    n_preds = min(len(pred_pointsets), 4)
    n_cols = 2 + n_preds

    if compute_advanced:
        fig, axes = plt.subplots(
            4, n_cols,
            figsize=(4.5 * n_cols, 4.5 * 3 + _ADV_ROW_EXTRA_HEIGHT),
            gridspec_kw={"height_ratios": [3, _ADV_ROW_HEIGHT_RATIO, 3, 3]},
        )
    else:
        fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 4.5 * 3))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    step_label = f" (step {step})" if step is not None else ""
    image_01 = source_img.astype(np.float64) / 255.0

    ax = axes[0, 0]
    ax.imshow(source_img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Source)", fontsize=_fs())
    ax.axis("off")

    ax = axes[0, 1]
    ax.scatter(gt_points[:, 0], 1 - gt_points[:, 1], c="black", s=point_size, alpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.set_facecolor("white")
    ax.set_title("GT (Target)", fontsize=_fs())
    ax.axis("off")

    if pred_labels is None:
        pred_labels = [f"Predict {i}{step_label}" for i in range(n_preds)]
    else:
        pred_labels = [str(l) for l in pred_labels[:n_preds]]
        if len(pred_labels) < n_preds:
            pred_labels.extend(f"Predict {i}{step_label}" for i in range(len(pred_labels), n_preds))

    for i in range(n_preds):
        ax = axes[0, 2 + i]
        pts = pred_pointsets[i]
        ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=point_size, alpha=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.set_facecolor("white")
        ax.set_title(pred_labels[i], fontsize=_fs())
        ax.axis("off")

    all_points = [gt_points] + [pred_pointsets[i] for i in range(n_preds)]
    cap_grid_shape = resolve_capacity_grid_size(image_01, capacity_grid_size)
    all_cap = [compute_grid_capacity(p, image_01, grid_size=cap_grid_shape) for p in all_points]
    all_spa = [compute_spacing_quality(p) for p in all_points]

    cap_row = 2 if compute_advanced else 1
    spa_row = 3 if compute_advanced else 2

    if gt_offsets is not None:
        try:
            from data.Transforms import to_pointset_optimal_transport
            pts_grid = to_pointset_optimal_transport(gt_offsets)
            pts_ot = pts_grid.reshape(2, -1).T
            n = gt_offsets.shape[-1]
            clipped = np.clip(pts_ot, 0.0, 1.0 - 1e-12)
            ij = np.floor(clipped * n).astype(np.int64)
            counts = np.zeros((n, n), dtype=np.int32)
            for x_idx, y_idx in ij:
                counts[y_idx, x_idx] += 1
            binary_img = (counts > 0).astype(np.uint8) * 255
            axes[cap_row, 0].imshow(binary_img, cmap="gray", vmin=0, vmax=255, origin="upper")
            axes[cap_row, 0].set_title("GT Binary\n(32×32 OT grid)", fontsize=_fs())
            axes[cap_row, 0].axis("off")
        except Exception:
            axes[cap_row, 0].axis("off")
    else:
        axes[cap_row, 0].axis("off")

    col_labels = ["GT (Target)"] + pred_labels
    for j, (cap, label) in enumerate(zip(all_cap, col_labels)):
        ax = axes[cap_row, 1 + j]
        status = cap["grid_status"]
        H_g, W_g = status.shape
        rgb = np.zeros((H_g, W_g, 3), dtype=np.float32)
        rgb[status == 0, :] = [0.0, 1.0, 0.0]   # green = ok
        rgb[status == -1, :] = [1.0, 0.0, 0.0]  # red = underfilled/misplaced
        rgb[status == 1, :] = [0.0, 0.0, 1.0]   # blue = overfilled
        ax.imshow(rgb, origin="upper", aspect="equal")
        ok_pct = 100.0 - cap["underfilled_pct"] - cap["overfilled_pct"]
        ax.set_title(
            f"{label} Capacity\n"
            f"Grid:{cap_grid_shape[0]}x{cap_grid_shape[1]} | "
            f"OK:{ok_pct:.0f}% Under:{cap['underfilled_pct']:.0f}% Over:{cap['overfilled_pct']:.0f}%\n"
            f"Score: {cap['score']:.3f}",
            fontsize=_fs(),
        )
        ax.axis("off")

    if gt_offsets is not None:
        try:
            n = gt_offsets.shape[-1]
            yy, xx = np.mgrid[0:n, 0:n]
            dx, dy = gt_offsets[0], gt_offsets[1]
            mag = np.sqrt(dx * dx + dy * dy)
            ax = axes[spa_row, 0]
            q = ax.quiver(xx, yy, dx, dy, mag, angles="xy", scale_units="xy", scale=1.0,
                          cmap="viridis", width=0.004)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_title("GT Offset Quiver", fontsize=_fs())
            ax.set_xlabel("grid x", fontsize=_fs())
            ax.set_ylabel("grid y", fontsize=_fs())
            ax.tick_params(labelsize=_fs())
            fig.colorbar(q, ax=ax, shrink=0.7, label="|offset|")
        except Exception:
            axes[spa_row, 0].axis("off")
    else:
        axes[spa_row, 0].axis("off")

    all_nn = [s["nn_distances"] for s in all_spa]
    vmin = min(d.min() for d in all_nn)
    vmax = max(d.max() for d in all_nn)

    for j, (spa, pts, label) in enumerate(zip(all_spa, all_points, col_labels)):
        ax = axes[spa_row, 1 + j]
        sc = ax.scatter(pts[:, 0], 1 - pts[:, 1], c=spa["nn_distances"],
                        cmap="RdYlBu", s=point_size * 3, alpha=0.8, vmin=vmin, vmax=vmax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.set_facecolor("white")
        ax.set_title(
            f"{label} Spacing\n"
            f"CV:{spa['nn_cv']:.3f}  Clumped:{spa['clumped_pct']:.1f}%\n"
            f"Score: {spa['spacing_score']:.3f}",
            fontsize=_fs(),
        )
        ax.axis("off")
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("NN dist", fontsize=_fs())
        cbar.ax.tick_params(labelsize=_fs())

    if compute_advanced:
        adv_points = [gt_points] + [pred_pointsets[i] for i in range(n_preds)]
        _render_advanced_metrics_row(axes[1, :], adv_points, col_labels, image_01, mc_approx=mc_approx)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def visualize_advanced_metrics_panel(
    image_01,
    pred_pointsets,
    save_path,
    pred_labels=None,
    gt_points=None,
    gt_label="GT",
    mc_approx=True,
):
    if not HAS_MPL:
        return None

    plt.rcParams.update({
        'font.size': _fs(),
        'axes.titlesize': _fs(),
        'axes.labelsize': _fs(),
        'xtick.labelsize': _fs(),
        'ytick.labelsize': _fs()
    })

    n_preds = min(len(pred_pointsets), 4)
    all_pointsets = []
    all_labels = []

    if gt_points is not None:
        all_pointsets.append(np.asarray(gt_points))
        all_labels.append(str(gt_label))

    for i in range(n_preds):
        all_pointsets.append(np.asarray(pred_pointsets[i]))

    if pred_labels is None:
        all_labels.extend([f"Pred {i}" for i in range(n_preds)])
    else:
        pred_labels = [str(l) for l in pred_labels[:n_preds]]
        if len(pred_labels) < n_preds:
            pred_labels.extend(f"Pred {i}" for i in range(len(pred_labels), n_preds))
        all_labels.extend(pred_labels)

    n_cols = len(all_pointsets)
    if n_cols == 0:
        return None

    all_metrics = [compute_all_advanced_metrics(pts, image_01, mc_approx=mc_approx) for pts in all_pointsets]

    fig, axes = plt.subplots(5, n_cols, figsize=(5 * n_cols, 5 * 5))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    metrics_names = [
        ("M1", "CVT Energy",          "M1_cvt_energy"),
        ("M2", "Capacity Constraint", "M2_voronoi_mass_cv"),
        ("M2v2", "Power Cell Cap CV", "M2_v2_power_cell_cap_cv"),
        ("M3", "EMD",                 "M3_emd_distance"),
        ("M4", "Sinkhorn Distance",   "M4_sinkhorn_ot_cost"),
        ("M5", "Spatial Measure",     "M5_spatial_measure_rho_mean"),
    ]

    for m_idx, (m_short, m_long, m_key) in enumerate(metrics_names):
        for p_idx in range(n_cols):
            ax = axes[m_idx, p_idx]
            value = all_metrics[p_idx].get(m_key, 0.0)
            ax.bar([0], [value], color=plt.cm.viridis(m_idx / 5), width=0.5)
            ax.set_ylim(0, max(1.0, value * 1.2))
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_title(f"{all_labels[p_idx]}\n{m_short}: {value:.3f}", fontsize=_fs(), fontweight="bold")
            if p_idx == 0:
                ax.set_ylabel(m_long, fontsize=_fs())
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def _extract_subject_contour(density_map, bg_threshold=0.05):
    from scipy.ndimage import binary_fill_holes, gaussian_filter

    H, W = density_map.shape

    mask = density_map > bg_threshold
    mask = binary_fill_holes(mask)
    sigma = max(0.5, min(H, W) / 512.0)
    smooth = gaussian_filter(mask.astype(np.float32), sigma=sigma)

    try:
        fig_tmp, ax_tmp = plt.subplots(1, 1)
        cs = ax_tmp.contour(smooth, levels=[0.5])   
        plt.close(fig_tmp)

        try:
            all_paths = list(cs.get_paths())
        except AttributeError:
            all_paths = [p for coll in cs.collections for p in coll.get_paths()]

        if not all_paths:
            return None

        largest = max(all_paths, key=lambda p: len(p.vertices))
        verts = largest.vertices   
        return np.column_stack([verts[:, 0] / W, verts[:, 1] / H])
    except Exception:
        return None


def _extract_subject_path(density_map, bg_threshold=0.05):
    from scipy.ndimage import binary_fill_holes, gaussian_filter
    from matplotlib.path import Path as MplPath
    import matplotlib.pyplot as plt

    H, W = density_map.shape

    mask = density_map > bg_threshold
    mask = binary_fill_holes(mask)
    sigma = max(0.5, min(H, W) / 512.0)
    smooth = gaussian_filter(mask.astype(np.float32), sigma=sigma)

    try:
        fig_tmp, ax_tmp = plt.subplots(1, 1)
        cs = ax_tmp.contour(smooth, levels=[0.5])   
        plt.close(fig_tmp)

        try:
            all_paths = list(cs.get_paths())
        except AttributeError:
            all_paths = [p for coll in cs.collections for p in coll.get_paths()]

        if not all_paths:
            return None

        vertices = []
        codes = []
        
        # Combine all disjoint leaves/regions into a single master Path
        for p in all_paths:
            v = p.vertices.copy()
            v[:, 0] /= W
            v[:, 1] /= H
            
            c = p.codes
            if c is None:
                c = np.full(len(v), MplPath.LINETO)
                c[0] = MplPath.MOVETO
                c[-1] = MplPath.CLOSEPOLY
                
            vertices.append(v)
            codes.append(c)

        if not vertices:
            return None
            
        master_path = MplPath(np.vstack(vertices), np.concatenate(codes))
        return master_path
    except Exception:
        return None


def plot_visual_m2_capacity_constraint(points, image_01, ax, clip_to_domain=True, show_colorbar=True, rng=None, mc_approx=True):
    from matplotlib.patches import Polygon as MplPolygon, PathPatch
    from matplotlib.collections import PatchCollection
    from matplotlib.path import Path, Path as MplPath
    from matplotlib import colors as mcolors
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy.spatial import Voronoi

    plot_title = "[M2] Capacity Constraint (Voronoi Mass)"

    density_map = 1.0 - image_01

    if rng is None:
        rng = np.random.default_rng(42)

    H, W = image_01.shape
    pts_clip = np.clip(points, 1e-5, 1 - 1e-5)

    try:
        vor = Voronoi(pts_clip)
    except Exception:
        ax.set_title(f"{plot_title}\n(insufficient points)", fontsize=_fs())
        return

    def _cell_mass(verts):
        sample = _polygon_sample_points(verts, (H, W), mc_approx=mc_approx, rng=rng)
        if sample is None:
            return 0.0
        _, px, py = sample
        return float(density_map[py, px].sum())

    cell_polys, cell_masses = [], []
    for pt_idx in vor.point_region:
        region = vor.regions[pt_idx]
        if len(region) < 3 or -1 in region:
            continue
        verts = np.clip(vor.vertices[region], 0, 1)
        cell_polys.append(MplPolygon(verts))
        cell_masses.append(_cell_mass(verts))

    cell_masses = np.array(cell_masses)

    mean_mass = cell_masses.mean() if len(cell_masses) > 0 else 1.0
    deviations = np.clip((cell_masses - mean_mass) / (mean_mass + 1e-8), -1.0, 1.0)
    dev_count = int(len(deviations))

    under_thr = -0.15
    over_thr = 0.15
    bucket_edges = np.array([-1.0, under_thr, over_thr, 1.0], dtype=np.float32)
    bucket_colors = ListedColormap(["#2b6cb0", "#d9d9d9", "#c53030"])
    bucket_norm = BoundaryNorm(bucket_edges, bucket_colors.N)

    # --- FIX: Calculate robust 100%-summing display percentages ---
    if dev_count > 0:
        under_raw = float((deviations < under_thr).mean() * 100.0)
        avg_raw = float(((deviations >= under_thr) & (deviations <= over_thr)).mean() * 100.0)
        over_raw = float((deviations > over_thr).mean() * 100.0)
        
        # Largest Remainder Method (Hare-Niemeyer) to ensure exact 100% sum
        raw_pcts = [under_raw, avg_raw, over_raw]
        floors = [int(np.floor(p)) for p in raw_pcts]
        remainders = [p - f for p, f in zip(raw_pcts, floors)]
        diff = 100 - sum(floors)
        
        for i in np.argsort(remainders)[::-1][:diff]:
            floors[i] += 1
            
        u_disp, a_disp, o_disp = floors
    else:
        u_disp = a_disp = o_disp = 0
    # -------------------------------------------------------------

    ax.set_facecolor("white")

    if cell_polys:
        col = PatchCollection(
            cell_polys, cmap=bucket_colors, norm=bucket_norm, alpha=0.8, edgecolor="black", linewidth=0.2
        )
        col.set_array(deviations)

        if clip_to_domain:
            # V1 - NOT ALWAYS RELIABLE
            # contour_pts = _extract_subject_contour(density_map, bg_threshold=0.05)
            # if contour_pts is not None and len(contour_pts) >= 3:
            #     cp = np.vstack([contour_pts, contour_pts[0]])
            #     codes = (
            #         [MplPath.MOVETO]
            #         + [MplPath.LINETO] * (len(cp) - 2)
            #         + [MplPath.CLOSEPOLY]
            #     )
            #     clip_patch = PathPatch(
            #         MplPath(cp, codes), transform=ax.transData, visible=False
            #     )
            #     ax.add_patch(clip_patch)
            #     col.set_clip_path(clip_patch)
            #     closed = np.vstack([contour_pts, contour_pts[0]])
            #     ax.plot(closed[:, 0], closed[:, 1],
            #             color="black", linewidth=0.8, zorder=10, alpha=0.5)

            # V2 - MUCH MORE RELIABLE
            subject_path = _extract_subject_path(density_map, bg_threshold=0.05)
            if subject_path is not None:
                clip_patch = PathPatch(
                    subject_path, transform=ax.transData, visible=False
                )
                ax.add_patch(clip_patch)
                col.set_clip_path(clip_patch)
                
                boundary_patch = PathPatch(
                    subject_path, transform=ax.transData, 
                    facecolor="none", edgecolor="black", 
                    linewidth=0.8, alpha=0.5, zorder=10
                )
                ax.add_patch(boundary_patch)

        ax.add_collection(col)

        if show_colorbar:
            tick_positions = [
                (bucket_edges[0] + bucket_edges[1]) / 2.0,
                (bucket_edges[1] + bucket_edges[2]) / 2.0,
                (bucket_edges[2] + bucket_edges[3]) / 2.0,
            ]
            tick_labels = [
                f"Under\n{bucket_edges[0]:.2f} to {bucket_edges[1]:.2f}\n{u_disp}%",
                f"Avg\n{bucket_edges[1]:.2f} to {bucket_edges[2]:.2f}\n{a_disp}%",
                f"Over\n{bucket_edges[2]:.2f} to {bucket_edges[3]:.2f}\n{o_disp}%",
            ]
            cbar = plt.colorbar(col, ax=ax, fraction=0.046, pad=0.04, ticks=tick_positions)
            cbar.ax.set_yticklabels(tick_labels, fontsize=_fs())
            cbar.set_label("Voronoi mass deviation", fontsize=_fs())
            cbar.ax.tick_params(length=0, labelsize=_fs())
            
            # Key fix #2: Detach the colorbar layout constraint so it doesn't permanently shrink the axes
            ax.set_axes_locator(None)

    ax.scatter(pts_clip[:, 0], pts_clip[:, 1], c="black", s=0.5, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    
    if dev_count > 0:
        ax.set_title(f"{plot_title}", fontsize=_fs())

        # Key Fix #1: Render the texts fully separated at uniform horizontal spacing
        ax.text(0.166, -0.06, f"Under: {u_disp}%", va="top", ha="center", fontsize=_fs(), color="#2b6cb0", transform=ax.transAxes)
        ax.text(0.500, -0.06, f"Avg: {a_disp}%", va="top", ha="center", fontsize=_fs(), color="#555555", transform=ax.transAxes)
        ax.text(0.833, -0.06, f"Over: {o_disp}%", va="top", ha="center", fontsize=_fs(), color="#c53030", transform=ax.transAxes)
    else:
        ax.set_title(f"{plot_title}\n(insufficient points)", fontsize=_fs())


def plot_visual_m5_spatial_measure(points, image_01, ax, show_colorbar=True):
    from scipy.spatial import cKDTree

    plot_title = "[M5] Spatial Measure ρ (Hotspots)"

    N = len(points)
    if N < 2:
        ax.set_title(f"{plot_title}\n(insufficient points)", fontsize=_fs())
        return

    tree = cKDTree(points)
    nn_dists, _ = tree.query(points, k=2)
    raw_rmin = nn_dists[:, 1]

    H, W = image_01.shape
    gx = np.clip((points[:, 0] * (W - 1)).astype(int), 0, W - 1)
    gy = np.clip((points[:, 1] * (H - 1)).astype(int), 0, H - 1)
    local_img_density = 1.0 - image_01[gy, gx]
    
    mean_density = local_img_density.mean()
    if mean_density < 1e-10:
        mean_density = 1.0
    D_x = N * (local_img_density / (mean_density + 1e-10))
    
    D_x_safe = np.maximum(D_x, 1e-8)
    # Density-warped spacing, consistent with compute_m5_spatial_measure.
    r_min = raw_rmin * np.sqrt(D_x_safe)
    r_max = np.sqrt(2.0 / np.sqrt(3.0))
    
    rho_values = np.clip(r_min / (r_max + 1e-10), 0.0, 2.0)

    sc = ax.scatter(points[:, 0], points[:, 1], c=rho_values,
                    cmap="RdYlBu", s=2.0, vmin=0.4, vmax=1.1)
                    
    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("ρ (uniformity)", fontsize=_fs())
        cbar.ax.tick_params(labelsize=_fs())
        
        # Key fix #2: Detach the colorbar layout constraint so it doesn't permanently shrink the axes
        ax.set_axes_locator(None)
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{plot_title}", fontsize=_fs())


def plot_visual_m1_cvt_vectors(points, image_01, ax):
    from scipy.spatial import Voronoi, voronoi_plot_2d

    plot_title = "[M1] CVT Energy (Offsets Error)"

    pts_clip = np.clip(points, 1e-5, 1 - 1e-5)

    try:
        vor = Voronoi(pts_clip)
    except Exception:
        ax.set_title(f"{plot_title}\n(insufficient points)", fontsize=_fs())
        return

    voronoi_plot_2d(
        vor, ax=ax,
        show_points=False, show_vertices=False,
        line_colors="gray", line_width=0.3, line_alpha=0.5,
    )

    for region_idx, point_idx in enumerate(vor.point_region):
        region = vor.regions[point_idx]
        if len(region) < 3 or -1 in region:
            continue
        vertices = np.clip(vor.vertices[region], 0, 1)
        cx, cy = vertices[:, 0].mean(), vertices[:, 1].mean()
        px_pt, py_pt = points[region_idx]
        ax.plot([px_pt, cx], [py_pt, cy], color="red", linewidth=0.8, alpha=0.7)
        ax.scatter(cx, cy, color="red", marker="x", s=5, linewidths=0.5)

    ax.scatter(pts_clip[:, 0], pts_clip[:, 1], c="black", s=1.0, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{plot_title}", fontsize=_fs())


def plot_visual_m5_spectrum(points, image_01, ax):
    plot_title = "[M5] Power Spectrum"

    H, W = image_01.shape
    grid_size = max(H, W)

    point_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    x_idx = np.clip((points[:, 0] * grid_size).astype(int), 0, grid_size - 1)
    y_idx = np.clip((points[:, 1] * grid_size).astype(int), 0, grid_size - 1)
    point_grid[y_idx, x_idx] = 1.0

    power = np.abs(np.fft.fft2(point_grid)) ** 2

    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    rr = np.maximum(
        np.sqrt((xx - grid_size / 2) ** 2 + (yy - grid_size / 2) ** 2), 1.0
    )
    radial_bins = np.arange(1, grid_size // 2, dtype=np.float32)
    radial_power = np.array([
        power[(rr >= r) & (rr < r + 1)].mean() if ((rr >= r) & (rr < r + 1)).sum() > 0 else 0.0
        for r in radial_bins
    ])

    valid = (radial_bins > 0) & (radial_power > 1e-10)
    if valid.sum() > 2:
        log_f = np.log(radial_bins[valid])
        log_p = np.log(radial_power[valid])
        slope, intercept = np.polyfit(log_f, log_p, 1)
        ax.plot(radial_bins[valid], radial_power[valid],
                color="steelblue", alpha=0.8, linewidth=1.0, label="Spectrum")
        ax.plot(
            radial_bins[valid],
            np.exp(intercept) * (radial_bins[valid] ** slope),
            color="red", linestyle="--", linewidth=1.2, label=f"Fit (slope {slope:.2f})",
        )
        ax.legend(fontsize=_fs())
        ax.set_title(f"{plot_title}\nSlope: {slope:.3f}", fontsize=_fs())
    else:
        ax.set_title(f"{plot_title}\n(insufficient data)", fontsize=_fs())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.tick_params(axis="both", labelsize=_fs())
    ax.set_xlabel("Radial frequency", fontsize=_fs())
    ax.set_ylabel("Power", fontsize=_fs())


def visualize_spatial_metrics_panel(
    image_01,
    pred_pointsets,
    save_path,
    pred_labels=None,
    gt_points=None,
    gt_label="GT",
    clip_to_domain=True,
    show_colorbar=True,
):
    if not HAS_MPL:
        return None

    plt.rcParams.update({
        'font.size': _fs(),
        'axes.titlesize': _fs(),
        'axes.labelsize': _fs(),
        'xtick.labelsize': _fs(),
        'ytick.labelsize': _fs()
    })

    all_pointsets = []
    all_labels = []

    if gt_points is not None:
        all_pointsets.append(np.asarray(gt_points, dtype=np.float64))
        all_labels.append(str(gt_label))

    for i in range(len(pred_pointsets)):
        all_pointsets.append(np.asarray(pred_pointsets[i], dtype=np.float64))

    if pred_labels is None:
        all_labels.extend([f"Pred {i}" for i in range(len(pred_pointsets))])
    else:
        pred_labels = [str(l) for l in pred_labels[:len(pred_pointsets)]]
        if len(pred_labels) < len(pred_pointsets):
            pred_labels.extend(f"Pred {i}" for i in range(len(pred_labels), len(pred_pointsets)))
        all_labels.extend(pred_labels)

    n_cols = len(all_pointsets)
    if n_cols == 0:
        return None

    row_titles = [
        "[M1] CVT Energy (Offsets Error)",
        "[M2] Capacity Constraint (Voronoi Mass)",
        "[M5] Spatial Measure ρ (Hotspots)",
    ]
    plot_fns = [
        plot_visual_m1_cvt_vectors,
        plot_visual_m2_capacity_constraint,
        plot_visual_m5_spatial_measure,
    ]
    n_rows = len(plot_fns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, (pts, label) in enumerate(zip(all_pointsets, all_labels)):
        for row_idx, fn in enumerate(plot_fns):
            ax = axes[row_idx, col_idx]
            try:
                if fn is plot_visual_m2_capacity_constraint:
                    fn(pts, image_01, ax, clip_to_domain=clip_to_domain, show_colorbar=show_colorbar)
                elif fn is plot_visual_m5_spatial_measure:
                    fn(pts, image_01, ax, show_colorbar=show_colorbar)
                else:
                    fn(pts, image_01, ax)
            except Exception as exc:
                ax.axis("off")
                ax.set_title(f"{row_titles[row_idx]}\n(error)", fontsize=_fs())
            if row_idx == 0:
                current = ax.get_title()
                ax.set_title(f"[{label}]  {current}", fontsize=_fs())

    for row_idx, rt in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(rt, fontsize=_fs(), fontweight="bold", labelpad=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
