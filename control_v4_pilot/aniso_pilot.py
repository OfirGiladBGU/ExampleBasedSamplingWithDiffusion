"""aniso_pilot.py -- core machinery for the anisotropy teacher-data pilot.

Reusable building blocks shared by the milestones in anisotropy_teacher_pilot.md.
All coordinates live in [0,1]^2. Pure numpy + scipy + matplotlib; no torch.

  poisson_disk_bridson       clean isotropic blue-noise set P0 (Bridson 2007)
  radius_for_count           Poisson-disk radius for a target density
  metric_M                   area-preserving map M(theta,kappa), det M = 1
  warp_constant              global constant-anisotropy stretch (M1)
  warp_divfree_swirl         density-preserving swirl (M1)
  render_dots                dots-ONLY renderer (no size/shape/orientation cues)
  global_near_field_anisotropy  PRIMARY estimator (constant-theta tiles)
  nn_vector_anisotropy       independent cross-check (k=1 NN-vector covariance)
  fixed_radius_anisotropy / local_metric_anisotropy / local_double_angle
                             earlier estimators, BROKEN here (see note); reference

Estimator note -- why the naive estimators fail, and what works:
  The warp is AREA-PRESERVING, so point DENSITY is uniform. All anisotropy lives
  in the near-field: the elliptical first-shell exclusion zone. Two traps:
    (1) k-NN selection fills a round disk -> covariance/double-angle read
        isotropic regardless of the true spacing.
    (2) Averaging over many neighbours (2-3 shells) drowns the anisotropic
        first shell in the isotropic far field, and plain double-angle CANCELS
        (close perp neighbours 2phi~180 vs far theta neighbours 2phi~0).
  Fixes, both used by global_near_field_anisotropy:
    * weight neighbours by closeness (Gaussian, bandwidth ~ one spacing) so the
      near perp-neighbours dominate and the cancellation breaks;
    * sum COHERENTLY across the whole (constant-theta) tile so the isotropic
      floor averages toward zero instead of the per-point noise level.
  For a stretch along theta the compressed axis is perpendicular, so the
  measured double-angle / NN-covariance axis = theta + 90deg.
"""

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -- isotropic blue noise -------------------------------------------------

def poisson_disk_bridson(radius, k=30, domain=(0.0, 0.0, 1.0, 1.0), rng=None):
    """Bridson (2007) Poisson-disk sampling. Returns (M,2) points in domain
    with all pairwise distances >= radius. This is our clean isotropic P0."""
    if rng is None:
        rng = np.random.default_rng()
    x0, y0, x1, y1 = domain
    W, H = x1 - x0, y1 - y0
    cell = radius / np.sqrt(2.0)
    gw = int(np.ceil(W / cell))
    gh = int(np.ceil(H / cell))
    grid = -np.ones((gw, gh), dtype=int)
    samples = []
    active = []

    def gcoord(p):
        return int((p[0] - x0) / cell), int((p[1] - y0) / cell)

    def fits(p):
        gx, gy = gcoord(p)
        for ix in range(max(gx - 2, 0), min(gx + 3, gw)):
            for iy in range(max(gy - 2, 0), min(gy + 3, gh)):
                j = grid[ix, iy]
                if j >= 0:
                    q = samples[j]
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < radius * radius:
                        return False
        return True

    p0 = np.array([x0 + rng.random() * W, y0 + rng.random() * H])
    samples.append(p0)
    active.append(0)
    gx, gy = gcoord(p0)
    grid[gx, gy] = 0

    while active:
        ai = int(rng.integers(len(active)))
        base = samples[active[ai]]
        found = False
        for _ in range(k):
            ang = rng.random() * 2 * np.pi
            rr = radius * (1.0 + rng.random())
            p = base + rr * np.array([np.cos(ang), np.sin(ang)])
            if x0 <= p[0] < x1 and y0 <= p[1] < y1 and fits(p):
                samples.append(p)
                gx, gy = gcoord(p)
                grid[gx, gy] = len(samples) - 1
                active.append(len(samples) - 1)
                found = True
                break
        if not found:
            active.pop(ai)
    return np.asarray(samples)


def radius_for_count(n_per_unit_area):
    """Approximate Poisson-disk radius giving ~n points per unit area."""
    return float(np.sqrt(0.75 / max(n_per_unit_area, 1.0)))


# -- area-preserving anisotropy -------------------------------------------

def metric_M(theta, kappa):
    """Symmetric area-preserving stretch, det = 1. Stretch by sqrt(kappa) along
    theta, compress by 1/sqrt(kappa) perpendicular."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([np.sqrt(kappa), 1.0 / np.sqrt(kappa)])
    return R @ D @ R.T


def warp_constant(points, theta, kappa, center=(0.5, 0.5)):
    """Global constant-anisotropy stretch about center. Density preserved."""
    M = metric_M(theta, kappa)
    c = np.asarray(center, dtype=float)
    return c + (points - c) @ M.T


def warp_divfree_swirl(points, strength=2.0, sigma=0.25, center=(0.5, 0.5)):
    """Density-preserving swirl (r,phi)->(r,phi+g(r)), det DPhi = 1, with
    g(r) = strength*exp(-(r/sigma)^2). Smooth spatially-varying orientation."""
    c = np.asarray(center, dtype=float)
    d = points - c
    r = np.hypot(d[:, 0], d[:, 1])
    phi = np.arctan2(d[:, 1], d[:, 0])
    g = strength * np.exp(-(r / sigma) ** 2)
    phi2 = phi + g
    return np.stack([r * np.cos(phi2), r * np.sin(phi2)], axis=1) + c


# -- objective anisotropy measurement -------------------------------------

def global_near_field_anisotropy(points, h_factor=1.0, r_factor=3.0,
                                 interior=True):
    """PRIMARY estimator for a CONSTANT-theta tile.

    Closeness-weighted, coherently-summed double-angle over near neighbours:
      m = sum_{i,j} w_ij exp(2i phi_ij) / sum w_ij,  w_ij = exp(-(d_ij/h)^2)
    with bandwidth h = h_factor * median-NN-distance and neighbour cutoff
    R = r_factor * median-NN-distance. Weighting by closeness makes the near
    perp-neighbours dominate (breaks the perp/theta cancellation); the coherent
    global sum drives the isotropic floor toward zero.
      strength = |m|            (~0 isotropic, grows with kappa)
      axis     = angle(m)/2     (= theta + 90deg, the compressed axis)
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    tree = cKDTree(pts)
    dnn, _ = tree.query(pts, k=2)
    nn = float(np.median(dnn[:, 1]))
    h = h_factor * nn
    R = r_factor * nn
    nbrs = tree.query_ball_point(pts, R)
    num = 0.0 + 0.0j
    den = 0.0
    used = 0
    for i in range(n):
        p = pts[i]
        if interior and (p[0] < R or p[0] > 1.0 - R or
                         p[1] < R or p[1] > 1.0 - R):
            continue
        used += 1
        js = [j for j in nbrs[i] if j != i]
        if not js:
            continue
        d = pts[js] - p
        dist = np.hypot(d[:, 0], d[:, 1])
        w = np.exp(-(dist / h) ** 2)
        phi = np.arctan2(d[:, 1], d[:, 0])
        num += np.sum(w * np.exp(2j * phi))
        den += float(np.sum(w))
    if den <= 0:
        return dict(strength=float("nan"), axis=float("nan"), nn=nn, R=R, n_used=used)
    m = num / den
    return dict(strength=float(abs(m)), axis=float(0.5 * np.angle(m)),
                nn=nn, R=R, n_used=used)


def nn_vector_anisotropy(points, kk=1):
    """Independent cross-check: covariance of the k=kk nearest-neighbour vectors
    over all points. No disk-selection dilution (kk is tiny). Nearest neighbours
    concentrate along the compressed axis, so:
      ratio = sqrt(lambda_max / lambda_min)  (grows with kappa, monotone)
      axis  = top-eigenvector angle          (= theta + 90deg)
    """
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=kk + 1)
    vecs = pts[idx[:, 1:kk + 1]] - pts[:, None, :]
    v = vecs.reshape(-1, 2)
    C = (v.T @ v) / len(v)
    w, V = np.linalg.eigh(C)
    lmin, lmax = float(w[0]), float(w[1])
    return dict(ratio=float(np.sqrt(lmax / max(lmin, 1e-18))),
                axis=float(np.arctan2(V[1, 1], V[0, 1])))


def fixed_radius_anisotropy(points, n_target=12, area=1.0, interior=True):
    """Uniform-weight fixed-radius double-angle. BROKEN here: far-field dilution
    + perp/theta cancellation keep it near the floor. Reference only."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    density = n / max(area, 1e-9)
    R = float(np.sqrt(n_target / (np.pi * density)))
    tree = cKDTree(pts)
    nbrs = tree.query_ball_point(pts, R)
    strength = []
    axis = []
    for i in range(n):
        p = pts[i]
        if interior and (p[0] < R or p[0] > 1.0 - R or
                         p[1] < R or p[1] > 1.0 - R):
            continue
        js = [j for j in nbrs[i] if j != i]
        if len(js) < 3:
            continue
        d = pts[js] - p
        phi = np.arctan2(d[:, 1], d[:, 0])
        mm = np.mean(np.exp(2j * phi))
        strength.append(abs(mm))
        axis.append(0.5 * np.angle(mm))
    strength = np.asarray(strength)
    axis = np.asarray(axis)
    return dict(strength=strength, axis=axis, R=R, n_used=len(strength),
                strength_median=float(np.median(strength)) if len(strength) else float("nan"))


def local_metric_anisotropy(points, k=12, metric_J=None):
    """Covariance of k-NN displacements. BROKEN: k-NN fills a disk. Reference."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    kq = min(k, n - 1)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=kq + 1)
    J = None if metric_J is None else np.asarray(metric_J, dtype=float)
    kappa = np.ones(n)
    axis = np.zeros(n)
    for i in range(n):
        d = pts[idx[i, 1:]] - pts[i]
        if J is not None:
            d = d @ J.T
        C = (d.T @ d) / len(d)
        w, V = np.linalg.eigh(C)
        lmin, lmax = float(w[0]), float(w[1])
        kappa[i] = np.sqrt(lmax / max(lmin, 1e-18))
        axis[i] = np.arctan2(V[1, 1], V[0, 1])
    return dict(kappa=kappa, axis=axis, kappa_median=float(np.median(kappa)))


def local_double_angle(points, k=8, metric_J=None):
    """k-NN double-angle. BROKEN here (see module note); reference only."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    kq = min(k, n - 1)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=kq + 1)
    strength = np.zeros(n)
    for i in range(n):
        d = pts[idx[i, 1:]] - pts[i]
        if metric_J is not None:
            d = d @ np.asarray(metric_J).T
        phi = np.arctan2(d[:, 1], d[:, 0])
        strength[i] = np.abs(np.mean(np.exp(2j * phi)))
    return dict(strength=strength, strength_median=float(np.median(strength)))


# -- rendering ------------------------------------------------------------

def render_dots(points, path, crop=(0.0, 0.0, 1.0, 1.0), size=5.0,
                pad=0.01, dpi=160, dot="black", bg="white", figsize=4.0):
    """Dots-only renderer. Uniform black dots on white -- no size/shape/
    orientation cues. This is the M0 output format under test."""
    x0, y0, x1, y1 = crop
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.scatter(points[:, 0], points[:, 1], s=size, c=dot, edgecolors="none")
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor=bg)
    plt.close(fig)


def crop_points(points, crop=(0.0, 0.0, 1.0, 1.0)):
    x0, y0, x1, y1 = crop
    m = (points[:, 0] >= x0) & (points[:, 0] < x1) & \
        (points[:, 1] >= y0) & (points[:, 1] < y1)
    return points[m]


def subsample(points, n, rng):
    if len(points) <= n:
        return points
    sel = rng.choice(len(points), size=n, replace=False)
    return points[sel]
