"""aniso_m2.py -- Milestone 2 (Constructive sampler) machinery.

Li-style anisotropic dart throwing against the local metric J(x) = A(x)/s, where
A(x) = R(theta) diag(sqrt(kappa), 1/sqrt(kappa)) R(theta)^T is the det-1 pure
anisotropy and s is the isotropic spacing scale (r0 = 1 in metric units). A dart
x is accepted iff, for every nearby accepted p, BOTH directed hard-core checks
hold (the metric is asymmetric):
    || J(x) (x - p) || >= 1   and   || J(p) (x - p) || >= 1
Because det A = 1 the real-space exclusion ellipse has area pi*s^2 regardless of
kappa -> anisotropy changes the SHAPE of exclusion, not its area, so it cannot
add or steal points (the determinant-metric mechanism from the pilot).

Hard-core spacing is an ACCEPTANCE INVARIANT, never a penalty. Exact N via a
stop-at-N rule. Periodic (torus) domain, uniform density (first pass).

Reuses aniso_m1 (capacity metrics) and aniso_pilot (anisotropy estimators).

Contents:
  metric_A_batch          batched A(theta,kappa), det 1
  const_field / smooth_orientation_field   (theta_fn, kappa_fn, kappa_max)
  sample_anisotropic      the dart thrower (grid hash, exact-N stop, auto-scale)
  hardcore_field          independent validator: q_min and violation count
  local_near_field_axis   per-point orientation (for smooth fields)
"""

import numpy as np
from scipy.spatial import cKDTree


def metric_A_batch(theta, kappa):
    """Batched det-1 anisotropy matrices A = R diag(sqrt k, 1/sqrt k) R^T."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
    n = len(theta)
    c, s = np.cos(theta), np.sin(theta)
    sq = np.sqrt(kappa)
    isq = 1.0 / sq
    R = np.empty((n, 2, 2))
    R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c
    D = np.zeros((n, 2, 2))
    D[:, 0, 0] = sq
    D[:, 1, 1] = isq
    return R @ D @ np.transpose(R, (0, 2, 1))


# -- anisotropy fields (theta_fn, kappa_fn take (m,2) -> (m,)) -------------

def const_field(theta_deg, kappa):
    th = np.deg2rad(theta_deg)
    return (lambda X: np.full(len(X), th),
            lambda X: np.full(len(X), float(kappa)),
            float(kappa))


def smooth_orientation_field(kappa, turns=1.0, theta0_deg=0.0):
    """Constant kappa, orientation rotating smoothly with y over `turns` half-
    turns across the domain -- a smooth commanded-orientation field."""
    th0 = np.deg2rad(theta0_deg)
    return (lambda X: th0 + np.pi * turns * X[:, 1],
            lambda X: np.full(len(X), float(kappa)),
            float(kappa))


# -- the constructive sampler ---------------------------------------------

def sample_anisotropic(theta_fn, kappa_fn, s, N, rng, kappa_max,
                       periodic=True, max_attempts=None):
    """Anisotropic dart throwing on [0,1)^2. Returns (points, reached_N, attempts).
    Search radius s*sqrt(kappa_max) provably bounds every possible violator
    (||J dv|| >= |dv| / (s sqrt kappa)), so no violating pair is missed."""
    if max_attempts is None:
        max_attempts = 80 * N
    search_r = s * np.sqrt(kappa_max)
    g = max(3, int(1.0 / search_r))
    cell = 1.0 / g
    grid = {}
    pts = []

    def cellof(x):
        return (int(x[0] / cell) % g, int(x[1] / cell) % g)

    attempts = 0
    while len(pts) < N and attempts < max_attempts:
        attempts += 1
        x = rng.random(2)
        th = theta_fn(x[None])[0]
        ka = kappa_fn(x[None])[0]
        Jx = metric_A_batch(th, ka)[0] / s
        cx, cy = cellof(x)
        ok = True
        for dx in (-1, 0, 1):
            if not ok:
                break
            for dy in (-1, 0, 1):
                for j in grid.get(((cx + dx) % g, (cy + dy) % g), ()):
                    dv = x - pts[j]
                    if periodic:
                        dv -= np.round(dv)
                    if dv[0] * dv[0] + dv[1] * dv[1] > search_r * search_r:
                        continue
                    if np.hypot(*(Jx @ dv)) < 1.0:
                        ok = False
                        break
                    thp = theta_fn(pts[j][None])[0]
                    kap = kappa_fn(pts[j][None])[0]
                    Jp = metric_A_batch(thp, kap)[0] / s
                    if np.hypot(*(Jp @ dv)) < 1.0:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            grid.setdefault((cx, cy), []).append(len(pts))
            pts.append(x)
    return np.asarray(pts), (len(pts) == N), attempts


def sample_exact_n(theta_fn, kappa_fn, s0, N, rng, kappa_max,
                   periodic=True, shrink=0.9, tries=8):
    """Call the sampler, auto-shrinking s until exactly N points are placed.
    Returns (points, s_used, attempts)."""
    s = s0
    for _ in range(tries):
        P, reached, att = sample_anisotropic(theta_fn, kappa_fn, s, N, rng,
                                             kappa_max, periodic=periodic)
        if reached:
            return P, s, att
        s *= shrink
    return P, s, att      # best effort (may be < N)


# -- validation -----------------------------------------------------------

def hardcore_field(points, theta_fn, kappa_fn, s, periodic=True, k=16):
    """Independent hard-core validator against J = A/s. Returns
    (q_min, q_1pct, n_violations). q >= 1 everywhere and 0 violations means the
    accepted set truly satisfies the anisotropic hard-core constraint."""
    n = len(points)
    A = metric_A_batch(theta_fn(points), kappa_fn(points))
    J = A / s
    tree = cKDTree(points, boxsize=1.0 if periodic else None)
    kk = min(k, n - 1)
    _, idx = tree.query(points, k=kk + 1)
    q_point = np.empty(n)
    viol = 0
    for i in range(n):
        nb = idx[i, 1:]
        dv = points[nb] - points[i]
        if periodic:
            dv -= np.round(dv)
        a = np.linalg.norm(dv @ J[i].T, axis=1)
        b = np.linalg.norm(np.einsum('mij,mj->mi', J[nb], dv), axis=1)
        q = np.minimum(a, b)
        q_point[i] = q.min()
        viol += int(np.count_nonzero(q < 1.0 - 1e-6))
    return float(q_point.min()), float(np.percentile(q_point, 1)), viol // 2


def local_near_field_axis(points, h_factor=1.0, r_factor=3.0, periodic=True):
    """Per-point closeness-weighted double-angle axis (line, period pi). For a
    stretch along theta the measured axis is theta+90 (compressed axis)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    tree = cKDTree(pts, boxsize=1.0 if periodic else None)
    dnn, _ = tree.query(pts, k=2)
    nn = float(np.median(dnn[:, 1]))
    h, R = h_factor * nn, r_factor * nn
    nbrs = tree.query_ball_point(pts, R)
    axis = np.full(n, np.nan)
    for i in range(n):
        js = [j for j in nbrs[i] if j != i]
        if len(js) < 3:
            continue
        dv = pts[js] - pts[i]
        if periodic:
            dv -= np.round(dv)
        dist = np.hypot(dv[:, 0], dv[:, 1])
        w = np.exp(-(dist / h) ** 2)
        phi = np.arctan2(dv[:, 1], dv[:, 0])
        m = np.sum(w * np.exp(2j * phi)) / np.sum(w)
        axis[i] = 0.5 * np.angle(m)
    return axis
