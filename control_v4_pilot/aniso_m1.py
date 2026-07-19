"""aniso_m1.py -- Milestone 1 (Oracle achievability) machinery.

Builds a GUARANTEED-CLEAN anisotropic oracle by applying density-preserving
diffeomorphisms to a clean isotropic blue-noise set, and measures whether the
requested anisotropy is jointly achievable with the capacity requirement --
independent of any sampler. Torus (periodic) domain, uniform density for this
first pass. Pure numpy + scipy; reuses aniso_pilot for M0-side helpers.

Why the oracle is 'clean' by construction:
  Phi is an area-preserving diffeomorphism (det DPhi = 1), so P_oracle = Phi(P0)
  has EXACT N (bijection), NO collisions (diffeomorphism of a min-distance set),
  PRESERVED measure (uniform density stays uniform), and exact correspondence
  to P0. If capacity still fails on THIS set, the requirement is wrong.

Contents:
  poisson_disk_periodic     clean isotropic P0 on the torus (toroidal Bridson)
  field_from_streamfn       v = grad^perp(psi) and its gradient L, via FD
  psi_shear / psi_taylor_green / psi_transition   stream functions
  divfree_flow              RK4 integrate points + deformation gradient F
  warp_linear               constant area-preserving map (F = M, non-periodic)
  F_to_kappa_theta          local anisotropy ratio kappa and orientation theta
  hardcore_quality          spec q-metric using J = F^-1 (must stay >= ~0.98)
  regional_count_error / capacity_cov_carrier / kde_error   proper capacity
  cvt_energy_isotropic      NAIVE metric shown to inflate with kappa (a trap)
"""

import numpy as np
from scipy.spatial import cKDTree


TWO_PI = 2.0 * np.pi


# -- clean isotropic P0 on the torus --------------------------------------

def poisson_disk_periodic(radius, k=30, rng=None):
    """Toroidal Bridson Poisson-disk on [0,1)^2 (wraps at all edges)."""
    if rng is None:
        rng = np.random.default_rng()
    cell = radius / np.sqrt(2.0)
    g = int(np.ceil(1.0 / cell))
    cell = 1.0 / g                      # exact tiling of the torus
    grid = -np.ones((g, g), dtype=int)
    samples = []
    active = []

    def gcoord(p):
        return int(p[0] / cell) % g, int(p[1] / cell) % g

    def fits(p):
        gx, gy = gcoord(p)
        for ix in range(gx - 2, gx + 3):
            for iy in range(gy - 2, gy + 3):
                j = grid[ix % g, iy % g]
                if j >= 0:
                    d = p - samples[j]
                    d -= np.round(d)             # toroidal min image
                    if d[0] * d[0] + d[1] * d[1] < radius * radius:
                        return False
        return True

    p0 = rng.random(2)
    samples.append(p0)
    active.append(0)
    gx, gy = gcoord(p0)
    grid[gx, gy] = 0

    while active:
        ai = int(rng.integers(len(active)))
        base = samples[active[ai]]
        found = False
        for _ in range(k):
            ang = rng.random() * TWO_PI
            rr = radius * (1.0 + rng.random())
            p = (base + rr * np.array([np.cos(ang), np.sin(ang)])) % 1.0
            if fits(p):
                samples.append(p)
                gx, gy = gcoord(p)
                grid[gx, gy] = len(samples) - 1
                active.append(len(samples) - 1)
                found = True
                break
        if not found:
            active.pop(ai)
    return np.asarray(samples)


# -- stream functions and the induced divergence-free velocity field ------

def psi_shear(A=0.15, k=1):
    """Streamfn for a periodic shear: v_x = A sin(2pi k y), v_y = 0."""
    return lambda P: (A / (TWO_PI * k)) * np.cos(TWO_PI * k * P[:, 1])


def psi_taylor_green(A=0.06, kx=1, ky=1):
    """Taylor-Green cellular vortices; has natural isotropic transition regions
    (near vortex centres and saddles the deformation is ~isotropic)."""
    return lambda P: A * np.sin(TWO_PI * kx * P[:, 0]) * np.sin(TWO_PI * ky * P[:, 1])


def psi_transition(A=0.10, k=1):
    """A single-direction wave modulated so the deformation vanishes in a band
    (an explicit isotropic transition region within an otherwise sheared field)."""
    return lambda P: (A / (TWO_PI * k)) * np.cos(TWO_PI * k * P[:, 1]) \
        * (0.5 - 0.5 * np.cos(TWO_PI * P[:, 0]))


def field_from_streamfn(psi, eps=1e-4):
    """Return f(X) -> (v, L): v = grad^perp(psi) = (-d psi/dy, d psi/dx), and
    L[:,a,b] = d v_a / d x_b, both by central finite differences. Guarantees a
    (numerically) divergence-free field regardless of the psi supplied."""
    ex = np.array([eps, 0.0])
    ey = np.array([0.0, eps])

    def gradpsi(P):
        dpx = (psi(P + ex) - psi(P - ex)) / (2 * eps)
        dpy = (psi(P + ey) - psi(P - ey)) / (2 * eps)
        return dpx, dpy

    def vel(P):
        dpx, dpy = gradpsi(P)
        return np.stack([-dpy, dpx], axis=1)

    def f(X):
        v = vel(X)
        vx1, vx0 = vel(X + ex), vel(X - ex)
        vy1, vy0 = vel(X + ey), vel(X - ey)
        L = np.zeros((len(X), 2, 2))
        L[:, :, 0] = (vx1 - vx0) / (2 * eps)      # d v / d x
        L[:, :, 1] = (vy1 - vy0) / (2 * eps)      # d v / d y
        return v, L

    return f


def divfree_flow(points, field, T=1.0, steps=48, periodic=True):
    """RK4-integrate points under a divergence-free field for time T, carrying
    the per-point deformation gradient F (dF/dt = L F, F(0) = I). Returns
    (P_oracle, F) with F shape (n,2,2). Density is preserved (div v = 0)."""
    X = points.astype(float).copy()
    F = np.tile(np.eye(2), (len(points), 1, 1))
    dt = T / steps

    def deriv(Xs, Fs):
        v, L = field(Xs)
        dF = np.einsum('nij,njk->nik', L, Fs)
        return v, dF

    for _ in range(steps):
        k1x, k1F = deriv(X, F)
        k2x, k2F = deriv(X + 0.5 * dt * k1x, F + 0.5 * dt * k1F)
        k3x, k3F = deriv(X + 0.5 * dt * k2x, F + 0.5 * dt * k2F)
        k4x, k4F = deriv(X + dt * k3x, F + dt * k3F)
        X = X + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        F = F + (dt / 6.0) * (k1F + 2 * k2F + 2 * k3F + k4F)
        if periodic:
            X = X % 1.0
    return X, F


def warp_linear(points, M, center=(0.5, 0.5)):
    """Constant area-preserving map (non-periodic; maps square -> parallelogram).
    Returns (P, F) with F = M broadcast to every point."""
    c = np.asarray(center, dtype=float)
    P = c + (points - c) @ M.T
    F = np.tile(np.asarray(M, dtype=float), (len(points), 1, 1))
    return P, F


# -- realized local anisotropy from the deformation gradient --------------

def F_to_kappa_theta(F):
    """Local anisotropy from F via SVD. For det F = 1 the singular values are
    (s, 1/s); the anisotropy ratio kappa = s/(1/s) = S0/S1 (matches metric_M's
    kappa), and theta is the direction of maximal stretch in the OUTPUT (the
    first left singular vector)."""
    U, S, _ = np.linalg.svd(F)
    kappa = S[:, 0] / np.maximum(S[:, 1], 1e-18)
    theta = np.arctan2(U[:, 1, 0], U[:, 0, 0])
    return kappa, theta


# -- exactness / hard-core spacing ----------------------------------------

def hardcore_quality(points, F, r0, periodic=True, k=12):
    """Spec hard-core metric. For each near pair, q = min over the two directed
    checks of || F^-1 (pj - pi) || / r0 (F^-1 whitens the anisotropy back to the
    isotropic frame where r0 is the guaranteed min distance). Returns
    (q_min, q_1pct). q_min >= ~0.98 means no spacing violation."""
    Finv = np.linalg.inv(F)
    tree = cKDTree(points, boxsize=1.0 if periodic else None)
    _, idx = tree.query(points, k=k + 1)
    q_point = np.empty(len(points))
    for i in range(len(points)):
        nb = idx[i, 1:]
        dv = points[nb] - points[i]
        if periodic:
            dv -= np.round(dv)
        a = np.linalg.norm(dv @ Finv[i].T, axis=1)
        b = np.linalg.norm(np.einsum('mij,mj->mi', Finv[nb], dv), axis=1)
        q_point[i] = np.min(np.minimum(a, b))
    q_point /= r0
    return float(q_point.min()), float(np.percentile(q_point, 1))


# -- capacity metrics (uniform density) -----------------------------------

def regional_count_error(points, G=16):
    """Normalized RMS regional count error vs uniform expectation N/G^2."""
    idx = (np.clip(points, 0, 1 - 1e-9) * G).astype(int)
    counts = np.zeros((G, G))
    np.add.at(counts, (idx[:, 1], idx[:, 0]), 1)
    exp = len(points) / (G * G)
    return float(np.sqrt(np.mean((counts - exp) ** 2)) / exp)


def capacity_cov_carrier(points, rng, carriers_per_point=40, periodic=True):
    """CCVT-style capacity: assign a dense uniform carrier set to nearest site
    and report the coefficient of variation of per-site carrier counts. This is
    a DENSITY-uniformity measure -- it is (correctly) insensitive to anisotropy
    at fixed density."""
    m = carriers_per_point * len(points)
    Q = rng.random((m, 2))
    tree = cKDTree(points, boxsize=1.0 if periodic else None)
    _, idx = tree.query(Q, k=1)
    counts = np.bincount(idx, minlength=len(points))
    return float(np.std(counts) / np.mean(counts))


def kde_error(points, rng, grid=64, sigma_factor=1.5, periodic=True):
    """Low-pass density error vs uniform: splat points with a Gaussian on a grid
    (toroidal) and report relative L1 deviation from the flat target."""
    n = len(points)
    h = sigma_factor / np.sqrt(n)
    xs = (np.arange(grid) + 0.5) / grid
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    dens = np.zeros((grid, grid))
    for p in points:
        dx = gx - p[0]
        dy = gy - p[1]
        if periodic:
            dx -= np.round(dx)
            dy -= np.round(dy)
        dens += np.exp(-(dx * dx + dy * dy) / (2 * h * h))
    dens /= dens.mean()
    return float(np.mean(np.abs(dens - 1.0)))


def cvt_energy_isotropic(points, rng, carriers_per_point=40, periodic=True):
    """NAIVE isotropic CVT energy: mean squared (toroidal) distance from carriers
    to nearest site. Included ONLY to demonstrate the trap the spec warns about:
    it inflates with anisotropy even at perfect density, so it must NOT be the
    primary capacity acceptance metric."""
    m = carriers_per_point * len(points)
    Q = rng.random((m, 2))
    tree = cKDTree(points, boxsize=1.0 if periodic else None)
    d, _ = tree.query(Q, k=1)
    return float(np.mean(d ** 2))
