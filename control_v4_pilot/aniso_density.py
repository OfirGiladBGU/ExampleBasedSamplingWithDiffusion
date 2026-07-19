"""aniso_density.py -- density-aware anisotropic sampler (M1 stage-2 / M4 icons).

Generalizes the uniform M2 dart thrower to a NON-UNIFORM target density:

    lambda(x) = N * rho(x) / integral(rho)        (desired local intensity)
    J(x)      = sqrt(lambda(x)) * A(theta, kappa) (A symmetric, det A = 1)
    accept x  iff  || J(x)(x-p) || >= r0  AND  || J(p)(x-p) || >= r0

det A = 1  =>  sqrt(det G) = lambda, so the exclusion ellipse has area
pi*r0^2/lambda(x): DENSITY sets the exclusion AREA, ANISOTROPY only reshapes it.
Candidates are drawn proportional to rho, so points concentrate on ink.

Performance notes (this used to be very slow):
  * each accepted point caches its own J and exclusion extent, so neighbour
    checks are pure vector math -- no per-neighbour metric/bilinear rebuilds;
  * neighbours come from a periodically rebuilt cKDTree with an ADAPTIVE query
    radius (max of the candidate's own extent and the largest accepted extent),
    instead of a global radius bounded by the clamped lambda floor;
  * both hard-core directions are checked vectorized over all neighbours.
Progress is reported so long runs are not silent.
"""

import sys

import numpy as np
from scipy.spatial import cKDTree

import aniso_m2 as m2


def ink_density(img01):
    """Grayscale [0,1] (dark = ink) -> ink density rho in [0,1] (1 = full ink)."""
    img = np.asarray(img01, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(1.0 - img, 0.0, 1.0)


def lambda_field(rho, N, floor_frac=0.05):
    """Desired intensity lambda(x) = N*rho/mean(rho), clamped away from zero."""
    rho = np.asarray(rho, dtype=np.float64)
    m = float(rho.mean())
    if m <= 1e-9:
        raise ValueError("density map is empty (mean rho ~ 0)")
    return np.maximum(N * rho / m, floor_frac * N)


def _bilinear(field, pts):
    """Sample a (H,W) field at points (n,2) given as (x,y) in [0,1)."""
    H, W = field.shape
    x = np.clip(pts[:, 0], 0.0, 1.0 - 1e-9) * W - 0.5
    y = np.clip(pts[:, 1], 0.0, 1.0 - 1e-9) * H - 0.5
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    x0c = np.clip(x0, 0, W - 1); y0c = np.clip(y0, 0, H - 1)
    fx = x - x0; fy = y - y0
    return (field[y0c, x0c] * (1 - fx) * (1 - fy) + field[y0c, x1] * fx * (1 - fy) +
            field[y1, x0c] * (1 - fx) * fy + field[y1, x1] * fx * fy)


def make_candidate_drawer(rho, rng):
    """Draw candidate positions proportional to rho (metric-volume sampling)."""
    p = np.asarray(rho, dtype=np.float64).ravel()
    tot = p.sum()
    if tot <= 0:
        raise ValueError("density map is empty")
    p = p / tot
    H, W = rho.shape
    idx = np.arange(p.size)

    def draw(n):
        k = rng.choice(idx, size=n, p=p)
        iy, ix = np.divmod(k, W)
        return np.stack([(ix + rng.random(n)) / W, (iy + rng.random(n)) / H], axis=1)

    return draw


def sample_density_anisotropic(rho, theta_fn, kappa_fn, N, rng, r0=1.0,
                               kappa_max=None, floor_frac=0.05,
                               max_attempts=None, batch=512, rebuild_every=64,
                               progress=True, progress_every=0.1):
    """Anisotropic dart throwing against a NON-UNIFORM density.
    Returns (points (M,2), reached_N bool, attempts)."""
    lam = lambda_field(rho, N, floor_frac=floor_frac)
    if kappa_max is None:
        kappa_max = float(np.max(kappa_fn(np.array([[0.5, 0.5]]))))
    if max_attempts is None:
        max_attempts = 150 * N

    draw = make_candidate_drawer(rho, rng)
    pts, Js = [], []
    tree, n_tree, r_max = None, 0, 0.0
    attempts = 0
    next_report = progress_every

    while len(pts) < N and attempts < max_attempts:
        cands = draw(batch)
        th = kappa = None
        th = theta_fn(cands)
        kappa = kappa_fn(cands)
        lm = _bilinear(lam, cands)
        sq = np.sqrt(lm)
        A = m2.metric_A_batch(th, kappa)
        r_cands = r0 * np.sqrt(kappa) / sq          # max exclusion extent

        for c in range(len(cands)):
            if len(pts) >= N:
                break
            attempts += 1
            x = cands[c]
            Jx = A[c] * sq[c]

            nb = []
            if tree is not None:
                nb = tree.query_ball_point(x, max(r_cands[c], r_max))
            if len(pts) > n_tree:
                nb = list(nb) + list(range(n_tree, len(pts)))

            ok = True
            if nb:
                Pn = np.asarray([pts[j] for j in nb])
                dv = x - Pn
                if np.min(np.linalg.norm(dv @ Jx.T, axis=1)) < r0:
                    ok = False
                else:
                    Jn = np.stack([Js[j] for j in nb])
                    if np.min(np.linalg.norm(np.einsum('mij,mj->mi', Jn, dv), axis=1)) < r0:
                        ok = False

            if ok:
                pts.append(x)
                Js.append(Jx)
                r_max = max(r_max, float(r_cands[c]))
                if len(pts) - n_tree >= rebuild_every:
                    tree = cKDTree(np.asarray(pts))
                    n_tree = len(pts)

        if progress and len(pts) >= next_report * N:
            sys.stdout.write("\r    placed %5d/%d  attempts %8d" % (len(pts), N, attempts))
            sys.stdout.flush()
            while len(pts) >= next_report * N:
                next_report += progress_every

    if progress:
        sys.stdout.write("\r    placed %5d/%d  attempts %8d\n" % (len(pts), N, attempts))
        sys.stdout.flush()
    return np.asarray(pts), (len(pts) == N), attempts


def sample_exact_n_density(rho, theta_fn, kappa_fn, N, rng, r0_init=1.0,
                           kappa_max=None, shrink=0.9, tries=10, **kw):
    """Auto-shrink r0 until exactly N points are placed. Returns (pts, r0, attempts)."""
    r0 = r0_init
    P, reached, att = None, False, 0
    for _ in range(tries):
        P, reached, att = sample_density_anisotropic(
            rho, theta_fn, kappa_fn, N, rng, r0=r0, kappa_max=kappa_max, **kw)
        if reached:
            return P, r0, att
        r0 *= shrink
    return P, r0, att


def density_capacity_error(points, rho, G=16):
    """Regional count error against the TARGET density (not uniform)."""
    H, W = rho.shape
    ys = (np.arange(H) * G // H)
    xs = (np.arange(W) * G // W)
    mass = np.zeros((G, G))
    np.add.at(mass, (ys[:, None].repeat(W, 1), xs[None, :].repeat(H, 0)), rho)
    tot = mass.sum()
    if tot <= 0:
        raise ValueError("empty density")
    expect = len(points) * mass / tot
    idx = np.clip((np.asarray(points) * G).astype(int), 0, G - 1)
    actual = np.zeros((G, G))
    np.add.at(actual, (idx[:, 1], idx[:, 0]), 1)
    nz = expect > 1e-6
    return float(np.sqrt(np.mean((actual[nz] - expect[nz]) ** 2)) / max(expect[nz].mean(), 1e-9))


if __name__ == "__main__":
    import time
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W] / H
    img = np.ones((H, W), dtype=np.float64)
    img[((xx - 0.5) ** 2 + (yy - 0.5) ** 2) < 0.25 ** 2] = 0.0
    rho = ink_density(img)
    N = 1024
    print("self-test: disk on white, N=%d" % N)
    for kappa in (1.0, 2.0):
        th_fn, ka_fn, kmax = m2.const_field(30.0, kappa)
        rng = np.random.default_rng(0)
        t0 = time.time()
        P, r0, att = sample_exact_n_density(rho, th_fn, ka_fn, N, rng,
                                            r0_init=1.0, kappa_max=kmax)
        dt = time.time() - t0
        print("  kappa=%.2f  placed %d/%d  r0=%.3f  attempts=%d  cap_err=%.3f  %.1fs"
              % (kappa, len(P), N, r0, att, density_capacity_error(P, rho), dt))
    print("capacity error should be SIMILAR for kappa=1 and kappa=2")
    print("(det-metric: anisotropy reshapes the exclusion ellipse, not its area)")
