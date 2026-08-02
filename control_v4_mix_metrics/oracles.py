"""Cheap "extreme" stipple oracles for the multi-oracle descriptor axis (M0).

Why these exist
---------------
The descriptor plan needs oracles that STRETCH descriptor space, not oracles that are good.
`icons-50_512_{GBN,WVS,BNOT}` already exist on disk and all three are spacing-optimised samplers
that live near the "regular" end of the axis. Nothing on disk occupies the low-regularity or
hard-structure corners, so with only those three the axis is short and every intermediate is
subtle. This module adds the cheap extremes:

  * `floyd_steinberg`  -- serpentine error diffusion; the classic "worm" structures. Maximally
                          distinct from a blue-noise sampler, and directional (should be the one
                          oracle that lights up the anisotropy descriptor on flat icons).
  * `ordered_dither`   -- Bayer threshold matrix; a hard periodic grid. Anchors the
                          maximum-structure corner.
  * `white_noise`      -- independent draws from rho. The zero-structure corner: clumps and gaps
                          at every scale.
  * `jittered_grid`    -- (bonus, near-free) stratified grid; sits between ordered and blue noise.

Everything is torch-free (numpy / scipy / PIL only) and returns points as float64 `(N, 2)` in
`[0, 1]`, x-then-y, matching `train_control.extract_points_from_target`.

Matched point count (the thing that must not drift)
---------------------------------------------------
The plan requires every oracle to run at MATCHED N per image, and `control_v4` fixes
`n_points = GRID_SIZE ** 2` (= 1024 at G=32, `train_control.py:404`), so N is a constant, not a
per-image quantity. Dithering methods do not take N as an argument -- they emit however many
pixels cross threshold -- so each one is driven through a GAIN `g` applied to rho.

The naive way to hit N is to bisect on g re-running the dither each time. That is ~15 error
diffusion passes per image and error diffusion is inherently sequential, so it dominates the
whole M0 budget. Instead we exploit the defining property of error diffusion: it conserves total
mass. The dithered count is therefore already ~= `sum(clip(rho * g, 0, 1))`, which is a monotone
scalar function of `g` that can be solved on the rho ARRAY with no dithering at all. So we solve
for g first (cheap), dither ONCE, and repair the handful of leftover points exactly. One pass per
image instead of fifteen.

Working resolution: why it is derived, not chosen (measured, not assumed)
-------------------------------------------------------------------------
The obvious move is to dither at the native 512 (or higher, to keep positions accurate). Measured,
that is wrong in both directions, and badly:

    oracle    raster    tone L1 err   corr(count, rho mass)
    fs        1024        0.448           0.774
    ordered   1024        1.452           0.477

At N=1024 in a 1024^2 raster the ink fill is ~0.1%. Two separate failures follow. Ordered
dithering cannot resolve tone at all, because an 8x8 Bayer matrix's finest level is 1/64 = 1.6%,
an order of magnitude coarser than the requested fill -- only the single lowest-threshold cell of
each tile ever fires, mid-greys receive nothing, and the output collapses to a bare 8px lattice.
Error diffusion does not collapse, but it discharges accumulated error hundreds of pixels away
from the ink that produced it, dropping points into white background ("highlight dirt"). Both
violate the plan's hard requirement that every oracle respect the density map.

The deeper point is that at 0.1% fill FS emits isolated dots and NO worms. The worm structure --
the entire reason Floyd-Steinberg is in the oracle set -- only exists near mid-tone, where dots
nearly touch on the raster. High resolution destroys the very character we are sampling for.

So the raster is derived from the point budget: R = sqrt(N / (target_fill * ink_area_fraction)),
targeting ~25% fill over the inked region, clamped to [48, 512]. Dot spacing then lands at a few
raster cells, which is what halftone structure requires. Measured after the fix, on a disc+band,
a sparse ring and a full gradient, every oracle reproduces tone at corr >= 0.95 (white noise sits
at 0.78 on the full gradient, which is its own Poisson counting variance at ~4 points/cell, not a
tone failure).

The cost is real and is NOT swept under the rug: point positions are quantised to 1/R, which at
N=1024 is roughly half an expected spacing. That is intrinsic -- a halftone's dots ARE on its
raster -- but it is also a confound for the NN-distance descriptor the axis rests on. `m0_run.py`
therefore carries a mandatory quantisation control: GBN/WVS centroids are requantised to the same
lattice and re-measured, so the share of any separation attributable to the lattice is reported as
a number rather than argued away.
"""

import numpy as np

EPS = 1e-12

# ---------------------------------------------------------------------------
# Optional numba acceleration for the sequential error-diffusion loop.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on cluster env
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap


@njit(cache=True)
def _fs_kernel(f, serpentine):
    """In-place serpentine Floyd-Steinberg error diffusion. Returns a uint8 mask.

    `f` is modified. Standard FS weights (7, 3, 5, 1)/16. Serpentine scanning (alternating row
    direction) is what produces the characteristic diagonal worms rather than a directional
    smear, and it is the variant with actual print history, so it is the default.
    """
    h, w = f.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        if serpentine and (y % 2 == 1):
            x_start, x_end, step = w - 1, -1, -1
        else:
            x_start, x_end, step = 0, w, 1
        x = x_start
        while x != x_end:
            old = f[y, x]
            new = 1.0 if old > 0.5 else 0.0
            if new > 0.5:
                mask[y, x] = 1
            err = old - new
            # Right-then-down neighbours, mirrored on right-to-left rows.
            if step == 1:
                if x + 1 < w:
                    f[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x - 1 >= 0:
                        f[y + 1, x - 1] += err * (3.0 / 16.0)
                    f[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w:
                        f[y + 1, x + 1] += err * (1.0 / 16.0)
            else:
                if x - 1 >= 0:
                    f[y, x - 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x + 1 < w:
                        f[y + 1, x + 1] += err * (3.0 / 16.0)
                    f[y + 1, x] += err * (5.0 / 16.0)
                    if x - 1 >= 0:
                        f[y + 1, x - 1] += err * (1.0 / 16.0)
            x += step
    return mask


# ---------------------------------------------------------------------------
# rho handling and gain solving
# ---------------------------------------------------------------------------

def rho_from_gray(gray01):
    """Project-canonical ink probability: rho = clip(1 - image, 0, 1) (`smart_init.py:18`)."""
    return np.clip(1.0 - np.asarray(gray01, dtype=np.float64), 0.0, 1.0)


def resample_rho(rho, size):
    """Bilinear resize of rho to (size, size) without pulling in PIL."""
    h, w = rho.shape
    if (h, w) == (size, size):
        return rho.copy()
    yi = (np.arange(size) + 0.5) * (h / size) - 0.5
    xi = (np.arange(size) + 0.5) * (w / size) - 0.5
    yi = np.clip(yi, 0, h - 1)
    xi = np.clip(xi, 0, w - 1)
    y0 = np.floor(yi).astype(np.int64)
    x0 = np.floor(xi).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    top = rho[np.ix_(y0, x0)] * (1 - wx) + rho[np.ix_(y0, x1)] * wx
    bot = rho[np.ix_(y1, x0)] * (1 - wx) + rho[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def solve_gain(rho, n_points, tol=1e-9, max_iter=80):
    """Smallest gain g with `sum(clip(rho * g, 0, 1)) == n_points`.

    Monotone non-decreasing in g, so plain bisection is exact to machine precision. Saturation
    (clip at 1) only binds if the image cannot hold n_points at all, which the caller checks.
    """
    rho = np.asarray(rho, dtype=np.float64)
    total = rho.sum()
    if total <= EPS:
        return 0.0
    lo = 0.0
    hi = max(n_points / total, 1.0)
    # Grow hi until it over-shoots (or the image saturates).
    for _ in range(64):
        if np.clip(rho * hi, 0.0, 1.0).sum() >= n_points:
            break
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if np.clip(rho * mid, 0.0, 1.0).sum() < n_points:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(hi, 1.0):
            break
    return 0.5 * (lo + hi)


def _mask_to_points(mask, rng, n_points, weight=None):
    """Pixel mask -> exactly `n_points` continuous coords in [0,1], x then y.

    Error diffusion conserves mass only approximately (boundary rows leak), so the raw count
    lands within a few of N. Rather than re-running the dither we repair the remainder here:
    surplus points are dropped where rho is LOWEST (those are the ones the dither was least
    committed to) and any deficit is drawn from the highest-rho unused pixels. Both edits touch
    O(few) points out of 1024, far below descriptor noise.
    """
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    n = len(xs)
    if n > n_points:
        if weight is not None:
            keep = np.argsort(weight[ys, xs])[::-1][:n_points]
        else:
            keep = rng.choice(n, size=n_points, replace=False)
        ys, xs = ys[keep], xs[keep]
    elif n < n_points:
        deficit = n_points - n
        free = np.nonzero(~mask.astype(bool))
        if len(free[0]) == 0:
            raise ValueError("cannot reach n_points: every pixel already selected")
        if weight is not None:
            order = np.argsort(weight[free])[::-1][:deficit]
        else:
            order = rng.choice(len(free[0]), size=min(deficit, len(free[0])), replace=False)
        ys = np.concatenate([ys, free[0][order]])
        xs = np.concatenate([xs, free[1][order]])
    # Pixel centres, x-then-y, matching extract_points_from_target's (cx/w, cy/h).
    pts = np.stack([(xs + 0.5) / w, (ys + 0.5) / h], axis=1).astype(np.float64)
    return pts


def _check_capacity(rho, n_points):
    inked = int((rho > 0).sum())
    if inked < n_points:
        raise ValueError(
            f"image has only {inked} inked pixels but {n_points} points requested; raise work_res"
        )


def halftone_raster(rho, n_points, target_fill=0.25, lo=48, hi=512, ink_thresh=0.02):
    """Raster size at which a dither of `n_points` dots sits at ~`target_fill` over the ink.

    See the module docstring: this is the parameter that decides whether FS/ordered reproduce tone
    and whether FS shows worms at all. Derived per image from the inked area fraction, because a
    sparse outline icon and a solid blob need very different screen frequencies to place the same
    1024 dots at the same fill.
    """
    a = float((np.asarray(rho) > ink_thresh).mean())
    if a <= EPS:
        return lo
    r = int(round(np.sqrt(n_points / (target_fill * a))))
    return int(np.clip(r, lo, hi))


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

def floyd_steinberg(rho, n_points, work_res=None, serpentine=True, seed=0):
    """Serpentine Floyd-Steinberg error diffusion at matched N.

    `work_res=None` derives the halftone raster from the point budget (see module docstring);
    passing an explicit value is supported but will silently cost tone fidelity and worm structure
    if it is far from `halftone_raster`.
    """
    rng = np.random.RandomState(seed)
    if work_res is None:
        work_res = halftone_raster(rho, n_points)
    r = resample_rho(rho, work_res)
    _check_capacity(r, n_points)
    g = solve_gain(r, n_points)
    f = np.clip(r * g, 0.0, 1.0)
    mask = _fs_kernel(f, bool(serpentine))
    return _mask_to_points(mask, rng, n_points, weight=r)


def _bayer_matrix(order=3):
    """2^order x 2^order normalised Bayer matrix in (0, 1)."""
    m = np.array([[0.0]])
    for _ in range(order):
        m = np.block([
            [4 * m + 0, 4 * m + 2],
            [4 * m + 3, 4 * m + 1],
        ])
    return (m + 0.5) / m.size


def ordered_dither(rho, n_points, work_res=None, order=3, seed=0):
    """Bayer / ordered dithering at matched N. Hard periodic structure by construction.

    `work_res=None` derives the halftone raster from the point budget. This oracle is the one that
    fails hardest at a mismatched raster -- an 8x8 Bayer matrix resolves tone only down to 1/64,
    so at low fill it degenerates into a bare lattice that ignores mid-greys entirely.
    """
    rng = np.random.RandomState(seed)
    if work_res is None:
        work_res = halftone_raster(rho, n_points)
    r = resample_rho(rho, work_res)
    _check_capacity(r, n_points)
    g = solve_gain(r, n_points)
    f = np.clip(r * g, 0.0, 1.0)
    b = _bayer_matrix(order)
    reps = work_res // b.shape[0] + 1
    thr = np.tile(b, (reps, reps))[:work_res, :work_res]
    mask = (f > thr).astype(np.uint8)
    return _mask_to_points(mask, rng, n_points, weight=r)


def white_noise(rho, n_points, work_res=512, seed=0):
    """Independent draws from rho (inverse-CDF over pixels + uniform within pixel).

    Deliberately the zero-structure extreme: clumps and voids at every scale. Sampling WITHOUT
    replacement would quietly inject a hard-core repulsion at the pixel scale and lift the
    regularity floor, so this samples WITH replacement and relies on within-pixel jitter to
    separate collisions.
    """
    rng = np.random.RandomState(seed)
    r = resample_rho(rho, work_res)
    flat = r.ravel()
    total = flat.sum()
    if total <= EPS:
        raise ValueError("all-white image: no ink to sample")
    idx = rng.choice(flat.size, size=n_points, replace=True, p=flat / total)
    ys, xs = np.divmod(idx, work_res)
    jx = rng.rand(n_points)
    jy = rng.rand(n_points)
    return np.stack([(xs + jx) / work_res, (ys + jy) / work_res], axis=1).astype(np.float64)


def jittered_grid(rho, n_points, work_res=512, seed=0):
    """Stratified sample of rho: one point per equal-mass cell of a sqrt(N) x sqrt(N) grid.

    Sits between `ordered_dither` and blue noise -- grid-anchored but not periodic. Included
    because it is near-free; drop it if it adds no descriptor separation.
    """
    rng = np.random.RandomState(seed)
    r = resample_rho(rho, work_res)
    side = int(round(np.sqrt(n_points)))
    if side * side != n_points:
        raise ValueError(f"jittered_grid needs a square n_points, got {n_points}")
    row_mass = r.sum(axis=1)
    cum = np.cumsum(row_mass)
    if cum[-1] <= EPS:
        raise ValueError("all-white image: no ink to sample")
    cum = cum / cum[-1]
    edges_y = np.searchsorted(cum, np.arange(side + 1) / side)
    edges_y = np.clip(edges_y, 0, work_res)
    pts = []
    for i in range(side):
        y0 = int(edges_y[i])
        y1 = int(max(edges_y[i + 1], y0 + 1))
        band = r[y0:y1, :].sum(axis=0)
        cb = np.cumsum(band)
        if cb[-1] <= EPS:
            xs = np.linspace(0, work_res - 1, side)
        else:
            cb = cb / cb[-1]
            xs = np.searchsorted(cb, (np.arange(side) + rng.rand(side)) / side)
            xs = np.clip(xs, 0, work_res - 1).astype(np.float64)
        yy = y0 + rng.rand(side) * (y1 - y0)
        pts.append(np.stack([(xs + rng.rand(side)) / work_res, yy / work_res], axis=1))
    out = np.concatenate(pts, axis=0).astype(np.float64)
    return np.clip(out, 0.0, 1.0 - 1e-9)


ORACLES = {
    "fs": floyd_steinberg,
    "ordered": ordered_dither,
    "white": white_noise,
    "jitgrid": jittered_grid,
}

# Oracles that already exist on disk as rendered target PNGs (points come from centroids).
DISK_ORACLES = ("gbn", "wvs", "bnot")


def render_points_pixels(points, size=512, resolve_collisions=True):
    """Render points as SINGLE BLACK PIXELS on white -- the convention the existing targets use.

    Measured on `icons-50_512_{GBN,WVS,BNOT}`: mean connected-component area is exactly 1.00 px, so
    every dot in those datasets is one pixel. Matching that matters for two independent reasons.

    Consistency: a new oracle rendered as a 5 px antialiased disc is not rasterised comparably to a
    1 px disk oracle, so any descriptor measured through the PNG channel would differ by dot size
    rather than by oracle.

    Matched N survival: at 512 with N=1024 the expected spacing is 16 px, but the dither oracles sit
    on a raster as fine as ~4 px and white noise clumps arbitrarily tightly. With a 2.2 px dot
    diameter, adjacent dots merge into one blob and centroid extraction returns fewer than N --
    measured at -157 points/icon for white noise and -48 for the jittered grid. One-pixel dots reduce
    that to the birthday-collision rate alone (~1024^2 / 2*512^2 ~ 2 expected coincidences).

    Those residual coincidences are resolved by nudging the loser to the nearest free pixel rather
    than dropping it, so the PNG holds exactly N points. A 1 px nudge is 1/16 of an expected spacing
    -- far below any descriptor's resolution -- whereas a dropped point breaks matched N outright.
    """
    from PIL import Image

    idx = np.clip((np.asarray(points, dtype=np.float64) * size).astype(np.int64), 0, size - 1)
    occupied = np.zeros((size, size), dtype=bool)
    collisions = 0
    for x, y in idx:
        if not occupied[y, x]:
            occupied[y, x] = True
            continue
        collisions += 1
        placed = False
        for r in range(1, 8):                       # expanding ring search for a free pixel
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < size and 0 <= nx < size and not occupied[ny, nx]:
                        occupied[ny, nx] = True
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
    arr = np.where(occupied, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L"), collisions


def render_dots_png(points, size=512, radius=1.6, supersample=3):
    """Render points as uniform black dots on white -- the M0 visibility format.

    Matches the existing target PNGs closely enough that centroid re-extraction is a fair
    round-trip control. Anti-aliased via supersampling so dot centroids stay sub-pixel accurate.
    """
    from PIL import Image, ImageDraw

    s = size * supersample
    img = Image.new("L", (s, s), 255)
    d = ImageDraw.Draw(img)
    r = radius * supersample
    for x, y in points:
        cx, cy = x * s, y * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
    return img.resize((size, size), Image.LANCZOS)


# Oracles whose points are BOUND TO THE DITHER RASTER by construction (see the working-resolution
# note above). Their positions are quantised to 1/R; every other oracle here is continuous.
# `m0_run.py` uses this split to decide when a comparison needs the common-raster control: a
# continuous-vs-raster pair is confounded by the lattice, a same-family pair is not.
RASTER_BOUND = ("fs", "ordered")
