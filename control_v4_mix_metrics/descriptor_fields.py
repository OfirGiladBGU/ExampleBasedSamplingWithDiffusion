"""Local descriptor FIELDS for multi-oracle descriptor control (plan section "Descriptors").

The hard requirement from the plan is that every descriptor be computable on a LOCAL WINDOW. A
single global number per image permits only global conditioning, and "region A in one style,
region B in another" becomes unreachable -- there is no way to say *where*. So every descriptor
here is emitted as a `(G, G)` spatial field at the offsets-grid resolution (G=32 to match
`control_v4`'s `GRID_SIZE`), computed in a sliding window a few expected spacings wide.

This supersedes `control_v4_mix/spacing_regularity.py`, which computes the same underlying
quantities but reduces them to one scalar per image. The per-POINT primitives are unchanged and
deliberately so -- `norm_nn_cv` is the measure that cleared Gate 0 on the WVS<->GBN axis, and
re-deriving it differently here would invalidate that result. What changes is only the pooling:
scalar -> windowed field.

Descriptors (numbered as in the plan)
-------------------------------------
1. `nn_mean`, `nn_cv`   -- local NN distance normalised by expected spacing sqrt(1/lambda), and its
                           CV. The most direct regularity measure; separates a relaxed CVT (low)
                           from white noise (high). `nn_cv` is the primary.
2. `cap_cv`             -- local capacity CV from POWER cell areas, not Voronoi. The plan calls
                           this out explicitly (R4's objection) -- see `power_cell_areas`.
3. `edge_align`         -- mean image-gradient magnitude at point locations, normalised against
                           what rho alone predicts, so it measures alignment rather than "edges
                           are darker anyway".
   `edge_align_strat`   -- the same quantity computed within intensity strata. Reported as a free
                           DIAGNOSTIC, not a gate: on the WVS<->GBN pair the unstratified version
                           was found to be a tone-transfer (rho-decomposable) effect, and the
                           stratified one is what survives tone control. Costs one extra pass.
4. `aniso`              -- local double-angle magnitude. Should read ~0 for every isotropic oracle
                           on flat icons and light up only for Floyd-Steinberg's directional worms.
5. `pcf_peak`,          -- local pair-correlation first-peak height and position, in units of the
   `pcf_pos`               local expected spacing. ADDED after M0 measured that descriptors 1-4
                           separate 20/21 oracle pairs but leave `fs|ordered` tied: ordered dither
                           is exactly PERIODIC and Floyd-Steinberg is aperiodic, yet the two have a
                           similar NN-distance CV and both read ~0 anisotropy, so nothing in 1-4
                           sees the difference. The plan names this descriptor for exactly this
                           trigger ("add ONLY if 1-4 fail to separate the oracles").

Torch-free (numpy / scipy / PIL only). Points are `(N, 2)` in [0, 1], x-then-y.
"""

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-12

# Every field M0 computes and reports, in a fixed order so the layout is stable pipeline-wide.
FIELD_KEYS = ("nn_mean", "nn_cv", "cap_cv", "edge_align", "aniso", "pcf_peak", "pcf_pos")

# The subset M1/M2 actually condition on. Decided BY the M0 measurement, not by taste:
#   * `nn_mean` is out -- r = -0.957 with `nn_cv`, i.e. one axis wearing two hats. `nn_cv` is kept
#     because it is the measure with prior validation (it cleared the WVS<->GBN gate).
#   * `pcf_pos` is out -- it separates 17/21 pairs but adds no UNIQUE coverage (all 21 pairs stay
#     covered without it) while carrying the highest surviving correlation, r = 0.835 with `nn_cv`.
#     Still computed and reported as a diagnostic; just not a conditioning channel.
# These five separate 21/21 oracle pairs and map one-to-one onto the plan's descriptors 1-5.
CONDITIONING_KEYS = ("nn_cv", "cap_cv", "edge_align", "aniso", "pcf_peak")
# Extra per-icon diagnostics that are not part of the conditioning stack.
DIAG_KEYS = ("edge_align_strat",)
PRIMARY_KEY = "nn_cv"


# ---------------------------------------------------------------------------
# Per-point primitives
# ---------------------------------------------------------------------------

def drop_exact_duplicates(pts):
    """Remove coincident points, keeping first occurrences. Returns (pts, n_dropped).

    `train_control._fit_points_to_n` repairs a short point set by DUPLICATING existing points
    (never by inventing uniform-random ones, which cannot be undone). A duplicate has
    nearest-neighbour distance exactly 0, which is the extreme left tail of the NN distribution and
    would inflate `nn_cv` by ~7% per duplicate -- on the very statistic this axis rests on.

    Dropping them recovers the true value exactly (measured: -0.2% vs intact, identical to the
    genuine short set). This lives inside `point_primitives` rather than at each call site, because
    "remember to de-duplicate" as a distributed convention is precisely the kind of rule that gets
    forgotten in one place and silently corrupts a number.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts, 0
    _, keep = np.unique(pts, axis=0, return_index=True)
    if len(keep) == len(pts):
        return pts, 0
    keep = np.sort(keep)
    return pts[keep], int(len(pts) - len(keep))


def point_primitives(pts, k=8):
    """Per-point quantities every field descriptor pools from. One KD-tree for all of them.

    Returns dict with:
      d1     -- nearest-neighbour distance
      lam    -- local intensity (points per unit area) from the k-th neighbour: k / (pi r_k^2)
      s      -- implied local spacing 1/sqrt(lam)
      u      -- d1 / s, the DENSITY-INVARIANT normalised NN distance. This is the quantity whose
                dispersion separated WVS from GBN; normalising by the local spacing is what keeps
                it from simply re-reading the density gradient.
      gap    -- d1 / d_k (higher = more regular)
      dbl2   -- DEBIASED squared double-angle resultant over the k neighbour directions.

    On `dbl2`: the naive statistic |mean exp(2 i phi)| is what `control_v4_pilot/aniso_pilot.py`
    implements and marks BROKEN, and this is the reason. The resultant of k random phasors has
    expected squared length 1/k, so an isotropic point set reads sqrt(pi/4k) -- 0.31 at k=8, which
    is measured here and is indistinguishable from a genuinely structured set. Every oracle would
    have scored ~0.31 and the descriptor would have looked like signal while carrying none.
    Rayleigh's correction removes it exactly: A^2 = (k |R|^2 - 1) / (k - 1), which is 0 in
    expectation for isotropic and 1 for perfectly aligned. It is kept SQUARED per point (and so
    may be negative) because clipping before pooling would reintroduce a positive bias; the field
    takes the square root only after the window mean.
    """
    pts, _n_dup = drop_exact_duplicates(pts)
    n = len(pts)
    k_eff = max(1, min(k, n - 1))
    tree = cKDTree(pts)
    dists, idx = tree.query(pts, k=k_eff + 1)
    d1 = dists[:, 1]
    dk = dists[:, k_eff]
    lam = k_eff / (np.pi * np.maximum(dk, EPS) ** 2)
    s = 1.0 / np.sqrt(np.maximum(lam, EPS))

    # Double-angle over the k-NN directions. A set with no directional preference averages to ~0;
    # a set organised into lines/worms keeps a consistent axis and averages high.
    d = pts[idx[:, 1:]] - pts[:, None, :]          # (n, k, 2)
    phi = np.arctan2(d[:, :, 1], d[:, :, 0])
    r2 = np.abs(np.mean(np.exp(2j * phi), axis=1)) ** 2
    dbl2 = (k_eff * r2 - 1.0) / max(k_eff - 1, 1)   # Rayleigh-debiased; may be negative

    return {
        # The DEDUPED set. Callers must use this, not the array they passed in: every per-point
        # array below is aligned to it, and pooling a length-N value array over length-(N-dup)
        # points is a shape error at best and a silent misalignment at worst.
        "pts": pts,
        "n_dup": _n_dup,
        "d1": d1,
        "lam": lam,
        "s": s,
        "u": d1 / np.maximum(s, EPS),
        "gap": d1 / np.maximum(dk, EPS),
        "dbl2": dbl2,
        "tree": tree,
    }


def power_cell_areas(pts, weights=None, n_probe=48, rng=None):
    """Monte-Carlo POWER (Laguerre) cell areas -- the plan's descriptor 2.

    Why power and not Voronoi. Under a varying density the correct capacity notion is the power
    diagram with weights set from the local density; plain Voronoi cells are systematically larger
    in light regions, so a Voronoi area CV reads the density gradient as if it were capacity
    disorder. `control_v4_mix/spacing_regularity.voronoi_area_cv_norm` normalises post-hoc by
    multiplying area by local lambda, which corrects the mean but not the shape of the cells, and
    it was already observed to be noisy and to disagree with the other regularity measures. The
    plan names this explicitly as R4's objection and asks that it be fixed here rather than
    inherited.

    Exact power diagrams need a 3-D convex hull lift and are slow at 1024 points x 7 oracles x
    10k icons. Monte-Carlo assignment is O(n_probe * n log n), unbiased, and its variance is
    controlled by `n_probe`; the CV of the resulting areas is what we pool, and at 48 probes per
    point the estimator noise sits well below the between-oracle spread.

    Returns (areas, owners, probe_xy) with areas summing to 1.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    rng = np.random.RandomState(0) if rng is None else rng
    if weights is None:
        weights = np.zeros(n)
    weights = np.asarray(weights, dtype=np.float64)

    m = int(n_probe * n)
    probe = rng.rand(m, 2)
    tree = cKDTree(pts)
    # Power distance is |x - p|^2 - w. Query a candidate shortlist by Euclidean distance, then
    # re-rank by power distance -- with bounded weights the true power-owner is within the
    # shortlist, and this stays O(m log n) instead of O(m n).
    kq = min(n, 16)
    dd, ii = tree.query(probe, k=kq)
    if kq == 1:
        dd = dd[:, None]
        ii = ii[:, None]
    pw = dd ** 2 - weights[ii]
    owners = ii[np.arange(m), np.argmin(pw, axis=1)]
    counts = np.bincount(owners, minlength=n).astype(np.float64)
    return counts / max(m, 1), owners, probe


# ---------------------------------------------------------------------------
# Image-side primitives
# ---------------------------------------------------------------------------

def gradient_magnitude(gray01):
    """Sobel gradient magnitude of the source image, normalised to max 1."""
    from scipy import ndimage

    g = np.asarray(gray01, dtype=np.float64)
    gx = ndimage.sobel(g, axis=1, mode="nearest")
    gy = ndimage.sobel(g, axis=0, mode="nearest")
    mag = np.hypot(gx, gy)
    mx = mag.max()
    return mag / mx if mx > EPS else mag


def _bilinear_sample(field, pts):
    """Sample a (H, W) field at (N, 2) xy points in [0, 1]."""
    h, w = field.shape
    x = np.clip(pts[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip(pts[:, 1] * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    top = field[y0, x0] * (1 - wx) + field[y0, x1] * wx
    bot = field[y1, x0] * (1 - wx) + field[y1, x1] * wx
    return top * (1 - wy) + bot * wy


# ---------------------------------------------------------------------------
# Windowed pooling
# ---------------------------------------------------------------------------

def _window_stats(pts, values, G, window, min_count):
    """Sliding-window mean and CV of a per-point value on a (G, G) grid.

    Implemented as a box filter over splatted sums rather than a per-cell neighbour query: splat
    each point's value, value^2 and count into its own cell, then convolve all three with a box of
    the window width. Cost is independent of the window size and of N.
    """
    from scipy import ndimage

    ix = np.clip((pts[:, 0] * G).astype(np.int64), 0, G - 1)
    iy = np.clip((pts[:, 1] * G).astype(np.int64), 0, G - 1)

    cnt = np.zeros((G, G))
    s1 = np.zeros((G, G))
    s2 = np.zeros((G, G))
    np.add.at(cnt, (iy, ix), 1.0)
    np.add.at(s1, (iy, ix), values)
    np.add.at(s2, (iy, ix), values ** 2)

    # Window width in cells; "a few expected spacings" -- at G=32 and N=1024 one cell IS one
    # expected spacing, so `window` is directly in units of expected spacings.
    size = max(1, int(round(window)))
    if size % 2 == 0:
        size += 1
    kw = dict(size=size, mode="constant", cval=0.0)
    cnt_w = ndimage.uniform_filter(cnt, **kw) * size * size
    s1_w = ndimage.uniform_filter(s1, **kw) * size * size
    s2_w = ndimage.uniform_filter(s2, **kw) * size * size

    valid = cnt_w >= min_count
    c = np.maximum(cnt_w, EPS)
    mean = s1_w / c
    var = np.maximum(s2_w / c - mean ** 2, 0.0)
    cv = np.sqrt(var) / np.maximum(np.abs(mean), EPS)

    mean = np.where(valid, mean, np.nan)
    cv = np.where(valid, cv, np.nan)
    return mean, cv, valid, cnt_w


def _pcf_fields(pts, prim, G, window, min_count, k_pcf=32, u_max=2.5, n_bins=25,
                peak_lo=0.7, peak_hi=1.8, smooth_bins=1):
    """Local pair-correlation first-peak height and position -- descriptor 5.

    The PCF is a per-NEIGHBOURHOOD quantity, not a per-point one: a single point's 24 neighbour
    distances are far too few to locate a peak. But a per-IMAGE PCF is exactly what the plan
    forbids, since a global number cannot say *where*. So the histogram is accumulated per point
    into that point's own cell and then box-filtered over the window, which yields a PCF estimated
    from every point in the window (~25 points x 24 neighbours = ~600 distances per cell) while
    still being local. Same trick as `_window_stats`, one histogram bin at a time.

    Distances are normalised by each point's own local spacing before binning, so the estimate is
    density-invariant for the same reason `nn_cv` is -- otherwise the peak would simply track rho.
    In those units a Poisson process has g = 1 everywhere, a square lattice a tall sharp peak at
    u = 1, and a periodic dither a taller/narrower peak than an aperiodic one at the same density.
    That contrast is what separates `ordered` from `fs`.

    `k_pcf=24` reaches out to u ~ 2.8 for a typical point (a disc of radius u holds ~pi*u^2
    neighbours), so the peak search stops at u = 2.0 to stay clear of the truncation edge where the
    histogram is biased low by the finite neighbour count.
    """
    from scipy import ndimage

    n = len(pts)
    k_eff = max(1, min(k_pcf, n - 1))
    dists, _ = prim["tree"].query(pts, k=k_eff + 1)
    u = dists[:, 1:] / np.maximum(prim["s"][:, None], EPS)      # (n, k) density-normalised

    du = u_max / n_bins
    bi = np.floor(u / du).astype(np.int64)
    keep = (bi >= 0) & (bi < n_bins)

    ix = np.clip((pts[:, 0] * G).astype(np.int64), 0, G - 1)
    iy = np.clip((pts[:, 1] * G).astype(np.int64), 0, G - 1)
    iy_r = np.repeat(iy[:, None], k_eff, axis=1)
    ix_r = np.repeat(ix[:, None], k_eff, axis=1)

    hist = np.zeros((n_bins, G, G))
    np.add.at(hist, (bi[keep], iy_r[keep], ix_r[keep]), 1.0)
    cnt = np.zeros((G, G))
    np.add.at(cnt, (iy, ix), 1.0)

    size = max(1, int(round(window)))
    if size % 2 == 0:
        size += 1
    # uniform_filter returns a MEAN; rescale to window sums so `cnt_w` is a point count (the
    # scale cancels in g, but the validity threshold is in points).
    hist_w = ndimage.uniform_filter(hist, size=(1, size, size), mode="constant",
                                    cval=0.0) * size * size
    cnt_w = ndimage.uniform_filter(cnt, size=size, mode="constant", cval=0.0) * size * size

    # Poisson expectation in normalised units: lambda * 2*pi*u*du * s^2 = 2*pi*u*du, per centre.
    u_mid = (np.arange(n_bins) + 0.5) * du
    expect = cnt_w[None, :, :] * (2.0 * np.pi * u_mid * du)[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        g = hist_w / np.maximum(expect, EPS)

    # What this statistic does and does not claim, measured rather than assumed.
    #
    # It is a LATTICE-PERIODICITY detector: an exact lattice reads 5.26 while a jittered lattice
    # reads 1.75, white noise 1.92 and a line process 2.33 -- a factor 2.3-3.0 above every
    # aperiodic process. That is the contrast `ordered` vs `fs` needs.
    #
    # It is NOT a monotone "amount of structure" ranking, and white noise sitting slightly ABOVE a
    # jittered lattice is not a bug to be smoothed away. Each point's distances are divided by its
    # OWN locally-estimated spacing, which is what buys density-invariance; but for a clustered
    # process that estimate itself fluctuates, and a ratio of two fluctuating quantities
    # concentrates, putting a spurious peak in the histogram. Raising `k_pcf` (32 -> 48) or the
    # window (5 -> 9) changes nothing, which is how we know it is this and not sample noise.
    #
    # Smoothing is therefore left OFF by default. It does suppress the max-of-noise bias, but it
    # also crushed the lattice peak from 5.26 to 1.77 -- destroying the sharpness that is the entire
    # point. The bias is common to all oracles at matched N, so it offsets every column equally and
    # cancels in the paired between-oracle comparison, which is the only comparison made.
    if smooth_bins > 1:
        g = ndimage.uniform_filter1d(g, size=smooth_bins, axis=0, mode="nearest")

    sel = (u_mid >= peak_lo) & (u_mid <= peak_hi)
    gs = g[sel]
    arg = np.argmax(gs, axis=0)
    peak = np.take_along_axis(gs, arg[None, :, :], axis=0)[0]
    pos = u_mid[sel][arg]

    valid = cnt_w >= min_count
    return (np.where(valid, peak, np.nan), np.where(valid, pos, np.nan))


def descriptor_fields(pts, gray01, G=32, window=5, min_count=12, k=8, n_probe=48, seed=0,
                      grad=None, with_pcf=True):
    """Full local descriptor field stack for one point set.

    Returns `(fields, diag)` where `fields` maps each key in FIELD_KEYS to a `(G, G)` float array
    (NaN where the window holds too few points to estimate), plus a boolean `valid` mask, and
    `diag` holds per-icon scalars that are not part of the conditioning stack.

    `window=5` at G=32, N=1024 means a 5x5-cell window ~ 5 expected spacings across, holding ~25
    points -- enough for a stable mean and a usable CV, small enough that a two-region request has
    somewhere to put the boundary.
    """
    pts = np.asarray(pts, dtype=np.float64)
    rng = np.random.RandomState(seed)
    prim = point_primitives(pts, k=k)
    pts = prim["pts"]          # rebind: duplicates dropped, all per-point arrays align to this
    if grad is None:
        grad = gradient_magnitude(gray01)

    fields = {}

    # 1. Local NN statistics (density-normalised). PRIMARY.
    nn_mean, nn_cv, valid, cnt_w = _window_stats(pts, prim["u"], G, window, min_count)
    fields["nn_mean"] = nn_mean
    fields["nn_cv"] = nn_cv

    # 2. Local capacity CV from power cells. Weights from local density: a point in a dense region
    #    legitimately owns a smaller cell, and the power weight is what encodes that.
    lam = prim["lam"]
    w = (1.0 / np.maximum(lam, EPS)) / np.pi        # r^2-scale weight
    w = w - w.mean()
    areas, _, _ = power_cell_areas(pts, weights=w, n_probe=n_probe, rng=rng)
    # Normalise each cell by the area the local density says it should have, so the CV reads
    # capacity DISORDER and not the density gradient.
    expect = 1.0 / np.maximum(lam, EPS)
    ratio = areas / np.maximum(expect / max(expect.sum(), EPS), EPS)
    _, cap_cv, _, _ = _window_stats(pts, ratio, G, window, min_count)
    fields["cap_cv"] = cap_cv

    # 3. Edge / structure alignment, normalised against rho.
    rho = np.clip(1.0 - np.asarray(gray01, dtype=np.float64), 0.0, 1.0)
    g_at_pts = _bilinear_sample(grad, pts)
    # What tone alone predicts: rho-weighted mean gradient, computed locally so the normalisation
    # is itself spatially varying (a global constant would reintroduce the attempt-1 confound).
    pred = _rho_weighted_grad_field(grad, rho, G, window)
    ea_mean, _, _, _ = _window_stats(pts, g_at_pts, G, window, min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        fields["edge_align"] = ea_mean / pred

    # 4. Local anisotropy (double-angle). Free -- reuses the same k-NN query. Pool the debiased
    #    SQUARED statistic, then take the root, so isotropic oracles land at ~0 instead of ~0.31.
    aniso2, _, _, _ = _window_stats(pts, prim["dbl2"], G, window, min_count)
    fields["aniso"] = np.sqrt(np.clip(aniso2, 0.0, None))

    # 5. Local PCF first peak -- periodicity, which 1-4 do not measure.
    if with_pcf:
        fields["pcf_peak"], fields["pcf_pos"] = _pcf_fields(pts, prim, G, window, min_count)
    else:
        fields["pcf_peak"] = np.full((G, G), np.nan)
        fields["pcf_pos"] = np.full((G, G), np.nan)

    fields["valid"] = valid
    fields["count"] = cnt_w

    diag = {"edge_align_strat": _edge_align_stratified(g_at_pts, pts, grad, rho)}
    return fields, diag


def _rho_weighted_grad_field(grad, rho, G, window, min_mass_frac=0.02, min_pred_frac=0.10):
    """The gradient a point would see if it were placed by tone alone, pooled over the SAME window
    as the measured value. Dividing one by the other turns "points sit on edges" into "points sit
    on edges MORE than being dark explains".

    Three things here are load-bearing, each learned from a measured failure.

    The window must MATCH the measured side: that side is a windowed mean, so a per-cell normaliser
    would divide a 5x5-cell quantity by a 1-cell one and the ratio would carry the mismatch rather
    than the alignment.

    Cells whose window holds almost no ink get NaN, not a ratio. With a bare epsilon guard a window
    at rho ~ 0 divides by ~1e-12 and yields ~1e9, which then swamps every downstream mean.

    Cells whose window has ink but essentially NO GRADIENT also get NaN, and this is the subtle
    one. Inside a solid black icon region the image gradient is genuinely zero, so "alignment
    relative to structure" is not large there -- it is UNDEFINED, because there is no structure to
    align to. Flat icons are mostly such regions, so masking on mass alone still left the pooled
    edge_align means at ~1e10 and the descriptor separated 0/21 oracle pairs. The floor is absolute
    (a fraction of the image's own rho-weighted mean gradient), not relative to the cell, because a
    relative floor cannot distinguish "no structure" from "weak structure".
    """
    from scipy import ndimage

    h, w = grad.shape
    ys = np.clip(np.arange(h) * G // h, 0, G - 1)
    xs = np.clip(np.arange(w) * G // w, 0, G - 1)
    yy = np.repeat(ys, w)
    xx = np.tile(xs, h)
    num = np.zeros((G, G))
    den = np.zeros((G, G))
    np.add.at(num, (yy, xx), (grad * rho).ravel())
    np.add.at(den, (yy, xx), rho.ravel())

    size = max(1, int(round(window)))
    if size % 2 == 0:
        size += 1
    kw = dict(size=size, mode="constant", cval=0.0)
    num_w = ndimage.uniform_filter(num, **kw)
    den_w = ndimage.uniform_filter(den, **kw)

    mass_ok = den_w > min_mass_frac * den.sum() / (G * G)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.where(mass_ok, num_w / np.maximum(den_w, EPS), np.nan)
    g_ref = float(num.sum() / max(den.sum(), EPS))          # image-wide rho-weighted mean gradient
    return np.where(np.isfinite(pred) & (pred > min_pred_frac * g_ref), pred, np.nan)


def _edge_align_stratified(g_at_pts, pts, grad, rho, n_strata=8):
    """Tone-stratified edge alignment -- DIAGNOSTIC ONLY, does not gate anything.

    Attempt 1 of the WVS<->GBN Gate 0 found that the raw edge-alignment difference between those
    two oracles was a tone-transfer effect: GBN simply puts more points on dark pixels, which is
    reproducible by editing rho and running a single oracle, i.e. rho-decomposable and therefore
    not a contribution. Comparing points only against other points in the same intensity stratum
    removes that channel. Reported alongside the raw value so the two can be read against each
    other for the new oracles; it is not a blocker here.
    """
    rho_at_pts = _bilinear_sample(rho, pts)
    edges = np.quantile(rho_at_pts, np.linspace(0, 1, n_strata + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    out = []
    flat_rho = rho.ravel()
    flat_grad = grad.ravel()
    for i in range(n_strata):
        lo, hi = edges[i], edges[i + 1]
        sel = (rho_at_pts > lo) & (rho_at_pts <= hi)
        if sel.sum() < 8:
            continue
        px = (flat_rho > lo) & (flat_rho <= hi)
        if px.sum() < 8:
            continue
        out.append(float(g_at_pts[sel].mean() / max(flat_grad[px].mean(), EPS)))
    return float(np.mean(out)) if out else float("nan")


def pool_fields(fields):
    """Reduce a field stack to one scalar per descriptor (valid cells only).

    M0's separation test operates on these scalars; the fields themselves are what M1 stores and
    M2 conditions on.
    """
    valid = fields["valid"]
    out = {}
    for key in FIELD_KEYS:
        v = fields[key][valid]
        v = v[np.isfinite(v)]
        out[key] = float(v.mean()) if v.size else float("nan")
    out["valid_frac"] = float(valid.mean())
    return out
