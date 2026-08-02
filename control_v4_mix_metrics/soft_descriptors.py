"""Differentiable descriptor fields, for a descriptor-CONSISTENCY loss on the decoded x0.

Closes the loop on the control signal. Today the model learns the descriptor -> output link only
implicitly, by imitating paired (descriptor, target) data. With these, the loss can measure the
descriptor the model ACTUALLY produced against the one it was asked for, and optimise the control
objective directly.

How exact is the "surrogate"? Mostly: exactly exact.
------------------------------------------------------------------------------
The obvious worry is that a differentiable approximation drifts from the metric being reported --
the same failure as an estimator that looks informative and carries nothing. Measured against
`descriptor_fields.py`, three of the four here are not approximations at all:

  nn_cv       EXACT. `torch.topk` selects neighbours discretely, but the returned DISTANCES carry
              gradient w.r.t. the coordinates. The selection is piecewise constant, so the value
              equals the numpy version and the gradient is a valid subgradient almost everywhere.
              No temperature, no softmin, nothing to tune.
  aniso       EXACT, same argument: neighbour indices from topk, then a differentiable gather of
              the direction vectors. Rayleigh debiasing is plain arithmetic.
  edge_align  EXACT -- but only once rho_map is supplied. The sampling side was never the issue;
              the NORMALISER is. descriptor_fields divides by a RHO-WEIGHTED window mean with
              masking, and an unweighted pool is a different quantity: calibration measured r = 0.34
              before this was fixed. It is exact with the weighting reproduced.
  pcf_peak    CONDITIONING ONLY -- do NOT put it in a loss. Its VALUES are faithful (r = 0.9996 at
              sigma_bins = 0.08), but its GRADIENT is not: the per-neighbour normaliser divides by
              the summed bin weights, and at that sigma the Gaussian is 0.008 wide against 0.1-wide
              bins, so a u landing between bin centres divides by ~3e-9 and amplifies gradients by
              ~3e8. Measured as NaN in |g_desc| on ~5% of steps while the loss VALUE stayed finite.
              Making it loss-safe needs many more bins so the Gaussian is always well sampled, which
              is memory-bound in the (B, N, k, bins) tensor. Until then use nn_cv + aniso.
              APPROXIMATE, in one place only: hard histogram binning is replaced by a Gaussian
              soft-binning of width `pcf_sigma_bins`. Everything else matches. Converges to the
              exact statistic as sigma -> 0, at the cost of gradient sharpness.

`cap_cv` is deliberately NOT implemented. Power-cell areas need a Monte-Carlo argmin assignment,
and softening that is a genuine approximation with its own temperature -- the one descriptor where
optimising the surrogate could plausibly diverge from the reported metric. It is also the weakest
separator in M0 (9/21 pairs). Add it only if the other four prove insufficient, and only after
`calibrate_soft_descriptors.py` shows the soft and hard versions actually track.

Cost
----
A (B, N, N) pairwise distance matrix: at N=1024, B=8 that is 8M entries, ~32 MB in fp32. The
existing C2 density loss already runs `torch.cdist` at this scale in `gaussian_kde_map`, so this is
within the established budget.

Apply it at LOW t only. A descriptor computed on a high-noise x0_pred measures noise, not style --
the same reason `density_match_loss` is masked by `--density-loss-t-frac`.
"""

import torch
import torch.nn.functional as F

EPS = 1e-12
# Added INSIDE every sqrt. d/dx sqrt(x) = 1/(2 sqrt(x)) is INFINITE at x = 0, and zero is not a rare
# input here: a window whose points all share the same u has var exactly 0, and `aniso` clamps its
# argument at 0 by construction. Both produced inf gradients that turned every parameter NaN within
# a few thousand steps (observed: |g_main|=nan |g_desc|=nan, then cKDTree failing on NaN points).
# The value shift is ~1e-6 in a quantity of order 0.1-1; the gradient fix is the point.
SQRT_EPS = 1e-12
# Floor on neighbour distances, in [0,1] coordinate units. lam = k / (pi dk^2) has derivative
# -2k/(pi dk^3), which reaches 5e36 at dk = 1e-12 -- one coincident pair in a predicted point set is
# enough to swamp fp32. At 1e-3 (about 3% of the 1/32 expected spacing) the derivative is 5e9:
# large, but finite and harmless. Predicted point sets DO collapse points early in training, and the
# ground truth now contains deliberate duplicates from short-set repair, so this is not hypothetical.
MIN_DIST = 1e-3
# Floors chosen so no LOCAL DERIVATIVE can reach inf, which matters more than keeping values finite.
# `nan_to_num` on the output does not help: masking an invalid cell gives it zero incoming gradient,
# and 0 * inf = NaN, so a single inf local derivative poisons the whole backward pass regardless of
# the mask. Empirically |g_desc| was NaN on ~11% of steps until these were physical rather than
# epsilon-sized.
#   MIN_COUNT_DENOM: a window with no points divides by 1 instead of 1e-12 (1e12 -> 1 derivative).
#   MIN_MEAN_DENOM : u is O(1), so 1e-3 is far below any real mean but caps d(cv)/d(mean) at 1e6.
MIN_COUNT_DENOM = 1.0
MIN_MEAN_DENOM = 1e-3

# Order matches descriptor_fields.CONDITIONING_KEYS minus cap_cv (see module note).
SOFT_KEYS = ("nn_cv", "edge_align", "aniso", "pcf_peak")


def _box_filter(x, size):
    """Sum over a size x size window, same padding. x: (B, C, G, G)."""
    if size <= 1:
        return x
    C = x.shape[1]
    kernel = x.new_ones(C, 1, size, size)
    return F.conv2d(x, kernel, padding=size // 2, groups=C)


def _splat(coords, values, G):
    """Scatter per-point values into their own (G, G) cell. coords (B,N,2), values (B,C,N)."""
    B, C, N = values.shape
    ix = (coords[..., 0] * G).long().clamp_(0, G - 1)
    iy = (coords[..., 1] * G).long().clamp_(0, G - 1)
    flat = (iy * G + ix).unsqueeze(1).expand(B, C, N)          # (B, C, N)
    out = values.new_zeros(B, C, G * G)
    out.scatter_add_(2, flat, values)
    return out.view(B, C, G, G)


def _window_mean_cv(coords, values, G, size, min_count):
    """Windowed mean and CV of a per-point value. Mirrors descriptor_fields._window_stats."""
    B, N = values.shape
    ones = torch.ones_like(values)
    stacked = torch.stack([ones, values, values * values], dim=1)   # (B, 3, N)
    s = _box_filter(_splat(coords, stacked, G), size)
    cnt, s1, s2 = s[:, 0:1], s[:, 1:2], s[:, 2:3]
    c = cnt.clamp(min=MIN_COUNT_DENOM)
    mean = s1 / c
    var = (s2 / c - mean * mean).clamp(min=0.0)
    cv = (var + SQRT_EPS).sqrt() / mean.abs().clamp(min=MIN_MEAN_DENOM)
    valid = (cnt >= min_count).float()
    return mean, cv, valid


def point_primitives_soft(coords, k=8, k_pcf=32):
    """Per-point d1, local spacing, u = d1/s, and the debiased double-angle -- all differentiable.

    `torch.topk` is what makes this exact rather than a surrogate: the neighbour SELECTION is
    discrete (piecewise constant, so it contributes no gradient), but the returned distances and the
    gathered direction vectors are smooth functions of the coordinates.
    """
    B, N, _ = coords.shape
    k_eff = max(1, min(k, N - 1))
    # ONE cdist for both uses. The PCF needs far more neighbours than the spacing statistics
    # (k_pcf=32 vs k=8); reusing the k=8 list left the peak-search range undersampled and dropped
    # soft-vs-exact agreement to r = 0.67.
    k_max = max(k_eff, min(k_pcf, N - 1))
    d = torch.cdist(coords, coords)                                     # (B, N, N)
    # EXCLUDE SELF EXPLICITLY rather than assuming it lands in topk column 0. cdist self-distances
    # are not exactly 0 in fp32, so if any other point is nearer than that error, THAT point takes
    # column 0 and self takes column 1 -- making delta = (0,0) below and atan2(0,0) return a NaN
    # gradient. detect_anomaly named Atan2Backward0, and it fired even on uniformly random points
    # where genuine coincidences do not occur.
    eye = torch.eye(N, dtype=torch.bool, device=coords.device).unsqueeze(0)
    d = d.masked_fill(eye, float("inf"))
    vals, idx = torch.topk(d, k_max, dim=-1, largest=False)             # self is never selected now
    d1 = vals[..., 0]
    dk = vals[..., k_eff - 1]

    lam = k_eff / (torch.pi * dk.clamp(min=MIN_DIST) ** 2)
    s = lam.clamp(min=EPS).rsqrt()
    u = d1.clamp(min=0.0) / s.clamp(min=MIN_DIST * 0.1)

    nb = torch.gather(coords.unsqueeze(1).expand(B, N, N, 2), 2,
                      idx[..., :k_eff].unsqueeze(-1).expand(B, N, k_eff, 2))
    delta = nb - coords.unsqueeze(2)                                    # (B, N, k, 2)

    # A COINCIDENT neighbour is legitimate here -- short point sets are repaired by duplication and
    # an undertrained model collapses points -- but atan2(0, 0) has a 0/0 gradient. Substitute a
    # constant for those entries (torch.where gives them zero gradient, so the constant direction
    # never reaches the loss) and drop them from the average rather than letting them vote.
    ok = ((delta * delta).sum(-1, keepdim=True) > MIN_DIST ** 2)
    delta_safe = torch.where(ok, delta, torch.full_like(delta, MIN_DIST))
    phi = torch.atan2(delta_safe[..., 1], delta_safe[..., 0])

    w = ok.squeeze(-1).to(delta.dtype)                                  # (B, N, k)
    n_used = w.sum(-1).clamp(min=1.0)
    # |mean exp(2i phi)|^2 without complex tensors, over the usable neighbours only
    c2 = (torch.cos(2 * phi) * w).sum(-1) / n_used
    s2 = (torch.sin(2 * phi) * w).sum(-1) / n_used
    r2 = c2 * c2 + s2 * s2
    # Rayleigh debias against the ACTUAL neighbour count: E[|R|^2] = 1/n for n random phasors, so a
    # fixed k where some were dropped would under-correct and read as false anisotropy.
    dbl2 = (n_used * r2 - 1.0) / (n_used - 1.0).clamp(min=1.0)
    return {"d1": d1, "s": s, "u": u, "dbl2": dbl2, "vals": vals, "k_eff": k_eff}


def soft_pcf_peak(prim, coords, G, size, min_count, u_max=2.5, n_bins=25,
                  peak_lo=0.7, peak_hi=1.8, sigma_bins=0.75, k_pcf=32):
    """Local PCF first-peak height with Gaussian soft-binning (the one approximation here)."""
    vals = prim["vals"]
    k_use = min(k_pcf, vals.shape[-1] - 1)
    u = vals[..., 1:k_use + 1] / prim["s"].unsqueeze(-1).clamp(min=EPS)  # (B, N, k)
    du = u_max / n_bins
    centres = (torch.arange(n_bins, device=u.device, dtype=u.dtype) + 0.5) * du
    sigma = max(sigma_bins, 1e-3) * du
    w = torch.exp(-0.5 * ((u.unsqueeze(-1) - centres) / sigma) ** 2)     # (B, N, k, bins)
    # descriptor_fields discards neighbours beyond u_max outright; without this they leak Gaussian
    # tails into the top bins and inflate the peak.
    w = w * (u < u_max).to(w.dtype).unsqueeze(-1)
    # Normalise PER NEIGHBOUR so each contributes exactly 1 across the bins, as hard binning does.
    # The analytic 1/(sigma*sqrt(2pi))*du factor only sums to 1 when sigma >> bin width; below that
    # the Gaussian is sampled at bin CENTRES and most neighbours fall between them, so the histogram
    # empties out. That aliasing is why agreement got WORSE as sigma narrowed (r: 0.80 at 0.35 ->
    # -0.17 at 0.08) instead of converging on the exact statistic. With this, sigma -> 0 approaches
    # hard binning correctly and sigma is a pure smoothness knob again.
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
    per_point = w.sum(2).permute(0, 2, 1)                                # (B, bins, N)

    hist = _box_filter(_splat(coords, per_point, G), size)               # (B, bins, G, G)
    cnt = _box_filter(_splat(coords, torch.ones_like(per_point[:, :1]), G), size)
    expect = cnt * (2 * torch.pi * centres * du).view(1, -1, 1, 1)
    g = hist / expect.clamp(min=1e-6)

    sel = (centres >= peak_lo) & (centres <= peak_hi)
    peak = g[:, sel].max(dim=1, keepdim=True).values                     # subgradient through max
    valid = (cnt[:, :1] >= min_count).float()
    return peak, valid


def soft_descriptor_fields(coords, grad_map, rho_map=None, G=32, window=5, min_count=12, k=8,
                           k_pcf=32, keys=SOFT_KEYS, stats=None, **pcf_kwargs):
    """Differentiable (B, K, G, G) descriptor field, normalised exactly as the dataset normalises.

    `coords`   (B, N, 2) in [0, 1], x then y -- e.g. `offsets_to_coords_gpu(x0_pred, ...)`.
    `grad_map` (B, 1, H, W) Sobel magnitude of the source image, as descriptor_fields uses.
    `stats`    dict key -> (lo, hi) from DESCRIPTOR_STATS.json. Required: the loss must live in the
               same normalised space as the conditioning, or the two are not comparable.

    Returns (fields, valid) with fields (B, K, G, G) in [0, 1] and valid (B, 1, G, G).
    """
    size = int(window) | 1
    prim = point_primitives_soft(coords, k=k, k_pcf=k_pcf)
    out, valid = [], None

    for key in keys:
        if key == "nn_cv":
            _, cv, v = _window_mean_cv(coords, prim["u"], G, size, min_count)
            f = cv
        elif key == "aniso":
            m, _, v = _window_mean_cv(coords, prim["dbl2"], G, size, min_count)
            f = (m.clamp(min=0.0) + SQRT_EPS).sqrt()
        elif key == "edge_align":
            if rho_map is None:
                raise ValueError("edge_align needs rho_map = clip(1 - gray, 0, 1)")
            g_at = _sample_bilinear(grad_map, coords)                     # (B, N)
            m, _, v = _window_mean_cv(coords, g_at, G, size, min_count)
            pred, pred_ok = _rho_weighted_grad(grad_map, rho_map, G, size)
            f = m / pred.clamp(min=EPS)
            v = v * pred_ok
        elif key == "pcf_peak":
            f, v = soft_pcf_peak(prim, coords, G, size, min_count, k_pcf=k_pcf, **pcf_kwargs)
        else:
            raise ValueError(f"no differentiable implementation for '{key}' (see module note)")
        valid = v if valid is None else valid * v
        if stats is not None:
            lo, hi = stats[key]
            f = ((f - lo) / max(hi - lo, 1e-9)).clamp(0.0, 1.0)
        out.append(f)
    return torch.cat(out, dim=1), valid


def _sample_bilinear(field, coords):
    """Sample (B,1,H,W) at (B,N,2) xy in [0,1] -> (B,N). grid_sample wants [-1,1]."""
    grid = (coords * 2.0 - 1.0).unsqueeze(1)                              # (B, 1, N, 2)
    s = F.grid_sample(field, grid, mode="bilinear", align_corners=False)
    return s[:, 0, 0, :]


def _rho_weighted_grad(grad_map, rho_map, G, size, min_mass_frac=0.02, min_pred_frac=0.10):
    """The gradient tone alone predicts: a RHO-WEIGHTED window mean, with the same masking.

    The weighting is the whole point of the descriptor -- dividing by an unweighted mean gradient
    measures something else entirely (measured: soft-vs-exact r fell to 0.34). Cells whose window
    holds almost no ink, or essentially no gradient, are masked exactly as descriptor_fields does:
    inside a solid region "alignment relative to structure" is undefined, not large.
    """
    num = F.adaptive_avg_pool2d(grad_map * rho_map, (G, G))
    den = F.adaptive_avg_pool2d(rho_map, (G, G))
    num_w = _box_filter(num, size)
    den_w = _box_filter(den, size)
    mass_ok = den_w > min_mass_frac * den.mean(dim=(2, 3), keepdim=True)
    pred = num_w / den_w.clamp(min=EPS)
    g_ref = num.sum(dim=(2, 3), keepdim=True) / den.sum(dim=(2, 3), keepdim=True).clamp(min=EPS)
    ok = mass_ok & (pred > min_pred_frac * g_ref)
    return pred, ok.to(pred.dtype)


def descriptor_consistency_loss(coords, grad_map, requested, rho_map=None, valid_mask=None,
                                stats=None, keys=SOFT_KEYS, weights=None, **kwargs):
    """Mean squared error between the descriptor ACHIEVED and the descriptor REQUESTED.

    `requested` is the (B, K_all, G, G) conditioning field; the channels named by `keys` are
    selected from it, so the loss compares like with like in normalised space.

    Cells where too few points landed are excluded: a descriptor estimated from three points is
    noise, and asking the model to match noise teaches nothing.
    """
    achieved, valid = soft_descriptor_fields(coords, grad_map, rho_map=rho_map, keys=keys,
                                            stats=stats, **kwargs)
    if valid_mask is not None:
        valid = valid * valid_mask
    # Replace non-finite cells BEFORE masking. `err * valid` does not neutralise them: NaN * 0 is
    # NaN, so a single bad cell would poison the whole reduction.
    achieved = torch.nan_to_num(achieved, nan=0.0, posinf=0.0, neginf=0.0)
    err = (achieved - requested) ** 2
    if weights is not None:
        err = err * torch.as_tensor(weights, device=err.device,
                                    dtype=err.dtype).view(1, -1, 1, 1)
    denom = (valid.sum() * err.shape[1]).clamp(min=1.0)
    return (err * valid).sum() / denom
