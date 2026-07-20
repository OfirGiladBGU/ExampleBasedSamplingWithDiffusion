"""Density-warped pair-correlation-function (PCF) surrogate -- the one genuinely new piece.

Training-only surrogate. The REPORTED blue-noise diagnostic is the hard numpy/FFT
``plot_visual_m5_spectrum`` / ``compute_m5_spatial_measure`` in
``stippling_metrics_advance.py``; this soft PCF is only an optimization signal and a
target-matching statistic (guardrails #1/#2).

Pipeline (README "Loss specifications"):

    D       = cdist(pts, pts)                 # pairwise distances, gradient flows here
    D_x     = N * local_rho / mean_rho        # M5 density warp (Wei & Wang 2011)
    D_warp  = D * sqrt(D_x)                    # per-source-point warp; sqrt(D_x) stop-grad by default
    pcf     = soft_bin(D_warp, edges)         # Gaussian kernel-splat, self-pair excluded
    L_pcf   = || pcf - pcf_target ||          # target = EMPIRICAL PCF of a GBN reference set

The warp makes the statistic capacity-aware and scale-invariant: after warping, a
blue-noise set has a PCF that dips near 0 and settles to a plateau; clumping shows a peak
near 0. Matching an *aggregate* GBN signature (never a per-sample target) is what keeps the
student off the master's ceiling.
"""

import math

import torch

from control_gt_free.losses.soft_membership import sample_local_rho


def default_edges(n_bins=48, r_max=2.5, device="cpu", dtype=torch.float32):
    """Radial bin edges in the *warped* domain. r_max ~2.5 covers the first shells.

    In the warped domain the nearest-neighbour packing distance is
    ``r_pack = sqrt(2/sqrt(3)) ~ 1.075`` (see ``compute_m5_spatial_measure``), so bins out
    to ~2.5 capture the exclusion dip and the settle-to-plateau region.
    """
    return torch.linspace(0.0, float(r_max), int(n_bins) + 1, device=device, dtype=dtype)


def _centers_and_width(edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1:] - edges[:-1]).mean()
    return centers, width


def soft_bin(dist, edges, sigma=None, mask=None, normalize=True, eps=1e-8):
    """Differentiable radial histogram: each distance splats a Gaussian across bin centers.

    Parameters
    ----------
    dist : (B, N, M) pairwise distances (warped).
    edges : (n_bins + 1,) radial bin edges.
    sigma : Gaussian kernel width; defaults to one bin width.
    mask  : (B, N, M) bool/float; entries with 0 are excluded (e.g. the self-distance
            diagonal). If None, no masking.
    normalize : make each sample's histogram sum to 1 (a distribution) so the target/pred
            comparison is invariant to the pair count. The same normalization is applied to
            the precomputed target, so matching is well-posed.

    Returns
    -------
    pcf : (B, n_bins).
    """
    centers, width = _centers_and_width(edges)
    if sigma is None:
        sigma = float(width)
    sigma = max(float(sigma), 1e-6)

    d = dist.unsqueeze(-1)                         # (B, N, M, 1)
    c = centers.view(1, 1, 1, -1)                  # (1, 1, 1, n_bins)
    bumps = torch.exp(-((d - c) ** 2) / (2.0 * sigma * sigma))  # (B, N, M, n_bins)

    if mask is not None:
        bumps = bumps * mask.unsqueeze(-1)

    hist = bumps.sum(dim=(1, 2))                   # (B, n_bins)
    if normalize:
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
    return hist


def density_warp_distances(points, rho_map, mean_rho=None, stop_grad_warp=True, eps=1e-8):
    """Return warped pairwise distances D_warp (B, N, N) and an off-diagonal mask.

    D_x = N * local_rho / mean_rho  (per source point); D_warp[i,j] = ||s_i - s_j|| * sqrt(D_x_i).
    ``stop_grad_warp`` detaches the sqrt(D_x) factor (README default: gradient flows only
    through cdist). Flip it on only if the warp itself needs to move (needs a
    differentiable rho lookup, which ``sample_local_rho`` already provides).
    """
    b, n, _ = points.shape
    D = torch.cdist(points, points)               # (B, N, N), grad flows here

    local_rho = sample_local_rho(points, rho_map)  # (B, N)
    if mean_rho is None:
        mean_rho = local_rho.mean(dim=1, keepdim=True)
    else:
        mean_rho = mean_rho.view(b, 1)
    D_x = n * local_rho / mean_rho.clamp_min(eps)  # (B, N)
    warp = torch.sqrt(D_x.clamp_min(eps))          # (B, N)
    if stop_grad_warp:
        warp = warp.detach()

    D_warp = D * warp.unsqueeze(2)                 # warp by the source point's factor

    mask = 1.0 - torch.eye(n, device=points.device, dtype=points.dtype).unsqueeze(0)
    return D_warp, mask


def pcf_from_points(points, rho_map, edges, sigma=None, mean_rho=None,
                    stop_grad_warp=True, normalize=True):
    """Convenience: points (B, N, 2) + rho map -> density-warped soft PCF (B, n_bins).

    Used both to build the empirical GBN target (``precompute_gbn_bar.py``) and to score the
    model output during training, so target and prediction share one code path.
    """
    D_warp, mask = density_warp_distances(
        points, rho_map, mean_rho=mean_rho, stop_grad_warp=stop_grad_warp
    )
    return soft_bin(D_warp, edges, sigma=sigma, mask=mask, normalize=normalize)


def loss_pcf(points, rho_map, pcf_target, edges, sigma=None, mean_rho=None,
             stop_grad_warp=True, normalize=True, p=2):
    """L_pcf = || pcf(points) - pcf_target ||_p. ``pcf_target`` is (n_bins,) or (B, n_bins)."""
    pcf = pcf_from_points(
        points, rho_map, edges, sigma=sigma, mean_rho=mean_rho,
        stop_grad_warp=stop_grad_warp, normalize=normalize,
    )
    tgt = pcf_target.to(pcf.device, pcf.dtype)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)
    diff = pcf - tgt
    if p == 1:
        l = diff.abs().mean()
    else:
        l = (diff ** 2).mean()
    return l, {"pcf": pcf.detach()}
