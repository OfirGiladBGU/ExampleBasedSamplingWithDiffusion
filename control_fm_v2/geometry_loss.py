"""Geometry loss on the one-step-decoded point set (control_fm_v2).

Why
---
Velocity-MSE is a *per-point regression* error: "did you predict the right direction
here?". Two point configurations can share a velocity error and differ wildly in spacing
quality, so the model can drive v-MSE down while still clumping. Nothing in the velocity
loss ever says "these two points are too close together". This module supplies exactly
that missing signal, at nearly zero cost -- no ODE solve, no trajectory unroll.

Mechanism
---------
    x0_pred = x_t - t * v_pred            # exact inverse of the linear interpolant
    points  = offsets_to_coords(x0_pred)  # grid_center + offset / G
    L_geo   = lambda * w(t) * spacing(points)  [+ lambda_cap * w(t) * capacity(points)]

`w(t)` must concentrate on LOW t: x0_pred is a rough guess near the noise end (t -> 1)
and accurate near data (t -> 0). Applying the term uniformly feeds the model noise.

Gradient subtlety (drives the default)
--------------------------------------
Because x0 = x_t - t*v, we have  dL/dv_pred = dL/dx0 * (-t).  The geometry gradient is
proportional to t and therefore VANISHES as t -> 0, whatever w(t) says. The effective
signal is w(t) * t, so the useful band is a middling-low t. That is precisely why a hard
mask (flat on [0, t_max)) beats a smooth (1-t)^k ramp here: the ramp puts its largest
weight exactly where the gradient is smallest. Default is the mask.

Spacing term (target-free -- the default)
-----------------------------------------
Work in the density-warped domain (Wei & Wang 2011; metric M5). Warp each pairwise
distance by sqrt(D_x_i), with D_x_i = N * rho(s_i) / mean_rho, so the criterion becomes
capacity-aware: dense regions are not punished for being dense. In that domain a
blue-noise set has nearest-neighbour distance r_pack = sqrt(2/sqrt(3)) ~ 1.0746 and its
pair-correlation dips to ~0 below it; clumping shows up as mass near zero. We penalise
that mass directly with a one-sided hinge:

    L_spacing = mean_{i != j} [ relu(r_target - D_warp_ij) / r_target ]^2

No reference set, no precomputed target, no histogram -- so it needs no ``gbn_bar.pt``
and costs one cdist. (The target-matching alternative lives in
``control_gt_free.losses.loss_pcf`` and requires ``eval/precompute_gbn_bar.py``.)

Cautions
--------
* Auxiliary ONLY. Keep ``weight`` modest; watch that the velocity loss does not degrade.
  This never replaces the velocity loss -- it shapes the output, it does not learn the field.
* Training-only surrogate. Report the hard scipy/FFT validators in
  ``utils/stippling_metrics_advance.py``, never this number.
* Interacts with min-SNR (Tier 1.3): both reweight across t. Ablate separately.
* Training-only => deliberately NOT part of ``arch.ARCH_KEYS``. It changes no weight shape,
  so a checkpoint trained with it stays loadable by a sampler that knows nothing about it.
"""

import math

import torch
import torch.nn.functional as F

# Reuse the audited primitives rather than fork them. These are pure functions of
# (points, density); nothing in control_gt_free's training loop comes along.
from control_gt_free.losses.soft_membership import (
    density_grid,
    density_to_rho,
    offsets_to_coords,
    sample_local_rho,
    soft_membership,
)

# Nearest-neighbour distance of an optimal (hexagonal) packing in the warped domain --
# the same constant `compute_m5_spatial_measure` uses as r_max.
R_PACK = math.sqrt(2.0 / math.sqrt(3.0))  # ~1.0746
_EPS = 1e-8


def t_weight(t, mode="hard", t_max=0.4, ramp_k=2.0):
    """Per-sample w(t) in [0, 1]. See the gradient note in the module docstring."""
    if mode == "off":
        return torch.ones_like(t)
    if mode == "hard":
        return (t < float(t_max)).to(t.dtype)
    if mode == "ramp":
        return (1.0 - t).clamp_min(0.0) ** float(ramp_k)
    raise ValueError("unknown t_mode %r (expected 'hard' | 'ramp' | 'off')" % (mode,))


def warped_pair_distances(points, rho_map, rho_floor=1e-3, stop_grad_warp=True):
    """(B,N,2) points + (B,1,H,W) rho -> warped distances (B,N,N) and an off-diagonal mask.

        D_warp[i, j] = ||s_i - s_j|| * sqrt(D_x_i),    D_x_i = N * rho(s_i) / mean_rho

    ``rho_floor`` is load-bearing. x0_pred can place points in pure-white regions where
    rho == 0; the warp would collapse to 0, every pair would look infinitely close, and the
    hinge would saturate. Flooring rho keeps D_x well-conditioned: in the fully-white limit
    D_x -> N and the warp -> sqrt(N), which is exactly the uniform-grid normalisation.

    ``stop_grad_warp`` detaches sqrt(D_x) so the gradient flows only through cdist (the
    control_gt_free default). The warp depends on the target density, not on the points'
    quality, so letting it move mostly adds variance.
    """
    b, n, _ = points.shape
    D = torch.cdist(points, points)  # (B, N, N) -- gradient flows here

    # x0_pred may leave [0, 1]; grid_sample would zero-pad. Clamp the LOOKUP only, and
    # detach it: the warp factor is a property of the target density.
    lookup = points.detach().clamp(0.0, 1.0)
    local_rho = sample_local_rho(lookup, rho_map).clamp_min(rho_floor)  # (B, N)
    mean_rho = local_rho.mean(dim=1, keepdim=True).clamp_min(rho_floor)
    warp = torch.sqrt((n * local_rho / mean_rho).clamp_min(_EPS))  # (B, N)
    if stop_grad_warp:
        warp = warp.detach()

    D_warp = D * warp.unsqueeze(2)  # warp by the SOURCE point's factor (asymmetric, per M5)
    mask = 1.0 - torch.eye(n, device=points.device, dtype=points.dtype).unsqueeze(0)
    return D_warp, mask


def spacing_hinge(D_warp, mask, r_target):
    """Per-sample (B,) penalty on warped-PCF mass below ``r_target``. Clumping -> large."""
    viol = (r_target - D_warp).clamp_min(0.0) / float(r_target)
    return (viol * viol * mask).sum(dim=(1, 2)) / mask.sum().clamp_min(1.0)


def capacity_cv(points, grid_xy, rho, tau):
    """Per-sample (B,) soft capacity CV: a smooth relaxation of the validator's delta_c.

    Allocative, not spacing-aware -- it will not punish clumping on its own. Optional.
    """
    w, _ = soft_membership(points, grid_xy, tau)  # (B, N, G)
    c = (w * rho.unsqueeze(1)).sum(dim=2)  # (B, N)
    c_mean = c.mean(dim=1, keepdim=True).clamp_min(_EPS)
    return ((c / c_mean - 1.0) ** 2).mean(dim=1)


class GeometryLoss:
    """Auxiliary geometry term on the one-step-decoded points.

    Call with the model's own ``x0_pred`` (see ``FlowMatching.x0_from_velocity``):

        total, comps = geo(x0_pred, t, target_density, global_step)
        loss = denoise_loss + (total if total is not None else 0)

    Returns ``(None, comps)`` -- not a zero tensor -- whenever the term is inactive
    (disabled, inside warmup, or no sample in the batch passed the t-mask), so the caller
    never builds a graph for nothing.
    """

    def __init__(
        self,
        weight=0.05,
        cap_weight=0.0,
        t_mode="hard",
        t_max=0.4,
        t_ramp_k=2.0,
        warmup_steps=2000,
        ramp_steps=1000,
        subsample=0,
        spacing_scale=1.0,
        rho_floor=1e-3,
        warp_grid=0,
        stop_grad_warp=True,
        cap_tau=5e-3,
        cap_grid=32,
    ):
        self.weight = float(weight)
        self.cap_weight = float(cap_weight)
        self.t_mode = str(t_mode)
        self.t_max = float(t_max)
        self.t_ramp_k = float(t_ramp_k)
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.subsample = int(subsample)
        self.spacing_scale = float(spacing_scale)
        self.r_target = self.spacing_scale * R_PACK
        self.rho_floor = float(rho_floor)
        self.warp_grid = int(warp_grid)
        self.stop_grad_warp = bool(stop_grad_warp)
        self.cap_tau = float(cap_tau)
        self.cap_grid = int(cap_grid)

        # Validate the t-mask up front rather than at step 2001.
        t_weight(torch.zeros(1), self.t_mode, self.t_max, self.t_ramp_k)

        if self.subsample > 0 and self.cap_weight > 0.0:
            raise ValueError(
                "subsample>0 is incompatible with the capacity term: soft-Voronoi capacity "
                "is defined over the FULL point set, so scoring a subset measures a "
                "different partition. Use subsample only with the spacing term."
            )

    @property
    def enabled(self):
        return self.weight > 0.0 or self.cap_weight > 0.0

    def scale(self, global_step):
        """0 during warmup, then a linear ramp to 1 over ``ramp_steps``.

        The ramp is not cosmetic: switching a loss term on as a step function makes the
        total jump, which the spike detector would read as a bad batch and skip.
        """
        if not self.enabled or global_step < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return 1.0
        return min(1.0, (global_step - self.warmup_steps) / float(self.ramp_steps))

    def __call__(self, x0_pred, t, target_density, global_step=0):
        comps = {"geo/scale": 0.0, "geo/frac_active": 0.0}

        r = self.scale(global_step)
        if r <= 0.0:
            return None, comps

        w = t_weight(t, self.t_mode, self.t_max, self.t_ramp_k)
        keep = w > 0
        n_keep = int(keep.sum().item())
        comps["geo/frac_active"] = n_keep / max(int(t.shape[0]), 1)
        if n_keep == 0:
            # Possible under a hard mask when every drawn t landed above t_max.
            return None, comps

        w = w[keep]
        # Subsetting the batch (not just zero-weighting it) is what keeps the O(N^2) cdist
        # off the samples the mask discards.
        points = offsets_to_coords(x0_pred[keep])  # (b, N, 2); G inferred -> transfer-safe
        dens = target_density[keep]
        if dens.dim() == 3:
            dens = dens.unsqueeze(1)

        if 0 < self.subsample < points.shape[1]:
            idx = torch.randperm(points.shape[1], device=points.device)[: self.subsample]
            points = points[:, idx]

        d = dens
        if self.warp_grid > 0 and d.shape[-1] != self.warp_grid:
            d = F.interpolate(d, size=(self.warp_grid, self.warp_grid), mode="area")
        rho_map = density_to_rho(d).clamp_min(self.rho_floor)

        wsum = w.sum().clamp_min(_EPS)
        total = points.new_zeros(())

        if self.weight > 0.0:
            D_warp, mask = warped_pair_distances(
                points, rho_map, rho_floor=self.rho_floor, stop_grad_warp=self.stop_grad_warp
            )
            l_sp = (w * spacing_hinge(D_warp, mask, self.r_target)).sum() / wsum
            total = total + self.weight * l_sp
            comps["geo/spacing"] = float(l_sp.detach())

        if self.cap_weight > 0.0:
            rho_flat, grid_xy = density_grid(dens, self.cap_grid, device=points.device)
            l_cap = (w * capacity_cv(points, grid_xy, rho_flat, self.cap_tau)).sum() / wsum
            total = total + self.cap_weight * l_cap
            comps["geo/capacity"] = float(l_cap.detach())

        total = r * total
        comps["geo/scale"] = r
        comps["geo/total"] = float(total.detach())
        return total, comps

    def describe(self):
        if not self.enabled:
            return "GeometryLoss: DISABLED"
        bits = [
            "weight=%.4g" % self.weight,
            "cap_weight=%.4g" % self.cap_weight,
            "t_mode=%s" % self.t_mode,
        ]
        if self.t_mode == "hard":
            bits.append("t_max=%.3g" % self.t_max)
        elif self.t_mode == "ramp":
            bits.append("k=%.3g" % self.t_ramp_k)
        bits += [
            "r_target=%.4g" % self.r_target,
            "warmup=%d" % self.warmup_steps,
            "ramp=%d" % self.ramp_steps,
        ]
        if self.subsample > 0:
            bits.append("subsample=%d" % self.subsample)
        return "GeometryLoss: " + ", ".join(bits)
