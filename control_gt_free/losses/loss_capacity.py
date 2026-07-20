"""Differentiable capacity-constraint surrogate (soft delta_c).

Training-only surrogate for the hard validator
``stippling_metrics_advance.compute_m2_capacity_constraint`` (``voronoi_mass_cv`` = delta_c).
Never reported.

    c_i   = sum_x rho(x) * w_i(x)                 # soft capacity of point i
    L_cap = mean_i ( c_i / mean_i(c_i) - 1 )^2     # squared CV == delta_c

As tau -> 0 the soft w_i(x) approach hard Voronoi membership and c_i approaches the
integral of rho over point i's Voronoi cell -- exactly the validator's capacity. So L_cap
is a smooth relaxation of the reported delta_c.
"""

import torch

from control_gt_free.losses.soft_membership import soft_membership


def soft_capacities(points, grid_xy, rho, tau, w=None, sq_dist=None):
    """Return (c_i, w, sq_dist). ``rho`` is (B, G), ``c_i`` is (B, N)."""
    if w is None:
        w, sq_dist = soft_membership(points, grid_xy, tau, sq_dist=sq_dist)
    # c_i = sum_x rho(x) * w_i(x)   ; rho broadcast over the points axis.
    c = (w * rho.unsqueeze(1)).sum(dim=2)
    return c, w, sq_dist


def loss_capacity(points, grid_xy, rho, tau, w=None, sq_dist=None, eps=1e-8):
    """Soft delta_c. Returns (loss, aux) where aux carries reusable w / sq_dist / c."""
    c, w, sq_dist = soft_capacities(points, grid_xy, rho, tau, w=w, sq_dist=sq_dist)
    c_mean = c.mean(dim=1, keepdim=True).clamp_min(eps)
    l = ((c / c_mean - 1.0) ** 2).mean()
    return l, {"w": w, "sq_dist": sq_dist, "capacities": c}


@torch.no_grad()
def soft_capacity_cv(points, grid_xy, rho, tau):
    """Scalar soft delta_c for logging the soft-vs-hard capacity gap (tau health)."""
    c, _, _ = soft_capacities(points, grid_xy, rho, tau)
    c_mean = c.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return float(((c / c_mean - 1.0) ** 2).mean().item())
