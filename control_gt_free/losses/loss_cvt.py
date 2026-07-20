"""Differentiable CVT-energy surrogate (soft weighted-variance).

Training-only surrogate for ``stippling_metrics_advance.compute_m1_cvt_energy``
(``cvt_energy``). Never reported.

    L_cvt = sum_i sum_x rho(x) * w_i(x) * ||x - s_i||^2

This is the soft-Voronoi CVT energy: with hard membership (tau -> 0) it is the standard
weighted centroidal-Voronoi energy the validator estimates. Reuses the membership ``w`` and
squared distances already computed for the capacity loss.

``normalize`` divides by the total mass ``sum_x rho(x)`` per sample so the scale is a
mean squared spacing (comparable across images / point counts) rather than an extensive
sum -- keeps its gradient magnitude in a sane range next to L_cap and L_pcf.
"""

from control_gt_free.losses.soft_membership import soft_membership


def loss_cvt(points, grid_xy, rho, tau, w=None, sq_dist=None, normalize=True, eps=1e-8):
    if w is None:
        w, sq_dist = soft_membership(points, grid_xy, tau, sq_dist=sq_dist)
    # per grid location x: sum_i w_i(x) * ||x - s_i||^2  -> (B, G)
    weighted = (w * sq_dist).sum(dim=1)
    energy = (rho * weighted).sum(dim=1)  # (B,)
    if normalize:
        energy = energy / rho.sum(dim=1).clamp_min(eps)
    return energy.mean(), {"w": w, "sq_dist": sq_dist}
