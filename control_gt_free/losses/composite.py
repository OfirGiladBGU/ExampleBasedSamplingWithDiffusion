"""Composite objective: L_total = w_cap*L_cap + w_cvt*L_cvt + w_pcf*L_pcf.

Returns the total and a dict of the (detached) components for logging. The soft-Voronoi
membership ``w`` and squared distances are computed ONCE (README) and shared by the
capacity and CVT terms.

None of these numbers are ever reported as results -- they are the training signal only.
"""

import torch

from control_gt_free.losses.soft_membership import (
    soft_membership, density_grid, offsets_to_coords, density_to_rho,
)
from control_gt_free.losses.loss_capacity import loss_capacity
from control_gt_free.losses.loss_cvt import loss_cvt
from control_gt_free.losses.loss_pcf import loss_pcf, default_edges


class CompositeObjective:
    """Callable objective bound to a fixed config (weights, tau source, edges, grids).

    Usage
    -----
        obj = CompositeObjective(loss_grid=32, w_cap=1, w_cvt=1, w_pcf=1,
                                 pcf_target=target, edges=edges)
        total, comps = obj(pred_offsets, target_density, tau)

    ``pred_offsets`` is the model's own output (B, 2, G, G); ``target_density`` is the raw
    density channel (white=1). Everything downstream converts to points and rho internally.
    """

    def __init__(self, loss_grid=32, w_cap=1.0, w_cvt=1.0, w_pcf=1.0,
                 pcf_target=None, edges=None, pcf_sigma=None, pcf_p=2,
                 stop_grad_warp=True, cvt_normalize=True, pcf_warp_grid=None):
        self.loss_grid = int(loss_grid)
        self.w_cap = float(w_cap)
        self.w_cvt = float(w_cvt)
        self.w_pcf = float(w_pcf)
        self.pcf_target = pcf_target
        self.edges = edges if edges is not None else default_edges()
        self.pcf_sigma = pcf_sigma
        self.pcf_p = int(pcf_p)
        self.stop_grad_warp = bool(stop_grad_warp)
        self.cvt_normalize = bool(cvt_normalize)
        # rho map resolution used for the PCF density warp (full density channel by default).
        self.pcf_warp_grid = pcf_warp_grid

    def __call__(self, pred_offsets, target_density, tau, points=None):
        if points is None:
            points = offsets_to_coords(pred_offsets)  # (B, N, 2), differentiable

        rho, grid_xy = density_grid(target_density, self.loss_grid, device=points.device)

        # Shared soft-Voronoi membership (computed once).
        w = sq = None
        comps = {}
        total = points.new_zeros(())

        if self.w_cap > 0.0:
            l_cap, aux = loss_capacity(points, grid_xy, rho, tau)
            w, sq = aux["w"], aux["sq_dist"]
            total = total + self.w_cap * l_cap
            comps["cap"] = float(l_cap.detach())
        if self.w_cvt > 0.0:
            l_cvt, aux = loss_cvt(points, grid_xy, rho, tau, w=w, sq_dist=sq,
                                  normalize=self.cvt_normalize)
            total = total + self.w_cvt * l_cvt
            comps["cvt"] = float(l_cvt.detach())
        if self.w_pcf > 0.0 and self.pcf_target is not None:
            rho_map = self._rho_map_for_warp(target_density)
            l_pcf, aux = loss_pcf(points, rho_map, self.pcf_target, self.edges,
                                  sigma=self.pcf_sigma, stop_grad_warp=self.stop_grad_warp,
                                  p=self.pcf_p)
            total = total + self.w_pcf * l_pcf
            comps["pcf"] = float(l_pcf.detach())

        comps["total"] = float(total.detach())
        return total, comps

    def _rho_map_for_warp(self, target_density):
        d = target_density
        if d.dim() == 3:
            d = d.unsqueeze(0)
        if self.pcf_warp_grid is not None and d.shape[-1] != self.pcf_warp_grid:
            d = torch.nn.functional.interpolate(
                d, size=(self.pcf_warp_grid, self.pcf_warp_grid), mode="area")
        return density_to_rho(d)
