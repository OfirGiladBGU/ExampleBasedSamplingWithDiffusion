"""Soft-Voronoi membership and shared point/grid geometry for the GT-free objective.

Everything here is a *differentiable training surrogate*. It is NEVER reported as a
result -- the reported numbers come from the scipy/FFT validators in
``utils/stippling_metrics_advance.py`` (see README_objective_driven_stippling.md,
guardrail #1 and #2).

Convention (matches the metric validators and ``train_control.offsets_to_coords_gpu``):
    * points / coords are (B, N, 2) in [0, 1], column 0 = x (image col), column 1 = y (row).
    * an OT offset grid is (B, 2, G, G) in ~[-1, 1]; coord = cell_center + offset / G.
    * the target density fed to the network is the raw image (white=1, dark=0). The metrics
      weight by rho = 1 - image (dark = HIGH mass). Callers must pass that inverted rho here
      -- use :func:`density_to_rho`.
"""

import torch
import torch.nn.functional as F


# ── point / grid geometry (shared, differentiable) ──────────────────────────

def grid_centers_flat(grid_size, device, dtype=torch.float32):
    """Cell-center coordinates of a ``grid_size`` x ``grid_size`` grid, shape (1, G*G, 2).

    Mirrors ``to_pointset_optimal_transport`` / ``train_control._grid_centers_flat``:
    linspace(0,1, endpoint=False) + 1/(2G), meshgrid with ``indexing='xy'`` so column 0
    is x and column 1 is y. Row-major flatten matches ``offsets.reshape(B, 2, G*G)``.
    """
    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    return torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)


def offsets_to_coords(offsets, grid_size=None, centers=None, clamp=False):
    """OT offsets (B, 2, G, G) -> coords (B, N, 2) in [0, 1]. Differentiable.

    ``clamp`` is off by default: clamping to [0,1] zeroes the gradient of any point that
    lands on the boundary, which is unnecessary here (coords = center +/- 1/G stay well
    inside the domain) and hurts optimization. The hard validator clamps on its own.
    """
    b, _, g, _ = offsets.shape
    if grid_size is None:
        grid_size = g
    if centers is None:
        centers = grid_centers_flat(grid_size, offsets.device, offsets.dtype)
    offs = offsets.permute(0, 2, 3, 1).reshape(b, grid_size * grid_size, 2)
    coords = centers.to(offsets.dtype).expand(b, -1, -1) + offs / float(grid_size)
    return coords.clamp(0.0, 1.0) if clamp else coords


def density_to_rho(target_density):
    """Convert the raw-image density channel (white=1) into rho (dark=HIGH mass).

    Matches ``stippling_metrics_advance._to_density`` exactly so the soft losses optimize
    the same quantity the hard validators score.
    """
    return (1.0 - target_density).clamp(0.0, 1.0)


def density_grid(target_density, loss_grid, device=None, dtype=torch.float32):
    """Downsample the density channel to the (cheap) loss grid and flatten.

    Returns
    -------
    rho   : (B, G_l*G_l)   rho(x) = 1 - image, area-pooled to the loss grid.
    grid_xy : (1, G_l*G_l, 2)  cell-center coords of the loss grid (for cdist to points).
    """
    if target_density.dim() == 3:
        target_density = target_density.unsqueeze(0)
    d = target_density
    if d.shape[-1] != loss_grid or d.shape[-2] != loss_grid:
        d = F.interpolate(d, size=(loss_grid, loss_grid), mode="area")
    rho = density_to_rho(d)  # (B, 1, G_l, G_l)
    rho = rho.reshape(rho.shape[0], -1)
    dev = device or target_density.device
    gxy = grid_centers_flat(loss_grid, dev, dtype)
    return rho.to(dev), gxy


def sample_local_rho(points, rho_map):
    """Bilinearly sample rho at each point. Returns (B, N). Used for the PCF warp factor.

    ``rho_map`` is (B, 1, H, W) with rho = 1 - image (dark = high mass). grid_sample wants
    coords in [-1, 1]. This is detached by the caller when the warp is stop-grad (default).
    """
    if rho_map.dim() == 3:
        rho_map = rho_map.unsqueeze(0)
    b, n, _ = points.shape
    if rho_map.shape[0] != b:
        rho_map = rho_map.expand(b, -1, -1, -1)
    grid = (points * 2.0 - 1.0).view(b, n, 1, 2)  # (B, N, 1, 2), last dim = (x, y)
    sampled = F.grid_sample(rho_map, grid, mode="bilinear", align_corners=False)
    return sampled.view(b, n)


# ── soft-Voronoi membership ─────────────────────────────────────────────────

def pairwise_sq_dist(points, grid_xy):
    """Squared Euclidean distances (B, N, G) between points and grid locations."""
    if grid_xy.dim() == 2:
        grid_xy = grid_xy.unsqueeze(0)
    if grid_xy.shape[0] != points.shape[0]:
        grid_xy = grid_xy.expand(points.shape[0], -1, -1)
    d = torch.cdist(points, grid_xy.to(points.dtype))
    return d * d


def soft_membership(points, grid_xy, tau, sq_dist=None):
    """Soft-Voronoi weights ``w_i(x) = softmax_i(-||x - s_i||^2 / tau)``.

    Softmax is over the *points* axis (dim=1) so ``sum_i w_i(x) = 1`` for every grid
    location x -- the cells form a soft partition, matching the hard Voronoi partition the
    validators use. Returns (w, sq_dist) so capacity/CVT can share both.

    As ``tau -> 0`` the weights approach a hard nearest-point assignment (the scipy
    Voronoi cells); as ``tau -> inf`` they wash out to uniform. See the tau-schedule note
    in the README: log the soft-vs-hard capacity gap to keep tau honest.
    """
    if sq_dist is None:
        sq_dist = pairwise_sq_dist(points, grid_xy)
    logits = -sq_dist / float(tau)
    w = torch.softmax(logits, dim=1)
    return w, sq_dist
