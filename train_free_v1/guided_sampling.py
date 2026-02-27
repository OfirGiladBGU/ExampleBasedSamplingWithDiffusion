"""x0-hat guided diffusion sampling with boundary repulsion.

Runs the frozen unconditional diffusion model independently inside each
quadtree cell, stitching the results together with a differentiable
repulsion energy that prevents point overlap at cell boundaries.

The guidance loop:
  1. Predict noise with the frozen U-Net (chunked for memory safety).
  2. Estimate x0_hat from the noise prediction.
  3. Map x0_hat to global image coordinates (differentiable affine).
  4. Compute boundary repulsion energy across neighbour-cell pairs.
  5. Backprop through autograd to get d(energy)/d(y_t).
  6. Clamp the gradient, then apply standard DDPM step minus guidance.
  7. After the loop, post-hoc cull each cell to its target budget.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from train_free_v1.quadtree import Cell


GRID_SIZE = 8  # the frozen model generates 8x8 offset grids


# ── differentiable local-to-global mapping ───────────────────────────

def _build_grid_centers(grid_size: int, device: torch.device) -> torch.Tensor:
    """Return (1, 2, G, G) tensor of cell-centre coords in [0, 1]."""
    coords = torch.arange(grid_size, dtype=torch.float32, device=device)
    coords = coords / grid_size + 0.5 / grid_size
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, G, G)


def local_to_global(
    x0_hat: torch.Tensor,
    cell_origins: torch.Tensor,
    cell_widths: torch.Tensor,
    grid_centers: torch.Tensor,
) -> torch.Tensor:
    """Map OT offsets to global image coordinates (fully differentiable).

    Parameters
    ----------
    x0_hat : (K, 2, G, G)   predicted clean offsets in [-1, 1]
    cell_origins : (K, 2, 1, 1)  top-left corner of each cell in [0, 1]
    cell_widths  : (K, 1, 1, 1)  side length of each cell in [0, 1]
    grid_centers : (1, 2, G, G)  normalised cell centers in [0, 1]

    Returns
    -------
    (K, 2, G, G) global coordinates in [0, 1]
    """
    local_pts = x0_hat / GRID_SIZE + grid_centers
    return local_pts * cell_widths + cell_origins


def global_to_flat_points(global_coords: torch.Tensor) -> torch.Tensor:
    """Reshape (K, 2, G, G) -> (K, 64, 2) for distance computation."""
    K = global_coords.shape[0]
    return global_coords.reshape(K, 2, -1).permute(0, 2, 1)  # (K, 64, 2)


# ── boundary repulsion energy ────────────────────────────────────────

def boundary_repulsion_energy(
    global_pts_flat: torch.Tensor,
    cells: List[Cell],
    neighbor_pairs: List[Tuple[int, int]],
) -> torch.Tensor:
    """Compute repulsion energy across all neighbour-cell boundaries.

    Uses adaptive radius: r = 0.5 * (spacing_A + spacing_B).
    Energy per pair: sum(max(0, r - dist)^2).

    Parameters
    ----------
    global_pts_flat : (K, 64, 2) global point coordinates
    cells : list[Cell]
    neighbor_pairs : list of (i, j) index pairs

    Returns
    -------
    Scalar energy tensor (differentiable w.r.t. global_pts_flat).
    """
    if not neighbor_pairs:
        return torch.tensor(0.0, device=global_pts_flat.device,
                            requires_grad=True)

    total_energy = torch.tensor(0.0, device=global_pts_flat.device)

    for i, j in neighbor_pairs:
        pts_a = global_pts_flat[i]  # (64, 2)
        pts_b = global_pts_flat[j]  # (64, 2)

        dists = torch.cdist(pts_a.unsqueeze(0),
                            pts_b.unsqueeze(0)).squeeze(0)  # (64, 64)

        r = 0.5 * (cells[i].spacing + cells[j].spacing)
        violations = torch.clamp(r - dists, min=0.0)
        total_energy = total_energy + (violations ** 2).sum()

    return total_energy


# ── lambda schedule ──────────────────────────────────────────────────

def make_lambda_schedule(
    betas: torch.Tensor,
    scale: float = 50.0,
) -> torch.Tensor:
    """Guidance strength proportional to beta_t.

    Naturally stronger in mid-range timesteps and weaker at the end
    where fine spatial details are resolved.
    """
    return scale * betas


# ── the main guided sampling loop ────────────────────────────────────

@torch.no_grad()
def _extract(data: torch.Tensor, t: torch.Tensor,
             shape: torch.Size) -> torch.Tensor:
    """Gather schedule values at timestep t, broadcast to shape."""
    out = torch.gather(data, 0, t)
    return out.view(t.shape[0], *([1] * (len(shape) - 1)))


def guided_sample(
    model: nn.Module,
    cells: List[Cell],
    *,
    betas: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    sqrt_recip_alphas_cumprod: torch.Tensor,
    sqrt_recipm1_alphas_cumprod: torch.Tensor,
    posterior_mean_coef1: torch.Tensor,
    posterior_mean_coef2: torch.Tensor,
    posterior_variance: torch.Tensor,
    posterior_log_variance_clipped: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
    lambda_scale: float = 50.0,
    grad_clip: float = 1.0,
    chunk_size: int = 256,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Run the full guided reverse-diffusion loop.

    Parameters
    ----------
    model : nn.Module
        Frozen unconditional denoiser (in eval mode).
    cells : list[Cell]
        Quadtree leaf cells with budgets and neighbour lists.
    betas ... posterior_log_variance_clipped :
        Diffusion schedule tensors (extracted from DiffusionModel).
    num_timesteps : int
    device : torch.device
    lambda_scale : float
        Multiplier for the guidance strength schedule.
    grad_clip : float
        Hard clamp on the guidance gradient magnitude.
    chunk_size : int
        Max cells per U-Net forward pass (GPU memory safety).
    seed : int, optional

    Returns
    -------
    points : ndarray (N_total, 2)
        Assembled global point set in [0, 1]^2 after post-hoc culling.
    """
    if seed is not None:
        torch.manual_seed(seed)

    K = len(cells)
    if K == 0:
        return np.empty((0, 2), dtype=np.float64)

    # ── precompute cell geometry tensors ──────────────────────────────
    origins = torch.tensor(
        [[c.x, c.y] for c in cells],
        dtype=torch.float32, device=device
    ).view(K, 2, 1, 1)

    widths = torch.tensor(
        [c.width for c in cells],
        dtype=torch.float32, device=device
    ).view(K, 1, 1, 1)

    grid_centers = _build_grid_centers(GRID_SIZE, device)  # (1, 2, 8, 8)

    # ── collect unique neighbour pairs ───────────────────────────────
    pair_set = set()
    for i, cell in enumerate(cells):
        for j in cell.neighbors:
            pair_set.add((min(i, j), max(i, j)))
    neighbor_pairs = list(pair_set)

    # ── lambda schedule ──────────────────────────────────────────────
    lambdas = make_lambda_schedule(betas, lambda_scale)

    # ── model variance schedule (matches Diffusion.py p_mean_variance)
    model_logvar_schedule = torch.log(
        torch.cat([posterior_variance[1:2], betas[1:]])
    )

    # ── logging setup ───────────────────────────────────────────────
    log_every = max(1, (num_timesteps - 1) // 20)  # ~20 log lines
    print(f"  Cells: {K} | Neighbour pairs: {len(neighbor_pairs)} | "
          f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Lambda scale: {lambda_scale} | Grad clip: {grad_clip} | "
          f"Chunk size: {chunk_size}")
    print(f"  Logging every {log_every} steps\n")
    print(f"  {'Step':>6}  {'t':>5}  {'Energy':>12}  "
          f"{'|grad|':>10}  {'lambda_t':>10}")
    print(f"  {'-'*53}")

    # ── initialise from pure noise ───────────────────────────────────
    y_t = torch.randn(K, 2, GRID_SIZE, GRID_SIZE,
                       dtype=torch.float32, device=device)

    from tqdm import tqdm
    total_steps = num_timesteps - 1
    for step_idx, step_i in enumerate(tqdm(
            reversed(range(total_steps)),
            total=total_steps, desc="Guided sampling")):
        t_val = step_i
        t_tensor = torch.full((K,), t_val, dtype=torch.int64, device=device)

        # ── forward pass (chunked, with grad through assembled tensor)
        y_t_grad = y_t.detach().requires_grad_(True)

        noise_preds = []
        for ci in range(0, K, chunk_size):
            ce = min(ci + chunk_size, K)
            chunk_pred = model(y_t_grad[ci:ce], t_tensor[ci:ce])
            noise_preds.append(chunk_pred)
        noise_pred = torch.cat(noise_preds, dim=0)

        # ── estimate x0_hat ──────────────────────────────────────────
        sqrt_recip = _extract(sqrt_recip_alphas_cumprod, t_tensor, y_t.shape)
        sqrt_rm1 = _extract(sqrt_recipm1_alphas_cumprod, t_tensor, y_t.shape)
        x0_hat = sqrt_recip * y_t_grad - sqrt_rm1 * noise_pred

        # ── map to global coordinates (differentiable) ───────────────
        global_coords = local_to_global(x0_hat, origins, widths, grid_centers)
        global_flat = global_to_flat_points(global_coords)  # (K, 64, 2)

        # ── boundary repulsion energy ────────────────────────────────
        energy = boundary_repulsion_energy(global_flat, cells, neighbor_pairs)

        # ── guidance gradient ────────────────────────────────────────
        energy_val = energy.item()
        if energy.requires_grad and energy_val > 0:
            grad = torch.autograd.grad(energy, y_t_grad)[0]
            grad = torch.clamp(grad, -grad_clip, grad_clip)
        else:
            grad = torch.zeros_like(y_t_grad)

        grad_norm = grad.norm().item()

        # ── standard DDPM posterior step ─────────────────────────────
        with torch.no_grad():
            coef1 = _extract(posterior_mean_coef1, t_tensor, y_t.shape)
            coef2 = _extract(posterior_mean_coef2, t_tensor, y_t.shape)
            posterior_mean = coef1 * x0_hat.detach() + coef2 * y_t_grad.detach()

            logvar = _extract(model_logvar_schedule, t_tensor, y_t.shape)

            noise = torch.randn_like(y_t)
            t_broadcast = t_tensor.view(K, 1, 1, 1)
            noise = torch.where(t_broadcast == 0, torch.zeros_like(noise), noise)

            y_t_prev = posterior_mean + torch.exp(0.5 * logvar) * noise

            # ── apply guidance ───────────────────────────────────────
            lam = _extract(lambdas, t_tensor, y_t.shape)
            lam_scalar = lam.flatten()[0].item()
            y_t = y_t_prev - lam * grad

        # ── periodic log ─────────────────────────────────────────────
        if step_idx % log_every == 0 or step_i == 0:
            tqdm.write(
                f"  {step_idx:6d}  {t_val:5d}  {energy_val:12.4f}  "
                f"{grad_norm:10.4f}  {lam_scalar:10.6f}"
            )

    # ── convert final offsets to global points ───────────────────────
    with torch.no_grad():
        final_global = local_to_global(y_t, origins, widths, grid_centers)
        final_flat = global_to_flat_points(final_global)  # (K, 64, 2)
        final_np = final_flat.cpu().numpy()

    # ── post-hoc culling ─────────────────────────────────────────────
    rng = np.random.RandomState(seed if seed is not None else 42)
    all_points = []
    for k, cell in enumerate(cells):
        pts = final_np[k]  # (64, 2)
        if cell.budget < 64:
            idx = rng.permutation(64)[:cell.budget]
            pts = pts[idx]
        all_points.append(pts)

    return np.concatenate(all_points, axis=0)
