"""
Guided Diffusion Sampling via Sinkhorn Posterior Guidance (DPS)

This module implements Diffusion Posterior Sampling (DPS) where a frozen U-Net
is guided toward target density distributions using the gradient of a Sinkhorn
optimal transport loss.

Key insight: Compute gradient only w.r.t. predicted clean state (x0_hat),
not through the U-Net, to maintain constant VRAM footprint.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Tuple


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from a 1D schedule tensor at indices specified by t.
    
    Args:
        a (torch.Tensor): 1D tensor of schedule values (e.g., betas, alphas).
        t (torch.Tensor): 1D tensor of integer indices (batch_size,).
        x_shape (Tuple): Shape of the target tensor (for proper broadcasting).
    
    Returns:
        torch.Tensor: Extracted and reshaped values for broadcasting.
    """
    batch_size = t.shape[0]
    t = t.to(a.device)
    out = a.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def offsets_to_points01(offsets: torch.Tensor, clamp: bool = False) -> torch.Tensor:
    """
    Convert model offsets (B, 2, H, W) to absolute point coordinates in [0, 1].

    The training transform uses: offset = (point - grid_center) * n,
    so inverse is: point = offset / n + grid_center.
    """
    B, C, H, W = offsets.shape
    if C != 2:
        raise ValueError(f"Expected 2 channels for offsets, got {C}")

    n = H
    grid_1d = (torch.arange(n, device=offsets.device, dtype=offsets.dtype) + 0.5) / n
    grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    grid_centers = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

    # (B, 2, H, W) -> (B, H, W, 2)
    off_hw2 = offsets.permute(0, 2, 3, 1)
    pts_hw2 = (off_hw2 / n) + grid_centers.unsqueeze(0)
    pts = pts_hw2.reshape(B, -1, 2)

    # For guidance, avoid hard clamp to keep non-zero gradients when points drift off-domain.
    if clamp:
        return torch.clamp(pts, 0.0, 1.0)
    return pts


def sample_with_sinkhorn_guidance(
    model,
    target_image: torch.Tensor,
    shape: Tuple[int, ...] = (1, 2, 32, 32),
    timesteps: int = 1000,
    lambda_0: float = 1.0,
    grad_clip: float = 1.0,
    device: str = 'cuda',
    cond=None,
    sinkhorn_loss_fn=None,
    with_tqdm: bool = True,
    debug_guidance: bool = False,
    debug_every: int = 100
) -> torch.Tensor:
    """
    Approximated Diffusion Posterior Sampling (DPS) for point clouds.
    
    Guides the frozen U-Net using the gradient of a Sinkhorn density loss.
    The gradient is computed only w.r.t. predicted clean state (x0_hat), not
    through the U-Net, to maintain constant VRAM footprint.
    
    Args:
        model: Frozen diffusion model (already loaded with checkpoint).
        target_image (torch.Tensor): Target density image (B, 1, H, W) in [0, 1].
        shape (Tuple): Shape of noise to generate, e.g., (B, C, H, W).
        timesteps (int): Number of reverse diffusion steps (default: 1000).
        lambda_0 (float): Initial guidance strength (default: 1.0).
        grad_clip (float): Gradient clipping magnitude (default: 1.0).
        device (str): Device to run on ('cuda' or 'cpu').
        cond (torch.Tensor, optional): Conditioning signal (for class-conditional models).
        sinkhorn_loss_fn: Sinkhorn loss function object. If None, creates one internally.
        with_tqdm (bool): Show progress bar (default: True).
    
    Returns:
        torch.Tensor: Generated point offsets (B, C, H, W) in [-1, 1] range.
                     Expected to be reshaped to (B, N, 2) or passed through inverse OT.
    """
    
    # ===== SETUP: DDPM SCHEDULE =====
    # Linear beta schedule (same as training)
    beta_start, beta_end = 0.0001, 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    # Pre-calculate DDPM coefficients for efficiency
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    
    # Posterior variance (for adding noise in reverse process)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Posterior mean coefficients
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    
    posterior_log_variance_clipped = torch.log(
        torch.clamp(posterior_variance, min=1e-20)
    )
    
    # ===== SETUP: SINKHORN LOSS =====
    if sinkhorn_loss_fn is None:
        from .sinkhorn_loss import SinkhornDensityLoss
        sinkhorn_loss_fn = SinkhornDensityLoss(blur=0.05, grid_size=32).to(device)
    
    # ===== INITIALIZE: PURE NOISE =====
    x_t = torch.randn(shape, device=device)
    
    # ===== REVERSE DIFFUSION LOOP WITH GUIDANCE =====
    iterator = tqdm(
        reversed(range(timesteps)),
        desc="Sinkhorn Guided Sampling",
        total=timesteps
    ) if with_tqdm else reversed(range(timesteps))
    
    for i in iterator:
        t = torch.tensor([i] * shape[0], device=device, dtype=torch.long)
        
        # ---------- A. FROZEN U-NET PREDICTION ----------
        with torch.no_grad():
            # Predict noise from frozen U-Net
            noise_pred = model.model(x_t, t, cond=cond)
            
            # Reconstruct clean state x0_hat using DDPM formula
            x0_hat_untracked = (
                extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise_pred
            )
        
        # ---------- B. VRAM FIX: DETACH AND TRACK ONLY x0_hat ----------
        # Sever computational graph from U-Net
        # Create new leaf node with gradient tracking ONLY for Sinkhorn loss
        x0_hat = x0_hat_untracked.detach().requires_grad_(True)
        
        # ---------- C. SINKHORN MATH & GRADIENT ----------
        with torch.enable_grad():
            # Convert predicted offsets to absolute point coordinates in [0, 1]
            # before comparing to image density with OT.
            x0_hat_points = offsets_to_points01(x0_hat, clamp=False)
            
            # Calculate macro-routing guidance loss
            loss = sinkhorn_loss_fn(x0_hat_points, target_image)
            
            # Backprop to get gradient ONLY w.r.t. x0_hat (not through U-Net!)
            # create_graph=False ensures we don't build a second-order graph
            grad_x0 = torch.autograd.grad(loss, x0_hat, create_graph=False)[0]
        
        # ---------- D. NORMALIZATION & CLIPPING ----------
        # Normalize gradient for consistent behavior across OOD images
        # Prevents runaway guidance on high-density regions
        grad_norm = torch.norm(grad_x0, p=2, dim=list(range(1, len(grad_x0.shape))), keepdim=True)
        grad_normalized = grad_x0 / (grad_norm + 1e-8)
        
        # Optional: Clip normalized gradient for extreme safety
        grad_clipped = torch.clamp(grad_normalized, -grad_clip, grad_clip)
        
        # ---------- E. POSTERIOR STEP WITH GUIDANCE ----------
        with torch.no_grad():
            # Compute standard DDPM posterior mean
            posterior_mean = (
                extract(posterior_mean_coef1, t, x_t.shape) * x0_hat_untracked +
                extract(posterior_mean_coef2, t, x_t.shape) * x_t
            )
            
            # Dynamic lambda schedule: λ_t = λ₀ × β_t
            # Strong guidance at t=T (all noise, macro-routing needed)
            # Weak guidance at t=0 (nearly clean, let U-Net finalize blue-noise)
            current_beta = extract(betas, t, x_t.shape)
            lambda_t = lambda_0 * current_beta
            
            # Inject guidance: shift posterior mean away from gradient direction
            guided_mean = posterior_mean - (lambda_t * grad_clipped)

            if debug_guidance and (i % max(1, debug_every) == 0):
                nudge_magnitude = (lambda_t * grad_clipped).abs().mean().item()
                unet_step_magnitude = (posterior_mean - x_t).abs().mean().item()
                grad_mag = grad_x0.abs().mean().item()
                loss_val = loss.item()
                x0_min = x0_hat_untracked.min().item()
                x0_max = x0_hat_untracked.max().item()
                p_min = x0_hat_points.min().item()
                p_max = x0_hat_points.max().item()
                print(
                    f"Step {i:4d} | Sinkhorn Nudge: {nudge_magnitude:.6f} | "
                    f"U-Net Step: {unet_step_magnitude:.6f} | "
                    f"Sinkhorn Loss: {loss_val:.6f} | Grad|x0|: {grad_mag:.6e} | "
                    f"x0_hat range: [{x0_min:.3f}, {x0_max:.3f}] | "
                    f"points range: [{p_min:.3f}, {p_max:.3f}]"
                )
            
            # Add posterior variance noise (Langevin dynamics)
            # Except at final step (t=0) where we don't add noise
            if i > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(extract(posterior_variance, t, x_t.shape))
                x_t = guided_mean + sigma_t * noise
            else:
                x_t = guided_mean
    
    return x_t


def sample_with_sinkhorn_guidance_ddim(
    model,
    target_image: torch.Tensor,
    shape: Tuple[int, ...] = (1, 2, 32, 32),
    timesteps: int = 100,
    lambda_0: float = 1.0,
    grad_clip: float = 1.0,
    device: str = 'cuda',
    cond=None,
    sinkhorn_loss_fn=None,
    with_tqdm: bool = True
) -> torch.Tensor:
    """
    DDIM variant of guided sampling (for faster inference).
    
    ⚠️ DDIM is mathematically different from DDPM - it gives deterministic
    sampling with fewer steps but potentially different convergence properties.
    
    This is a placeholder for future work. For now, use full DDPM (1000 steps)
    and verify results, then optimize to DDIM.
    
    Args:
        Same as sample_with_sinkhorn_guidance().
    
    Returns:
        torch.Tensor: Generated point offsets.
    
    Note:
        Current implementation raises NotImplementedError.
        To implement: use accumulated variance schedule (not learned).
    """
    raise NotImplementedError(
        "DDIM variant not yet implemented. Use full DDPM (1000 steps) for now. "
        "DDIM migration: later refactor after DDPM verification."
    )
