"""In-loop CCVT-guided reverse diffusion for train_free_v3."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_free_v3.ccvt_guidance import CCVTConfig, DifferentiableCCVTGuidance


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    bs = t.shape[0]
    t = t.to(a.device)
    out = a.gather(0, t)
    return out.reshape(bs, *((1,) * (len(x_shape) - 1)))


def offsets_to_points01(offsets: torch.Tensor) -> torch.Tensor:
    B, C, H, W = offsets.shape
    if C != 2:
        raise ValueError(f"Expected 2 channels, got {C}")
    n = H
    grid_1d = (torch.arange(n, device=offsets.device, dtype=offsets.dtype) + 0.5) / n
    gy, gx = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    grid_centers = torch.stack([gx, gy], dim=-1)

    off_hw2 = offsets.permute(0, 2, 3, 1)
    pts_hw2 = (off_hw2 / n) + grid_centers.unsqueeze(0)
    return pts_hw2.reshape(B, -1, 2)


def points_to_offset_guidance(g_points: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Map point-space guidance (B,N,2) back to offset tensor space (B,2,H,W)."""
    B, N, D = g_points.shape
    if D != 2 or N != h * w:
        raise ValueError("Invalid point guidance shape for requested grid")
    g_hw2 = g_points.reshape(B, h, w, 2)
    g_offsets = g_hw2.permute(0, 3, 1, 2) * h
    return g_offsets


def sample_with_ccvt_guidance(
    model,
    target_density: torch.Tensor,
    shape: Tuple[int, ...] = (1, 2, 32, 32),
    timesteps: int = 1000,
    lambda_0: float = 1.0,
    grad_clip: float = 0.0,
    device: str = "cuda",
    cond=None,
    with_tqdm: bool = True,
    debug_guidance: bool = False,
    debug_every: int = 100,
    ccvt_mode: str = "ccvt",
    ccvt_grid_size: int = 64,
    tau: float = 0.01,
    num_steps: int = 1,
    repulsion_strength: float = 0.2,
    repulsion_radius: float = 0.03,
    resample_jumps: int = 0,
    jump_length: int = 10,
) -> torch.Tensor:
    beta_start, beta_end = 1e-4, 2e-2
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    cfg = CCVTConfig(
        grid_size=ccvt_grid_size,
        tau=tau,
        num_steps=num_steps,
        mode=ccvt_mode,
        repulsion_strength=repulsion_strength,
        repulsion_radius=repulsion_radius,
    )
    ccvt = DifferentiableCCVTGuidance(cfg)

    x_t = torch.randn(shape, device=device)
    jump_counter = 0

    def take_guided_step(x_curr: torch.Tensor, i_step: int, log_debug: bool = False) -> torch.Tensor:
        t = torch.tensor([i_step] * shape[0], device=device, dtype=torch.long)

        with torch.no_grad():
            noise_pred = model.model(x_curr, t, cond=cond)
            x0_hat_untracked = (
                extract(sqrt_recip_alphas_cumprod, t, x_curr.shape) * x_curr
                - extract(sqrt_recipm1_alphas_cumprod, t, x_curr.shape) * noise_pred
            )

        # Critical VRAM rule: detach from U-Net graph before CCVT geometry ops.
        x0_hat = x0_hat_untracked.detach().requires_grad_(True)

        points = offsets_to_points01(x0_hat)
        _, bary = ccvt.refine(points, target_density)
        g_points = points - bary

        # IMPORTANT: Do NOT normalize OT guidance here.
        # Spring physics requires magnitude proportional to distance (x - b).
        guidance_points = g_points

        # Optional safety clip (disabled by default) for extreme instability only.
        if grad_clip and grad_clip > 0:
            guidance_points = torch.clamp(guidance_points, -grad_clip, grad_clip)

        _, _, H, W = x_curr.shape
        g_offsets = points_to_offset_guidance(guidance_points, H, W)

        with torch.no_grad():
            posterior_mean = (
                extract(posterior_mean_coef1, t, x_curr.shape) * x0_hat_untracked
                + extract(posterior_mean_coef2, t, x_curr.shape) * x_curr
            )
            # Guidance decays naturally with beta_t.
            lambda_t = lambda_0 * extract(betas, t, x_curr.shape)
            guided_mean = posterior_mean - lambda_t * g_offsets

            if log_debug and debug_guidance and (i_step % max(1, debug_every) == 0):
                nudge_mag = (lambda_t * g_offsets).abs().mean().item()
                step_mag = (posterior_mean - x_curr).abs().mean().item()
                print(
                    f"Step {i_step:4d} | CCVT Nudge: {nudge_mag:.6f} | "
                    f"U-Net Step: {step_mag:.6f} | "
                    f"points range: [{points.min().item():.3f}, {points.max().item():.3f}]"
                )

            if i_step > 0:
                sigma = torch.sqrt(extract(posterior_variance, t, x_curr.shape))
                x_prev = guided_mean + sigma * torch.randn_like(x_curr)
            else:
                x_prev = guided_mean

        return x_prev

    iterator = tqdm(reversed(range(timesteps)), total=timesteps, desc="CCVT Guided Sampling") if with_tqdm else reversed(range(timesteps))

    for i in iterator:
        x_t = take_guided_step(x_t, i, log_debug=True)

        # Optional RePaint-style micro loops at fixed intervals.
        if resample_jumps > 0 and i > 0 and (i % max(1, jump_length) == 0):
            beta_i = betas[i]
            for _ in range(resample_jumps):
                # Re-noise x_{i-1} back toward x_i and denoise again with CCVT guidance.
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(1.0 - beta_i) * x_t + torch.sqrt(beta_i) * noise
                x_t = take_guided_step(x_t, i, log_debug=False)
                jump_counter += 1

    if debug_guidance and resample_jumps > 0:
        print(
            f"Resample jumps summary | jump_length={jump_length} | "
            f"resample_jumps={resample_jumps} | total_extra_guided_steps={jump_counter}"
        )

    return x_t
