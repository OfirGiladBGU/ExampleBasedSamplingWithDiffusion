"""End-to-end unconditional OT warp pipeline for train_free_v4."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from train_free_v2.utils_guidance import inverse_ot_transform, load_target_image
from train_free_v4.backends import CDFWarpBackend, IdentityWarpBackend


def _load_lloyd_step():
    """Import DifferentiableLloydStep from the projection-conditioned repo."""
    root = Path("/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_pointdit_v6")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sinkhorn_lloyd_losses import DifferentiableLloydStep  # type: ignore
    return DifferentiableLloydStep


def _preprocess_density_for_lloyd(
    density_image: torch.Tensor,
    density_mode: str,
    density_gamma: float,
    grid_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return (B, 1, grid_size, grid_size) density map ready for Lloyd relaxation."""
    density = F.interpolate(density_image, size=(grid_size, grid_size), mode="area")
    if density_mode == "dark":
        density = 1.0 - density
    density = torch.clamp(density, min=0.0)
    if density_gamma != 1.0:
        density = torch.pow(density, density_gamma)
    density = density + eps
    density = density / density.sum(dim=(2, 3), keepdim=True)
    return density


def build_backend(
    backend: str,
    warp_grid_size: int,
    density_mode: str,
    density_gamma: float,
    cdf_eps: float,
    interpolation: bool,
):
    if backend == "none":
        return IdentityWarpBackend()
    if backend == "cdf":
        return CDFWarpBackend(
            warp_grid_size=warp_grid_size,
            density_mode=density_mode,
            density_gamma=density_gamma,
            eps=cdf_eps,
            interpolation=interpolation,
        )
    raise ValueError(f"Unknown backend: {backend}")


@torch.no_grad()
def sample_unconditional_offsets(model, shape: Tuple[int, ...], with_tqdm: bool = True) -> torch.Tensor:
    return model.p_sample_loop(shape, img=None, cond=None, with_tqdm=with_tqdm, with_sampling=True)


@torch.no_grad()
def run_unconditional_ot_warp_pipeline(
    model,
    image_path: str,
    batch_size: int = 1,
    model_grid_size: int = 32,
    warp_grid_size: int = 256,
    backend: str = "cdf",
    density_mode: str = "dark",
    density_gamma: float = 1.0,
    cdf_eps: float = 1e-8,
    interpolation: bool = True,
    lloyd_relax_steps: int = 0,
    lloyd_tau: float = 0.005,
    lloyd_grid_size: int = 64,
    device: str = "cuda",
    with_tqdm: bool = True,
) -> Dict[str, torch.Tensor]:
    high_res_image, _ = load_target_image(image_path, grid_size=model_grid_size, device=device)
    if batch_size > 1:
        high_res_image = high_res_image.repeat(batch_size, 1, 1, 1)

    offsets = sample_unconditional_offsets(
        model,
        shape=(batch_size, 2, model_grid_size, model_grid_size),
        with_tqdm=with_tqdm,
    )
    uniform_points = inverse_ot_transform(offsets, grid_size=model_grid_size)

    warp_backend = build_backend(
        backend=backend,
        warp_grid_size=warp_grid_size,
        density_mode=density_mode,
        density_gamma=density_gamma,
        cdf_eps=cdf_eps,
        interpolation=interpolation,
    )
    warped_points = warp_backend.warp(uniform_points, high_res_image)

    if lloyd_relax_steps > 0:
        DifferentiableLloydStep = _load_lloyd_step()
        lloyd = DifferentiableLloydStep(
            grid_size=lloyd_grid_size, tau=lloyd_tau, num_steps=1
        ).to(device)
        density_for_lloyd = _preprocess_density_for_lloyd(
            high_res_image, density_mode, density_gamma, lloyd_grid_size, cdf_eps
        )
        pts = warped_points * 2.0 - 1.0
        for _ in range(lloyd_relax_steps):
            pts = lloyd(pts, density_for_lloyd)
        warped_points = torch.clamp((pts + 1.0) * 0.5, 0.0, 1.0)

    return {
        "offsets": offsets,
        "uniform_points": uniform_points,
        "warped_points": warped_points,
        "density_image": high_res_image,
    }
