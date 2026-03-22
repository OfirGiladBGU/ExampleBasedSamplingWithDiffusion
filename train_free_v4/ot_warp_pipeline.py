"""End-to-end unconditional OT warp pipeline for train_free_v4."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from train_free_v2.utils_guidance import inverse_ot_transform, load_target_image
from train_free_v4.backends import CDFWarpBackend, IdentityWarpBackend


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

    return {
        "offsets": offsets,
        "uniform_points": uniform_points,
        "warped_points": warped_points,
        "density_image": high_res_image,
    }
