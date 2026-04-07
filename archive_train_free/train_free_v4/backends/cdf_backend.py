"""Separable 2D CDF inverse-warp backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CDFWarpBackend:
    name = "cdf"

    def __init__(
        self,
        warp_grid_size: int = 256,
        density_mode: str = "dark",
        density_gamma: float = 1.0,
        eps: float = 1e-8,
        contrast_stretch: bool = True,
        interpolation: bool = True,
    ):
        if density_mode not in {"dark", "light"}:
            raise ValueError("density_mode must be 'dark' or 'light'")
        self.warp_grid_size = warp_grid_size
        self.density_mode = density_mode
        self.density_gamma = density_gamma
        self.eps = eps
        self.contrast_stretch = contrast_stretch
        self.interpolation = interpolation

    def preprocess_density(self, density_image: torch.Tensor) -> torch.Tensor:
        density = F.interpolate(
            density_image,
            size=(self.warp_grid_size, self.warp_grid_size),
            mode="area",
        )

        if self.contrast_stretch:
            mins = density.amin(dim=(2, 3), keepdim=True)
            maxs = density.amax(dim=(2, 3), keepdim=True)
            density = (density - mins) / (maxs - mins + self.eps)

        if self.density_mode == "dark":
            density = 1.0 - density

        density = torch.clamp(density, min=0.0)
        if self.density_gamma != 1.0:
            density = torch.pow(density, self.density_gamma)

        density = density + self.eps
        density = density / density.sum(dim=(2, 3), keepdim=True)
        return density

    def warp(self, uniform_points: torch.Tensor, density_image: torch.Tensor) -> torch.Tensor:
        pdf = self.preprocess_density(density_image).squeeze(1)
        return apply_cdf_inverse_warp(
            uniform_points,
            pdf,
            eps=self.eps,
            interpolation=self.interpolation,
        )


def apply_cdf_inverse_warp(
    uniform_points: torch.Tensor,
    pdf: torch.Tensor,
    eps: float = 1e-8,
    interpolation: bool = True,
) -> torch.Tensor:
    """Warp uniform points into a target density using separable 2D CDF inversion.

    Args:
        uniform_points: (B, N, 2) in [0, 1].
        pdf: (B, H, W) normalized density with mass summing to 1.
    """
    if uniform_points.ndim != 3 or uniform_points.shape[-1] != 2:
        raise ValueError("uniform_points must have shape (B, N, 2)")
    if pdf.ndim != 3:
        raise ValueError("pdf must have shape (B, H, W)")

    B, N, _ = uniform_points.shape
    _, H, W = pdf.shape
    device = uniform_points.device
    pdf = pdf.to(device)

    pdf = pdf + eps
    pdf = pdf / pdf.sum(dim=(1, 2), keepdim=True)

    pdf_y = pdf.sum(dim=2)
    pdf_x_given_y = pdf / pdf_y.unsqueeze(2).clamp_min(eps)

    cdf_y = torch.cumsum(pdf_y, dim=1)
    cdf_y = cdf_y / cdf_y[:, -1:].clamp_min(eps)

    cdf_x_given_y = torch.cumsum(pdf_x_given_y, dim=2)
    cdf_x_given_y = cdf_x_given_y / cdf_x_given_y[:, :, -1:].clamp_min(eps)

    zero_pad_y = torch.zeros((B, 1), device=device, dtype=pdf.dtype)
    cdf_y = torch.cat([zero_pad_y, cdf_y], dim=1)

    zero_pad_x = torch.zeros((B, H, 1), device=device, dtype=pdf.dtype)
    cdf_x_given_y = torch.cat([zero_pad_x, cdf_x_given_y], dim=2)

    warped_points = torch.zeros_like(uniform_points)

    for b in range(B):
        u = uniform_points[b, :, 0].contiguous()
        v = uniform_points[b, :, 1].contiguous()

        y_idx = torch.searchsorted(cdf_y[b].contiguous(), v, right=False)
        y_idx = torch.clamp(y_idx, 1, H)

        cdf_y_lower = cdf_y[b, y_idx - 1]
        if interpolation:
            cdf_y_upper = cdf_y[b, y_idx]
            y_fraction = (v - cdf_y_lower) / (cdf_y_upper - cdf_y_lower + eps)
        else:
            y_fraction = torch.zeros_like(v)
        y_continuous = (y_idx - 1 + y_fraction) / H
        warped_points[b, :, 1] = y_continuous

        row_indices = (y_idx - 1).long()
        selected_cdfs = cdf_x_given_y[b, row_indices, :]

        x_continuous = torch.empty_like(u)
        for n_idx in range(N):
            row_cdf = selected_cdfs[n_idx].contiguous()
            x_bin = torch.searchsorted(row_cdf, u[n_idx:n_idx + 1], right=False).squeeze(0)
            x_bin = torch.clamp(x_bin, 1, W)
            x_lower = row_cdf[x_bin - 1]
            if interpolation:
                x_upper = row_cdf[x_bin]
                x_fraction = (u[n_idx] - x_lower) / (x_upper - x_lower + eps)
            else:
                x_fraction = torch.zeros((), device=device, dtype=u.dtype)
            x_continuous[n_idx] = (x_bin - 1 + x_fraction) / W

        warped_points[b, :, 0] = x_continuous

    return torch.clamp(warped_points, 0.0, 1.0)
