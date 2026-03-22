"""Identity warp backend for unconditional baseline reproduction."""

import torch


class IdentityWarpBackend:
    name = "none"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def warp(self, uniform_points: torch.Tensor, density_image: torch.Tensor) -> torch.Tensor:
        return uniform_points.clone()
