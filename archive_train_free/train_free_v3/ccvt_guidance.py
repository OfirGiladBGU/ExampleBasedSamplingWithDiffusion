"""CCVT guidance wrapper using existing DifferentiableLloydStep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import sys

import torch
import torch.nn.functional as F


def _import_differentiable_lloyd_step():
    """Import DifferentiableLloydStep from projection-conditioned repo.

    Keeps this module lightweight while reusing the tested implementation.
    """
    root = Path("/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/experiments_pointdit_v6")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sinkhorn_lloyd_losses import DifferentiableLloydStep  # type: ignore

    return DifferentiableLloydStep

Mode = Literal["lloyd", "ccvt", "repulsion"]


@dataclass
class CCVTConfig:
    grid_size: int = 64
    tau: float = 0.01
    num_steps: int = 1
    mode: Mode = "ccvt"
    repulsion_strength: float = 0.2
    repulsion_radius: float = 0.03
    eps: float = 1e-8


class DifferentiableCCVTGuidance:
    """Wrapper that computes continuous barycenters via DifferentiableLloydStep.

    Inputs and outputs use absolute point coordinates in [0, 1]^2.
    Internal Lloyd module operates in [-1, 1]^2, so we convert in/out.
    """

    def __init__(self, cfg: CCVTConfig):
        self.cfg = cfg
        DifferentiableLloydStep = _import_differentiable_lloyd_step()
        self.lloyd_step = DifferentiableLloydStep(
            grid_size=cfg.grid_size,
            tau=cfg.tau,
            num_steps=cfg.num_steps,
        )

    @staticmethod
    def _to_lloyd_space(points01: torch.Tensor) -> torch.Tensor:
        # [0, 1] -> [-1, 1]
        return points01 * 2.0 - 1.0

    @staticmethod
    def _to_image_space(points11: torch.Tensor) -> torch.Tensor:
        # [-1, 1] -> [0, 1]
        return torch.clamp((points11 + 1.0) * 0.5, 0.0, 1.0)

    def compute_barycenters(self, points: torch.Tensor, density_image: torch.Tensor) -> torch.Tensor:
        """Compute continuous target barycenters b_i for each point x_i.

        Args:
            points: (B, N, 2) in [0,1]
            density_image: (B,1,H,W)
        Returns:
            barycenters: (B, N, 2)
        """
        self.lloyd_step = self.lloyd_step.to(points.device)
        points_lloyd = self._to_lloyd_space(points)

        with torch.enable_grad():
            bary_lloyd = self.lloyd_step(points_lloyd, density_image)

        bary = self._to_image_space(bary_lloyd)
        if self.cfg.mode == "repulsion":
            bary = self._apply_repulsion(bary)
        return bary

    def _apply_repulsion(self, points: torch.Tensor) -> torch.Tensor:
        """Simple differentiable repulsion post-step to reduce point collapse."""
        B, N, _ = points.shape
        rad = self.cfg.repulsion_radius
        strength = self.cfg.repulsion_strength
        eps = self.cfg.eps

        delta = points.unsqueeze(2) - points.unsqueeze(1)  # (B,N,N,2)
        d = torch.norm(delta, dim=-1).clamp_min(eps)  # (B,N,N)

        mask = (d < rad).float() - torch.eye(N, device=points.device, dtype=points.dtype).unsqueeze(0)
        weight = torch.clamp((rad - d) / rad, min=0.0) * mask

        force = (weight.unsqueeze(-1) * (delta / d.unsqueeze(-1))).sum(dim=2)
        out = points + strength * force / max(N, 1)
        return torch.clamp(out, 0.0, 1.0)

    def refine(self, points: torch.Tensor, density_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Iteratively compute barycenters and move points toward them.

        Returns:
            refined_points, last_barycenters
        """
        x = points
        b = points
        for _ in range(self.cfg.num_steps):
            b = self.compute_barycenters(x, density_image)
            x = x + 0.5 * (b - x)
            x = torch.clamp(x, 0.0, 1.0)
        return x, b


class CCVTGuidance:
    """Thin API wrapper requested by the v3 implementation spec."""

    def __init__(self, grid_size: int = 64, tau: float = 0.01, num_steps: int = 1, mode: Mode = "ccvt"):
        self.operator = DifferentiableCCVTGuidance(
            CCVTConfig(grid_size=grid_size, tau=tau, num_steps=num_steps, mode=mode)
        )

    def compute_barycenters(self, points: torch.Tensor, density_image: torch.Tensor) -> torch.Tensor:
        return self.operator.compute_barycenters(points, density_image)
