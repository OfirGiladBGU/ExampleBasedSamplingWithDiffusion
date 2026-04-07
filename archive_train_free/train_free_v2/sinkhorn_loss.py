"""
Sinkhorn Density Loss for Diffusion Posterior Sampling

This module implements a differentiable Sinkhorn optimal transport loss
that can be used to guide diffusion models toward target density distributions.

The loss computes the Wasserstein distance between:
- Source: Uniform distribution over predicted points (B, N, 2)
- Target: Weighted distribution over image density (B, 1, H, W)

Uses log-domain Sinkhorn-Knopp algorithm for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornDensityLoss(nn.Module):
    """
    Differentiable Sinkhorn optimal transport loss for point clouds.
    
    Computes the Earth Mover's Distance (Wasserstein-2) between predicted points
    and a target density image using the Sinkhorn algorithm.
    
    Args:
        blur (float): Entropy regularization parameter (default: 0.05).
                     Lower = sharper matching, higher = smoother.
        grid_size (int): Resolution for density grid (default: 32).
                        Downsamples image to grid_size x grid_size.
        scaling (float): Sinkhorn convergence scaling factor (default: 0.5).
        p (int): Wasserstein distance order (default: 2 for Wasserstein-2).
        max_iters (int): Maximum Sinkhorn iterations (default: 50).
        eps_clamp (float): Minimum value for numerical stability (default: 1e-8).
    """
    
    def __init__(
        self,
        blur: float = 0.05,
        grid_size: int = 32,
        scaling: float = 0.5,
        p: int = 2,
        max_iters: int = 50,
        eps_clamp: float = 1e-8,
        density_mode: str = "dark",
        density_gamma: float = 1.0,
        contrast_stretch: bool = True
    ):
        super().__init__()
        self.blur = blur
        self.grid_size = grid_size
        self.scaling = scaling
        self.p = p
        self.max_iters = max_iters
        self.eps_clamp = eps_clamp
        self.density_mode = density_mode
        self.density_gamma = density_gamma
        self.contrast_stretch = contrast_stretch

        if self.density_mode not in ["dark", "light"]:
            raise ValueError("density_mode must be either 'dark' or 'light'")
        
        # Pre-compute grid coordinates in [0, 1] (image domain)
        grid_1d = torch.linspace(0.0, 1.0, grid_size)
        grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # (grid_size, grid_size, 2)
        self.register_buffer('grid_coords', grid_coords)
    
    def forward(self, pred_points: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn loss between predicted points and target density.
        
        Args:
            pred_points (torch.Tensor): Predicted points, shape (B, N, 2) in [0, 1].
            target_image (torch.Tensor): Target density image, shape (B, 1, H, W) in [0, 1].
                                        Dark pixels (high intensity) → high density.
        
        Returns:
            torch.Tensor: Scalar loss value (differentiable). Sum over batch.
        """
        B, N, _ = pred_points.shape
        device = pred_points.device
        
        # 1. Extract and invert density from target image
        # Downsample to grid_size via area averaging (conservative)
        target_density = F.interpolate(
            target_image,
            size=(self.grid_size, self.grid_size),
            mode='area'
        )  # (B, 1, grid_size, grid_size)
        
        # Optional per-image contrast stretch to amplify density signal.
        if self.contrast_stretch:
            mins = target_density.amin(dim=(2, 3), keepdim=True)
            maxs = target_density.amax(dim=(2, 3), keepdim=True)
            target_density = (target_density - mins) / (maxs - mins + self.eps_clamp)

        # Map image intensity to density by polarity.
        if self.density_mode == "dark":
            target_density = 1.0 - target_density
        else:
            target_density = target_density

        # Gamma shaping: >1 sharpens peaks, <1 flattens.
        if self.density_gamma != 1.0:
            target_density = torch.pow(torch.clamp(target_density, min=0.0), self.density_gamma)

        target_density = target_density.view(B, -1)  # (B, grid_size^2)
        
        # Ensure positivity for probability weights
        target_density = torch.clamp(target_density, min=self.eps_clamp)
        
        # 2. Normalize to probability distributions
        # Source: Uniform over N points
        source_weights = torch.ones(B, N, device=device) / N
        
        # Target: Normalized density
        target_weights = target_density / target_density.sum(dim=1, keepdim=True)
        
        # 3. Compute cost matrix: ||pred_points[i] - grid[j]||^p
        grid_flat = self.grid_coords.view(-1, 2).to(device)  # (grid_size^2, 2)
        grid_expanded = grid_flat.unsqueeze(0).expand(B, -1, -1)  # (B, grid_size^2, 2)
        
        # Cost matrix: (B, N, grid_size^2)
        # Use squared Euclidean cost for Wasserstein-2 behavior.
        cost_matrix = torch.cdist(pred_points, grid_expanded, p=self.p) ** 2
        
        # 4. Sinkhorn algorithm in log-domain for numerical stability
        log_P = self._sinkhorn_log_domain(
            cost_matrix,
            source_weights,
            target_weights
        )
        
        # 5. Compute loss: sum of transport matrix weighted by cost
        # Transport matrix in probability space
        P = torch.exp(log_P)  # (B, N, grid_size^2)
        
        # Loss per batch: sum_ij P[b,i,j] * cost[b,i,j]
        loss_per_sample = (P * cost_matrix).sum(dim=[1, 2])  # (B,)
        
        # Return sum over batch (not mean, to keep per-sample gradient constant)
        return loss_per_sample.sum()
    
    def _sinkhorn_log_domain(
        self,
        cost: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        num_iters: int = None
    ) -> torch.Tensor:
        """
        Sinkhorn algorithm in log-domain for numerical stability.
        
        Solves: max_{u,v} <u, a> + <v, b> - <exp(-u/eps - cost/eps - v/eps), 1>
        
        Args:
            cost (torch.Tensor): Cost matrix, shape (B, N, M).
            a (torch.Tensor): Source weights, shape (B, N).
            b (torch.Tensor): Target weights, shape (B, M).
            num_iters (int): Number of iterations (default: self.max_iters).
        
        Returns:
            torch.Tensor: Log of transport matrix (B, N, M).
        """
        if num_iters is None:
            num_iters = self.max_iters
        
        B, N, M = cost.shape
        device = cost.device
        
        # Entropy regularization: smaller blur = sharper
        eps = self.blur ** 2
        
        # Log of kernel: K = exp(-cost / eps)
        # Clamp to avoid numerical saturation when points are far from the image domain.
        log_K = torch.clamp(-cost / eps, min=-80.0, max=0.0)
        
        # Initialize dual variables in log-space
        log_u = torch.zeros(B, N, device=device)
        log_v = torch.zeros(B, M, device=device)
        
        # Sinkhorn-Knopp iterations
        for iteration in range(num_iters):
            # Update log_u
            log_sum_exp_v = torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            log_u = torch.log(a + self.eps_clamp) - log_sum_exp_v
            
            # Update log_v
            log_sum_exp_u = torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
            log_v = torch.log(b + self.eps_clamp) - log_sum_exp_u
        
        # Compute log of transport matrix
        # P = diag(u) @ K @ diag(v)  in log domain: log(P) = log(u) + log(K) + log(v)
        log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
        
        return log_P


class SinkhornDensityLossSimple(nn.Module):
    """
    Simplified Sinkhorn loss without geomloss dependency (fallback).
    
    Same interface as SinkhornDensityLoss but uses basic PyTorch operations.
    Slightly slower but fully portable.
    """
    
    def __init__(
        self,
        blur: float = 0.05,
        grid_size: int = 32,
        max_iters: int = 50,
        eps_clamp: float = 1e-8
    ):
        super().__init__()
        self.blur = blur
        self.grid_size = grid_size
        self.max_iters = max_iters
        self.eps_clamp = eps_clamp
        
        # Pre-compute grid in [0, 1] (image domain)
        grid_1d = torch.linspace(0.0, 1.0, grid_size)
        grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer('grid_coords', grid_coords)
    
    def forward(self, pred_points: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        """Same interface as SinkhornDensityLoss."""
        B, N, _ = pred_points.shape
        device = pred_points.device
        
        # Downsample target image
        target_density = F.interpolate(
            target_image,
            size=(self.grid_size, self.grid_size),
            mode='area'
        )
        target_density = 1.0 - target_density
        target_density = target_density.view(B, -1)
        target_density = torch.clamp(target_density, min=self.eps_clamp)
        
        # Normalize to probabilities
        source_weights = torch.ones(B, N, device=device) / N
        target_weights = target_density / target_density.sum(dim=1, keepdim=True)
        
        # Compute cost matrix (squared Euclidean)
        grid_flat = self.grid_coords.view(-1, 2).to(device)
        grid_expanded = grid_flat.unsqueeze(0).expand(B, -1, -1)
        cost_matrix = torch.cdist(pred_points, grid_expanded, p=2) ** 2
        
        # Sinkhorn in log-domain
        log_P = self._sinkhorn_log_domain(cost_matrix, source_weights, target_weights)
        P = torch.exp(log_P)
        
        # Loss
        loss_per_sample = (P * cost_matrix).sum(dim=[1, 2])
        return loss_per_sample.sum()
    
    def _sinkhorn_log_domain(
        self,
        cost: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Log-domain Sinkhorn algorithm."""
        B, N, M = cost.shape
        device = cost.device
        eps = self.blur ** 2
        
        log_K = torch.clamp(-cost / eps, min=-80.0, max=0.0)
        log_u = torch.zeros(B, N, device=device)
        log_v = torch.zeros(B, M, device=device)
        
        for _ in range(self.max_iters):
            log_sum_exp_v = torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            log_u = torch.log(a + self.eps_clamp) - log_sum_exp_v
            
            log_sum_exp_u = torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)
            log_v = torch.log(b + self.eps_clamp) - log_sum_exp_u
        
        log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
        return log_P
