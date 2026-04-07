"""
Utility functions for train_free_v2 guided sampling.

Includes:
- Image loading and preprocessing
- Inverse optimal transport transform
- Point cloud saving and visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings


def load_target_image(
    image_path: str,
    grid_size: int = 32,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess target density image for guidance.
    
    Args:
        image_path (str): Path to grayscale PNG/JPG image.
        grid_size (int): Target grid resolution (default: 32).
        device (str): Device to load tensor on ('cuda' or 'cpu').
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - high_res (1, 1, H, W): Original resolution image, normalized [0, 1]
            - target_density (1, 1, grid_size, grid_size): Downsampled density map
    
    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If image is not grayscale or has invalid format.
    """
    from PIL import Image
    
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    try:
        img = Image.open(path).convert('L')  # Convert to grayscale
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")
    
    # Convert to tensor and normalize to [0, 1]
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    high_res = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    high_res = high_res.to(device)
    
    # Downsample via area-weighted averaging (conservative)
    target_density = F.interpolate(
        high_res,
        size=(grid_size, grid_size),
        mode='area'
    )
    
    return high_res, target_density


def inverse_ot_transform(
    offsets: torch.Tensor,
    grid_size: int = 32
) -> torch.Tensor:
    """
    Inverse optimal transport transform: (B, 2, grid_size, grid_size) -> (B, N, 2).
    
    Converts model output (offset grid) back to point set in [0, 1]^2.
    
    Args:
        offsets (torch.Tensor): Model output, shape (B, 2, grid_size, grid_size)
                               with values in approximately [-1, 1].
        grid_size (int): Grid resolution (should match model output size).
    
    Returns:
        torch.Tensor: Point set (B, N, 2) where N = grid_size^2, values in [0, 1].
    
    Mathematical Details:
        - Each grid cell center: grid_c = (i + 0.5) / grid_size for i in [0, grid_size)
        - Point in cell: point = grid_c + offset / grid_size
        - After clipping: point in [0, 1]
    """
    B, C, H, W = offsets.shape
    
    if C != 2:
        raise ValueError(f"Expected 2 channels, got {C}")
    if H != grid_size or W != grid_size:
        raise ValueError(f"Expected grid {grid_size}x{grid_size}, got {H}x{W}")
    
    # Create grid of cell centers in [0, 1]
    grid_1d = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
    grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    grid_centers = torch.stack([grid_x, grid_y], dim=-1)  # (grid_size, grid_size, 2)
    grid_centers = grid_centers.to(offsets.device)
    
    # Reshape offsets: (B, 2, H, W) -> (B, H, W, 2)
    offsets_reshaped = offsets.permute(0, 2, 3, 1)  # (B, H, W, 2)
    
    # Convert offsets from [-1, 1] to point displacement
    # Offsets are typically in [-1, 1] range representing displacement within a grid cell
    # Scale: offset of ±1 represents ±0.5 cell width
    point_displacement = offsets_reshaped / (2.0 * grid_size)
    
    # Add displacement to grid centers
    points = grid_centers.unsqueeze(0) + point_displacement  # (B, H, W, 2)
    
    # Flatten grid: (B, H, W, 2) -> (B, H*W, 2)
    points = points.reshape(B, -1, 2)  # (B, N, 2) where N = grid_size^2
    
    # Clip to [0, 1]
    points = torch.clamp(points, min=0.0, max=1.0)
    
    return points


def inverse_ot_transform_legacy(
    offsets: np.ndarray,
    grid_size: int = 32
) -> np.ndarray:
    """
    NumPy version of inverse OT transform (for compatibility with existing code).
    
    Args:
        offsets (np.ndarray): (N, 2, grid_size, grid_size) or similar.
        grid_size (int): Grid resolution.
    
    Returns:
        np.ndarray: Points (N, grid_size^2, 2) in [0, 1]^2.
    """
    # Create grid
    grid_1d = (np.arange(grid_size) + 0.5) / grid_size
    grid_y, grid_x = np.meshgrid(grid_1d, grid_1d, indexing='ij')
    grid_centers = np.stack([grid_x, grid_y], axis=-1)  # (grid_size, grid_size, 2)
    
    # Reshape offsets if needed
    if offsets.ndim == 4:  # (N, 2, H, W)
        N = offsets.shape[0]
        points = []
        for i in range(N):
            offset_reshaped = offsets[i].transpose(1, 2, 0)  # (H, W, 2)
            point_displacement = offset_reshaped / (2.0 * grid_size)
            pts = grid_centers + point_displacement
            pts = np.clip(pts, 0, 1.0)
            pts = pts.reshape(-1, 2)
            points.append(pts)
        return np.array(points)
    else:
        raise ValueError(f"Expected 4D array, got {offsets.ndim}D")


def save_point_cloud(
    points: torch.Tensor,
    output_path: str,
    format: str = 'npy'
) -> None:
    """
    Save point cloud to disk.
    
    Args:
        points (torch.Tensor): Point set (B, N, 2) or (N, 2).
        output_path (str): Path to save file (without extension).
        format (str): 'npy' (NumPy) or 'ply' (PLY format).
    
    Raises:
        ValueError: If format is not supported.
    """
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().detach().numpy()
    else:
        points_np = points
    
    output_path = str(output_path)
    
    if format == 'npy':
        filepath = output_path if output_path.endswith('.npy') else output_path + '.npy'
        np.save(filepath, points_np)
        print(f"Saved point cloud to {filepath}")
    
    elif format == 'ply':
        raise NotImplementedError("PLY format not yet implemented")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npy' or 'ply'.")


def visualize_point_cloud_grid(
    points: torch.Tensor,
    output_path: Optional[str] = None,
    title: str = "Generated Point Cloud",
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 100,
    show: bool = False
) -> None:
    """
    Visualize point cloud on a 2D grid.
    
    Args:
        points (torch.Tensor): Points (B, N, 2), (N, 2), or (B, 1, N, 2).
        output_path (str, optional): Path to save figure (no extension).
        title (str): Plot title.
        figsize (Tuple): Figure size.
        dpi (int): DPI for saved figure.
        show (bool): Display plot in notebook/window (default: False).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available, skipping visualization")
        return
    
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().detach().numpy()
    else:
        points_np = points
    
    # Handle batches: take first sample
    if points_np.ndim == 3:  # (B, N, 2)
        points_np = points_np[0]
    elif points_np.ndim == 4:  # (B, 1, N, 2)
        points_np = points_np[0, 0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(points_np[:, 0], points_np[:, 1], s=2, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if output_path:
        filepath = str(output_path) if str(output_path).endswith('.png') else str(output_path) + '.png'
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_density_guidance(
    points: torch.Tensor,
    target_image: torch.Tensor,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 100,
    show: bool = False
) -> None:
    """
    Visualize points overlaid on target density image for comparison.
    
    Args:
        points (torch.Tensor): Points (B, N, 2) or (N, 2).
        target_image (torch.Tensor): Target density (B, 1, H, W) or (1, 1, H, W).
        output_path (str, optional): Path to save figure.
        figsize (Tuple): Figure size.
        dpi (int): DPI.
        show (bool): Display plot (default: False).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available, skipping visualization")
        return
    
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().detach().numpy()
    else:
        points_np = points
    
    if isinstance(target_image, torch.Tensor):
        target_np = target_image.cpu().detach().numpy()
    else:
        target_np = target_image
    
    # Handle batches
    if points_np.ndim == 3:
        points_np = points_np[0]
    if target_np.ndim == 4:
        target_np = target_np[0, 0]
    elif target_np.ndim == 3:
        target_np = target_np[0]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Target image
    axes[0].imshow(target_np, cmap='gray', origin='upper')
    axes[0].set_title('Target Density Image')
    axes[0].set_aspect('equal')
    
    # Points overlaid on density
    axes[1].imshow(target_np, cmap='gray', origin='upper', alpha=0.5)
    axes[1].scatter(
        points_np[:, 0] * target_np.shape[1],
        points_np[:, 1] * target_np.shape[0],
        s=2, alpha=0.5, c='red'
    )
    axes[1].set_title('Generated Points on Density')
    axes[1].set_aspect('equal')
    
    if output_path:
        filepath = str(output_path) if str(output_path).endswith('.png') else str(output_path) + '.png'
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved comparison visualization to {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_density_statistics(
    points: torch.Tensor,
    grid_size: int = 32,
    bins: int = 10
) -> dict:
    """
    Compute statistics about point distribution in generated cloud.
    
    Args:
        points (torch.Tensor): Points (B, N, 2) or (N, 2).
        grid_size (int): Grid resolution for histogram.
        bins (int): Number of histogram bins.
    
    Returns:
        dict: Statistics including point count, coverage, etc.
    """
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().detach().numpy()
    else:
        points_np = points
    
    if points_np.ndim == 3:
        points_np = points_np[0]
    
    N = points_np.shape[0]
    
    # Spatial coverage
    x_min, x_max = points_np[:, 0].min(), points_np[:, 0].max()
    y_min, y_max = points_np[:, 1].min(), points_np[:, 1].max()
    coverage = (x_max - x_min) * (y_max - y_min)
    
    # Point density
    hist, _ = np.histogram(points_np[:, 0], bins=bins)
    density_std = hist.std() / hist.mean() if hist.mean() > 0 else 0
    
    stats = {
        'num_points': N,
        'x_range': (float(x_min), float(x_max)),
        'y_range': (float(y_min), float(y_max)),
        'spatial_coverage': float(coverage),
        'density_uniformity': 1.0 - min(float(density_std), 1.0)  # Higher is more uniform
    }
    
    return stats
