import torch


def _base_grid(grid_size, device, dtype):
    coords = torch.arange(grid_size, device=device, dtype=dtype) / grid_size + 0.5 / grid_size
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


def offsets_to_positions(offsets):
    """Convert offset tensors (B,2,H,W) to point positions (B,N,2) in [0,1]."""
    b, _, h, w = offsets.shape
    assert h == w, "Expected square offset grids"
    grid = _base_grid(h, offsets.device, offsets.dtype).unsqueeze(0).expand(b, -1, -1)
    pts = offsets.permute(0, 2, 3, 1).reshape(b, -1, 2)
    pts = grid + pts / h
    return torch.clamp(pts, 0.0, 1.0 - 1e-6)


def positions_to_offsets(positions, grid_size):
    """Convert point positions (B,N,2) in [0,1] to offset tensors (B,2,H,W)."""
    b, n, _ = positions.shape
    expected = grid_size * grid_size
    if n != expected:
        raise ValueError(f"Expected N={expected}, got {n}")
    grid = _base_grid(grid_size, positions.device, positions.dtype).unsqueeze(0).expand(b, -1, -1)
    offsets = (positions - grid) * grid_size
    return offsets.reshape(b, grid_size, grid_size, 2).permute(0, 3, 1, 2).contiguous()


def chamfer_distance_offsets(pred_offsets, gt_offsets):
    """Symmetric Chamfer distance between predicted and GT offsets."""
    pred_pts = offsets_to_positions(pred_offsets)
    gt_pts = offsets_to_positions(gt_offsets)
    dmat = torch.cdist(pred_pts, gt_pts)
    return dmat.min(dim=2).values.mean() + dmat.min(dim=1).values.mean()
