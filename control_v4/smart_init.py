import numpy as np
import torch

from data.Transforms import to_image_optimal_transport


def generate_smart_init_points_from_density(image_2d, n_points, seed=42):
    """Sample points from dark regions using rejection sampling.

    Input image is expected in [0,1], where 0 is black (ink) and 1 is white.
    Returned points are normalized to [0,1] with shape (N, 2).
    """
    image = np.asarray(image_2d, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Ink probability: black pixels should accept more points.
    prob = np.clip(1.0 - image, 0.0, 1.0)
    h, w = prob.shape
    rng = np.random.RandomState(seed)

    # Guard: if the image has no ink, rejection sampling will loop forever.
    if prob.sum() <= 0.0:
        print(
            "WARNING: generate_smart_init_points_from_density received an all-white image "
            "(no ink detected). Falling back to uniform random initialization."
        )
        x = rng.uniform(0.0, 1.0, n_points)
        y = rng.uniform(0.0, 1.0, n_points)
        return np.stack([x, y], axis=1).astype(np.float32)

    points = []
    batch = max(4096, n_points * 8)
    while len(points) < n_points:
        xs = rng.uniform(0.0, w, size=batch)
        ys = rng.uniform(0.0, h, size=batch)
        ps = rng.uniform(0.0, 1.0, size=batch)

        xi = np.clip(xs.astype(np.int32), 0, w - 1)
        yi = np.clip(ys.astype(np.int32), 0, h - 1)
        keep = ps < prob[yi, xi]

        accepted_x = xs[keep] / max(w, 1)
        accepted_y = ys[keep] / max(h, 1)
        for x, y in zip(accepted_x, accepted_y):
            points.append([x, y])
            if len(points) >= n_points:
                break

    return np.asarray(points, dtype=np.float32)


def smart_init_points_to_offsets(points):
    """Convert smart-init points (N,2 in [0,1]) to offset grid (2,G,G)."""
    offsets = to_image_optimal_transport(points.astype(np.float32))
    return offsets.astype(np.float32, copy=False)


def render_smart_init_grid(points, grid_size=32):
    """Render smart-init points into a normalized low-res occupancy map (1,G,G)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points shape (N,2), got {pts.shape}")

    g = int(grid_size)
    grid = np.zeros((g, g), dtype=np.float32)

    px = np.clip((pts[:, 0] * g).astype(np.int32), 0, g - 1)
    py = np.clip((pts[:, 1] * g).astype(np.int32), 0, g - 1)
    np.add.at(grid, (py, px), 1.0)

    if grid.max() > 0.0:
        grid /= grid.max()

    return grid[None, ...]


def build_smart_init_from_image(image_2d, grid_size=32, n_points=None, seed=42):
    """Build points, offset grid, and rendered smart-init map from an image."""
    g = int(grid_size)
    if n_points is None:
        n_points = g * g

    points = generate_smart_init_points_from_density(image_2d, n_points=n_points, seed=seed)
    offsets = smart_init_points_to_offsets(points)
    smart_grid = render_smart_init_grid(points, grid_size=g)
    return points, offsets, smart_grid


def save_smart_init_debug(out_dir, points, offsets, smart_grid, model_input_grid=None):
    """Save smart-init artifacts for debugging."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "smart_init_points.npy"), np.asarray(points, dtype=np.float32))
    np.save(os.path.join(out_dir, "smart_init_offsets.npy"), np.asarray(offsets, dtype=np.float32))
    np.save(os.path.join(out_dir, "smart_init_grid.npy"), np.asarray(smart_grid, dtype=np.float32))
    if model_input_grid is not None:
        np.save(
            os.path.join(out_dir, "smart_init_grid_model_input.npy"),
            np.asarray(model_input_grid, dtype=np.float32),
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_cols = 3 if model_input_grid is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), dpi=140)
        axes[0].imshow(np.squeeze(smart_grid), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("smart_init_grid_raw")
        axes[0].axis("off")

        axes[1].scatter(points[:, 0], 1.0 - points[:, 1], c="black", s=0.7, alpha=0.8)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_aspect("equal")
        axes[1].set_title("smart_init_points")
        axes[1].axis("off")

        if model_input_grid is not None:
            axes[2].imshow(np.squeeze(model_input_grid), cmap="gray", vmin=0.0, vmax=1.0)
            axes[2].set_title("smart_init_grid_model_input")
            axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "smart_init_collage.png"), dpi=140, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def add_noise_at_t(x_init, alpha_cumprod_t, noise=None):
    """Apply explicit forward noising: sqrt(a)*x + sqrt(1-a)*eps."""
    if noise is None:
        noise = torch.randn_like(x_init)
    a = torch.as_tensor(alpha_cumprod_t, dtype=x_init.dtype, device=x_init.device)
    while a.ndim < x_init.ndim:
        a = a.unsqueeze(-1)
    return a.sqrt() * x_init + (1.0 - a).sqrt() * noise
