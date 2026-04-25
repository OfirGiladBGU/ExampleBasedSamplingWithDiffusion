"""Advanced stippling metrics and GBN-style density visualization.

Extends stippling_metrics.py with:
  - M1–M6 quality metrics (Voronoi, Sinkhorn, Adaptive-NND, CVT, Spectrum, EMD)
  - GBN Algorithm 2 adaptive density reconstruction
  - 4-row overfit panel with M1-M6 text row
  - AKDE density-map visualization panel (GBN paper style)
"""

import os
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from utils.stippling_metrics import (
    compute_grid_capacity,
    compute_spacing_quality,
    resolve_capacity_grid_size,
)


# ── Advanced text formatting ──────────────────────────────────────────

def _format_advanced_text(metrics):
    """Format M1–M6 metrics as a compact monospace string for text axes."""
    return (
        f"M1 Voronoi CV : {metrics.get('M1_voronoi_mass_cv', 0.0):.4f}\n"
        f"M2 Sinkhorn OT: {metrics.get('M2_sinkhorn_ot_cost', 0.0):.4f}\n"
        f"M3 Adapt NND  : {metrics.get('M3_adaptive_nnd_cv', 0.0):.4f}\n"
        f"M4 CVT Energy : {metrics.get('M4_cvt_energy', 0.0):.4f}\n"
        f"M5 Spec Slope : {metrics.get('M5_spectrum_slope', 0.0):.4f}\n"
        f"M6 EMD Dist   : {metrics.get('M6_emd_distance', 0.0):.4f}"
    )


# ── constants for the shared advanced-metrics row ──────────────────
_ADV_ROW_FONTSIZE       = 14
_ADV_ROW_TITLE_FONTSIZE = 13
_ADV_ROW_HEIGHT_RATIO   = 2.0
_ADV_ROW_EXTRA_HEIGHT   = 3.5  # inches added to base figure height


def _render_advanced_metrics_row(axes_row, points_list, labels, image_01):
    """Fill one matplotlib axes row with M1-M6 metric text boxes.

    Parameters
    ----------
    axes_row : sequence of Axes
        ``axes_row[0]`` is left blank (INPUT/spacer column);
        ``axes_row[1+j]`` receives metrics for ``points_list[j]``.
    points_list : list of ndarray (N, 2)
        One array per column that should show metrics (GT + preds, or just preds).
    labels : list of str
        Column label for each entry in *points_list*.
    image_01 : ndarray (H, W) float32
        Reference density image used by the metric functions.
    """
    axes_row[0].axis("off")
    for j, (pts, label) in enumerate(zip(points_list, labels)):
        ax = axes_row[1 + j]
        ax.axis("off")
        try:
            m    = compute_all_advanced_metrics(pts, image_01)
            text = _format_advanced_text(m)
        except Exception:
            text = "(metrics unavailable)"
        ax.text(
            0.5, 0.95, text,
            ha="center", va="top",
            fontsize=_ADV_ROW_FONTSIZE,
            fontfamily="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.85),
        )
        ax.set_title(f"{label} Adv. Metrics", fontsize=_ADV_ROW_TITLE_FONTSIZE)


# ── Advanced Metrics M1–M6 ──────────────────────────────────────────

def compute_voronoi_mass_variance(points, image_01):
    """M1: Voronoi-based density mass variance.

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]

    Returns
    -------
    dict with voronoi_mass_cv, voronoi_mass_std, voronoi_mass_mean
    """
    try:
        from scipy.spatial import Voronoi
        import warnings
        warnings.filterwarnings("ignore")

        N = len(points)
        if N < 3:
            return {"voronoi_mass_cv": 0.0, "voronoi_mass_std": 0.0, "voronoi_mass_mean": 0.0}

        pts_clip = np.clip(points, 1e-6, 1 - 1e-6)
        try:
            vor = Voronoi(pts_clip)
        except Exception:
            return {"voronoi_mass_cv": 0.0, "voronoi_mass_std": 0.0, "voronoi_mass_mean": 0.0}

        masses = np.zeros(N)
        H_img, W_img = image_01.shape

        for region_idx, point_idx in enumerate(vor.point_region):
            try:
                region = vor.regions[point_idx]
                if len(region) < 3 or -1 in region:
                    masses[region_idx] = 0.0
                    continue
                vertices = vor.vertices[region]
                vertices_clip = np.clip(vertices, 0, 1)
                x_min, x_max = vertices_clip[:, 0].min(), vertices_clip[:, 0].max()
                y_min, y_max = vertices_clip[:, 1].min(), vertices_clip[:, 1].max()
                if x_max - x_min < 1e-8 or y_max - y_min < 1e-8:
                    masses[region_idx] = 0.0
                    continue
                n_samples = max(10, min(50, int((x_max - x_min) * (y_max - y_min) * W_img * H_img)))
                xs = np.random.uniform(x_min, x_max, n_samples)
                ys = np.random.uniform(y_min, y_max, n_samples)
                from matplotlib.path import Path
                poly_path = Path(vertices_clip)
                points_test = np.column_stack([xs, ys])
                inside = poly_path.contains_points(points_test)
                if inside.sum() > 0:
                    sample_density = []
                    for x, y in points_test[inside]:
                        px = int(np.clip(x * W_img, 0, W_img - 1))
                        py = int(np.clip(y * H_img, 0, H_img - 1))
                        sample_density.append(image_01[py, px])
                    masses[region_idx] = np.mean(sample_density)
                else:
                    masses[region_idx] = 0.0
            except Exception:
                masses[region_idx] = 0.0

        masses = np.maximum(masses, 0.0)
        if masses.sum() < 1e-8:
            return {"voronoi_mass_cv": 0.0, "voronoi_mass_std": 0.0, "voronoi_mass_mean": 0.0}

        mass_mean = masses.mean()
        mass_std = masses.std()
        mass_cv = mass_std / (mass_mean + 1e-8)
        return {
            "voronoi_mass_cv": float(np.clip(mass_cv, 0, 10)),
            "voronoi_mass_std": float(mass_std),
            "voronoi_mass_mean": float(mass_mean),
        }
    except Exception:
        return {"voronoi_mass_cv": 0.0, "voronoi_mass_std": 0.0, "voronoi_mass_mean": 0.0}


def compute_sinkhorn_wasserstein(points, image_01, target_density=None):
    """M2: Sinkhorn-Wasserstein distance to target density.

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]
    target_density : ndarray (H, W) or None

    Returns
    -------
    dict with sinkhorn_ot_cost
    """
    try:
        import ot
        if target_density is None:
            target_density = image_01
        N = len(points)
        H, W = image_01.shape
        if N == 0:
            return {"sinkhorn_ot_cost": 0.0}
        yy, xx = np.mgrid[0:H, 0:W]
        grid_points = np.column_stack([xx.ravel() / W, yy.ravel() / H])
        density_weights = target_density.ravel() / (target_density.sum() + 1e-8)
        density_weights = np.maximum(density_weights, 1e-10)
        density_weights /= density_weights.sum()
        if len(grid_points) > 10000:
            idx = np.random.choice(len(grid_points), 10000, p=density_weights)
            target_pts = grid_points[idx]
            target_w = np.ones(len(target_pts)) / len(target_pts)
        else:
            target_pts = grid_points
            target_w = density_weights
        source_w = np.ones(N) / N
        M = ot.dist(points, target_pts, metric="euclidean")
        try:
            P = ot.sinkhorn(source_w, target_w, M, reg=0.01, numItermax=100)
            ot_cost = np.sum(P * M)
        except Exception:
            ot_cost = 0.0
        return {"sinkhorn_ot_cost": float(np.clip(ot_cost, 0, 100))}
    except Exception:
        return {"sinkhorn_ot_cost": 0.0}


def compute_adaptive_nnd(points, image_01):
    """M3: Adaptive nearest-neighbor distance (density-weighted).

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]

    Returns
    -------
    dict with adaptive_nnd_mean, adaptive_nnd_cv
    """
    try:
        from scipy.spatial import cKDTree
        N = len(points)
        if N < 2:
            return {"adaptive_nnd_mean": 0.0, "adaptive_nnd_cv": 0.0}
        tree = cKDTree(points)
        nn_dists, _ = tree.query(points, k=2)
        nn_dists = nn_dists[:, 1]
        H, W = image_01.shape
        local_density = np.zeros(N)
        for i, (px, py) in enumerate(points):
            grid_x = int(np.clip(px * (W - 1), 0, W - 1))
            grid_y = int(np.clip(py * (H - 1), 0, H - 1))
            local_density[i] = image_01[grid_y, grid_x]
        expected_nn = (local_density + 0.1) ** (-0.5) * 0.5
        adaptive_nnd = nn_dists / (expected_nn + 1e-8)
        adaptive_mean = adaptive_nnd.mean()
        adaptive_std = adaptive_nnd.std()
        adaptive_cv = adaptive_std / (adaptive_mean + 1e-8)
        return {
            "adaptive_nnd_mean": float(np.clip(adaptive_mean, 0, 10)),
            "adaptive_nnd_cv": float(np.clip(adaptive_cv, 0, 10)),
        }
    except Exception:
        return {"adaptive_nnd_mean": 0.0, "adaptive_nnd_cv": 0.0}


def compute_cvt_energy(points, image_01):
    """M4: CVT-like energy (mass-weighted second moment in Voronoi cells).

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]

    Returns
    -------
    dict with cvt_energy
    """
    try:
        from scipy.spatial import Voronoi
        N = len(points)
        if N < 3:
            return {"cvt_energy": 0.0}
        pts_clip = np.clip(points, 1e-6, 1 - 1e-6)
        try:
            vor = Voronoi(pts_clip)
        except Exception:
            return {"cvt_energy": 0.0}
        H, W = image_01.shape
        total_energy = 0.0
        for region_idx, point_idx in enumerate(vor.point_region):
            try:
                region = vor.regions[point_idx]
                if len(region) < 3 or -1 in region:
                    continue
                vertices = vor.vertices[region]
                vertices_clip = np.clip(vertices, 0, 1)
                point = points[region_idx]
                x_min, x_max = vertices_clip[:, 0].min(), vertices_clip[:, 0].max()
                y_min, y_max = vertices_clip[:, 1].min(), vertices_clip[:, 1].max()
                if x_max - x_min < 1e-8 or y_max - y_min < 1e-8:
                    continue
                n_samples = max(10, min(50, int((x_max - x_min) * (y_max - y_min) * W * H)))
                xs = np.random.uniform(x_min, x_max, n_samples)
                ys = np.random.uniform(y_min, y_max, n_samples)
                from matplotlib.path import Path
                poly_path = Path(vertices_clip)
                points_test = np.column_stack([xs, ys])
                inside = poly_path.contains_points(points_test)
                if inside.sum() > 0:
                    for x, y in points_test[inside]:
                        px = int(np.clip(x * W, 0, W - 1))
                        py = int(np.clip(y * H, 0, H - 1))
                        density = image_01[py, px]
                        dist_sq = (x - point[0]) ** 2 + (y - point[1]) ** 2
                        total_energy += density * dist_sq
            except Exception:
                continue
        return {"cvt_energy": float(np.clip(total_energy, 0, 1000))}
    except Exception:
        return {"cvt_energy": 0.0}


def compute_warped_spectrum(points, image_01):
    """M5: Spectral analysis of warped point distribution.

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]

    Returns
    -------
    dict with spectrum_slope, spectrum_peak_freq
    """
    try:
        N = len(points)
        if N < 10:
            return {"spectrum_slope": 0.0, "spectrum_peak_freq": 0.0}
        H, W = image_01.shape
        grid_size = max(H, W)
        point_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        for px, py in points:
            x_idx = int(np.clip(px * grid_size, 0, grid_size - 1))
            y_idx = int(np.clip(py * grid_size, 0, grid_size - 1))
            point_grid[y_idx, x_idx] = 1.0
        fft_result = np.fft.fft2(point_grid)
        power = np.abs(fft_result) ** 2
        yy, xx = np.mgrid[0:grid_size, 0:grid_size]
        rr = np.sqrt((xx - grid_size / 2) ** 2 + (yy - grid_size / 2) ** 2)
        rr = np.maximum(rr, 1.0)
        radial_bins = np.arange(1, grid_size // 2, dtype=np.float32)
        radial_power = []
        for r in radial_bins:
            mask = (rr >= r) & (rr < r + 1)
            if mask.sum() > 0:
                radial_power.append(power[mask].mean())
            else:
                radial_power.append(0.0)
        radial_power = np.array(radial_power)
        if radial_power.max() < 1e-8:
            return {"spectrum_slope": 0.0, "spectrum_peak_freq": 0.0}
        valid_idx = (radial_bins > 0) & (radial_power > 1e-10)
        if valid_idx.sum() > 2:
            log_freq = np.log(radial_bins[valid_idx])
            log_power = np.log(radial_power[valid_idx])
            slope = np.polyfit(log_freq, log_power, 1)[0]
        else:
            slope = 0.0
        peak_freq_idx = np.argmax(radial_power)
        peak_freq = radial_bins[peak_freq_idx] if len(radial_bins) > 0 else 0.0
        return {
            "spectrum_slope": float(np.clip(slope, -5, 5)),
            "spectrum_peak_freq": float(peak_freq / grid_size),
        }
    except Exception:
        return {"spectrum_slope": 0.0, "spectrum_peak_freq": 0.0}


def compute_emd_stats(points, target_points=None, image_01=None):
    """M6: Earth Mover's Distance (EMD) statistics.

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    target_points : ndarray (N, 2) or None
    image_01 : ndarray (H, W) or None

    Returns
    -------
    dict with emd_distance
    """
    try:
        import ot
        N = len(points)
        if N < 2:
            return {"emd_distance": 0.0}
        if target_points is None:
            if image_01 is None:
                target_points = points.copy()
            else:
                H, W = image_01.shape
                yy, xx = np.mgrid[0:H, 0:W]
                grid_points = np.column_stack([xx.ravel() / W, yy.ravel() / H])
                density_weights = image_01.ravel() / (image_01.sum() + 1e-8)
                density_weights = np.maximum(density_weights, 1e-10)
                if len(grid_points) > 5000:
                    idx = np.random.choice(len(grid_points), 5000, p=density_weights)
                    target_points = grid_points[idx]
                else:
                    target_points = grid_points
        if len(target_points) == 0:
            return {"emd_distance": 0.0}
        M = ot.dist(points, target_points, metric="euclidean")
        source_w = np.ones(len(points)) / len(points)
        target_w = np.ones(len(target_points)) / len(target_points)
        try:
            emd = ot.emd2(source_w, target_w, M)
        except Exception:
            emd = 0.0
        return {"emd_distance": float(np.clip(emd, 0, 100))}
    except Exception:
        return {"emd_distance": 0.0}


def compute_all_advanced_metrics(points, image_01, image_input_u8=None):
    """Compute all M1–M6 metrics and return merged dict.

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]
    image_input_u8 : ndarray (H, W) uint8 or None

    Returns
    -------
    dict with all M1–M6 keys prefixed accordingly
    """
    result = {}
    m1 = compute_voronoi_mass_variance(points, image_01)
    result.update({f"M1_{k}": v for k, v in m1.items()})
    m2 = compute_sinkhorn_wasserstein(points, image_01)
    result.update({f"M2_{k}": v for k, v in m2.items()})
    m3 = compute_adaptive_nnd(points, image_01)
    result.update({f"M3_{k}": v for k, v in m3.items()})
    m4 = compute_cvt_energy(points, image_01)
    result.update({f"M4_{k}": v for k, v in m4.items()})
    m5 = compute_warped_spectrum(points, image_01)
    result.update({f"M5_{k}": v for k, v in m5.items()})
    m6 = compute_emd_stats(points, None, image_01)
    result.update({f"M6_{k}": v for k, v in m6.items()})
    return result


# ── GBN Algorithm 2 + density reconstruction ─────────────────────────

@torch.no_grad()
def gbn_algorithm_2(points, true_sigma, iterations=10):
    """Algorithm 2 from the Gaussian Blue Noise paper.

    Iteratively optimises per-point shaping factors (a_k) so that the
    adaptive Gaussian kernels faithfully reproduce the sampled density field.

    Parameters
    ----------
    points : torch.Tensor (N, 2) in [0, 1]
    true_sigma : float – per-point bandwidth (alpha / sqrt(N))
    iterations : int – refinement iterations (paper uses 10)

    Returns
    -------
    torch.Tensor (N,) – optimised shaping factors a_k
    """
    import torch as _torch

    N = points.shape[0]
    device = points.device
    dist_sq = _torch.cdist(points, points) ** 2
    self_mask = _torch.eye(N, device=device, dtype=_torch.bool)
    a = _torch.ones(N, device=device, dtype=_torch.float32)

    for _ in range(iterations):
        a_l = a.unsqueeze(0)
        exponent = -a_l * dist_sq / (2 * true_sigma ** 2)
        exponent = exponent.masked_fill(self_mask, -float("inf"))
        d_k = (a_l * _torch.exp(exponent)).sum(dim=1)
        # 50 % damping: prevents oscillation on sparse point sets
        a = 0.5 * a + 0.5 * d_k
        mean_sq = (a ** 2).mean()
        a = a / _torch.sqrt(mean_sq + 1e-8)

    return a


@torch.no_grad()
def reconstruct_density_map(samples, grid_size=(512, 512), alpha=1.0, iterations=10, batch_size=8192, **kwargs):
    """Reconstruct a density map via GBN Algorithm 2 + Equation 34.

    Parameters
    ----------
    samples : torch.Tensor (N, 2) in [0, 1]
    grid_size : (H, W) output resolution
    alpha : float – bandwidth multiplier (larger = softer/blurrier)
    iterations : int – Algorithm 2 iterations (paper: 10)
    batch_size : int – pixels per GPU batch
    **kwargs : absorbs legacy k / sigma args for call-site compatibility

    Returns
    -------
    torch.Tensor (H, W) normalised [0, 1], on same device as `samples`
    """
    import torch as _torch

    device = samples.device
    H, W = grid_size
    N = samples.shape[0]

    true_sigma = alpha / (N ** 0.5)
    a_k = gbn_algorithm_2(samples, true_sigma, iterations=iterations)

    y = _torch.linspace(0, 1, H, device=device)
    x = _torch.linspace(0, 1, W, device=device)
    grid_y, grid_x = _torch.meshgrid(y, x, indexing="ij")
    grid_coords = _torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    density_map = _torch.zeros(H * W, device=device)
    a_k_row = a_k.unsqueeze(0)

    for i in range(0, H * W, batch_size):
        batch_coords = grid_coords[i : i + batch_size]
        dist_sq = _torch.cdist(batch_coords, samples) ** 2
        weights = a_k_row * _torch.exp(-a_k_row * dist_sq / (2 * true_sigma ** 2))
        density_map[i : i + batch_size] = weights.sum(dim=1)

    density_map = density_map.view(H, W)
    d_min, d_max = density_map.min(), density_map.max()
    if d_max - d_min > 1e-4:
        density_map = (density_map - d_min) / (d_max - d_min)
    else:
        density_map = _torch.zeros_like(density_map)

    return density_map


def process_target_density(source_img_u8, device):
    """Convert an input image into a continuous target density field.

    Inverts so dark = high density, then applies a slight Gaussian blur.
    """
    import torch as _torch
    import torchvision.transforms.functional as TF

    if source_img_u8.ndim == 3:
        img_float = _torch.tensor(source_img_u8, dtype=_torch.float32).mean(dim=-1) / 255.0
    else:
        img_float = _torch.tensor(source_img_u8, dtype=_torch.float32) / 255.0

    img_float = img_float.to(device)
    density_map = 1.0 - img_float
    density_map = density_map.unsqueeze(0).unsqueeze(0)
    smoothed = TF.gaussian_blur(density_map, kernel_size=[5, 5], sigma=[1.0, 1.0])
    smoothed = smoothed.squeeze()
    d_min, d_max = smoothed.min(), smoothed.max()
    if d_max > d_min:
        smoothed = (smoothed - d_min) / (d_max - d_min)
    return smoothed


def visualize_adaptive_sampling_density_map(
    source_img_u8,
    pred_pointsets,
    save_path,
    gt_points=None,
    grid_size=(512, 512),
    k=4,
    alpha=1.0,
    device=None,
    pred_labels=None,
):
    """GBN-style panel: row 0 = point clouds, row 1 = AKDE density maps.

    Columns: Input | GT (if provided) | Pred 1 | Pred 2 …

    Parameters
    ----------
    source_img_u8 : ndarray (H, W) uint8
    pred_pointsets : list of ndarray (N, 2) in [0, 1]
    save_path : str
    gt_points : ndarray (N, 2) or None
    grid_size : (H, W) AKDE output resolution
    alpha : float – GBN bandwidth multiplier
    device : str or None

    Returns
    -------
    save_path or None
    """
    if not HAS_MPL:
        return None

    try:
        import torch as _torch
    except ImportError:
        print("Warning: torch not available, skipping AKDE visualisation")
        return None

    _device = _torch.device(device) if device else _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    target_density = process_target_density(source_img_u8, _device).cpu().numpy()

    try:
        from PIL import Image as _PILImage
        source_resized = np.array(
            _PILImage.fromarray(source_img_u8).resize((grid_size[1], grid_size[0]), _PILImage.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    except Exception:
        from scipy.ndimage import zoom as _zoom
        h, w = source_img_u8.shape
        source_resized = _zoom(source_img_u8.astype(np.float32) / 255.0,
                               (grid_size[0] / h, grid_size[1] / w), order=1)

    # columns: (title, points_np_or_None, density_np)
    columns = [("Input", None, source_resized)]

    if gt_points is not None:
        print(f"  AKDE: reconstructing GT map from {len(gt_points)} points …")
        gt_np = np.asarray(gt_points, dtype=np.float32)
        gt_map = reconstruct_density_map(_torch.from_numpy(gt_np).to(_device),
                                         grid_size=grid_size, alpha=alpha)
        columns.append(("GT", gt_np, gt_map.cpu().numpy()))

    for i, pts in enumerate(pred_pointsets):
        label = pred_labels[i] if (pred_labels and i < len(pred_labels)) else f"Pred {i + 1}"
        print(f"  AKDE: reconstructing {label} map from {len(pts)} points …")
        pts_np = np.asarray(pts, dtype=np.float32)
        pred_map = reconstruct_density_map(_torch.from_numpy(pts_np).to(_device),
                                           grid_size=grid_size, alpha=alpha)
        columns.append((label, pts_np, pred_map.cpu().numpy()))

    n_cols = len(columns)
    if n_cols == 0:
        return None

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 9.0))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, (title, points, density) in enumerate(columns):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        if title == "Input":
            ax_top.imshow(source_resized, cmap="gray", vmin=0, vmax=1, origin="upper", aspect="equal")
            ax_top.set_title("Input Image", fontsize=12, fontweight="bold", pad=6)
            ax_top.axis("off")
            ax_bot.imshow(target_density, cmap="gray_r", vmin=0, vmax=1, origin="upper", aspect="equal")
            ax_bot.set_title("Target Density Field", fontsize=12, fontweight="bold", pad=6)
            ax_bot.axis("off")
            continue

        ax_top.scatter(points[:, 0], points[:, 1], s=0.5, c="black", marker=".")
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.invert_yaxis()
        ax_top.set_aspect("equal")
        ax_top.set_title(f"{title} Points", fontsize=12, fontweight="bold", pad=6)
        ax_top.axis("off")

        ax_bot.imshow(density, cmap="gray_r", vmin=0, vmax=1, origin="upper", aspect="equal")
        ax_bot.set_title(f"{title} Density", fontsize=12, fontweight="bold", pad=6)
        ax_bot.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    return save_path


# ── 4-row overfit panel (with optional M1-M6 text row) ───────────────

def visualize_overfit_metrics(
    source_img,
    target_img,
    gt_points,
    pred_pointsets,
    save_path,
    step=None,
    point_size=0.5,
    gt_offsets=None,
    capacity_grid_size=16,
    pred_labels=None,
    compute_advanced=False,
):
    """4-row comparison panel (adds M1-M6 text row when compute_advanced=True).

    Columns: INPUT | GT | Pred0 | Pred1 | …
      Row 0  Point clouds
      Row 1  Grid Capacity heatmaps
      Row 2  Spacing Quality scatter
      Row 3  (optional) M1-M6 numeric metrics text
    """
    if not HAS_MPL:
        return None

    n_preds = min(len(pred_pointsets), 4)
    n_cols = 2 + n_preds

    if compute_advanced:
        fig, axes = plt.subplots(
            4, n_cols,
            figsize=(4.5 * n_cols, 4.5 * 3 + _ADV_ROW_EXTRA_HEIGHT),
            gridspec_kw={"height_ratios": [3, 3, 3, _ADV_ROW_HEIGHT_RATIO]},
        )
    else:
        fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 4.5 * 3))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    step_label = f" (step {step})" if step is not None else ""
    image_01 = source_img.astype(np.float64) / 255.0

    # Row 0: point clouds
    ax = axes[0, 0]
    ax.imshow(source_img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Source)")
    ax.axis("off")

    ax = axes[0, 1]
    ax.scatter(gt_points[:, 0], 1 - gt_points[:, 1], c="black", s=point_size, alpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.set_facecolor("white")
    ax.set_title("GT (Target)")
    ax.axis("off")

    if pred_labels is None:
        pred_labels = [f"Predict {i}{step_label}" for i in range(n_preds)]
    else:
        pred_labels = [str(l) for l in pred_labels[:n_preds]]
        if len(pred_labels) < n_preds:
            pred_labels.extend(f"Predict {i}{step_label}" for i in range(len(pred_labels), n_preds))

    for i in range(n_preds):
        ax = axes[0, 2 + i]
        pts = pred_pointsets[i]
        ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=point_size, alpha=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.set_facecolor("white")
        ax.set_title(pred_labels[i])
        ax.axis("off")

    all_points = [gt_points] + [pred_pointsets[i] for i in range(n_preds)]
    cap_grid_shape = resolve_capacity_grid_size(image_01, capacity_grid_size)
    all_cap = [compute_grid_capacity(p, image_01, grid_size=cap_grid_shape) for p in all_points]
    all_spa = [compute_spacing_quality(p) for p in all_points]

    # Row 1: grid capacity
    if gt_offsets is not None:
        try:
            from data.Transforms import to_pointset_optimal_transport
            pts_grid = to_pointset_optimal_transport(gt_offsets)
            pts_ot = pts_grid.reshape(2, -1).T
            n = gt_offsets.shape[-1]
            clipped = np.clip(pts_ot, 0.0, 1.0 - 1e-12)
            ij = np.floor(clipped * n).astype(np.int64)
            counts = np.zeros((n, n), dtype=np.int32)
            for x_idx, y_idx in ij:
                counts[y_idx, x_idx] += 1
            binary_img = (counts > 0).astype(np.uint8) * 255
            axes[1, 0].imshow(binary_img, cmap="gray", vmin=0, vmax=255, origin="upper")
            axes[1, 0].set_title("GT Binary\n(32×32 OT grid)", fontsize=9)
            axes[1, 0].axis("off")
        except Exception:
            axes[1, 0].axis("off")
    else:
        axes[1, 0].axis("off")

    col_labels = ["GT (Target)"] + pred_labels
    for j, (cap, label) in enumerate(zip(all_cap, col_labels)):
        ax = axes[1, 1 + j]
        status = cap["grid_status"]
        H_g, W_g = status.shape
        rgb = np.zeros((H_g, W_g, 3), dtype=np.float32)
        rgb[status == 0, 1] = 1.0
        rgb[status == -1, 0] = 1.0
        rgb[status == 1, 2] = 1.0
        ax.imshow(rgb, origin="upper", aspect="equal")
        ok_pct = 100.0 - cap["underfilled_pct"] - cap["overfilled_pct"]
        ax.set_title(
            f"{label} Capacity\n"
            f"Grid:{cap_grid_shape[0]}x{cap_grid_shape[1]} | "
            f"OK:{ok_pct:.0f}% Under:{cap['underfilled_pct']:.0f}% Over:{cap['overfilled_pct']:.0f}%\n"
            f"Score: {cap['score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    # Row 2: spacing quality
    if gt_offsets is not None:
        try:
            n = gt_offsets.shape[-1]
            yy, xx = np.mgrid[0:n, 0:n]
            dx, dy = gt_offsets[0], gt_offsets[1]
            mag = np.sqrt(dx * dx + dy * dy)
            ax = axes[2, 0]
            q = ax.quiver(xx, yy, dx, dy, mag, angles="xy", scale_units="xy", scale=1.0,
                          cmap="viridis", width=0.004)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_title("GT Offset Quiver", fontsize=9)
            ax.set_xlabel("grid x", fontsize=8)
            ax.set_ylabel("grid y", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(q, ax=ax, shrink=0.7, label="|offset|")
        except Exception:
            axes[2, 0].axis("off")
    else:
        axes[2, 0].axis("off")

    all_nn = [s["nn_distances"] for s in all_spa]
    vmin = min(d.min() for d in all_nn)
    vmax = max(d.max() for d in all_nn)

    for j, (spa, pts, label) in enumerate(zip(all_spa, all_points, col_labels)):
        ax = axes[2, 1 + j]
        sc = ax.scatter(pts[:, 0], 1 - pts[:, 1], c=spa["nn_distances"],
                        cmap="RdYlBu", s=point_size * 3, alpha=0.8, vmin=vmin, vmax=vmax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.set_facecolor("white")
        ax.set_title(
            f"{label} Spacing\n"
            f"CV:{spa['nn_cv']:.3f}  Clumped:{spa['clumped_pct']:.1f}%\n"
            f"Score: {spa['spacing_score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")
        plt.colorbar(sc, ax=ax, shrink=0.7, label="NN dist")

    # Row 3: M1-M6 text (optional)
    if compute_advanced:
        adv_points = [gt_points] + [pred_pointsets[i] for i in range(n_preds)]
        _render_advanced_metrics_row(axes[3, :], adv_points, col_labels, image_01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ── Advanced metrics bar-chart panel ─────────────────────────────────

def visualize_advanced_metrics_panel(
    image_01,
    pred_pointsets,
    save_path,
    pred_labels=None,
    gt_points=None,
    gt_label="GT",
):
    """Bar-chart panel of M1–M6 for predictions and optional GT."""
    if not HAS_MPL:
        return None

    n_preds = min(len(pred_pointsets), 4)
    all_pointsets = []
    all_labels = []

    if gt_points is not None:
        all_pointsets.append(np.asarray(gt_points))
        all_labels.append(str(gt_label))

    for i in range(n_preds):
        all_pointsets.append(np.asarray(pred_pointsets[i]))

    if pred_labels is None:
        all_labels.extend([f"Pred {i}" for i in range(n_preds)])
    else:
        pred_labels = [str(l) for l in pred_labels[:n_preds]]
        if len(pred_labels) < n_preds:
            pred_labels.extend(f"Pred {i}" for i in range(len(pred_labels), n_preds))
        all_labels.extend(pred_labels)

    n_cols = len(all_pointsets)
    if n_cols == 0:
        return None

    all_metrics = [compute_all_advanced_metrics(pts, image_01) for pts in all_pointsets]

    fig, axes = plt.subplots(6, n_cols, figsize=(5 * n_cols, 5 * 6))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    metrics_names = [
        ("M1", "Voronoi Mass CV", "M1_voronoi_mass_cv"),
        ("M2", "Sinkhorn OT Cost", "M2_sinkhorn_ot_cost"),
        ("M3", "Adaptive NND CV", "M3_adaptive_nnd_cv"),
        ("M4", "CVT Energy", "M4_cvt_energy"),
        ("M5", "Spectrum Slope", "M5_spectrum_slope"),
        ("M6", "EMD Distance", "M6_emd_distance"),
    ]

    for m_idx, (m_short, m_long, m_key) in enumerate(metrics_names):
        for p_idx in range(n_cols):
            ax = axes[m_idx, p_idx]
            value = all_metrics[p_idx].get(m_key, 0.0)
            ax.bar([0], [value], color=plt.cm.viridis(m_idx / 6), width=0.5)
            ax.set_ylim(0, max(1.0, value * 1.2))
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_title(f"{all_labels[p_idx]}\n{m_short}: {value:.3f}", fontsize=10, fontweight="bold")
            if p_idx == 0:
                ax.set_ylabel(m_long, fontsize=9)
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ── Per-axis spatial visual functions ────────────────────────────────

# -- Voronoi clipping helpers -------------------------------------------------

def _clip_polygon_to_bbox(vertices, bbox=(0.0, 1.0, 0.0, 1.0)):
    """Sutherland-Hodgman polygon clipping to axis-aligned bounding box.

    Parameters
    ----------
    vertices : ndarray (N, 2)
    bbox : (xmin, xmax, ymin, ymax)

    Returns
    -------
    ndarray (M, 2) – clipped polygon vertices (may be empty)
    """
    xmin, xmax, ymin, ymax = bbox

    def _clip_edge(poly, ax0, ay0, ax1, ay1):
        """Clip *poly* against the half-plane to the left of (a0→a1)."""
        if not poly:
            return []

        def _inside(p):
            return (ax1 - ax0) * (p[1] - ay0) - (ay1 - ay0) * (p[0] - ax0) >= 0

        def _intersect(p1, p2):
            dx, dy = ax1 - ax0, ay1 - ay0
            dpx, dpy = p2[0] - p1[0], p2[1] - p1[1]
            # denom = dx*dpy - dy*dpx  (correct Sutherland-Hodgman sign)
            denom = dx * dpy - dy * dpx
            if abs(denom) < 1e-12:
                return p1
            t = (dy * (p1[0] - ax0) - dx * (p1[1] - ay0)) / denom
            return (p1[0] + t * dpx, p1[1] + t * dpy)

        out = []
        for i, curr in enumerate(poly):
            prev = poly[i - 1]
            if _inside(curr):
                if not _inside(prev):
                    out.append(_intersect(prev, curr))
                out.append(curr)
            elif _inside(prev):
                out.append(_intersect(prev, curr))
        return out

    poly = [(v[0], v[1]) for v in vertices]
    # CCW winding of [xmin,xmax]×[ymin,ymax]; "inside" = left of each directed edge
    poly = _clip_edge(poly, xmin, ymin, xmax, ymin)   # bottom: y >= ymin
    poly = _clip_edge(poly, xmax, ymin, xmax, ymax)   # right:  x <= xmax
    poly = _clip_edge(poly, xmax, ymax, xmin, ymax)   # top:    y <= ymax
    poly = _clip_edge(poly, xmin, ymax, xmin, ymin)   # left:   x >= xmin
    return np.array(poly) if poly else np.zeros((0, 2))


def _voronoi_regions_clipped(vor, bbox=(0.0, 1.0, 0.0, 1.0)):
    """Return a list of (clipped_vertices, is_originally_finite) per input point.

    Infinite ridges are extended far enough to be clipped cleanly to *bbox*.
    The returned order matches ``vor.points`` (one entry per point).

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
    bbox : (xmin, xmax, ymin, ymax)

    Returns
    -------
    list of (ndarray (M,2), bool)
    """
    xmin, xmax, ymin, ymax = bbox
    center = vor.points.mean(axis=0)
    # Extend far enough that any infinite ridge will be cut by the bbox
    radius = max(xmax - xmin, ymax - ymin) * 4.0

    # Build ridge lookup: point_index → [(other_point, v1, v2), ...]
    ridge_map = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_map.setdefault(p1, []).append((p2, v1, v2))
        ridge_map.setdefault(p2, []).append((p1, v1, v2))

    # Extended vertex list (we may append far-points)
    ext_verts = vor.vertices.tolist()

    result = []
    for pt_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        is_finite = all(v >= 0 for v in region)

        if is_finite:
            raw = vor.vertices[region]
        else:
            # Start from finite vertices in the region
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridge_map.get(pt_idx, []):
                # Swap so that v2 is the finite endpoint
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue  # both endpoints finite – already in new_region

                # Build a "far point" for the infinite end
                tangent = vor.points[p2] - vor.points[pt_idx]
                norm = np.linalg.norm(tangent)
                if norm < 1e-12:
                    continue
                tangent /= norm
                normal = np.array([-tangent[1], tangent[0]])
                midpoint = vor.points[[pt_idx, p2]].mean(axis=0)
                sign = np.sign(np.dot(midpoint - center, normal))
                far_pt = vor.vertices[v2] + sign * normal * radius
                ext_verts.append(far_pt.tolist())
                new_region.append(len(ext_verts) - 1)

            if len(new_region) < 3:
                result.append((np.zeros((0, 2)), is_finite))
                continue

            ext_arr = np.array(ext_verts)
            vs = ext_arr[new_region]
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            raw = vs[np.argsort(angles)]

        clipped = _clip_polygon_to_bbox(raw, bbox)
        result.append((clipped, is_finite))

    return result


def _extract_subject_contour(density_map, bg_threshold=0.05):
    """Extract the dominant subject-area contour as (N, 2) float in [0, 1].

    Uses ``matplotlib.contour`` on a thresholded + hole-filled mask so no
    extra dependencies (skimage, shapely) are required.

    Parameters
    ----------
    density_map : ndarray (H, W), values in [0, 1]; 1 = subject, 0 = bg
    bg_threshold : float

    Returns
    -------
    ndarray (N, 2) in [0, 1] or None
    """
    from scipy.ndimage import binary_fill_holes, gaussian_filter

    H, W = density_map.shape

    # Binary mask: subject pixels
    mask = density_map > bg_threshold
    # Fill interior holes (e.g. white eyes inside the monkey)
    mask = binary_fill_holes(mask)
    # Very slight smoothing just to avoid 1-pixel jags in the contour path
    sigma = max(0.5, min(H, W) / 512.0)
    smooth = gaussian_filter(mask.astype(np.float32), sigma=sigma)

    try:
        fig_tmp, ax_tmp = plt.subplots(1, 1)
        cs = ax_tmp.contour(smooth, levels=[0.5])   # threshold on the smooth mask
        plt.close(fig_tmp)

        # matplotlib >= 3.8 removed cs.collections; use cs.get_paths() instead
        try:
            all_paths = list(cs.get_paths())
        except AttributeError:
            all_paths = [p for coll in cs.collections for p in coll.get_paths()]

        if not all_paths:
            return None

        # Pick the path enclosing the largest area (proxy: most vertices)
        largest = max(all_paths, key=lambda p: len(p.vertices))
        verts = largest.vertices   # (col, row) in pixel indices
        # Scale to [0, 1]
        return np.column_stack([verts[:, 0] / W, verts[:, 1] / H])
    except Exception:
        return None


def plot_visual_m1_voronoi_mass(points, image_01, ax, clip_to_domain=True):
    """Visual M1: Voronoi cells colored by mass deviation from the mean.

    Red = over-filled (too much density mass), Blue = under-filled.

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]  (0=dark/dense, 1=white/bg)
    ax : matplotlib Axes
    clip_to_domain : bool, default True
        When True, the subject shape is extracted from the image and used as
        a clip path — only the Voronoi mosaic inside the subject silhouette
        is rendered; the white background is fully masked.
        When False, all finite Voronoi cells are colored by deviation without
        any contour clipping or background classification.
    """
    from matplotlib.patches import Polygon as MplPolygon, PathPatch
    from matplotlib.collections import PatchCollection
    from matplotlib.path import Path, Path as MplPath
    from scipy.spatial import Voronoi

    # Stippling density: dark = subject (high), white bg = 0
    density_map = 1.0 - image_01

    H, W = image_01.shape
    pts_clip = np.clip(points, 1e-5, 1 - 1e-5)

    try:
        vor = Voronoi(pts_clip)
    except Exception:
        ax.set_title("M1: Voronoi Mass Deviation\n(insufficient points)", fontsize=9)
        return

    def _cell_mass(verts):
        """Mean stippling density inside a polygon (Monte-Carlo sampling)."""
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
        n = max(10, int((x_max - x_min) * (y_max - y_min) * W * H * 0.5))
        xs = np.random.uniform(x_min, x_max, n)
        ys = np.random.uniform(y_min, y_max, n)
        inside = Path(verts).contains_points(np.column_stack([xs, ys]))
        if inside.sum() == 0:
            return 0.0
        px = np.clip((xs[inside] * W).astype(int), 0, W - 1)
        py = np.clip((ys[inside] * H).astype(int), 0, H - 1)
        return float(density_map[py, px].mean())

    # ── Build Voronoi cell polygons (finite cells only in both modes) ─────
    cell_polys, cell_masses = [], []
    for pt_idx in vor.point_region:
        region = vor.regions[pt_idx]
        if len(region) < 3 or -1 in region:
            continue
        verts = np.clip(vor.vertices[region], 0, 1)
        cell_polys.append(MplPolygon(verts))
        cell_masses.append(_cell_mass(verts))

    cell_masses = np.array(cell_masses)

    # Deviation relative to the mean of all cells
    mean_mass = cell_masses.mean() if len(cell_masses) > 0 else 1.0
    deviations = np.clip((cell_masses - mean_mass) / (mean_mass + 1e-8), -1.0, 1.0)

    ax.set_facecolor("white")

    if cell_polys:
        col = PatchCollection(
            cell_polys, cmap="coolwarm", alpha=0.8, edgecolor="black", linewidth=0.2
        )
        col.set_array(deviations)
        col.set_clim(-1.0, 1.0)

        if clip_to_domain:
            # ── clip_to_domain=True: extract subject contour and clip the
            #    collection so only the inside of the shape is rendered. ──────
            contour_pts = _extract_subject_contour(density_map, bg_threshold=0.05)
            if contour_pts is not None and len(contour_pts) >= 3:
                cp = np.vstack([contour_pts, contour_pts[0]])
                codes = (
                    [MplPath.MOVETO]
                    + [MplPath.LINETO] * (len(cp) - 2)
                    + [MplPath.CLOSEPOLY]
                )
                clip_patch = PathPatch(
                    MplPath(cp, codes), transform=ax.transData, visible=False
                )
                ax.add_patch(clip_patch)
                col.set_clip_path(clip_patch)
                # Draw the contour outline on top for reference
                closed = np.vstack([contour_pts, contour_pts[0]])
                ax.plot(closed[:, 0], closed[:, 1],
                        color="black", linewidth=0.8, zorder=10, alpha=0.5)

        # clip_to_domain=False: draw all finite cells as-is, no contour
        ax.add_collection(col)

    ax.scatter(pts_clip[:, 0], pts_clip[:, 1], c="black", s=0.5, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("M1: Voronoi Mass Deviation\n(Red=Over, Blue=Under)", fontsize=9)


def plot_visual_m3_adaptive_nnd(points, image_01, ax):
    """Visual M3: Points colored by density-adapted nearest-neighbor distance.

    Clumped regions (blue-noise violations) appear as bright hotspots.
    """
    from scipy.spatial import cKDTree

    N = len(points)
    if N < 2:
        ax.set_title("M3: Adaptive NND Hotspots\n(insufficient points)", fontsize=9)
        return

    tree = cKDTree(points)
    nn_dists, _ = tree.query(points, k=2)
    nn_dists = nn_dists[:, 1]

    H, W = image_01.shape
    gx = np.clip((points[:, 0] * (W - 1)).astype(int), 0, W - 1)
    gy = np.clip((points[:, 1] * (H - 1)).astype(int), 0, H - 1)
    # Stippling density: dark = dense (1), white background = 0
    local_density = 1.0 - image_01[gy, gx]

    expected_nn = (local_density + 0.1) ** (-0.5) * 0.5
    adaptive_nnd = nn_dists / (expected_nn + 1e-8)

    sc = ax.scatter(points[:, 0], points[:, 1], c=adaptive_nnd, cmap="viridis", s=2.0)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="Adaptive NND")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("M3: Adaptive NND Hotspots", fontsize=9)


def plot_visual_m4_cvt_vectors(points, image_01, ax):
    """Visual M4: Voronoi boundaries + red lines to each cell's geometric centroid.

    Arrow length/direction show how far and which way each point should move
    to reach the density-weighted CVT optimum.
    """
    from scipy.spatial import Voronoi, voronoi_plot_2d

    pts_clip = np.clip(points, 1e-5, 1 - 1e-5)

    try:
        vor = Voronoi(pts_clip)
    except Exception:
        ax.set_title("M4: CVT Relaxation Vectors\n(insufficient points)", fontsize=9)
        return

    voronoi_plot_2d(
        vor, ax=ax,
        show_points=False, show_vertices=False,
        line_colors="gray", line_width=0.3, line_alpha=0.5,
    )

    for region_idx, point_idx in enumerate(vor.point_region):
        region = vor.regions[point_idx]
        if len(region) < 3 or -1 in region:
            continue
        vertices = np.clip(vor.vertices[region], 0, 1)
        cx, cy = vertices[:, 0].mean(), vertices[:, 1].mean()
        px_pt, py_pt = points[region_idx]
        ax.plot([px_pt, cx], [py_pt, cy], color="red", linewidth=0.8, alpha=0.7)
        ax.scatter(cx, cy, color="red", marker="x", s=5, linewidths=0.5)

    ax.scatter(pts_clip[:, 0], pts_clip[:, 1], c="black", s=1.0, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("M4: CVT Relaxation Vectors", fontsize=9)


def plot_visual_m5_spectrum(points, image_01, ax):
    """Visual M5: Log-log radial power spectrum with regression line overlay.

    Shows whether the spectrum follows a blue-noise power law (straight line)
    and reveals any frequency spikes causing spatial artifacts.
    """
    H, W = image_01.shape
    grid_size = max(H, W)

    point_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    x_idx = np.clip((points[:, 0] * grid_size).astype(int), 0, grid_size - 1)
    y_idx = np.clip((points[:, 1] * grid_size).astype(int), 0, grid_size - 1)
    point_grid[y_idx, x_idx] = 1.0

    power = np.abs(np.fft.fft2(point_grid)) ** 2

    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    rr = np.maximum(
        np.sqrt((xx - grid_size / 2) ** 2 + (yy - grid_size / 2) ** 2), 1.0
    )
    radial_bins = np.arange(1, grid_size // 2, dtype=np.float32)
    radial_power = np.array([
        power[(rr >= r) & (rr < r + 1)].mean() if ((rr >= r) & (rr < r + 1)).sum() > 0 else 0.0
        for r in radial_bins
    ])

    valid = (radial_bins > 0) & (radial_power > 1e-10)
    if valid.sum() > 2:
        log_f = np.log(radial_bins[valid])
        log_p = np.log(radial_power[valid])
        slope, intercept = np.polyfit(log_f, log_p, 1)
        ax.plot(radial_bins[valid], radial_power[valid],
                color="steelblue", alpha=0.8, linewidth=1.0, label="Spectrum")
        ax.plot(
            radial_bins[valid],
            np.exp(intercept) * (radial_bins[valid] ** slope),
            color="red", linestyle="--", linewidth=1.2, label=f"Fit (slope {slope:.2f})",
        )
        ax.legend(fontsize=7)
        ax.set_title(f"M5: Power Spectrum\nSlope: {slope:.3f}", fontsize=9)
    else:
        ax.set_title("M5: Power Spectrum\n(insufficient data)", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_xlabel("Radial frequency", fontsize=7)
    ax.set_ylabel("Power", fontsize=7)


# ── Spatial visual metrics panel ─────────────────────────────────────

def visualize_spatial_metrics_panel(
    image_01,
    pred_pointsets,
    save_path,
    pred_labels=None,
    gt_points=None,
    gt_label="GT",
    clip_to_domain=True,
):
    """4-row spatial-visual panel: M1 Voronoi / M3 NND / M4 CVT / M5 Spectrum.

    Each column is one point set (GT first if provided, then predictions).
    Each row is one spatial metric visualisation.

    Parameters
    ----------
    image_01 : ndarray (H, W) float in [0, 1]
    pred_pointsets : list of ndarray (N, 2) in [0, 1]
    save_path : str
    pred_labels : list of str or None
    gt_points : ndarray (N, 2) or None
    gt_label : str
    clip_to_domain : bool, default True
        Passed to plot_visual_m1_voronoi_mass.  When True, boundary Voronoi
        cells are clipped to [0,1] and drawn white instead of being omitted.

    Returns
    -------
    save_path or None
    """
    if not HAS_MPL:
        return None

    n_preds = min(len(pred_pointsets), 4)
    all_pointsets = []
    all_labels = []

    if gt_points is not None:
        all_pointsets.append(np.asarray(gt_points, dtype=np.float64))
        all_labels.append(str(gt_label))

    for i in range(n_preds):
        all_pointsets.append(np.asarray(pred_pointsets[i], dtype=np.float64))

    if pred_labels is None:
        all_labels.extend([f"Pred {i}" for i in range(n_preds)])
    else:
        pred_labels = [str(l) for l in pred_labels[:n_preds]]
        if len(pred_labels) < n_preds:
            pred_labels.extend(f"Pred {i}" for i in range(len(pred_labels), n_preds))
        all_labels.extend(pred_labels)

    n_cols = len(all_pointsets)
    if n_cols == 0:
        return None

    row_titles = [
        "M1: Voronoi Mass Deviation",
        "M3: Adaptive NND Hotspots",
        "M4: CVT Relaxation Vectors",
        "M5: Power Spectrum",
    ]
    plot_fns = [
        plot_visual_m1_voronoi_mass,
        plot_visual_m3_adaptive_nnd,
        plot_visual_m4_cvt_vectors,
        plot_visual_m5_spectrum,
    ]
    n_rows = len(plot_fns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, (pts, label) in enumerate(zip(all_pointsets, all_labels)):
        for row_idx, fn in enumerate(plot_fns):
            ax = axes[row_idx, col_idx]
            try:
                if fn is plot_visual_m1_voronoi_mass:
                    fn(pts, image_01, ax, clip_to_domain=clip_to_domain)
                else:
                    fn(pts, image_01, ax)
            except Exception as exc:
                ax.axis("off")
                ax.set_title(f"{row_titles[row_idx]}\n(error)", fontsize=8)
            # Prepend column label on top row
            if row_idx == 0:
                current = ax.get_title()
                ax.set_title(f"[{label}]  {current}", fontsize=9)

    # Row labels on left spine
    for row_idx, rt in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(rt, fontsize=10, fontweight="bold", labelpad=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
