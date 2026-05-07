"""Advanced stippling metrics and GBN-style density visualization.

Extends stippling_metrics.py with:
    - M1-M5 quality metrics (Voronoi, Sinkhorn, EMD, Adaptive-NND, CVT)
    - GBN Algorithm 2 adaptive density reconstruction
    - 4-row overfit panel with M1-M5 text row
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

# Dynamically fetch the current matplotlib global font size so it 
# perfectly syncs with compare_advance_metrics.py
def _fs():
    return plt.rcParams.get('font.size', 12) if plt else 12

# ── Advanced text formatting ──────────────────────────────────────────

def _format_advanced_text(metrics):
    """Format M1-M5 metrics as a compact monospace string for text axes."""
    return (
        f"M1 CVT Energy        : {metrics.get('M1_cvt_energy', 0.0):.4f}\n"
        f"M2 Capacity Constraint: {metrics.get('M2_voronoi_mass_cv', 0.0):.4f}\n"
        f"M3 EMD               : {metrics.get('M3_emd_distance', 0.0):.4f}\n"
        f"M4 Sinkhorn Distance : {metrics.get('M4_sinkhorn_ot_cost', 0.0):.4f}\n"
        f"M5 Spatial Measure   : {metrics.get('M5_spatial_measure_rho_mean', 0.0):.4f}"
    )


def _polygon_sample_points(vertices_clip, image_shape, mc_approx=True, rng=None):
    """Return points and pixel indices inside a polygon.

    When mc_approx is True, samples points randomly inside the polygon bbox.
    When mc_approx is False, uses all pixel centers inside the bbox.
    """
    from matplotlib.path import Path

    H, W = image_shape
    x_min, x_max = vertices_clip[:, 0].min(), vertices_clip[:, 0].max()
    y_min, y_max = vertices_clip[:, 1].min(), vertices_clip[:, 1].max()
    if x_max - x_min < 1e-8 or y_max - y_min < 1e-8:
        return None

    poly_path = Path(vertices_clip)

    if mc_approx:
        if rng is None:
            rng = np.random.default_rng(42)
        n_samples = max(10, min(50, int((x_max - x_min) * (y_max - y_min) * W * H * 0.5)))
        xs = rng.uniform(x_min, x_max, n_samples)
        ys = rng.uniform(y_min, y_max, n_samples)
        points_test = np.column_stack([xs, ys])
    else:
        x0 = max(0, int(np.floor(x_min * W)))
        x1 = min(W - 1, int(np.ceil(x_max * W)))
        y0 = max(0, int(np.floor(y_min * H)))
        y1 = min(H - 1, int(np.ceil(y_max * H)))
        if x1 < x0 or y1 < y0:
            return None
        xs = (np.arange(x0, x1 + 1, dtype=np.float64) + 0.5) / float(W)
        ys = (np.arange(y0, y1 + 1, dtype=np.float64) + 0.5) / float(H)
        gx, gy = np.meshgrid(xs, ys)
        points_test = np.column_stack([gx.ravel(), gy.ravel()])

    inside = poly_path.contains_points(points_test)
    if inside.sum() == 0:
        return None

    points_inside = points_test[inside]
    px = np.clip((points_inside[:, 0] * W).astype(int), 0, W - 1)
    py = np.clip((points_inside[:, 1] * H).astype(int), 0, H - 1)
    return points_inside, px, py


# ── constants for the shared advanced-metrics row ──────────────────
_ADV_ROW_HEIGHT_RATIO   = 2.0
_ADV_ROW_EXTRA_HEIGHT   = 3.5  # inches added to base figure height


def _render_advanced_metrics_row(axes_row, points_list, labels, image_01, metrics_list=None, mc_approx=True):
    """Fill one matplotlib axes row with M1-M5 metric text boxes."""
    axes_row[0].axis("off")
    for j, (pts, label) in enumerate(zip(points_list, labels)):
        ax = axes_row[1 + j]
        ax.axis("off")
        try:
            if metrics_list is not None and j < len(metrics_list) and metrics_list[j] is not None:
                m = metrics_list[j]
            else:
                m = compute_all_advanced_metrics(pts, image_01, mc_approx=mc_approx)
            text = _format_advanced_text(m)
        except Exception:
            text = "(metrics unavailable)"
        ax.text(
            0.5, 0.95, text,
            ha="center", va="top",
            fontsize=_fs(),
            fontfamily="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.85),
        )
        ax.set_title(f"{label} Adv. Metrics", fontsize=_fs())


# ── Advanced Metrics M1-M5 ──────────────────────────────────────────

def compute_m2_capacity_constraint(points, image_01, rng=None, mc_approx=True):
    try:
        from scipy.spatial import Voronoi
        import warnings
        warnings.filterwarnings("ignore")

        if rng is None:
            rng = np.random.default_rng(42)

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
                sample = _polygon_sample_points(vertices_clip, (H_img, W_img), mc_approx=mc_approx, rng=rng)
                if sample is None:
                    masses[region_idx] = 0.0
                    continue
                _, px, py = sample
                masses[region_idx] = float(image_01[py, px].mean())
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


def compute_m4_sinkhorn(points, image_01, target_density=None):
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


def compute_m5_spatial_measure(points, image_01):
    try:
        from scipy.spatial import cKDTree
        N = len(points)
        if N < 2:
            return {
                "spatial_measure_rho_mean": 0.0,
                "spatial_measure_rho_cv": 0.0
            }
        
        # Compute nearest-neighbor distances (r_min)
        tree = cKDTree(points)
        nn_dists, _ = tree.query(points, k=2)
        r_min = nn_dists[:, 1]
        
        # Get local image density at each point
        H, W = image_01.shape
        local_img_density = np.zeros(N)
        for i, (px, py) in enumerate(points):
            grid_x = int(np.clip(px * (W - 1), 0, W - 1))
            grid_y = int(np.clip(py * (H - 1), 0, H - 1))
            local_img_density[i] = image_01[grid_y, grid_x]
        
        # Compute local point density: D_x = N * (local_density / mean(local_density))
        mean_density = local_img_density.mean()
        if mean_density < 1e-10:
            mean_density = 1.0
        D_x = N * (local_img_density / (mean_density + 1e-10))
        
        D_x_safe = np.maximum(D_x, 1e-8)
        r_max = np.sqrt(2.0 / (np.sqrt(3.0) * D_x_safe))
        
        rho_values = np.clip(r_min / (r_max + 1e-10), 0.0, 2.0)
        rho_mean = float(rho_values.mean())
        rho_std = float(rho_values.std())
        rho_cv = rho_std / (rho_mean + 1e-8)
        
        return {
            "spatial_measure_rho_mean": rho_mean,
            "spatial_measure_rho_cv": float(np.clip(rho_cv, 0, 10))
        }
    except Exception:
        return {
            "spatial_measure_rho_mean": 0.0,
            "spatial_measure_rho_cv": 0.0
        }


def compute_m1_cvt_energy(points, image_01, rng=None, mc_approx=True):
    try:
        from scipy.spatial import Voronoi
        if rng is None:
            rng = np.random.default_rng(42)
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
                sample = _polygon_sample_points(vertices_clip, (H, W), mc_approx=mc_approx, rng=rng)
                if sample is None:
                    continue
                points_inside, px, py = sample
                if mc_approx:
                    for x, y, ix, iy in zip(points_inside[:, 0], points_inside[:, 1], px, py):
                        density = image_01[iy, ix]
                        dist_sq = (x - point[0]) ** 2 + (y - point[1]) ** 2
                        total_energy += density * dist_sq
                else:
                    dist_sq = (points_inside[:, 0] - point[0]) ** 2 + (points_inside[:, 1] - point[1]) ** 2
                    total_energy += float(np.sum(image_01[py, px] * dist_sq))
            except Exception:
                continue
        return {"cvt_energy": float(np.clip(total_energy, 0, 1000))}
    except Exception:
        return {"cvt_energy": 0.0}


def compute_m3_emd(points, target_points=None, image_01=None, rng=None, mc_approx=True):
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
                grid_points = np.column_stack([(xx.ravel() + 0.5) / W, (yy.ravel() + 0.5) / H])
                density_weights = image_01.ravel().astype(np.float64)
                density_sum = density_weights.sum()
                if density_sum <= 1e-12:
                    return {"emd_distance": 0.0}
                density_weights = density_weights / density_sum
                if mc_approx and len(grid_points) > 5000:
                    if rng is None:
                        rng = np.random.default_rng(42)
                    idx = rng.choice(len(grid_points), 5000, replace=False, p=density_weights)
                    target_points = grid_points[idx]
                elif len(grid_points) > 20000:
                    stride = max(1, int(np.ceil(np.sqrt(len(grid_points) / 20000.0))))
                    target_points = grid_points.reshape(H, W, 2)[::stride, ::stride].reshape(-1, 2)
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


def compute_all_advanced_metrics(points, image_01, image_input_u8=None, mc_approx=True):
    result = {}
    rng = np.random.default_rng(42)
    m1 = compute_m1_cvt_energy(points, image_01, rng=rng, mc_approx=mc_approx)
    result.update({f"M1_{k}": v for k, v in m1.items()})
    m2 = compute_m2_capacity_constraint(points, image_01, rng=rng, mc_approx=mc_approx)
    result.update({f"M2_{k}": v for k, v in m2.items()})
    m3 = compute_m3_emd(points, None, image_01, rng=rng, mc_approx=mc_approx)
    result.update({f"M3_{k}": v for k, v in m3.items()})
    m4 = compute_m4_sinkhorn(points, image_01)
    result.update({f"M4_{k}": v for k, v in m4.items()})
    m5 = compute_m5_spatial_measure(points, image_01)
    result.update({f"M5_{k}": v for k, v in m5.items()})
    return result


# ── GBN Algorithm 2 + density reconstruction ─────────────────────────

if HAS_TORCH:
    @torch.no_grad()
    def gbn_algorithm_2(points, true_sigma, iterations=10):
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
            a = 0.5 * a + 0.5 * d_k
            mean_sq = (a ** 2).mean()
            a = a / _torch.sqrt(mean_sq + 1e-8)
        return a

    @torch.no_grad()
    def reconstruct_density_map(samples, grid_size=(512, 512), alpha=1.0, iterations=10, batch_size=8192, **kwargs):
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
else:
    def gbn_algorithm_2(*args, **kwargs):
        raise ImportError("gbn_algorithm_2 requires PyTorch; torch not available in this environment")

    def reconstruct_density_map(*args, **kwargs):
        raise ImportError("reconstruct_density_map requires PyTorch; torch not available in this environment")


def process_target_density(source_img_u8, device):
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

    columns = [("Input", None, source_resized)]

    if gt_points is not None:
        gt_np = np.asarray(gt_points, dtype=np.float32)
        gt_map = reconstruct_density_map(_torch.from_numpy(gt_np).to(_device),
                                         grid_size=grid_size, alpha=alpha)
        columns.append(("GT", gt_np, gt_map.cpu().numpy()))

    for i, pts in enumerate(pred_pointsets):
        label = pred_labels[i] if (pred_labels and i < len(pred_labels)) else f"Pred {i + 1}"
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
            ax_top.set_title("Input Image", fontsize=_fs(), fontweight="bold", pad=6)
            ax_top.axis("off")
            ax_bot.imshow(target_density, cmap="gray_r", vmin=0, vmax=1, origin="upper", aspect="equal")
            ax_bot.set_title("Target Density Field", fontsize=_fs(), fontweight="bold", pad=6)
            ax_bot.axis("off")
            continue

        ax_top.scatter(points[:, 0], points[:, 1], s=0.5, c="black", marker=".")
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.invert_yaxis()
        ax_top.set_aspect("equal")
        ax_top.set_title(f"{title} Points", fontsize=_fs(), fontweight="bold", pad=6)
        ax_top.axis("off")

        ax_bot.imshow(density, cmap="gray_r", vmin=0, vmax=1, origin="upper", aspect="equal")
        ax_bot.set_title(f"{title} Density", fontsize=_fs(), fontweight="bold", pad=6)
        ax_bot.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    return save_path


# ── 4-row overfit panel (with optional M1-M5 text row) ───────────────

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
    mc_approx=True,
):
    if not HAS_MPL:
        return None

    plt.rcParams.update({
        'font.size': _fs(),
        'axes.titlesize': _fs(),
        'axes.labelsize': _fs(),
        'xtick.labelsize': _fs(),
        'ytick.labelsize': _fs()
    })

    n_preds = min(len(pred_pointsets), 4)
    n_cols = 2 + n_preds

    if compute_advanced:
        fig, axes = plt.subplots(
            4, n_cols,
            figsize=(4.5 * n_cols, 4.5 * 3 + _ADV_ROW_EXTRA_HEIGHT),
            gridspec_kw={"height_ratios": [3, _ADV_ROW_HEIGHT_RATIO, 3, 3]},
        )
    else:
        fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 4.5 * 3))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    step_label = f" (step {step})" if step is not None else ""
    image_01 = source_img.astype(np.float64) / 255.0

    ax = axes[0, 0]
    ax.imshow(source_img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Source)", fontsize=_fs())
    ax.axis("off")

    ax = axes[0, 1]
    ax.scatter(gt_points[:, 0], 1 - gt_points[:, 1], c="black", s=point_size, alpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.set_facecolor("white")
    ax.set_title("GT (Target)", fontsize=_fs())
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
        ax.set_title(pred_labels[i], fontsize=_fs())
        ax.axis("off")

    all_points = [gt_points] + [pred_pointsets[i] for i in range(n_preds)]
    cap_grid_shape = resolve_capacity_grid_size(image_01, capacity_grid_size)
    all_cap = [compute_grid_capacity(p, image_01, grid_size=cap_grid_shape) for p in all_points]
    all_spa = [compute_spacing_quality(p) for p in all_points]

    cap_row = 2 if compute_advanced else 1
    spa_row = 3 if compute_advanced else 2

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
            axes[cap_row, 0].imshow(binary_img, cmap="gray", vmin=0, vmax=255, origin="upper")
            axes[cap_row, 0].set_title("GT Binary\n(32×32 OT grid)", fontsize=_fs())
            axes[cap_row, 0].axis("off")
        except Exception:
            axes[cap_row, 0].axis("off")
    else:
        axes[cap_row, 0].axis("off")

    col_labels = ["GT (Target)"] + pred_labels
    for j, (cap, label) in enumerate(zip(all_cap, col_labels)):
        ax = axes[cap_row, 1 + j]
        status = cap["grid_status"]
        H_g, W_g = status.shape
        rgb = np.zeros((H_g, W_g, 3), dtype=np.float32)
        rgb[status == 0, :] = [0.0, 1.0, 0.0]   # green = ok
        rgb[status == -1, :] = [1.0, 0.0, 0.0]  # red = underfilled/misplaced
        rgb[status == 1, :] = [0.0, 0.0, 1.0]   # blue = overfilled
        ax.imshow(rgb, origin="upper", aspect="equal")
        ok_pct = 100.0 - cap["underfilled_pct"] - cap["overfilled_pct"]
        ax.set_title(
            f"{label} Capacity\n"
            f"Grid:{cap_grid_shape[0]}x{cap_grid_shape[1]} | "
            f"OK:{ok_pct:.0f}% Under:{cap['underfilled_pct']:.0f}% Over:{cap['overfilled_pct']:.0f}%\n"
            f"Score: {cap['score']:.3f}",
            fontsize=_fs(),
        )
        ax.axis("off")

    if gt_offsets is not None:
        try:
            n = gt_offsets.shape[-1]
            yy, xx = np.mgrid[0:n, 0:n]
            dx, dy = gt_offsets[0], gt_offsets[1]
            mag = np.sqrt(dx * dx + dy * dy)
            ax = axes[spa_row, 0]
            q = ax.quiver(xx, yy, dx, dy, mag, angles="xy", scale_units="xy", scale=1.0,
                          cmap="viridis", width=0.004)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_title("GT Offset Quiver", fontsize=_fs())
            ax.set_xlabel("grid x", fontsize=_fs())
            ax.set_ylabel("grid y", fontsize=_fs())
            ax.tick_params(labelsize=_fs())
            fig.colorbar(q, ax=ax, shrink=0.7, label="|offset|")
        except Exception:
            axes[spa_row, 0].axis("off")
    else:
        axes[spa_row, 0].axis("off")

    all_nn = [s["nn_distances"] for s in all_spa]
    vmin = min(d.min() for d in all_nn)
    vmax = max(d.max() for d in all_nn)

    for j, (spa, pts, label) in enumerate(zip(all_spa, all_points, col_labels)):
        ax = axes[spa_row, 1 + j]
        sc = ax.scatter(pts[:, 0], 1 - pts[:, 1], c=spa["nn_distances"],
                        cmap="RdYlBu", s=point_size * 3, alpha=0.8, vmin=vmin, vmax=vmax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.set_facecolor("white")
        ax.set_title(
            f"{label} Spacing\n"
            f"CV:{spa['nn_cv']:.3f}  Clumped:{spa['clumped_pct']:.1f}%\n"
            f"Score: {spa['spacing_score']:.3f}",
            fontsize=_fs(),
        )
        ax.axis("off")
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("NN dist", fontsize=_fs())
        cbar.ax.tick_params(labelsize=_fs())

    if compute_advanced:
        adv_points = [gt_points] + [pred_pointsets[i] for i in range(n_preds)]
        _render_advanced_metrics_row(axes[1, :], adv_points, col_labels, image_01, mc_approx=mc_approx)

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
    mc_approx=True,
):
    if not HAS_MPL:
        return None

    plt.rcParams.update({
        'font.size': _fs(),
        'axes.titlesize': _fs(),
        'axes.labelsize': _fs(),
        'xtick.labelsize': _fs(),
        'ytick.labelsize': _fs()
    })

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

    all_metrics = [compute_all_advanced_metrics(pts, image_01, mc_approx=mc_approx) for pts in all_pointsets]

    fig, axes = plt.subplots(5, n_cols, figsize=(5 * n_cols, 5 * 5))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    metrics_names = [
        ("M1", "CVT Energy",          "M1_cvt_energy"),
        ("M2", "Capacity Constraint", "M2_voronoi_mass_cv"),
        ("M3", "EMD",                 "M3_emd_distance"),
        ("M4", "Sinkhorn Distance",   "M4_sinkhorn_ot_cost"),
        ("M5", "Spatial Measure",     "M5_spatial_measure_rho_mean"),
    ]

    for m_idx, (m_short, m_long, m_key) in enumerate(metrics_names):
        for p_idx in range(n_cols):
            ax = axes[m_idx, p_idx]
            value = all_metrics[p_idx].get(m_key, 0.0)
            ax.bar([0], [value], color=plt.cm.viridis(m_idx / 5), width=0.5)
            ax.set_ylim(0, max(1.0, value * 1.2))
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_title(f"{all_labels[p_idx]}\n{m_short}: {value:.3f}", fontsize=_fs(), fontweight="bold")
            if p_idx == 0:
                ax.set_ylabel(m_long, fontsize=_fs())
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ── Per-axis spatial visual functions ────────────────────────────────

def _clip_polygon_to_bbox(vertices, bbox=(0.0, 1.0, 0.0, 1.0)):
    xmin, xmax, ymin, ymax = bbox

    def _clip_edge(poly, ax0, ay0, ax1, ay1):
        if not poly:
            return []

        def _inside(p):
            return (ax1 - ax0) * (p[1] - ay0) - (ay1 - ay0) * (p[0] - ax0) >= 0

        def _intersect(p1, p2):
            dx, dy = ax1 - ax0, ay1 - ay0
            dpx, dpy = p2[0] - p1[0], p2[1] - p1[1]
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
    poly = _clip_edge(poly, xmin, ymin, xmax, ymin)   
    poly = _clip_edge(poly, xmax, ymin, xmax, ymax)   
    poly = _clip_edge(poly, xmax, ymax, xmin, ymax)   
    poly = _clip_edge(poly, xmin, ymax, xmin, ymin)   
    return np.array(poly) if poly else np.zeros((0, 2))


def _voronoi_regions_clipped(vor, bbox=(0.0, 1.0, 0.0, 1.0)):
    xmin, xmax, ymin, ymax = bbox
    center = vor.points.mean(axis=0)
    radius = max(xmax - xmin, ymax - ymin) * 4.0

    ridge_map = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridge_map.setdefault(p1, []).append((p2, v1, v2))
        ridge_map.setdefault(p2, []).append((p1, v1, v2))

    ext_verts = vor.vertices.tolist()

    result = []
    for pt_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        is_finite = all(v >= 0 for v in region)

        if is_finite:
            raw = vor.vertices[region]
        else:
            new_region = [v for v in region if v >= 0]

            for p2, v1, v2 in ridge_map.get(pt_idx, []):
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue  

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
    from scipy.ndimage import binary_fill_holes, gaussian_filter

    H, W = density_map.shape

    mask = density_map > bg_threshold
    mask = binary_fill_holes(mask)
    sigma = max(0.5, min(H, W) / 512.0)
    smooth = gaussian_filter(mask.astype(np.float32), sigma=sigma)

    try:
        fig_tmp, ax_tmp = plt.subplots(1, 1)
        cs = ax_tmp.contour(smooth, levels=[0.5])   
        plt.close(fig_tmp)

        try:
            all_paths = list(cs.get_paths())
        except AttributeError:
            all_paths = [p for coll in cs.collections for p in coll.get_paths()]

        if not all_paths:
            return None

        largest = max(all_paths, key=lambda p: len(p.vertices))
        verts = largest.vertices   
        return np.column_stack([verts[:, 0] / W, verts[:, 1] / H])
    except Exception:
        return None


def plot_visual_m2_capacity_constraint(points, image_01, ax, clip_to_domain=True, show_colorbar=True, rng=None, mc_approx=True):
    from matplotlib.patches import Polygon as MplPolygon, PathPatch
    from matplotlib.collections import PatchCollection
    from matplotlib.path import Path, Path as MplPath
    from matplotlib import colors as mcolors
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy.spatial import Voronoi

    plot_title = "[M2] Capacity Constraint (Voronoi Mass)"

    density_map = 1.0 - image_01

    if rng is None:
        rng = np.random.default_rng(42)

    H, W = image_01.shape
    pts_clip = np.clip(points, 1e-5, 1 - 1e-5)

    try:
        vor = Voronoi(pts_clip)
    except Exception:
        ax.set_title(f"{plot_title}\n(insufficient points)", fontsize=_fs())
        return

    def _cell_mass(verts):
        sample = _polygon_sample_points(verts, (H, W), mc_approx=mc_approx, rng=rng)
        if sample is None:
            return 0.0
        _, px, py = sample
        return float(density_map[py, px].mean())

    cell_polys, cell_masses = [], []
    for pt_idx in vor.point_region:
        region = vor.regions[pt_idx]
        if len(region) < 3 or -1 in region:
            continue
        verts = np.clip(vor.vertices[region], 0, 1)
        cell_polys.append(MplPolygon(verts))
        cell_masses.append(_cell_mass(verts))

    cell_masses = np.array(cell_masses)

    mean_mass = cell_masses.mean() if len(cell_masses) > 0 else 1.0
    deviations = np.clip((cell_masses - mean_mass) / (mean_mass + 1e-8), -1.0, 1.0)
    dev_count = int(len(deviations))

    under_thr = -0.15
    over_thr = 0.15
    bucket_edges = np.array([-1.0, under_thr, over_thr, 1.0], dtype=np.float32)
    bucket_colors = ListedColormap(["#2b6cb0", "#d9d9d9", "#c53030"])
    bucket_norm = BoundaryNorm(bucket_edges, bucket_colors.N)

    # --- FIX: Calculate robust 100%-summing display percentages ---
    if dev_count > 0:
        under_raw = float((deviations < under_thr).mean() * 100.0)
        avg_raw = float(((deviations >= under_thr) & (deviations <= over_thr)).mean() * 100.0)
        over_raw = float((deviations > over_thr).mean() * 100.0)
        
        # Largest Remainder Method (Hare-Niemeyer) to ensure exact 100% sum
        raw_pcts = [under_raw, avg_raw, over_raw]
        floors = [int(np.floor(p)) for p in raw_pcts]
        remainders = [p - f for p, f in zip(raw_pcts, floors)]
        diff = 100 - sum(floors)
        
        for i in np.argsort(remainders)[::-1][:diff]:
            floors[i] += 1
            
        u_disp, a_disp, o_disp = floors
    else:
        u_disp = a_disp = o_disp = 0
    # -------------------------------------------------------------

    ax.set_facecolor("white")

    if cell_polys:
        col = PatchCollection(
            cell_polys, cmap=bucket_colors, norm=bucket_norm, alpha=0.8, edgecolor="black", linewidth=0.2
        )
        col.set_array(deviations)

        if clip_to_domain:
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
                closed = np.vstack([contour_pts, contour_pts[0]])
                ax.plot(closed[:, 0], closed[:, 1],
                        color="black", linewidth=0.8, zorder=10, alpha=0.5)

        ax.add_collection(col)

        if show_colorbar:
            tick_positions = [
                (bucket_edges[0] + bucket_edges[1]) / 2.0,
                (bucket_edges[1] + bucket_edges[2]) / 2.0,
                (bucket_edges[2] + bucket_edges[3]) / 2.0,
            ]
            tick_labels = [
                f"Under\n{bucket_edges[0]:.2f} to {bucket_edges[1]:.2f}\n{u_disp}%",
                f"Avg\n{bucket_edges[1]:.2f} to {bucket_edges[2]:.2f}\n{a_disp}%",
                f"Over\n{bucket_edges[2]:.2f} to {bucket_edges[3]:.2f}\n{o_disp}%",
            ]
            cbar = plt.colorbar(col, ax=ax, fraction=0.046, pad=0.04, ticks=tick_positions)
            cbar.ax.set_yticklabels(tick_labels, fontsize=_fs())
            cbar.set_label("Voronoi mass deviation", fontsize=_fs())
            cbar.ax.tick_params(length=0, labelsize=_fs())
            
            # Key fix #2: Detach the colorbar layout constraint so it doesn't permanently shrink the axes
            ax.set_axes_locator(None)

    ax.scatter(pts_clip[:, 0], pts_clip[:, 1], c="black", s=0.5, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    
    if dev_count > 0:
        ax.set_title(f"{plot_title}", fontsize=_fs())
        
        # Key Fix #1: Draw dashed third lines mimicking the capacity grid style
        for x in [0.333, 0.666]:
            ax.plot([x, x], [0, 1], transform=ax.transAxes, color="0.75", linestyle="--", linewidth=0.5, zorder=0)

        # Key Fix #1: Render the texts fully separated at uniform horizontal spacing
        ax.text(0.166, -0.06, f"Under: {u_disp}%", va="top", ha="center", fontsize=_fs(), color="#2b6cb0", transform=ax.transAxes)
        ax.text(0.500, -0.06, f"Avg: {a_disp}%", va="top", ha="center", fontsize=_fs(), color="#555555", transform=ax.transAxes)
        ax.text(0.833, -0.06, f"Over: {o_disp}%", va="top", ha="center", fontsize=_fs(), color="#c53030", transform=ax.transAxes)
    else:
        ax.set_title(f"{plot_title}\n(insufficient points)", fontsize=_fs())


def plot_visual_m5_spatial_measure(points, image_01, ax, show_colorbar=True):
    from scipy.spatial import cKDTree

    plot_title = "[M5] Spatial Measure ρ (Hotspots)"

    N = len(points)
    if N < 2:
        ax.set_title(f"{plot_title}\n(insufficient points)", fontsize=_fs())
        return

    tree = cKDTree(points)
    nn_dists, _ = tree.query(points, k=2)
    r_min = nn_dists[:, 1]

    H, W = image_01.shape
    gx = np.clip((points[:, 0] * (W - 1)).astype(int), 0, W - 1)
    gy = np.clip((points[:, 1] * (H - 1)).astype(int), 0, H - 1)
    local_img_density = 1.0 - image_01[gy, gx]
    
    mean_density = local_img_density.mean()
    if mean_density < 1e-10:
        mean_density = 1.0
    D_x = N * (local_img_density / (mean_density + 1e-10))
    
    D_x_safe = np.maximum(D_x, 1e-8)
    r_max = np.sqrt(2.0 / (np.sqrt(3.0) * D_x_safe))
    
    rho_values = np.clip(r_min / (r_max + 1e-10), 0.0, 2.0)

    sc = ax.scatter(points[:, 0], points[:, 1], c=rho_values,
                    cmap="RdYlBu", s=2.0, vmin=0.4, vmax=1.1)
                    
    if show_colorbar:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("ρ (uniformity)", fontsize=_fs())
        cbar.ax.tick_params(labelsize=_fs())
        
        # Key fix #2: Detach the colorbar layout constraint so it doesn't permanently shrink the axes
        ax.set_axes_locator(None)
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{plot_title}", fontsize=_fs())


def plot_visual_m1_cvt_vectors(points, image_01, ax):
    from scipy.spatial import Voronoi, voronoi_plot_2d

    plot_title = "[M1] CVT Energy (Offsets Error)"

    pts_clip = np.clip(points, 1e-5, 1 - 1e-5)

    try:
        vor = Voronoi(pts_clip)
    except Exception:
        ax.set_title(f"{plot_title}\n(insufficient points)", fontsize=_fs())
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
    ax.set_title(f"{plot_title}", fontsize=_fs())


def plot_visual_m5_spectrum(points, image_01, ax):
    plot_title = "[M5] Power Spectrum"

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
        ax.legend(fontsize=_fs())
        ax.set_title(f"{plot_title}\nSlope: {slope:.3f}", fontsize=_fs())
    else:
        ax.set_title(f"{plot_title}\n(insufficient data)", fontsize=_fs())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.tick_params(axis="both", labelsize=_fs())
    ax.set_xlabel("Radial frequency", fontsize=_fs())
    ax.set_ylabel("Power", fontsize=_fs())


# ── Spatial visual metrics panel ─────────────────────────────────────

def visualize_spatial_metrics_panel(
    image_01,
    pred_pointsets,
    save_path,
    pred_labels=None,
    gt_points=None,
    gt_label="GT",
    clip_to_domain=True,
    show_colorbar=True,
):
    if not HAS_MPL:
        return None

    plt.rcParams.update({
        'font.size': _fs(),
        'axes.titlesize': _fs(),
        'axes.labelsize': _fs(),
        'xtick.labelsize': _fs(),
        'ytick.labelsize': _fs()
    })

    all_pointsets = []
    all_labels = []

    if gt_points is not None:
        all_pointsets.append(np.asarray(gt_points, dtype=np.float64))
        all_labels.append(str(gt_label))

    for i in range(len(pred_pointsets)):
        all_pointsets.append(np.asarray(pred_pointsets[i], dtype=np.float64))

    if pred_labels is None:
        all_labels.extend([f"Pred {i}" for i in range(len(pred_pointsets))])
    else:
        pred_labels = [str(l) for l in pred_labels[:len(pred_pointsets)]]
        if len(pred_labels) < len(pred_pointsets):
            pred_labels.extend(f"Pred {i}" for i in range(len(pred_labels), len(pred_pointsets)))
        all_labels.extend(pred_labels)

    n_cols = len(all_pointsets)
    if n_cols == 0:
        return None

    row_titles = [
        "[M1] CVT Energy (Offsets Error)",
        "[M2] Capacity Constraint (Voronoi Mass)",
        "[M5] Spatial Measure ρ (Hotspots)",
    ]
    plot_fns = [
        plot_visual_m1_cvt_vectors,
        plot_visual_m2_capacity_constraint,
        plot_visual_m5_spatial_measure,
    ]
    n_rows = len(plot_fns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, (pts, label) in enumerate(zip(all_pointsets, all_labels)):
        for row_idx, fn in enumerate(plot_fns):
            ax = axes[row_idx, col_idx]
            try:
                if fn is plot_visual_m2_capacity_constraint:
                    fn(pts, image_01, ax, clip_to_domain=clip_to_domain, show_colorbar=show_colorbar)
                elif fn is plot_visual_m5_spatial_measure:
                    fn(pts, image_01, ax, show_colorbar=show_colorbar)
                else:
                    fn(pts, image_01, ax)
            except Exception as exc:
                ax.axis("off")
                ax.set_title(f"{row_titles[row_idx]}\n(error)", fontsize=_fs())
            if row_idx == 0:
                current = ax.get_title()
                ax.set_title(f"[{label}]  {current}", fontsize=_fs())

    for row_idx, rt in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(rt, fontsize=_fs(), fontweight="bold", labelpad=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
