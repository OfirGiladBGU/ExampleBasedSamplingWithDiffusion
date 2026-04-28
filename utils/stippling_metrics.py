"""Shared stippling quality metrics and visualization.

Adapted from PointDiT v5 stippling_metrics.py for the [0,1] coordinate
system used by this project (ExampleBasedSamplingWithDiffusion).

Provides two metrics:
  1. Grid Capacity  -- CCVT-style: do grid cells have the right point count?
  2. Spacing Quality -- Blue-noise check: are NN distances uniform, no clumping?

And a combined 3-row visualization:
  Row 0: Source image | GT scatter | Pred scatters ...
  Row 1: (empty)      | GT capacity | Pred capacities ...
  Row 2: (empty)      | GT spacing  | Pred spacings ...
"""

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False


# ── metric computations (pure numpy, points in [0,1]) ───────────────

def compute_grid_capacity(points, image_01, grid_size=(16, 16), ignore_white=True, white_threshold=0.9):
    """CCVT-style grid capacity fulfillment.

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]
    image_01 : ndarray (H, W) float in [0, 1]  (0=black/dark, 1=white/light)
    grid_size : (rows, cols)
    ignore_white : bool
        If True, cells whose downsampled value exceeds *white_threshold* are
        treated as background (expected = 0 points).  An empty white cell is
        shown green (OK); a white cell that contains points is shown red
        (misplaced dots penalise the score).
    white_threshold : float
        Pixel value above which a cell is considered "white" (default 0.9).

    Returns
    -------
    dict with grid_status (rows, cols), score, underfilled_pct, overfilled_pct
    grid_status values: 0=ok (green), -1=underfilled/misplaced (red), 1=overfilled (blue)
    """
    from scipy.ndimage import zoom

    N = len(points)
    H_grid, W_grid = grid_size
    H_img, W_img = image_01.shape

    scale_h = H_grid / H_img
    scale_w = W_grid / W_img
    image_down = zoom(image_01, (scale_h, scale_w), order=1)

    white_mask = (image_down > white_threshold) if ignore_white else None

    # Build expected density — white cells get 0 expected points
    expected_weight = 1.0 - image_down + 0.01
    if ignore_white and white_mask is not None:
        expected_weight[white_mask] = 0.0
    total_weight = expected_weight.sum()
    if total_weight > 0:
        expected_weight /= total_weight
    grid_expected = expected_weight * N

    pts = np.clip(points, 0, 1 - 1e-6)
    col_idx = (pts[:, 0] * W_grid).astype(int)
    row_idx = (pts[:, 1] * H_grid).astype(int)

    grid_actual = np.zeros((H_grid, W_grid), dtype=np.float64)
    np.add.at(grid_actual, (row_idx, col_idx), 1)

    grid_ratio = grid_actual / (grid_expected + 1e-6)

    grid_status = np.zeros_like(grid_ratio, dtype=int)
    grid_status[grid_ratio < 0.5] = -1
    grid_status[grid_ratio > 2.0] = 1

    # Override white cells: empty=green (OK), any points=red (misplaced)
    if ignore_white and white_mask is not None:
        grid_status[white_mask & (grid_actual == 0)] = 0   # green — correctly empty
        grid_status[white_mask & (grid_actual > 0)] = -1   # red — dots in background

    # All cells are significant: dark cells by expected>0.5, white cells always
    if ignore_white and white_mask is not None:
        significant = (grid_expected > 0.5) | white_mask
    else:
        significant = grid_expected > 0.5
    n_sig = significant.sum()
    if n_sig > 0:
        ok = ((grid_status == 0) & significant).sum()
        under = ((grid_status == -1) & significant).sum()
        over = ((grid_status == 1) & significant).sum()
        score = ok / n_sig
        underfilled_pct = 100.0 * under / n_sig
        overfilled_pct = 100.0 * over / n_sig
    else:
        score, underfilled_pct, overfilled_pct = 1.0, 0.0, 0.0

    return {
        "grid_status": grid_status,
        "score": float(score),
        "underfilled_pct": float(underfilled_pct),
        "overfilled_pct": float(overfilled_pct),
    }


def compute_spacing_quality(points):
    """Nearest-neighbour spacing quality (blue-noise check).

    Parameters
    ----------
    points : ndarray (N, 2) in [0, 1]

    Returns
    -------
    dict with nn_distances (N,), nn_cv, clumped_pct, spacing_score
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    nn_dists, _ = tree.query(points, k=2)
    nn_dists = nn_dists[:, 1]

    nn_mean = nn_dists.mean()
    nn_std = nn_dists.std()
    nn_cv = nn_std / (nn_mean + 1e-8)

    N = len(points)
    expected_nn = 0.5 * (1.0 / N) ** 0.5
    clump_threshold = 0.3 * expected_nn
    clumped_mask = nn_dists < clump_threshold
    clumped_pct = 100.0 * clumped_mask.mean()

    cv_penalty = min(nn_cv, 1.0)
    clump_penalty = clumped_pct / 100.0
    spacing_score = max(0.0, 1.0 - 0.5 * cv_penalty - 0.5 * clump_penalty)

    return {
        "nn_distances": nn_dists,
        "nn_cv": float(nn_cv),
        "clumped_pct": float(clumped_pct),
        "spacing_score": float(spacing_score),
    }


def geometric_validation_score(pointsets, clump_weight=5.0):
    """Aggregate CV/clumping score used by v3/v4 train scripts.

    Accepts either a single point set ``(N, 2)`` or an iterable/batch of point
    sets, and returns a dict with mean ``cv``, ``clumped_pct``, and ``score``.
    """
    arr = np.asarray(pointsets)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        pointsets_iter = [arr]
    else:
        pointsets_iter = [np.asarray(p) for p in pointsets]

    cvs = []
    clumped_pcts = []
    per_sample_scores = []

    for pts in pointsets_iter:
        spacing = compute_spacing_quality(pts)
        cv = float(spacing["nn_cv"])
        clumped_pct = float(spacing["clumped_pct"])
        score = cv + clump_weight * (clumped_pct / 100.0)
        cvs.append(cv)
        clumped_pcts.append(clumped_pct)
        per_sample_scores.append(score)

    return {
        "cv": float(np.mean(cvs)) if cvs else 0.0,
        "clumped_pct": float(np.mean(clumped_pcts)) if clumped_pcts else 0.0,
        "score": float(np.mean(per_sample_scores)) if per_sample_scores else 0.0,
    }


def resolve_capacity_grid_size(image_01, capacity_grid_size):
    """Resolve capacity grid size from user option.

        capacity_grid_size semantics:
            - int > 0: use square (k, k)
            - int == -1: use full image resolution (H, W)
            - tuple/list (h, w): use explicit grid shape
    """
    h_img, w_img = image_01.shape
    if isinstance(capacity_grid_size, (tuple, list)) and len(capacity_grid_size) == 2:
        h_grid = max(1, int(capacity_grid_size[0]))
        w_grid = max(1, int(capacity_grid_size[1]))
        return h_grid, w_grid

    k = int(capacity_grid_size)
    if k == -1:
        return h_img, w_img
    if k <= 0:
        raise ValueError(
            "capacity_grid_size must be > 0, or -1 for full image resolution"
        )
    return k, k


# ── visualization ────────────────────────────────────────────────────

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
):
    """Create 3-row comparison figure with metrics.

    Layout (columns: INPUT | GT | Pred0 | Pred1 | ...):
      Row 0  Point clouds (INPUT shows source image, others show scatter)
      Row 1  GT binary occupancy (32x32) | Grid Capacity per column
      Row 2  GT offset quiver           | Spacing Quality per column

    Parameters
    ----------
    source_img : ndarray (H, W) uint8
    target_img : ndarray (H, W) uint8
    gt_points : ndarray (N, 2) in [0, 1]
    pred_pointsets : list of ndarray (N, 2) in [0, 1]
    save_path : str
    step : int or None
    """
    if not HAS_MPL:
        return None

    n_preds = min(len(pred_pointsets), 4)
    n_cols = 2 + n_preds  # INPUT + GT + preds
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    step_label = f" (step {step})" if step is not None else ""

    image_01 = source_img.astype(np.float64) / 255.0

    # ── Row 0: point clouds ──────────────────────────────────────────
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
        pred_labels = [str(label) for label in pred_labels[:n_preds]]
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

    # ── compute metrics for GT + each prediction ─────────────────────
    all_points = [gt_points] + [pred_pointsets[i] for i in range(n_preds)]
    cap_grid_shape = resolve_capacity_grid_size(image_01, capacity_grid_size)
    all_cap = [compute_grid_capacity(p, image_01, grid_size=cap_grid_shape) for p in all_points]
    all_spa = [compute_spacing_quality(p) for p in all_points]

    # ── Row 1: grid capacity ─────────────────────────────────────────
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
            fontsize=9,
        )
        ax.axis("off")

    # ── Row 2: spacing quality ───────────────────────────────────────
    if gt_offsets is not None:
        try:
            n = gt_offsets.shape[-1]
            yy, xx = np.mgrid[0:n, 0:n]
            dx, dy = gt_offsets[0], gt_offsets[1]
            mag = np.sqrt(dx * dx + dy * dy)
            ax = axes[2, 0]
            q = ax.quiver(
                xx, yy, dx, dy, mag,
                angles="xy", scale_units="xy", scale=1.0,
                cmap="viridis", width=0.004,
            )
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

    for j, (spa, pts, label) in enumerate(
        zip(all_spa, all_points, col_labels)
    ):
        ax = axes[2, 1 + j]
        nn = spa["nn_distances"]
        sc = ax.scatter(
            pts[:, 0], 1 - pts[:, 1],
            c=nn, cmap="RdYlBu", s=point_size * 3, alpha=0.8, vmin=vmin, vmax=vmax,
        )
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
