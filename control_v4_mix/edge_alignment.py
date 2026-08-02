"""Edge / structure-alignment descriptor for the WVS <-> GBN style axis (Gate 0).

This is the BLOCKING kill-shot measure from style_axis_wvs_gbn_plan.md and, if Gate 0
passes, the evaluation metric for the whole downstream experiment. So it is written as a
clean, reusable library with NO torch dependency (numpy / scipy / PIL only) -- it runs cheaply
on CPU and can be imported by later eval code.

What it measures
----------------
Point position *relative to image structure*, normalized against the tone density rho so it
captures **arrangement**, not tone. This is deliberately NOT a spectral / pair-correlation
measure: those are rho-blind and isotropically averaged, would find WVS and GBN nearly
identical, and would wrongly kill a visible axis.

Conventions (established from the control_v4 codebase)
-----------------------------------------------------
* Source grayscale is float32 in [0, 1]; dark = ink (image < 0.5), see control_v4/conditioning.py.
* Point density / "ink probability" rho = clip(1 - image, 0, 1) -- exactly the field the
  smart-init sampler draws from (control_v4/smart_init.py:18).
* Points are (N, 2) in [0, 1] ordered (x, y) where x = col / W, y = row / H
  -- matches extract_points_from_target (control_v4/train_control.py:297-319).
* Gridded-point convention: coord = grid_center + offset / G (train_control.py:621-626).
"""

import numpy as np
from PIL import Image
from scipy import ndimage

EPS = 1e-8


# -----------------------------------------------------------------------------
# I/O + point extraction
# -----------------------------------------------------------------------------

def load_gray01(path):
    """Load an image as float32 grayscale in [0, 1] (dark = ink)."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def extract_centroids(target_png, min_area=1, max_points=None, seed=42):
    """Detect stipple-dot centroids in a rendered target PNG.

    Mirrors control_v4/train_control.py:extract_points_from_target (invert, threshold > 127,
    scipy.ndimage.label + center_of_mass) but returns ALL real centroids -- it does NOT
    random-pad up to a fixed N and does NOT truncate, because padding with uniform-random
    points would corrupt an arrangement measure. Optional min_area drops single-pixel noise
    blobs; max_points (if given) subsamples deterministically for speed only.

    Returns
    -------
    np.ndarray (N, 2) float64 in [0, 1], ordered (x, y).
    """
    img_np = np.asarray(Image.open(target_png).convert("L"), dtype=np.uint8)
    binary = ((255 - img_np) > 127).astype(np.uint8)

    labelled, n_labels = ndimage.label(binary)
    if n_labels == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if min_area > 1:
        areas = ndimage.sum(np.ones_like(binary), labelled, index=np.arange(1, n_labels + 1))
        keep_labels = np.nonzero(areas >= min_area)[0] + 1
    else:
        keep_labels = np.arange(1, n_labels + 1)

    if keep_labels.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    centroids = ndimage.center_of_mass(binary, labelled, list(keep_labels))
    h, w = img_np.shape
    points = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)

    if max_points is not None and len(points) > max_points:
        rng = np.random.RandomState(seed)
        points = points[rng.choice(len(points), max_points, replace=False)]

    return points


def points_from_offsets(offsets, grid_size=None):
    """Decode OT offset grid (2, G, G) to points (N, 2) in [0, 1], (x, y) order.

    Uses the same coord = grid_center + offset / G convention as
    control_v4/train_control.py:offsets_to_coords_gpu and to_pointset_optimal_transport.
    Grid centers are (i + 0.5) / G.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 3 or offsets.shape[0] != 2:
        raise ValueError(f"Expected offsets of shape (2, G, G), got {offsets.shape}")
    g = grid_size or offsets.shape[-1]
    centers = (np.arange(g) + 0.5) / g
    cx, cy = np.meshgrid(centers, centers, indexing="xy")  # (G, G)
    px = cx + offsets[0] / g
    py = cy + offsets[1] / g
    pts = np.stack([px.reshape(-1), py.reshape(-1)], axis=1)
    return np.clip(pts, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Fields: edges, density, sampling
# -----------------------------------------------------------------------------

def density_field(gray01):
    """Tone density rho = clip(1 - image, 0, 1) -- the project canonical ink probability."""
    return np.clip(1.0 - np.asarray(gray01, dtype=np.float32), 0.0, 1.0)


def build_edge_map(gray01, method="sobel"):
    """Return a per-image normalized edge/gradient-magnitude map in [0, 1].

    method="sobel" uses scipy.ndimage.sobel (always available). method="canny" uses OpenCV if
    installed and returns a binary {0,1} edge map; it falls back to sobel otherwise.
    """
    g = np.asarray(gray01, dtype=np.float32)
    if method == "canny":
        try:
            import cv2
            edges = cv2.Canny((g * 255.0).astype(np.uint8), 100, 200)
            return (edges > 0).astype(np.float32)
        except Exception:
            method = "sobel"
    gx = ndimage.sobel(g, axis=1, mode="reflect")
    gy = ndimage.sobel(g, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    m = float(mag.max())
    return (mag / m) if m > EPS else np.zeros_like(mag)


def _pixel_index(pts_xy, h, w):
    """Nearest integer pixel (row, col) for points in [0,1], (x, y) order."""
    xi = np.clip(np.floor(pts_xy[:, 0] * w).astype(np.int64), 0, w - 1)
    yi = np.clip(np.floor(pts_xy[:, 1] * h).astype(np.int64), 0, h - 1)
    return yi, xi


def sample_bilinear(field, pts_xy):
    """Bilinearly sample a scalar field (H, W) at points (N, 2) in [0,1], (x, y) order."""
    field = np.asarray(field, dtype=np.float64)
    h, w = field.shape
    fx = np.clip(pts_xy[:, 0] * w - 0.5, 0, w - 1)
    fy = np.clip(pts_xy[:, 1] * h - 0.5, 0, h - 1)
    x0 = np.floor(fx).astype(np.int64); x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(fy).astype(np.int64); y1 = np.minimum(y0 + 1, h - 1)
    wx = fx - x0; wy = fy - y0
    v00 = field[y0, x0]; v01 = field[y0, x1]
    v10 = field[y1, x0]; v11 = field[y1, x1]
    top = v00 * (1 - wx) + v01 * wx
    bot = v10 * (1 - wx) + v11 * wx
    return top * (1 - wy) + bot * wy


def edge_band_and_distance(gray01, method="sobel", edge_quantile=0.80, ink_thresh=0.05):
    """Build an edge-band mask and a distance-to-edge map (in pixels).

    The band threshold is a per-image quantile of the edge magnitude taken over the ink support
    only (so a mostly-white icon does not have its threshold set by background zeros). For a
    binary canny map every edge pixel is the band. Returns
    (band_mask, dist_px, band_coverage, ink_mask).
    """
    E = build_edge_map(gray01, method=method)
    rho = density_field(gray01)
    ink_mask = rho > ink_thresh

    if method == "canny":
        band = E > 0.5
    else:
        support = E[ink_mask] if ink_mask.any() else E.reshape(-1)
        tau = float(np.quantile(support, edge_quantile)) if support.size else 1.0
        band = E >= max(tau, EPS)
    band_cov = float(band.mean())

    if band.any():
        dist_px = ndimage.distance_transform_edt(~band).astype(np.float32)
    else:
        shp = np.asarray(gray01).shape
        dist_px = np.full(shp, float(np.hypot(shp[0], shp[1])), dtype=np.float32)
    return band, dist_px, band_cov, ink_mask


# -----------------------------------------------------------------------------
# Descriptors
# -----------------------------------------------------------------------------

def precompute_fields(gray01, method="sobel", edge_quantile=0.80, ink_thresh=0.05):
    """Precompute every image-derived field the descriptors need, ONCE.

    The distance transform is by far the most expensive part of the bundle, and WVS and GBN
    share the same source image -- so the caller should build this once per icon and pass it to
    both oracles' bundles via the ``fields=`` argument. Without this the runner recomputes the
    same EDT four times per icon.
    """
    gray01 = np.asarray(gray01, dtype=np.float32)
    band, dist_px, band_cov, ink_mask = edge_band_and_distance(
        gray01, method, edge_quantile, ink_thresh)
    rho = density_field(gray01)
    rho_total = float(rho.sum())
    expected_frac = float(rho[band].sum() / (rho_total + EPS)) if rho_total > EPS else 0.0
    return {
        "shape": gray01.shape,
        "band": band,
        "dist_px": dist_px,
        "band_coverage": band_cov,
        "ink_mask": ink_mask,
        "expected_frac": expected_frac,
        "grad_mag": build_edge_map(gray01, method="sobel"),
    }


def rho_normalized_alignment(gray01, pts_xy, method="sobel", edge_quantile=0.80, fields=None):
    """Primary descriptor: rho-normalized edge over-representation.

    expected_frac = fraction of tone-density (rho) mass inside the edge band -- what tone alone
    predicts. observed_frac = fraction of oracle points landing in the edge band.
    s_raw = observed_frac / expected_frac. GBN (hugs contours) >> 1; WVS (fills interiors) ~= 1.
    This is orthogonal to rho, i.e. it measures arrangement, not "edges are darker".
    """
    gray01 = np.asarray(gray01, dtype=np.float32)
    h, w = gray01.shape
    if fields is None:
        fields = precompute_fields(gray01, method=method, edge_quantile=edge_quantile)
    band = fields["band"]
    band_cov = fields["band_coverage"]
    expected_frac = fields["expected_frac"]

    if len(pts_xy) == 0:
        return dict(s_raw=np.nan, observed_frac=np.nan, expected_frac=expected_frac,
                    band_coverage=band_cov, n_points=0)

    yi, xi = _pixel_index(np.asarray(pts_xy), h, w)
    observed_frac = float(band[yi, xi].mean())
    s_raw = observed_frac / (expected_frac + EPS)
    return dict(s_raw=float(s_raw), observed_frac=observed_frac, expected_frac=expected_frac,
                band_coverage=band_cov, n_points=int(len(pts_xy)))


def compute_descriptor_bundle(gray01, pts_xy, method="sobel", edge_quantile=0.80,
                              k_pixels=(1, 2, 4, 8), fields=None):
    """Compute the full descriptor bundle for one (source, point-set) pair.

    Returns a flat dict with the primary rho-normalized descriptor plus corroborating measures:
      * s_raw               -- primary (rho-normalized edge over-representation)
      * observed_frac / expected_frac
      * dist_mean / dist_median -- distance (px) from points to nearest edge
      * frac_within_k       -- fraction of points within k px of an edge, per k
      * grad_at_points      -- mean normalized gradient magnitude sampled at points
      * n_points
    """
    gray01 = np.asarray(gray01, dtype=np.float32)
    h, w = gray01.shape
    pts_xy = np.asarray(pts_xy, dtype=np.float64)

    if fields is None:
        fields = precompute_fields(gray01, method=method, edge_quantile=edge_quantile)
    align = rho_normalized_alignment(gray01, pts_xy, method=method,
                                     edge_quantile=edge_quantile, fields=fields)
    out = {
        "n_points": align["n_points"],
        "s_raw": align["s_raw"],
        "observed_frac": align["observed_frac"],
        "expected_frac": align["expected_frac"],
    }
    if len(pts_xy) == 0:
        out.update({"dist_mean": np.nan, "dist_median": np.nan, "grad_at_points": np.nan})
        for k in k_pixels:
            out[f"frac_within_{k}px"] = np.nan
        return out

    dist_px = fields["dist_px"]
    yi, xi = _pixel_index(pts_xy, h, w)
    d = dist_px[yi, xi].astype(np.float64)
    out["dist_mean"] = float(np.mean(d))
    out["dist_median"] = float(np.median(d))
    for k in k_pixels:
        out[f"frac_within_{k}px"] = float(np.mean(d <= k))

    E = fields["grad_mag"]  # continuous magnitude for grad-at-points
    out["grad_at_points"] = float(np.mean(sample_bilinear(E, pts_xy)))
    return out


# Descriptors where a LARGER value means "more edge-aligned" (GBN-like). Used by the runner to
# orient separation stats consistently (GBN should exceed WVS on every one of these).
GBN_HIGHER_KEYS = (
    "s_raw", "observed_frac", "grad_at_points",
    "frac_within_1px", "frac_within_2px", "frac_within_4px", "frac_within_8px",
)
# Descriptors where a SMALLER value means "more edge-aligned" (GBN-like).
GBN_LOWER_KEYS = ("dist_mean", "dist_median")
