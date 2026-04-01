import numpy as np
import torch
import torch.nn.functional as F


def compute_signed_distance_field(image_2d, truncate_px=0.0):
    """Return a normalized signed distance field for a grayscale image.

    Dark pixels are treated as ink. Negative values lie inside ink regions,
    positive values lie in the background, and the result is normalized to
    [-1, 1] per image for stable conditioning.
    """
    from scipy import ndimage

    image = np.asarray(image_2d, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}")

    ink_mask = image < 0.5
    dist_to_ink = ndimage.distance_transform_edt(~ink_mask)
    dist_to_bg = ndimage.distance_transform_edt(ink_mask)
    sdf = dist_to_ink - dist_to_bg

    if truncate_px is not None and truncate_px > 0.0:
        sdf = np.clip(sdf, -truncate_px, truncate_px)

    max_abs = float(np.max(np.abs(sdf)))
    if max_abs > 0.0:
        sdf = sdf / max_abs

    return sdf.astype(np.float32, copy=False)


def build_condition_tensors_from_image(image_2d, grid_size, device=None, sdf_truncate_px=0.0):
    """Build high-res image, low-res density, high-res SDF, and low-res SDF tensors."""
    image = np.asarray(image_2d, dtype=np.float32)
    if image.max() > 1.0 or image.min() < 0.0:
        image = image / 255.0

    high_res = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    high_res_sdf = torch.from_numpy(
        compute_signed_distance_field(image, truncate_px=sdf_truncate_px)
    ).unsqueeze(0).unsqueeze(0)

    target_density = F.interpolate(high_res, size=(grid_size, grid_size), mode="area")
    target_sdf = F.interpolate(high_res_sdf, size=(grid_size, grid_size), mode="bilinear", align_corners=False)

    if device is not None:
        high_res = high_res.to(device)
        target_density = target_density.to(device)
        high_res_sdf = high_res_sdf.to(device)
        target_sdf = target_sdf.to(device)

    return high_res, target_density, high_res_sdf, target_sdf