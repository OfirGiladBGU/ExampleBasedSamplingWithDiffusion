"""Generate stipple point sets using Dynamic ControlNet V4 (Truncated Control)."""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from control_v4.conditioning import build_condition_tensors_from_image
from control_v4.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_v4.smart_init import build_smart_init_from_image, save_smart_init_debug, add_noise_at_t
from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from utils.stippling_metrics import (
    compute_grid_capacity,
    compute_spacing_quality,
    geometric_validation_score,
    resolve_capacity_grid_size,
)
from utils.stippling_metrics_advance import (
    compute_all_advanced_metrics,
    visualize_adaptive_sampling_density_map,
    visualize_spatial_metrics_panel,
    visualize_overfit_metrics as visualize_overfit_metrics_advance,
    _render_advanced_metrics_row,
    _ADV_ROW_FONTSIZE,
    _ADV_ROW_TITLE_FONTSIZE,
    _ADV_ROW_HEIGHT_RATIO,
    _ADV_ROW_EXTRA_HEIGHT,
)


# Editable defaults 
CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT = "config/GBN/model.ckpt"
CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep1100.pt"
# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/gradient0deg.png"
# GT_IMAGE_PATH = ""
INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/sample_with_GT/source/emoji-one_4_monkey.png"
GT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/sample_outputs_data/sample_with_GT/target/emoji-one_4_monkey.png"
OUTPUT_DIR = "control_v4/sample_outputs_advance"
N_SAMPLES = 1
TIMESTEPS = 1000
GRID_SIZE = 32
RESAMPLE_JUMPS = 2
ENABLE_GECCO = True
DEVICE = "cuda"
SDF_TRUNCATE_PX = 8.0
USE_SDF = True
SHOW_DENOISING = False
SHOW_DENOISING_INTERVAL = 50
TRUNCATION_RATIO = 0.30
T_START_STEP = -1
SMART_INIT_SEED = 42
SMART_INIT_SPLAT_SIGMA_PX = 0.5
CAPACITY_GRID_SIZE = 32
# CAPACITY_GRID_SIZE = -1  # -1 for full input resolution
METRICS_ADVANCE = True
ADAPTIVE_SAMPLING_DENSITY_MAP = True
CLIP_TO_DOMAIN = True  # Whether to clip predicted points to [0,1]² before metrics and visualisation (recommended for truncated control)


def extract_points_from_target(img_path, n_points):
    """Detect dot centroids in a stippled target and return (N, 2) in [0, 1]."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    inv = 255 - img_np
    binary = (inv > 127).astype(np.uint8)

    from scipy import ndimage
    labelled, n_labels = ndimage.label(binary)
    centroids = ndimage.center_of_mass(binary, labelled, range(1, n_labels + 1))

    h, w = img_np.shape
    points = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)

    rng = np.random.RandomState(42)
    if len(points) > n_points:
        points = points[rng.choice(len(points), n_points, replace=False)]
    elif len(points) < n_points:
        deficit = n_points - len(points)
        points = np.vstack([points, rng.rand(deficit, 2)])

    return points


def visualize_sample_metrics_no_gt(source_img_u8, pred_pointsets, save_path, point_size=0.5, capacity_grid_size=16, compute_advanced=False):
    """Create overfit-style metrics panel without GT column."""
    if not HAS_MPL:
        return None
    if len(pred_pointsets) == 0:
        return None

    from utils.stippling_metrics_advance import compute_all_advanced_metrics, _format_advanced_text

    n_preds = min(len(pred_pointsets), 4)
    n_cols = 1 + n_preds  # INPUT + predictions

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

    image_01 = source_img_u8.astype(np.float64) / 255.0

    ax = axes[0, 0]
    ax.imshow(source_img_u8, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Condition (Input)")
    ax.axis("off")

    for i in range(n_preds):
        ax = axes[0, 1 + i]
        pts = pred_pointsets[i]
        ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=point_size, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(f"Predict {i}")
        ax.axis("off")

    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    cap_grid_shape = resolve_capacity_grid_size(image_01, capacity_grid_size)
    pred_caps = [compute_grid_capacity(pred_pointsets[i], image_01, grid_size=cap_grid_shape) for i in range(n_preds)]
    pred_spa = [compute_spacing_quality(pred_pointsets[i]) for i in range(n_preds)]

    for i in range(n_preds):
        cap = pred_caps[i]
        ax = axes[1, 1 + i]
        status = cap["grid_status"]
        h_grid, w_grid = status.shape
        rgb = np.zeros((h_grid, w_grid, 3), dtype=np.float32)
        rgb[status == 0, 1] = 1.0
        rgb[status == -1, 0] = 1.0
        rgb[status == 1, 2] = 1.0
        ax.imshow(rgb, origin="upper", aspect="equal")
        ok_pct = 100.0 - cap["underfilled_pct"] - cap["overfilled_pct"]
        ax.set_title(
            f"Predict {i} Capacity\n"
            f"Grid:{cap_grid_shape[0]}x{cap_grid_shape[1]} | "
            f"OK:{ok_pct:.0f}% Under:{cap['underfilled_pct']:.0f}% Over:{cap['overfilled_pct']:.0f}%\n"
            f"Score: {cap['score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    all_nn = [s["nn_distances"] for s in pred_spa]
    vmin = min(d.min() for d in all_nn)
    vmax = max(d.max() for d in all_nn)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin, vmax = 0.0, 1.0

    for i in range(n_preds):
        spa = pred_spa[i]
        pts = pred_pointsets[i]
        ax = axes[2, 1 + i]
        sc = ax.scatter(
            pts[:, 0],
            1 - pts[:, 1],
            c=spa["nn_distances"],
            cmap="RdYlBu",
            s=point_size * 3,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.set_title(
            f"Predict {i} Spacing\n"
            f"CV:{spa['nn_cv']:.3f}  Clumped:{spa['clumped_pct']:.1f}%\n"
            f"Score: {spa['spacing_score']:.3f}",
            fontsize=9,
        )
        ax.axis("off")
        plt.colorbar(sc, ax=ax, shrink=0.7, label="NN dist")

    # ── Row 3: advanced M1–M6 numeric text (optional) ────────────────
    if compute_advanced:
        pred_labels = [f"Predict {i}" for i in range(n_preds)]
        _render_advanced_metrics_row(axes[3, :], pred_pointsets[:n_preds], pred_labels, image_01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_sample_image(image_path, pts, out_png_path):
    if not HAS_MPL:
        return

    cond_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cond_img is None:
        return
    cond_img = cond_img.astype(float) / 255.0

    fig, (ax_cond, ax_stipple) = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    ax_cond.imshow(cond_img, cmap="gray", vmin=0.0, vmax=1.0)
    ax_cond.axis("off")
    ax_cond.set_title("Condition")

    ax_stipple.scatter(pts[:, 0], 1.0 - pts[:, 1], c="black", s=0.8, alpha=0.8)
    ax_stipple.set_xlim(0, 1)
    ax_stipple.set_ylim(0, 1)
    ax_stipple.set_aspect("equal")
    ax_stipple.axis("off")
    ax_stipple.set_title("Predicted stipple")

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_denoise_step(img_tensor, timestep_i, t_start, out_path):
    if not HAS_MPL:
        return
    offsets = img_tensor[0].detach().cpu().float().numpy()
    h, w = offsets.shape[1], offsets.shape[2]
    cx = (np.arange(w) + 0.5) / w
    cy = (np.arange(h) + 0.5) / h
    gx, gy = np.meshgrid(cx, cy)
    px = np.clip(gx + offsets[0] / w, 0.0, 1.0).flatten()
    py = np.clip(gy + offsets[1] / h, 0.0, 1.0).flatten()

    elapsed = t_start - 1 - timestep_i
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=110)
    ax.scatter(px, 1.0 - py, c="black", s=0.5, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"step {elapsed}/{max(t_start - 1, 1)} (t={timestep_i})", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def _save_condition_debug_tensors(
    high_res,
    high_res_sdf,
    target_density,
    target_sdf,
    smart_init_grid_raw,
    smart_init_grid_model,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)
    cond_map = {
        "high_res": high_res,
        "high_res_sdf": high_res_sdf,
        "target_density": target_density,
        "target_sdf": target_sdf,
        "smart_init_grid_raw": smart_init_grid_raw,
        "smart_init_grid_model_input": smart_init_grid_model,
    }
    for name, tensor in cond_map.items():
        arr = tensor.detach().cpu().float().numpy().squeeze()
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    if HAS_MPL:
        ordered_names = [
            "high_res",
            "high_res_sdf",
            "target_density",
            "target_sdf",
            "smart_init_grid_raw",
            "smart_init_grid_model_input",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=140)
        for ax, name in zip(axes.flat, ordered_names):
            if name == "_":
                ax.axis("off")
                continue
            arr = cond_map[name].detach().cpu().float().numpy().squeeze()
            if arr.ndim != 2:
                ax.axis("off")
                ax.set_title(f"{name} (invalid)")
                continue
            if "sdf" in name:
                vis = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
            else:
                vis = np.clip(arr, 0.0, 1.0)
            ax.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            ax.set_title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "conditions_collage.png"), dpi=140, bbox_inches="tight")
        plt.close()


def _grid_centers_flat(grid_size, device, dtype):
    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    return torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)


def _offsets_to_coords_gpu(offsets, grid_size, grid_centers_flat):
    bsz = offsets.shape[0]
    offs = offsets.permute(0, 2, 3, 1).reshape(bsz, grid_size * grid_size, 2)
    coords = grid_centers_flat.expand(bsz, -1, -1) + offs / float(grid_size)
    return coords.clamp(0.0, 1.0)


def _render_smart_init_gpu(coords, grid_size, sigma_px, device):
    """Gaussian soft splatting of (1, N, 2) coords to (1, 1, G, G) -- matches training."""
    lin = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    pixel_centers = torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)
    sigma = max(float(sigma_px), 1e-4) / float(grid_size)
    dist = torch.cdist(pixel_centers, coords, p=2)
    gauss = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
    return gauss.amax(dim=2).reshape(1, 1, grid_size, grid_size).clamp(0.0, 1.0)


def load_condition(image_path, grid_size, device, sdf_truncate_px=0.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_01 = img.astype(np.float32) / 255.0
    high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_image(
        image_01,
        grid_size,
        device,
        sdf_truncate_px=sdf_truncate_px,
    )
    return image_01, high_res, target_density, high_res_sdf, target_sdf


def _extract_control_state_dict(ctrl_state):
    if not isinstance(ctrl_state, dict):
        raise TypeError(f"Control checkpoint must be a dict, got {type(ctrl_state)}")

    for key in ("control_net", "model_state_dict", "state_dict"):
        value = ctrl_state.get(key)
        if isinstance(value, dict):
            return value

    if ctrl_state and all(hasattr(v, "shape") for v in ctrl_state.values()):
        return ctrl_state

    keys_preview = ", ".join(list(ctrl_state.keys())[:8])
    raise KeyError(
        "Could not find control weights in checkpoint. "
        f"Expected one of ['control_net', 'model_state_dict', 'state_dict']; found keys: [{keys_preview}]"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--base_ckpt", default=BASE_CKPT)
    parser.add_argument("--control_ckpt", default=CONTROL_CKPT)
    parser.add_argument("--input-image", "--image", dest="input_image", default=INPUT_IMAGE_PATH)
    parser.add_argument(
        "--gt-image",
        default=GT_IMAGE_PATH,
        help="Optional GT stipple image path. If empty, GT column is omitted in the metrics panel.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--no_ot", action="store_true")
    parser.add_argument("--enable-gecco", default=ENABLE_GECCO, action=argparse.BooleanOptionalAction)
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)
    parser.add_argument("--use-sdf", action=argparse.BooleanOptionalAction, default=USE_SDF)
    parser.add_argument("--show-denoising", action=argparse.BooleanOptionalAction, default=SHOW_DENOISING)
    parser.add_argument(
        "--show-denoising-interval",
        "--denoise-interval",
        dest="show_denoising_interval",
        type=int,
        default=SHOW_DENOISING_INTERVAL,
    )
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO)
    parser.add_argument("--t-start-step", type=int, default=T_START_STEP,
                        help="If >=0, overrides truncation-ratio derived start step")
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--smart-init-splat-sigma-px", type=float, default=SMART_INIT_SPLAT_SIGMA_PX,
                        help="Gaussian sigma in grid-pixel units for Smart Init soft splatting (match training default)")
    parser.add_argument(
        "--capacity-grid-size",
        type=int,
        default=CAPACITY_GRID_SIZE,
        help="Capacity grid size: >0 uses KxK, -1 uses full input image resolution",
    )
    parser.add_argument(
        "--metrics-advance",
        action="store_true",
        default=METRICS_ADVANCE,
        help="Enable advanced M1-M6 metrics (Voronoi, Sinkhorn, Adaptive-NND, CVT, Spectrum, EMD)",
    )
    parser.add_argument(
        "--adaptive-sampling-density-map",
        action=argparse.BooleanOptionalAction,
        default=ADAPTIVE_SAMPLING_DENSITY_MAP,
        help="Enable GBN-style AKDE density map visualisation (saved to adaptive_sampling_density_map/)",
    )
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    if not (0.0 < args.truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0,1]")
    if args.capacity_grid_size == 0 or args.capacity_grid_size < -1:
        raise ValueError("--capacity-grid-size must be > 0, or -1 for full input resolution")

    device = torch.device(args.device)

    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    denoiser = diffusion.model
    denoiser.eval()

    control_net = DynamicControlNet(
        denoiser,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
    ).to(device)
    ctrl_state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.load_state_dict(_extract_control_state_dict(ctrl_state), strict=False)
    control_net.eval()

    img_stem = os.path.splitext(os.path.basename(args.input_image))[0]
    sample_base_dir = os.path.join(args.out_dir, img_stem)
    os.makedirs(sample_base_dir, exist_ok=True)

    image_01, high_res, target_density, high_res_sdf, target_sdf = load_condition(
        args.input_image,
        args.grid_size,
        device,
        sdf_truncate_px=args.sdf_truncate_px,
    )

    smart_points, smart_offsets_np, smart_grid_np = build_smart_init_from_image(
        image_01,
        grid_size=args.grid_size,
        n_points=args.grid_size * args.grid_size,
        seed=args.smart_init_seed,
    )
    smart_init_grid_raw = torch.from_numpy(smart_grid_np).unsqueeze(0).to(device)
    smart_init_offsets = torch.from_numpy(smart_offsets_np).unsqueeze(0).to(device)
    grid_centers_flat = _grid_centers_flat(args.grid_size, device, smart_init_offsets.dtype)
    smart_coords = _offsets_to_coords_gpu(smart_init_offsets, args.grid_size, grid_centers_flat)
    smart_init_grid = _render_smart_init_gpu(
        smart_coords,
        args.grid_size,
        args.smart_init_splat_sigma_px,
        device,
    )

    if not args.use_sdf:
        high_res_sdf = torch.zeros_like(high_res_sdf)
        target_sdf = torch.zeros_like(target_sdf)

    conditions_dir = os.path.join(sample_base_dir, "conditions")
    _save_condition_debug_tensors(
        high_res,
        high_res_sdf,
        target_density,
        target_sdf,
        smart_init_grid_raw,
        smart_init_grid,
        conditions_dir,
    )

    smart_dir = os.path.join(sample_base_dir, "smart_init")
    save_smart_init_debug(
        smart_dir,
        smart_points,
        smart_offsets_np,
        smart_grid_np,
        model_input_grid=smart_init_grid.detach().cpu().numpy(),
    )

    controlled = DynamicControlledDenoiser(denoiser, control_net)
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
    diffusion.model = controlled
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    t_start = args.t_start_step if args.t_start_step >= 0 else int(args.timesteps * args.truncation_ratio)
    t_start = int(np.clip(t_start, 1, max(args.timesteps - 1, 1)))

    n_samples = args.n_samples
    x_init = smart_init_offsets
    if x_init.shape[0] != n_samples:
        x_init = x_init.expand(n_samples, -1, -1, -1).contiguous()

    alpha_t = diffusion.alphas_cumprod[t_start]
    img = add_noise_at_t(x_init, alpha_t)

    print(f"Loaded checkpoint : {args.control_ckpt}")
    print(f"GECCO enabled     : {args.enable_gecco}")
    print(f"SDF enabled       : {args.use_sdf}")
    print(f"Timesteps         : {args.timesteps}")
    print(f"t_start           : {t_start}")
    print(f"Resample jumps    : {args.resample_jumps}")
    if args.capacity_grid_size == -1:
        print("Capacity grid     : full input resolution")
    else:
        print(f"Capacity grid     : {args.capacity_grid_size}x{args.capacity_grid_size}")

    steps_dir = None
    if args.show_denoising:
        steps_dir = os.path.join(sample_base_dir, "denoising_steps")
        os.makedirs(steps_dir, exist_ok=True)

    from tqdm import tqdm
    with torch.no_grad() if args.resample_jumps == 0 else torch.enable_grad():
        for i in tqdm(reversed(range(t_start)), total=t_start, desc="sampling_v4"):
            t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
            for u in range(args.resample_jumps + 1):
                with torch.no_grad():
                    img = diffusion.p_sample(
                        img,
                        cond=None,
                        t=t_tensor,
                        clip_denoised=diffusion.sample_clip,
                        with_sampling=True,
                    )

                if u == args.resample_jumps or i == 0:
                    break
                beta_i = diffusion.betas[i]
                noise = torch.randn_like(img)
                img = (1.0 - beta_i).sqrt() * img + beta_i.sqrt() * noise

            if steps_dir is not None:
                elapsed = t_start - 1 - i
                if elapsed % args.show_denoising_interval == 0:
                    step_path = os.path.join(steps_dir, f"step_{elapsed:04d}.png")
                    _save_denoise_step(img, i, t_start, step_path)

    samples_raw = img.detach().cpu().numpy()

    npy_dir = os.path.join(sample_base_dir, "npy")
    png_dir = os.path.join(sample_base_dir, "png")
    metrics_dir = os.path.join(sample_base_dir, "metrics")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    pred_pointsets = []

    for idx, s in enumerate(samples_raw):
        suffix = f"_{idx + 1}"
        npy_path = os.path.join(npy_dir, f"{img_stem}{suffix}.npy")
        png_path = os.path.join(png_dir, f"{img_stem}{suffix}.png")

        if not args.no_ot:
            pts = to_pointset_optimal_transport(s)
            pts = pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T
            pred_pointsets.append(pts)
            np.save(npy_path, pts)
            save_sample_image(args.input_image, pts, png_path)
        else:
            np.save(npy_path, s)

    if args.no_ot:
        print("Skipped metrics panel: --no_ot was enabled.")
    else:
        input_img_u8 = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
        panel_path = os.path.join(metrics_dir, "results_panel.png")
        panel_saved = None
        gt_points = None
        if input_img_u8 is None:
            print(f"Skipped metrics panel: failed to read input image: {args.input_image}")
        elif len(pred_pointsets) == 0:
            print("Skipped metrics panel: no predicted point sets were generated.")
        elif args.gt_image:
            gt_img_u8 = cv2.imread(args.gt_image, cv2.IMREAD_GRAYSCALE)
            if gt_img_u8 is None:
                print(f"GT image was provided but could not be read, falling back to no-GT panel: {args.gt_image}")
                panel_saved = visualize_sample_metrics_no_gt(
                    input_img_u8,
                    pred_pointsets,
                    panel_path,
                    capacity_grid_size=args.capacity_grid_size,
                    compute_advanced=args.metrics_advance,
                )
            else:
                gt_points = extract_points_from_target(args.gt_image, pred_pointsets[0].shape[0])
                panel_saved = visualize_overfit_metrics_advance(
                    input_img_u8,
                    gt_img_u8,
                    gt_points,
                    pred_pointsets,
                    panel_path,
                    step=None,
                    gt_offsets=None,
                    capacity_grid_size=args.capacity_grid_size,
                    compute_advanced=args.metrics_advance,
                )
        else:
            panel_saved = visualize_sample_metrics_no_gt(
                input_img_u8,
                pred_pointsets,
                panel_path,
                capacity_grid_size=args.capacity_grid_size,
                compute_advanced=args.metrics_advance,
            )

        if panel_saved is not None:
            print(f"Saved metrics panel: {panel_saved}")
        else:
            print("Skipped metrics panel: matplotlib is not available.")

        geom = geometric_validation_score(pred_pointsets)
        print(
            "Geometry summary | "
            f"CV={geom['cv']:.4f} | "
            f"Clumped={geom['clumped_pct']:.2f}% | "
            f"Score={geom['score']:.4f}"
        )

        # Advanced metrics M1–M6 (optional) — text row already embedded in panel above
        if args.metrics_advance and not args.no_ot:
            try:
                metrics_advance_dir = os.path.join(sample_base_dir, "metrics_advance")
                os.makedirs(metrics_advance_dir, exist_ok=True)

                gt_metrics = None
                if gt_points is not None:
                    gt_metrics = compute_all_advanced_metrics(gt_points, image_01)
                    gt_json_path = os.path.join(metrics_advance_dir, "metrics_gt.json")
                    with open(gt_json_path, "w") as f:
                        json.dump(gt_metrics, f, indent=2)
                    print(
                        "GT advanced metrics | "
                        f"M1_CV={gt_metrics.get('M1_voronoi_mass_cv', 0.0):.4f} | "
                        f"M2_OT={gt_metrics.get('M2_sinkhorn_ot_cost', 0.0):.4f} | "
                        f"M3_NND={gt_metrics.get('M3_adaptive_nnd_cv', 0.0):.4f} | "
                        f"M6_EMD={gt_metrics.get('M6_emd_distance', 0.0):.4f}"
                    )

                # Compute and save detailed M1-M6 metrics for each prediction
                for idx, pts in enumerate(pred_pointsets):
                    metrics_dict = compute_all_advanced_metrics(pts, image_01)
                    metrics_json_path = os.path.join(metrics_advance_dir, f"metrics_pred_{idx + 1}.json")

                    with open(metrics_json_path, "w") as f:
                        json.dump(metrics_dict, f, indent=2)

                    if gt_metrics is not None:
                        comparable_keys = sorted(set(gt_metrics.keys()) & set(metrics_dict.keys()))
                        compare = {k: float(metrics_dict[k] - gt_metrics[k]) for k in comparable_keys}
                        compare_json_path = os.path.join(metrics_advance_dir, f"metrics_compare_pred_{idx + 1}_minus_gt.json")
                        with open(compare_json_path, "w") as f:
                            json.dump(compare, f, indent=2)

                    # Print summary
                    print(f"Pred {idx + 1} advanced metrics | "
                          f"M1_CV={metrics_dict.get('M1_voronoi_mass_cv', 0.0):.4f} | "
                          f"M2_OT={metrics_dict.get('M2_sinkhorn_ot_cost', 0.0):.4f} | "
                          f"M3_NND={metrics_dict.get('M3_adaptive_nnd_cv', 0.0):.4f} | "
                          f"M6_EMD={metrics_dict.get('M6_emd_distance', 0.0):.4f}")

                    if gt_metrics is not None:
                        print(
                            f"Pred {idx + 1} - GT deltas | "
                            f"dM1_CV={metrics_dict.get('M1_voronoi_mass_cv', 0.0) - gt_metrics.get('M1_voronoi_mass_cv', 0.0):+.4f} | "
                            f"dM2_OT={metrics_dict.get('M2_sinkhorn_ot_cost', 0.0) - gt_metrics.get('M2_sinkhorn_ot_cost', 0.0):+.4f} | "
                            f"dM3_NND={metrics_dict.get('M3_adaptive_nnd_cv', 0.0) - gt_metrics.get('M3_adaptive_nnd_cv', 0.0):+.4f} | "
                            f"dM6_EMD={metrics_dict.get('M6_emd_distance', 0.0) - gt_metrics.get('M6_emd_distance', 0.0):+.4f}"
                        )
            except Exception as e:
                print(f"Warning: advanced metrics computation failed: {e}")

        # ── AKDE density map (GBN-style) ─────────────────────────────────
        if args.adaptive_sampling_density_map and input_img_u8 is not None and len(pred_pointsets) > 0:
            try:
                akde_dir = os.path.join(sample_base_dir, "adaptive_sampling_density_map")
                os.makedirs(akde_dir, exist_ok=True)
                akde_path = os.path.join(akde_dir, "density_map.png")
                akde_saved = visualize_adaptive_sampling_density_map(
                    input_img_u8,
                    pred_pointsets,
                    akde_path,
                    gt_points=gt_points,
                    device=str(device),
                )
                if akde_saved:
                    print(f"Saved AKDE density map : {akde_saved}")
            except Exception as e:
                print(f"Warning: AKDE density map visualisation failed: {e}")

        # ── Spatial visual metrics panel (M1/M3/M4/M5) ───────────────────
        if args.metrics_advance and input_img_u8 is not None and len(pred_pointsets) > 0:
            try:
                spatial_dir = os.path.join(sample_base_dir, "spatial_metrics")
                os.makedirs(spatial_dir, exist_ok=True)
                spatial_path = os.path.join(spatial_dir, "spatial_metrics_panel.png")
                image_01 = input_img_u8.astype(np.float64) / 255.0
                spatial_saved = visualize_spatial_metrics_panel(
                    image_01,
                    pred_pointsets,
                    spatial_path,
                    gt_points=gt_points,
                    clip_to_domain=CLIP_TO_DOMAIN,
                )
                if spatial_saved:
                    print(f"Saved spatial metrics panel: {spatial_saved}")
            except Exception as e:
                print(f"Warning: spatial metrics panel failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
