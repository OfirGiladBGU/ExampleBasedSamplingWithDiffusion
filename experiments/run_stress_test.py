import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from control_v4.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_v4.conditioning import build_condition_tensors_from_image
from control_v4.smart_init import add_noise_at_t, build_smart_init_from_image
from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from utils.stippling_metrics import compute_spacing_quality, visualize_overfit_metrics


# Stress 1:
DATA_ROOT_DIR = r"/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1"
OUTPUT_ROOT_DIR = os.path.join("experiments", "outputs_stress1")
# Baseline configuration
BASELINE_CONFIG_PATH = "config_trained/GBN_stress1/config.json"
BASELINE_CKPT_PATH = "config_trained/GBN_stress1/model.ckpt"
# Control V4 configuration
CONTROL_BASE_CONFIG_PATH = "config/GBN/config.json"
CONTROL_BASE_CKPT_PATH = "config/GBN/model.ckpt"
CONTROLNET_CKPT_PATH = "control_v4/train_outputs_data_stress1_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"


# Stress 2:
# DATA_ROOT_DIR = r"/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2"
# OUTPUT_ROOT_DIR = os.path.join("experiments", "outputs_stress2")
# # Baseline configuration
# BASELINE_CONFIG_PATH = "config_trained/GBN_stress2/config.json"
# BASELINE_CKPT_PATH = "config_trained/GBN_stress2/model.ckpt"
# # Control V4 configuration
# CONTROL_BASE_CONFIG_PATH = "config/GBN/config.json"
# CONTROL_BASE_CKPT_PATH = "config/GBN/model.ckpt"
# CONTROLNET_CKPT_PATH = "control_v4/train_outputs_data_stress2/checkpoints/dynamic_controlnet_v4_ep1500.pt"


# Stress V2:
# DATA_ROOT_DIR = r"/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2"
# OUTPUT_ROOT_DIR = os.path.join("experiments", "outputs_stress2_V2")
# # Baseline configuration
# BASELINE_CONFIG_PATH = "config_trained/GBN_stress2_V2/config.json"
# BASELINE_CKPT_PATH = "config_trained/GBN_stress2_V2/model.ckpt"
# # Control V4 configuration
# CONTROL_BASE_CONFIG_PATH = "config/GBN/config.json"
# CONTROL_BASE_CKPT_PATH = "config/GBN/model.ckpt"
# CONTROLNET_CKPT_PATH = "control_v4/train_outputs_data_stress2_V2_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"


# Common settings
DEVICE = "cuda"
N_EXAMPLES = 4
GRID_SIZE = 32
TIMESTEPS = 1000

# Baseline
BASELINE_TRUNCATION_RATIO = 1.0

# Control V4
TRUNCATION_RATIO = 0.30
RESAMPLE_JUMPS = 0
ENABLE_GECCO = True
SDF_FEATURES = True
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SEED = 42
SMART_INIT_FEATURES = True
BATCH_COORDS_FEATURES = True
ENABLE_SMART_INIT_SPLAT_SIGMA = False
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CALCULATE_METRICS = True
CAPACITY_GRID_SIZE = 32


def _is_image_file(name):
    return os.path.splitext(name)[1].lower() in VALID_EXTS


def _pick_condition_image(original_dir):
    if not os.path.isdir(original_dir):
        raise FileNotFoundError(f"Missing original directory: {original_dir}")
    candidates = sorted([f for f in os.listdir(original_dir) if _is_image_file(f)])
    if not candidates:
        raise FileNotFoundError(f"No images found in original directory: {original_dir}")
    return os.path.join(original_dir, candidates[0])


def _pick_matched_source_target_examples(source_dir, original_dir, target_dir, n_examples):
        """Pick (source_path, target_path) pairs.

        Priority:
            1) If source/target share filenames, use matched source per target.
            2) Otherwise fallback to a single image from original/ repeated for all targets.
        """
        target_paths = _pick_target_images(target_dir, n_examples)

        if os.path.isdir(source_dir):
                source_candidates = sorted([f for f in os.listdir(source_dir) if _is_image_file(f)])
                source_set = set(source_candidates)
                matched = [p for p in target_paths if os.path.basename(p) in source_set]
                if len(matched) == len(target_paths):
                        source_paths = [os.path.join(source_dir, os.path.basename(p)) for p in target_paths]
                        return source_paths, target_paths, "source"

        # Fallback path for older layouts with a single original image
        fallback = _pick_condition_image(original_dir)
        source_paths = [fallback for _ in target_paths]
        return source_paths, target_paths, "original"


def _pick_target_images(target_dir, n_examples):
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Missing target directory: {target_dir}")
    candidates = sorted([f for f in os.listdir(target_dir) if _is_image_file(f)])
    if len(candidates) < n_examples:
        raise ValueError(
            f"Requested {n_examples} examples but only found {len(candidates)} target images in {target_dir}"
        )
    return [os.path.join(target_dir, f) for f in candidates[:n_examples]]



def _extract_control_state_dict(ctrl_state):
    if not isinstance(ctrl_state, dict):
        raise TypeError(f"Control checkpoint must be a dict, got {type(ctrl_state)}")

    for key in ("control_net", "model_state_dict", "state_dict"):
        maybe = ctrl_state.get(key)
        if isinstance(maybe, dict) and maybe and all(hasattr(v, "shape") for v in maybe.values()):
            return maybe

    if ctrl_state and all(hasattr(v, "shape") for v in ctrl_state.values()):
        return ctrl_state

    keys_preview = ", ".join(list(ctrl_state.keys())[:8])
    raise KeyError(
        "Could not find control weights in checkpoint. "
        f"Expected one of ['control_net', 'model_state_dict', 'state_dict']; found keys: [{keys_preview}]"
    )


def _resolve_ckpt_path(path):
    if os.path.exists(path):
        return path

    root, ext = os.path.splitext(path)
    if ext == ".ckpt":
        alt = root + ".ckp"
        if os.path.exists(alt):
            return alt
    elif ext == ".ckp":
        alt = root + ".ckpt"
        if os.path.exists(alt):
            return alt

    raise FileNotFoundError(
        "Checkpoint not found. Tried: "
        f"{path}"
        + (f" and {root + '.ckp'}" if ext == ".ckpt" else "")
        + (f" and {root + '.ckpt'}" if ext == ".ckp" else "")
    )


def _grid_centers_flat(grid_size, device, dtype):
    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    return torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)


def offsets_to_coords_gpu(offsets, grid_size, grid_centers_flat):
    bsz = offsets.shape[0]
    offs = offsets.permute(0, 2, 3, 1).reshape(bsz, grid_size * grid_size, 2)
    coords = grid_centers_flat.expand(bsz, -1, -1) + offs / float(grid_size)
    return coords.clamp(0.0, 1.0)


def render_smart_init_gpu(coords, grid_size=32, sigma_px=0.5):
    bsz = coords.shape[0]
    dtype = coords.dtype
    device = coords.device

    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    pixel_centers = torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2).expand(bsz, -1, -1)

    sigma = max(float(sigma_px), 1e-4) / float(grid_size)
    diff = pixel_centers.unsqueeze(2) - coords.unsqueeze(1)
    dist2 = (diff * diff).sum(dim=-1)
    gauss = torch.exp(-dist2 / (2.0 * sigma * sigma))
    grid = gauss.max(dim=2).values.reshape(bsz, 1, grid_size, grid_size)
    return grid.clamp(0.0, 1.0)


def render_occupancy_grid_gpu(coords, grid_size=32):
    """Render (B,N,2) coords to (B,1,G,G) normalized occupancy map (no Gaussian)."""
    bsz, n, _ = coords.shape
    device = coords.device
    dtype = coords.dtype
    px = (coords[:, :, 0] * grid_size).clamp(0, grid_size - 1).long()
    py = (coords[:, :, 1] * grid_size).clamp(0, grid_size - 1).long()
    flat_idx = py * grid_size + px
    grid_flat = torch.zeros(bsz, grid_size * grid_size, device=device, dtype=dtype)
    ones = torch.ones(bsz, n, device=device, dtype=dtype)
    grid_flat.scatter_add_(1, flat_idx, ones)
    mx = grid_flat.amax(dim=1, keepdim=True).clamp(min=1.0)
    return (grid_flat / mx).reshape(bsz, 1, grid_size, grid_size)


def load_condition(image_path, grid_size, device, sdf_features=True, sdf_truncate_px=0.0):
    image_01 = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    if not sdf_features:
        high_res = torch.from_numpy(image_01).unsqueeze(0).unsqueeze(0).to(device)
        target_density = torch.nn.functional.interpolate(high_res, size=(grid_size, grid_size), mode="area")
        return image_01, high_res, target_density, None, None
    high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_image(
        image_01,
        grid_size,
        device=device,
        sdf_truncate_px=sdf_truncate_px,
    )
    return image_01, high_res, target_density, high_res_sdf, target_sdf


def run_sdedit_branch(diffusion, model, x_noisy, device, timesteps, truncation_ratio, resample_jumps, desc):
    original_model = diffusion.model
    diffusion.model = model
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    t_start = int(np.clip(int(diffusion.num_timesteps * truncation_ratio), 1, diffusion.num_timesteps - 1))
    img = x_noisy.clone()

    with torch.no_grad():
        step_iter = tqdm(reversed(range(t_start)), total=t_start, desc=desc, leave=False)
        for i in step_iter:
            t_tensor = torch.full((img.shape[0],), i, dtype=torch.int64, device=device)
            for u in range(resample_jumps + 1):
                img = diffusion.p_sample(
                    img,
                    cond=None,
                    t=t_tensor,
                    clip_denoised=diffusion.sample_clip,
                    with_sampling=True,
                )
                if u == resample_jumps or i == 0:
                    break
                beta_i = diffusion.betas[i]
                noise = torch.randn_like(img)
                img = (1.0 - beta_i).sqrt() * img + beta_i.sqrt() * noise

    raw = img.detach().cpu().numpy()
    diffusion.model = original_model
    diffusion.reset_timesteps()
    diffusion.train()
    return raw


def offsets_to_pointset(offsets):
    pts = to_pointset_optimal_transport(offsets)
    return pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T


def offsets_batch_to_pointsets(offsets_batch):
    return np.stack([offsets_to_pointset(offsets_batch[i]) for i in range(offsets_batch.shape[0])], axis=0)


def target_image_to_points(image_01):
    """Detect stipple dot centroids in a [0,1] grayscale image and return (N, 2) in [0,1]."""
    from scipy import ndimage
    inv = 1.0 - image_01
    binary = (inv > 0.5).astype(np.uint8)
    labelled, n_labels = ndimage.label(binary)
    if n_labels == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centroids = ndimage.center_of_mass(binary, labelled, range(1, n_labels + 1))
    h, w = image_01.shape
    # centroids are (row, col) -> convert to (x, y) in [0, 1]
    points = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float32)
    return points

def print_metrics(name, points):
    spacing = compute_spacing_quality(points)
    print(
        f"{name:<12} CV={spacing['nn_cv']:.4f} | "
        f"Clumped={spacing['clumped_pct']:.2f}% | "
        f"SpacingScore={spacing['spacing_score']:.4f}"
    )
    return spacing


def save_panel(save_path, condition_image_01, gt_points_batch, baseline_points_batch, control_points_batch):
    if not HAS_MPL:
        print("matplotlib unavailable; skipping panel save")
        return False

    n_cols = len(gt_points_batch) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 8.6), dpi=180)

    axes[0, 0].imshow(condition_image_01, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Condition")
    axes[0, 0].axis("off")

    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Baseline", ha="center", va="center", fontsize=11)
    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, "Control V4", ha="center", va="center", fontsize=11)

    for i, gt_points in enumerate(gt_points_batch):
        col = i + 1
        axes[0, col].scatter(gt_points[:, 0], 1.0 - gt_points[:, 1], s=0.5, c="black")
        axes[0, col].set_xlim(0, 1)
        axes[0, col].set_ylim(0, 1)
        axes[0, col].set_aspect("equal")
        axes[0, col].set_title(f"GT{i+1}")
        axes[0, col].axis("off")

        baseline_points = baseline_points_batch[i]
        axes[1, col].scatter(baseline_points[:, 0], 1.0 - baseline_points[:, 1], s=0.5, c="black")
        axes[1, col].set_xlim(0, 1)
        axes[1, col].set_ylim(0, 1)
        axes[1, col].set_aspect("equal")
        axes[1, col].axis("off")

        control_points = control_points_batch[i]
        axes[2, col].scatter(control_points[:, 0], 1.0 - control_points[:, 1], s=0.5, c="black")
        axes[2, col].set_xlim(0, 1)
        axes[2, col].set_ylim(0, 1)
        axes[2, col].set_aspect("equal")
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Inference-only stress comparison between a trained baseline and a trained Control V4 model over a dataset root with original/target subfolders."
    )
    parser.add_argument("--data-root", "--dataset-root", dest="data_root", default=DATA_ROOT_DIR)

    baseline_paths = parser.add_argument_group("Baseline paths (2)")
    baseline_paths.add_argument("--baseline-config", default=BASELINE_CONFIG_PATH)
    baseline_paths.add_argument("--baseline-ckpt", default=BASELINE_CKPT_PATH)

    control_paths = parser.add_argument_group("Control paths (3)")
    control_paths.add_argument("--control-base-config", default=CONTROL_BASE_CONFIG_PATH)
    control_paths.add_argument("--control-base-ckpt", default=CONTROL_BASE_CKPT_PATH)
    control_paths.add_argument(
        "--control-ckpt",
        "--controlnet-ckpt",
        dest="control_ckpt",
        default=CONTROLNET_CKPT_PATH,
    )

    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", default=OUTPUT_ROOT_DIR)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--sample-timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO,
                        help="Truncation ratio for Control V4 (SDEdit from smart init, e.g. 0.30 = 300 steps)")
    parser.add_argument("--baseline-truncation-ratio", type=float, default=BASELINE_TRUNCATION_RATIO,
                        help="Truncation ratio for Baseline (1.0 = full denoising from pure noise)")
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--smart-init-features", action=argparse.BooleanOptionalAction, default=SMART_INIT_FEATURES)
    parser.add_argument("--batch-coords-features", action=argparse.BooleanOptionalAction, default=BATCH_COORDS_FEATURES)
    parser.add_argument("--smart-init-splat-sigma-px", type=float, default=0.5)
    parser.add_argument(
        "--enable-smart-init-splat-sigma",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_SMART_INIT_SPLAT_SIGMA,
        help="Use Gaussian soft-splat smart-init hint (otherwise use hard occupancy, matching no_splat runs)",
    )
    parser.add_argument("--enable-gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    parser.add_argument("--sdf-features", action=argparse.BooleanOptionalAction, default=SDF_FEATURES)
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--n-examples", type=int, default=N_EXAMPLES,
                        help="Number of target examples (columns) to compare")
    parser.add_argument(
        "--calculate-metrics",
        action=argparse.BooleanOptionalAction,
        default=CALCULATE_METRICS,
        help="Save one overfit-style metrics panel per sample into output_dir/<dataset>/metrics",
    )
    parser.add_argument(
        "--capacity-grid-size",
        type=int,
        default=CAPACITY_GRID_SIZE,
        help="Capacity grid size: >0 uses KxK, -1 uses full input image resolution",
    )
    args = parser.parse_args()

    if not (0.0 < args.truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0,1]")
    if not (0.0 < args.baseline_truncation_ratio <= 1.0):
        raise ValueError("--baseline-truncation-ratio must be in (0,1]")
    if args.n_examples <= 0:
        raise ValueError("--n-examples must be >= 1")
    if args.capacity_grid_size == 0 or args.capacity_grid_size < -1:
        raise ValueError("--capacity-grid-size must be > 0, or -1 for full input resolution")

    source_dir = os.path.join(args.data_root, "source")
    original_dir = os.path.join(args.data_root, "original")
    target_dir = os.path.join(args.data_root, "target")
    source_image_paths, target_image_paths, condition_source_kind = _pick_matched_source_target_examples(
        source_dir,
        original_dir,
        target_dir,
        args.n_examples,
    )

    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    baseline_ckpt_path = _resolve_ckpt_path(args.baseline_ckpt)
    control_base_ckpt_path = _resolve_ckpt_path(args.control_base_ckpt)

    baseline_diffusion = ParseSampleConfig(args.baseline_config)
    baseline_diffusion.load_state_dict(torch.load(baseline_ckpt_path, map_location="cpu")["diffu"])
    baseline_diffusion.to(device)
    baseline_denoiser = baseline_diffusion.model
    baseline_denoiser.eval()
    for param in baseline_denoiser.parameters():
        param.requires_grad = False

    control_diffusion = ParseSampleConfig(args.control_base_config)
    control_diffusion.load_state_dict(torch.load(control_base_ckpt_path, map_location="cpu")["diffu"])
    control_diffusion.to(device)
    control_backbone = control_diffusion.model
    control_backbone.eval()
    for param in control_backbone.parameters():
        param.requires_grad = False

    control_net = DynamicControlNet(
        control_backbone,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        batch_coords_features=args.batch_coords_features,
    ).to(device)
    ctrl_state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.load_state_dict(_extract_control_state_dict(ctrl_state), strict=True)
    control_net.eval()

    condition_images_01 = []
    high_res_list = []
    target_density_list = []
    high_res_sdf_list = []
    target_sdf_list = []
    for src_path in source_image_paths:
        image_01, hi, den, hi_sdf, den_sdf = load_condition(
            src_path,
            args.grid_size,
            device,
            sdf_features=args.sdf_features,
            sdf_truncate_px=args.sdf_truncate_px,
        )
        condition_images_01.append(image_01)
        high_res_list.append(hi)
        target_density_list.append(den)
        high_res_sdf_list.append(hi_sdf)
        target_sdf_list.append(den_sdf)

    # Stress datasets are expected to be shape-consistent; fail loudly otherwise.
    first_shape = high_res_list[0].shape
    for idx, tensor in enumerate(high_res_list[1:], start=1):
        if tensor.shape != first_shape:
            raise ValueError(
                "Source images have different shapes; this script currently expects a fixed shape. "
                f"Sample 0 shape={first_shape}, sample {idx} shape={tensor.shape}."
            )

    high_res = torch.cat(high_res_list, dim=0)
    target_density = torch.cat(target_density_list, dim=0)
    if args.sdf_features:
        high_res_sdf = torch.cat(high_res_sdf_list, dim=0)
        target_sdf = torch.cat(target_sdf_list, dim=0)
    else:
        high_res_sdf = None
        target_sdf = None

    gt_points_batch = [
        target_image_to_points(np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0)
        for path in tqdm(target_image_paths, desc="Extracting GT points", leave=False)
    ]

    if args.smart_init_features:
        smart_init_offsets_tensors = []
        for image_01 in condition_images_01:
            _, smart_offsets_np, _ = build_smart_init_from_image(
                image_01,
                grid_size=args.grid_size,
                n_points=args.grid_size * args.grid_size,
                seed=args.smart_init_seed,
            )
            smart_init_offsets_tensors.append(torch.from_numpy(smart_offsets_np).unsqueeze(0))
        smart_init_offsets_base = torch.cat(smart_init_offsets_tensors, dim=0).to(device)
        grid_centers_flat = _grid_centers_flat(args.grid_size, device, smart_init_offsets_base.dtype)
        smart_points_base = offsets_to_coords_gpu(smart_init_offsets_base, args.grid_size, grid_centers_flat)
        if args.enable_smart_init_splat_sigma:
            smart_init_grid = render_smart_init_gpu(
                smart_points_base,
                grid_size=args.grid_size,
                sigma_px=args.smart_init_splat_sigma_px,
            )
        else:
            smart_init_grid = render_occupancy_grid_gpu(
                smart_points_base,
                grid_size=args.grid_size,
            )
        x_init = smart_init_offsets_base.contiguous()
    else:
        smart_init_grid = None
        x_init = torch.randn((high_res.shape[0], 2, args.grid_size, args.grid_size), device=device)

    controlled = DynamicControlledDenoiser(control_backbone, control_net)
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
    controlled.eval()

    baseline_diffusion.set_num_timesteps(args.sample_timesteps)

    # Baseline: starts from pure noise (full schedule)
    x_noisy_baseline = torch.randn_like(x_init)

    # Control: SDEdit from smart_init noised at truncation t_start
    control_diffusion.set_num_timesteps(args.sample_timesteps)
    control_t_start = int(np.clip(
        int(control_diffusion.num_timesteps * args.truncation_ratio),
        1, control_diffusion.num_timesteps - 1,
    ))
    noise = torch.randn_like(x_init)
    alpha_t = control_diffusion.alphas_cumprod[control_t_start]
    x_noisy_control = add_noise_at_t(x_init, alpha_t, noise=noise)

    baseline_diffusion.reset_timesteps()
    control_diffusion.reset_timesteps()

    print(f"Baseline: full denoising ({int(args.baseline_truncation_ratio * args.sample_timesteps)} steps from pure noise)")
    print(f"Control V4: SDEdit from smart_init ({control_t_start} steps, truncation_ratio={args.truncation_ratio})")
    print(f"Smart Init enabled: {args.smart_init_features}")
    print(f"Batch coords      : {args.batch_coords_features}")
    print(f"SDF enabled       : {args.sdf_features}")

    models_to_run = [
        ("Baseline",   baseline_diffusion, baseline_denoiser, "baseline",   x_noisy_baseline, args.baseline_truncation_ratio),
        ("Control V4", control_diffusion,   controlled,        "control_v4", x_noisy_control,  args.truncation_ratio),
    ]
    results = {}
    for model_name, diffusion_obj, model_obj, desc, x_noisy, trunc_ratio in tqdm(models_to_run, desc="Models"):
        results[desc] = run_sdedit_branch(
            diffusion_obj,
            model_obj,
            x_noisy,
            device,
            timesteps=args.sample_timesteps,
            truncation_ratio=trunc_ratio,
            resample_jumps=args.resample_jumps,
            desc=model_name,
        )
    baseline_raw = results["baseline"]
    control_raw  = results["control_v4"]

    baseline_points_batch = offsets_batch_to_pointsets(baseline_raw)
    control_points_batch = offsets_batch_to_pointsets(control_raw)

    print(f"Data root: {args.data_root}")
    print(f"Condition source kind: {condition_source_kind}")
    print("Condition/target pairs:")
    for src_path, tgt_path in zip(source_image_paths, target_image_paths):
        print(f"  - source: {src_path}")
        print(f"    target: {tgt_path}")
    print(f"Target images ({args.n_examples}):")
    for target_path in target_image_paths:
        print(f"  - {target_path}")
    print(f"Baseline config: {args.baseline_config}")
    print(f"Baseline ckpt: {baseline_ckpt_path}")
    print(f"Control base config: {args.control_base_config}")
    print(f"Control base ckpt: {control_base_ckpt_path}")
    print(f"Control ckpt: {args.control_ckpt}")
    print(f"Control smart-init soft-splat enabled: {args.enable_smart_init_splat_sigma}")
    if args.enable_smart_init_splat_sigma:
        print(f"Control smart-init splat sigma (px): {args.smart_init_splat_sigma_px}")
    print(f"Device: {args.device}")
    print(f"Samples per model: {args.n_examples}")
    print(f"Control t_start: {control_t_start}/{args.sample_timesteps}")
    for i in range(args.n_examples):
        print_metrics(f"Baseline[{i}]", baseline_points_batch[i])
    for i in range(args.n_examples):
        print_metrics(f"Control[{i}]", control_points_batch[i])

    data_stem = os.path.basename(os.path.normpath(args.data_root))
    out_dir = os.path.join(args.output_dir, data_stem)
    os.makedirs(out_dir, exist_ok=True)

    # Keep a representative condition snapshot for backward compatibility.
    Image.fromarray((condition_images_01[0] * 255.0).astype(np.uint8), mode="L").save(os.path.join(out_dir, "condition.png"))
    for i, gt_img_for_save in enumerate(gt_points_batch):
        # save a synthetic dot-on-white image reconstructed from the extracted points
        pass  # raw GT PNGs are preserved in target_dir; no need to re-save
    np.save(os.path.join(out_dir, "gt_points.npy"), np.array(gt_points_batch, dtype=object))
    np.save(os.path.join(out_dir, "baseline_offsets.npy"), baseline_raw)
    np.save(os.path.join(out_dir, "control_v4_offsets.npy"), control_raw)
    np.save(os.path.join(out_dir, "baseline_points.npy"), baseline_points_batch)
    np.save(os.path.join(out_dir, "control_v4_points.npy"), control_points_batch)

    panel_path = os.path.join(out_dir, "comparison_panel_paper_style.png")
    saved = save_panel(panel_path, condition_images_01[0], gt_points_batch, baseline_points_batch, control_points_batch)
    if saved:
        print(f"Saved panel to: {panel_path}")

    if args.calculate_metrics:
        metrics_dir = os.path.join(out_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metric_saved_count = 0

        for i, target_path in enumerate(target_image_paths):
            condition_image_u8 = (condition_images_01[i] * 255.0).astype(np.uint8)
            target_image_u8 = np.array(Image.open(target_path).convert("L"), dtype=np.uint8)
            sample_stem = os.path.splitext(os.path.basename(target_path))[0]
            metrics_path = os.path.join(metrics_dir, f"{i:03d}_{sample_stem}_metrics.png")
            saved_metrics = visualize_overfit_metrics(
                condition_image_u8,
                target_image_u8,
                gt_points_batch[i],
                [baseline_points_batch[i], control_points_batch[i]],
                metrics_path,
                step=None,
                gt_offsets=None,
                capacity_grid_size=args.capacity_grid_size,
                pred_labels=["Baseline", "Control V4"],
            )
            if saved_metrics:
                metric_saved_count += 1

        print(f"Saved {metric_saved_count}/{args.n_examples} metrics panels to: {metrics_dir}")


if __name__ == "__main__":
    main()
