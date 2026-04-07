import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

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
from utils.stippling_metrics import compute_spacing_quality


COMPARE_IMAGE_PATH = os.path.join("experiments", "stress_test_density.png")
OUTPUT_ROOT_DIR = os.path.join("experiments", "outputs")
DEVICE = "cuda"
N_SAMPLES = 1

# Baseline configuration
BASELINE_CONFIG_PATH = "GBN/config.json"
BASELINE_CKPT_PATH = "GBN/model.ckpt"

# Control V4 configuration
CONTROL_BASE_CONFIG_PATH = "config/GBN/config.json"
CONTROL_BASE_CKPT_PATH = "config/GBN/model.ckpt"
CONTROLNET_CKPT_PATH = "control_v4/train_outputs/checkpoints/best_controlnet_ep0002_score0.373_cv0.354_clumped1.86.pt"
GRID_SIZE = 32
TIMESTEPS = 1000
TRUNCATION_RATIO = 0.30
RESAMPLE_JUMPS = 0
ENABLE_GECCO = True
USE_SDF = True
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SEED = 42



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


def load_condition(image_path, grid_size, device, sdf_truncate_px=0.0):
    image_01 = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
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
        for i in reversed(range(t_start)):
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


def print_metrics(name, points):
    spacing = compute_spacing_quality(points)
    print(
        f"{name:<12} CV={spacing['nn_cv']:.4f} | "
        f"Clumped={spacing['clumped_pct']:.2f}% | "
        f"SpacingScore={spacing['spacing_score']:.4f}"
    )
    return spacing


def save_panel(save_path, image_01, smart_init_points, baseline_points, control_points):
    if not HAS_MPL:
        print("matplotlib unavailable; skipping panel save")
        return False

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=180)

    axes[0].imshow(image_01, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Condition")
    axes[0].axis("off")

    for ax, points, title in (
        (axes[1], smart_init_points, "Smart Init"),
        (axes[2], baseline_points, "Baseline"),
        (axes[3], control_points, "Control V4"),
    ):
        ax.scatter(points[:, 0], 1.0 - points[:, 1], s=0.5, c="black")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Inference-only stress comparison between a trained baseline and a trained Control V4 model."
    )
    parser.add_argument("--compare-image", "--image", dest="compare_image", default=COMPARE_IMAGE_PATH)

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
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO)
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--smart-init-splat-sigma-px", type=float, default=0.5)
    parser.add_argument("--enable-gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    parser.add_argument("--use-sdf", action=argparse.BooleanOptionalAction, default=USE_SDF)
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--panel-sample-index", type=int, default=0)
    args = parser.parse_args()

    if not (0.0 < args.truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0,1]")

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
    ).to(device)
    ctrl_state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.load_state_dict(_extract_control_state_dict(ctrl_state), strict=False)
    control_net.eval()

    image_01, high_res, target_density, high_res_sdf, target_sdf = load_condition(
        args.compare_image,
        args.grid_size,
        device,
        sdf_truncate_px=args.sdf_truncate_px,
    )

    if not args.use_sdf:
        high_res_sdf = torch.zeros_like(high_res_sdf)
        target_sdf = torch.zeros_like(target_sdf)

    smart_points_np, smart_offsets_np, _ = build_smart_init_from_image(
        image_01,
        grid_size=args.grid_size,
        n_points=args.grid_size * args.grid_size,
        seed=args.smart_init_seed,
    )
    smart_init_offsets_base = torch.from_numpy(smart_offsets_np).unsqueeze(0).to(device)
    grid_centers_flat = _grid_centers_flat(args.grid_size, device, smart_init_offsets_base.dtype)
    smart_points_base = offsets_to_coords_gpu(smart_init_offsets_base, args.grid_size, grid_centers_flat)
    smart_init_grid = render_smart_init_gpu(
        smart_points_base,
        grid_size=args.grid_size,
        sigma_px=args.smart_init_splat_sigma_px,
    )

    controlled = DynamicControlledDenoiser(control_backbone, control_net)
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
    controlled.eval()

    baseline_diffusion.set_num_timesteps(args.sample_timesteps)
    t_start = int(
        np.clip(
            int(baseline_diffusion.num_timesteps * args.truncation_ratio),
            1,
            baseline_diffusion.num_timesteps - 1,
        )
    )
    x_init = smart_init_offsets_base
    if x_init.shape[0] != args.n_samples:
        x_init = x_init.expand(args.n_samples, -1, -1, -1).contiguous()
    noise = torch.randn_like(x_init)
    alpha_t = baseline_diffusion.alphas_cumprod[t_start]
    x_noisy = add_noise_at_t(x_init, alpha_t, noise=noise)
    baseline_diffusion.reset_timesteps()

    baseline_raw = run_sdedit_branch(
        baseline_diffusion,
        baseline_denoiser,
        x_noisy,
        device,
        timesteps=args.sample_timesteps,
        truncation_ratio=args.truncation_ratio,
        resample_jumps=args.resample_jumps,
        desc="baseline",
    )
    control_raw = run_sdedit_branch(
        control_diffusion,
        controlled,
        x_noisy,
        device,
        timesteps=args.sample_timesteps,
        truncation_ratio=args.truncation_ratio,
        resample_jumps=args.resample_jumps,
        desc="control_v4",
    )

    smart_init_points = offsets_to_pointset(smart_offsets_np)
    baseline_points_batch = offsets_batch_to_pointsets(baseline_raw)
    control_points_batch = offsets_batch_to_pointsets(control_raw)

    panel_sample_index = int(np.clip(args.panel_sample_index, 0, args.n_samples - 1))
    baseline_points = baseline_points_batch[panel_sample_index]
    control_points = control_points_batch[panel_sample_index]

    print(f"Compare image: {args.compare_image}")
    print(f"Baseline config: {args.baseline_config}")
    print(f"Baseline ckpt: {baseline_ckpt_path}")
    print(f"Control base config: {args.control_base_config}")
    print(f"Control base ckpt: {control_base_ckpt_path}")
    print(f"Control ckpt: {args.control_ckpt}")
    print(f"Device: {args.device}")
    print(f"Samples per model: {args.n_samples}")
    print(f"Panel sample index: {panel_sample_index}")
    print(f"t_start: {t_start}/{args.sample_timesteps}")
    print_metrics("Smart Init", smart_init_points)
    for i in range(args.n_samples):
        print_metrics(f"Baseline[{i}]", baseline_points_batch[i])
    for i in range(args.n_samples):
        print_metrics(f"Control[{i}]", control_points_batch[i])

    image_stem = os.path.splitext(os.path.basename(args.compare_image))[0]
    out_dir = os.path.join(args.output_dir, image_stem)
    os.makedirs(out_dir, exist_ok=True)

    Image.fromarray((image_01 * 255.0).astype(np.uint8), mode="L").save(os.path.join(out_dir, "condition.png"))
    np.save(os.path.join(out_dir, "smart_init_offsets.npy"), smart_offsets_np)
    np.save(os.path.join(out_dir, "baseline_offsets.npy"), baseline_raw)
    np.save(os.path.join(out_dir, "control_v4_offsets.npy"), control_raw)
    np.save(os.path.join(out_dir, "smart_init_points.npy"), smart_init_points)
    np.save(os.path.join(out_dir, "baseline_points.npy"), baseline_points_batch)
    np.save(os.path.join(out_dir, "control_v4_points.npy"), control_points_batch)

    panel_path = os.path.join(out_dir, "comparison_panel.png")
    saved = save_panel(panel_path, image_01, smart_init_points, baseline_points, control_points)
    if saved:
        print(f"Saved panel to: {panel_path}")


if __name__ == "__main__":
    main()
