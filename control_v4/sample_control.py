"""Generate stipple point sets using Dynamic ControlNet V4 (Truncated Control)."""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

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


# ?? editable defaults ????????????????????????????????????????????????
CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT = "config/GBN/model.ckpt"
CONTROL_CKPT = "control_v4/train_outputs/dynamic_controlnet_v3_ep1.pt"
IMAGE_PATH = "control_v4/sample_test/source.png"
OUTPUT_DIR = "control_v4/sample_outputs"
N_SAMPLES = 1
TIMESTEPS = 1000
GRID_SIZE = 32
RESAMPLE_JUMPS = 2
ENABLE_GECCO = True
DEVICE = "cuda"
SDF_TRUNCATE_PX = 8.0
USE_SDF = True
SHOW_DENOISING = False
DENOISE_INTERVAL = 50
TRUNCATION_RATIO = 0.30
T_START_STEP = -1
SMART_INIT_SEED = 42


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


def _save_condition_debug_tensors(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cond_map = {
        "high_res": high_res,
        "high_res_sdf": high_res_sdf,
        "target_density": target_density,
        "target_sdf": target_sdf,
        "smart_init_grid": smart_init_grid,
    }
    for name, tensor in cond_map.items():
        arr = tensor.detach().cpu().float().numpy().squeeze()
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    if HAS_MPL:
        ordered_names = ["high_res", "high_res_sdf", "target_density", "target_sdf", "smart_init_grid"]
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=140)
        for ax, name in zip(axes.flat, ordered_names + ["_"]):
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
    parser.add_argument("--image", default=IMAGE_PATH)
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
    parser.add_argument("--denoise-interval", type=int, default=DENOISE_INTERVAL)
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO)
    parser.add_argument("--t-start-step", type=int, default=T_START_STEP,
                        help="If >=0, overrides truncation-ratio derived start step")
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    if not (0.0 < args.truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0,1]")

    device = torch.device(args.device)

    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    denoiser = diffusion.model

    control_net = DynamicControlNet(
        denoiser,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
    ).to(device)
    ctrl_state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.load_state_dict(_extract_control_state_dict(ctrl_state), strict=False)
    control_net.eval()

    img_stem = os.path.splitext(os.path.basename(args.image))[0]
    sample_base_dir = os.path.join(args.out_dir, img_stem)
    os.makedirs(sample_base_dir, exist_ok=True)

    image_01, high_res, target_density, high_res_sdf, target_sdf = load_condition(
        args.image,
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
    smart_init_grid = torch.from_numpy(smart_grid_np).unsqueeze(0).to(device)

    if not args.use_sdf:
        high_res_sdf = torch.zeros_like(high_res_sdf)
        target_sdf = torch.zeros_like(target_sdf)

    conditions_dir = os.path.join(sample_base_dir, "conditions")
    _save_condition_debug_tensors(
        high_res,
        high_res_sdf,
        target_density,
        target_sdf,
        smart_init_grid,
        conditions_dir,
    )

    smart_dir = os.path.join(sample_base_dir, "smart_init")
    save_smart_init_debug(smart_dir, smart_points, smart_offsets_np, smart_grid_np)

    controlled = DynamicControlledDenoiser(denoiser, control_net)
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
    diffusion.model = controlled
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    t_start = args.t_start_step if args.t_start_step >= 0 else int(args.timesteps * args.truncation_ratio)
    t_start = int(np.clip(t_start, 1, max(args.timesteps - 1, 1)))

    n_samples = args.n_samples
    shape = [n_samples, 2, args.grid_size, args.grid_size]
    x_init = torch.from_numpy(smart_offsets_np).unsqueeze(0).to(device)
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
                if elapsed % args.denoise_interval == 0:
                    step_path = os.path.join(steps_dir, f"step_{elapsed:04d}.png")
                    _save_denoise_step(img, i, t_start, step_path)

    samples_raw = img.detach().cpu().numpy()

    npy_dir = os.path.join(sample_base_dir, "npy")
    png_dir = os.path.join(sample_base_dir, "png")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for idx, s in enumerate(samples_raw):
        suffix = f"_{idx + 1}"
        npy_path = os.path.join(npy_dir, f"{img_stem}{suffix}.npy")
        png_path = os.path.join(png_dir, f"{img_stem}{suffix}.png")

        if not args.no_ot:
            pts = to_pointset_optimal_transport(s)
            pts = pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T
            np.save(npy_path, pts)
            save_sample_image(args.image, pts, png_path)
        else:
            np.save(npy_path, s)

    print("Done.")


if __name__ == "__main__":
    main()
