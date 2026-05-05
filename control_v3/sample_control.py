"""Generate stipple point sets using the Dynamic ControlNet V3.8.

Usage (from project root):
    python control_v3/sample_control.py \\
        --config       config/GBN/config.json \\
        --base_ckpt    config/GBN/model.ckpt \\
        --control_ckpt control_v3/train_outputs_wave/dynamic_controlnet_v3_ep25.pt \\
        --image        /path/to/image.png \\
        --n-samples    16 \\
        --timesteps    1000 \\
        --resample-jumps 2 \\
        --out-dir      control_v3/sample_outputs
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── editable defaults ────────────────────────────────────────────────
CONFIG_PATH   = "config/GBN/config.json"
BASE_CKPT     = "config/GBN/model.ckpt"
CONTROL_CKPT  = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v3/train_outputs_icons50_512/dynamic_controlnet_v3_ep250.pt"
IMAGE_PATH    = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source/Icons-50/monkey/emoji-one_4_monkey.png"
OUTPUT_DIR    = "control_v3/sample_outputs"
N_SAMPLES     = 2
TIMESTEPS     = 1000
GRID_SIZE     = 32
RESAMPLE_JUMPS = 2
ENABLE_GECCO  = True
DEVICE        = "cuda"
SDF_TRUNCATE_PX = 8.0
USE_SDF = True
SHOW_DENOISING = True
DENOISE_INTERVAL = 50

import cv2
import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from control_v3.conditioning import build_condition_tensors_from_image
from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from control_v3.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser


def save_sample_image(image_path, pts, out_png_path):
    """Save a single stipple scatter alongside its condition image.

    Layout: 2 columns — left: condition, right: stipple scatter.
    """
    if not HAS_MPL:
        print("matplotlib unavailable; skipping image export")
        return

    cond_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cond_img is None:
        print(f"Warning: could not read condition image {image_path!r} for vis")
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
    print(f"  -> {out_png_path}")


def _save_denoise_step(img_tensor, timestep_i, total_timesteps, grid_size, out_path):
    """Save a scatter-plot preview of sample 0 at an intermediate denoising step."""
    if not HAS_MPL:
        return
    offsets = img_tensor[0].cpu().float().numpy()  # (2, H, W)
    H, W = offsets.shape[1], offsets.shape[2]
    cx = (np.arange(W) + 0.5) / W
    cy = (np.arange(H) + 0.5) / H
    gx, gy = np.meshgrid(cx, cy)           # (H, W)
    px = np.clip(gx + offsets[0] / W, 0.0, 1.0).flatten()
    py = np.clip(gy + offsets[1] / H, 0.0, 1.0).flatten()
    step_num = total_timesteps - 1 - timestep_i
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.scatter(px, 1.0 - py, c="black", s=0.5, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"step {step_num}/{total_timesteps - 1}  (t={timestep_i})", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def _save_condition_debug_tensors(high_res, high_res_sdf, target_density, target_sdf, out_dir):
    """Export model condition tensors for debugging."""
    os.makedirs(out_dir, exist_ok=True)

    cond_map = {
        "high_res": high_res,
        "high_res_sdf": high_res_sdf,
        "target_density": target_density,
        "target_sdf": target_sdf,
    }

    for name, tensor in cond_map.items():
        arr = tensor.detach().cpu().float().numpy().squeeze()
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

    if HAS_MPL:
        ordered_names = ["high_res", "high_res_sdf", "target_density", "target_sdf"]
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=140)
        for ax, name in zip(axes.flat, ordered_names):
            arr = cond_map[name].detach().cpu().float().numpy().squeeze()
            if arr.ndim != 2:
                ax.axis("off")
                ax.set_title(f"{name} (invalid shape)")
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
    """Load a grayscale image and return image, density, and SDF condition tensors."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return build_condition_tensors_from_image(
        img.astype(np.float32) / 255.0,
        grid_size,
        device,
        sdf_truncate_px=sdf_truncate_px,
    )


def _extract_control_state_dict(ctrl_state):
    """Return a ControlNet state_dict from common checkpoint layouts."""
    if not isinstance(ctrl_state, dict):
        raise TypeError(f"Control checkpoint must be a dict, got {type(ctrl_state)}")

    for key in ("control_net", "model_state_dict", "state_dict"):
        value = ctrl_state.get(key)
        if isinstance(value, dict):
            return value

    # Some checkpoints are saved as a raw state_dict (param_name -> tensor).
    if ctrl_state and all(hasattr(v, "shape") for v in ctrl_state.values()):
        return ctrl_state

    keys_preview = ", ".join(list(ctrl_state.keys())[:8])
    raise KeyError(
        "Could not find control weights in checkpoint. "
        f"Expected one of ['control_net', 'model_state_dict', 'state_dict']; "
        f"found keys: [{keys_preview}]"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--base_ckpt", default=BASE_CKPT)
    parser.add_argument("--control_ckpt", default=CONTROL_CKPT,
                        help="Path to trained Dynamic ControlNet V3 .pt file")
    parser.add_argument("--image", default=IMAGE_PATH,
                        help="Grayscale condition image")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help="Number of independent samples to generate")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    parser.add_argument("--out-dir", default=OUTPUT_DIR,
                        help="Output directory; files are named {image_stem}_{i}.npy / .png")
    parser.add_argument("--no_ot", action="store_true",
                        help="Skip inverse OT (save raw offset grids)")
    parser.add_argument("--enable-gecco", default=ENABLE_GECCO,
                        action=argparse.BooleanOptionalAction,
                        help="Enable GECCO dynamic features (must match training; default: True)")
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS,
                        help="RePaint micro-loops per timestep (0=disabled)")
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX,
                        help="Truncate signed distance magnitudes before max-normalization (0 disables)")
    parser.add_argument(
        "--use-sdf",
        action=argparse.BooleanOptionalAction,
        default=USE_SDF,
        help="Pass real SDF channels to the model (--no-use-sdf zeroes them out for ablation)",
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--show-denoising",
        action=argparse.BooleanOptionalAction,
        default=SHOW_DENOISING,
        help="Save intermediate denoising previews (sample 0) to <out-dir>/<stem>_steps/",
    )
    parser.add_argument(
        "--denoise-interval",
        type=int,
        default=DENOISE_INTERVAL,
        help="Save a preview every N denoising steps when --show-denoising is set (default: 50)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── load pretrained diffusion model ──────────────────────────────
    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(
        torch.load(args.base_ckpt, map_location="cpu")["diffu"]
    )
    diffusion.to(device)
    denoiser = diffusion.model

    # ── build & load Dynamic ControlNet V3 ───────────────────────────
    control_net = DynamicControlNet(
        denoiser,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
    ).to(device)
    ctrl_state = torch.load(args.control_ckpt, map_location="cpu")
    control_state_dict = _extract_control_state_dict(ctrl_state)
    control_net.load_state_dict(control_state_dict)
    control_net.eval()

    print(f"Loaded checkpoint : {args.control_ckpt}")
    print(f"GECCO enabled     : {args.enable_gecco}")
    print(f"SDF enabled       : {args.use_sdf}")

    img_stem = os.path.splitext(os.path.basename(args.image))[0]
    sample_base_dir = os.path.join(args.out_dir, img_stem)
    os.makedirs(sample_base_dir, exist_ok=True)

    # ── wire up DynamicControlledDenoiser ─────────────────────────────
    controlled = DynamicControlledDenoiser(denoiser, control_net)
    high_res, target_density, high_res_sdf, target_sdf = load_condition(
        args.image,
        args.grid_size,
        device,
        sdf_truncate_px=args.sdf_truncate_px,
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
        conditions_dir,
    )

    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf)

    diffusion.model = controlled
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    # ── sample ───────────────────────────────────────────────────────
    n_samples = args.n_samples
    shape = [n_samples, 2, args.grid_size, args.grid_size]
    print(f"Generating {n_samples} samples  |  shape {shape}  |  "
          f"T={args.timesteps}  |  resample_jumps={args.resample_jumps}")

    # ── denoising-step preview setup ─────────────────────────────────
    steps_dir = None
    if args.show_denoising:
        steps_dir = os.path.join(sample_base_dir, "denoising_steps")
        os.makedirs(steps_dir, exist_ok=True)
        print(f"Denoising previews: {steps_dir}/  (every {args.denoise_interval} steps)")

    apply_manual_loop = args.resample_jumps > 0 or args.show_denoising
    with torch.no_grad() if not apply_manual_loop else torch.enable_grad():
        if not apply_manual_loop:
            samples_raw = diffusion.p_sample_loop(
                shape, img=None, cond=None, with_tqdm=True, with_sampling=True
            )
        else:
            from tqdm import tqdm
            img = diffusion.noise_fn(shape).to(device)
            for i in tqdm(reversed(range(diffusion.num_timesteps - 1)),
                          total=diffusion.num_timesteps - 1, desc="sampling"):
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

                # ── save intermediate preview ─────────────────────────
                if steps_dir is not None:
                    elapsed = diffusion.num_timesteps - 1 - i
                    if elapsed % args.denoise_interval == 0:
                        step_path = os.path.join(steps_dir, f"step_{elapsed:04d}.png")
                        _save_denoise_step(img, i, diffusion.num_timesteps, args.grid_size, step_path)

            samples_raw = img

    samples_raw = samples_raw.cpu().numpy()

    # ── save outputs: one .npy + .png per sample ─────────────────────
    os.makedirs(sample_base_dir, exist_ok=True)
    npy_dir = os.path.join(sample_base_dir, "npy")
    png_dir = os.path.join(sample_base_dir, "png")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    print(f"Saving {n_samples} samples to {sample_base_dir}/ (npy/, png/)")

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
