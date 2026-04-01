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
CONTROL_CKPT  = "control_v3/train_outputs_wave/dynamic_controlnet_v3_ep25.pt"
IMAGE_PATH    = "/groups/asharf_group/ofirgila/ControlNet/training/data_taksim/source/taksim-circle.png"
OUTPUT_DIR    = "control_v3/sample_outputs"
N_SAMPLES     = 2
TIMESTEPS     = 1000
GRID_SIZE     = 32
RESAMPLE_JUMPS = 2
ENABLE_GECCO  = True
DEVICE        = "cuda"
SDF_TRUNCATE_PX = 8.0
USE_SDF = True

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
    control_net.load_state_dict(ctrl_state["control_net"])
    control_net.eval()

    print(f"Loaded checkpoint : {args.control_ckpt}")
    print(f"GECCO enabled     : {args.enable_gecco}")
    print(f"SDF enabled       : {args.use_sdf}")

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
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf)

    diffusion.model = controlled
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    # ── sample ───────────────────────────────────────────────────────
    n_samples = args.n_samples
    shape = [n_samples, 2, args.grid_size, args.grid_size]
    print(f"Generating {n_samples} samples  |  shape {shape}  |  "
          f"T={args.timesteps}  |  resample_jumps={args.resample_jumps}")

    apply_manual_loop = args.resample_jumps > 0
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
            samples_raw = img

    samples_raw = samples_raw.cpu().numpy()

    # ── save outputs: one .npy + .png per sample ─────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    img_stem = os.path.splitext(os.path.basename(args.image))[0]
    print(f"Saving {n_samples} samples to {args.out_dir}/")

    for idx, s in enumerate(samples_raw):
        suffix = f"_{idx + 1}"
        npy_path = os.path.join(args.out_dir, f"{img_stem}{suffix}.npy")
        png_path = os.path.join(args.out_dir, f"{img_stem}{suffix}.png")

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
