"""Generate stipple point sets using the Dynamic ControlNet V3.

Usage (from project root):
    python control_v3/sample_control.py \
        --config       config/GBN/config.json \
        --base_ckpt    config/GBN/model.ckpt \
        --control_ckpt control_v3/control_out/dynamic_controlnet_v3_ep100.pt \
        --image        my_photo.png \
        --batch        16 \
        --timesteps    1000 \
        --output       stippled.npy
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from control_v3.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser


def load_condition(image_path, grid_size, device):
    """Load a grayscale image and return high-res + target density tensors."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    high_res = torch.from_numpy(img).float() / 255.0
    high_res = high_res.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    target_density = F.interpolate(
        high_res, size=(grid_size, grid_size), mode="area"
    )  # (1, 1, grid_size, grid_size)

    return high_res, target_density


def compute_sdf(target_density_tensor):
    """Compute normalised distance-to-nearest-dot SDF of empty space."""
    from scipy import ndimage as ndi
    binary = (target_density_tensor.cpu().numpy() > 0.5).astype(np.float32)
    sdf_batch = []
    for i in range(binary.shape[0]):
        empty_mask = 1.0 - binary[i, 0]
        dist = ndi.distance_transform_edt(empty_mask)
        max_dist = dist.max() if dist.max() > 0 else 1.0
        sdf_batch.append(dist / max_dist)
    sdf_np = np.array(sdf_batch, dtype=np.float32)[:, np.newaxis]
    return torch.from_numpy(sdf_np).to(target_density_tensor.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/GBN/config.json")
    parser.add_argument("--base_ckpt", default="config/GBN/model.ckpt")
    parser.add_argument("--control_ckpt", required=True,
                        help="Path to trained Dynamic ControlNet V3 .pt file")
    parser.add_argument("--image", required=True,
                        help="Grayscale condition image")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--output", default="stippled.npy")
    parser.add_argument("--no_ot", action="store_true",
                        help="Skip inverse OT (save raw offset grids)")
    parser.add_argument("--binary-threshold", type=float, default=0.5,
                        help="Threshold to binarize target density (default: 0.5)")
    parser.add_argument("--resample-jumps", type=int, default=2,
                        help="RePaint micro-loops per timestep (0=disabled)")
    parser.add_argument("--device", default="cuda")
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
    control_net = DynamicControlNet(denoiser, grid_size=args.grid_size).to(device)
    ctrl_state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.load_state_dict(ctrl_state["control_net"])
    control_net.eval()

    # ── wire up DynamicControlledDenoiser ─────────────────────────────
    controlled = DynamicControlledDenoiser(denoiser, control_net)
    high_res, target_density = load_condition(args.image, args.grid_size, device)
    target_density = (target_density > args.binary_threshold).float()
    sdf = compute_sdf(target_density)
    controlled.set_condition(high_res, target_density, sdf_map=sdf)

    diffusion.model = controlled
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    # ── sample ───────────────────────────────────────────────────────
    shape = [args.batch, 2, args.grid_size, args.grid_size]
    print(f"Sampling {args.batch} point sets  |  shape {shape}  |  "
          f"T={args.timesteps}  |  resample_jumps={args.resample_jumps}")

    with torch.no_grad():
        if args.resample_jumps == 0:
            samples_raw = diffusion.p_sample_loop(
                shape, img=None, cond=None, with_tqdm=True, with_sampling=True
            )
        else:
            from tqdm import tqdm
            img = diffusion.noise_fn(shape).to(device)
            for i in tqdm(reversed(range(diffusion.num_timesteps - 1)),
                          total=diffusion.num_timesteps - 1, desc="sampling"):
                t_tensor = torch.full((args.batch,), i, dtype=torch.int64, device=device)
                for u in range(args.resample_jumps + 1):
                    img = diffusion.p_sample(img, cond=None, t=t_tensor,
                                             clip_denoised=diffusion.sample_clip,
                                             with_sampling=True)
                    if u == args.resample_jumps or i == 0:
                        break
                    beta_i = diffusion.betas[i]
                    noise = torch.randn_like(img)
                    img = (1.0 - beta_i).sqrt() * img + beta_i.sqrt() * noise
            samples_raw = img

    samples_raw = samples_raw.cpu().numpy()

    # ── inverse OT -> point sets in [0, 1]^2 ────────────────────────
    if not args.no_ot:
        samples = []
        for s in samples_raw:
            pts = to_pointset_optimal_transport(s)
            pts = pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T
            samples.append(pts)
        np.save(args.output, samples)
    else:
        np.save(args.output, samples_raw)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
