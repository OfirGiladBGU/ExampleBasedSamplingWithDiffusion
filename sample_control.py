"""Generate stipple point sets conditioned on a grayscale image.

Usage:
    python sample_control.py \
        --config       config/GBN/config.json \
        --base_ckpt    config/GBN/model.ckpt \
        --control_ckpt control_out/controlnet_ep100.pt \
        --image        my_photo.png \
        --batch        16 \
        --timesteps    1000 \
        --output       stippled.npy
"""

import argparse
import numpy as np
import cv2
import torch

from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from models.ControlNet import ControlNet, ControlledDenoiser


def load_condition(image_path, grid_size, device):
    """Load a grayscale image, resize to grid_size, return (1, 1, H, W)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (grid_size, grid_size),
                     interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/GBN/config.json",
                        help="Path to config.json (same as training)")
    parser.add_argument("--base_ckpt", default="config/GBN/model.ckpt",
                        help="Path to pretrained model.ckpt")
    parser.add_argument("--control_ckpt", required=True,
                        help="Path to trained ControlNet .pt file")
    parser.add_argument("--image", required=True,
                        help="Grayscale condition image")
    parser.add_argument("--batch", type=int, default=16,
                        help="Number of samples to generate")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--output", default="stippled.npy")
    parser.add_argument("--no_ot", action="store_true",
                        help="Skip inverse OT (save raw offset grids)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Load pretrained diffusion model ----
    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(
        torch.load(args.base_ckpt, map_location="cpu")["diffu"]
    )
    diffusion.to(device)

    denoiser = diffusion.model

    # ---- Build & load ControlNet ----
    control_net = ControlNet(denoiser).to(device)
    ctrl_state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.load_state_dict(ctrl_state["control_net"])
    control_net.eval()

    # ---- Wire up ControlledDenoiser ----
    controlled = ControlledDenoiser(denoiser, control_net)
    cond_img = load_condition(args.image, args.grid_size, device)
    controlled.set_condition(cond_img)

    diffusion.model = controlled
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    # ---- Sample ----
    shape = [args.batch, 2, args.grid_size, args.grid_size]
    print(f"Sampling {args.batch} point sets  |  shape {shape}  |  "
          f"T={args.timesteps}")

    with torch.no_grad():
        samples_raw = diffusion.p_sample_loop(
            shape, img=None, cond=None, with_tqdm=True, with_sampling=True
        )

    samples_raw = samples_raw.cpu().numpy()

    # ---- Inverse OT -> point sets in [0, 1]^2 ----
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
