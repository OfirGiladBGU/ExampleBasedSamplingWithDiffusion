"""Generate stipple point sets using the image-GECCO early-fusion wrapper (control_v5).

Usage (from project root):
    python control_v5/sample_control.py \
        --config    config/GBN/config.json \
        --ckpt      config/GBN/model.ckpt  \
        --wrapper   control_v5/train_out/checkpoints/gecco_wrapper_ep100.pt \
        --image     /path/to/condition.png \
        --n         1024                   \
        --grid-size 32                     \
        --out       control_v5/sample_out
"""

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

from utils.Config import ParseSampleConfig
from control_v5.LightweightAdapter import ImageGECCOWrapper
from data.Transforms import to_pointset_optimal_transport

# ── editable defaults ─────────────────────────────────────────────────────────

CONFIG_PATH  = "config/GBN/config.json"
BASE_CKPT    = "config/GBN/model.ckpt"
WRAPPER_CKPT = ""   # filled via --wrapper
IMAGE_PATH   = ""   # filled via --image
OUTPUT_DIR   = "control_v5/sample_outputs"

GECCO_CH  = 8   # overridden from checkpoint
GRID_SIZE = 32
N_SAMPLES = 1
TIMESTEPS = 1000
DEVICE    = "cuda"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_condition(image_path: str, device: torch.device) -> torch.Tensor:
    """Load a grayscale condition image as (1, 1, H, W) float32 in [0, 1]."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def save_pointset(pts: np.ndarray, path: str, title: str = "") -> None:
    """Save an (N, 2) point set as a scatter plot PNG."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=1.0, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_sampling(diffusion, wrapper, n_samples, grid_size, timesteps, show_tqdm=True):
    """Run full reverse diffusion and return raw offsets."""
    shape = [n_samples, 2, grid_size, grid_size]
    diffusion.set_num_timesteps(timesteps)
    with torch.no_grad():
        raw = diffusion.p_sample_loop(shape, img=None, cond=None,
                                      with_tqdm=show_tqdm, with_sampling=True)
    diffusion.reset_timesteps()
    return raw


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",     default=CONFIG_PATH)
    parser.add_argument("--ckpt",       default=BASE_CKPT, help="Base diffusion checkpoint")
    parser.add_argument("--wrapper",    default=WRAPPER_CKPT, required=not WRAPPER_CKPT,
                        help="Path to gecco_wrapper_ep*.pt from training")
    parser.add_argument("--image",      default=IMAGE_PATH, required=not IMAGE_PATH,
                        help="Condition image (grayscale, any resolution)")
    parser.add_argument("--out",        default=OUTPUT_DIR)
    parser.add_argument("--grid-size",  type=int, default=GRID_SIZE,
                        help="Grid resolution G (model trained on 32 works on others too)")
    parser.add_argument("--n",          type=int, default=N_SAMPLES,
                        help="Number of independent samples to draw")
    parser.add_argument("--gecco-ch",  type=int, default=GECCO_CH,
                        help="Overrides checkpoint gecco_ch when provided explicitly")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--device",    default=DEVICE)
    args = parser.parse_args()

    if not args.wrapper:
        parser.error("--wrapper is required")
    if not args.image:
        parser.error("--image is required")

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # ── load base diffusion ───────────────────────────────────────────
    print(f"Loading base diffusion: {args.ckpt}")
    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    diffusion.eval()

    # ── load wrapper checkpoint ───────────────────────────────────────
    print(f"Loading wrapper checkpoint: {args.wrapper}")
    state = torch.load(args.wrapper, map_location=device)
    gecco_ch = int(state.get("gecco_ch", args.gecco_ch))
    print(f"  gecco_ch from checkpoint: {gecco_ch}")

    wrapper = ImageGECCOWrapper(diffusion.model, gecco_ch=gecco_ch).to(device)
    wrapper.load_state_dict(state["wrapper"])
    wrapper.eval()

    # ── load condition image ──────────────────────────────────────────
    print(f"Loading condition image: {args.image}")
    high_res = load_condition(args.image, device)                  # (1, 1, H, W)
    high_res_batch = high_res.expand(args.n, -1, -1, -1)           # (N, 1, H, W)
    print(f"  image shape (H×W): {high_res.shape[-2]}×{high_res.shape[-1]}")
    print(f"  grid size G: {args.grid_size}  →  {args.grid_size**2} points per sample")
    print(f"  generating {args.n} sample(s)")

    # ── swap model and sample ─────────────────────────────────────────
    original_model = diffusion.model
    wrapper.set_condition(high_res_batch)
    diffusion.model = wrapper

    raw = run_sampling(
        diffusion, wrapper,
        n_samples=args.n,
        grid_size=args.grid_size,
        timesteps=args.timesteps,
        show_tqdm=True,
    )

    diffusion.model = original_model

    # ── save outputs ──────────────────────────────────────────────────
    stem = os.path.splitext(os.path.basename(args.image))[0]
    for i, offsets in enumerate(raw):
        pts = to_pointset_optimal_transport(offsets.detach().cpu().numpy())   # (2, G, G)
        pts_flat = pts.reshape(2, -1).T                                        # (N, 2)

        npy_path = os.path.join(args.out, f"{stem}_sample{i:03d}.npy")
        np.save(npy_path, pts_flat)
        print(f"  -> {npy_path}")

        png_path = os.path.join(args.out, f"{stem}_sample{i:03d}.png")
        save_pointset(pts_flat, png_path, title=f"{stem} sample {i}")
        if HAS_MPL:
            print(f"  -> {png_path}")

    # ── side-by-side panel (condition + all samples) ──────────────────
    if HAS_MPL and args.n > 0:
        ncols = min(args.n + 1, 5)
        nrows = max((args.n + 1 + ncols - 1) // ncols, 1)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=150)
        axes_flat = np.array(axes).flatten() if nrows > 1 or ncols > 1 else [axes]

        axes_flat[0].imshow(high_res[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes_flat[0].set_title("Condition")
        axes_flat[0].axis("off")

        for i, offsets in enumerate(raw):
            pts = to_pointset_optimal_transport(offsets.detach().cpu().numpy()).reshape(2, -1).T
            ax = axes_flat[i + 1]
            ax.scatter(pts[:, 0], 1 - pts[:, 1], c="black", s=0.8, alpha=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_title(f"Sample {i}")
            ax.axis("off")

        for ax in axes_flat[args.n + 1:]:
            ax.axis("off")

        panel_path = os.path.join(args.out, f"{stem}_panel.png")
        plt.tight_layout()
        plt.savefig(panel_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> panel: {panel_path}")

    print("Done.")


if __name__ == "__main__":
    main()
