"""Overfit the Dynamic ControlNet V2 on a single (source, target) pair.

Trains on one example for many steps, periodically samples from the
diffusion model, and saves comparison visualizations + weights.

Usage (from project root):
    python control_v2/test_overfit.py --steps 2000
    python control_v2/test_overfit.py --steps 5000 --sample-index 42 --vis-every 200
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from data.Transforms import to_image_optimal_transport, to_pointset_optimal_transport
from control_v2.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from utils.Config import ParseSampleConfig

# ── paths ────────────────────────────────────────────────────────────
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024"
SOURCE_DIR = os.path.join(DATA_ROOT, "source")
TARGET_DIR = os.path.join(DATA_ROOT, "target")
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"
GRID_SIZE = 32
N_POINTS = GRID_SIZE ** 2
WANDB_ENV = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/.env"


# ── helpers ──────────────────────────────────────────────────────────
def load_wandb_key():
    if os.path.exists(WANDB_ENV):
        with open(WANDB_ENV) as f:
            for line in f:
                line = line.strip()
                if line.startswith("WANDB_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["WANDB_API_KEY"] = key
                    return True
    return False


def extract_points_from_image(img_path, n_points):
    """Detect dot centroids in a stippled image -> (N, 2) in [0, 1]."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    inv = 255 - img_np
    binary = (inv > 127).astype(np.uint8)

    from scipy import ndimage
    labelled, n_labels = ndimage.label(binary)
    centroids = ndimage.center_of_mass(binary, labelled, range(1, n_labels + 1))

    h, w = img_np.shape
    pts = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)

    rng = np.random.RandomState(42)
    if len(pts) > n_points:
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    elif len(pts) < n_points:
        deficit = n_points - len(pts)
        pts = np.vstack([pts, rng.rand(deficit, 2)])
        print(f"  WARNING: padded {deficit} random points "
              f"(only {len(pts) - deficit} detected)")
    return pts


def load_condition(img_path, grid_size, device):
    """Load source image and return (high_res, target_density) tensors."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0

    high_res = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    target_density = F.interpolate(
        high_res, size=(grid_size, grid_size), mode="area"
    )
    return high_res, target_density


def sample_from_model(diffusion, control_net, denoiser, high_res, target_density,
                      device, n_samples=4, timesteps=200):
    """Run the full reverse diffusion loop and return point sets."""
    controlled = DynamicControlledDenoiser(denoiser, control_net)
    controlled.set_condition(high_res, target_density)

    orig_model = diffusion.model
    diffusion.model = controlled
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    shape = [n_samples, 2, GRID_SIZE, GRID_SIZE]
    with torch.no_grad():
        raw = diffusion.p_sample_loop(shape, img=None, cond=None,
                                      with_tqdm=False, with_sampling=True)
    raw_np = raw.cpu().numpy()

    diffusion.model = orig_model
    diffusion.reset_timesteps()
    diffusion.train()

    pointsets = []
    for s in raw_np:
        ps = to_pointset_optimal_transport(s)
        ps = ps.reshape(ps.shape[0], np.prod(ps.shape[1:])).T
        pointsets.append(ps)
    return np.array(pointsets), raw_np


def visualize(source_img, target_img, pointsets, save_path, step=None):
    """Save a comparison figure: source | target GT | generated samples."""
    if not HAS_MPL:
        return None
    n_samples = min(len(pointsets), 4)
    fig, axes = plt.subplots(1, 2 + n_samples,
                             figsize=(5 * (2 + n_samples), 5))

    axes[0].imshow(source_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Source (condition)")
    axes[0].axis("off")

    axes[1].imshow(target_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Target GT (stippled)")
    axes[1].axis("off")

    for i in range(n_samples):
        pts = pointsets[i]
        axes[2 + i].scatter(pts[:, 0], 1 - pts[:, 1],
                            c="black", s=0.5, alpha=0.8)
        axes[2 + i].set_xlim(0, 1)
        axes[2 + i].set_ylim(0, 1)
        axes[2 + i].set_aspect("equal")
        axes[2 + i].set_facecolor("white")
        title = f"Sample {i}"
        if step is not None:
            title += f" (step {step})"
        axes[2 + i].set_title(title)
        axes[2 + i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vis-every", type=int, default=500,
                        help="Visualise & sample every N steps")
    parser.add_argument("--sample-timesteps", type=int, default=200,
                        help="Diffusion timesteps when sampling")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── wandb init ───────────────────────────────────────────────────
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        load_wandb_key()
        wandb.init(
            project="dynamic-controlnet-overfit",
            config=vars(args),
            name=f"v2-overfit-idx{args.sample_index}-{args.steps}steps",
        )

    # ── pick the single example ──────────────────────────────────────
    source_files = sorted(os.listdir(SOURCE_DIR))
    if args.sample_index >= len(source_files):
        sys.exit(f"sample-index {args.sample_index} out of range "
                 f"(dataset has {len(source_files)} files)")

    fname = source_files[args.sample_index]
    stem = os.path.splitext(fname)[0]
    source_path = os.path.join(SOURCE_DIR, fname)
    target_path = os.path.join(TARGET_DIR, fname)

    if not os.path.exists(target_path):
        sys.exit(f"Target not found: {target_path}")

    out_dir = os.path.join("control_v2", "overfit_outputs", stem)
    os.makedirs(out_dir, exist_ok=True)

    # ── prepare GT offset tensor on the fly ──────────────────────────
    print(f"Example: {fname}")
    print(f"  source: {source_path}")
    print(f"  target: {target_path}")

    gt_points = extract_points_from_image(target_path, N_POINTS)
    gt_offsets = to_image_optimal_transport(gt_points)
    x_0 = torch.from_numpy(gt_offsets).float().unsqueeze(0).to(device)

    high_res, target_density = load_condition(source_path, GRID_SIZE, device)

    source_np = np.array(Image.open(source_path).convert("L"))
    target_np = np.array(Image.open(target_path).convert("L"))

    Image.fromarray(source_np).save(os.path.join(out_dir, "source.png"))
    Image.fromarray(target_np).save(os.path.join(out_dir, "target.png"))
    np.save(os.path.join(out_dir, "gt_offsets.npy"), gt_offsets)

    if use_wandb:
        wandb.log({
            "source": wandb.Image(source_np, caption="Source (condition)"),
            "target": wandb.Image(target_np, caption="Target GT"),
        }, step=0)

    # ── load pretrained diffusion + build Dynamic ControlNet ─────────
    diffusion = ParseSampleConfig(CONFIG_PATH)
    diffusion.load_state_dict(
        torch.load(CKPT_PATH, map_location="cpu")["diffu"])
    diffusion.to(device)

    denoiser = diffusion.model
    num_timesteps = diffusion.num_timesteps

    for p in denoiser.parameters():
        p.requires_grad = False

    control_net = DynamicControlNet(denoiser).to(device)
    control_net.train()

    trainable = sum(p.numel() for p in control_net.parameters()
                    if p.requires_grad)
    print(f"  DynamicControlNet trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(control_net.parameters(), lr=args.lr)

    # ── training loop ────────────────────────────────────────────────
    print(f"\n{'Step':>6}  {'Loss':>12}")
    print("-" * 22)

    losses = []
    for step in range(1, args.steps + 1):
        t = torch.randint(0, num_timesteps, (1,), device=device)
        noise = torch.randn_like(x_0)
        offsets_t = diffusion.q_sample(x_0, t, noise)

        controls = control_net(offsets_t, t, high_res, target_density)
        noise_pred = denoiser(offsets_t, t, controls=controls)

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(control_net.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if use_wandb:
            wandb.log({"loss": loss_val}, step=step)

        if step % 50 == 0 or step == 1:
            print(f"{step:6d}  {loss_val:12.6f}")

        if step % args.vis_every == 0 or step == args.steps:
            control_net.eval()
            pts, raw = sample_from_model(
                diffusion, control_net, denoiser, high_res, target_density,
                device, n_samples=args.n_samples,
                timesteps=args.sample_timesteps)
            vis_path = os.path.join(out_dir, f"vis_step{step:05d}.png")
            saved = visualize(source_np, target_np, pts, vis_path, step=step)
            np.save(os.path.join(out_dir, f"points_step{step:05d}.npy"), pts)
            print(f"  -> saved visualisation: {vis_path}")

            if use_wandb and saved:
                wandb.log({
                    "comparison": wandb.Image(saved, caption=f"Step {step}"),
                }, step=step)

            control_net.train()

    # ── save final weights ───────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "dynamic_controlnet_overfit.pt")
    torch.save({
        "control_net": control_net.state_dict(),
        "step": args.steps,
        "loss_history": losses,
        "example": fname,
    }, ckpt_path)
    print(f"  -> saved weights: {ckpt_path}")

    # ── save loss curve ──────────────────────────────────────────────
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(losses) + 1), losses, linewidth=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"V2 Overfit Loss ({fname})")
        ax.grid(True, alpha=0.3)
        loss_path = os.path.join(out_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"  -> saved loss curve: {loss_path}")
        if use_wandb:
            wandb.log({"loss_curve": wandb.Image(loss_path)})

    # ── save metrics ─────────────────────────────────────────────────
    metrics = {
        "example": fname,
        "steps": args.steps,
        "final_loss": float(losses[-1]),
        "min_loss": float(min(losses)),
        "mean_loss_last100": float(np.mean(losses[-100:])),
        "lr": args.lr,
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if use_wandb:
        wandb.log(metrics)
        wandb.finish()

    print(f"\nFinal loss: {losses[-1]:.6f}  (min: {min(losses):.6f})")
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
