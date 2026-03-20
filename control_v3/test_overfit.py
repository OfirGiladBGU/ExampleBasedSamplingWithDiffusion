"""Overfit the ControlNet V3.2 on a single (source, target) pair.

V3.2 uses static-only conditioning (5ch hint encoder, no dynamic sampling)
with Kaiming-initialized injection layers to avoid the zero-init trap.

Usage (from project root):
    python control_v3/test_overfit.py --steps 5000
    python control_v3/test_overfit.py --steps 5000 --sample-index 42 --vis-every 200
"""

import argparse
import json
import os
import sys
from datetime import datetime

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
from control_v3.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from utils.Config import ParseSampleConfig
from utils.stippling_metrics import visualize_overfit_metrics

# ── paths ────────────────────────────────────────────────────────────
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/small_target_image"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_1024"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_taksim"
SOURCE_DIR = os.path.join(DATA_ROOT, "source")
TARGET_DIR = os.path.join(DATA_ROOT, "target")
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"
OUTPUT_DIR = "control_v3/overfit_outputs"
GRID_SIZE = 32
N_POINTS = GRID_SIZE ** 2

# ── default run parameters (edit here for quick experiments) ───────
STEPS = 1000
SAMPLE_INDEX = 0
LR = 5e-4
VIS_EVERY = 500
SAMPLE_TIMESTEPS = 1000

ENABLE_GECCO = True
MIN_SNR_GAMMA = 5.0
RESAMPLE_JUMPS = 2

N_SAMPLES = 2
SEED = 42
DEVICE = "cuda"
EXPORT_GT_OFFSET = True

WANDB_ENV = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/.env"
WANDB_ACTIVE = False


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


def export_gt_offset_artifacts(out_dir, gt_offsets):
    """Save GT offset diagnostics inside out_dir/gt_offset/."""
    gt_dir = os.path.join(out_dir, "gt_offset")
    os.makedirs(gt_dir, exist_ok=True)

    # 1) Raw target tensor view: offset magnitude on the 32x32 grid.
    mag = np.sqrt(gt_offsets[0] ** 2 + gt_offsets[1] ** 2)
    if mag.max() > 0:
        mag_u8 = np.round((mag / mag.max()) * 255.0).astype(np.uint8)
    else:
        mag_u8 = np.zeros_like(mag, dtype=np.uint8)
    Image.fromarray(mag_u8).save(os.path.join(gt_dir, "gt_offsets_magnitude_32x32.png"))

    # 2) Inverse OT mapping then exact 32x32 occupancy (no interpolation).
    pts_grid = to_pointset_optimal_transport(gt_offsets)
    pts = pts_grid.reshape(2, -1).T

    n = gt_offsets.shape[-1]
    clipped = np.clip(pts, 0.0, 1.0 - 1e-12)
    ij = np.floor(clipped * n).astype(np.int64)
    counts = np.zeros((n, n), dtype=np.int32)
    for x_idx, y_idx in ij:
        counts[y_idx, x_idx] += 1

    Image.fromarray(((counts > 0).astype(np.uint8) * 255), mode="L").save(
        os.path.join(gt_dir, "gt_points_binary_32x32.png")
    )

    # 3) Quiver view of offset vectors for intuitive direction/magnitude reading.
    if HAS_MPL:
        dx, dy = gt_offsets[0], gt_offsets[1]
        yy, xx = np.mgrid[0:n, 0:n]
        fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
        q = ax.quiver(
            xx,
            yy,
            dx,
            dy,
            np.sqrt(dx * dx + dy * dy),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            cmap="viridis",
            width=0.004,
        )
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("grid x")
        ax.set_ylabel("grid y")
        ax.set_title("Offset Vector Field")
        fig.colorbar(q, ax=ax, label="|offset|")
        plt.tight_layout()
        plt.savefig(os.path.join(gt_dir, "offset_quiver.png"))
        plt.close()
    else:
        print("  WARNING: matplotlib unavailable, skipping offset_quiver.png")

    print(f"  -> saved gt_offset artifacts: {gt_dir}/")


def sample_from_model(diffusion, control_net, denoiser, high_res, target_density,
                      device, n_samples=2, timesteps=200, resample_jumps=0):
    """Run the full reverse diffusion loop and return point sets."""
    from tqdm import tqdm as _tqdm
    controlled = DynamicControlledDenoiser(denoiser, control_net)
    controlled.set_condition(high_res, target_density)

    orig_model = diffusion.model
    diffusion.model = controlled
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    shape = [n_samples, 2, GRID_SIZE, GRID_SIZE]
    with torch.no_grad():
        if resample_jumps == 0:
            raw = diffusion.p_sample_loop(shape, img=None, cond=None,
                                          with_tqdm=True, with_sampling=True)
        else:
            img = diffusion.noise_fn(shape).to(device)
            for i in _tqdm(reversed(range(diffusion.num_timesteps - 1)),
                           total=diffusion.num_timesteps - 1,
                           desc="sampling"):
                t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
                for u in range(resample_jumps + 1):
                    img = diffusion.p_sample(img, cond=None, t=t_tensor,
                                             clip_denoised=diffusion.sample_clip,
                                             with_sampling=True)
                    if u == resample_jumps or i == 0:
                        break
                    # Re-noise: bring x_{i-1} back to x_i
                    beta_i = diffusion.betas[i]
                    noise = torch.randn_like(img)
                    img = (1.0 - beta_i).sqrt() * img + beta_i.sqrt() * noise
            raw = img
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


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--vis-every", type=int, default=VIS_EVERY,
                        help="Visualise & sample every N steps")
    parser.add_argument("--sample-timesteps", type=int, default=SAMPLE_TIMESTEPS,
                        help="Diffusion timesteps when sampling")
    parser.add_argument(
        "--enable-gecco",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_GECCO,
        help="Enable GECCO dynamic feature sampling in the control hint path",
    )
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=MIN_SNR_GAMMA,
        help="Gamma for Min-SNR loss weighting (0 disables)",
    )
    parser.add_argument(
        "--resample-jumps",
        type=int,
        default=RESAMPLE_JUMPS,
        help="RePaint-style micro-loops per timestep during sampling (0=disabled)",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--export-gt-offset",
        type=bool,
        default=EXPORT_GT_OFFSET,
        help="Export GT offset diagnostics into out_dir/gt_offset",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── wandb init ───────────────────────────────────────────────────
    use_wandb = HAS_WANDB and WANDB_ACTIVE
    if use_wandb:
        load_wandb_key()
        run_name = datetime.now().strftime("v3-overfit-%Y%m%d-%H%M%S")
        wandb.init(
            project="Stipple-ControlNet",
            config=vars(args),
            name=run_name,
        )
        print(f"wandb run name: {run_name}")

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

    out_dir = os.path.join(OUTPUT_DIR, stem)
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

    if args.export_gt_offset:
        export_gt_offset_artifacts(out_dir, gt_offsets)

    if use_wandb:
        wandb.log({
            "source": wandb.Image(source_np, caption="Source (condition)"),
            "target": wandb.Image(target_np, caption="Target GT"),
        }, step=0)

    # ── load pretrained diffusion + build Dynamic ControlNet V3 ──────
    diffusion = ParseSampleConfig(CONFIG_PATH)
    diffusion.load_state_dict(
        torch.load(CKPT_PATH, map_location="cpu")["diffu"])
    diffusion.to(device)

    denoiser = diffusion.model
    num_timesteps = diffusion.num_timesteps

    for p in denoiser.parameters():
        p.requires_grad = False
    denoiser.eval()

    control_net = DynamicControlNet(denoiser, enable_gecco=args.enable_gecco).to(device)
    control_net.train()

    trainable = sum(p.numel() for p in control_net.parameters()
                    if p.requires_grad)
    print(f"  DynamicControlNet V3 trainable params: {trainable:,}")
    print(f"  GECCO dynamic features enabled: {args.enable_gecco}")
    print(f"  Min-SNR gamma: {args.min_snr_gamma}")
    print(f"  Resample jumps (RePaint): {args.resample_jumps}")

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

        per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none")
        per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))

        alphas_cumprod_t = diffusion.alphas_cumprod.gather(0, t)
        snr = alphas_cumprod_t / torch.clamp(1.0 - alphas_cumprod_t, min=1e-8)
        min_snr_weight = torch.clamp(snr, max=args.min_snr_gamma) / torch.clamp(snr, min=1e-8)

        loss = (per_sample_mse * min_snr_weight).mean()

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
                timesteps=args.sample_timesteps,
                resample_jumps=args.resample_jumps,
            )
            vis_path = os.path.join(out_dir, f"vis_step{step:05d}.png")
            saved = visualize_overfit_metrics(
                source_np, target_np, gt_points,
                list(pts), vis_path, step=step,
                gt_offsets=gt_offsets,
            )
            np.save(os.path.join(out_dir, f"points_step{step:05d}.npy"), pts)
            print(f"  -> saved visualisation: {vis_path}")

            if use_wandb and saved:
                wandb.log({
                    "comparison": wandb.Image(saved, caption=f"Step {step}"),
                }, step=step)

            control_net.train()
            denoiser.eval()

    # ── save final weights ───────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "dynamic_controlnet_v3_overfit.pt")
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
        ax.set_title(f"V3 Overfit Loss ({fname})")
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
