"""Overfit the image-GECCO early-fusion wrapper on a single (source, target) pair.

Usage (from project root):
    python control_early_fusion/test_overfit.py --steps 5000
    python control_early_fusion/test_overfit.py --steps 5000 --sample-index 42 --vis-every 200
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from data.Transforms import to_image_optimal_transport, to_pointset_optimal_transport
from control_early_fusion.LightweightAdapter import ImageGECCOWrapper
from utils.Config import ParseSampleConfig
from utils.stippling_metrics import compute_spacing_quality

# ── editable defaults ─────────────────────────────────────────────────────────

# DATA_ROOT  = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1"
# DATA_ROOT  = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1"
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_taksim"
SOURCE_DIR = os.path.join(DATA_ROOT, "source")
TARGET_DIR = os.path.join(DATA_ROOT, "target")
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH   = "config/GBN/model.ckpt"
OUTPUT_DIR  = "control_early_fusion/overfit_outputs"
WANDB_ENV   = ".env"

GECCO_CH         = 8
GRID_SIZE        = 32
N_POINTS         = GRID_SIZE ** 2
SAMPLE_TIMESTEPS = 1000

WANDB_ACTIVE   = False
STEPS          = 10000
SAMPLE_INDEX   = 0
LR             = 5e-4
VIS_EVERY      = 500
N_SAMPLES      = 2
SEED           = 42
RESAMPLE_JUMPS = 2
DEVICE         = "cuda"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_wandb_key():
    if os.path.exists(WANDB_ENV):
        with open(WANDB_ENV) as f:
            for line in f:
                if line.strip().startswith("WANDB_API_KEY"):
                    key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["WANDB_API_KEY"] = key
                    return True
    return False


def extract_points_from_target(img_path, n_points):
    """Detect dot centroids in a stippled target -> (N, 2) in [0, 1]."""
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
        pts = np.vstack([pts, rng.rand(n_points - len(pts), 2)])
    return pts


def load_source_image(img_path, device):
    """Load grayscale source as (1, 1, H, W) float32 in [0, 1]."""
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def save_vis(out_path, high_res_np, gt_offsets_np, pred_offsets_np, step):
    """Save 4-column panel: Source | GT | Predict | GT Quiver."""
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)

    axes[0].imshow(high_res_np[0, 0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Source"); axes[0].axis("off")

    gt_pts = to_pointset_optimal_transport(gt_offsets_np[0]).reshape(2, -1).T
    axes[1].scatter(gt_pts[:, 0], 1 - gt_pts[:, 1], c="black", s=1.5, alpha=0.8)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].set_aspect("equal")
    axes[1].set_title("GT"); axes[1].axis("off")

    pred_pts = to_pointset_optimal_transport(pred_offsets_np[0]).reshape(2, -1).T
    axes[2].scatter(pred_pts[:, 0], 1 - pred_pts[:, 1], c="black", s=1.5, alpha=0.8)
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1); axes[2].set_aspect("equal")
    axes[2].set_title(f"Predict @ step {step}"); axes[2].axis("off")

    dx, dy = gt_offsets_np[0][0], gt_offsets_np[0][1]
    n_g = dx.shape[-1]
    yy, xx = np.mgrid[0:n_g, 0:n_g]
    q = axes[3].quiver(xx, yy, dx, dy, np.sqrt(dx**2 + dy**2),
                       angles="xy", scale_units="xy", scale=1.0,
                       cmap="viridis", width=0.004)
    axes[3].invert_yaxis(); axes[3].set_aspect("equal")
    axes[3].set_title("GT Quiver")
    fig.colorbar(q, ax=axes[3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def sample_from_wrapper(diffusion, wrapper, high_res, grid_size, n_samples, timesteps, resample_jumps=2):
    """Run reverse diffusion conditioned on high_res, with resample jumps."""
    device = high_res.device
    high_res_batch = high_res.expand(n_samples, -1, -1, -1)
    orig_model = diffusion.model
    wrapper.set_condition(high_res_batch)
    diffusion.model = wrapper
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    shape = [n_samples, 2, grid_size, grid_size]
    img = torch.randn(shape, device=device)
    with torch.no_grad():
        for i in tqdm(reversed(range(timesteps)), total=timesteps, desc="sampling"):
            t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
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

    diffusion.model = orig_model
    diffusion.reset_timesteps()
    diffusion.train()
    return img


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gecco-ch",       type=int,   default=GECCO_CH)
    parser.add_argument("--steps",          type=int,   default=STEPS)
    parser.add_argument("--sample-index",   type=int,   default=SAMPLE_INDEX)
    parser.add_argument("--lr",             type=float, default=LR)
    parser.add_argument("--vis-every",      type=int,   default=VIS_EVERY)
    parser.add_argument("--n-samples",      type=int,   default=N_SAMPLES)
    parser.add_argument("--grid-size",       type=int, default=GRID_SIZE)
    parser.add_argument("--sample-timesteps", type=int, default=SAMPLE_TIMESTEPS)
    parser.add_argument("--resample-jumps",   type=int, default=RESAMPLE_JUMPS,
                        help="Resample jumps per timestep during eval sampling (0 = plain DDPM)")
    parser.add_argument("--seed",            type=int, default=SEED)
    parser.add_argument("--device",         default=DEVICE)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    n_points = args.grid_size ** 2

    # ── wandb init ────────────────────────────────────────────────────
    use_wandb = HAS_WANDB and WANDB_ACTIVE
    if use_wandb:
        load_wandb_key()
        run_name = datetime.now().strftime("v5-overfit-%Y%m%d-%H%M%S")
        wandb.init(project="Stipple-ControlNet", config=vars(args), name=run_name)
        print(f"wandb run: {run_name}")

    # ── pick single example ───────────────────────────────────────────
    source_files = sorted(os.listdir(SOURCE_DIR))
    if args.sample_index >= len(source_files):
        sys.exit(f"sample-index {args.sample_index} out of range "
                 f"(found {len(source_files)} files in source dir)")

    fname       = source_files[args.sample_index]
    stem        = os.path.splitext(fname)[0]
    source_path = os.path.join(SOURCE_DIR, fname)
    target_path = os.path.join(TARGET_DIR, fname)
    if not os.path.exists(target_path):
        sys.exit(f"Target not found: {target_path}")

    out_dir  = os.path.join(OUTPUT_DIR, stem)
    vis_dir  = os.path.join(out_dir, "vis")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    for d in (out_dir, vis_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    print(f"Overfitting on: {fname}")
    print(f"  source: {source_path}")
    print(f"  target: {target_path}")
    print(f"  output: {out_dir}")

    # ── load data ─────────────────────────────────────────────────────
    gt_points  = extract_points_from_target(target_path, n_points)
    gt_offsets = to_image_optimal_transport(gt_points)            # (2, G, G)
    x_0        = torch.from_numpy(gt_offsets).float().unsqueeze(0).to(device)  # (1,2,G,G)
    high_res   = load_source_image(source_path, device)           # (1,1,H,W)

    # Save reference images
    np.save(os.path.join(out_dir, "gt_offsets.npy"), gt_offsets)
    Image.fromarray(np.array(Image.open(source_path).convert("L"))).save(
        os.path.join(out_dir, "source.png"))
    Image.fromarray(np.array(Image.open(target_path).convert("L"))).save(
        os.path.join(out_dir, "target.png"))

    # ── load pretrained diffusion ─────────────────────────────────────
    diffusion = ParseSampleConfig(CONFIG_PATH)
    diffusion.load_state_dict(torch.load(CKPT_PATH, map_location="cpu")["diffu"])
    diffusion.to(device)

    denoiser      = diffusion.model
    num_timesteps = diffusion.num_timesteps

    # ── build wrapper ─────────────────────────────────────────────────
    wrapper = ImageGECCOWrapper(denoiser, gecco_ch=args.gecco_ch).to(device)
    wrapper.train()

    total_p = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    gecco_p = sum(p.numel() for p in wrapper.gecco_extractor.parameters())
    print(f"Trainable params: {total_p:,} (GECCO extractor: {gecco_p:,})")

    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=args.lr)

    gt_offsets_np = gt_offsets[np.newaxis]          # (1,2,G,G) for vis
    high_res_np   = high_res.cpu().numpy()           # (1,1,H,W) for vis

    if use_wandb:
        wandb.log({
            "source": wandb.Image(high_res_np[0, 0], caption="Source (condition)"),
        }, step=0)

    losses = []
    best_score = float("inf")
    best_ckpt  = None

    # ── overfit loop ─────────────────────────────────────────────────
    header = f"{'Step':>6}  {'Loss':>12}"
    print(header)
    print("-" * 22)

    for step in range(1, args.steps + 1):
        t     = torch.randint(0, num_timesteps, (1,), device=device)
        noise = torch.randn_like(x_0)
        offsets_t = diffusion.q_sample(x_0, t, noise)

        wrapper.set_condition(high_res)
        noise_pred = wrapper(offsets_t, t)

        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if use_wandb:
            wandb.log({"loss": loss_val}, step=step)

        if step % 50 == 0 or step == 1:
            avg = np.mean(losses[-50:])
            print(f"{step:>6}  {avg:>12.6f}")

        if step % args.vis_every == 0 or step == args.steps:
            wrapper.eval()
            raw = sample_from_wrapper(
                diffusion, wrapper, high_res, args.grid_size,
                n_samples=args.n_samples, timesteps=args.sample_timesteps,
                resample_jumps=args.resample_jumps,
            )
            raw_np = raw.detach().cpu().numpy()

            vis_path = os.path.join(vis_dir, f"step_{step:06d}.png")
            save_vis(vis_path, high_res_np, gt_offsets_np, raw_np, step)

            # Geometry metrics from first sample
            pts = to_pointset_optimal_transport(raw_np[0]).reshape(2, -1).T
            spacing = compute_spacing_quality(pts)
            cv    = float(spacing["nn_cv"])
            clumped_pct = float(spacing["clumped_pct"])
            score = cv + 5.0 * clumped_pct / 100.0
            print(f"  -> vis: {vis_path}")
            print(f"  -> geom  CV={cv:.4f}  Clumped={clumped_pct:.2f}%  Score={score:.4f}")

            if use_wandb:
                log_dict = {
                    "geo_cv": cv,
                    "geo_clumped_pct": clumped_pct,
                    "geo_score": score,
                }
                if HAS_MPL and os.path.exists(vis_path):
                    log_dict["vis"] = wandb.Image(vis_path)
                wandb.log(log_dict, step=step)

            if score < best_score:
                best_score = score
                new_best = os.path.join(ckpt_dir, f"best_ep{step:06d}_score{score:.3f}.pt")
                torch.save({
                    "wrapper": wrapper.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "gecco_ch": args.gecco_ch,
                    "best_score": best_score,
                }, new_best)
                if best_ckpt and os.path.exists(best_ckpt):
                    try:
                        os.remove(best_ckpt)
                    except OSError:
                        pass
                best_ckpt = new_best
                print(f"  -> new best: {new_best}")

            wrapper.train()

    # ── save final checkpoint ─────────────────────────────────────────
    final_ckpt = os.path.join(ckpt_dir, f"final_step{args.steps}.pt")
    torch.save({
        "wrapper": wrapper.state_dict(),
        "step": args.steps,
        "gecco_ch": args.gecco_ch,
    }, final_ckpt)
    print(f"Final checkpoint: {final_ckpt}")

    if use_wandb:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
