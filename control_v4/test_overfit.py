"""Overfit Dynamic ControlNet V4 on a single (source, target) pair.

Usage (from project root):
    python control_v4/test_overfit.py --steps 5000
    python control_v4/test_overfit.py --steps 5000 --sample-index 42 --vis-every 200
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
from control_v4.conditioning import build_condition_tensors_from_image
from control_v4.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_v4.smart_init import build_smart_init_from_image, add_noise_at_t
from utils.Config import ParseSampleConfig
from utils.stippling_metrics import compute_spacing_quality, visualize_overfit_metrics

# ── paths ────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\Users\User\PycharmProjects\ExampleBasedSamplingWithDiffusion\training\monkey"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_1024"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_taksim"
SOURCE_DIR = os.path.join(DATA_ROOT, "source")
TARGET_DIR = os.path.join(DATA_ROOT, "target")
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"
OUTPUT_DIR = "control_v4/overfit_outputs"
GRID_SIZE = 32
N_POINTS = GRID_SIZE ** 2

# ── default run parameters (edit here for quick experiments) ───────
STEPS = 10000
SAMPLE_INDEX = 0
LR = 5e-4
VIS_EVERY = 500
SAMPLE_TIMESTEPS = 1000
TRUNCATION_RATIO = 0.30

ENABLE_GECCO = True
MIN_SNR_GAMMA = 5.0
RESAMPLE_JUMPS = 2
SDF_TRUNCATE_PX = 8.0
USE_SDF = True

GEOM_CLUMP_WEIGHT = 5.0

N_SAMPLES = 2
SEED = 42
SMART_INIT_SEED = 42
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


def load_condition(img_path, grid_size, device, sdf_truncate_px=0.0):
    """Load source image and return image, density, and SDF condition tensors."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0
    return build_condition_tensors_from_image(
        img_np,
        grid_size,
        device,
        sdf_truncate_px=sdf_truncate_px,
    )


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


def sample_from_model(diffusion, control_net, denoiser, high_res, high_res_sdf,
                      target_density, target_sdf, smart_init_grid, smart_init_offsets,
                      device, n_samples=2, timesteps=200, resample_jumps=0, truncation_ratio=0.30):
    """Run truncated reverse diffusion from Smart Init (SDEdit-style)."""
    from tqdm import tqdm as _tqdm
    controlled = DynamicControlledDenoiser(denoiser, control_net)
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)

    orig_model = diffusion.model
    diffusion.model = controlled
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    shape = [n_samples, 2, GRID_SIZE, GRID_SIZE]
    t_start = int(np.clip(int(diffusion.num_timesteps * truncation_ratio), 1, diffusion.num_timesteps - 1))

    x_init = smart_init_offsets
    if x_init.shape[0] != n_samples:
        x_init = x_init.expand(n_samples, -1, -1, -1).contiguous()

    alpha_t = diffusion.alphas_cumprod[t_start]
    img = add_noise_at_t(x_init, alpha_t)

    for i in _tqdm(reversed(range(t_start)),
                   total=t_start,
                   desc="sampling"):
        t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
        for u in range(resample_jumps + 1):
            with torch.no_grad():
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


def geometric_validation_score(pointsets, clump_weight=5.0):
    """Compute aggregate geometric validation score from predicted point sets."""
    cvs = []
    clumped_pcts = []
    per_sample_scores = []

    for pts in pointsets:
        spacing = compute_spacing_quality(pts)
        cv = float(spacing["nn_cv"])
        clumped_pct = float(spacing["clumped_pct"])
        score = cv + clump_weight * (clumped_pct / 100.0)

        cvs.append(cv)
        clumped_pcts.append(clumped_pct)
        per_sample_scores.append(score)

    return {
        "cv": float(np.mean(cvs)) if cvs else 0.0,
        "clumped_pct": float(np.mean(clumped_pcts)) if clumped_pcts else 0.0,
        "score": float(np.mean(per_sample_scores)) if per_sample_scores else 0.0,
    }


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
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX,
                        help="Truncate signed distance magnitudes before max-normalization (0 disables)")
    parser.add_argument(
        "--use-sdf",
        action=argparse.BooleanOptionalAction,
        default=USE_SDF,
        help="Pass real SDF channels to the model (--no-use-sdf zeroes them out for ablation)",
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
    vis_dir = os.path.join(out_dir, "vis")
    points_dir = os.path.join(out_dir, "points")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(points_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── prepare GT offset tensor on the fly ──────────────────────────
    print(f"Example: {fname}")
    print(f"  source: {source_path}")
    print(f"  target: {target_path}")

    gt_points = extract_points_from_image(target_path, N_POINTS)
    gt_offsets = to_image_optimal_transport(gt_points)
    x_0 = torch.from_numpy(gt_offsets).float().unsqueeze(0).to(device)

    high_res, target_density, high_res_sdf, target_sdf = load_condition(
        source_path,
        GRID_SIZE,
        device,
        sdf_truncate_px=args.sdf_truncate_px,
    )

    source_img_01 = np.array(Image.open(source_path).convert("L"), dtype=np.float32) / 255.0
    smart_points, smart_offsets_np, smart_grid_np = build_smart_init_from_image(
        source_img_01,
        grid_size=GRID_SIZE,
        n_points=N_POINTS,
        seed=SMART_INIT_SEED,
    )
    smart_init_grid = torch.from_numpy(smart_grid_np).unsqueeze(0).to(device)
    smart_init_offsets = torch.from_numpy(smart_offsets_np).unsqueeze(0).to(device)

    if not args.use_sdf:
        high_res_sdf = torch.zeros_like(high_res_sdf)
        target_sdf = torch.zeros_like(target_sdf)

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
    truncation_cutoff = max(1, int(num_timesteps * TRUNCATION_RATIO))

    for p in denoiser.parameters():
        p.requires_grad = False
    denoiser.eval()

    control_net = DynamicControlNet(denoiser, grid_size=GRID_SIZE, enable_gecco=args.enable_gecco).to(device)
    control_net.train()

    trainable = sum(p.numel() for p in control_net.parameters()
                    if p.requires_grad)
    print(f"  DynamicControlNet V3 trainable params: {trainable:,}")
    print(f"  GECCO dynamic features enabled: {args.enable_gecco}")
    print(f"  Min-SNR gamma: {args.min_snr_gamma}")
    print(f"  Resample jumps (RePaint): {args.resample_jumps}")
    print(f"  SDF truncation (px): {args.sdf_truncate_px}")
    print(f"  SDF conditioning enabled: {args.use_sdf}")
    print(f"  Truncation ratio: {TRUNCATION_RATIO:.3f} -> cutoff {truncation_cutoff}/{num_timesteps}")

    optimizer = torch.optim.AdamW(control_net.parameters(), lr=args.lr)
    best_val_score = float("inf")
    best_ckpt_path = None
    last_geom = {"cv": None, "clumped_pct": None, "score": None}

    # ── training loop ────────────────────────────────────────────────
    print(f"\n{'Step':>6}  {'Loss':>12}")
    print("-" * 22)

    losses = []
    for step in range(1, args.steps + 1):
        t = torch.randint(0, truncation_cutoff, (1,), device=device)
        noise = torch.randn_like(x_0)
        offsets_t = diffusion.q_sample(x_0, t, noise)

        controls = control_net(offsets_t, t, high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
        noise_pred = denoiser(offsets_t, t, controls=controls)

        per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none")
        per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))

        alphas_cumprod_t = diffusion.alphas_cumprod.gather(0, t)
        snr = alphas_cumprod_t / torch.clamp(1.0 - alphas_cumprod_t, min=1e-8)
        min_snr_weight = torch.clamp(snr, max=args.min_snr_gamma) / torch.clamp(snr, min=1e-8)

        denoise_loss = (per_sample_mse * min_snr_weight).mean()
        loss = denoise_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(control_net.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if use_wandb:
            wandb.log({
                "loss": loss_val,
                "loss_denoise": float(denoise_loss.item()),
            }, step=step)

        if step % 50 == 0 or step == 1:
            print(f"{step:6d}  {loss_val:12.6f}")

        if step % args.vis_every == 0 or step == args.steps:
            control_net.eval()
            pts, raw = sample_from_model(
                diffusion,
                control_net,
                denoiser,
                high_res,
                high_res_sdf,
                target_density,
                target_sdf,
                smart_init_grid,
                smart_init_offsets,
                device,
                n_samples=args.n_samples,
                timesteps=args.sample_timesteps,
                resample_jumps=args.resample_jumps,
                truncation_ratio=TRUNCATION_RATIO,
            )
            vis_path = os.path.join(vis_dir, f"vis_step{step:05d}.png")
            saved = visualize_overfit_metrics(
                source_np, target_np, gt_points,
                list(pts), vis_path, step=step,
                gt_offsets=gt_offsets,
            )
            np.save(os.path.join(points_dir, f"points_step{step:05d}.npy"), pts)
            print(f"  -> saved visualisation: {vis_path}")

            geom = geometric_validation_score(pts, clump_weight=GEOM_CLUMP_WEIGHT)
            last_geom = geom
            print(
                "  -> geometry "
                f"CV={geom['cv']:.4f} | Clumped={geom['clumped_pct']:.2f}% | "
                f"Score={geom['score']:.4f}"
            )

            checkpoint = {
                "step": step,
                "model_state_dict": control_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_score": min(best_val_score, geom["score"]),
                "current_val_score": geom["score"],
                "cv_score": geom["cv"],
                "clumped_score": geom["clumped_pct"],
                "loss": float(loss_val),
                "loss_denoise": float(denoise_loss.item()),
                "example": fname,
                "config": vars(args),
            }

            latest_ckpt_path = os.path.join(ckpt_dir, "latest_controlnet.pt")
            torch.save(checkpoint, latest_ckpt_path)

            if geom["score"] < best_val_score:
                best_val_score = geom["score"]
                best_filename = (
                    f"best_controlnet_step{step:05d}"
                    f"_score{best_val_score:.3f}"
                    f"_cv{geom['cv']:.3f}"
                    f"_clumped{geom['clumped_pct']:.2f}.pt"
                )
                new_best_path = os.path.join(ckpt_dir, best_filename)
                checkpoint["best_val_score"] = best_val_score
                torch.save(checkpoint, new_best_path)

                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    try:
                        os.remove(best_ckpt_path)
                    except OSError:
                        pass
                best_ckpt_path = new_best_path
                print(f"  -> new best checkpoint: {new_best_path}")

            if use_wandb and saved:
                wandb.log({
                    "comparison": wandb.Image(saved, caption=f"Step {step}"),
                    "geom/cv": geom["cv"],
                    "geom/clumped_pct": geom["clumped_pct"],
                    "geom/score": geom["score"],
                }, step=step)

            control_net.train()
            denoiser.eval()

    # ── save final weights ───────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "dynamic_controlnet_v3_overfit.pt")
    torch.save({
        "model_state_dict": control_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": args.steps,
        "loss_history": losses,
        "best_val_score": best_val_score,
        "cv_score": last_geom["cv"],
        "clumped_score": last_geom["clumped_pct"],
        "current_val_score": last_geom["score"],
        "example": fname,
        "config": vars(args),
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
        "best_val_score": None if best_val_score == float("inf") else float(best_val_score),
        "last_cv": last_geom["cv"],
        "last_clumped_pct": last_geom["clumped_pct"],
        "last_geom_score": last_geom["score"],
        "best_checkpoint": best_ckpt_path,
        "latest_checkpoint": os.path.join(ckpt_dir, "latest_controlnet.pt"),
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
