"""Train the Dynamic ControlNet V3.7 for stipple generation.

V3.7 training mirrors the overfit winner components:
    - AdaptiveGateInjection (V3.2 revert)
    - optional GECCO dynamic conditioning (enabled by default)
    - Min-SNR-gamma weighted denoising loss
    - continuous (non-binarized) target density

Optional intermediate evaluation sampling supports full-schedule
RePaint-style resampling via ``--resample-jumps``.

Usage (from project root):
    python control_v3/train_control.py \
        --config  config/GBN/config.json \
        --ckpt    config/GBN/model.ckpt \
        --source  /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/source \
        --offsets /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets \
        --epochs  100 \
        --batch_size 16 \
        --lr 1e-4 \
        --out control_v3/control_out
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
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

from utils.Config import ParseSampleConfig
from control_v3.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_v3.DynamicStippleDataset import DynamicStippleDataset
from data.Transforms import to_image_optimal_transport, to_pointset_optimal_transport

# ── default globals (edit here for quick experiments) ───────────────
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/target"
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/target"
# If empty, offsets are auto-exported (if needed) to a default processed_offsets folder.
OFFSETS_DIR = ""
OUTPUT_DIR = "control_v3/train_outputs"
GRID_SIZE = 32

# EPOCHS = 100
EPOCHS = 2
BATCH_SIZE = 16
LR = 1e-4
SAVE_EVERY = 10
DEVICE = "cuda"

ENABLE_GECCO = True
MIN_SNR_GAMMA = 5.0

EVAL_EVERY = 0
EVAL_BATCH = 4
EVAL_TIMESTEPS = 1000
RESAMPLE_JUMPS = 2

NUM_WORKERS = 4
PIN_MEMORY = True
VAL_SPLIT = 0.1

WANDB_ENV = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/.env"
WANDB_ACTIVE = True
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


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
    """Detect dot centroids in a stippled target and return (N, 2) in [0, 1]."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    inv = 255 - img_np
    binary = (inv > 127).astype(np.uint8)

    from scipy import ndimage
    labelled, n_labels = ndimage.label(binary)
    centroids = ndimage.center_of_mass(binary, labelled, range(1, n_labels + 1))

    h, w = img_np.shape
    points = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)

    rng = np.random.RandomState(42)
    if len(points) > n_points:
        points = points[rng.choice(len(points), n_points, replace=False)]
    elif len(points) < n_points:
        deficit = n_points - len(points)
        points = np.vstack([points, rng.rand(deficit, 2)])

    return points


def ensure_offsets_dir(source_dir, target_dir, offsets_dir, grid_size):
    """Ensure offsets exist; auto-export from targets when missing/empty."""
    if offsets_dir and offsets_dir.strip():
        resolved_offsets_dir = offsets_dir
    else:
        resolved_offsets_dir = os.path.join(
            os.path.dirname(os.path.normpath(target_dir)), "processed_offsets"
        )

    os.makedirs(resolved_offsets_dir, exist_ok=True)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"Offsets dir is empty and target dir was not found: {target_dir}"
        )

    source_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(source_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    }

    # Build target stem -> filepath map for files that have matching source stems.
    target_map = {}
    for fname in sorted(os.listdir(target_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXT:
            continue
        stem = os.path.splitext(fname)[0]
        if stem in source_stems:
            target_map[stem] = os.path.join(target_dir, fname)

    expected_stems = sorted(target_map.keys())
    if not expected_stems:
        raise RuntimeError(
            "No matching source/target stems found for offset export. "
            "Check SOURCE_DIR/TARGET_DIR filename stems."
        )

    existing_files = [
        f for f in os.listdir(resolved_offsets_dir)
        if f.endswith(".npy")
    ]
    existing_stems = {os.path.splitext(f)[0] for f in existing_files}

    missing_stems = [stem for stem in expected_stems if stem not in existing_stems]

    if not missing_stems:
        print(
            f"Offsets already complete: {len(expected_stems)} / {len(expected_stems)} "
            f"in {resolved_offsets_dir}"
        )
        return resolved_offsets_dir

    print("Offsets export is incomplete. Resuming export from target images...")

    # Safety: re-export the most recently written existing file in case last write was corrupted.
    reexport_stem = None
    if existing_files:
        latest_file = max(
            existing_files,
            key=lambda f: os.path.getmtime(os.path.join(resolved_offsets_dir, f)),
        )
        latest_stem = os.path.splitext(latest_file)[0]
        if latest_stem in target_map:
            reexport_stem = latest_stem

    export_stems = list(missing_stems)
    if reexport_stem is not None and reexport_stem not in export_stems:
        export_stems.append(reexport_stem)

    print(
        f"  -> existing: {len(existing_stems)} | missing: {len(missing_stems)} | "
        f"to export now: {len(export_stems)}"
    )

    n_points = grid_size ** 2
    exported = 0
    for stem in tqdm(export_stems, desc="Exporting offsets", unit="img"):
        pts = extract_points_from_target(target_map[stem], n_points)
        offsets = to_image_optimal_transport(pts)

        # Write through a temporary file then atomically replace target file.
        final_path = os.path.join(resolved_offsets_dir, stem + ".npy")
        tmp_prefix = final_path + ".tmp"
        tmp_path = tmp_prefix + ".npy"
        np.save(tmp_prefix, offsets)
        os.replace(tmp_path, final_path)
        exported += 1

    final_existing_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(resolved_offsets_dir)
        if f.endswith(".npy")
    }
    final_missing = [stem for stem in expected_stems if stem not in final_existing_stems]
    if final_missing:
        raise RuntimeError(
            f"Offset export incomplete after resume: missing {len(final_missing)} files."
        )

    print(
        f"  -> exported/re-exported {exported} files; "
        f"offsets complete: {len(expected_stems)} / {len(expected_stems)} in {resolved_offsets_dir}"
    )
    return resolved_offsets_dir


def sample_eval_batch(diffusion, denoiser, control_net, high_res_img, target_density,
                      device, n_samples=4, timesteps=1000, resample_jumps=2,
                      show_tqdm=False, tqdm_desc="sampling"):
    """Sample offset grids for intermediate eval with optional resampling."""
    controlled = DynamicControlledDenoiser(denoiser, control_net)
    controlled.set_condition(high_res_img, target_density)

    original_model = diffusion.model
    diffusion.model = controlled
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    h, w = target_density.shape[-2], target_density.shape[-1]
    shape = [n_samples, 2, h, w]
    with torch.no_grad():
        if resample_jumps == 0:
            raw = diffusion.p_sample_loop(shape, img=None, cond=None,
                                          with_tqdm=show_tqdm, with_sampling=True)
        else:
            img = diffusion.noise_fn(shape).to(device)
            iter_steps = reversed(range(diffusion.num_timesteps - 1))
            if show_tqdm:
                iter_steps = tqdm(
                    iter_steps,
                    total=diffusion.num_timesteps - 1,
                    desc=tqdm_desc,
                    leave=False,
                )
            for i in iter_steps:
                t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
                for u in range(resample_jumps + 1):
                    img = diffusion.p_sample(img, cond=None, t=t_tensor,
                                             clip_denoised=diffusion.sample_clip,
                                             with_sampling=True)
                    if u == resample_jumps or i == 0:
                        break
                    beta_i = diffusion.betas[i]
                    noise = torch.randn_like(img)
                    img = (1.0 - beta_i).sqrt() * img + beta_i.sqrt() * noise
            raw = img

    diffusion.model = original_model
    diffusion.reset_timesteps()
    diffusion.train()
    return raw


def save_val_panel(save_path, cond_batch, gt_offsets_batch, pred_offsets_batch):
    """Save a 4-column panel per validation sample.

    Columns: Condition | GT | Predict | GT Offset Quiver
    """
    if not HAS_MPL:
        print("matplotlib unavailable; skipping validation panel export")
        return False

    n = min(4, cond_batch.shape[0], gt_offsets_batch.shape[0], pred_offsets_batch.shape[0])
    if n <= 0:
        return False

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), dpi=150)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        cond = cond_batch[i, 0]
        gt_offsets = gt_offsets_batch[i]
        pred_offsets = pred_offsets_batch[i]

        gt_pts_grid = to_pointset_optimal_transport(gt_offsets)
        gt_pts = gt_pts_grid.reshape(2, -1).T

        pred_pts_grid = to_pointset_optimal_transport(pred_offsets)
        pred_pts = pred_pts_grid.reshape(2, -1).T

        ax = axes[i, 0]
        ax.imshow(cond, cmap="gray", vmin=0.0, vmax=1.0)
        if i == 0:
            ax.set_title("Condition")
        ax.axis("off")

        ax = axes[i, 1]
        ax.scatter(gt_pts[:, 0], 1.0 - gt_pts[:, 1], c="black", s=0.5, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        if i == 0:
            ax.set_title("GT")
        ax.axis("off")

        ax = axes[i, 2]
        ax.scatter(pred_pts[:, 0], 1.0 - pred_pts[:, 1], c="black", s=0.5, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        if i == 0:
            ax.set_title("Predict")
        ax.axis("off")

        ax = axes[i, 3]
        n_grid = gt_offsets.shape[-1]
        yy, xx = np.mgrid[0:n_grid, 0:n_grid]
        dx, dy = gt_offsets[0], gt_offsets[1]
        mag = np.sqrt(dx * dx + dy * dy)
        q = ax.quiver(
            xx,
            yy,
            dx,
            dy,
            mag,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            cmap="viridis",
            width=0.004,
        )
        ax.invert_yaxis()
        ax.set_aspect("equal")
        if i == 0:
            ax.set_title("GT Offset Quiver")
        ax.tick_params(labelsize=6)
        fig.colorbar(q, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--source",
                        default=SOURCE_DIR,
                        help="Dir of full-resolution grayscale source images")
    parser.add_argument("--target",
                        default=TARGET_DIR,
                        help="Dir of stippled target images (used to auto-export offsets if needed)")
    parser.add_argument("--offsets",
                        default=OFFSETS_DIR,
                        help="Dir of .npy offset files; if empty/missing, offsets are exported from --target")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE,
                        help="Grid resolution for offset export and dataset loading")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--enable-gecco",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_GECCO,
        help="Enable GECCO dynamic feature sampling in control hint path",
    )
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=MIN_SNR_GAMMA,
        help="Gamma for Min-SNR loss weighting (0 disables)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=EVAL_EVERY,
        help="Run intermediate eval sampling every N epochs (0 disables)",
    )
    parser.add_argument("--eval-batch", type=int, default=EVAL_BATCH,
                        help="Number of samples for each intermediate eval")
    parser.add_argument("--eval-timesteps", type=int, default=EVAL_TIMESTEPS,
                        help="Timesteps used in intermediate eval sampling")
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS,
                        help="RePaint micro-loops per timestep during eval sampling")
    parser.add_argument("--out", default=OUTPUT_DIR,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT,
                        help="Validation split ratio in [0,1). Example: 0.1 = 10% val")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    args.offsets = ensure_offsets_dir(args.source, args.target, args.offsets, args.grid_size)

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # ── wandb ────────────────────────────────────────────────────────
    use_wandb = WANDB_ACTIVE
    if use_wandb:
        try:
            import wandb
            load_wandb_key()
            run_name = datetime.now().strftime("v3-train-%Y%m%d-%H%M%S")
            wandb.init(
                project="Stipple-ControlNet",
                name=run_name,
                config=vars(args),
            )
            print(f"wandb run name: {run_name}")
        except ImportError:
            print("wandb not installed, logging disabled")
            use_wandb = False

    # ── load pretrained diffusion model ──────────────────────────────
    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    diffusion.eval()

    denoiser = diffusion.model
    num_timesteps = diffusion.num_timesteps

    for p in denoiser.parameters():
        p.requires_grad = False

    # ── build Dynamic ControlNet V3 ──────────────────────────────────
    control_net = DynamicControlNet(
        denoiser,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
    ).to(device)
    control_net.train()

    trainable = sum(p.numel() for p in control_net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in denoiser.parameters())
    print(f"Trainable DynamicControlNet V3 params : {trainable:,}")
    print(f"Frozen denoiser params                : {frozen:,}")
    print(f"GECCO dynamic features enabled        : {args.enable_gecco}")
    print(f"Min-SNR gamma                         : {args.min_snr_gamma}")
    print(f"Eval resample-jumps                   : {args.resample_jumps}")

    # ── dataset ──────────────────────────────────────────────────────
    dataset = DynamicStippleDataset(args.source, args.offsets, grid_size=args.grid_size)
    val_len = int(len(dataset) * args.val_split)
    val_len = min(max(val_len, 0), max(len(dataset) - 1, 0))
    train_len = len(dataset) - val_len

    if val_len > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    print(f"Dataset split: train={train_len}, val={val_len}")

    # ── optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(control_net.parameters(), lr=args.lr)

    # ── training loop ────────────────────────────────────────────────
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        preview_high_res = None
        preview_target_density = None

        control_net.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False)
        for high_res_img, target_density, x_0 in train_pbar:
            high_res_img = high_res_img.to(device)
            target_density = target_density.to(device)
            x_0 = x_0.to(device)

            if preview_high_res is None:
                preview_high_res = high_res_img[:1].detach()
                preview_target_density = target_density[:1].detach()

            t = torch.randint(0, num_timesteps, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)
            offsets_t = diffusion.q_sample(x_0, t, noise)

            controls = control_net(offsets_t, t, high_res_img, target_density)
            noise_pred = denoiser(offsets_t, t, controls=controls)

            per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none")
            per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))

            if args.min_snr_gamma > 0:
                alphas_cumprod_t = diffusion.alphas_cumprod.gather(0, t)
                snr = alphas_cumprod_t / torch.clamp(1.0 - alphas_cumprod_t, min=1e-8)
                min_snr_weight = torch.clamp(snr, max=args.min_snr_gamma) / torch.clamp(snr, min=1e-8)
                loss = (per_sample_mse * min_snr_weight).mean()
            else:
                loss = per_sample_mse.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(control_net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if use_wandb:
                wandb.log({"loss/step": loss.item()}, step=global_step)
            global_step += 1
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)

        # Validation loop with tqdm.
        val_avg_loss = None
        val_preview_high_res = None
        val_preview_target_density = None
        val_preview_offsets = None
        if val_loader is not None:
            control_net.eval()
            val_loss_sum = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False)
            with torch.no_grad():
                for high_res_img, target_density, x_0 in val_pbar:
                    high_res_img = high_res_img.to(device)
                    target_density = target_density.to(device)
                    x_0 = x_0.to(device)

                    if val_preview_high_res is None:
                        keep = min(4, high_res_img.shape[0])
                        val_preview_high_res = high_res_img[:keep].detach()
                        val_preview_target_density = target_density[:keep].detach()
                        val_preview_offsets = x_0[:keep].detach()

                    t = torch.randint(0, num_timesteps, (x_0.shape[0],), device=device)
                    noise = torch.randn_like(x_0)
                    offsets_t = diffusion.q_sample(x_0, t, noise)

                    controls = control_net(offsets_t, t, high_res_img, target_density)
                    noise_pred = denoiser(offsets_t, t, controls=controls)

                    per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none")
                    per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))

                    if args.min_snr_gamma > 0:
                        alphas_cumprod_t = diffusion.alphas_cumprod.gather(0, t)
                        snr = alphas_cumprod_t / torch.clamp(1.0 - alphas_cumprod_t, min=1e-8)
                        min_snr_weight = torch.clamp(snr, max=args.min_snr_gamma) / torch.clamp(snr, min=1e-8)
                        val_loss = (per_sample_mse * min_snr_weight).mean()
                    else:
                        val_loss = per_sample_mse.mean()

                    val_loss_sum += val_loss.item()
                    val_pbar.set_postfix(loss=f"{val_loss.item():.6f}")

            val_avg_loss = val_loss_sum / max(len(val_loader), 1)
            control_net.train()

            # Export per-epoch qualitative val panel (4 samples, 4 columns).
            control_net.eval()
            pred_raw = sample_eval_batch(
                diffusion,
                denoiser,
                control_net,
                val_preview_high_res,
                val_preview_target_density,
                device,
                n_samples=val_preview_high_res.shape[0],
                timesteps=args.eval_timesteps,
                resample_jumps=args.resample_jumps,
                show_tqdm=True,
                tqdm_desc=f"Epoch {epoch+1}/{args.epochs} [predict]",
            )
            panel_path = os.path.join(args.out, f"val_panel_ep{epoch+1}.png")
            saved = save_val_panel(
                panel_path,
                val_preview_high_res.cpu().numpy(),
                val_preview_offsets.cpu().numpy(),
                pred_raw.cpu().numpy(),
            )
            if saved:
                print(f"  -> saved validation panel: {panel_path}")
                if use_wandb:
                    wandb.log({"val/panel": wandb.Image(panel_path)}, step=global_step)
            control_net.train()

        if use_wandb:
            log_payload = {"loss/epoch": avg_loss, "epoch": epoch}
            if val_avg_loss is not None:
                log_payload["loss/val_epoch"] = val_avg_loss
            wandb.log(log_payload, step=global_step)
        if val_avg_loss is None:
            print(f"Epoch {epoch:>4d}  |  train loss = {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch:>4d}  |  train loss = {avg_loss:.6f}  |  val loss = {val_avg_loss:.6f}")

        # Optional intermediate eval sampling with full resampling workflow.
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            control_net.eval()
            eval_raw = sample_eval_batch(
                diffusion,
                denoiser,
                control_net,
                preview_high_res,
                preview_target_density,
                device,
                n_samples=args.eval_batch,
                timesteps=args.eval_timesteps,
                resample_jumps=args.resample_jumps,
                show_tqdm=True,
                tqdm_desc=f"Epoch {epoch+1}/{args.epochs} [eval-sample]",
            )
            eval_path = os.path.join(args.out, f"eval_offsets_ep{epoch+1}.pt")
            torch.save(eval_raw.cpu(), eval_path)
            print(f"  -> saved eval samples: {eval_path}")
            if use_wandb:
                wandb.log({"eval/sample_path": eval_path}, step=global_step)
            control_net.train()

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.out, f"dynamic_controlnet_v3_ep{epoch+1}.pt")
            torch.save({
                "control_net": control_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            }, save_path)
            print(f"  -> saved {save_path}")

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
