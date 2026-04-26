"""Train the image-GECCO early-fusion wrapper (control_early_fusion).

The model is a lightweight wrapper around the pretrained diffusion denoiser.
Two conv layers extract image features which are sampled at the current noisy
offset positions (GECCO) and concatenated to the denoiser's input -- no
parallel U-Net branch.

Usage (from project root):
    python control_early_fusion/train_control.py \\
        --config  config/GBN/config.json \\
        --ckpt    config/GBN/model.ckpt  \\
        --source  /path/to/source        \\
        --target  /path/to/target        \\
        --epochs  100                    \\
        --batch_size 16                  \\
        --lr 1e-4                        \\
        --out control_early_fusion/train_out
"""

import os
import sys
import argparse
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from control_early_fusion.LightweightAdapter import ImageGECCOWrapper
from control_early_fusion.DynamicStippleDataset import StippleDataset
from data.Transforms import to_image_optimal_transport, to_pointset_optimal_transport
from utils.stippling_metrics import geometric_validation_score

# -- editable defaults ------------------------------------------------

WANDB_ENV = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/.env"

CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH   = "config/GBN/model.ckpt"

# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/target"
# OUTPUT_DIR = "control_early_fusion/train_outputs_gecco"

# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/target"
# OUTPUT_DIR = "control_early_fusion/train_outputs_data_stress1"
# GRID_SIZE = 32
# VAL_SPLIT = 0.0
# EPOCHS = 1000
# SAVE_EVERY = 100


SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
OUTPUT_DIR = "control_v4/train_outputs_icons50_512"
GRID_SIZE = 32
VAL_SPLIT = 0.1
EPOCHS = 10000
SAVE_EVERY = 10


OFFSETS_DIR = ""
PRELOAD_RAM = False
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

GECCO_CH      = 8
# GRID_SIZE     = 32
TRUNCATION_RATIO  = 0.30
EVAL_TIMESTEPS    = 1000
RESAMPLE_JUMPS    = 2

WANDB_ACTIVE      = True
# EPOCHS            = 10000
BATCH_SIZE        = 16
LR                = 1e-4
# SAVE_EVERY        = 10
# VAL_SPLIT         = 0.0
DEVICE            = "cuda"
RESUME_LATEST     = True
NUM_WORKERS       = 4
PIN_MEMORY        = True

WANDB_VALID_IMAGES = 8
WANDB_TRAIN_IMAGES = 8

BEST_MAX_CV          = 1e9
BEST_MAX_CLUMPED_PCT = 100.0
GEOM_CLUMP_WEIGHT    = 1.0


# -- helpers ----------------------------------------------------------

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
    points = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)
    rng = np.random.RandomState(42)
    if len(points) > n_points:
        points = points[rng.choice(len(points), n_points, replace=False)]
    elif len(points) < n_points:
        points = np.vstack([points, rng.rand(n_points - len(points), 2)])
    return points


def ensure_offsets_dir(source_dir, target_dir, offsets_dir, grid_size):
    """Ensure offsets exist; auto-export from targets when missing."""
    resolved = offsets_dir.strip() if offsets_dir else ""
    if not resolved:
        resolved = os.path.join(os.path.dirname(os.path.normpath(target_dir)), "processed_offsets")
    os.makedirs(resolved, exist_ok=True)

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Target dir not found: {target_dir}")

    source_stems = set()
    for root, _, files in os.walk(source_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in VALID_EXT:
                continue
            rel = os.path.relpath(os.path.join(root, f), source_dir)
            source_stems.add(os.path.splitext(rel)[0])

    target_map = {}
    for root, _, files in os.walk(target_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in VALID_EXT:
                continue
            rel = os.path.relpath(os.path.join(root, fname), target_dir)
            stem = os.path.splitext(rel)[0]
            if stem in source_stems:
                target_map[stem] = os.path.join(root, fname)

    expected_stems = sorted(target_map.keys())
    if not expected_stems:
        raise RuntimeError("No matching source/target stems found.")

    existing_stems = {
        os.path.splitext(os.path.relpath(os.path.join(r, f), resolved))[0]
        for r, _, fs in os.walk(resolved) for f in fs if f.endswith(".npy")
    }
    missing = [s for s in expected_stems if s not in existing_stems]
    if not missing:
        print(f"Offsets complete: {len(expected_stems)}/{len(expected_stems)} in {resolved}")
        return resolved

    print(f"Exporting {len(missing)} missing offsets...")
    n_points = grid_size ** 2
    for stem in tqdm(missing, desc="Exporting offsets", unit="img"):
        pts = extract_points_from_target(target_map[stem], n_points)
        offsets = to_image_optimal_transport(pts)
        final = os.path.join(resolved, stem + ".npy")
        os.makedirs(os.path.dirname(final), exist_ok=True)
        np.save(final[:-4], offsets)  # np.save appends .npy
    print(f"Offsets export done: {len(expected_stems)}/{len(expected_stems)} in {resolved}")
    return resolved


def collate_fn(batch):
    """Collate variable-size images by padding to max H x W within the batch."""
    imgs, offsets = zip(*batch)
    max_h = max(t.shape[-2] for t in imgs)
    max_w = max(t.shape[-1] for t in imgs)
    padded = []
    for img in imgs:
        ph = max_h - img.shape[-2]
        pw = max_w - img.shape[-1]
        padded.append(F.pad(img, (0, pw, 0, ph), value=1.0).contiguous())
    return torch.stack(padded), torch.stack([o.contiguous() for o in offsets])


def save_val_panel(save_path, cond_batch, gt_offsets_batch, pred_offsets_batch, max_samples=4):
    """Save a 4-column panel: Condition | GT | Predict | GT Quiver."""
    if not HAS_MPL:
        return False
    n = min(max_samples, cond_batch.shape[0], gt_offsets_batch.shape[0], pred_offsets_batch.shape[0])
    if n <= 0:
        return False
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), dpi=150)
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n):
        cond = cond_batch[i, 0]
        gt_pts = to_pointset_optimal_transport(gt_offsets_batch[i]).reshape(2, -1).T
        pred_pts = to_pointset_optimal_transport(pred_offsets_batch[i]).reshape(2, -1).T

        axes[i, 0].imshow(cond, cmap="gray", vmin=0, vmax=1)
        if i == 0:
            axes[i, 0].set_title("Condition")
        axes[i, 0].axis("off")

        axes[i, 1].scatter(gt_pts[:, 0], 1 - gt_pts[:, 1], c="black", s=0.5, alpha=0.8)
        axes[i, 1].set_xlim(0, 1); axes[i, 1].set_ylim(0, 1); axes[i, 1].set_aspect("equal")
        if i == 0:
            axes[i, 1].set_title("GT")
        axes[i, 1].axis("off")

        axes[i, 2].scatter(pred_pts[:, 0], 1 - pred_pts[:, 1], c="black", s=0.5, alpha=0.8)
        axes[i, 2].set_xlim(0, 1); axes[i, 2].set_ylim(0, 1); axes[i, 2].set_aspect("equal")
        if i == 0:
            axes[i, 2].set_title("Predict")
        axes[i, 2].axis("off")

        dx = gt_offsets_batch[i][0]
        dy = gt_offsets_batch[i][1]
        n_g = dx.shape[-1]
        yy, xx = np.mgrid[0:n_g, 0:n_g]
        q = axes[i, 3].quiver(xx, yy, dx, dy, np.sqrt(dx**2 + dy**2),
                               angles="xy", scale_units="xy", scale=1.0,
                               cmap="viridis", width=0.004)
        axes[i, 3].invert_yaxis()
        axes[i, 3].set_aspect("equal")
        if i == 0:
            axes[i, 3].set_title("GT Quiver")
        fig.colorbar(q, ax=axes[i, 3], shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def sample_eval_batch(diffusion, wrapper, high_res_img, grid_size, device,
                      n_samples=4, timesteps=1000, resample_jumps=2,
                      show_tqdm=False, tqdm_desc="sampling"):
    """Run full reverse diffusion conditioned on high_res_img.

    Returns predicted offsets, shape (n_samples, 2, grid_size, grid_size).
    """
    original_model = diffusion.model
    wrapper.set_condition(high_res_img)
    diffusion.model = wrapper
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    shape = [n_samples, 2, grid_size, grid_size]
    with torch.no_grad():
        if resample_jumps <= 0:
            raw = diffusion.p_sample_loop(shape, img=None, cond=None,
                                          with_tqdm=show_tqdm, with_sampling=True)
        else:
            img = diffusion.noise_fn(shape).to(device)
            steps = reversed(range(diffusion.num_timesteps - 1))
            if show_tqdm:
                steps = tqdm(steps, total=diffusion.num_timesteps - 1,
                             desc=tqdm_desc, leave=False)
            for i in steps:
                t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
                for u in range(resample_jumps + 1):
                    img = diffusion.p_sample(img, cond=None, t=t_tensor,
                                             clip_denoised=diffusion.sample_clip,
                                             with_sampling=True)
                    if u == resample_jumps or i == 0:
                        break
                    beta_i = diffusion.betas[i]
                    img = (1 - beta_i).sqrt() * img + beta_i.sqrt() * torch.randn_like(img)
            raw = img

    diffusion.model = original_model
    diffusion.reset_timesteps()
    diffusion.train()
    return raw


# -- main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Paths
    parser.add_argument("--config",   default=CONFIG_PATH)
    parser.add_argument("--ckpt",     default=CKPT_PATH)
    parser.add_argument("--source",   default=SOURCE_DIR)
    parser.add_argument("--target",   default=TARGET_DIR)
    parser.add_argument("--offsets",  default=OFFSETS_DIR,
                        help="Dir of .npy offsets; empty -> auto-export from --target")
    parser.add_argument("--out",      default=OUTPUT_DIR)
    parser.add_argument("--preload-ram", action="store_true", default=PRELOAD_RAM)

    # Model
    parser.add_argument("--gecco-ch",  type=int,   default=GECCO_CH)
    parser.add_argument("--grid-size", type=int,   default=GRID_SIZE)
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO)
    parser.add_argument("--eval-timesteps",   type=int,   default=EVAL_TIMESTEPS)
    parser.add_argument("--resample-jumps",   type=int,   default=RESAMPLE_JUMPS)

    # Training
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--val-split",   type=float, default=VAL_SPLIT)
    parser.add_argument("--save_every",  type=int,   default=SAVE_EVERY)
    parser.add_argument("--resume-latest", action=argparse.BooleanOptionalAction,
                        default=RESUME_LATEST)
    parser.add_argument("--device",  default=DEVICE)

    # W&B
    parser.add_argument("--wandb-valid-images", type=int, default=WANDB_VALID_IMAGES)
    parser.add_argument("--wandb-train-images", type=int, default=WANDB_TRAIN_IMAGES)

    # Geometry gating
    parser.add_argument("--geom-clump-weight",     type=float, default=GEOM_CLUMP_WEIGHT)
    parser.add_argument("--best-max-cv",           type=float, default=BEST_MAX_CV)
    parser.add_argument("--best-max-clumped-pct",  type=float, default=BEST_MAX_CLUMPED_PCT)

    args = parser.parse_args()

    if not (0.0 < args.truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0, 1]")
    if args.save_every <= 0:
        raise ValueError("--save_every must be >= 1")

    args.offsets = ensure_offsets_dir(args.source, args.target, args.offsets, args.grid_size)

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    checkpoints_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # -- wandb --------------------------------------------------------
    use_wandb = WANDB_ACTIVE
    if use_wandb:
        try:
            import wandb
            load_wandb_key()
            run_name = datetime.now().strftime("v5-gecco-%Y%m%d-%H%M%S")
            wandb.init(project="Stipple-ControlNet", name=run_name, config=vars(args))
            wandb.define_metric("epoch")
            for m in ("metrics/train_loss", "metrics/valid_loss", "metrics/geo_cv",
                      "metrics/geo_clumped_pct", "metrics/geo_score"):
                wandb.define_metric(m, step_metric="epoch")
            wandb.define_metric("visual/*", step_metric="epoch")
            print(f"wandb run: {run_name}")
        except ImportError:
            print("wandb not installed; logging disabled")
            use_wandb = False

    # -- load pretrained diffusion ------------------------------------
    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)

    denoiser = diffusion.model
    num_timesteps = diffusion.num_timesteps
    truncation_cutoff = max(1, int(num_timesteps * args.truncation_ratio))

    # -- build wrapper ------------------------------------------------
    wrapper = ImageGECCOWrapper(denoiser, gecco_ch=args.gecco_ch).to(device)
    wrapper.train()

    total_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    gecco_params = sum(p.numel() for p in wrapper.gecco_extractor.parameters())
    print(f"Wrapper total trainable params : {total_params:,}")
    print(f"  of which GECCO extractor     : {gecco_params:,}")
    print(f"  of which denoiser            : {total_params - gecco_params:,}")
    print(f"GECCO channels                 : {args.gecco_ch}")
    print(f"Truncation ratio               : {args.truncation_ratio:.3f}  ({truncation_cutoff}/{num_timesteps} steps)")

    # -- dataset ------------------------------------------------------
    full_dataset = StippleDataset(args.source, args.offsets)
    if len(full_dataset) == 0:
        raise RuntimeError(
            "StippleDataset has 0 samples. Check that source images and "
            "offset .npy files share matching relative stems."
        )

    val_len   = min(max(int(len(full_dataset) * args.val_split), 0), max(len(full_dataset) - 1, 0))
    train_len = len(full_dataset) - val_len

    all_idx = torch.randperm(len(full_dataset),
                             generator=torch.Generator().manual_seed(42)).tolist()
    train_fnames = [full_dataset.filenames[i] for i in all_idx[:train_len]]
    val_fnames   = [full_dataset.filenames[i] for i in all_idx[train_len:]]

    train_dataset = StippleDataset(args.source, args.offsets,
                                   filenames=train_fnames, preload_ram=args.preload_ram)
    val_dataset   = StippleDataset(args.source, args.offsets,
                                   filenames=val_fnames,   preload_ram=args.preload_ram) if val_len > 0 else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)                    if val_dataset is not None else None

    print(f"Dataset split: train={train_len}, val={val_len}")

    # -- optimizer ----------------------------------------------------
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=args.lr)

    # -- optional resume ----------------------------------------------
    start_epoch    = 0
    global_step    = 0
    best_geom_score = float("inf")
    best_geom_ckpt_path = None
    last_geom = {"cv": float("nan"), "clumped_pct": float("nan"), "score": float("nan")}
    epoch_history = []
    train_loss_history = []
    val_loss_history   = []

    ckpt_re = re.compile(r"^gecco_wrapper_ep(\d+)\.pt$")
    if args.resume_latest:
        latest_path, latest_ep = None, -1
        for fname in os.listdir(checkpoints_dir):
            m = ckpt_re.match(fname)
            if m and int(m.group(1)) > latest_ep:
                latest_ep = int(m.group(1))
                latest_path = os.path.join(checkpoints_dir, fname)
        if latest_path is None:
            print("Resume requested but no checkpoint found -- starting from scratch.")
        else:
            state = torch.load(latest_path, map_location=device)
            wrapper.load_state_dict(state["wrapper"])
            optimizer.load_state_dict(state["optimizer"])
            global_step     = int(state.get("global_step", 0))
            best_geom_score = float(state.get("best_geom_score", best_geom_score))
            start_epoch     = int(state.get("epoch", latest_ep - 1)) + 1
            print(f"Resumed: {latest_path}  start_epoch={start_epoch}  step={global_step}")

    # -- training loop ------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        should_save = ((epoch + 1) % args.save_every == 0) or ((epoch + 1) == args.epochs)
        epoch_loss = 0.0
        preview_imgs = preview_offsets = None

        wrapper.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False)
        for high_res, x_0 in train_pbar:
            high_res = high_res.to(device)
            x_0      = x_0.to(device)

            if preview_imgs is None:
                k = max(1, min(args.wandb_train_images, high_res.shape[0]))
                preview_imgs    = high_res[:k].detach()
                preview_offsets = x_0[:k].detach()

            t = torch.randint(0, truncation_cutoff, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)
            offsets_t = diffusion.q_sample(x_0, t, noise)

            wrapper.set_condition(high_res)
            noise_pred = wrapper(offsets_t, t)

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)

        # -- train prediction panel -----------------------------------
        if should_save and args.wandb_train_images > 0 and preview_imgs is not None:
            wrapper.eval()
            train_pred = sample_eval_batch(
                diffusion, wrapper, preview_imgs, args.grid_size, device,
                n_samples=preview_imgs.shape[0], timesteps=args.eval_timesteps,
                resample_jumps=args.resample_jumps, show_tqdm=True,
                tqdm_desc=f"Epoch {epoch+1} [train-predict]",
            )
            panel_path = os.path.join(args.out, f"train_panel_ep{epoch+1}.png")
            if save_val_panel(panel_path, preview_imgs.cpu().numpy(),
                              preview_offsets.cpu().numpy(), train_pred.cpu().numpy(),
                              max_samples=args.wandb_train_images):
                print(f"  -> train panel: {panel_path}")
                if use_wandb:
                    wandb.log({"epoch": epoch+1, "visual/train": wandb.Image(panel_path)},
                              step=epoch+1)
            wrapper.train()

        # -- validation loop ------------------------------------------
        val_avg_loss = None
        val_preview_imgs = val_preview_offsets = None
        pred_raw_for_geom = None

        if val_loader is not None:
            wrapper.eval()
            val_loss_sum = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False)
            with torch.no_grad():
                for high_res, x_0 in val_pbar:
                    high_res = high_res.to(device)
                    x_0      = x_0.to(device)

                    if val_preview_imgs is None:
                        k = min(args.wandb_valid_images, high_res.shape[0])
                        val_preview_imgs    = high_res[:k].detach()
                        val_preview_offsets = x_0[:k].detach()

                    t = torch.randint(0, truncation_cutoff, (x_0.shape[0],), device=device)
                    noise = torch.randn_like(x_0)
                    offsets_t = diffusion.q_sample(x_0, t, noise)

                    wrapper.set_condition(high_res)
                    noise_pred = wrapper(offsets_t, t)

                    val_loss = F.mse_loss(noise_pred, noise)
                    val_loss_sum += val_loss.item()
                    val_pbar.set_postfix(loss=f"{val_loss.item():.6f}")

            val_avg_loss = val_loss_sum / max(len(val_loader), 1)
            wrapper.train()

            # val prediction panel
            if should_save and args.wandb_valid_images > 0 and val_preview_imgs is not None:
                wrapper.eval()
                pred_raw_for_geom = sample_eval_batch(
                    diffusion, wrapper, val_preview_imgs, args.grid_size, device,
                    n_samples=val_preview_imgs.shape[0], timesteps=args.eval_timesteps,
                    resample_jumps=args.resample_jumps, show_tqdm=True,
                    tqdm_desc=f"Epoch {epoch+1} [predict]",
                )
                panel_path = os.path.join(args.out, f"val_panel_ep{epoch+1}.png")
                if save_val_panel(panel_path, val_preview_imgs.cpu().numpy(),
                                  val_preview_offsets.cpu().numpy(),
                                  pred_raw_for_geom.cpu().numpy(),
                                  max_samples=args.wandb_valid_images):
                    print(f"  -> val panel: {panel_path}")
                    if use_wandb:
                        wandb.log({"epoch": epoch+1, "visual/valid": wandb.Image(panel_path)},
                                  step=epoch+1)
                wrapper.train()

            # geometry-gated best checkpoint
            if should_save and val_preview_imgs is not None:
                wrapper.eval()
                if pred_raw_for_geom is None:
                    pred_raw_for_geom = sample_eval_batch(
                        diffusion, wrapper, val_preview_imgs[:1], args.grid_size, device,
                        n_samples=1, timesteps=args.eval_timesteps,
                        resample_jumps=args.resample_jumps,
                    )
                pred_pts = []
                for raw in pred_raw_for_geom:
                    pts = to_pointset_optimal_transport(raw.detach().cpu().numpy())
                    pred_pts.append(pts.reshape(pts.shape[0], -1).T)
                geom = geometric_validation_score(pred_pts, clump_weight=args.geom_clump_weight)
                last_geom = geom
                geom_score = float(geom["score"])
                print(f"  -> geom CV={geom['cv']:.4f} Clumped={geom['clumped_pct']:.2f}% Score={geom_score:.4f}")

                if (geom["cv"] <= args.best_max_cv and
                        geom["clumped_pct"] <= args.best_max_clumped_pct and
                        geom_score < best_geom_score):
                    best_geom_score = geom_score
                    bname = (f"best_wrapper_ep{epoch+1:04d}"
                             f"_score{best_geom_score:.3f}"
                             f"_cv{geom['cv']:.3f}"
                             f"_clumped{geom['clumped_pct']:.2f}.pt")
                    new_best = os.path.join(checkpoints_dir, bname)
                    torch.save({
                        "wrapper": wrapper.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "gecco_ch": args.gecco_ch,
                        "best_geom_score": best_geom_score,
                    }, new_best)
                    if best_geom_ckpt_path and os.path.exists(best_geom_ckpt_path):
                        try:
                            os.remove(best_geom_ckpt_path)
                        except OSError:
                            pass
                    best_geom_ckpt_path = new_best
                    print(f"  -> new best-geom checkpoint: {new_best}")

                if use_wandb:
                    wandb.log({
                        "epoch": epoch+1,
                        "metrics/geo_cv": float(geom["cv"]),
                        "metrics/geo_clumped_pct": float(geom["clumped_pct"]),
                        "metrics/geo_score": geom_score,
                    }, step=epoch+1)
                wrapper.train()

        # -- logging --------------------------------------------------
        if use_wandb:
            epoch_history.append(epoch + 1)
            train_loss_history.append(float(avg_loss))
            val_loss_history.append(float(val_avg_loss) if val_avg_loss is not None else float("nan"))
            wandb.log({"epoch": epoch+1, "metrics/train_loss": avg_loss}, step=epoch+1)
            if val_avg_loss is not None:
                wandb.log({"epoch": epoch+1, "metrics/valid_loss": val_avg_loss}, step=epoch+1)
            if len(epoch_history) > 0:
                wandb.log({"epoch": epoch+1,
                           "visual/compare": wandb.plot.line_series(
                               xs=epoch_history,
                               ys=[train_loss_history, val_loss_history],
                               keys=["train", "valid"],
                               title="Train vs Valid Loss",
                               xname="epoch")}, step=epoch+1)

        if val_avg_loss is None:
            print(f"Epoch {epoch:>4d}  |  train loss = {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch:>4d}  |  train loss = {avg_loss:.6f}  |  val loss = {val_avg_loss:.6f}")

        # -- periodic checkpoint --------------------------------------
        if should_save:
            save_path = os.path.join(checkpoints_dir, f"gecco_wrapper_ep{epoch+1}.pt")
            torch.save({
                "wrapper": wrapper.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "gecco_ch": args.gecco_ch,
                "best_geom_score": best_geom_score,
                "cv_score": float(last_geom["cv"]),
                "clumped_score": float(last_geom["clumped_pct"]),
            }, save_path)
            print(f"  -> saved {save_path}")

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
