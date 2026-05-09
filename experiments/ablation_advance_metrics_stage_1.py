"""Ablation export script.

Run one active configuration at a time (by uncommenting its block), then export
validation predictions for all checkpoints in numerical epoch order.

Output layout:
    OUTPUT_DIR/{RESULTS_DIR}/epoch_{epoch_id}/{image_stem}.npy
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_v4.conditioning import build_condition_tensors_from_image
from control_v4.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_v4.smart_init import (
    add_noise_at_t,
    generate_smart_init_points_from_density,
    render_smart_init_grid,
    smart_init_points_to_offsets,
)
from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig


CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"

# Vanilla
WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_vanilla/checkpoints"
RESULTS_DIR = "vanilla"
GRID_SIZE = 32
ENABLE_GECCO = False
ENABLE_ADAPTIVE_GATE_INJECTION = False
EVAL_TIMESTEPS = 1000
TRUNCATION_RATIO = None
RESAMPLE_JUMPS = 0
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False

# GECCO
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_gecco/checkpoints"
# RESULTS_DIR = "gecco"
# GRID_SIZE = 32
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = False
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = None
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Adaptive gate injection
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_agi/checkpoints"
# RESULTS_DIR = "agi"
# GRID_SIZE = 32
# ENABLE_GECCO = False
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = None
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Full
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints"
# RESULTS_DIR = "full"
# GRID_SIZE = 32
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = None
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Full + SDEdit
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints"
# RESULTS_DIR = "sdedit"
# GRID_SIZE = 32
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = 0.3
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Full + SDEdit + resample jumps
# WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints"
# RESULTS_DIR = "sdedit_resample"
# GRID_SIZE = 32
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = 0.3
# RESAMPLE_JUMPS = 2
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
OUTPUT_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/ablation_advance_metrics"

SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SPLAT_SIGMA_PX = 0.5

DEVICE = "cuda"
SPLIT_SEED = 42
VAL_SPLIT = 0.1


def parse_args():
    p = argparse.ArgumentParser(description="Export ablation predictions to .npy per epoch")
    p.add_argument("--output", default=OUTPUT_DIR, help="Base output folder for exports")
    p.add_argument("--source", default=SOURCE_DIR, help="Source images folder (validation pool)")
    p.add_argument("--target", default=TARGET_DIR, help="Target images folder (ground truth)")
    p.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Fraction for validation split")
    p.add_argument("--seed", type=int, default=SPLIT_SEED, help="Deterministic seed for split")
    p.add_argument("--checkpoints", default="all", help="'all' or comma-separated filename substrings")
    p.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--base-ckpt", default=CKPT_PATH)
    p.add_argument("--timesteps", type=int, default=EVAL_TIMESTEPS)
    p.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO)
    p.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    p.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    p.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)
    p.add_argument("--smart-init-splat-sigma-px", type=float, default=SMART_INIT_SPLAT_SIGMA_PX)
    return p.parse_args()


def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    p = Path(folder)
    return [str(f) for f in sorted(p.rglob("*")) if f.suffix.lower() in exts]


def select_validation_images(all_images, val_frac, seed):
    imgs = sorted(all_images)
    n_total = len(imgs)
    val_len = int(n_total * float(val_frac))
    val_len = min(max(val_len, 0), max(n_total - 1, 0))
    train_len = n_total - val_len

    all_indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(int(seed)),
    ).tolist()
    val_indices = all_indices[train_len:]
    return [imgs[i] for i in val_indices]


def backup_validation_images(val_images, out_base, target_dir):
    out_base_p = Path(out_base)
    val_data_dir = out_base_p / "validation_data"
    source_backup_dir = val_data_dir / "source"
    target_backup_dir = val_data_dir / "target"
    manifest_path = out_base_p / "validation_manifest.json"

    if manifest_path.exists():
        return source_backup_dir, target_backup_dir

    source_backup_dir.mkdir(parents=True, exist_ok=True)
    target_backup_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    
    # Build a map of target filenames to full paths (recursive search)
    target_path_obj = Path(target_dir)
    target_map = {}
    for target_file in target_path_obj.rglob("*"):
        if target_file.is_file() and target_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            target_map[target_file.name] = target_file
    
    for img in val_images:
        src = Path(img)
        # Backup source image
        src_dst = source_backup_dir / src.name
        if not src_dst.exists():
            shutil.copy2(src, src_dst)
        
        # Backup corresponding target image (search by filename)
        if src.name in target_map:
            tgt_src = target_map[src.name]
            tgt_dst = target_backup_dir / src.name
            if not tgt_dst.exists():
                shutil.copy2(tgt_src, tgt_dst)
        
        manifest.append(src.name)
    
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return source_backup_dir, target_backup_dir


def find_checkpoints(weights_dir, pattern_filter=None):
    p = Path(weights_dir)
    candidates = []
    for pat in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(c for c in p.glob(pat) if c.is_file())

    # Ignore best-checkpoint snapshots; keep only regular epoch checkpoints.
    candidates = [c for c in candidates if not c.name.startswith("best_")]

    if pattern_filter and pattern_filter != "all":
        wanted = [s.strip() for s in pattern_filter.split(",") if s.strip()]
        candidates = [c for c in candidates if any(w in c.name for w in wanted)]

    def epoch_of_name(name):
        m = re.search(r"(\d{2,6})", name)
        return int(m.group(1)) if m else None

    def sort_key(pth):
        ep = epoch_of_name(pth.name)
        if ep is not None:
            return (0, ep)
        return (1, int(pth.stat().st_mtime))

    candidates.sort(key=sort_key)
    return [str(c) for c in candidates]


def epoch_id_from_name(name):
    m = re.search(r"(\d{2,6})", name)
    if m:
        return m.group(1)
    return str(int(time.time()))


def _grid_centers_flat(grid_size, device, dtype):
    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    return torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)


def _offsets_to_coords_gpu(offsets, grid_size, grid_centers_flat):
    bsz = offsets.shape[0]
    offs = offsets.permute(0, 2, 3, 1).reshape(bsz, grid_size * grid_size, 2)
    coords = grid_centers_flat.expand(bsz, -1, -1) + offs / float(grid_size)
    return coords.clamp(0.0, 1.0)


def _render_smart_init_gpu(coords, grid_size, sigma_px, device):
    lin = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    pixel_centers = torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)
    sigma = max(float(sigma_px), 1e-4) / float(grid_size)
    dist = torch.cdist(pixel_centers, coords, p=2)
    gauss = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
    return gauss.amax(dim=2).reshape(1, 1, grid_size, grid_size).clamp(0.0, 1.0)


def load_condition(image_path, grid_size, device, sdf_features, sdf_truncate_px):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_01 = img.astype(np.float32) / 255.0
    if not sdf_features:
        high_res = torch.from_numpy(image_01).unsqueeze(0).unsqueeze(0).to(device)
        target_density = torch.nn.functional.interpolate(high_res, size=(grid_size, grid_size), mode="area")
        return image_01, high_res, target_density, None, None

    high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_image(
        image_01,
        grid_size,
        device,
        sdf_truncate_px=sdf_truncate_px,
    )
    return image_01, high_res, target_density, high_res_sdf, target_sdf


def build_runtime(args):
    diffusion = ParseSampleConfig(args.config)
    # Initialize models and move them to the requested device, mirroring train script.
    device = torch.device(args.device)

    diffusion = ParseSampleConfig(args.config)
    # Load the base diffusion weights to CPU as in training script.
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    diffusion.eval()

    denoiser = diffusion.model

    # Build DynamicControlNet and move to device (same order as training)
    control_net = DynamicControlNet(
        denoiser,
        grid_size=GRID_SIZE,
        enable_gecco=ENABLE_GECCO,
        smart_init_features=SMART_INIT_FEATURES,
        sdf_features=SDF_FEATURES,
        batch_coords_features=BATCH_COORDS_FEATURES,
        enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION,
    ).to(device)
    control_net.eval()

    controlled = DynamicControlledDenoiser(denoiser, control_net).to(device)
    controlled.eval()

    diffusion.model = controlled
    diffusion.set_num_timesteps(args.timesteps)
    diffusion.eval()

    return diffusion, control_net, controlled, device


def load_checkpoint_into_model(control_net, diffusion, checkpoint_path, device):
    # Load checkpoint tensors directly to the target device, then move models there.
    state = torch.load(checkpoint_path, map_location=device)
    control_net.safe_load_state_dict(state, strict=False)

    # Move control net and diffusion (which wraps the controlled denoiser) to device
    control_net.to(device)
    try:
        diffusion.to(device)
    except Exception:
        # diffusion may not implement .to cleanly; ensure its model is moved
        if hasattr(diffusion, "model"):
            diffusion.model.to(device)

    control_net.eval()
    diffusion.eval()


def sample_points_for_image(diffusion, controlled, device, image_path, args):
    image_01, high_res, target_density, high_res_sdf, target_sdf = load_condition(
        image_path,
        GRID_SIZE,
        device,
        sdf_features=SDF_FEATURES,
        sdf_truncate_px=args.sdf_truncate_px,
    )

    smart_points = generate_smart_init_points_from_density(
        image_01,
        n_points=GRID_SIZE * GRID_SIZE,
        seed=args.smart_init_seed,
    )
    smart_offsets_np = smart_init_points_to_offsets(smart_points)
    smart_init_offsets = torch.from_numpy(smart_offsets_np).unsqueeze(0).to(device)

    if SMART_INIT_FEATURES:
        smart_grid_np = render_smart_init_grid(smart_points, grid_size=GRID_SIZE)
        smart_init_grid_raw = torch.from_numpy(smart_grid_np).unsqueeze(0).to(device)
        if ENABLE_SMART_INIT_SPLAT_SIGMA:
            grid_centers_flat = _grid_centers_flat(GRID_SIZE, device, smart_init_offsets.dtype)
            smart_coords = _offsets_to_coords_gpu(smart_init_offsets, GRID_SIZE, grid_centers_flat)
            smart_init_grid = _render_smart_init_gpu(
                smart_coords,
                GRID_SIZE,
                args.smart_init_splat_sigma_px,
                device,
            )
        else:
            smart_init_grid = smart_init_grid_raw
    else:
        smart_init_grid = None

    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)

    trunc_ratio = args.truncation_ratio
    if trunc_ratio is None:
        t_start = args.timesteps
    else:
        t_start = int(args.timesteps * trunc_ratio)
    t_start = int(np.clip(t_start, 1, max(args.timesteps - 1, 1)))

    alpha_t = diffusion.alphas_cumprod[t_start]
    img = add_noise_at_t(smart_init_offsets, alpha_t)

    with torch.no_grad() if args.resample_jumps == 0 else torch.enable_grad():
        for i in reversed(range(t_start)):
            t_tensor = torch.full((1,), i, dtype=torch.int64, device=device)
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

    s = img.detach().cpu().numpy()[0]
    pts = to_pointset_optimal_transport(s)
    pts = pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T
    return np.asarray(pts, dtype=np.float32)


def main():
    args = parse_args()
    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)

    manifest_path = out_base / "validation_manifest.json"
    val_data_dir = out_base / "validation_data"
    source_backup_dir = val_data_dir / "source"
    target_backup_dir = val_data_dir / "target"
    
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        val_images = [str(source_backup_dir / name) for name in manifest]
        missing = [p for p in val_images if not Path(p).exists()]
        if missing:
            print("Validation manifest exists but some backed-up images are missing.")
            print(f"Missing count: {len(missing)}")
            return 2
        print(f"Loaded {len(val_images)} validation images from existing manifest")
    else:
        all_images = list_images(args.source)
        if len(all_images) == 0:
            print(f"No source images found in {args.source}")
            return 2

        val_images = select_validation_images(all_images, args.val_split, args.seed)
        print(f"Selected {len(val_images)} validation images")
        source_backup_dir, target_backup_dir = backup_validation_images(val_images, out_base, args.target)
        print(f"Backed up validation data to {val_data_dir}")
        val_images = [str(source_backup_dir / Path(img).name) for img in val_images]

    ckpts = find_checkpoints(WEIGHTS_DIR, pattern_filter=args.checkpoints)
    if len(ckpts) == 0:
        print(f"No checkpoints found in {WEIGHTS_DIR}")
        return 2

    print(f"Model {RESULTS_DIR}: found {len(ckpts)} checkpoints")
    model_out_base = out_base / RESULTS_DIR
    model_out_base.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for ckpt in ckpts:
            eid = epoch_id_from_name(os.path.basename(ckpt))
            print(f"DRY epoch {eid} -> {(model_out_base / f'epoch_{eid}_npy')}")
        return 0

    diffusion, control_net, controlled, device = build_runtime(args)

    total_ckpts = len(ckpts)
    for ckpt_idx, ckpt in enumerate(ckpts, start=1):
        eid = epoch_id_from_name(os.path.basename(ckpt))
        epoch_dir = model_out_base / f"epoch_{eid}_npy"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{ckpt_idx}/{total_ckpts}] running epoch {eid} -> {epoch_dir}", flush=True)
        load_checkpoint_into_model(control_net, diffusion, ckpt, device)

        for img_idx, img in enumerate(val_images, start=1):
            print(f"epoch {eid} image {img_idx}: {Path(img).name}", flush=True)
            pts = sample_points_for_image(diffusion, controlled, device, img, args)
            np.save(epoch_dir / f"{Path(img).stem}.npy", pts)

        print(f"Finished checkpoint {ckpt}")

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
