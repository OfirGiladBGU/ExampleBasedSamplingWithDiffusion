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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_v4.sample_control import load_pipeline, process_single_image


BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = "config/GBN/model.ckpt"

# Vanilla
WEIGHTS_DIR = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_vanilla/checkpoints"
RESULTS_DIR = "vanilla"
GRID_SIZE = 32
ENABLE_GECCO = False
ENABLE_ADAPTIVE_GATE_INJECTION = False
EVAL_TIMESTEPS = 1000
TRUNCATION_RATIO = 1.0
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
# ENABLE_GECCO = True  # NOTE
# ENABLE_ADAPTIVE_GATE_INJECTION = False
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = 1.0
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
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = 1.0
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
# ENABLE_GECCO = True  # NOTE
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = 1.0
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
# ENABLE_GECCO = True  # NOTE
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = 0.3  # NOTE
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
# ENABLE_GECCO = True  # NOTE
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = 0.3  # NOTE
# RESAMPLE_JUMPS = 2  # NOTE
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# Default folders
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
# OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics"
OUTPUT_DIR = "experiments/outputs/ablation_advance_metrics_e400_b50_1024"

SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SPLAT_SIGMA_PX = 0.5

DEVICE = "cuda"
SPLIT_SEED = 42
VAL_SPLIT = 0.1
# NUM_SAMPLES = -1  # Use all validation samples by default
NUM_SAMPLES = 50
# NUM_EPOCHS = -1  # Use all checkpoints by default
NUM_EPOCHS = 400
# NOTE: 40 is epoch 1K

def parse_args():
    p = argparse.ArgumentParser(description="Export ablation predictions to .npy per epoch")
    p.add_argument("--output", default=OUTPUT_DIR, help="Base output folder for exports")
    p.add_argument("--source", default=SOURCE_DIR, help="Source images folder (validation pool)")
    p.add_argument("--target", default=TARGET_DIR, help="Target images folder (ground truth)")
    p.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Fraction for validation split")
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                   help="Number of validation samples to use in order after selection; -1 means use all")
    p.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                   help="Number of sorted checkpoints/epochs to export; -1 means use all")
    p.add_argument("--seed", type=int, default=SPLIT_SEED, help="Deterministic seed for split")
    p.add_argument("--checkpoints", default="all", help="'all' or comma-separated filename substrings")
    p.add_argument("--dry-run", action="store_true", help="Only show what would be done")
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


def limit_validation_images(val_images, num_samples):
    if int(num_samples) < 0:
        return val_images
    return val_images[: int(num_samples)]


def limit_checkpoints(ckpts, num_epochs):
    if int(num_epochs) < 0:
        return ckpts
    return ckpts[: int(num_epochs)]


def backup_validation_images(val_images, out_base, target_dir):
    out_base_p = Path(out_base)
    val_data_dir = out_base_p / "validation_data"
    source_backup_dir = val_data_dir / "source"
    target_backup_dir = val_data_dir / "target"
    manifest_path = out_base_p / "validation_manifest.json"

    source_backup_dir.mkdir(parents=True, exist_ok=True)
    target_backup_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    
    # Build a map of target filenames to full paths (recursive search)
    target_path_obj = Path(target_dir)
    target_map = {}
    for target_file in target_path_obj.rglob("*"):
        if target_file.is_file() and target_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            target_map[target_file.name] = target_file
    
    for img in tqdm(val_images, desc="Backing up validation images"):
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
        val_images = limit_validation_images(val_images, args.num_samples)
        print(f"Loaded {len(val_images)} validation images from existing manifest")
    else:
        all_images = list_images(args.source)
        if len(all_images) == 0:
            print(f"No source images found in {args.source}")
            return 2

        manifest_images = select_validation_images(all_images, args.val_split, args.seed)
        source_backup_dir, target_backup_dir = backup_validation_images(manifest_images, out_base, args.target)
        print(f"Backed up validation data to {val_data_dir}")
        val_images = limit_validation_images(manifest_images, args.num_samples)
        print(f"Selected {len(val_images)} validation images")
        val_images = [str(source_backup_dir / Path(img).name) for img in val_images]

    ckpts = find_checkpoints(WEIGHTS_DIR, pattern_filter=args.checkpoints)
    ckpts = limit_checkpoints(ckpts, args.num_epochs)
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

    total_ckpts = len(ckpts)
    for ckpt_idx, ckpt in enumerate(ckpts, start=1):
        eid = epoch_id_from_name(os.path.basename(ckpt))
        epoch_dir = model_out_base / f"epoch_{eid}_npy"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        diffusion, control_net = load_pipeline(
            base_config_path=BASE_CONFIG_PATH, 
            base_ckpt_path=BASE_CKPT_PATH, 
            control_ckpt_path=ckpt, # NOTE: Load current checkpoint for each epoch
            grid_size=GRID_SIZE, 
            enable_gecco=ENABLE_GECCO, 
            enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION, 
            smart_init_features=SMART_INIT_FEATURES, 
            sdf_features=SDF_FEATURES, 
            batch_coords_features=BATCH_COORDS_FEATURES, 
            device=DEVICE
        )

        print(f"[{ckpt_idx}/{total_ckpts}] running epoch {eid} -> {epoch_dir}", flush=True)
        for img_idx, img in enumerate(val_images, start=1):
            print(f"epoch {eid} image {img_idx}: {Path(img).name}", flush=True)

            process_single_image(
                image_path=Path(img), 
                diffusion=diffusion, 
                control_net=control_net,
                grid_size=GRID_SIZE,
                truncation_ratio=TRUNCATION_RATIO,
                eval_timesteps=EVAL_TIMESTEPS,
                smart_init_features=SMART_INIT_FEATURES,
                sdf_features=SDF_FEATURES,
                resample_jumps=RESAMPLE_JUMPS,
                sdf_truncate_px=SDF_TRUNCATE_PX,
                t_start_step=-1,
                smart_init_seed=SMART_INIT_SEED,
                smart_init_splat_sigma_px=SMART_INIT_SPLAT_SIGMA_PX,
                enable_smart_init_splat_sigma=ENABLE_SMART_INIT_SPLAT_SIGMA,
                show_denoising_interval=50,
                device=DEVICE,
                # Exports
                export_conditions=False,
                export_png=False,
                export_npy=True,  # NOTE: Export NPY
                track_time=False,
                show_denoising=False,
                conditions_dir=None,
                png_dir=None,
                npy_dir=Path(epoch_dir),  # NOTE: Export NPY
                timestamps_dir=None,
                denoising_dir=None,
            )

        print(f"Finished checkpoint {ckpt}")

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
