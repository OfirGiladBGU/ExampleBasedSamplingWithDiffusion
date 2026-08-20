"""ablation_runtime_results_stage_1.py -- Part 1: measure inference RUNTIME.

Uncomment ONE method block below and run. It loads that method's ep5000 checkpoint
(the last weights only) and runs inference over the validation images with timing on,
writing one per-image timing .txt into:

    OUTPUT_DIR/<RESULTS_DIR>/timestamps/<image_stem>.txt

No predictions are exported -- this only measures speed. Reuses the config blocks and
the seed-42 validation split from ablation_advance_metrics_stage_1.py.

Part 2 (ablation_runtime_results_stage_2.py) averages each method's timestamps; part 3
merges the per-method means.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_v4.sample_control import load_pipeline, process_single_image


BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = "config/GBN/model.ckpt"
FREEZE_DENOISER = True  # informational (load_pipeline auto-detects from the checkpoint)
GRID_SIZE = 32
ENABLE_GECCO = False
ENABLE_ADAPTIVE_GATE_INJECTION = False
EVAL_TIMESTEPS = 1000
INFER_TRUNCATION_RATIO = 1.0
RESAMPLE_JUMPS = 0
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False


# ── Uncomment ONE method block ────────────────────────────────────────────────

# Vanilla
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_vanilla/checkpoints"
# RESULTS_DIR = "vanilla"


# Unfrozen
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_unfrozen/checkpoints"
# RESULTS_DIR = "unfrozen"
# BASE_CKPT_PATH = ""  # NOTE
# FREEZE_DENOISER = False  # NOTE


# GECCO
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_gecco/checkpoints"
# RESULTS_DIR = "gecco"
# ENABLE_GECCO = True  # NOTE


# Adaptive gate injection
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_agi/checkpoints"
# RESULTS_DIR = "agi"
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE


# Full
WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints"
RESULTS_DIR = "full"
BASE_CKPT_PATH = ""  # NOTE
FREEZE_DENOISER = False  # NOTE
ENABLE_GECCO = True  # NOTE
ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE


# Full + SDEdit
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints"
# RESULTS_DIR = "sdedit"
# BASE_CKPT_PATH = ""  # NOTE
# FREEZE_DENOISER = False  # NOTE
# ENABLE_GECCO = True  # NOTE
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE
# INFER_TRUNCATION_RATIO = 0.5  # NOTE


# ── Shared config ─────────────────────────────────────────────────────────────
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/source"
OUTPUT_DIR = "experiments/outputs/ablation_runtime_results"

SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SPLAT_SIGMA_PX = 0.5

DEVICE = "cuda"
SPLIT_SEED = 42
VAL_SPLIT = 0.1
NUM_SAMPLES = 50            # number of validation images to time
EPOCH = 5000               # measure only the last (ep5000) weights
WARMUP_RUNS = 1            # untimed inferences to warm up CUDA before timing (0 = none)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def list_images(folder):
    p = Path(folder)
    return [str(f) for f in sorted(p.rglob("*")) if f.suffix.lower() in IMG_EXTS]


def select_validation_images(all_images, val_frac, seed):
    imgs = sorted(all_images)
    n_total = len(imgs)
    val_len = min(max(int(n_total * float(val_frac)), 0), max(n_total - 1, 0))
    train_len = n_total - val_len
    order = torch.randperm(n_total, generator=torch.Generator().manual_seed(int(seed))).tolist()
    return [imgs[i] for i in order[train_len:]]


def limit_validation_images(val_images, num_samples):
    if int(num_samples) < 0:
        return val_images
    return val_images[: int(num_samples)]


def find_checkpoint_at_epoch(weights_dir, epoch):
    p = Path(weights_dir)
    for pat in ("*.ckpt", "*.pt", "*.pth"):
        for c in sorted(p.glob(pat)):
            if c.name.startswith("best_"):
                continue
            m = re.search(r"ep(\d+)", c.name)
            if m and int(m.group(1)) == int(epoch):
                return str(c)
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Time the active method's ep5000 inference over the val set.")
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--source", default=SOURCE_DIR)
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="-1 = all val images")
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--epoch", type=int, default=EPOCH)
    p.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_one(img, diffusion, control_net, ts_dir, track_time):
    process_single_image(
        image_path=Path(img), diffusion=diffusion, control_net=control_net,
        grid_size=GRID_SIZE, truncation_ratio=INFER_TRUNCATION_RATIO, eval_timesteps=EVAL_TIMESTEPS,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        resample_jumps=RESAMPLE_JUMPS, sdf_truncate_px=SDF_TRUNCATE_PX, t_start_step=-1,
        smart_init_seed=SMART_INIT_SEED, smart_init_splat_sigma_px=SMART_INIT_SPLAT_SIGMA_PX,
        enable_smart_init_splat_sigma=ENABLE_SMART_INIT_SPLAT_SIGMA,
        show_denoising_interval=50, device=DEVICE,
        export_conditions=False, export_png=False, export_npy=False,   # timing only
        track_time=track_time, show_denoising=False,
        conditions_dir=None, png_dir=None, npy_dir=None,
        timestamps_dir=(Path(ts_dir) if track_time else None), denoising_dir=None,
    )


def main():
    args = parse_args()
    all_images = list_images(args.source)
    if not all_images:
        print(f"No source images in {args.source}"); return 2
    val_images = limit_validation_images(
        select_validation_images(all_images, args.val_split, args.seed), args.num_samples)

    ckpt = find_checkpoint_at_epoch(WEIGHTS_DIR, args.epoch)
    if ckpt is None:
        print(f"No ep{args.epoch} checkpoint in {WEIGHTS_DIR}"); return 2

    ts_dir = Path(args.output) / RESULTS_DIR / "timestamps"
    print(f"Method '{RESULTS_DIR}' ep{args.epoch}: {ckpt}")
    print(f"  gecco={ENABLE_GECCO} agi={ENABLE_ADAPTIVE_GATE_INJECTION} "
          f"base_ckpt={'<from-scratch>' if not BASE_CKPT_PATH else BASE_CKPT_PATH} "
          f"trunc={INFER_TRUNCATION_RATIO} resample={RESAMPLE_JUMPS}")
    print(f"  timing {len(val_images)} images -> {ts_dir}")

    if args.dry_run:
        print(f"DRY: would time {len(val_images)} images ({args.warmup} warmup)")
        return 0

    ts_dir.mkdir(parents=True, exist_ok=True)
    diffusion, control_net = load_pipeline(
        base_config_path=BASE_CONFIG_PATH, base_ckpt_path=BASE_CKPT_PATH, control_ckpt_path=ckpt,
        grid_size=GRID_SIZE, enable_gecco=ENABLE_GECCO,
        enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        batch_coords_features=BATCH_COORDS_FEATURES, device=DEVICE,
    )

    for w in range(max(0, args.warmup)):
        print(f"  warmup {w + 1}/{args.warmup}", flush=True)
        run_one(val_images[0], diffusion, control_net, ts_dir, track_time=False)

    for i, img in enumerate(val_images, start=1):
        print(f"  [{i}/{len(val_images)}] {Path(img).name}", flush=True)
        run_one(img, diffusion, control_net, ts_dir, track_time=True)

    print(f"\nStage 1 done -> {ts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
