"""Train the Dynamic ControlNet V4 (Truncated Control) for stippling.

V4 keeps the frozen base denoiser + trainable control branch, adds Smart Init as
an extra hint channel, and trains on truncated late-timestep noise.

Optional intermediate evaluation sampling supports full-schedule
RePaint-style resampling via ``--resample-jumps``.

Usage (from project root):
    python control_v4/train_control.py \
        --config  config/GBN/config.json \
        --ckpt    config/GBN/model.ckpt \
        --source  /path/to/source \
        --offsets /path/to/processed_offsets \
        --epochs  100 \
        --batch_size 16 \
        --lr 1e-4 \
        --out control_v4/control_out
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_v4.train_control import run  # noqa: E402

# ── default globals (edit here for quick experiments) ───────────────

SHOW_SELECTED_INPUTS = True
SHOW_SELECTED_GT = True
SHOW_SELECTED_PREDICT = True
SHOW_SELECTED_GT_OFFSETS = True

# Paths and I/O
WANDB_ENV = ".env"

SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/target"

BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = "config/GBN/model.ckpt"
FREEZE_DENOISER = True  # TODO: Add in the other scripts
GRID_SIZE = 32
VAL_SPLIT = 0.1
# EPOCHS = 10000
EPOCHS = 5000
SAVE_EVERY = 25
KEEP_EVERY = 100                    # 0 disables pruning; if >0 must be a multiple of SAVE_EVERY
ENABLE_GECCO = False
ENABLE_ADAPTIVE_GATE_INJECTION = False
EVAL_TIMESTEPS = 1000
TRAIN_TRUNCATION_RATIO = 1.0
INFER_TRUNCATION_RATIO = 1.0
RESAMPLE_JUMPS = 0
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False


# Vanilla
OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_vanilla"


# Unfrozen
# OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_unfrozen"
# BASE_CKPT_PATH = ""
# FREEZE_DENOISER = False


# GECCO
# OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_gecco"
# ENABLE_GECCO = True


# Adaptive gate injection
# OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_agi"
# ENABLE_ADAPTIVE_GATE_INJECTION = True


# Full
# OUTPUT_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_full"
# BASE_CKPT_PATH = ""
# FREEZE_DENOISER = False
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True


# If empty, offsets are auto-exported (if needed) to a default processed_offsets folder.
OFFSETS_DIR = ""
CACHE_DATA_DIR = ""
PRELOAD_RAM = False  # Preload all cached data to RAM (eliminates disk I/O per batch)
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Model parameters
# GRID_SIZE = 32
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_JITTER_PX = 0.5
SMART_INIT_SPLAT_SIGMA_PX = 0.5

# Loss component weights
MIN_SNR_GAMMA = 5.0
GEOM_CLUMP_WEIGHT = 1.0
BEST_MAX_CV = 1e9
BEST_MAX_CLUMPED_PCT = 100.0

# ── Synced with train_control.py run(): these args are READ by run() and must be
#    provided or run(args) raises AttributeError. Defaults preserve the original
#    ablation behaviour. Values below MATCH train_control's current defaults
#    (Component 2 ON at 0.8); override here to diverge for a given ablation.
POINTS_SOURCE = "npy"
DROP_WHITE_POINTS = True
WHITE_THRESHOLD = 255
T_SAMPLING = "uniform"            # Component 1 OFF
LOGIT_NORMAL_M = 0.0
LOGIT_NORMAL_S = 1.0
DENSITY_LOSS_WEIGHT = 0.8         # Component 2 ON (train_control default); set 0.0 to disable
DENSITY_KDE_GRID = 32
DENSITY_KDE_SIGMA_PX = 1.0
DENSITY_LOSS_T_FRAC = 0.4
DENSITY_LOSS_T_SOFT = 0.0
DENSITY_LOSS_WARMUP_EPOCHS = 5
DENSITY_LOSS_GRAD_LOG_EVERY = 20

# Training configuration
WANDB_ACTIVE = True

BATCH_SIZE = 16
LR = 1e-4
DEVICE = "cuda"
RESUME_LATEST = True

NUM_WORKERS = 4
PIN_MEMORY = True

WANDB_VALID_IMAGES = 8
WANDB_TRAIN_IMAGES = 8
SHOW_LABELS = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Paths and I/O
    parser.add_argument("--base_config_path", default=BASE_CONFIG_PATH)
    parser.add_argument("--base_ckpt_path", default=BASE_CKPT_PATH)
    parser.add_argument("--source",
                        default=SOURCE_DIR,
                        help="Dir of full-resolution grayscale source images")
    parser.add_argument("--target",
                        default=TARGET_DIR,
                        help="Dir of stippled target images (used to auto-export offsets if needed)")
    parser.add_argument("--offsets",
                        default=OFFSETS_DIR,
                        help="Dir of .npy offset files; if empty/missing, offsets are exported from --target")
    parser.add_argument("--cache-data-dir", default=CACHE_DATA_DIR,
                        help="Optional directory to cache feature artifacts (SDF and/or Smart Init) as .npy")
    parser.add_argument("--preload-ram", action="store_true", default=PRELOAD_RAM,
                        help="Preload all cached data (SDF, smart init) to RAM for zero disk I/O per batch")
    parser.add_argument("--out", default=OUTPUT_DIR,
                        help="Output directory for checkpoints and logs")

    # Model parameters
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE,
                        help="Grid resolution for offset export and dataset loading")
    parser.add_argument(
        "--enable-gecco",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_GECCO,
        help="Enable GECCO dynamic feature sampling in control hint path",
    )
    parser.add_argument(
        "--smart-init-features",
        action=argparse.BooleanOptionalAction,
        default=SMART_INIT_FEATURES,
        help="Enable Smart Init conditioning features in the control hint path",
    )
    parser.add_argument(
        "--sdf-features",
        action=argparse.BooleanOptionalAction,
        default=SDF_FEATURES,
        help="Enable SDF conditioning features in the control hint path",
    )
    parser.add_argument(
        "--batch-coords-features",
        action=argparse.BooleanOptionalAction,
        default=BATCH_COORDS_FEATURES,
        help="Enable coordinate-grid conditioning features in the control hint path",
    )
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX,
                        help="Truncate signed distance magnitudes before max-normalization (0 disables)")
    parser.add_argument("--eval-timesteps", type=int, default=EVAL_TIMESTEPS,
                        help="Timesteps used in intermediate eval sampling")
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS,
                        help="RePaint micro-loops per timestep during eval sampling")
    # NOTE: run() (control_v4.train_control) reads BOTH of these, so the ablation parser MUST
    # define them. The 4 trainable approaches use 1.0/1.0 (full-range training, full denoise).
    # The SDEdit variant is an INFERENCE-only change: --infer-truncation-ratio 0.3.
    parser.add_argument("--train-truncation-ratio", type=float, default=TRAIN_TRUNCATION_RATIO,
                        help="TRAINING only: draw diffusion timesteps from [0, ratio * total_timesteps)")
    parser.add_argument("--infer-truncation-ratio", type=float, default=INFER_TRUNCATION_RATIO,
                        help="INFERENCE only: SDEdit start level for eval sampling (0.3 = SDEdit variant)")
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED,
                        help="Random seed used when generating Smart Init")
    parser.add_argument(
        "--smart-init-jitter-px",
        type=float,
        default=SMART_INIT_JITTER_PX,
        help="Train-only Gaussian micro-jitter strength in grid-pixel units for Smart Init points (0 disables)",
    )
    parser.add_argument(
        "--smart-init-splat-sigma-px",
        type=float,
        default=SMART_INIT_SPLAT_SIGMA_PX,
        help="Gaussian sigma in grid-pixel units for GPU Smart Init soft splatting",
    )
    parser.add_argument(
        "--enable-smart-init-jitter",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_SMART_INIT_JITTER,
        help="Enable train-only Gaussian micro-jitter on Smart Init points (requires --smart-init-jitter-px > 0)",
    )
    parser.add_argument(
        "--enable-smart-init-splat-sigma",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_SMART_INIT_SPLAT_SIGMA,
        help="Enable Gaussian soft splatting for Smart Init grid; when disabled, uses occupancy rendering",
    )
    parser.add_argument(
        "--enable-adaptive-gate-injection",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_ADAPTIVE_GATE_INJECTION,
        help="Use sigmoid-gated adaptive injection (default); use --no-enable-adaptive-gate-injection for simple zero convolutions",
    )

    # Loss parameters
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=MIN_SNR_GAMMA,
        help="Gamma for Min-SNR loss weighting (0 disables)",
    )
    parser.add_argument("--geom-clump-weight", type=float, default=GEOM_CLUMP_WEIGHT,
                        help="Weight of clumped_pct in geometric score: score = cv + w*(clumped_pct/100)")
    parser.add_argument("--best-max-cv", type=float, default=BEST_MAX_CV,
                        help="Only save best-geom checkpoint if CV <= this value")
    parser.add_argument("--best-max-clumped-pct", type=float, default=BEST_MAX_CLUMPED_PCT,
                        help="Only save best-geom checkpoint if clumped_pct <= this value")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--freeze-denoiser",
        action=argparse.BooleanOptionalAction,
        default=FREEZE_DENOISER,
        help="Freeze the base denoiser (default); use --no-freeze-denoiser to train jointly",
    )
    parser.add_argument(
        "--resume-latest",
        action=argparse.BooleanOptionalAction,
        default=RESUME_LATEST,
        help="Resume training from the latest checkpoint found in --out",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=SAVE_EVERY,
        help="Every N epochs: save standard checkpoints and export/log train+valid+hints panels (best-geom checkpoints are saved independently)",
    )
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT,
                        help="Validation split ratio in [0,1). Example: 0.1 = 10%% val")
    parser.add_argument(
        "--wandb-valid-images",
        type=int,
        default=WANDB_VALID_IMAGES,
        help="Number of validation predictions to include in the wandb visual panel each epoch (0 disables valid panel upload)",
    )
    parser.add_argument(
        "--wandb-train-images",
        type=int,
        default=WANDB_TRAIN_IMAGES,
        help="Number of training predictions to include in the wandb visual panel each epoch (0 disables train panel upload)",
    )
    parser.add_argument(
        "--show-labels",
        action=argparse.BooleanOptionalAction,
        default=SHOW_LABELS,
        help="Show column headers (Condition, GT, Predict, GT Offset Quiver) on top row of panels",
    )
    parser.add_argument(
        "--show-selected-inputs",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SELECTED_INPUTS,
        help="Show the first panel column (Condition)",
    )
    parser.add_argument(
        "--show-selected-gt",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SELECTED_GT,
        help="Show the second panel column (GT)",
    )
    parser.add_argument(
        "--show-selected-predict",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SELECTED_PREDICT,
        help="Show the third panel column (Predict)",
    )
    parser.add_argument(
        "--show-selected-gt-offsets",
        action=argparse.BooleanOptionalAction,
        default=SHOW_SELECTED_GT_OFFSETS,
        help="Show the last panel column (GT Offset Quiver)",
    )
    parser.add_argument("--device", default=DEVICE)

    # ── Synced with train_control.py: run() reads all of the following ──────────
    parser.add_argument("--points-source", dest="points_source",
                        choices=("npy", "png"), default=POINTS_SOURCE,
                        help="npy: exact <stem>.npy beside each target (PNG-centroid fallback); png: always PNG centroids.")
    parser.add_argument("--drop-white-points", action=argparse.BooleanOptionalAction, default=DROP_WHITE_POINTS,
                        help="Drop GT points on background (rho < --white-threshold) and duplicate survivors. Changes cached OFFSETS.")
    parser.add_argument("--white-threshold", type=float, default=WHITE_THRESHOLD,
                        help="Source pixel value 0-255; a point is dropped when the pixel under it is >= this.")
    parser.add_argument("--keep-every", type=int, default=KEEP_EVERY,
                        help="Keep only periodic checkpoints at multiples of this (0 disables; must be a multiple of --save_every).")
    parser.add_argument("--t-sampling", choices=["uniform", "logit_normal"], default=T_SAMPLING,
                        help="Component 1: training timestep sampling.")
    parser.add_argument("--logit-normal-m", type=float, default=LOGIT_NORMAL_M,
                        help="Mean m of logit-normal t sampling.")
    parser.add_argument("--logit-normal-s", type=float, default=LOGIT_NORMAL_S,
                        help="Std s of logit-normal t sampling.")
    parser.add_argument("--density-loss-weight", type=float, default=DENSITY_LOSS_WEIGHT,
                        help="Component 2: density-match (KDE) loss weight (0 disables).")
    parser.add_argument("--density-kde-grid", type=int, default=DENSITY_KDE_GRID,
                        help="Resolution of the KDE density map for Component 2.")
    parser.add_argument("--density-kde-sigma-px", type=float, default=DENSITY_KDE_SIGMA_PX,
                        help="Gaussian sigma (KDE-grid pixels) for Component 2 splatting.")
    parser.add_argument("--density-loss-t-frac", type=float, default=DENSITY_LOSS_T_FRAC,
                        help="Component 2 applied only where t < frac * eval_timesteps.")
    parser.add_argument("--density-loss-t-soft", type=float, default=DENSITY_LOSS_T_SOFT,
                        help="Soft-ramp width (timesteps) at the Component 2 t cutoff; 0 = hard mask.")
    parser.add_argument("--density-loss-warmup-epochs", type=int, default=DENSITY_LOSS_WARMUP_EPOCHS,
                        help="Ramp Component 2 weight in over N epochs (0 = no ramp).")
    parser.add_argument("--density-loss-grad-log-every", type=int, default=DENSITY_LOSS_GRAD_LOG_EVERY,
                        help="Log grad-balance ratio every N steps (0 = off).")

    args = parser.parse_args()
    run(args=args)


if __name__ == "__main__":
    main()
