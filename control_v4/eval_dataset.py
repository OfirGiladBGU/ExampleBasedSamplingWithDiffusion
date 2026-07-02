"""Evaluate selected train/valid dataset samples and export a train-style panel.

This debug utility reproduces the exact train/valid split logic used in
``control_v4/train_control.py``:

    all_indices = torch.randperm(N, generator=torch.Generator().manual_seed(42))

You can pick rows from either split by their split-local indices, e.g.:

    --train-samples "[]" --valid-samples "[0,1,2,3]"

The output panel matches the 4-column format used during training:
Condition | GT | Predict | GT Offset Quiver
"""

import argparse
import ast
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from control_v4.DynamicControlNet import DynamicControlNet
from control_v4.DynamicStippleDataset import DynamicStippleDataset
from control_v4.train_control import (
    _grid_centers_flat,
    dynamic_collate,
    ensure_offsets_dir,
    offsets_to_coords_gpu,
    render_smart_init_gpu,
    sample_eval_batch,
    save_val_panel,
)

#################
# Defualt Setup #
#################

# Export Configuration
STORE_SELECTED_INPUTS = False
STORE_SELECTED_GT = False
STORE_SELECTED_PREDICT = False
STORE_SELECTED_GT_OFFSETS = False

# Panel visibility
SHOW_SELECTED_INPUTS = True
SHOW_SELECTED_GT = True
SHOW_SELECTED_PREDICT = True
SHOW_SELECTED_GT_OFFSETS = True

# Export Paths
OFFSETS_DIR = ""
CACHE_DATA_DIR = ""
OUTPUT_DIR = "control_v4/eval_outputs"
PANEL_NAME = "eval_panel.png"
META_NAME = "eval_selection.json"

# Editable defaults (copied locally so this script is self-contained)
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"

# [Default] Model Component
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
EVAL_TIMESTEPS = 1000
TRUNCATION_RATIO = 0.30
RESAMPLE_JUMPS = 2
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False

# [Default] Model parameters
GRID_SIZE = 32
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
# SMART_INIT_JITTER_PX = 0.5
SMART_INIT_SPLAT_SIGMA_PX = 0.5

VAL_SPLIT = 0.1
SPLIT_SEED = 42
DEVICE = "cuda"
SHOW_LABELS = False


#####################
# Customized Subset #
#####################

# ICONS
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep8120.pt"
CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
TRAIN_SAMPLES = []
VALID_SAMPLES = [1, 2, 3, 4]


# FACES
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_data_celeba_5K_1024_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_celeba_5K_1024/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_celeba_5K_1024/target"
# TRAIN_SAMPLES = []
# VALID_SAMPLES = [1, 2, 3, 4, 6, 8, 10, 17, 18, 19, 21, 23, 24, 25, 26, 28, 30, 32, 35, 36]


# ANIMALS
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_data_AM-2K_1024_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AM-2K_1024/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AM-2K_1024/target"
# TRAIN_SAMPLES = []
# VALID_SAMPLES = [0, 2, 4, 8, 9, 10, 11, 12, 13, 14]


# ICONS - TIMES
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
# TRAIN_SAMPLES = []
# VALID_SAMPLES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


# ICONS - TEASER
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
# TRAIN_SAMPLES = []
# VALID_SAMPLES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# STORE_SELECTED_INPUTS = True
# STORE_SELECTED_GT = True
# STORE_SELECTED_PREDICT = True
# STORE_SELECTED_GT_OFFSETS = True


# Ablations #

# Condition config (0.0)
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
# TRAIN_SAMPLES = []
# VALID_SAMPLES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# STORE_SELECTED_INPUTS = True
# STORE_SELECTED_GT = True
# STORE_SELECTED_PREDICT = True
# STORE_SELECTED_GT_OFFSETS = True
# SHOW_SELECTED_INPUTS = True
# SHOW_SELECTED_GT = False
# SHOW_SELECTED_PREDICT = False
# SHOW_SELECTED_GT_OFFSETS = False
# RESULTS_DIR = "condition"
# OUTPUT_DIR = f"control_v4/eval_outputs_{RESULTS_DIR}"


# Common config (1.0)
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
# TRAIN_SAMPLES = []
# VALID_SAMPLES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# STORE_SELECTED_INPUTS = True
# STORE_SELECTED_GT = True
# STORE_SELECTED_PREDICT = True
# STORE_SELECTED_GT_OFFSETS = True
# SHOW_SELECTED_INPUTS = False
# SHOW_SELECTED_GT = False
# SHOW_SELECTED_PREDICT = True
# SHOW_SELECTED_GT_OFFSETS = False

# Vanilla (1.1)
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_vanilla/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# RESULTS_DIR = "vanilla"
# GRID_SIZE = 32
# OUTPUT_DIR = f"control_v4/eval_outputs_{RESULTS_DIR}"
# ENABLE_GECCO = False
# ENABLE_ADAPTIVE_GATE_INJECTION = False
# EVAL_TIMESTEPS = 1000
# TRUNCATION_RATIO = 1.0
# RESAMPLE_JUMPS = 0
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False

# GECCO (1.2)
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_gecco/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# RESULTS_DIR = "gecco"
# GRID_SIZE = 32
# OUTPUT_DIR = f"control_v4/eval_outputs_{RESULTS_DIR}"
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

# Adaptive gate injection (1.3)
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_agi/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# RESULTS_DIR = "agi"
# GRID_SIZE = 32
# OUTPUT_DIR = f"control_v4/eval_outputs_{RESULTS_DIR}"
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

# Full (1.4)
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# RESULTS_DIR = "full"
# GRID_SIZE = 32
# OUTPUT_DIR = f"control_v4/eval_outputs_{RESULTS_DIR}"
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

# Full + SDEdit (1.5)
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# RESULTS_DIR = "sdedit"
# GRID_SIZE = 32
# OUTPUT_DIR = f"control_v4/eval_outputs_{RESULTS_DIR}"
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

# Full + SDEdit + resample jumps (1.6)
# CONTROL_CKPT = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full/checkpoints/dynamic_controlnet_v4_ep10000.pt"
# RESULTS_DIR = "sdedit_resample"
# GRID_SIZE = 32
# OUTPUT_DIR = f"control_v4/eval_outputs_{RESULTS_DIR}"
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


def _parse_index_list(raw_text, arg_name):
    """Parse a Python-list style argument into a list of non-negative ints."""
    try:
        parsed = ast.literal_eval(raw_text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"{arg_name} must be a Python list literal, got: {raw_text}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"{arg_name} must be a list, got: {type(parsed).__name__}")

    out = []
    for i, value in enumerate(parsed):
        if not isinstance(value, int):
            raise ValueError(f"{arg_name}[{i}] must be int, got: {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{arg_name}[{i}] must be >= 0, got: {value}")
        out.append(value)
    return out


def _validate_indices(indices, split_size, split_name):
    """Ensure all requested split-local indices are in range."""
    bad = [idx for idx in indices if idx >= split_size]
    if bad:
        raise IndexError(
            f"{split_name} indices out of range for split size {split_size}: {bad}"
        )


def _safe_sample_name(row, fallback_index):
    filename = os.path.splitext(os.path.basename(row["filename"]))[0]
    return f"{fallback_index:03d}_{row['split']}_{row['split_index']:04d}_{filename}"


def _save_condition_image(save_path, cond_image):
    image_u8 = np.clip(np.round(cond_image * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(save_path)


def _save_point_scatter(save_path, offsets, title=None):
    if not HAS_MPL:
        return False

    pts_grid = to_pointset_optimal_transport(offsets)
    pts = pts_grid.reshape(2, -1).T

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.scatter(pts[:, 0], 1.0 - pts[:, 1], c="black", s=0.5, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_offset_quiver(save_path, offsets, title=None):
    if not HAS_MPL:
        return False

    n_grid = offsets.shape[-1]
    yy, xx = np.mgrid[0:n_grid, 0:n_grid]
    dx, dy = offsets[0], offsets[1]
    mag = np.sqrt(dx * dx + dy * dy)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
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
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.colorbar(q, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _export_selected_columns(out_dir, meta_rows, cond_batch, gt_offsets_batch, pred_offsets_batch,
                             store_inputs=False, store_gt=False, store_predict=False, store_gt_offsets=False):
    if not (store_inputs or store_gt or store_predict or store_gt_offsets):
        return

    os.makedirs(out_dir, exist_ok=True)
    input_dir = os.path.join(out_dir, "inputs")
    gt_dir = os.path.join(out_dir, "gt")
    predict_dir = os.path.join(out_dir, "predict")
    gt_offsets_dir = os.path.join(out_dir, "gt_offsets")

    if store_inputs:
        os.makedirs(input_dir, exist_ok=True)
    if store_gt:
        os.makedirs(gt_dir, exist_ok=True)
    if store_predict:
        os.makedirs(predict_dir, exist_ok=True)
    if store_gt_offsets:
        os.makedirs(gt_offsets_dir, exist_ok=True)

    for i, row in enumerate(meta_rows):
        sample_name = _safe_sample_name(row, i)

        if store_inputs:
            _save_condition_image(os.path.join(input_dir, f"{sample_name}.png"), cond_batch[i, 0])

        if store_gt:
            if not _save_point_scatter(os.path.join(gt_dir, f"{sample_name}.png"), gt_offsets_batch[i]):
                print("matplotlib unavailable; skipping GT export")
                return

        if store_predict:
            if not _save_point_scatter(os.path.join(predict_dir, f"{sample_name}.png"), pred_offsets_batch[i]):
                print("matplotlib unavailable; skipping predict export")
                return

        if store_gt_offsets:
            if not _save_offset_quiver(os.path.join(gt_offsets_dir, f"{sample_name}.png"), gt_offsets_batch[i]):
                print("matplotlib unavailable; skipping GT offset export")
                return


def _load_models(args, device):
    """Load diffusion backbone and trained ControlNet checkpoint."""
    if not args.control_ckpt:
        raise ValueError("--control-ckpt is required")
    if not os.path.isfile(args.control_ckpt):
        raise FileNotFoundError(f"Control checkpoint was not found: {args.control_ckpt}")

    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    diffusion.eval()
    denoiser = diffusion.model

    control_net = DynamicControlNet(
        denoiser,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        batch_coords_features=args.batch_coords_features,
        enable_adaptive_gate_injection=args.enable_adaptive_gate_injection,
    ).to(device)
    state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.safe_load_state_dict(state, strict=False)
    control_net.eval()
    return diffusion, denoiser, control_net


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--control-ckpt", default=CONTROL_CKPT, help="Trained control_v4 checkpoint path")

    parser.add_argument("--source", default=SOURCE_DIR)
    parser.add_argument("--target", default=TARGET_DIR)
    parser.add_argument("--offsets", default=OFFSETS_DIR)
    parser.add_argument("--cache-data-dir", default=CACHE_DATA_DIR)
    parser.add_argument("--out", default=OUTPUT_DIR)

    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)

    parser.add_argument("--eval-timesteps", type=int, default=EVAL_TIMESTEPS)
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO)

    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--enable-gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    parser.add_argument(
        "--smart-init-features",
        action=argparse.BooleanOptionalAction,
        default=SMART_INIT_FEATURES,
    )
    parser.add_argument(
        "--sdf-features",
        action=argparse.BooleanOptionalAction,
        default=SDF_FEATURES,
    )
    parser.add_argument(
        "--batch-coords-features",
        action=argparse.BooleanOptionalAction,
        default=BATCH_COORDS_FEATURES,
    )
    parser.add_argument(
        "--enable-smart-init-splat-sigma",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_SMART_INIT_SPLAT_SIGMA,
        help="Re-render smart init grid from smart_init_offsets with Gaussian splat before sampling",
    )
    parser.add_argument(
        "--smart-init-splat-sigma-px",
        type=float,
        default=SMART_INIT_SPLAT_SIGMA_PX,
        help="Gaussian sigma in grid-pixel units for optional smart init soft splatting",
    )

    parser.add_argument("--train-samples", default=str(TRAIN_SAMPLES), help="Python list literal, e.g. [] or [0,2,4]")
    parser.add_argument("--valid-samples", default=str(VALID_SAMPLES), help="Python list literal")
    parser.add_argument("--panel-name", default=PANEL_NAME)
    parser.add_argument("--meta-name", default=META_NAME)
    parser.add_argument(
        "--show-labels",
        action=argparse.BooleanOptionalAction,
        default=SHOW_LABELS,
        help="Show column headers (Condition, GT, Predict, GT Offset Quiver) on top row of panel",
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
    parser.add_argument(
        "--store-selected-inputs",
        action=argparse.BooleanOptionalAction,
        default=STORE_SELECTED_INPUTS,
        help="Export the first panel column into <out>/inputs/ as separate images",
    )
    parser.add_argument(
        "--store-selected-gt",
        action=argparse.BooleanOptionalAction,
        default=STORE_SELECTED_GT,
        help="Export the second panel column into <out>/gt/ as separate images",
    )
    parser.add_argument(
        "--store-selected-predict",
        action=argparse.BooleanOptionalAction,
        default=STORE_SELECTED_PREDICT,
        help="Export the third panel column into <out>/predict/ as separate images",
    )
    parser.add_argument(
        "--store-selected-gt-offsets",
        action=argparse.BooleanOptionalAction,
        default=STORE_SELECTED_GT_OFFSETS,
        help="Export the last panel column into <out>/gt_offsets/ as separate images",
    )
    parser.add_argument(
        "--enable-adaptive-gate-injection",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_ADAPTIVE_GATE_INJECTION,
        help="Use sigmoid-gated adaptive injection (default); use --no-enable-adaptive-gate-injection for simple zero convolutions",
    )
    args = parser.parse_args()

    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("--val-split must be in [0, 1)")
    if not (0.0 < args.truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0, 1]")
    if not (args.show_selected_inputs or args.show_selected_gt or args.show_selected_predict or args.show_selected_gt_offsets):
        raise ValueError("At least one of --show-selected-inputs, --show-selected-gt, --show-selected-predict, or --show-selected-gt-offsets must be enabled")

    train_samples = _parse_index_list(args.train_samples, "--train-samples")
    valid_samples = _parse_index_list(args.valid_samples, "--valid-samples")
    if len(train_samples) + len(valid_samples) == 0:
        raise ValueError("At least one sample is required across --train-samples and --valid-samples")

    args.offsets = ensure_offsets_dir(args.source, args.target, args.offsets, args.grid_size)
    os.makedirs(args.out, exist_ok=True)

    cache_data_dir = args.cache_data_dir
    if not (args.smart_init_features or args.sdf_features):
        cache_data_dir = None
    elif not cache_data_dir:
        cache_data_dir = os.path.join(args.out, "cache_data")

    base_dataset = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        preload_ram=False,
    )
    if len(base_dataset) == 0:
        raise RuntimeError("Dataset is empty after matching source images and offsets")

    val_len = int(len(base_dataset) * args.val_split)
    val_len = min(max(val_len, 0), max(len(base_dataset) - 1, 0))
    train_len = len(base_dataset) - val_len

    all_indices = torch.randperm(
        len(base_dataset),
        generator=torch.Generator().manual_seed(args.split_seed),
    ).tolist()
    train_indices = all_indices[:train_len]
    val_indices = all_indices[train_len:]

    train_filenames = [base_dataset.filenames[i] for i in train_indices]
    val_filenames = [base_dataset.filenames[i] for i in val_indices]

    train_dataset = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        filenames=train_filenames,
        preload_ram=False,
    )
    val_dataset = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        filenames=val_filenames,
        preload_ram=False,
    )

    _validate_indices(train_samples, len(train_dataset), "train")
    _validate_indices(valid_samples, len(val_dataset), "valid")

    selected = []
    meta_rows = []
    for idx in train_samples:
        selected.append(train_dataset[idx])
        meta_rows.append(
            {
                "split": "train",
                "split_index": idx,
                "filename": train_dataset.filenames[idx],
            }
        )
    for idx in valid_samples:
        selected.append(val_dataset[idx])
        meta_rows.append(
            {
                "split": "valid",
                "split_index": idx,
                "filename": val_dataset.filenames[idx],
            }
        )

    batch = dynamic_collate(selected)
    device = torch.device(args.device)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    diffusion, denoiser, control_net = _load_models(args, device)

    if (
        args.smart_init_features
        and args.enable_smart_init_splat_sigma
        and ("smart_init_offsets" in batch)
    ):
        grid_centers_flat = _grid_centers_flat(args.grid_size, device=device, dtype=torch.float32)
        smart_coords = offsets_to_coords_gpu(batch["smart_init_offsets"], args.grid_size, grid_centers_flat)
        batch["smart_init_grid"] = render_smart_init_gpu(
            smart_coords,
            grid_size=args.grid_size,
            sigma_px=args.smart_init_splat_sigma_px,
            grid_centers_flat=grid_centers_flat,
        )

    pred_raw = sample_eval_batch(
        diffusion,
        denoiser,
        control_net,
        batch,
        device,
        n_samples=batch["high_res"].shape[0],
        eval_timesteps=args.eval_timesteps,
        resample_jumps=args.resample_jumps,
        show_tqdm=True,
        tqdm_desc="eval_dataset sampling",
        truncation_ratio=args.truncation_ratio,
    )

    panel_path = os.path.join(args.out, args.panel_name)
    saved = save_val_panel(
        panel_path,
        batch["high_res"].detach().cpu().numpy(),
        batch["offsets"].detach().cpu().numpy(),
        pred_raw.detach().cpu().numpy(),
        max_samples=len(selected),
        show_labels=args.show_labels,
        show_selected_inputs=args.show_selected_inputs,
        show_selected_gt=args.show_selected_gt,
        show_selected_predict=args.show_selected_predict,
        show_selected_gt_offsets=args.show_selected_gt_offsets,
    )
    if not saved:
        raise RuntimeError("Panel export failed (matplotlib unavailable or no samples)")

    _export_selected_columns(
        args.out,
        meta_rows,
        batch["high_res"].detach().cpu().numpy(),
        batch["offsets"].detach().cpu().numpy(),
        pred_raw.detach().cpu().numpy(),
        store_inputs=args.store_selected_inputs,
        store_gt=args.store_selected_gt,
        store_predict=args.store_selected_predict,
        store_gt_offsets=args.store_selected_gt_offsets,
    )

    meta = {
        "source": args.source,
        "offsets": args.offsets,
        "grid_size": args.grid_size,
        "val_split": args.val_split,
        "split_seed": args.split_seed,
        "train_size": len(train_dataset),
        "valid_size": len(val_dataset),
        "selected_rows": meta_rows,
        "panel_path": panel_path,
        "control_ckpt": args.control_ckpt,
        "store_selected_inputs": args.store_selected_inputs,
        "store_selected_gt": args.store_selected_gt,
        "store_selected_predict": args.store_selected_predict,
        "store_selected_gt_offsets": args.store_selected_gt_offsets,
    }
    meta_path = os.path.join(args.out, args.meta_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved panel: {panel_path}")
    print(f"Saved metadata: {meta_path}")
    print("Selection order:")
    for i, row in enumerate(meta_rows):
        print(f"  row {i}: {row['split']}[{row['split_index']}] -> {row['filename']}")


if __name__ == "__main__":
    main()
