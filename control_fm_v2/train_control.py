"""Train the Dynamic ControlNet FM (Truncated Control) for stippling.

FM keeps the frozen base denoiser + trainable control branch, adds Smart Init as
an extra hint channel, and trains on truncated late-timestep noise.

Optional intermediate evaluation sampling supports full-schedule
RePaint-style resampling via ``--resample-jumps``.

Usage (from project root):
    python control_fm/train_control.py \
        --source  /path/to/source \
        --offsets /path/to/processed_offsets \
        --epochs  100 \
        --batch_size 16 \
        --lr 1e-4 \
        --out control_fm/control_out
"""

import contextlib
import os
import sys
import argparse
import re
from datetime import datetime
import math

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

from control_fm.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_fm.DynamicStippleDataset import DynamicStippleDataset
from control_fm.smart_init import build_smart_init_from_image
from control_fm_v2.flow_matching import MODEL_CONFIG, FlowMatching, build_velocity_network
from control_fm_v2.geometry_loss import GeometryLoss
from control_fm_v2.ema import WeightEMA
from control_fm_v2.arch import arch_from
from data.Transforms import to_image_optimal_transport, to_pointset_optimal_transport
from utils.stippling_metrics import geometric_validation_score

# ── default globals (edit here for quick experiments) ───────────────

# Paths and I/O
WANDB_ENV = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/.env"

# control_fm_v2 has NO config file. The architecture lives in
# control_fm_v2/flow_matching.py::MODEL_CONFIG.
# NOTE: control_fm has NO pretrained baseline. build_velocity_network() / the single-stage
# builders read the model config only -- they never call torch.load. Both the base U-Net and
# the control branch are Flow-Matching weights trained from scratch and stored together in
# the control_fm checkpoint. There is deliberately no "model.ckpt" path here.

# ICONS 1024 WVS
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/target"
# OUTPUT_DIR = "control_fm_v2/train_outputs_icons50_512_no_random_CN_TRUNC_HEUN"

# ICONS 1024 GBN
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN/target"
OUTPUT_DIR = "control_fm_v2/train_outputs_icons50_512_GBN_SPADE_GEO"

GRID_SIZE = 32
VAL_SPLIT = 0.1
EPOCHS = 2000
SAVE_EVERY = 10
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False

MIN_SNR_GAMMA = 5.0

# GEO_WEIGHT = 0.0
GEO_WEIGHT = 0.05          # lambda_geo -- primary term (density-warped spacing hinge)


# # CONCAT
# ODE_METHOD = "euler"
# TRUNCATION_RATIO = 1.00
# FM_COUPLING = "gaussian"  # "smartinit" or "gaussian"
# CONDITIONING = "concat"  # "controlnet", "concat", or "spade"

# CN 
# ODE_METHOD = "euler"
# TRUNCATION_RATIO = 1.00
# FM_COUPLING = "gaussian"  # "smartinit" or "gaussian"
# CONDITIONING = "controlnet"  # "controlnet", "concat", or "spade"

# CN + TRUC
# ODE_METHOD = "euler"
# TRUNCATION_RATIO = 0.30
# FM_COUPLING = "gaussian"  # "smartinit" or "gaussian"
# CONDITIONING = "controlnet"  # "controlnet", "concat", or "spade"

# SPADE
ODE_METHOD = "euler"
TRUNCATION_RATIO = 1.00
FM_COUPLING = "gaussian"  # "smartinit" or "gaussian"
CONDITIONING = "spade"  # "controlnet", "concat", or "spade"

# SPADE + SMART
# MIN_SNR_GAMMA = 0.0
# ODE_METHOD = "euler"
# TRUNCATION_RATIO = 1.00
# FM_COUPLING = "smartinit"
# CONDITIONING = "spade"

# CN + TRUC + HEUN
# ODE_METHOD = "heun"
# TRUNCATION_RATIO = 0.30
# FM_COUPLING = "gaussian"  # "smartinit" or "gaussian"
# CONDITIONING = "controlnet"  # "controlnet", "concat", or "spade"


# FACES 1024
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_celeba_5K_1024/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_celeba_5K_1024/target"
# OUTPUT_DIR = "control_fm/train_outputs_data_celeba_5K_1024_no_random"
# GRID_SIZE = 32
# VAL_SPLIT = 0.1
# EPOCHS = 10000
# SAVE_EVERY = 10
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# TRUNCATION_RATIO = 0.30
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# ANIMALS 1024
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AM-2K_1024/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AM-2K_1024/target"
# OUTPUT_DIR = "control_fm/train_outputs_data_AM-2K_1024_no_random"
# GRID_SIZE = 32
# VAL_SPLIT = 0.1
# EPOCHS = 20000
# SAVE_EVERY = 10
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# TRUNCATION_RATIO = 0.30
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# Stress 1 1024
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/target"
# OUTPUT_DIR = "control_fm/train_outputs_data_stress1_no_random"
# GRID_SIZE = 32
# VAL_SPLIT = 0.0
# EPOCHS = 10000
# SAVE_EVERY = 50
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# TRUNCATION_RATIO = 0.30
# SMART_INIT_FEATURES = False    # NOTE
# SDF_FEATURES = False           # NOTE
# BATCH_COORDS_FEATURES = False  # NOTE
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# Stress 2 1024 V2
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2/target"
# OUTPUT_DIR = "control_fm/train_outputs_data_stress2_V2_no_random"
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False
# GRID_SIZE = 32
# VAL_SPLIT = 0.0
# EPOCHS = 10000
# SAVE_EVERY = 50
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# TRUNCATION_RATIO = 0.30
# SMART_INIT_FEATURES = False    # NOTE
# SDF_FEATURES = False           # NOTE
# BATCH_COORDS_FEATURES = False  # NOTE
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# # Stress 2 1024 V1 - IGNORE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2/target"
# OUTPUT_DIR = "control_fm/train_outputs_data_stress2_no_random"
# GRID_SIZE = 32
# VAL_SPLIT = 0.0
# EPOCHS = 10000
# SAVE_EVERY = 10
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# TRUNCATION_RATIO = 0.30
# SMART_INIT_FEATURES = True    # NOTE
# SDF_FEATURES = True           # NOTE
# BATCH_COORDS_FEATURES = True  # NOTE
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# If empty, offsets are auto-exported (if needed) to a default processed_offsets folder.
OFFSETS_DIR = ""
CACHE_DATA_DIR = ""
PRELOAD_RAM = False  # Preload all cached data to RAM (eliminates disk I/O per batch)
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Model parameters
FREEZE_DENOISER = False  # single-model Flow Matching: train base + control jointly
# GRID_SIZE = 32
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_JITTER_PX = 0.5
SMART_INIT_SPLAT_SIGMA_PX = 0.5

# Loss component weights
# Tier 1.3: 0.0 disables SNR weighting (plain velocity MSE).
# NOTE: partially redundant with logit-normal t -- both concentrate gradient on mid-t.
# Also w(t) -> 0 as t -> 1, i.e. exactly where gaussian sampling STARTS. Ablate it
# separately; this is the least certain of the Tier-1 changes.
GEOM_CLUMP_WEIGHT = 1.0
BEST_MAX_CV = 1e9
BEST_MAX_CLUMPED_PCT = 100.0

# Training configuration
# ── control_fm_v2 upgrades (guide: enable ONE at a time, validate after each) ──
USE_EMA = True             # Tier 1.1 -- highest priority, on by default
EMA_DECAY = 0.9999
EMA_WARMUP_STEPS = 500
T_DIST = "logitnormal"     # Tier 1.2 -- 'uniform' | 'logitnormal'
T_LOGITNORM_M = 0.0
T_LOGITNORM_S = 1.0
# Tier 2 -- single source of truth is flow_matching.MODEL_CONFIG
ATTN_MIDDLE = MODEL_CONFIG["attn_middle"]
ATTN_HEADS = MODEL_CONFIG["attn_heads"]

# ── Geometry loss on the one-step-decoded points (see geometry_loss.py) ──────
# Auxiliary spacing signal that velocity-MSE is structurally blind to. Enabled by default
# with a deliberately small weight; the velocity loss must stay dominant.
GEO_CAP_WEIGHT = 0.0       # lambda_cap -- optional allocative term; leave off unless density drifts
GEO_T_MODE = "hard"        # 'hard' | 'ramp' | 'off'; hard mask is the reasoned default
GEO_T_MAX = 0.4            # apply only where the one-step x0 decode is trustworthy
GEO_T_RAMP_K = 2.0         # exponent for t_mode='ramp'
GEO_WARMUP_STEPS = 2000    # x0_pred is garbage everywhere early on; wait for v-loss to settle
GEO_RAMP_STEPS = 1000      # then ease lambda in linearly (avoids a step-change spike-skip)
GEO_SUBSAMPLE = 0          # 0 = all G*G points; >0 caps the O(N^2) cdist (spacing term only)
GEO_SPACING_SCALE = 1.0    # r_target = scale * r_pack, r_pack = sqrt(2/sqrt(3)) ~ 1.0746
GEO_RHO_FLOOR = 1e-3       # guards the density warp where rho -> 0 (white regions)
GEO_WARP_GRID = 0          # 0 = warp against the full-res density map
GEO_STOP_GRAD_WARP = True  # gradient flows through cdist only
GEO_CAP_TAU = 5e-3         # soft-Voronoi temperature (control_gt_free: 0.02 -> 0.002)
GEO_CAP_GRID = 32          # soft-Voronoi grid for the capacity term

WANDB_ACTIVE = True

BATCH_SIZE = 16
LR = 1e-4

# Flow-Matching sampling / stability
ODE_STEPS = 50
ETA = 0.0  # 0 = deterministic ODE; > 0 = stochastic reverse SDE (1.0 = canonical)
T_SCALE = 1000.0
DEVICE = "cuda"
RESUME_LATEST = True

NUM_WORKERS = 4
PIN_MEMORY = True

WANDB_VALID_IMAGES = 8
WANDB_TRAIN_IMAGES = 8
SHOW_LABELS = True
SHOW_SELECTED_INPUTS = True
SHOW_SELECTED_GT = True
SHOW_SELECTED_PREDICT = True
SHOW_SELECTED_GT_OFFSETS = True


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
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source dir was not found: {source_dir}")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"Offsets dir is empty and target dir was not found: {target_dir}"
        )

    source_stems = set()
    for root, _, files in os.walk(source_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in VALID_EXT:
                continue
            rel_path = os.path.relpath(os.path.join(root, f), source_dir)
            source_stems.add(os.path.splitext(rel_path)[0])

    # Build target stem -> filepath map for files that have matching source stems.
    target_map = {}
    for root, _, files in os.walk(target_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXT:
                continue
            rel_path = os.path.relpath(os.path.join(root, fname), target_dir)
            stem = os.path.splitext(rel_path)[0]
            if stem in source_stems:
                target_map[stem] = os.path.join(root, fname)

    expected_stems = sorted(target_map.keys())
    if not expected_stems:
        raise RuntimeError(
            "No matching source/target stems found for offset export. "
            "Check SOURCE_DIR/TARGET_DIR filename stems."
        )

    existing_files = []
    for root, _, files in os.walk(resolved_offsets_dir):
        for f in files:
            if f.endswith(".npy"):
                existing_files.append(os.path.relpath(os.path.join(root, f), resolved_offsets_dir))
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
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        tmp_prefix = final_path + ".tmp"
        tmp_path = tmp_prefix + ".npy"
        np.save(tmp_prefix, offsets)
        os.replace(tmp_path, final_path)
        exported += 1

    final_existing_stems = set()
    for root, _, files in os.walk(resolved_offsets_dir):
        for f in files:
            if f.endswith(".npy"):
                rel_path = os.path.relpath(os.path.join(root, f), resolved_offsets_dir)
                final_existing_stems.add(os.path.splitext(rel_path)[0])
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


def sample_eval_batch(fm, denoiser, control_net, batch, device, n_samples=4,
                      ode_steps=50, ode_method="euler", t_start=1.0,
                      show_tqdm=False, tqdm_desc="sampling", conditioner=None, eta=0.0,
                      x_start=None):
    """Sample offset grids for intermediate eval via Flow-Matching ODE integration."""
    high_res_img = batch["high_res"]
    target_density = batch["target_density"]
    high_res_sdf = batch.get("high_res_sdf")
    target_sdf = batch.get("target_sdf")
    smart_init_grid = batch.get("smart_init_grid")

    h, w = target_density.shape[-2], target_density.shape[-1]
    shape = [n_samples, 2, h, w]

    if conditioner is not None:
        conditioner.set_condition(target_density, smart_init_grid if conditioner.use_smart_init_grid else None)
        conditioner.eval()

        def velocity_fn(x, t_net):
            return conditioner(x, t_net)
    else:
        controlled = DynamicControlledDenoiser(denoiser, control_net)
        controlled.set_condition(high_res_img, high_res_sdf, target_density, target_sdf, smart_init_grid)

        def velocity_fn(x, t_net):
            return controlled(x, t_net)

    denoiser.eval()
    smart_src = batch.get("smart_init_offsets")
    if x_start is None and fm.coupling == "smartinit":
        if smart_src is None:
            raise ValueError("coupling='smartinit' requires 'smart_init_offsets' in the eval batch")
        x_start = fm.start_state(shape, device=device, smart_init=smart_src.to(device))
    with torch.no_grad():
        raw = fm.ode_sample(
            velocity_fn, shape, device=device,
            n_steps=ode_steps, method=ode_method, t_start=t_start,
            x_start=x_start,
            eta=eta,
            source_mean=(smart_src.to(device) if (fm.coupling == "smartinit" and smart_src is not None) else None),
            show_tqdm=show_tqdm,
        )
    denoiser.train()
    return raw

def _grid_centers_flat(grid_size, device, dtype):
    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    return torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)


def offsets_to_coords_gpu(offsets, grid_size, grid_centers_flat):
    """Convert OT offsets (B,2,G,G) to normalized coords (B,N,2)."""
    bsz = offsets.shape[0]
    offs = offsets.permute(0, 2, 3, 1).reshape(bsz, grid_size * grid_size, 2)
    coords = grid_centers_flat.expand(bsz, -1, -1) + offs / float(grid_size)
    return coords.clamp(0.0, 1.0)


def coords_to_offsets_gpu(coords, grid_size, grid_centers_flat):
    """Convert normalized coords (B,N,2) back to OT offsets (B,2,G,G)."""
    delta = (coords - grid_centers_flat.expand(coords.shape[0], -1, -1)) * float(grid_size)
    return delta.reshape(coords.shape[0], grid_size, grid_size, 2).permute(0, 3, 1, 2).contiguous()


def apply_gpu_jitter(coords, jitter_strength_px=0.0, grid_size=32):
    """Apply Gaussian coordinate jitter in pixel units to (B,N,2) coords."""
    if jitter_strength_px <= 0.0:
        return coords
    sigma = float(jitter_strength_px) / float(grid_size)
    return (coords + torch.randn_like(coords) * sigma).clamp(0.0, 1.0)


def render_occupancy_grid_gpu(coords, grid_size=32):
    """Render (B,N,2) coords to (B,1,G,G) simple normalized occupancy map (no Gaussian)."""
    bsz, n, _ = coords.shape
    device = coords.device
    dtype = coords.dtype
    px = (coords[:, :, 0] * grid_size).clamp(0, grid_size - 1).long()
    py = (coords[:, :, 1] * grid_size).clamp(0, grid_size - 1).long()
    flat_idx = py * grid_size + px  # (B, N)
    grid_flat = torch.zeros(bsz, grid_size * grid_size, device=device, dtype=dtype)
    ones = torch.ones(bsz, n, device=device, dtype=dtype)
    grid_flat.scatter_add_(1, flat_idx, ones)
    mx = grid_flat.amax(dim=1, keepdim=True).clamp(min=1.0)
    return (grid_flat / mx).reshape(bsz, 1, grid_size, grid_size)


def render_smart_init_gpu(coords, grid_size=32, sigma_px=0.5, grid_centers_flat=None):
    """Render (B,N,2) coords to (B,1,G,G) with Gaussian soft splatting."""
    bsz = coords.shape[0]
    device = coords.device
    dtype = coords.dtype
    if grid_centers_flat is None:
        grid_centers_flat = _grid_centers_flat(grid_size, device=device, dtype=dtype)
    pixel_centers = grid_centers_flat.expand(bsz, -1, -1)

    sigma = max(float(sigma_px), 1e-4) / float(grid_size)
    dist = torch.cdist(pixel_centers, coords, p=2)
    gauss = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
    grid = gauss.amax(dim=2).reshape(bsz, 1, grid_size, grid_size)
    return grid.clamp(0.0, 1.0)


def sdf_to_display(sdf_2d):
    """Map SDF from [-1, 1] to display space [0, 1]."""
    return np.clip((sdf_2d + 1.0) * 0.5, 0.0, 1.0)


def save_val_panel(
    save_path,
    cond_batch,
    gt_offsets_batch,
    pred_offsets_batch,
    max_samples=4,
    show_labels=True,
    show_selected_inputs=True,
    show_selected_gt=True,
    show_selected_predict=True,
    show_selected_gt_offsets=True,
):
    """Save a validation panel with a configurable subset of columns.

    Columns, in order: Condition | GT | Predict | GT Offset Quiver.

    Args:
        show_labels: If False, column headers will not be displayed on the top row.
        show_selected_*: Control whether each column is visible in the panel.
    """
    if not HAS_MPL:
        print("matplotlib unavailable; skipping validation panel export")
        return False

    visible_columns = [
        ("Condition", show_selected_inputs),
        ("GT", show_selected_gt),
        ("Predict", show_selected_predict),
        ("GT Offset Quiver", show_selected_gt_offsets),
    ]
    visible_columns = [column for column in visible_columns if column[1]]
    if not visible_columns:
        raise ValueError("At least one show_selected_* flag must be True")

    n = min(max_samples, cond_batch.shape[0], gt_offsets_batch.shape[0], pred_offsets_batch.shape[0])
    if n <= 0:
        return False

    num_columns = len(visible_columns)
    fig, axes = plt.subplots(n, num_columns, figsize=(4 * num_columns, 4 * n), dpi=150)
    axes = np.array(axes, dtype=object).reshape(n, num_columns)

    for i in range(n):
        cond = cond_batch[i, 0]
        gt_offsets = gt_offsets_batch[i]
        pred_offsets = pred_offsets_batch[i]

        gt_pts_grid = to_pointset_optimal_transport(gt_offsets)
        gt_pts = gt_pts_grid.reshape(2, -1).T

        pred_pts_grid = to_pointset_optimal_transport(pred_offsets)
        pred_pts = pred_pts_grid.reshape(2, -1).T

        column_index = 0

        if show_selected_inputs:
            ax = axes[i, column_index]
            ax.imshow(cond, cmap="gray", vmin=0.0, vmax=1.0)
            if i == 0 and show_labels:
                ax.set_title("Condition")
            ax.axis("off")
            column_index += 1

        if show_selected_gt:
            ax = axes[i, column_index]
            ax.scatter(gt_pts[:, 0], 1.0 - gt_pts[:, 1], c="black", s=0.5, alpha=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            if i == 0 and show_labels:
                ax.set_title("GT")
            ax.axis("off")
            column_index += 1

        if show_selected_predict:
            ax = axes[i, column_index]
            ax.scatter(pred_pts[:, 0], 1.0 - pred_pts[:, 1], c="black", s=0.5, alpha=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            if i == 0 and show_labels:
                ax.set_title("Predict")
            ax.axis("off")
            column_index += 1

        if show_selected_gt_offsets:
            ax = axes[i, column_index]
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
            if i == 0 and show_labels:
                ax.set_title("GT Offset Quiver")
            ax.tick_params(labelsize=6)
            fig.colorbar(q, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def dynamic_collate(batch):
    """Collate samples with variable high-res image sizes by padding per batch."""
    sample0 = batch[0]

    max_h = max(sample["high_res"].shape[-2] for sample in batch)
    max_w = max(sample["high_res"].shape[-1] for sample in batch)

    padded_high_res = []
    for sample in batch:
        img = sample["high_res"]
        pad_h = max_h - img.shape[-2]
        pad_w = max_w - img.shape[-1]
        padded = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        padded_high_res.append(padded.contiguous())

    collated = {
        "high_res": torch.stack(padded_high_res, dim=0),
        "target_density": torch.stack([sample["target_density"].contiguous() for sample in batch], dim=0),
        "offsets": torch.stack([sample["offsets"].contiguous() for sample in batch], dim=0),
    }

    if "high_res_sdf" in sample0:
        padded_high_res_sdf = []
        for sample in batch:
            sdf = sample["high_res_sdf"]
            pad_h = max_h - sdf.shape[-2]
            pad_w = max_w - sdf.shape[-1]
            padded = F.pad(sdf, (0, pad_w, 0, pad_h), mode="constant", value=1.0)
            padded_high_res_sdf.append(padded.contiguous())
        collated["high_res_sdf"] = torch.stack(padded_high_res_sdf, dim=0)
        collated["target_sdf"] = torch.stack([sample["target_sdf"].contiguous() for sample in batch], dim=0)

    if "smart_init_grid" in sample0:
        collated["smart_init_grid"] = torch.stack([sample["smart_init_grid"].contiguous() for sample in batch], dim=0)
        collated["smart_init_offsets"] = torch.stack([sample["smart_init_offsets"].contiguous() for sample in batch], dim=0)

    return collated


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Paths and I/O
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
    parser.add_argument("--warm-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Build + validate the smart-init/SDF cache to disk up front, before wandb/model "
                             "setup, so epoch 1 is fast (default ON; use --no-warm-cache to skip)")
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
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO,
                        help="Train only on timesteps [0, truncation_ratio * total_timesteps)")
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
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=USE_EMA,
                        help="Tier 1.1: maintain a weight EMA and use it for eval/geometry/checkpoints")
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--ema-warmup-steps", type=int, default=EMA_WARMUP_STEPS,
                        help="Hard-copy the shadow below this many accepted steps")
    parser.add_argument("--t-dist", choices=["uniform", "logitnormal"], default=T_DIST,
                        help="Tier 1.2: timestep sampling distribution")
    parser.add_argument("--t-logitnorm-m", type=float, default=T_LOGITNORM_M)
    parser.add_argument("--t-logitnorm-s", type=float, default=T_LOGITNORM_S)
    parser.add_argument("--attn-middle", action=argparse.BooleanOptionalAction, default=ATTN_MIDDLE,
                        help="Tier 2: zero-init multi-head bottleneck attention (no positional encoding)")
    parser.add_argument("--attn-heads", type=int, default=ATTN_HEADS)
    # Geometry loss (geometry_loss.py). NOTE: unrelated to --geom-clump-weight below,
    # which weights a *reported metric* in the best-checkpoint score, not the loss.
    parser.add_argument("--geo-weight", type=float, default=GEO_WEIGHT,
                        help="lambda_geo for the density-warped spacing hinge (0 disables)")
    parser.add_argument("--geo-cap-weight", type=float, default=GEO_CAP_WEIGHT,
                        help="lambda_cap for the optional soft-capacity term (0 disables)")
    parser.add_argument("--geo-t-mode", choices=["hard", "ramp", "off"], default=GEO_T_MODE,
                        help="w(t): 'hard' masks t<t_max; 'ramp' uses (1-t)^k; 'off' is uniform")
    parser.add_argument("--geo-t-max", type=float, default=GEO_T_MAX)
    parser.add_argument("--geo-t-ramp-k", type=float, default=GEO_T_RAMP_K)
    parser.add_argument("--geo-warmup-steps", type=int, default=GEO_WARMUP_STEPS)
    parser.add_argument("--geo-ramp-steps", type=int, default=GEO_RAMP_STEPS)
    parser.add_argument("--geo-subsample", type=int, default=GEO_SUBSAMPLE)
    parser.add_argument("--geo-spacing-scale", type=float, default=GEO_SPACING_SCALE)
    parser.add_argument("--geo-rho-floor", type=float, default=GEO_RHO_FLOOR)
    parser.add_argument("--geo-warp-grid", type=int, default=GEO_WARP_GRID)
    parser.add_argument("--geo-stop-grad-warp", action=argparse.BooleanOptionalAction,
                        default=GEO_STOP_GRAD_WARP)
    parser.add_argument("--geo-cap-tau", type=float, default=GEO_CAP_TAU)
    parser.add_argument("--geo-cap-grid", type=int, default=GEO_CAP_GRID)
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
    parser.add_argument("--ode-steps", type=int, default=ODE_STEPS,
                        help="Number of Euler/Heun steps for Flow-Matching ODE eval sampling")
    parser.add_argument("--ode-method", choices=["euler", "heun"], default=ODE_METHOD,
                        help="ODE integrator for Flow-Matching sampling")
    parser.add_argument("--eta", type=float, default=ETA,
                        help="Stochasticity for eval sampling: 0 = deterministic ODE, "
                             "1.0 = canonical reverse SDE. Inference-time only, no retraining. "
                             "Needs --fm-source-jitter-px > 0 when --fm-coupling smartinit.")
    parser.add_argument("--t-start", type=float, default=1.0,
                        help="Starting t for ODE sampling (1.0 = full noise->data)")
    parser.add_argument("--t-scale", type=float, default=T_SCALE,
                        help="Scale applied to continuous t before sinusoidal timestep embedding")
    parser.add_argument("--fm-coupling", choices=["gaussian", "smartinit"], default=FM_COUPLING,
                        help="FM source endpoint: 'gaussian' (noise->data) or 'smartinit' (smart-init->data coupling)")
    parser.add_argument("--fm-source-jitter-px", type=float, default=0.0,
                        help="Gaussian jitter (offset/pixel units) on the smart-init source during training (smartinit coupling)")
    parser.add_argument("--conditioning", choices=["controlnet", "concat", "spade"], default=CONDITIONING,
                        help="Conditioning architecture: 'controlnet' (dual-branch), 'concat' or 'spade' (single-stage, attention-free)")
    parser.add_argument("--concat-smart-init-grid", action=argparse.BooleanOptionalAction, default=False,
                        help="(concat/spade) also feed the smart-init occupancy grid as a condition channel")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max global grad-norm for clipping")
    parser.add_argument("--spike-factor", type=float, default=25.0,
                        help="Skip an update if loss > spike_factor * EMA(loss) (0 disables)")
    parser.add_argument("--spike-ema-beta", type=float, default=0.05,
                        help="EMA smoothing for the spike-detection loss average")
    parser.add_argument("--lr-warmup", type=int, default=200,
                        help="Linear LR warmup steps")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="Final LR for cosine decay")
    parser.add_argument("--lr-schedule", action=argparse.BooleanOptionalAction, default=True,
                        help="Cosine LR decay after warmup (use --no-lr-schedule for constant LR)")
    args = parser.parse_args()
    run(args=args)


def run(args):
    if args.wandb_valid_images < 0:
        raise ValueError("--wandb-valid-images must be >= 0")
    if args.wandb_train_images < 0:
        raise ValueError("--wandb-train-images must be >= 0")
    if args.save_every <= 0:
        raise ValueError("--save_every must be >= 1")
    if not (0.0 < args.truncation_ratio <= 1.0):
        raise ValueError("--truncation-ratio must be in (0, 1]")
    if not (args.show_selected_inputs or args.show_selected_gt or args.show_selected_predict or args.show_selected_gt_offsets):
        raise ValueError("At least one of --show-selected-inputs, --show-selected-gt, --show-selected-predict, or --show-selected-gt-offsets must be enabled")
    args.offsets = ensure_offsets_dir(args.source, args.target, args.offsets, args.grid_size)

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # ── build/validate the on-disk cache up front (BEFORE wandb/model setup) ──
    if args.warm_cache:
        _warm_cache_dir = args.cache_data_dir
        if not (args.smart_init_features or args.sdf_features or args.fm_coupling == "smartinit"):
            _warm_cache_dir = None
        elif not _warm_cache_dir:
            _warm_cache_dir = os.path.join(args.out, "cache_data")
        if _warm_cache_dir:
            warm_ds = DynamicStippleDataset(
                args.source, args.offsets,
                grid_size=args.grid_size,
                sdf_truncate_px=args.sdf_truncate_px,
                cache_data_dir=_warm_cache_dir,
                smart_init_seed=args.smart_init_seed,
                smart_init_features=args.smart_init_features,
                sdf_features=args.sdf_features,
                provide_smart_init_source=(args.fm_coupling == "smartinit"),
                preload_ram=False,
            )
            print(f"[warm-cache] building/validating cache for {len(warm_ds)} samples -> {_warm_cache_dir}")
            for _i in tqdm(range(len(warm_ds)), desc="warm-cache"):
                _ = warm_ds[_i]  # first touch computes+saves smart-init (and SDF); cached entries just load
            print("[warm-cache] complete. Proceeding to wandb/model setup and training.")

    # ── wandb ────────────────────────────────────────────────────────
    use_wandb = WANDB_ACTIVE
    if use_wandb:
        try:
            import wandb
            load_wandb_key()
            run_name = datetime.now().strftime("fm-train-%Y%m%d-%H%M%S")
            wandb.init(
                project="Stipple-ControlNet",
                name=run_name,
                config=vars(args),
            )
            # Log metrics organized by section: metrics, chart, visual.
            wandb.define_metric("epoch")
            wandb.define_metric("metrics/train_loss", step_metric="epoch")
            wandb.define_metric("metrics/valid_loss", step_metric="epoch")
            wandb.define_metric("metrics/geo_cv", step_metric="epoch")
            wandb.define_metric("metrics/geo_clumped_pct", step_metric="epoch")
            wandb.define_metric("metrics/geo_score", step_metric="epoch")
            wandb.define_metric("visual/*", step_metric="epoch")
            print(f"wandb run name: {run_name}")
        except ImportError:
            print("wandb not installed, logging disabled")
            use_wandb = False

    # ── build Flow-Matching velocity network from scratch (no diffusion baseline) ──
    fm = FlowMatching(device=device, t_scale=args.t_scale,
                      coupling=args.fm_coupling, source_jitter_px=args.fm_source_jitter_px,
                      t_dist=args.t_dist, t_logitnorm_m=args.t_logitnorm_m,
                      t_logitnorm_s=args.t_logitnorm_s,
                      min_snr_gamma=args.min_snr_gamma)

    geo_loss = GeometryLoss(
        weight=args.geo_weight, cap_weight=args.geo_cap_weight,
        t_mode=args.geo_t_mode, t_max=args.geo_t_max, t_ramp_k=args.geo_t_ramp_k,
        warmup_steps=args.geo_warmup_steps, ramp_steps=args.geo_ramp_steps,
        subsample=args.geo_subsample, spacing_scale=args.geo_spacing_scale,
        rho_floor=args.geo_rho_floor, warp_grid=args.geo_warp_grid,
        stop_grad_warp=args.geo_stop_grad_warp,
        cap_tau=args.geo_cap_tau, cap_grid=args.geo_cap_grid,
    )
    print(geo_loss.describe())

    conditioner = None
    if args.conditioning in ("concat", "spade"):
        # Single-stage, attention-free conditioning; no ControlNet dual-branch.
        n_cond = 1 + (1 if args.concat_smart_init_grid else 0)
        if args.conditioning == "spade":
            from control_fm_v2.single_stage import build_spade_velocity_network, SPADEConditioner
            denoiser = build_spade_velocity_network(
                cond_channels=n_cond, device=device,
                attn_middle=args.attn_middle, attn_heads=args.attn_heads)
            conditioner = SPADEConditioner(denoiser, use_smart_init_grid=args.concat_smart_init_grid).to(device)
        else:
            from control_fm_v2.single_stage import build_conditional_velocity_network, SingleStageConditioner
            denoiser = build_conditional_velocity_network(
                extra_in_channels=n_cond, device=device,
                attn_middle=args.attn_middle, attn_heads=args.attn_heads)
            conditioner = SingleStageConditioner(denoiser, use_smart_init_grid=args.concat_smart_init_grid).to(device)
        conditioner.train()
        denoiser.train()
        control_net = None
        train_module = conditioner
    else:
        # Tier 2: attention is attached to denoiser.middle HERE, before DynamicControlNet
        # deep-copies that middle -- otherwise the control branch silently lacks it.
        denoiser = build_velocity_network(
            device=device, attn_middle=args.attn_middle, attn_heads=args.attn_heads)

        # NOTE: Create control_net BEFORE freezing denoiser so deep copies have requires_grad=True
        control_net = DynamicControlNet(
            denoiser,
            grid_size=args.grid_size,
            enable_gecco=args.enable_gecco,
            smart_init_features=args.smart_init_features,
            sdf_features=args.sdf_features,
            batch_coords_features=args.batch_coords_features,
            enable_adaptive_gate_injection=args.enable_adaptive_gate_injection,
        ).to(device)
        train_module = control_net
        train_module.train()

        # Freeze/unfreeze denoiser based on flag (AFTER control_net creation)
        if args.freeze_denoiser:
            for p in denoiser.parameters():
                p.requires_grad = False
        else:
            denoiser.train()

    if conditioner is not None:
        trainable_total = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        print(f"Conditioning architecture             : {args.conditioning} (single-stage, attention-free)")
        print(f"Trainable single-stage velocity net   : {trainable_total:,}")
    else:
        control_params = sum(p.numel() for p in control_net.parameters() if p.requires_grad)
        denoiser_params = sum(p.numel() for p in denoiser.parameters())
        if args.freeze_denoiser:
            print(f"Trainable ControlNet params           : {control_params:,}")
            print(f"Frozen denoiser params                : {denoiser_params:,}")
        else:
            trainable_total = control_params + sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
            print(f"Trainable params (control + denoiser) : {trainable_total:,}")
    print(f"Grid size                             : {args.grid_size}x{args.grid_size}")
    print(f"GECCO dynamic features enabled        : {args.enable_gecco}")
    print(f"Adaptive gate injection enabled       : {args.enable_adaptive_gate_injection}")
    print(f"ODE sampling steps                    : {args.ode_steps}")
    print(f"ODE method                            : {args.ode_method}")
    print(f"t-scale (timestep embedding)          : {args.t_scale}")
    print(f"t distribution (Tier 1.2)             : {args.t_dist}"
          + (f" (m={args.t_logitnorm_m}, s={args.t_logitnorm_s})" if args.t_dist == "logitnormal" else ""))
    print(f"min-SNR gamma (Tier 1.3)              : {args.min_snr_gamma}"
          + ("  [disabled]" if args.min_snr_gamma <= 0 else ""))
    print(f"Bottleneck attention (Tier 2)         : {args.attn_middle}"
          + (f" ({args.attn_heads} heads, zero-init, no pos-enc)" if args.attn_middle else ""))
    print(f"Smart Init features enabled           : {args.smart_init_features}")
    print(f"SDF features enabled                  : {args.sdf_features}")
    print(f"Batch coords features enabled         : {args.batch_coords_features}")
    print(f"Min-SNR gamma                         : {args.min_snr_gamma}")
    print(f"SDF truncation (px)                   : {args.sdf_truncate_px}")
    print(f"Smart Init micro-jitter (train, px)  : {args.smart_init_jitter_px}")
    print(f"Smart Init soft-splat sigma (px)     : {args.smart_init_splat_sigma_px}")
    print(f"Smart Init jitter enabled            : {args.enable_smart_init_jitter}")
    print(f"Smart Init splat-sigma enabled       : {args.enable_smart_init_splat_sigma}")

    # ── dataset ──────────────────────────────────────────────────────
    cache_data_dir = args.cache_data_dir
    if not (args.smart_init_features or args.sdf_features or args.fm_coupling == "smartinit"):
        cache_data_dir = None
    elif not cache_data_dir:
        cache_data_dir = os.path.join(args.out, "cache_data")
    dataset = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        provide_smart_init_source=(args.fm_coupling == "smartinit"),
        preload_ram=False,  # Initial scan only, no preload needed
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "DynamicStippleDataset has 0 samples. Ensure source images and offsets share matching "
            "relative stems (including subfolders), e.g. source/a/b/img.png with offsets/a/b/img.npy."
        )
    val_len = int(len(dataset) * args.val_split)
    val_len = min(max(val_len, 0), max(len(dataset) - 1, 0))
    train_len = len(dataset) - val_len

    all_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = all_indices[:train_len]
    val_indices = all_indices[train_len:]

    train_filenames = [dataset.filenames[i] for i in train_indices]
    train_dataset = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        provide_smart_init_source=(args.fm_coupling == "smartinit"),
        filenames=train_filenames,
        preload_ram=args.preload_ram,
    )

    val_dataset = None
    if val_len > 0:
        val_filenames = [dataset.filenames[i] for i in val_indices]
        val_dataset = DynamicStippleDataset(
            args.source,
            args.offsets,
            grid_size=args.grid_size,
            sdf_truncate_px=args.sdf_truncate_px,
            cache_data_dir=cache_data_dir,
            smart_init_seed=args.smart_init_seed,
            smart_init_features=args.smart_init_features,
            sdf_features=args.sdf_features,
            provide_smart_init_source=(args.fm_coupling == "smartinit"),
            filenames=val_filenames,
            preload_ram=args.preload_ram,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dynamic_collate,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=dynamic_collate,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    print(f"Dataset split: train={train_len}, val={val_len}")
    grid_centers_flat = _grid_centers_flat(args.grid_size, device=device, dtype=torch.float32) if args.smart_init_features else None

    # ── optimizer ────────────────────────────────────────────────────
    if conditioner is not None:
        optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.lr)
    elif args.freeze_denoiser:
        optimizer = torch.optim.AdamW(control_net.parameters(), lr=args.lr)
    else:
        all_params = list(control_net.parameters()) + list(denoiser.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=args.lr)

    total_opt_steps = max(1, args.epochs * max(1, len(train_loader)))

    def _lr_lambda(cur):
        if cur < args.lr_warmup:
            return float(cur + 1) / float(max(1, args.lr_warmup))
        if not args.lr_schedule:
            return 1.0
        progress = float(cur - args.lr_warmup) / float(max(1, total_opt_steps - args.lr_warmup))
        progress = min(1.0, max(0.0, progress))
        floor = (args.lr_min / args.lr) if args.lr > 0 else 0.0
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── Tier 1.1: weight EMA over every jointly-trained module ────────────────
    # controlnet path: control_net and denoiser are separate modules, both optimised.
    # concat/spade path: train_module IS the conditioner and already holds the network.
    ema = None
    if args.use_ema:
        ema_targets = {"control_net": train_module}
        if control_net is not None:
            ema_targets["denoiser"] = denoiser
        ema = WeightEMA(ema_targets, decay=args.ema_decay, warmup_steps=args.ema_warmup_steps)
        print(f"Weight EMA enabled                    : decay={args.ema_decay} warmup={args.ema_warmup_steps}")

    def ema_ctx():
        """Swap EMA weights in for eval/geometry/sampling. THE bug this prevents: keeping an
        EMA but still sampling from the raw weights."""
        return ema.averaged() if ema is not None else contextlib.nullcontext()

    # ── optional resume from latest checkpoint ───────────────────────
    start_epoch = 0
    global_step = 0
    loss_ema = None
    n_skipped = 0
    best_geom_score = float("inf")
    last_geom = {"cv": float("nan"), "clumped_pct": float("nan"), "score": float("nan")}
    checkpoints_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_geom_ckpt_path = None
    epoch_history = []
    train_epoch_history = []
    val_epoch_history = []
    if args.resume_latest:
        ckpt_re = re.compile(r"^dynamic_controlnet_fm_ep(\d+)\.pt$")
        latest_path = None
        latest_epoch_num = -1
        for fname in os.listdir(checkpoints_dir):
            match = ckpt_re.match(fname)
            if not match:
                continue
            ep_num = int(match.group(1))
            if ep_num > latest_epoch_num:
                latest_epoch_num = ep_num
                latest_path = os.path.join(checkpoints_dir, fname)

        if latest_path is None:
            print("Resume requested but no checkpoint found. Starting from scratch.")
        else:
            state = torch.load(latest_path, map_location="cpu")
            if conditioner is not None:
                train_module.load_state_dict(state["control_net"], strict=False)
            else:
                control_net.safe_load_state_dict(state, strict=False)
                if state.get("denoiser") is not None:
                    denoiser.load_state_dict(state["denoiser"], strict=False)
                else:
                    print(
                        "WARNING: checkpoint contains no 'denoiser' weights. The jointly-trained "
                        "base U-Net will RESTART FROM RANDOM INIT. (Checkpoints written before the "
                        "denoiser-saving fix are affected.)"
                    )
            if "optimizer" in state and state["optimizer"] is not None:
                optimizer.load_state_dict(state["optimizer"])
            if ema is not None:
                if state.get("ema") is not None:
                    ema.load_state_dict(state["ema"])
                    print("  -> restored EMA shadow from checkpoint")
                else:
                    print("  -> WARNING: checkpoint has no EMA shadow; EMA restarts from the "
                          "resumed weights (it will re-warm over --ema-warmup-steps).")
            global_step = int(state.get("global_step", 0))
            best_geom_score = float(state.get("best_geom_score", best_geom_score))
            # epoch in checkpoint is 0-based. Continue from next epoch.
            start_epoch = int(state.get("epoch", latest_epoch_num - 1)) + 1
            print(
                f"Resumed from checkpoint: {latest_path} | "
                f"start_epoch={start_epoch} | global_step={global_step}"
            )

    # ── training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        should_save_epoch = ((epoch + 1) % args.save_every == 0) or ((epoch + 1) == args.epochs)
        epoch_loss = 0.0
        epoch_geo = {"spacing": 0.0, "capacity": 0.0, "total": 0.0, "frac_active": 0.0, "n": 0}
        preview_batch = None

        train_module.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=False)
        for batch in train_pbar:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            high_res_img = batch["high_res"]
            target_density = batch["target_density"]
            x_0 = batch["offsets"]
            high_res_sdf = batch.get("high_res_sdf")
            target_sdf = batch.get("target_sdf")
            smart_init_grid = batch.get("smart_init_grid")
            smart_init_offsets = batch.get("smart_init_offsets")

            if preview_batch is None:
                keep_train = max(1, min(args.wandb_train_images, high_res_img.shape[0]))
                preview_batch = {
                    key: value[:keep_train].detach()
                    for key, value in batch.items()
                    if torch.is_tensor(value)
                }

                if should_save_epoch and use_wandb and HAS_MPL and args.sdf_features and args.smart_init_features:
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=140)
                    axes[0].imshow(preview_batch["target_density"][0, 0].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                    axes[0].set_title("target_density")
                    axes[0].axis("off")
                    axes[1].imshow(sdf_to_display(preview_batch["target_sdf"][0, 0].cpu().numpy()), cmap="gray", vmin=0.0, vmax=1.0)
                    axes[1].set_title("target_sdf")
                    axes[1].axis("off")
                    axes[2].imshow(preview_batch["smart_init_grid"][0, 0].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                    axes[2].set_title("smart_init_grid")
                    axes[2].axis("off")
                    plt.tight_layout()
                    debug_path = os.path.join(args.out, "wandb_hint_channels_first_batch.png")
                    plt.savefig(debug_path, dpi=140, bbox_inches="tight")
                    plt.close()
                    wandb.log({
                        "epoch": epoch + 1,
                        "visual/hint_channels": wandb.Image(debug_path),
                    }, step=epoch + 1)

            if args.smart_init_features:
                smart_coords = None
                if args.enable_smart_init_jitter or args.enable_smart_init_splat_sigma:
                    smart_coords = offsets_to_coords_gpu(smart_init_offsets, args.grid_size, grid_centers_flat)

                if args.enable_smart_init_jitter:
                    smart_coords = apply_gpu_jitter(
                        smart_coords,
                        jitter_strength_px=args.smart_init_jitter_px,
                        grid_size=args.grid_size,
                    )
                    smart_init_offsets = coords_to_offsets_gpu(smart_coords, args.grid_size, grid_centers_flat)

                if args.enable_smart_init_splat_sigma:
                    smart_init_grid = render_smart_init_gpu(
                        smart_coords,
                        grid_size=args.grid_size,
                        sigma_px=args.smart_init_splat_sigma_px,
                        grid_centers_flat=grid_centers_flat,
                    )
                elif args.enable_smart_init_jitter:
                    smart_init_grid = render_occupancy_grid_gpu(smart_coords, grid_size=args.grid_size)

            if not args.sdf_features:
                high_res_sdf = None
                target_sdf = None

            t = fm.sample_t(x_0.shape[0], device=device)
            # source endpoint (t=1): pure noise (gaussian) OR smart-init offsets (smartinit coupling)
            source = fm.sample_source(x_0, smart_init=smart_init_offsets)
            offsets_t = fm.interpolate(x_0, source, t)
            v_target = fm.velocity_target(x_0, source)
            t_net = fm.net_t(t)
            if conditioner is not None:
                conditioner.set_condition(
                    target_density,
                    smart_init_grid if args.concat_smart_init_grid else None,
                )
                v_pred = conditioner(offsets_t, t_net)
            else:
                controls = control_net(
                    offsets_t,
                    t_net,
                    high_res_img,
                    target_density,
                    high_res_sdf=high_res_sdf,
                    target_sdf_map=target_sdf,
                    target_smart_init_map=smart_init_grid,
                )
                v_pred = denoiser(offsets_t, t_net, controls=controls)

            per_sample_mse = F.mse_loss(v_pred, v_target, reduction="none")
            per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))
            # Tier 1.3: min-SNR weighting, derived for THIS interpolant (see
            # flow_matching.FlowMatching.loss_weight). Returns ones when disabled.
            snr_w = fm.loss_weight(t)
            denoise_loss = (snr_w * per_sample_mse).mean()
            loss = denoise_loss

            # Auxiliary geometry loss on the one-step-decoded points. Returns None (not a
            # zero tensor) while inside warmup or when the t-mask empties the batch.
            geo_comps = {}
            if geo_loss.enabled:
                x0_pred = fm.x0_from_velocity(offsets_t, t, v_pred)
                geo_total, geo_comps = geo_loss(x0_pred, t, target_density, global_step)
                if geo_total is not None:
                    loss = loss + geo_total
                    epoch_geo["n"] += 1
                    for _k in ("spacing", "capacity", "total", "frac_active"):
                        if "geo/" + _k in geo_comps:
                            epoch_geo[_k] += geo_comps["geo/" + _k]

            loss_val = loss.item()
            # Spike detection tracks the VELOCITY loss, not the total: the geometry term
            # legitimately raises the total when it ramps in, and a step-change there would
            # otherwise be misread as a bad batch and skipped.
            denoise_val = denoise_loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if conditioner is not None:
                params_to_opt = list(denoiser.parameters())
            elif args.freeze_denoiser:
                params_to_opt = list(control_net.parameters())
            else:
                params_to_opt = list(control_net.parameters()) + list(denoiser.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_opt, args.grad_clip)

            grads_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in params_to_opt)
            is_spike = (
                args.spike_factor > 0
                and loss_ema is not None
                and denoise_val > args.spike_factor * loss_ema
            )
            if (not math.isfinite(loss_val)) or (not grads_ok) or is_spike:
                optimizer.zero_grad(set_to_none=True)
                n_skipped += 1
                train_pbar.set_postfix(loss=f"{loss_val:.4f}", skipped=n_skipped)
            else:
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update()
                loss_ema = denoise_val if loss_ema is None else (1.0 - args.spike_ema_beta) * loss_ema + args.spike_ema_beta * denoise_val
                epoch_loss += loss_val
                train_pbar.set_postfix(loss=f"{loss_val:.6f}")
            global_step += 1

        avg_loss = epoch_loss / max(len(train_loader), 1)

        if should_save_epoch and args.wandb_train_images > 0 and preview_batch is not None and preview_batch["high_res"].shape[0] > 0:
            train_module.eval()
            with ema_ctx():
                train_pred_raw = sample_eval_batch(
                fm,
                denoiser,
                control_net,
                preview_batch,
                device,
                n_samples=preview_batch["high_res"].shape[0],
                ode_steps=args.ode_steps,
                ode_method=args.ode_method,
                show_tqdm=True,
                tqdm_desc=f"Epoch {epoch+1}/{args.epochs} [train-predict]",
                t_start=args.t_start,
                eta=args.eta,
                    conditioner=conditioner,
                )
            train_panel_path = os.path.join(args.out, f"train_panel_ep{epoch+1}.png")
            train_saved = save_val_panel(
                train_panel_path,
                preview_batch["high_res"].cpu().numpy(),
                preview_batch["offsets"].cpu().numpy(),
                train_pred_raw.cpu().numpy(),
                max_samples=args.wandb_train_images,
                show_labels=args.show_labels,
                show_selected_inputs=args.show_selected_inputs,
                show_selected_gt=args.show_selected_gt,
                show_selected_predict=args.show_selected_predict,
                show_selected_gt_offsets=args.show_selected_gt_offsets,
            )
            if train_saved:
                print(f"  -> saved train panel: {train_panel_path}")
                if use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "visual/train": wandb.Image(train_panel_path),
                    }, step=epoch + 1)
            train_module.train()

        # Validation loop with tqdm.
        val_avg_loss = None
        val_preview_batch = None
        pred_raw_for_geom = None
        if val_loader is not None:
            train_module.eval()
            val_loss_sum = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                    high_res_img = batch["high_res"]
                    target_density = batch["target_density"]
                    high_res_sdf = batch.get("high_res_sdf")
                    target_sdf = batch.get("target_sdf")
                    x_0 = batch["offsets"]
                    smart_init_grid = batch.get("smart_init_grid")
                    smart_init_offsets = batch.get("smart_init_offsets")

                    if args.smart_init_features and args.enable_smart_init_splat_sigma and smart_init_offsets is not None:
                        smart_coords = offsets_to_coords_gpu(smart_init_offsets, args.grid_size, grid_centers_flat)
                        smart_init_grid = render_smart_init_gpu(
                            smart_coords,
                            grid_size=args.grid_size,
                            sigma_px=args.smart_init_splat_sigma_px,
                            grid_centers_flat=grid_centers_flat,
                        )

                    if not args.sdf_features:
                        high_res_sdf = None
                        target_sdf = None

                    if val_preview_batch is None:
                        keep = min(args.wandb_valid_images, high_res_img.shape[0])
                        val_preview_batch = {
                            key: value[:keep].detach()
                            for key, value in batch.items()
                            if torch.is_tensor(value)
                        }
                        if args.smart_init_features and args.enable_smart_init_splat_sigma and val_preview_batch.get("smart_init_offsets") is not None:
                            smart_coords = offsets_to_coords_gpu(val_preview_batch["smart_init_offsets"], args.grid_size, grid_centers_flat)
                            val_preview_batch["smart_init_grid"] = render_smart_init_gpu(
                                smart_coords,
                                grid_size=args.grid_size,
                                sigma_px=args.smart_init_splat_sigma_px,
                                grid_centers_flat=grid_centers_flat,
                            )

                    t = fm.sample_t(x_0.shape[0], device=device)
                    # source endpoint (t=1): pure noise (gaussian) OR smart-init offsets (smartinit coupling)
                    source = fm.sample_source(x_0, smart_init=smart_init_offsets)
                    offsets_t = fm.interpolate(x_0, source, t)
                    v_target = fm.velocity_target(x_0, source)
                    t_net = fm.net_t(t)
                    if conditioner is not None:
                        conditioner.set_condition(
                            target_density,
                            smart_init_grid if args.concat_smart_init_grid else None,
                        )
                        v_pred = conditioner(offsets_t, t_net)
                    else:
                        controls = control_net(
                            offsets_t,
                            t_net,
                            high_res_img,
                            target_density,
                            high_res_sdf=high_res_sdf,
                            target_sdf_map=target_sdf,
                            target_smart_init_map=smart_init_grid,
                        )
                        v_pred = denoiser(offsets_t, t_net, controls=controls)

                    # Deliberately UNWEIGHTED: keeps val curves comparable across
                    # --min-snr-gamma settings.
                    per_sample_mse = F.mse_loss(v_pred, v_target, reduction="none")
                    per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))
                    val_loss = per_sample_mse.mean()

                    val_loss_sum += val_loss.item()
                    val_pbar.set_postfix(loss=f"{val_loss.item():.6f}")

            val_avg_loss = val_loss_sum / max(len(val_loader), 1)
            train_module.train()

            if should_save_epoch and args.wandb_valid_images > 0:
                # Export per-epoch qualitative val panel (N samples, 4 columns).
                train_module.eval()
                with ema_ctx():
                    pred_raw = sample_eval_batch(
                    fm,
                    denoiser,
                    control_net,
                    val_preview_batch,
                    device,
                    n_samples=val_preview_batch["high_res"].shape[0],
                    ode_steps=args.ode_steps,
                    ode_method=args.ode_method,
                    show_tqdm=True,
                    tqdm_desc=f"Epoch {epoch+1}/{args.epochs} [predict]",
                    t_start=args.t_start,
                    eta=args.eta,
                        conditioner=conditioner,
                    )
                pred_raw_for_geom = pred_raw
                panel_path = os.path.join(args.out, f"val_panel_ep{epoch+1}.png")
                saved = save_val_panel(
                    panel_path,
                    val_preview_batch["high_res"].cpu().numpy(),
                    val_preview_batch["offsets"].cpu().numpy(),
                    pred_raw.cpu().numpy(),
                    max_samples=args.wandb_valid_images,
                    show_labels=args.show_labels,
                    show_selected_inputs=args.show_selected_inputs,
                    show_selected_gt=args.show_selected_gt,
                    show_selected_predict=args.show_selected_predict,
                    show_selected_gt_offsets=args.show_selected_gt_offsets,
                )
                if saved:
                    print(f"  -> saved validation panel: {panel_path}")
                    if use_wandb:
                        wandb.log({
                            "epoch": epoch + 1,
                            "visual/valid": wandb.Image(panel_path),
                        }, step=epoch + 1)
                train_module.train()

            # Geometry-gated best checkpoint based on CV + clumped% score.
            # Only compute geometry on epochs where we save checkpoints.
            if should_save_epoch and val_preview_batch is not None and val_preview_batch["high_res"].shape[0] > 0:
                train_module.eval()
                if pred_raw_for_geom is None:
                    with ema_ctx():
                        pred_raw_for_geom = sample_eval_batch(
                        fm,
                        denoiser,
                        control_net,
                        {key: value[:1] for key, value in val_preview_batch.items()},
                        device,
                        n_samples=1,
                        ode_steps=args.ode_steps,
                        ode_method=args.ode_method,
                        show_tqdm=True,
                        tqdm_desc=f"Epoch {epoch+1}/{args.epochs} [geom]",
                        t_start=args.t_start,
                        eta=args.eta,
                            conditioner=conditioner,
                        )

                pred_pointsets = []
                for raw_sample in pred_raw_for_geom:
                    pts = to_pointset_optimal_transport(raw_sample.detach().cpu().numpy())
                    pts = pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T
                    pred_pointsets.append(pts)

                geom = geometric_validation_score(
                    pred_pointsets,
                    clump_weight=args.geom_clump_weight,
                )
                last_geom = geom

                cv_ok = geom["cv"] <= args.best_max_cv
                clump_ok = geom["clumped_pct"] <= args.best_max_clumped_pct
                geom_score = float(geom["score"])
                print(
                    "  -> geom "
                    f"CV={geom['cv']:.4f} | Clumped={geom['clumped_pct']:.2f}% | Score={geom_score:.4f}"
                )

                if cv_ok and clump_ok and geom_score < best_geom_score:
                    best_geom_score = geom_score
                    best_name = (
                        f"best_controlnet_ep{epoch+1:04d}"
                        f"_score{best_geom_score:.3f}"
                        f"_cv{geom['cv']:.3f}"
                        f"_clumped{geom['clumped_pct']:.2f}.pt"
                    )
                    new_best_path = os.path.join(checkpoints_dir, best_name)
                    best_payload = {
                        # Architecture identity -- the loader refuses a mismatching rebuild.
                        "arch": arch_from(args),
                        "control_net": train_module.state_dict(),
                        # controlnet path trains the base U-Net jointly but keeps it outside
                        # control_net, so it must be saved separately or it is lost.
                        "denoiser": (denoiser.state_dict() if control_net is not None else None),
                        "control_net_ema": (ema.shadow["control_net"] if ema is not None else None),
                        "denoiser_ema": (ema.shadow.get("denoiser") if ema is not None else None),
                        "ema": (ema.state_dict() if ema is not None else None),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_geom_score": best_geom_score,
                        "cv_score": float(geom["cv"]),
                        "clumped_score": float(geom["clumped_pct"]),
                        "current_geom_score": geom_score,
                    }
                    torch.save(best_payload, new_best_path)

                    if best_geom_ckpt_path is not None and os.path.exists(best_geom_ckpt_path):
                        try:
                            os.remove(best_geom_ckpt_path)
                        except OSError:
                            pass
                    best_geom_ckpt_path = new_best_path
                    print(f"  -> new best-geom checkpoint: {new_best_path}")

                if use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "metrics/geo_cv": float(geom["cv"]),
                            "metrics/geo_clumped_pct": float(geom["clumped_pct"]),
                            "metrics/geo_score": geom_score,
                        },
                        step=epoch + 1
                    )
                train_module.train()

        if use_wandb:
            epoch_history.append(epoch + 1)
            train_epoch_history.append(float(avg_loss))
            val_epoch_history.append(float(val_avg_loss) if val_avg_loss is not None else float("nan"))

            # Log train loss separately
            wandb.log({"epoch": epoch + 1, "metrics/train_loss": avg_loss}, step=epoch + 1)

            # Geometry-loss components (training surrogate only -- never a reported result)
            if epoch_geo["n"] > 0:
                _n = epoch_geo["n"]
                wandb.log({
                    "epoch": epoch + 1,
                    "geo/spacing": epoch_geo["spacing"] / _n,
                    "geo/capacity": epoch_geo["capacity"] / _n,
                    "geo/total": epoch_geo["total"] / _n,
                    "geo/frac_active": epoch_geo["frac_active"] / _n,
                    "geo/scale": geo_loss.scale(global_step),
                }, step=epoch + 1)

            # Log val loss separately if available
            if val_avg_loss is not None:
                wandb.log({"epoch": epoch + 1, "metrics/valid_loss": val_avg_loss}, step=epoch + 1)

            # Log compare chart separately
            if len(epoch_history) > 0:
                compare_chart = wandb.plot.line_series(
                    xs=epoch_history,
                    ys=[train_epoch_history, val_epoch_history],
                    keys=["train", "Valid"],
                    title="Train vs Valid Loss Across Epochs",
                    xname="epoch",
                )
                wandb.log({"epoch": epoch + 1, "visual/compare": compare_chart}, step=epoch + 1)
        if val_avg_loss is None:
            print(f"Epoch {epoch:>4d}  |  train loss = {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch:>4d}  |  train loss = {avg_loss:.6f}  |  val loss = {val_avg_loss:.6f}")

        if should_save_epoch:
            save_path = os.path.join(checkpoints_dir, f"dynamic_controlnet_fm_ep{epoch+1}.pt")
            torch.save({
                # Architecture identity -- the loader refuses a mismatching rebuild.
                "arch": arch_from(args),
                "control_net": train_module.state_dict(),
                "denoiser": (denoiser.state_dict() if control_net is not None else None),
                "control_net_ema": (ema.shadow["control_net"] if ema is not None else None),
                "denoiser_ema": (ema.shadow.get("denoiser") if ema is not None else None),
                "ema": (ema.state_dict() if ema is not None else None),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_geom_score": best_geom_score,
                "cv_score": float(last_geom["cv"]),
                "clumped_score": float(last_geom["clumped_pct"]),
                "current_geom_score": float(last_geom["score"]),
            }, save_path)
            print(f"  -> saved {save_path}")

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
