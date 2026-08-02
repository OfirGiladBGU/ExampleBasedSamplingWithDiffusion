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
from control_v4.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_v4.DynamicStippleDataset import DynamicStippleDataset
from control_v4_mix.DynamicControlNetMultiStyle import (
    DynamicControlNetMultiStyle,
    DynamicControlledMultiStyleDenoiser,
)
from control_v4_mix.MultiStyleStippleDataset import MultiStyleStippleDataset
from control_v4_mix.data_split import source_train_val_split, split_from_manifest
from control_v4.smart_init import add_noise_at_t
from control_v4.smart_init import build_smart_init_from_image
from data.Transforms import to_image_optimal_transport, to_pointset_optimal_transport
from utils.stippling_metrics import geometric_validation_score

# ?????? default globals (edit here for quick experiments) ?????????????????????????????????????????????

# Paths and I/O
WANDB_ENV = "/groups/asharf_group/ofirgila/projection-conditioned-point-cloud-diffusion/.env"

BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = "config/GBN/model.ckpt"

# ICONS 1024 GBN
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN/target"

# OUTPUT_DIR = "control_v4/train_outputs_icons50_512_GBN"
# FREEZE_DENOISER = True  # TODO: Add in the other scripts
# RESAMPLE_JUMPS = 2

OUTPUT_DIR = "control_v4_mix/train_outputs_multistyle"
FREEZE_DENOISER = False  # TODO: Add in the other scripts
RESAMPLE_JUMPS = 0

GRID_SIZE = 32
VAL_SPLIT = 0.1
EPOCHS = 5000
SAVE_EVERY = 10
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
EVAL_TIMESTEPS = 1000
TRAIN_TRUNCATION_RATIO = 1.00
INFER_TRUNCATION_RATIO = 0.30
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False


# ICONS 1024 WVS
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_WVS/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_WVS/target"
OUTPUT_DIR = "control_v4/train_outputs_icons50_512_WVS_UNIFIED"
FREEZE_DENOISER = False  # TODO: Add in the other scripts
GRID_SIZE = 32
VAL_SPLIT = 0.1
EPOCHS = 10000
SAVE_EVERY = 10
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
EVAL_TIMESTEPS = 1000
TRAIN_TRUNCATION_RATIO = 1.00
INFER_TRUNCATION_RATIO = 0.30
RESAMPLE_JUMPS = 2
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False


# FACES 1024 GBN
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_celeba_5K_1024/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/data_celeba_5K_1024/target"
# OUTPUT_DIR = "control_v4/train_outputs_data_celeba_5K_1024_no_random"
# GRID_SIZE = 32
# VAL_SPLIT = 0.1
# EPOCHS = 10000
# SAVE_EVERY = 10
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRAIN_TRUNCATION_RATIO = 0.30
# INFER_TRUNCATION_RATIO = 0.30
# RESAMPLE_JUMPS = 2
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# ANIMALS 1024 GBN
# SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AM-2K_1024/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/AM-2K_1024/target"
# OUTPUT_DIR = "control_v4/train_outputs_data_AM-2K_1024_no_random"
# GRID_SIZE = 32
# VAL_SPLIT = 0.1
# EPOCHS = 20000
# SAVE_EVERY = 10
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRAIN_TRUNCATION_RATIO = 0.30
# INFER_TRUNCATION_RATIO = 0.30
# RESAMPLE_JUMPS = 2
# SMART_INIT_FEATURES = False
# SDF_FEATURES = False
# BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# Stress 1 1024
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/target"
# OUTPUT_DIR = "control_v4/train_outputs_data_stress1_no_random"
# GRID_SIZE = 32
# VAL_SPLIT = 0.0
# EPOCHS = 10000
# SAVE_EVERY = 50
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRAIN_TRUNCATION_RATIO = 0.30
# INFER_TRUNCATION_RATIO = 0.30
# RESAMPLE_JUMPS = 2
# SMART_INIT_FEATURES = False    # NOTE
# SDF_FEATURES = False           # NOTE
# BATCH_COORDS_FEATURES = False  # NOTE
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# Stress 2 1024 V2
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2/target"
# OUTPUT_DIR = "control_v4/train_outputs_data_stress2_V2_no_random"
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False
# GRID_SIZE = 32
# VAL_SPLIT = 0.0
# EPOCHS = 10000
# SAVE_EVERY = 50
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRAIN_TRUNCATION_RATIO = 0.30
# INFER_TRUNCATION_RATIO = 0.30
# RESAMPLE_JUMPS = 2
# SMART_INIT_FEATURES = False    # NOTE
# SDF_FEATURES = False           # NOTE
# BATCH_COORDS_FEATURES = False  # NOTE
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# # Stress 2 1024 V1 - IGNORE
# SOURCE_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2/source"
# TARGET_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2/target"
# OUTPUT_DIR = "control_v4/train_outputs_data_stress2_no_random"
# GRID_SIZE = 32
# VAL_SPLIT = 0.0
# EPOCHS = 10000
# SAVE_EVERY = 10
# ENABLE_GECCO = True
# ENABLE_ADAPTIVE_GATE_INJECTION = True
# EVAL_TIMESTEPS = 1000
# TRAIN_TRUNCATION_RATIO = 0.30
# INFER_TRUNCATION_RATIO = 0.30
# RESAMPLE_JUMPS = 2
# SMART_INIT_FEATURES = True    # NOTE
# SDF_FEATURES = True           # NOTE
# BATCH_COORDS_FEATURES = True  # NOTE
# ENABLE_SMART_INIT_JITTER = False
# ENABLE_SMART_INIT_SPLAT_SIGMA = False


# If empty, offsets are auto-exported (if needed) to a default processed_offsets folder.
OFFSETS_DIR = ""
CACHE_DATA_DIR = ""

# Multi-oracle style axis. Each oracle shares the SAME source images (same rho); training uses
# ONE-HOT conditioning (WVS=[1,0,0], GBN=[0,1,0], DITHER=[0,0,1]) and hopes to interpolate convex
# combinations like [0.5,0.5,0]. Per oracle: source/ + target/ stipple + processed_offsets/.
from control_v4_mix.oracles_config import ORACLES_DEFAULT, resolve_oracles
PRELOAD_RAM = False  # Preload all cached data to RAM (eliminates disk I/O per batch)
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Space management: once an epoch is a multiple of KEEP_EVERY, delete the intermediate
# periodic checkpoints (and their train/val panels) whose epoch is NOT a multiple of
# KEEP_EVERY. With SAVE_EVERY=10 and KEEP_EVERY=100, epochs 10..90 are pruned once epoch
# 100 is saved, keeping 100, 200, ...  Set KEEP_EVERY=0 to disable. 'best_*' checkpoints
# are never touched. KEEP_EVERY must be 0 or a multiple of SAVE_EVERY.
KEEP_EVERY = 100

# Model parameters
# FREEZE_DENOISER = True  # TODO: Add in the other scripts
# GRID_SIZE = 32
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_JITTER_PX = 0.5
SMART_INIT_SPLAT_SIGMA_PX = 0.5

# Component 1 - training timestep sampling schedule.
#   "uniform"      -> original behaviour (torch.randint over [0, cutoff)).
#   "logit_normal" -> sample u ~ N(m, s), t = sigmoid(u) * cutoff. Concentrates the
#                     gradient budget on low-t (final-spacing) steps. m<0 pushes lower.
T_SAMPLING = "uniform"
LOGIT_NORMAL_M = 0.0
LOGIT_NORMAL_S = 1.0

# Component 2 - density-match (KDE) loss on the low-t x0 decode.
#   Decodes the one-step x0 estimate to a point set, splats both it and the teacher
#   point set to a low-res Gaussian density map, and penalises the difference. This
#   directly matches the teacher's density field (capacity), which raw eps-MSE only
#   matches indirectly. Weight 0.0 -> OFF (bit-exact original behaviour).
DENSITY_LOSS_WEIGHT = 0.0
DENSITY_KDE_GRID = 16          # KDE map resolution (coarser than GRID_SIZE = low-pass)
DENSITY_KDE_SIGMA_PX = 1.0     # Gaussian sigma, in KDE-grid pixels
DENSITY_LOSS_T_FRAC = 0.4      # only apply where t < frac * eval_timesteps
DENSITY_LOSS_T_SOFT = 0.0      # width of a soft ramp at that cutoff (0 = hard mask)
DENSITY_LOSS_WARMUP_EPOCHS = 0 # linearly ramp the weight in over N epochs (0 = no ramp)
DENSITY_LOSS_GRAD_LOG_EVERY = 0  # log ||grad L_main|| vs ||grad L_density|| every N steps

# Loss component weights
MIN_SNR_GAMMA = 5.0
GEOM_CLUMP_WEIGHT = 1.0
BEST_MAX_CV = 1e9
BEST_MAX_CLUMPED_PCT = 100.0

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


def sample_training_timesteps(batch_size, truncation_cutoff, device,
                              mode="uniform", m=0.0, s=1.0):
    """Draw diffusion timesteps in [0, truncation_cutoff) for a training step.

    Component 1. ``mode="uniform"`` reproduces the original ``torch.randint`` draw
    exactly. ``mode="logit_normal"`` samples ``u ~ N(m, s)``, maps ``t = sigmoid(u)``
    into ``[0, truncation_cutoff)`` and rounds to an integer timestep, concentrating
    the gradient budget on the low-t steps that establish final point spacing. A
    negative ``m`` pushes mass further toward low t.
    """
    if mode == "logit_normal":
        u = torch.randn(batch_size, device=device) * s + m
        t = (torch.sigmoid(u) * truncation_cutoff).long()
        return t.clamp_(0, max(truncation_cutoff - 1, 0))
    return torch.randint(0, truncation_cutoff, (batch_size,), device=device)


def gaussian_kde_map(coords, kde_grid, sigma_px, grid_centers_flat=None):
    """Differentiable Gaussian KDE of (B,N,2) coords onto a (B,K*K) density map.

    Component 2. Unlike ``render_occupancy_grid_gpu`` (hard ``.long()`` binning, no
    gradient) and ``render_smart_init_gpu`` (``amax`` -> occupancy, saturates), this
    *sums* the Gaussian contributions of every point, so the result is a genuine
    density that is differentiable w.r.t. ``coords``. Each map is normalised to
    *mean* 1 (i.e. sum = K*K), which makes the loss invariant to point count, kernel
    scale AND map resolution. Normalising to sum 1 instead would put cell values at
    ~1/K^2 and the squared-error loss at ~1e-7, forcing an ~1e6 density weight.
    """
    bsz = coords.shape[0]
    if grid_centers_flat is None:
        grid_centers_flat = _grid_centers_flat(kde_grid, device=coords.device, dtype=coords.dtype)
    centers = grid_centers_flat.expand(bsz, -1, -1)          # (B, K*K, 2)

    sigma = max(float(sigma_px), 1e-4) / float(kde_grid)
    dist = torch.cdist(centers, coords, p=2)                 # (B, K*K, N)
    gauss = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
    density = gauss.sum(dim=2)                               # (B, K*K)
    n_cells = density.shape[1]
    return density * (n_cells / density.sum(dim=1, keepdim=True).clamp(min=1e-8))


def density_match_loss(x0_pred, x0_true, t, grid_size, kde_grid, sigma_px,
                       t_cutoff, t_soft=0.0, kde_centers=None, point_centers=None):
    """Component 2 loss: || KDE(decode(x0_pred)) - KDE(decode(x0_true)) ||^2.

    The x0 estimate is meaningless at high t, so the per-sample loss is masked to
    ``t < t_cutoff``. ``t_soft > 0`` replaces the hard indicator with a linear ramp
    that decays to 0 over ``t_soft`` steps above the cutoff, avoiding a gradient
    discontinuity at the boundary. Returns ``(loss, active_fraction)``; the loss is
    the mask-weighted mean, so it stays on the same scale as the number of steps
    that actually contribute.
    """
    if t_soft > 0.0:
        w = ((t_cutoff + t_soft - t.float()) / float(t_soft)).clamp(0.0, 1.0)
    else:
        w = (t < t_cutoff).to(x0_pred.dtype)

    active = w.sum()
    if float(active) <= 0.0:
        return x0_pred.sum() * 0.0, 0.0

    coords_pred = offsets_to_coords_gpu(x0_pred, grid_size, point_centers)
    # The teacher side is a constant target - no gradient flows back through it.
    coords_true = offsets_to_coords_gpu(x0_true.detach(), grid_size, point_centers)

    kde_pred = gaussian_kde_map(coords_pred, kde_grid, sigma_px, kde_centers)
    kde_true = gaussian_kde_map(coords_true, kde_grid, sigma_px, kde_centers)

    per_sample = ((kde_pred - kde_true) ** 2).mean(dim=1)     # (B,)
    loss = (per_sample * w).sum() / active.clamp(min=1e-8)
    return loss, float(active) / float(t.shape[0])


def loss_grad_norm(loss, params):
    """L2 norm of d(loss)/d(params), for the Component 2 gradient-balance diagnostic.

    Uses ``retain_graph=True`` so the real backward pass is unaffected. Only call
    this periodically - it costs an extra backward.
    """
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True, create_graph=False,
    )
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum())
    return total ** 0.5


def sample_eval_batch(diffusion, denoiser, control_net, batch, device, n_samples=4, eval_timesteps=1000, resample_jumps=2,
                      show_tqdm=False, tqdm_desc="sampling", truncation_ratio=None):
    """Sample offset grids for intermediate eval with optional resampling.

    When ``truncation_ratio`` and ``smart_init_offsets`` are provided the
    function uses SDEdit-style truncated sampling: it adds noise at
    ``t_start = int(eval_timesteps * truncation_ratio)`` to ``smart_init_offsets``
    and denoises only from there, matching the truncated training regime.
    """
    controlled = DynamicControlledMultiStyleDenoiser(denoiser, control_net)
    high_res_img = batch["high_res"]
    target_density = batch["target_density"]
    high_res_sdf = batch.get("high_res_sdf")
    target_sdf = batch.get("target_sdf")
    smart_init_grid = batch.get("smart_init_grid")
    smart_init_offsets = batch.get("smart_init_offsets")
    if smart_init_offsets is None:
        smart_init_offsets_list = []
        for image_01 in high_res_img[:, 0].detach().cpu().numpy():
            _, smart_offsets_np, _ = build_smart_init_from_image(
                image_01,
                grid_size=target_density.shape[-1],
                n_points=target_density.shape[-1] * target_density.shape[-1],
                seed=42,
            )
            smart_init_offsets_list.append(torch.from_numpy(smart_offsets_np).unsqueeze(0))
        smart_init_offsets = torch.cat(smart_init_offsets_list, dim=0)

    style_vec_eval = batch.get("style_vec")
    controlled.set_condition(high_res_img, high_res_sdf, target_density, target_sdf, smart_init_grid, style_vec=style_vec_eval)

    original_model = diffusion.model
    diffusion.model = controlled
    diffusion.set_num_timesteps(eval_timesteps)
    diffusion.eval()

    h, w = target_density.shape[-2], target_density.shape[-1]
    shape = [n_samples, 2, h, w]
    use_sdedit = (truncation_ratio is not None) and (truncation_ratio < 1.0) and (smart_init_offsets is not None)
    apply_manual_loop = resample_jumps > 0 or use_sdedit
    with torch.no_grad() if not apply_manual_loop else torch.enable_grad():
        if not apply_manual_loop:
            raw = diffusion.p_sample_loop(shape, img=None, cond=None,
                                          with_tqdm=show_tqdm, with_sampling=True)
        else:
            if use_sdedit:
                t_start = int(np.clip(
                    int(diffusion.num_timesteps * truncation_ratio),
                    1, diffusion.num_timesteps - 1,
                ))
                x_init = smart_init_offsets.to(device)
                if x_init.shape[0] != n_samples:
                    x_init = x_init.expand(n_samples, -1, -1, -1).contiguous()
                alpha_t = diffusion.alphas_cumprod[t_start]
                img = add_noise_at_t(x_init, alpha_t)
                n_steps = t_start
            else:
                img = diffusion.noise_fn(shape).to(device)
                n_steps = diffusion.num_timesteps - 1
            iter_steps = reversed(range(n_steps))
            if show_tqdm:
                iter_steps = tqdm(
                    iter_steps,
                    total=n_steps,
                    desc=tqdm_desc,
                    leave=False,
                )

            for i in iter_steps:
                t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
                for u in range(resample_jumps + 1):
                    with torch.no_grad():
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

    if "style_vec" in sample0:
        collated["style_vec"] = torch.stack([sample["style_vec"] for sample in batch], dim=0)

    return collated


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
    parser.add_argument("--oracles", default=ORACLES_DEFAULT,
                        help="';'-separated NAME:root list; each root has source/,target/,processed_offsets/")
    parser.add_argument("--val-manifest", default="control_v4_mix/validation_manifest.json",
                        help="JSON list of val image basenames; if present, defines the EXACT val set")
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
    parser.add_argument("--train-truncation-ratio", type=float, default=TRAIN_TRUNCATION_RATIO,
                        help="TRAINING only: draw diffusion timesteps from [0, ratio * total_timesteps). "
                             "1.0 = full-range training.")
    parser.add_argument("--infer-truncation-ratio", type=float, default=INFER_TRUNCATION_RATIO,
                        help="INFERENCE only: SDEdit start level for eval sampling. <1.0 starts from a "
                             "noised smart-init at t = ratio * T; 1.0 = full denoise from pure noise.")
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

    parser.add_argument(
        "--t-sampling",
        choices=["uniform", "logit_normal"],
        default=T_SAMPLING,
        help="Component 1: training timestep sampling. 'logit_normal' concentrates the "
             "gradient budget on low-t (final-spacing) steps.",
    )
    parser.add_argument(
        "--logit-normal-m",
        type=float,
        default=LOGIT_NORMAL_M,
        help="Mean m of logit-normal t sampling; negative pushes mass toward low t.",
    )
    parser.add_argument(
        "--logit-normal-s",
        type=float,
        default=LOGIT_NORMAL_S,
        help="Std s of logit-normal t sampling.",
    )

    parser.add_argument(
        "--density-loss-weight",
        type=float,
        default=DENSITY_LOSS_WEIGHT,
        help="Component 2: weight of the density-match (KDE) loss on the low-t x0 "
             "decode. 0 disables it entirely (default).",
    )
    parser.add_argument(
        "--density-kde-grid",
        type=int,
        default=DENSITY_KDE_GRID,
        help="Resolution of the KDE density map for the Component 2 loss.",
    )
    parser.add_argument(
        "--density-kde-sigma-px",
        type=float,
        default=DENSITY_KDE_SIGMA_PX,
        help="Gaussian sigma (in KDE-grid pixels) used to splat points for Component 2.",
    )
    parser.add_argument(
        "--density-loss-t-frac",
        type=float,
        default=DENSITY_LOSS_T_FRAC,
        help="Component 2 is applied only where t < frac * eval_timesteps.",
    )
    parser.add_argument(
        "--density-loss-t-soft",
        type=float,
        default=DENSITY_LOSS_T_SOFT,
        help="Width (in timesteps) of a soft ramp at the Component 2 t cutoff; 0 = hard mask.",
    )
    parser.add_argument(
        "--density-loss-warmup-epochs",
        type=int,
        default=DENSITY_LOSS_WARMUP_EPOCHS,
        help="Linearly ramp the Component 2 weight in over this many epochs (0 = no ramp).",
    )
    parser.add_argument(
        "--density-loss-grad-log-every",
        type=int,
        default=DENSITY_LOSS_GRAD_LOG_EVERY,
        help="Every N steps, log ||grad L_main|| vs ||grad L_density|| to pick the "
             "Component 2 weight by gradient balance. 0 = off (costs an extra backward).",
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
    parser.add_argument(
        "--keep-every",
        type=int,
        default=KEEP_EVERY,
        help="Space saver: once an epoch is a multiple of this, delete earlier periodic "
             "checkpoints + panels that are NOT multiples of it (0 disables; must be 0 or a "
             "multiple of --save_every). 'best_*' checkpoints are kept.",
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
    args = parser.parse_args()
    run(args=args)


def prune_intermediate_saves(epoch_label, keep_every, checkpoints_dir, out_dir, ckpt_prefix):
    """Free disk once ``epoch_label`` is a multiple of ``keep_every``.

    Deletes every periodic checkpoint ``{ckpt_prefix}{N}.pt`` (and its train/val panels
    ``{train,val}_panel_ep{N}.png`` in ``out_dir``) for which N < epoch_label and
    N % keep_every != 0. Multiples of keep_every and the current/future epochs are kept, and
    the pattern deliberately does NOT match ``best_*`` checkpoints. No-op when keep_every <= 0
    or epoch_label is not a keep multiple.
    """
    if keep_every <= 0 or epoch_label % keep_every != 0:
        return
    removed = 0
    ckpt_re = re.compile(r"^" + re.escape(ckpt_prefix) + r"(\d+)\.pt$")
    if os.path.isdir(checkpoints_dir):
        for fname in os.listdir(checkpoints_dir):
            m = ckpt_re.match(fname)
            if not m:
                continue
            ep = int(m.group(1))
            if ep >= epoch_label or ep % keep_every == 0:
                continue
            try:
                os.remove(os.path.join(checkpoints_dir, fname))
                removed += 1
            except OSError as exc:
                print(f"  -> KEEP_EVERY: could not remove {fname}: {exc}")
    panel_re = re.compile(r"^(?:train|val)_panel_ep(\d+)\.png$")
    if os.path.isdir(out_dir):
        for fname in os.listdir(out_dir):
            m = panel_re.match(fname)
            if not m:
                continue
            ep = int(m.group(1))
            if ep >= epoch_label or ep % keep_every == 0:
                continue
            try:
                os.remove(os.path.join(out_dir, fname))
            except OSError as exc:
                print(f"  -> KEEP_EVERY: could not remove {fname}: {exc}")
    if removed:
        print(f"  -> KEEP_EVERY={keep_every}: pruned {removed} intermediate checkpoint(s) "
              f"below epoch {epoch_label} (kept multiples of {keep_every}; best_* untouched)")


def run(args):
    if args.wandb_valid_images < 0:
        raise ValueError("--wandb-valid-images must be >= 0")
    if args.wandb_train_images < 0:
        raise ValueError("--wandb-train-images must be >= 0")
    if args.save_every <= 0:
        raise ValueError("--save_every must be >= 1")
    if args.keep_every < 0:
        raise ValueError("--keep-every must be >= 0 (0 disables pruning)")
    if args.keep_every > 0 and args.keep_every % args.save_every != 0:
        raise ValueError(
            f"--keep-every ({args.keep_every}) must be a multiple of --save_every "
            f"({args.save_every}); otherwise the kept epochs never coincide with saved ones."
        )
    if not (0.0 < args.train_truncation_ratio <= 1.0):
        raise ValueError("--train-truncation-ratio must be in (0, 1]")
    if not (0.0 < args.infer_truncation_ratio <= 1.0):
        raise ValueError("--infer-truncation-ratio must be in (0, 1]")
    if not (args.show_selected_inputs or args.show_selected_gt or args.show_selected_predict or args.show_selected_gt_offsets):
        raise ValueError("At least one of --show-selected-inputs, --show-selected-gt, --show-selected-predict, or --show-selected-gt-offsets must be enabled")
    # Multi-oracle: parse --oracles, ensure each oracle offsets exist; first oracle source is the
    # shared rho input (all oracles share source content).
    oracle_names, oracle_offsets, first_root = [], {}, None
    for name, root in resolve_oracles(args.oracles):
        odir = ensure_offsets_dir(os.path.join(root, "source"), os.path.join(root, "target"),
                                  os.path.join(root, "processed_offsets"), args.grid_size)
        oracle_names.append(name)
        oracle_offsets[name] = odir
        if first_root is None:
            first_root = root
    args.source = os.path.join(first_root, "source")
    args.offsets = oracle_offsets[oracle_names[0]]

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # ?????? wandb ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    use_wandb = WANDB_ACTIVE
    if use_wandb:
        try:
            import wandb
            load_wandb_key()
            run_name = datetime.now().strftime("v4-train-%Y%m%d-%H%M%S")
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

    # ?????? load pretrained diffusion + build Dynamic ControlNet V4 ??????????????????
    diffusion = ParseSampleConfig(args.base_config_path, device=device)
    diffusion.load_state_dict(torch.load(args.base_ckpt_path, map_location="cpu")["diffu"], strict=False)
    diffusion.to(device)
    diffusion.eval()

    denoiser = diffusion.model
    # TRAINING truncation: restricts the diffusion timesteps sampled during training.
    truncation_cutoff = max(1, int(args.eval_timesteps * args.train_truncation_ratio))

    # NOTE: Create control_net BEFORE freezing denoiser so deep copies have requires_grad=True
    control_net = DynamicControlNetMultiStyle(
        denoiser,
        style_dim=len(oracle_names),
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
        smart_init_features=args.smart_init_features,
        sdf_features=args.sdf_features,
        batch_coords_features=args.batch_coords_features,
        enable_adaptive_gate_injection=args.enable_adaptive_gate_injection,
    ).to(device)
    control_net.train()

    # Freeze/unfreeze denoiser based on flag (AFTER control_net creation)
    if args.freeze_denoiser:
        for p in denoiser.parameters():
            p.requires_grad = False
    else:
        denoiser.train()

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
    print(f"Train truncation ratio                : {args.train_truncation_ratio:.3f}")
    print(f"Train truncation cutoff timesteps     : {truncation_cutoff}/{args.eval_timesteps}")
    print(f"Infer truncation ratio (SDEdit)       : {args.infer_truncation_ratio:.3f}")
    print(f"Eval resample-jumps                   : {args.resample_jumps}")
    print(f"Smart Init features enabled           : {args.smart_init_features}")
    print(f"SDF features enabled                  : {args.sdf_features}")
    print(f"Batch coords features enabled         : {args.batch_coords_features}")
    print(f"Min-SNR gamma                         : {args.min_snr_gamma}")
    density_loss_enabled = args.density_loss_weight > 0.0
    density_t_cutoff = args.density_loss_t_frac * float(args.eval_timesteps)
    _tsamp = args.t_sampling + (f" (m={args.logit_normal_m}, s={args.logit_normal_s})" if args.t_sampling == "logit_normal" else "")
    _dens = (
        f"on (w={args.density_loss_weight}, kde={args.density_kde_grid}, "
        f"sigma={args.density_kde_sigma_px}px, t<{density_t_cutoff:.0f})"
        if density_loss_enabled else "off"
    )
    print(f"Density-match (KDE) loss              : {_dens}")
    print(f"Timestep sampling                     : {_tsamp}")
    print(f"SDF truncation (px)                   : {args.sdf_truncate_px}")
    print(f"Smart Init micro-jitter (train, px)  : {args.smart_init_jitter_px}")
    print(f"Smart Init soft-splat sigma (px)     : {args.smart_init_splat_sigma_px}")
    print(f"Smart Init jitter enabled            : {args.enable_smart_init_jitter}")
    print(f"Smart Init splat-sigma enabled       : {args.enable_smart_init_splat_sigma}")

    # ?????? dataset ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    cache_data_dir = args.cache_data_dir
    if not (args.smart_init_features or args.sdf_features):
        cache_data_dir = None
    elif not cache_data_dir:
        cache_data_dir = os.path.join(args.out, "cache_data")
    dataset = MultiStyleStippleDataset(
        args.source, oracle_names, oracle_offsets, grid_size=args.grid_size,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "MultiStyleStippleDataset has 0 samples. Check the shared source dir and that each "
            "oracle processed_offsets share matching stems."
        )
    print(f"Multi-style dataset: {len(dataset)} samples over {len(dataset.filenames)} icons; "
          f"K={dataset.K} oracles={oracle_names}; per-oracle counts = {dataset.oracle_counts()}")
    # Held-out val set. Prefer an explicit manifest of val basenames (EXACTLY reproduces the
    # reference control_v4 split regardless of source-folder structure); else fall back to the
    # source-folder split.
    if args.val_manifest and os.path.isfile(args.val_manifest):
        train_filenames, val_filenames = split_from_manifest(args.source, args.val_manifest)
        print(f"Val set from manifest {args.val_manifest}: {len(val_filenames)} val icons "
              f"(train {len(train_filenames)})")
    else:
        train_filenames, val_filenames = source_train_val_split(args.source, args.val_split, seed=42)
        print(f"Val set from source-folder split: {len(val_filenames)} val icons")
    train_len, val_len = len(train_filenames), len(val_filenames)
    train_dataset = MultiStyleStippleDataset(
        args.source, oracle_names, oracle_offsets, grid_size=args.grid_size,
        filenames=train_filenames,
    )
    val_dataset = None
    if val_len > 0:
        val_dataset = MultiStyleStippleDataset(
            args.source, oracle_names, oracle_offsets, grid_size=args.grid_size,
            filenames=val_filenames,
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
    # Component 2 decodes offsets -> points, so it needs the point-grid centres too.
    # (density_loss_enabled / density_t_cutoff are defined earlier, with the banner.)
    _need_point_centers = args.smart_init_features or density_loss_enabled
    grid_centers_flat = _grid_centers_flat(args.grid_size, device=device, dtype=torch.float32) if _need_point_centers else None
    density_kde_centers = (
        _grid_centers_flat(args.density_kde_grid, device=device, dtype=torch.float32)
        if density_loss_enabled else None
    )

    # ?????? optimizer ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    if args.freeze_denoiser:
        optimizer = torch.optim.AdamW(control_net.parameters(), lr=args.lr)
    else:
        all_params = list(control_net.parameters()) + list(denoiser.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=args.lr)

    # ?????? optional resume from latest checkpoint ?????????????????????????????????????????????????????????????????????
    start_epoch = 0
    global_step = 0
    best_geom_score = float("inf")
    last_geom = {"cv": float("nan"), "clumped_pct": float("nan"), "score": float("nan")}
    checkpoints_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_geom_ckpt_path = None
    epoch_history = []
    train_epoch_history = []
    val_epoch_history = []
    if args.resume_latest:
        ckpt_re = re.compile(r"^dynamic_controlnet_v4_ep(\d+)\.pt$")
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
            control_net.safe_load_state_dict(state, strict=False)
            if state.get("denoiser") is not None:
                denoiser.load_state_dict(state["denoiser"], strict=False)
                print("  -> restored trained (unfrozen) base denoiser from checkpoint")
            if "optimizer" in state and state["optimizer"] is not None:
                optimizer.load_state_dict(state["optimizer"])
            global_step = int(state.get("global_step", 0))
            best_geom_score = float(state.get("best_geom_score", best_geom_score))
            # epoch in checkpoint is 0-based. Continue from next epoch.
            start_epoch = int(state.get("epoch", latest_epoch_num - 1)) + 1
            print(
                f"Resumed from checkpoint: {latest_path} | "
                f"start_epoch={start_epoch} | global_step={global_step}"
            )

    # ?????? training loop ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    # On (re)start: if a previous prune was interrupted mid-deletion (e.g. the process was
    # killed while deleting epochs 10..90 after saving epoch 100), finish it now. Re-run the
    # pruner at the highest keep-boundary already on disk so leftover intermediates are removed
    # rather than lingering until the next boundary. Idempotent -- a clean run is a no-op.
    if args.keep_every > 0 and os.path.isdir(checkpoints_dir):
        _keep_prefix = "dynamic_controlnet_v4_ep"
        _keep_re = re.compile(r"^" + re.escape(_keep_prefix) + r"(\d+)\.pt$")
        _keep_eps = []
        for _keep_fname in os.listdir(checkpoints_dir):
            _keep_m = _keep_re.match(_keep_fname)
            if _keep_m:
                _keep_eps.append(int(_keep_m.group(1)))
        if _keep_eps:
            _keep_boundary = (max(_keep_eps) // args.keep_every) * args.keep_every
            if _keep_boundary >= args.keep_every:
                print(f"  -> KEEP_EVERY: startup catch-up prune at boundary {_keep_boundary}")
                prune_intermediate_saves(
                    _keep_boundary, args.keep_every, checkpoints_dir, args.out, _keep_prefix,
                )

    for epoch in range(start_epoch, args.epochs):
        should_save_epoch = ((epoch + 1) % args.save_every == 0) or ((epoch + 1) == args.epochs)
        epoch_loss = 0.0
        epoch_density_loss = 0.0
        preview_batch = None

        control_net.train()
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

            t = sample_training_timesteps(
                x_0.shape[0], truncation_cutoff, device,
                mode=args.t_sampling, m=args.logit_normal_m, s=args.logit_normal_s,
            )
            noise = torch.randn_like(x_0)
            offsets_t = diffusion.q_sample(x_0, t, noise)

            controls = control_net(
                offsets_t,
                t,
                high_res_img,
                target_density,
                high_res_sdf=high_res_sdf,
                target_sdf_map=target_sdf,
                target_smart_init_map=smart_init_grid,
                style_vec=batch["style_vec"],
            )
            noise_pred = denoiser(offsets_t, t, controls=controls)

            per_sample_mse = F.mse_loss(noise_pred, noise, reduction="none")
            per_sample_mse = per_sample_mse.mean(dim=(1, 2, 3))

            if args.min_snr_gamma > 0:
                alphas_cumprod_t = diffusion.alphas_cumprod.gather(0, t)
                snr = alphas_cumprod_t / torch.clamp(1.0 - alphas_cumprod_t, min=1e-8)
                min_snr_weight = torch.clamp(snr, max=args.min_snr_gamma) / torch.clamp(snr, min=1e-8)
                denoise_loss = (per_sample_mse * min_snr_weight).mean()
            else:
                denoise_loss = per_sample_mse.mean()

            loss = denoise_loss

            density_loss_item = 0.0
            if density_loss_enabled:
                density_w = args.density_loss_weight
                if args.density_loss_warmup_epochs > 0:
                    density_w *= min(1.0, float(epoch) / float(args.density_loss_warmup_epochs))
                if density_w > 0.0:
                    # Closed-form one-step x0 estimate from (x_t, eps_pred).
                    x0_pred = diffusion.predict_xstart_from_noise(offsets_t, t, noise_pred)
                    d_loss, d_active = density_match_loss(
                        x0_pred, x_0, t,
                        grid_size=args.grid_size,
                        kde_grid=args.density_kde_grid,
                        sigma_px=args.density_kde_sigma_px,
                        t_cutoff=density_t_cutoff,
                        t_soft=args.density_loss_t_soft,
                        kde_centers=density_kde_centers,
                        point_centers=grid_centers_flat,
                    )
                    if (args.density_loss_grad_log_every > 0
                            and global_step % args.density_loss_grad_log_every == 0
                            and d_active > 0.0):
                        # Mirror the clip_grad_norm_ scope below: when the base is
                        # unfrozen the density gradient flows through the denoiser too,
                        # so measuring the control branch alone understates the ratio.
                        _gp = [p for p in control_net.parameters() if p.requires_grad]
                        if not args.freeze_denoiser:
                            _gp += [p for p in denoiser.parameters() if p.requires_grad]
                        gn_main = loss_grad_norm(denoise_loss, _gp)
                        gn_dens = loss_grad_norm(density_w * d_loss, _gp)
                        _ratio = gn_dens / max(gn_main, 1e-12)
                        print(f"  [C2 grad] step {global_step}: |g_main|={gn_main:.4e} "
                              f"|g_density|={gn_dens:.4e} ratio={_ratio:.3f} "
                              f"(target 0.05-0.20) active={d_active:.2f}")
                        if use_wandb:
                            wandb.log({"c2/grad_ratio": _ratio,
                                       "c2/grad_main": gn_main,
                                       "c2/grad_density": gn_dens}, step=epoch + 1)
                    loss = denoise_loss + density_w * d_loss
                    density_loss_item = float(d_loss.detach())

            optimizer.zero_grad()
            loss.backward()
            if args.freeze_denoiser:
                torch.nn.utils.clip_grad_norm_(control_net.parameters(), 1.0)
            else:
                all_params = list(control_net.parameters()) + list(denoiser.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_density_loss += density_loss_item
            global_step += 1
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        avg_density_loss = epoch_density_loss / max(len(train_loader), 1)

        if should_save_epoch and args.wandb_train_images > 0 and preview_batch is not None and preview_batch["high_res"].shape[0] > 0:
            control_net.eval()
            train_pred_raw = sample_eval_batch(
                diffusion,
                denoiser,
                control_net,
                preview_batch,
                device,
                n_samples=preview_batch["high_res"].shape[0],
                eval_timesteps=args.eval_timesteps,
                resample_jumps=args.resample_jumps,
                show_tqdm=True,
                tqdm_desc=f"Epoch {epoch+1}/{args.epochs} [train-predict]",
                truncation_ratio=args.infer_truncation_ratio,
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
            control_net.train()

        # Validation loop with tqdm.
        val_avg_loss = None
        val_preview_batch = None
        pred_raw_for_geom = None
        if val_loader is not None:
            control_net.eval()
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

                    t = sample_training_timesteps(
                        x_0.shape[0], truncation_cutoff, device,
                        mode=args.t_sampling, m=args.logit_normal_m, s=args.logit_normal_s,
                    )
                    noise = torch.randn_like(x_0)
                    offsets_t = diffusion.q_sample(x_0, t, noise)

                    controls = control_net(
                        offsets_t,
                        t,
                        high_res_img,
                        target_density,
                        high_res_sdf=high_res_sdf,
                        target_sdf_map=target_sdf,
                        target_smart_init_map=smart_init_grid,
                        style_vec=batch["style_vec"],
                    )
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

            if should_save_epoch and args.wandb_valid_images > 0:
                # Export per-epoch qualitative val panel (N samples, 4 columns).
                control_net.eval()
                pred_raw = sample_eval_batch(
                    diffusion,
                    denoiser,
                    control_net,
                    val_preview_batch,
                    device,
                    n_samples=val_preview_batch["high_res"].shape[0],
                    eval_timesteps=args.eval_timesteps,
                    resample_jumps=args.resample_jumps,
                    show_tqdm=True,
                    tqdm_desc=f"Epoch {epoch+1}/{args.epochs} [predict]",
                    truncation_ratio=args.infer_truncation_ratio,
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
                control_net.train()

            # Geometry-gated best checkpoint based on CV + clumped% score.
            # Only compute geometry on epochs where we save checkpoints.
            if should_save_epoch and val_preview_batch is not None and val_preview_batch["high_res"].shape[0] > 0:
                control_net.eval()
                if pred_raw_for_geom is None:
                    pred_raw_for_geom = sample_eval_batch(
                        diffusion,
                        denoiser,
                        control_net,
                        {key: value[:1] for key, value in val_preview_batch.items()},
                        device,
                        n_samples=1,
                        eval_timesteps=args.eval_timesteps,
                        resample_jumps=args.resample_jumps,
                        show_tqdm=True,
                        tqdm_desc=f"Epoch {epoch+1}/{args.epochs} [geom]",
                        truncation_ratio=args.infer_truncation_ratio,
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
                        "control_net": control_net.state_dict(),
                        "denoiser": (None if args.freeze_denoiser else denoiser.state_dict()),
                        "freeze_denoiser": bool(args.freeze_denoiser),
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
                control_net.train()

        if use_wandb:
            epoch_history.append(epoch + 1)
            train_epoch_history.append(float(avg_loss))
            val_epoch_history.append(float(val_avg_loss) if val_avg_loss is not None else float("nan"))

            # Log train loss separately
            wandb.log({"epoch": epoch + 1, "metrics/train_loss": avg_loss}, step=epoch + 1)
            if density_loss_enabled:
                wandb.log({"epoch": epoch + 1, "metrics/train_density_loss": avg_density_loss},
                          step=epoch + 1)

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
            save_path = os.path.join(checkpoints_dir, f"dynamic_controlnet_v4_ep{epoch+1}.pt")
            torch.save({
                "control_net": control_net.state_dict(),
                # When the base is unfrozen its trained weights live only in `denoiser`
                # (DynamicControlNet does NOT hold the base), so they must be saved
                # explicitly or they are lost on reload. Frozen -> None (base == GBN base).
                "denoiser": (None if args.freeze_denoiser else denoiser.state_dict()),
                "freeze_denoiser": bool(args.freeze_denoiser),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_geom_score": best_geom_score,
                "cv_score": float(last_geom["cv"]),
                "clumped_score": float(last_geom["clumped_pct"]),
                "current_geom_score": float(last_geom["score"]),
            }, save_path)
            print(f"  -> saved {save_path}")
            prune_intermediate_saves(
                epoch + 1, args.keep_every, checkpoints_dir, args.out, "dynamic_controlnet_v4_ep",
            )

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
