"""Overfit Dynamic ControlNet FM on a single (source, target) pair.

Usage (from project root):
    python control_fm/test_overfit.py --steps 5000
    python control_fm/test_overfit.py --steps 5000 --sample-index 42 --vis-every 200
"""

import argparse
import json
import math
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
from control_fm.conditioning import (
    build_condition_tensors_from_cached_sdf,
    build_condition_tensors_from_image,
    compute_raw_sdf,
)
from control_fm.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_fm.smart_init import (
    generate_smart_init_points_from_density,
    render_smart_init_grid,
    smart_init_points_to_offsets,
)
from control_fm.flow_matching import FlowMatching, build_velocity_network
from control_gt_free.losses.composite import CompositeObjective
from control_gt_free.losses.loss_pcf import default_edges
from control_gt_free.train.tau_schedule import TauSchedule
from control_gt_free.train.unroll import unroll_ode
from utils.stippling_metrics import compute_spacing_quality, visualize_overfit_metrics

# ── default globals (edit here for quick experiments) ───────────────

# Paths and I/O
# DATA_ROOT = r"C:\Users\User\PycharmProjects\ExampleBasedSamplingWithDiffusion\training\monkey"
# DATA_ROOT = r"/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024"
# DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_1024"
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_taksim"
SOURCE_DIR = os.path.join(DATA_ROOT, "source")
TARGET_DIR = os.path.join(DATA_ROOT, "target")
BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = "config/GBN/model.ckpt"
OUTPUT_DIR = "control_gt_free/overfit_outputs"
WANDB_ENV = ".env"
EXPORT_GT_OFFSET = True

ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
EVAL_TIMESTEPS = 1000
TRUNCATION_RATIO = 1.00
RESAMPLE_JUMPS = 0
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Model parameters
FREEZE_DENOISER = False
GRID_SIZE = 32
N_POINTS = GRID_SIZE ** 2

SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_JITTER_PX = 0.5
SMART_INIT_SPLAT_SIGMA_PX = 0.5

# Loss parameters
MIN_SNR_GAMMA = 5.0
GEOM_CLUMP_WEIGHT = 5.0
BEST_MAX_CV = 1e9
BEST_MAX_CLUMPED_PCT = 100.0

# Training configuration
WANDB_ACTIVE = False

STEPS = 10000
SAMPLE_INDEX = 0
LR = 5e-4
VIS_EVERY = 500
RESUME_LATEST = False
N_SAMPLES = 2
SEED = 42
DEVICE = "cuda"
CAPACITY_GRID_SIZE = 16

# Flow-Matching sampling
ODE_STEPS = 50
ODE_METHOD = "euler"
T_SCALE = 1000.0
# CAPACITY_GRID_SIZE = -1  # -1 for full input resolution

# FM_COUPLING = "smartinit"
FM_COUPLING = "smartinit"  # overfit default: START THE UNROLL FROM THE SMART-INIT
                           # point cloud (not Gaussian noise) so the objective is
                           # far easier to descend. Use --fm-coupling gaussian to
                           # go back to noise->data. (train_control.py keeps gaussian.)
# CONDITIONING = "spade"
# CONDITIONING = "controlnet"  # "controlnet", "concat", or "spade"
CONDITIONING = "concat"  # "controlnet", "concat", or "spade"

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


def load_condition(img_path, grid_size, device, sdf_features=True, sdf_truncate_px=0.0, cache_dir=None):
    """Load source image and return image, density, and SDF condition tensors.
    
    If cache_dir is provided, caches the raw SDF to avoid scipy calls on repeated runs.
    """
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0

    high_res = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    target_density = F.interpolate(high_res, size=(grid_size, grid_size), mode="area")

    if not sdf_features:
        if device is not None:
            high_res = high_res.to(device)
            target_density = target_density.to(device)
        return high_res, target_density, None, None
    
    # Check for cached raw SDF
    if cache_dir:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        sdf_cache_path = os.path.join(cache_dir, stem + "_sdf_raw.npy")
        if os.path.exists(sdf_cache_path):
            raw_sdf = np.load(sdf_cache_path)
            high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_cached_sdf(
                img_np, raw_sdf, grid_size, sdf_truncate_px=sdf_truncate_px
            )
        else:
            raw_sdf = compute_raw_sdf(img_np)
            os.makedirs(cache_dir, exist_ok=True)
            np.save(sdf_cache_path, raw_sdf)
            high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_cached_sdf(
                img_np, raw_sdf, grid_size, sdf_truncate_px=sdf_truncate_px
            )
    else:
        high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_image(
            img_np,
            grid_size,
            sdf_truncate_px=sdf_truncate_px,
        )
    
    if device is not None:
        high_res = high_res.to(device)
        target_density = target_density.to(device)
        if high_res_sdf is not None:
            high_res_sdf = high_res_sdf.to(device)
        if target_sdf is not None:
            target_sdf = target_sdf.to(device)
    
    return high_res, target_density, high_res_sdf, target_sdf


def _grid_centers_flat(grid_size, device, dtype):
    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    centers = torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2)
    return centers


def offsets_to_coords_gpu(offsets, grid_size, grid_centers_flat):
    """Convert OT offsets (B,2,G,G) to normalized coords (B,N,2) on GPU."""
    bsz = offsets.shape[0]
    offs = offsets.permute(0, 2, 3, 1).reshape(bsz, grid_size * grid_size, 2)
    coords = grid_centers_flat.expand(bsz, -1, -1) + offs / float(grid_size)
    return coords.clamp(0.0, 1.0)


def coords_to_offsets_gpu(coords, grid_size, grid_centers_flat):
    """Convert normalized coords (B,N,2) back to OT offsets (B,2,G,G) on GPU."""
    delta = (coords - grid_centers_flat.expand(coords.shape[0], -1, -1)) * float(grid_size)
    return delta.reshape(coords.shape[0], grid_size, grid_size, 2).permute(0, 3, 1, 2).contiguous()


def apply_gpu_jitter(coords, jitter_strength_px=0.5, grid_size=32):
    """Apply Gaussian coordinate jitter in pixel units to (B,N,2) coords on GPU."""
    if jitter_strength_px <= 0.0:
        return coords
    sigma = float(jitter_strength_px) / float(grid_size)
    noise = torch.randn_like(coords) * sigma
    return (coords + noise).clamp(0.0, 1.0)


def render_smart_init_gpu(coords, grid_size=32, sigma_px=0.5):
    """Render (B,N,2) normalized coords to (B,1,G,G) with Gaussian soft splatting on GPU."""
    bsz = coords.shape[0]
    dtype = coords.dtype
    device = coords.device

    lin = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / float(grid_size)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    pixel_centers = torch.stack([gx, gy], dim=-1).reshape(1, grid_size * grid_size, 2).expand(bsz, -1, -1)

    sigma = max(float(sigma_px), 1e-4) / float(grid_size)
    diff = pixel_centers.unsqueeze(2) - coords.unsqueeze(1)  # (B, P, N, 2)
    dist2 = (diff * diff).sum(dim=-1)
    gauss = torch.exp(-dist2 / (2.0 * sigma * sigma))

    # Max aggregation keeps values naturally bounded and preserves local peaks.
    grid = gauss.max(dim=2).values.reshape(bsz, 1, grid_size, grid_size)
    return grid.clamp(0.0, 1.0)


def render_occupancy_grid_gpu(coords, grid_size=32):
    """Render (B,N,2) coords to (B,1,G,G) normalized occupancy map on GPU."""
    bsz, n, _ = coords.shape
    device = coords.device
    dtype = coords.dtype
    px = (coords[:, :, 0] * grid_size).clamp(0, grid_size - 1).long()
    py = (coords[:, :, 1] * grid_size).clamp(0, grid_size - 1).long()
    flat_idx = py * grid_size + px
    grid_flat = torch.zeros(bsz, grid_size * grid_size, device=device, dtype=dtype)
    ones = torch.ones(bsz, n, device=device, dtype=dtype)
    grid_flat.scatter_add_(1, flat_idx, ones)
    mx = grid_flat.amax(dim=1, keepdim=True).clamp(min=1.0)
    return (grid_flat / mx).reshape(bsz, 1, grid_size, grid_size)


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
    Image.fromarray(mag_u8).save(os.path.join(gt_dir, f"gt_offsets_magnitude_{GRID_SIZE}x{GRID_SIZE}.png"))

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
        os.path.join(gt_dir, f"gt_points_binary_{GRID_SIZE}x{GRID_SIZE}.png")
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


def sample_from_model(fm, control_net, denoiser, high_res, high_res_sdf,
                      target_density, target_sdf, smart_init_grid, x_init_offsets,
                      device, n_samples=2, ode_steps=50, ode_method="euler", t_start=1.0,
                      conditioner=None):
    """Integrate the Flow-Matching ODE from noise (t=1) to data (t=0)."""
    shape = [n_samples, 2, GRID_SIZE, GRID_SIZE]

    if conditioner is not None:
        # single-stage concat conditioning (no ControlNet)
        conditioner.set_condition(target_density, smart_init_grid if conditioner.use_smart_init_grid else None)
        conditioner.eval()

        def velocity_fn(x, t_net):
            return conditioner(x, t_net)
    else:
        controlled = DynamicControlledDenoiser(denoiser, control_net)
        controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
        controlled.eval()

        def velocity_fn(x, t_net):
            return controlled(x, t_net)

    x_start = None
    if fm.coupling == "smartinit":
        x_start = fm.start_state(shape, device=device, smart_init=x_init_offsets)
    with torch.no_grad():
        raw = fm.ode_sample(
            velocity_fn, shape, device=device,
            n_steps=ode_steps, method=ode_method, t_start=t_start,
            x_start=x_start,
            show_tqdm=True,
        )
    raw_np = raw.detach().cpu().numpy()

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

    # Paths and I/O
    parser.add_argument("--base_config_path", default=BASE_CONFIG_PATH)
    parser.add_argument("--base_ckpt_path", default=BASE_CKPT_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--export-gt-offset",
        type=bool,
        default=EXPORT_GT_OFFSET,
        help="Export GT offset diagnostics into out_dir/gt_offset",
    )

    # Model parameters
    parser.add_argument(
        "--grid-size", 
        type=int, 
        default=GRID_SIZE,
        help="Grid resolution for offset export and dataset loading",
    )
    parser.add_argument(
        "--enable-gecco",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_GECCO,
        help="Enable GECCO dynamic feature sampling in the control hint path",
    )
    parser.add_argument(
        "--enable-adaptive-gate-injection",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_ADAPTIVE_GATE_INJECTION,
        help="Use sigmoid-gated adaptive injection; disable for zero-conv style injections",
    )
    parser.add_argument(
        "--resample-jumps",
        type=int,
        default=RESAMPLE_JUMPS,
        help="RePaint-style micro-loops per timestep during sampling (0=disabled)",
    )
    parser.add_argument(
        "--eval-timesteps",
        type=int,
        default=EVAL_TIMESTEPS,
        help="Diffusion timesteps used when sampling the model for visualisation",
    )
    parser.add_argument(
        "--truncation-ratio",
        type=float,
        default=TRUNCATION_RATIO,
        help="Train/sampling truncation ratio for late-timestep diffusion",
    )
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX,
                        help="Truncate signed distance magnitudes before max-normalization (0 disables)")
    parser.add_argument(
        "--sdf-features",
        action=argparse.BooleanOptionalAction,
        default=SDF_FEATURES,
        help="Enable SDF feature channels for conditioning (disable for ablation)",
    )
    parser.add_argument(
        "--smart-init-features",
        action=argparse.BooleanOptionalAction,
        default=SMART_INIT_FEATURES,
        help="Enable Smart Init conditioning features in the control hint path",
    )
    parser.add_argument(
        "--batch-coords-features",
        action=argparse.BooleanOptionalAction,
        default=BATCH_COORDS_FEATURES,
        help="Enable batch-coordinate conditioning features in the control hint path",
    )
    parser.add_argument(
        "--smart-init-jitter-px",
        type=float,
        default=SMART_INIT_JITTER_PX,
        help="Gaussian micro-jitter strength in grid-pixel units for overfit Smart Init (0 disables)",
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
        help="Enable Gaussian micro-jitter on Smart Init points (requires --smart-init-jitter-px > 0)",
    )
    parser.add_argument(
        "--enable-smart-init-splat-sigma",
        "--enable-smart-init-splat",
        dest="enable_smart_init_splat_sigma",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_SMART_INIT_SPLAT_SIGMA,
        help="Enable Gaussian soft splatting for Smart Init grid; when disabled uses occupancy grid",
    )

    # Loss parameters
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=MIN_SNR_GAMMA,
        help="Gamma for Min-SNR loss weighting (0 disables)",
    )
    parser.add_argument(
        "--geom-clump-weight",
        type=float,
        default=GEOM_CLUMP_WEIGHT,
        help="Weight of clumped_pct in geometric score: score = cv + w*(clumped_pct/100)",
    )
    parser.add_argument(
        "--best-max-cv",
        type=float,
        default=BEST_MAX_CV,
        help="Only save best-geom checkpoint if CV <= this value",
    )
    parser.add_argument(
        "--best-max-clumped-pct",
        type=float,
        default=BEST_MAX_CLUMPED_PCT,
        help="Only save best-geom checkpoint if clumped_pct <= this value",
    )

    # Training configuration
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
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
        help="Resume from the latest checkpoint found in the output directory",
    )
    parser.add_argument(
        "--vis-every",
        "--save_every",
        dest="vis_every",
        type=int,
        default=VIS_EVERY,
        help="Visualise, sample, and checkpoint every N steps",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument(
        "--capacity-grid-size",
        type=int,
        default=CAPACITY_GRID_SIZE,
        help="Capacity grid size: >0 uses KxK, -1 uses full source image resolution",
    )
    parser.add_argument(
        "--show-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for CLI parity with train_control.py",
    )
    parser.add_argument(
        "--show-selected-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for CLI parity with train_control.py",
    )
    parser.add_argument(
        "--show-selected-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for CLI parity with train_control.py",
    )
    parser.add_argument(
        "--show-selected-predict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for CLI parity with train_control.py",
    )
    parser.add_argument(
        "--show-selected-gt-offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for CLI parity with train_control.py",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)

    parser.add_argument("--ode-steps", type=int, default=ODE_STEPS,
                        help="Number of Euler/Heun steps for Flow-Matching ODE sampling")
    parser.add_argument("--ode-method", choices=["euler", "heun"], default=ODE_METHOD,
                        help="ODE integrator for Flow-Matching sampling")
    parser.add_argument("--t-start", type=float, default=1.0,
                        help="Starting t for ODE sampling (1.0 = full noise->data)")
    parser.add_argument("--t-scale", type=float, default=T_SCALE,
                        help="Scale applied to continuous t before sinusoidal timestep embedding")
    parser.add_argument("--fm-coupling", choices=["gaussian", "smartinit"], default=FM_COUPLING,
                        help="FM source endpoint: 'gaussian' (noise->data) or 'smartinit' (smart-init->data coupling)")
    parser.add_argument("--fm-source-jitter-px", type=float, default=0.0,
                        help="Gaussian jitter (offset/pixel units) on the smart-init source during training (smartinit coupling)")
    parser.add_argument("--conditioning", choices=["controlnet", "concat", "spade"], default=CONDITIONING,
                        help="Conditioning architecture: 'controlnet' (dual-branch injections) or 'concat' (single-stage, attention-free)")
    parser.add_argument("--concat-smart-init-grid", action=argparse.BooleanOptionalAction, default=False,
                        help="(concat mode) also concat the smart-init occupancy grid as a condition channel")
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

    # ── GT-free objective-driven training (README_objective_driven_stippling.md) ──
    parser.add_argument("--gt-free", action=argparse.BooleanOptionalAction, default=True,
                        help="Objective-driven GT-free training: descend a differentiable "
                             "quality loss on the model's OWN points; no velocity MSE vs GT "
                             "offsets. Use --no-gt-free for the honest imitation baseline (A).")
    parser.add_argument("--unroll-steps", type=int, default=8,
                        help="ODE steps unrolled WITH grad each step (K); 1 = one-step x0")
    parser.add_argument("--unroll-method", choices=["euler", "heun"], default="euler")
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=False,
                        help="Checkpoint each unroll step to cut memory when K is large")
    parser.add_argument("--loss-grid", type=int, default=32,
                        help="Downsampled soft-Voronoi grid for the loss (O(N*G) memory)")
    parser.add_argument("--w-cap", type=float, default=1.0)
    parser.add_argument("--w-cvt", type=float, default=1.0)
    parser.add_argument("--w-pcf", type=float, default=1.0)
    parser.add_argument("--pcf-warp-grid", type=int, default=0,
                        help="rho-map resolution for the PCF density warp (0 = full res)")
    parser.add_argument("--stop-grad-warp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gbn-bar", default=None,
                        help="Bundle from eval/precompute_gbn_bar.py (empirical PCF target + bar)")
    parser.add_argument("--tau-start", type=float, default=0.02)
    parser.add_argument("--tau-end", type=float, default=0.002)
    parser.add_argument("--tau-mode", choices=["cosine", "linear", "exp"], default="cosine")
    parser.add_argument("--tau-warmup-frac", type=float, default=0.1)
    parser.add_argument("--gt-free-mode", choices=["unroll", "hybrid"], default="hybrid",
                        help="GT-free variant: 'hybrid' = imitation velocity loss (spacing "
                             "anchor) + geometry objective on the one-step x0 decode at low t "
                             "(robust, converges); 'unroll' = pure objective on the full sampler "
                             "unroll (attractive-only losses clump).")
    parser.add_argument("--geo-weight", type=float, default=0.1,
                        help="(hybrid) lambda on the auxiliary geometry objective (keep < velocity)")
    parser.add_argument("--geo-t-max", type=float, default=0.4,
                        help="(hybrid) apply geometry only where t < this (x0 decode reliable near data)")
    parser.add_argument("--geo-warmup", type=int, default=200,
                        help="(hybrid) velocity-only steps before enabling the geometry term")
    parser.add_argument("--data-root", default=DATA_ROOT,
                        help="Dataset root; source/ and target/ are read from here "
                             "(selects which image to overfit -- e.g. taksim vs monkey)")
    parser.add_argument("--init-ckpt", default=None,
                        help="Warm-start MODEL WEIGHTS ONLY from this checkpoint (fresh "
                             "optimizer/step). Transfer probe: Stage-A weights -> Stage-B "
                             "GT-free training on a NEW image.")
    args = parser.parse_args()

    if args.capacity_grid_size == 0 or args.capacity_grid_size < -1:
        raise ValueError("--capacity-grid-size must be > 0, or -1 for full source resolution")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── wandb init ───────────────────────────────────────────────────
    use_wandb = HAS_WANDB and WANDB_ACTIVE
    if use_wandb:
        load_wandb_key()
        run_name = datetime.now().strftime("fm-overfit-%Y%m%d-%H%M%S")
        wandb.init(
            project="Stipple-ControlNet",
            config=vars(args),
            name=run_name,
        )
        print(f"wandb run name: {run_name}")

    # ── pick the single example ──────────────────────────────────────
    source_dir = os.path.join(args.data_root, "source")
    target_dir = os.path.join(args.data_root, "target")
    source_files = sorted(os.listdir(source_dir))
    if args.sample_index >= len(source_files):
        sys.exit(f"sample-index {args.sample_index} out of range "
                 f"(dataset has {len(source_files)} files)")

    fname = source_files[args.sample_index]
    stem = os.path.splitext(fname)[0]
    source_path = os.path.join(source_dir, fname)
    target_path = os.path.join(target_dir, fname)

    if not os.path.exists(target_path):
        sys.exit(f"Target not found: {target_path}")

    out_dir = os.path.join(args.output_dir, stem)
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
        args.grid_size,
        device,
        sdf_features=args.sdf_features,
        sdf_truncate_px=args.sdf_truncate_px,
        cache_dir=out_dir,  # Cache raw SDF for faster repeated runs
    )

    source_img_01 = np.array(Image.open(source_path).convert("L"), dtype=np.float32) / 255.0
    smart_points_base = generate_smart_init_points_from_density(
        source_img_01,
        n_points=N_POINTS,
        seed=SMART_INIT_SEED,
    )
    smart_offsets_np = smart_init_points_to_offsets(smart_points_base)
    smart_init_offsets_base = torch.from_numpy(smart_offsets_np).unsqueeze(0).to(device)
    if args.smart_init_features:
        smart_grid_np = render_smart_init_grid(smart_points_base, grid_size=args.grid_size)
        smart_init_grid_base = torch.from_numpy(smart_grid_np).unsqueeze(0).to(device)
        smart_grid_centers_flat = _grid_centers_flat(args.grid_size, device, smart_init_offsets_base.dtype)
        smart_points_base = offsets_to_coords_gpu(smart_init_offsets_base, args.grid_size, smart_grid_centers_flat)
    else:
        smart_init_grid_base = None
        smart_grid_centers_flat = None

    if not args.sdf_features:
        high_res_sdf = None
        target_sdf = None

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

    # ── build Flow-Matching velocity network from scratch (no diffusion baseline) ──
    fm = FlowMatching(device=device, t_scale=args.t_scale,
                      coupling=args.fm_coupling, source_jitter_px=args.fm_source_jitter_px)

    conditioner = None
    if args.conditioning in ("concat", "spade"):
        # Single-stage, attention-free conditioning; no ControlNet dual-branch.
        n_cond = 1 + (1 if args.concat_smart_init_grid else 0)
        if args.conditioning == "spade":
            from control_fm.single_stage import build_spade_velocity_network, SPADEConditioner
            denoiser = build_spade_velocity_network(args.base_config_path, cond_channels=n_cond, device=device)
            conditioner = SPADEConditioner(denoiser, use_smart_init_grid=args.concat_smart_init_grid).to(device)
        else:
            from control_fm.single_stage import build_conditional_velocity_network, SingleStageConditioner
            denoiser = build_conditional_velocity_network(args.base_config_path, extra_in_channels=n_cond, device=device)
            conditioner = SingleStageConditioner(denoiser, use_smart_init_grid=args.concat_smart_init_grid).to(device)
        conditioner.train()
        denoiser.train()
        control_net = None
        train_module = conditioner
    else:
        denoiser = build_velocity_network(args.base_config_path, device=device)

        # NOTE: Create control_net BEFORE freezing denoiser so deep copies have requires_grad=True
        control_net = DynamicControlNet(
            denoiser,
            grid_size=args.grid_size,
            enable_gecco=args.enable_gecco,
            enable_adaptive_gate_injection=args.enable_adaptive_gate_injection,
            smart_init_features=args.smart_init_features,
            sdf_features=args.sdf_features,
            batch_coords_features=args.batch_coords_features,
        ).to(device)
        control_net.train()
        train_module = control_net

        # Freeze/unfreeze denoiser based on flag (AFTER control_net creation)
        if args.freeze_denoiser:
            for p in denoiser.parameters():
                p.requires_grad = False
            denoiser.eval()
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
    print(f"Smart Init features enabled           : {args.smart_init_features}")
    print(f"SDF features enabled                  : {args.sdf_features}")
    print(f"Batch coords features enabled         : {args.batch_coords_features}")
    print(f"Min-SNR gamma                         : {args.min_snr_gamma}")
    print(f"SDF truncation (px)                   : {args.sdf_truncate_px}")
    print(f"Smart Init micro-jitter (train, px)  : {args.smart_init_jitter_px}")
    print(f"Smart Init soft-splat sigma (px)     : {args.smart_init_splat_sigma_px}")
    print(f"Smart Init jitter enabled            : {args.enable_smart_init_jitter}")
    print(f"Smart Init splat-sigma enabled       : {args.enable_smart_init_splat_sigma}")
    if args.capacity_grid_size == -1:
        print("  Capacity grid size: full source resolution")
    else:
        print(f"  Capacity grid size: {args.capacity_grid_size}x{args.capacity_grid_size}")

    if conditioner is not None:
        optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.lr)
    elif args.freeze_denoiser:
        optimizer = torch.optim.AdamW(control_net.parameters(), lr=args.lr)
    else:
        all_params = list(control_net.parameters()) + list(denoiser.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=args.lr)

    def _lr_lambda(cur):
        if cur < args.lr_warmup:
            return float(cur + 1) / float(max(1, args.lr_warmup))
        if not args.lr_schedule:
            return 1.0
        progress = float(cur - args.lr_warmup) / float(max(1, args.steps - args.lr_warmup))
        progress = min(1.0, max(0.0, progress))
        floor = (args.lr_min / args.lr) if args.lr > 0 else 0.0
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── GT-free objective-driven setup ───────────────────────────────
    gt_free_obj = None
    tau_sched = None
    if getattr(args, "gt_free", False):
        _pcf_target, _edges = None, default_edges(device=device)
        if args.gbn_bar:
            _bar = torch.load(args.gbn_bar, map_location="cpu")
            _pcf_target = _bar["pcf_target"].to(device)
            _edges = _bar["edges"].to(device)
            print(f"[gt-free] loaded empirical PCF target from {args.gbn_bar}")
        elif args.w_pcf > 0:
            print("[gt-free] WARNING: --w-pcf>0 but no --gbn-bar; PCF term DISABLED. "
                  "Run eval/precompute_gbn_bar.py first.")
        gt_free_obj = CompositeObjective(
            loss_grid=args.loss_grid, w_cap=args.w_cap, w_cvt=args.w_cvt,
            w_pcf=(args.w_pcf if _pcf_target is not None else 0.0),
            pcf_target=_pcf_target, edges=_edges,
            stop_grad_warp=args.stop_grad_warp, pcf_warp_grid=(args.pcf_warp_grid or None))
        _tau_total = max(1, int(args.steps))
        tau_sched = TauSchedule(args.tau_start, args.tau_end,
                                warmup=int(args.tau_warmup_frac * _tau_total),
                                total=_tau_total, mode=args.tau_mode)
        print(f"[gt-free] objective-driven ON | K={args.unroll_steps} "
              f"weights cap/cvt/pcf={args.w_cap}/{args.w_cvt}/{gt_free_obj.w_pcf} | "
              f"tau {args.tau_start}->{args.tau_end} ({args.tau_mode})")
    latest_ckpt_path = os.path.join(ckpt_dir, "latest_controlnet.pt")
    best_val_score = float("inf")
    best_ckpt_path = None
    last_geom = {"cv": None, "clumped_pct": None, "score": None}

    if args.init_ckpt:
        _ck = torch.load(args.init_ckpt, map_location=device)
        _sd = _ck.get("model_state_dict", _ck.get("control_net", _ck)) if isinstance(_ck, dict) else _ck
        _missing, _unexpected = train_module.load_state_dict(_sd, strict=False)
        print(f"[init-ckpt] warm-started weights from {args.init_ckpt} "
              f"(weights only; fresh optimizer/step; missing={len(_missing)}, unexpected={len(_unexpected)})")
    start_step = 1
    if args.resume_latest and os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        train_module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_score = float(checkpoint.get("best_val_score", best_val_score))
        start_step = int(checkpoint.get("step", 0)) + 1
        print(f"Resumed from latest checkpoint: {latest_ckpt_path} (start_step={start_step})")

    if start_step > args.steps:
        print(f"Nothing to do: start_step={start_step} is already beyond steps={args.steps}")
        return

    # ── training loop ────────────────────────────────────────────────
    print(f"\n{'Step':>6}  {'Loss':>12}")
    print("-" * 22)

    losses = []
    loss_ema = None
    n_skipped = 0
    for step in range(start_step, args.steps + 1):
        # Reset to base values each step (mirrors train's fresh dataloader unpack)
        if args.smart_init_features:
            smart_init_grid = smart_init_grid_base
            smart_init_offsets = smart_init_offsets_base

            smart_coords = None
            if args.enable_smart_init_jitter or args.enable_smart_init_splat_sigma:
                smart_coords = smart_points_base.clone()

            if args.enable_smart_init_jitter:
                smart_coords = apply_gpu_jitter(
                    smart_coords,
                    jitter_strength_px=args.smart_init_jitter_px,
                    grid_size=GRID_SIZE,
                )
                smart_init_offsets = coords_to_offsets_gpu(smart_coords, GRID_SIZE, smart_grid_centers_flat)

            if args.enable_smart_init_splat_sigma:
                smart_init_grid = render_smart_init_gpu(
                    smart_coords,
                    grid_size=GRID_SIZE,
                    sigma_px=args.smart_init_splat_sigma_px,
                )
            elif args.enable_smart_init_jitter:
                smart_init_grid = render_occupancy_grid_gpu(smart_coords, grid_size=GRID_SIZE)
        else:
            smart_init_grid = None
            smart_init_offsets = smart_init_offsets_base

        if getattr(args, 'gt_free', False) and args.gt_free_mode == "unroll":
            # PURE GT-free: unroll the sampler from noise/smart-init and descend the composite
            # objective on the FINAL points -- no GT target. Attractive-only losses (cap/cvt) clump
            # here unless the PCF term dominates; kept for ablation.
            _g = x_0.shape[-1]
            _shape = [x_0.shape[0], 2, _g, _g]
            if conditioner is not None:
                conditioner.set_condition(
                    target_density,
                    smart_init_grid if args.concat_smart_init_grid else None,
                )
                _vfn = lambda _x, _tn: conditioner(_x, _tn)
            else:
                _controlled = DynamicControlledDenoiser(denoiser, control_net)
                _controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
                _vfn = lambda _x, _tn: _controlled(_x, _tn)
            _x_start = (fm.start_state(_shape, device=device, smart_init=smart_init_offsets)
                        if fm.coupling == "smartinit" else None)
            _tau = tau_sched.value(step)
            x0_pred = unroll_ode(fm, _vfn, _shape, device, args.unroll_steps, args.unroll_method,
                                 x_start=_x_start, grad_checkpoint=args.grad_checkpoint)
            denoise_loss, gtfree_comps = gt_free_obj(x0_pred, target_density, _tau)
            loss = denoise_loss
        else:
            # Imitation velocity loss -- the spacing prior inherited from the GT/GBN blue-noise targets.
            t = fm.sample_t(x_0.shape[0], device=device)
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
                    offsets_t, t_net, high_res, target_density,
                    high_res_sdf=high_res_sdf, target_sdf_map=target_sdf,
                    target_smart_init_map=smart_init_grid,
                )
                v_pred = denoiser(offsets_t, t_net, controls=controls)
            per_sample_mse = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=(1, 2, 3))
            denoise_loss = per_sample_mse.mean()
            gtfree_comps = {}
            if getattr(args, 'gt_free', False):
                # HYBRID (imitation-anchored objective refinement): add the geometry objective on the
                # cheap ONE-STEP x0 decode  x0 = x_t - t*v_pred, masked to LOW t (decode is a rough
                # guess near noise, reliable near data) and warmed up. Velocity loss stays dominant;
                # the geometry term only refines spacing on top of the inherited blue-noise prior.
                _lam = args.geo_weight if step >= args.geo_warmup else 0.0
                if _lam > 0.0:
                    _tt = t.view(-1, 1, 1, 1)
                    x0_pred = offsets_t - _tt * v_pred
                    _sel = (t < args.geo_t_max).nonzero(as_tuple=True)[0]
                    if _sel.numel() > 0:
                        _td = target_density[_sel] if target_density.shape[0] == t.shape[0] else target_density
                        _tau = tau_sched.value(step)
                        geo_loss, gtfree_comps = gt_free_obj(x0_pred.index_select(0, _sel), _td, _tau)
                        denoise_loss = denoise_loss + _lam * geo_loss
            loss = denoise_loss

        loss_val = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if conditioner is not None:
            params_to_opt = list(denoiser.parameters())
        elif args.freeze_denoiser:
            params_to_opt = list(control_net.parameters())
        else:
            params_to_opt = list(control_net.parameters()) + list(denoiser.parameters())
        torch.nn.utils.clip_grad_norm_(params_to_opt, args.grad_clip)

        # ── stability guard: skip the update on NaN/Inf grads/loss or a finite loss spike ──
        grads_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in params_to_opt)
        is_spike = (
            args.spike_factor > 0
            and loss_ema is not None
            and loss_val > args.spike_factor * loss_ema
        )
        if (not math.isfinite(loss_val)) or (not grads_ok) or is_spike:
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            reason = "nan/inf" if (not math.isfinite(loss_val) or not grads_ok) else "spike"
            ema_str = f"{loss_ema:.4f}" if loss_ema is not None else "nan"
            print(f"{step:6d}  SKIP({reason})  loss={loss_val:.4f} ema={ema_str}")
            losses.append(loss_ema if loss_ema is not None else loss_val)
            continue

        optimizer.step()
        scheduler.step()

        # robust EMA of loss for spike detection (updated only on accepted steps)
        loss_ema = loss_val if loss_ema is None else (1.0 - args.spike_ema_beta) * loss_ema + args.spike_ema_beta * loss_val
        losses.append(loss_val)

        if use_wandb:
            wandb.log({
                "loss": loss_val,
                "loss_denoise": float(denoise_loss.item()),
            }, step=step)

        if step % 50 == 0 or step == 1:
            print(f"{step:6d}  {loss_val:12.6f}")

        if step % args.vis_every == 0 or step == args.steps:
            train_module.eval()
            pts, raw = sample_from_model(
                fm,
                control_net,
                denoiser,
                high_res,
                high_res_sdf,
                target_density,
                target_sdf,
                smart_init_grid if args.smart_init_features else None,
                smart_init_offsets,
                device,
                n_samples=args.n_samples,
                ode_steps=args.ode_steps,
                ode_method=args.ode_method,
                t_start=args.t_start,
                conditioner=conditioner,
            )
            vis_path = os.path.join(vis_dir, f"vis_step{step:05d}.png")
            saved = visualize_overfit_metrics(
                source_np, target_np, gt_points,
                list(pts), vis_path, step=step,
                gt_offsets=gt_offsets,
                capacity_grid_size=args.capacity_grid_size,
            )
            np.save(os.path.join(points_dir, f"points_step{step:05d}.npy"), pts)
            print(f"  -> saved visualisation: {vis_path}")

            geom = geometric_validation_score(pts, clump_weight=args.geom_clump_weight)
            last_geom = geom
            print(
                "  -> geometry "
                f"CV={geom['cv']:.4f} | Clumped={geom['clumped_pct']:.2f}% | "
                f"Score={geom['score']:.4f}"
            )

            checkpoint = {
                "step": step,
                "model_state_dict": train_module.state_dict(),
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

            torch.save(checkpoint, latest_ckpt_path)

            if (
                geom["score"] < best_val_score
                and geom["cv"] <= args.best_max_cv
                and geom["clumped_pct"] <= args.best_max_clumped_pct
            ):
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

            train_module.train()
            if not args.freeze_denoiser:
                denoiser.train()
            else:
                denoiser.eval()

    # ── save final weights ───────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "dynamic_controlnet_fm_overfit.pt")
    torch.save({
        "model_state_dict": train_module.state_dict(),
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
        ax.set_title(f"FM Overfit Loss ({fname})")
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
