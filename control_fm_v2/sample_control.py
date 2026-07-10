"""Generate stipple point sets using Dynamic ControlNet fm (Truncated Control)."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from control_fm.conditioning import build_condition_tensors_from_image
from control_fm.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_fm.smart_init import (
    generate_smart_init_points_from_density,
    smart_init_points_to_offsets,
    render_smart_init_grid
)
from data.Transforms import to_pointset_optimal_transport
from control_fm_v2.flow_matching import MODEL_CONFIG, FlowMatching, build_velocity_network
from control_fm_v2.arch import assert_arch_matches, resolve_arch
from control_fm_v2.flow_matching import resolve_attention

# DEFUALTS
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
TRUNCATION_RATIO = 0.30
ODE_STEPS = 50
ODE_METHOD = "euler"  # keep in step with the active block in train_control.py
ETA = 0.0                  # 0 = deterministic ODE; > 0 = stochastic reverse SDE (1.0 = canonical)
FM_SOURCE_JITTER_PX = 0.0  # jitter the model was TRAINED with; required for SDE + smartinit
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
# ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False

# Model parameters
GRID_SIZE = 32
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
# SMART_INIT_JITTER_PX = 0.5
SMART_INIT_SPLAT_SIGMA_PX = 0.5

N_SAMPLES = 1
DEVICE = "cuda"
SHOW_DENOISING_INTERVAL = 50

EXPORT_CONDITIONS = True
EXPORT_PNG = True
EXPORT_NPY = True
TRACK_TIME = True
TRACK_TIME_FULL = False
PROFILE_TRACE = False  # NOTE: works only if track_time is True, and will add overhead to execution time
PROFILE_TRACE_CHROME = False  # NOTE: works only if profile_trace is True, and will add overhead to execution time
SHOW_DENOISING = False

# ----

# Editable defaults 
# control_fm_v2 has NO config file: the architecture lives in
# control_fm_v2/flow_matching.py::MODEL_CONFIG.
# Tier 2 -- must match the checkpoint. Single source of truth: flow_matching.MODEL_CONFIG
ATTN_MIDDLE = MODEL_CONFIG["attn_middle"]
ATTN_HEADS = MODEL_CONFIG["attn_heads"]
USE_EMA = True        # sample from the EMA weights when the ckpt has them
# NOTE: control_fm has NO pretrained baseline. build_velocity_network() / the single-stage
# builders read the model config only -- they never call torch.load. Both the base U-Net and
# the control branch are Flow-Matching weights trained from scratch and stored together in
# the control_fm checkpoint. There is deliberately no "model.ckpt" path here.

# CONTROL_CKPT_PATH = "control_fm/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_fm_ep8120.pt"
CONTROL_CKPT_PATH = "control_fm_v2/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_fm_ep10000.pt"

# Samples Compare
# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/monkey/source/emoji-one_4_monkey.png"
# # INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/quadratic_V2/source/quadratic_density_gradient.png"
# # INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/plant2/source/plant2_400x400.png"
# OUTPUT_DIR = "control_fm_v2/sample_outputs"
# TRUNCATION_RATIO = 0.30

# Sizes Compare
INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source/Icons-50/monkey/emoji-one_4_monkey.png"
# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress1/original/stress_test_density.png"
# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/GaussianBlueNoise/data_stress2_V2/original/plant4h_V2.png"
GRID_SIZE = 8
# GRID_SIZE = 16  # NOTE
# GRID_SIZE = 24  # NOTE
# GRID_SIZE = 32  # NOTE
# GRID_SIZE = 48  # NOTE
# GRID_SIZE = 64  # NOTE
# GRID_SIZE = 80
# GRID_SIZE = 96  
# GRID_SIZE = 112
OUTPUT_DIR = f"control_fm_v2/sample_outputs_{GRID_SIZE}_cpu"
TRUNCATION_RATIO = 0.30


# Teaser
# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/siggraph_icons/sig1.png"
# # INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/siggraph_icons/sig2.png"
# # INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/siggraph_icons/sig3.png"
# # INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/siggraph_icons/sig4.png"
# SHOW_DENOISING = True
# SHOW_DENOISING_INTERVAL = 100

# # OUTPUT_DIR = "control_fm/sample_outputs"
# # TRUNCATION_RATIO = 1.0

# OUTPUT_DIR = "control_fm/sample_outputs_trunc"
# TRUNCATION_RATIO = 0.3


# ── Helper Functions ──────────────────────────────────────────────────────────

def save_sample_image(image_path, pts, out_png_path):
    cond_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if cond_img is None:
        return

    h, w = cond_img.shape
    out_img = np.full((h, w), 255, dtype=np.uint8)

    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        cv2.imwrite(str(out_png_path), out_img)
        return

    px = np.rint(pts[:, 0] * (w - 1)).astype(np.int32)
    py = np.rint(pts[:, 1] * (h - 1)).astype(np.int32)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    out_img[py, px] = 0
    cv2.imwrite(str(out_png_path), out_img)


def _save_denoise_step(img_tensor, timestep_i, t_start, out_path):
    """Save a visualization of the denoising step as a scatter plot."""
    if not HAS_MPL:
        return
    offsets = img_tensor[0].detach().cpu().float().numpy()
    h, w = offsets.shape[1], offsets.shape[2]
    cx = (np.arange(w) + 0.5) / w
    cy = (np.arange(h) + 0.5) / h
    gx, gy = np.meshgrid(cx, cy)
    px = np.clip(gx + offsets[0] / w, 0.0, 1.0).flatten()
    py = np.clip(gy + offsets[1] / h, 0.0, 1.0).flatten()

    elapsed = t_start - 1 - timestep_i
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=110)
    ax.scatter(px, 1.0 - py, c="black", s=0.5, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"step {elapsed}/{max(t_start - 1, 1)} (t={timestep_i})", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


def _measure_elapsed_seconds(start_event, end_event, fallback_start, fallback_end):
    if start_event is not None and end_event is not None:
        end_event.synchronize()
        return start_event.elapsed_time(end_event) / 1000.0
    return fallback_end - fallback_start


def _summarize_timing_breakdown(timing_dict):
    if not timing_dict:
        return {}

    has_cuda_events = any(
        isinstance(value, list) and value and isinstance(value[0], tuple)
        for value in timing_dict.values()
    )
    if has_cuda_events and torch.cuda.is_available():
        torch.cuda.synchronize()

    summary = {}
    for key, value in timing_dict.items():
        if isinstance(value, list):
            total_s = 0.0
            for start_event, end_event in value:
                total_s += start_event.elapsed_time(end_event) / 1000.0
            summary[key] = total_s
        else:
            summary[key] = float(value)
    return summary


def _format_timing_dict(timing_dict):
    summary = _summarize_timing_breakdown(timing_dict)
    if not summary:
        return ""

    # Categorize keys dynamically
    unet_keys = {k: v for k, v in summary.items() if k.startswith("unet")}
    ctrl_keys = {k: v for k, v in summary.items() if k.startswith("ctrl") or k == "controlnet_forward"}
    diff_keys = {k: v for k, v in summary.items() if k.startswith("p_mean_variance") or k.startswith("p_sample")}
    other_keys = {k: v for k, v in summary.items() if k not in unet_keys and k not in ctrl_keys and k not in diff_keys and k != "p_mean_variance.model_forward"}

    parts = []
    
    # 1. Pull out the master total first so it stands alone at the front
    if "p_mean_variance.model_forward" in diff_keys:
        parts.append(f"model_forward={diff_keys.pop('p_mean_variance.model_forward'):.6f}s")
        
    def fmt_group(name, d):
        if not d: return ""
        # Sort alphabetically, but force macro "forward" keys to the front of their group
        sorted_items = sorted(d.items(), key=lambda x: (not x[0].endswith("forward"), x[0]))
        inner = ", ".join(f"{k}={v:.6f}s" for k, v in sorted_items)
        return f"{name}({inner})"

    # 2. Append the wrapped groups
    if unet_keys: parts.append(fmt_group("Baseline", unet_keys))
    if ctrl_keys: parts.append(fmt_group("ControlNet", ctrl_keys))
    if diff_keys: parts.append(fmt_group("DiffusionOps", diff_keys))
    if other_keys: parts.append(fmt_group("Misc", other_keys))

    return ", ".join(parts)


def _format_step_timing_line(step_index, timestep, step_total_s, sample_s, t_tensor_s, resample_noise_s, p_sample_breakdown=None):
    p_sample_breakdown_text = _format_timing_dict(p_sample_breakdown)
    p_sample_suffix = f"p_sample_ops=[{p_sample_breakdown_text}]" if p_sample_breakdown_text else ""
    return (
        f"step {step_index:04d} (t={timestep:04d}): "
        f"total={step_total_s:.6f}s, "
        f"p_sample={sample_s:.6f}s, "
        f"t_tensor={t_tensor_s:.6f}s, "
        f"resample_noise={resample_noise_s:.6f}s, "
        f"{p_sample_suffix}"
    )


def _save_condition_debug_tensors(
    high_res, high_res_sdf, target_density, target_sdf, 
    smart_init_grid_raw, smart_init_grid_model, out_dir, image_stem
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_map = {
        "high_res": high_res,
        "high_res_sdf": high_res_sdf,
        "target_density": target_density,
        "target_sdf": target_sdf,
        "smart_init_grid_raw": smart_init_grid_raw,
        "smart_init_grid_model_input": smart_init_grid_model,
    }
    for name, tensor in cond_map.items():
        if tensor is None: continue
        arr = tensor.detach().cpu().float().numpy().squeeze()
        np.save(out_dir / f"{image_stem}_{name}.npy", arr)

    if HAS_MPL:
        ordered_names = ["high_res", "high_res_sdf", "target_density", "target_sdf", "smart_init_grid_raw", "smart_init_grid_model_input"]
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=140)
        for ax, name in zip(axes.flat, ordered_names):
            tensor = cond_map[name]
            if tensor is None:
                ax.axis("off")
                ax.set_title(f"{name} (disabled)")
                continue
            arr = tensor.detach().cpu().float().numpy().squeeze()
            if arr.ndim != 2:
                ax.axis("off")
                ax.set_title(f"{name} (invalid)")
                continue
            vis = np.clip((arr + 1.0) * 0.5, 0.0, 1.0) if "sdf" in name else np.clip(arr, 0.0, 1.0)
            ax.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            ax.set_title(name)
        plt.tight_layout()
        plt.savefig(out_dir / f"{image_stem}_conditions_collage.png", dpi=140, bbox_inches="tight")
        plt.close()


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


def load_condition(image_path, grid_size, device, sdf_features=True, sdf_truncate_px=0.0):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_01 = img.astype(np.float32) / 255.0
    if not sdf_features:
        high_res = torch.from_numpy(image_01).unsqueeze(0).unsqueeze(0).to(device)
        target_density = torch.nn.functional.interpolate(high_res, size=(grid_size, grid_size), mode="area")
        return image_01, high_res, target_density, None, None
    high_res, target_density, high_res_sdf, target_sdf = build_condition_tensors_from_image(
        image_01, grid_size, device, sdf_truncate_px=sdf_truncate_px,
    )
    return image_01, high_res, target_density, high_res_sdf, target_sdf


# ── Core Inference Pipeline ───────────────────────────────────────────────────

def load_pipeline(control_ckpt_path, grid_size, enable_gecco, enable_adaptive_gate_injection,
                  smart_init_features, sdf_features, batch_coords_features, device,
                  coupling="gaussian", conditioning="controlnet", concat_smart_init_grid=False,
                  attn_middle=None, attn_heads=None, use_ema=True, strict_arch=True,
                  arch_from_ckpt=True):
    """Build the velocity network for the trained conditioning architecture and load its weights.

    Flow Matching: the network is instantiated from the model config only. There is no
    diffusion baseline and no pretrained ``model.ckpt`` -- every weight (base U-Net and,
    for the controlnet path, the control branch) comes from ``control_ckpt_path``.
    """
    conditioner = None
    attn_middle, attn_heads = resolve_attention(None, attn_middle, attn_heads)

    # Load the checkpoint FIRST and let its architecture stamp decide what to build. The
    # stamp describes the weights that exist, so it is authoritative; rebuilding from stale
    # CLI defaults + load_state_dict(strict=False) would silently sample a random network.
    state = torch.load(control_ckpt_path, map_location="cpu")
    eff = resolve_arch(
        state,
        {
            "conditioning": conditioning, "fm_coupling": coupling,
            "attn_middle": attn_middle, "attn_heads": attn_heads,
            "enable_gecco": enable_gecco,
            "enable_adaptive_gate_injection": enable_adaptive_gate_injection,
            "smart_init_features": smart_init_features, "sdf_features": sdf_features,
            "batch_coords_features": batch_coords_features,
            "concat_smart_init_grid": concat_smart_init_grid,
        },
        prefer_checkpoint=arch_from_ckpt, strict=strict_arch, ckpt_path=control_ckpt_path,
    )
    conditioning = eff["conditioning"]
    coupling = eff["fm_coupling"]
    attn_middle, attn_heads = eff["attn_middle"], eff["attn_heads"]
    enable_gecco = eff["enable_gecco"]
    enable_adaptive_gate_injection = eff["enable_adaptive_gate_injection"]
    smart_init_features = eff["smart_init_features"]
    sdf_features = eff["sdf_features"]
    batch_coords_features = eff["batch_coords_features"]
    concat_smart_init_grid = eff["concat_smart_init_grid"]
    model_config = eff.get("model_config")

    fm = FlowMatching(device=device, coupling=coupling)

    if conditioning in ("concat", "spade"):
        n_cond = 1 + (1 if concat_smart_init_grid else 0)
        if conditioning == "spade":
            from control_fm_v2.single_stage import build_spade_velocity_network, SPADEConditioner
            denoiser = build_spade_velocity_network(cond_channels=n_cond, device=device,
                                                    model_config=model_config,
                                                    attn_middle=attn_middle, attn_heads=attn_heads)
            conditioner = SPADEConditioner(denoiser, use_smart_init_grid=concat_smart_init_grid).to(device)
        else:
            from control_fm_v2.single_stage import build_conditional_velocity_network, SingleStageConditioner
            denoiser = build_conditional_velocity_network(extra_in_channels=n_cond, device=device,
                                                          model_config=model_config,
                                                          attn_middle=attn_middle, attn_heads=attn_heads)
            conditioner = SingleStageConditioner(denoiser, use_smart_init_grid=concat_smart_init_grid).to(device)
        sd = None
        if isinstance(state, dict):
            sd = _pick_weights(state, "control_net", "control_net_ema", use_ema, "conditioner")
            if sd is None:
                for k in ("model_state_dict", "state_dict"):
                    if isinstance(state.get(k), dict):
                        sd = state[k]
                        break
        if sd is None:
            sd = state
        conditioner.load_state_dict(sd, strict=False)
        conditioner.eval()
        return fm, denoiser, None, conditioner

    # Attention must be attached BEFORE DynamicControlNet deep-copies denoiser.middle.
    denoiser = build_velocity_network(device=device, model_config=model_config,
                                      attn_middle=attn_middle, attn_heads=attn_heads)
    denoiser.eval()

    control_net = DynamicControlNet(
        denoiser=denoiser,
        grid_size=grid_size,
        enable_gecco=enable_gecco,
        enable_adaptive_gate_injection=enable_adaptive_gate_injection,
        smart_init_features=smart_init_features,
        sdf_features=sdf_features,
        batch_coords_features=batch_coords_features,
    ).to(device)

    ctrl_sd = _pick_weights(state, "control_net", "control_net_ema", use_ema, "control_net")
    control_net.safe_load_state_dict(ctrl_sd if ctrl_sd is not None else state, strict=False)

    # The controlnet path trains the base U-Net jointly (FREEZE_DENOISER=False) but keeps it
    # OUTSIDE control_net, so its weights live under a separate "denoiser" key. Without this
    # the base network stays at its random init and every sample is meaningless.
    denoiser_sd = _pick_weights(state, "denoiser", "denoiser_ema", use_ema, "denoiser")
    if denoiser_sd is None:
        raise RuntimeError(
            f"Checkpoint '{control_ckpt_path}' has no 'denoiser' weights.\n"
            "The controlnet path trains the base U-Net jointly but older checkpoints only stored\n"
            "the control branch, so the base network cannot be restored and samples would be\n"
            "generated from a randomly initialised U-Net.\n"
            "Fix: retrain (or resume+re-save) with the current train_control.py, which now stores\n"
            "the 'denoiser' state dict. Alternatively use a concat/spade checkpoint, whose\n"
            "conditioner already contains the full network."
        )
    denoiser.load_state_dict(denoiser_sd, strict=False)
    denoiser.eval()
    control_net.eval()

    return fm, denoiser, control_net, conditioner



def _pick_weights(state, raw_key, ema_key, use_ema, label):
    """Return the state dict to load: EMA shadow when requested and present, else raw.

    The point of Tier 1.1 is to SAMPLE from the EMA weights. Silently loading the raw
    weights would discard the benefit, so the fallback is explicit and loud.
    """
    if use_ema:
        ema_sd = state.get(ema_key) if isinstance(state, dict) else None
        if ema_sd:
            print("  [%s] using EMA weights (%s)" % (label, ema_key))
            return ema_sd
        print("  [%s] WARNING: --use-ema requested but %r is absent "
              "(checkpoint predates the EMA fix). Falling back to raw weights." % (label, ema_key))
    return state.get(raw_key) if isinstance(state, dict) else state

# Defualts 
CONDITIONS_SUBDIR = "conditions"
PNG_SUBDIR = "png"
NPY_SUBDIR = "npy"
TIME_SUBDIR = "timestamps"
DENOISING_SUBDIR = "denoising_steps"

def process_single_image(
    image_path: Path, fm=None, denoiser=None, control_net=None, conditioner=None,
    ode_steps=50, ode_method="euler", eta=0.0, fm_source_jitter_px=0.0,
    grid_size=32, truncation_ratio=0.30,
    smart_init_features=False, sdf_features=False, smart_init_seed=42, sdf_truncate_px=8.0,
    enable_smart_init_splat_sigma=False, smart_init_splat_sigma_px=0.5,
    device="cuda", n_samples=1, show_denoising_interval=50,
    # Exports
    output_dir: Path | None=None,
    export_conditions=True, export_png=True, export_npy=True, 
    track_time=True, track_time_full=False, profile_trace=False, profile_trace_chrome=False, 
    show_denoising=False,
    conditions_dir=None, png_dir=None, npy_dir=None, timestamps_dir=None, denoising_dir=None,
):
    """Runs the inference pipeline for a single image, with exact timing tracking."""
    out_dir = output_dir if output_dir else Path("./output")
    img_stem = image_path.stem

    track_time_full = bool(track_time and track_time_full)
    profile_trace = bool(track_time and profile_trace)
    profile_trace_chrome = bool(track_time and profile_trace_chrome)

    conditions_dir = conditions_dir if conditions_dir else out_dir / CONDITIONS_SUBDIR
    png_dir = png_dir if png_dir else out_dir / PNG_SUBDIR
    npy_dir = npy_dir if npy_dir else out_dir / NPY_SUBDIR
    timestamps_dir = timestamps_dir if timestamps_dir else out_dir / TIME_SUBDIR
    denoising_dir = denoising_dir if denoising_dir else out_dir / DENOISING_SUBDIR

    if export_conditions: conditions_dir.mkdir(parents=True, exist_ok=True)
    if export_png: png_dir.mkdir(parents=True, exist_ok=True)
    if export_npy: npy_dir.mkdir(parents=True, exist_ok=True)
    if track_time: timestamps_dir.mkdir(parents=True, exist_ok=True)
    if show_denoising: denoising_dir.mkdir(parents=True, exist_ok=True)

    use_cuda_events = track_time and torch.cuda.is_available() and str(device).startswith("cuda")
    denoise_start_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    denoise_end_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None

    t_total_start = time.perf_counter()
    time_si, time_denoise, time_ot = 0.0, 0.0, 0.0

    # 1. Condition Loading
    image_01, high_res, target_density, high_res_sdf, target_sdf = load_condition(
        image_path, grid_size, device, sdf_features=sdf_features, sdf_truncate_px=sdf_truncate_px,
    )

    # 2. Smart Init / Prior (TIMED)
    t_si_start = time.perf_counter()
    smart_points = generate_smart_init_points_from_density(image_01, n_points=grid_size * grid_size, seed=smart_init_seed)
    smart_offsets_np = smart_init_points_to_offsets(smart_points)
    smart_init_offsets = torch.from_numpy(smart_offsets_np).unsqueeze(0).to(device)

    if smart_init_features:
        smart_grid_np = render_smart_init_grid(smart_points, grid_size=grid_size)
        smart_init_grid_raw = torch.from_numpy(smart_grid_np).unsqueeze(0).to(device)
        if enable_smart_init_splat_sigma:
            grid_centers_flat = _grid_centers_flat(grid_size, device, smart_init_offsets.dtype)
            smart_coords = _offsets_to_coords_gpu(smart_init_offsets, grid_size, grid_centers_flat)
            smart_init_grid = _render_smart_init_gpu(smart_coords, grid_size, smart_init_splat_sigma_px, device)
        else:
            smart_init_grid = smart_init_grid_raw
    else:
        smart_init_grid_raw, smart_init_grid = None, None
    time_si = time.perf_counter() - t_si_start

    # Condition Export
    if export_conditions:
        _save_condition_debug_tensors(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid_raw, smart_init_grid, conditions_dir, img_stem)

    # Setup velocity network (base denoiser) with control injection.
    if conditioner is not None:
        controlled = conditioner
        controlled.set_condition(target_density, smart_init_grid if conditioner.use_smart_init_grid else None)
        controlled.eval()
    else:
        base_denoiser = denoiser.locked if isinstance(denoiser, DynamicControlledDenoiser) else denoiser
        controlled = DynamicControlledDenoiser(base_denoiser, control_net)
        controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
        controlled.eval()

    n_steps = max(1, int(ode_steps))

    if getattr(fm, "coupling", "gaussian") == "smartinit":
        # Coupling: the flow's source IS the smart-init. Start the ODE exactly at the
        # smart-init (t=1) and integrate down to the data (t=0); no SDEdit noise blend.
        t_start_val = 1.0
        img = smart_init_offsets.expand(n_samples, -1, -1, -1).contiguous()
    else:
        # Flow-Matching ODE: integrate from t_start_val (noise side) down to 0 (data side).
        t_start_val = float(np.clip(truncation_ratio, 1e-4, 1.0))
        if t_start_val >= 1.0:
            # Pure-noise start: the smart-init prior is unused.
            img = torch.randn((n_samples, 2, grid_size, grid_size), device=device)
            time_si = 0.0
        else:
            # SDEdit-style truncated init: interpolate smart-init offsets toward noise at t_start_val.
            x_init = smart_init_offsets.expand(n_samples, -1, -1, -1).contiguous()
            eps0 = torch.randn_like(x_init)
            img = fm.interpolate(x_init, eps0, torch.full((n_samples,), t_start_val, device=device))

    t_schedule = torch.linspace(t_start_val, 0.0, n_steps + 1, device=device)

    # -- stochastic sampling setup (eta = 0 keeps the deterministic ODE path) --
    # Reverse SDE needs the source statistics (s, sigma) to recover the score from v.
    eta = float(eta)
    sde_source, sde_sigma = 0.0, 1.0
    if eta > 0.0:
        if getattr(fm, "coupling", "gaussian") == "smartinit":
            sde_source = smart_init_offsets.expand(n_samples, -1, -1, -1).contiguous()
            sde_sigma = float(fm_source_jitter_px)
            if sde_sigma <= 0.0:
                raise ValueError(
                    "eta > 0 with coupling='smartinit' requires --fm_source_jitter_px equal to the "
                    "jitter the model was TRAINED with. With zero jitter the interpolant is "
                    "deterministic, the score does not exist, and SDE sampling is undefined. "
                    "Use a gaussian-coupled checkpoint, pass the trained jitter, or set --eta 0."
                )
        else:
            sde_source, sde_sigma = 0.0, 1.0

    # 3. Denoising (TIMED)
    steps_png_dir = None
    steps_npy_dir = None
    if show_denoising:
        if show_denoising_interval <= 0:
            raise ValueError("show_denoising_interval must be > 0 when show_denoising is enabled")
        steps_png_dir = denoising_dir / "png"
        steps_npy_dir = denoising_dir / "npy"
        steps_png_dir.mkdir(parents=True, exist_ok=True)
        steps_npy_dir.mkdir(parents=True, exist_ok=True)

    profiler = None
    if profile_trace:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() and str(device).startswith("cuda") else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        profiler.__enter__()

    t_denoise_start = time.perf_counter()
    if denoise_start_event is not None:
        denoise_start_event.record()

    step_timing_lines = [] if track_time else None
    global_timing_summary = {}

    with torch.no_grad():
        for step_idx in tqdm(range(n_steps), total=n_steps, desc=f"FM-ODE {img_stem}"):
            if track_time:
                step_wall_start = time.perf_counter()
                t_tensor_start = time.perf_counter()
            t_cur = float(t_schedule[step_idx])
            t_next = float(t_schedule[step_idx + 1])
            dt = t_cur - t_next
            t_tensor = torch.full((n_samples,), t_cur, device=device)
            t_net = fm.net_t(t_tensor)
            t_tensor_time = time.perf_counter() - t_tensor_start if track_time else 0.0

            step_breakdown = {} if track_time_full else None
            if track_time:
                p_sample_start = time.perf_counter()
            v = controlled(img, t_net, timing_breakdown=step_breakdown)
            if eta > 0.0 and step_idx < n_steps - 1:
                # Reverse-SDE (Euler-Maruyama) step. g^2 = 2*eta*t*sigma^2 cancels the
                # (t*sigma^2) in score = -(x - s + (1-t)v)/(t*sigma^2), so no 1/t blow-up.
                # The last step is left deterministic so the output is not left noisy.
                corr = eta * (img - sde_source + (1.0 - t_cur) * v)
                noise_scale = sde_sigma * math.sqrt(2.0 * eta * t_cur * dt)
                img = img - dt * v - dt * corr + noise_scale * torch.randn_like(img)
            elif ode_method == "heun":
                x_euler = img - dt * v
                t_net_next = fm.net_t(torch.full((n_samples,), t_next, device=device))
                v_next = controlled(x_euler, t_net_next)
                img = img - dt * 0.5 * (v + v_next)
            else:
                img = img - dt * v
            p_sample_time = (time.perf_counter() - p_sample_start) if track_time else 0.0

            if show_denoising and (step_idx % show_denoising_interval == 0):
                step_png_path = steps_png_dir / f"{img_stem}_step_{step_idx:04d}.png"
                step_npy_path = steps_npy_dir / f"{img_stem}_step_{step_idx:04d}.npy"
                _save_denoise_step(img, step_idx, n_steps, str(step_png_path))
                np.save(step_npy_path, img.detach().cpu().numpy())

            if track_time:
                step_total_time = time.perf_counter() - step_wall_start
                step_timing_lines.append(
                    _format_step_timing_line(
                        step_index=step_idx,
                        timestep=int(round(t_cur * 1000)),
                        step_total_s=step_total_time,
                        sample_s=p_sample_time,
                        t_tensor_s=t_tensor_time,
                        resample_noise_s=0.0,
                        p_sample_breakdown=step_breakdown,
                    )
                )
                if track_time_full and step_breakdown:
                    step_summary = _summarize_timing_breakdown(step_breakdown)
                    for k, val in step_summary.items():
                        global_timing_summary[k] = global_timing_summary.get(k, 0.0) + val

    t_denoise_end = time.perf_counter()
    if denoise_end_event is not None:
        denoise_end_event.record()
    time_denoise = _measure_elapsed_seconds(denoise_start_event, denoise_end_event, t_denoise_start, t_denoise_end)

    inference_time = time.perf_counter() - t_total_start

    if profiler is not None:
        profiler.__exit__(None, None, None)
        summary_sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        summary_path = timestamps_dir / f"{img_stem}_profiler_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"--- Profiler Summary for {img_stem} ---\n")
            f.write(profiler.key_averages().table(sort_by=summary_sort_key, row_limit=25))
        print(f"Saved profiler summary to {summary_path}")

        if profile_trace_chrome:
            trace_path = timestamps_dir / f"{img_stem}_profiler_trace.json"
            profiler.export_chrome_trace(str(trace_path))
            print(f"Saved profiler chrome trace to {trace_path}")

    samples_raw = img.detach().cpu().numpy()

    # 4. Optimal Transport & Saving (TIMED)
    for idx, s in enumerate(samples_raw):
        suffix = f"_{idx + 1}" if n_samples > 1 else ""
        
        t_ot_start = time.perf_counter()
        pts = to_pointset_optimal_transport(s)
        pts = pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T
        time_ot += (time.perf_counter() - t_ot_start)

        if export_npy:
            npy_filename = npy_dir / f"{img_stem}{suffix}.npy"
            np.save(npy_filename, pts)
        if export_png:
            png_filename = png_dir / f"{img_stem}{suffix}.png"
            save_sample_image(image_path, pts, png_filename)

    time_total = time.perf_counter() - t_total_start

    # Time Tracking Export
    if track_time:
        with open(timestamps_dir / f"{img_stem}.txt", "w") as f:
            f.write(f"Smart Init Time: {time_si:.4f} s\n")
            f.write(f"Denoising Time: {time_denoise:.4f} s\n")
            f.write(f"Total Inference Time: {inference_time:.4f} s\n")
            f.write(f"Optimal Transport & Saving Time: {time_ot:.4f} s\n")
            f.write(f"Total Execution Time: {time_total:.4f} s\n")
            f.write("\n--- Denoising Step Timings ---\n")
            for line in step_timing_lines:
                f.write(line + "\n")
        
        # Export the expressive JSON breakdown
        if track_time_full:
            # Build the hierarchical structure
            unet_keys = {k: v for k, v in global_timing_summary.items() if k.startswith("unet")}
            ctrl_keys = {k: v for k, v in global_timing_summary.items() if k.startswith("ctrl")}
            diff_keys = {k: v for k, v in global_timing_summary.items() if k.startswith("p_mean_variance") or k.startswith("p_sample")}
            other_keys = {k: v for k, v in global_timing_summary.items() if k not in unet_keys and k not in ctrl_keys and k not in diff_keys}
            
            hierarchical_components = {}

            if unet_keys: hierarchical_components["Baseline"] = unet_keys
            if ctrl_keys: hierarchical_components["ControlNet"] = ctrl_keys
            if diff_keys: hierarchical_components["DiffusionOps"] = diff_keys
            if other_keys: hierarchical_components["Misc"] = other_keys

            json_data = {
                "image": img_stem,
                "grid_size": grid_size,
                "n_steps": n_steps,
                "total_inference_time": inference_time,
                "components": hierarchical_components
            }
            with open(timestamps_dir / f"{img_stem}_full.json", "w") as f:
                json.dump(json_data, f, indent=4)
            
    return time_total


# ── Public API ────────────────────────────────────────────────────────────────

def run_inference_on_directory(
    control_ckpt_path: str,
    grid_size: int = 32, enable_gecco: bool = True, enable_adaptive_gate_injection: bool = True,
    smart_init_features: bool = False, sdf_features: bool = False, batch_coords_features: bool = False,
    truncation_ratio: float = 0.30, smart_init_seed: int = 42,
    sdf_truncate_px: float = 8.0,
    ode_steps: int = 50, ode_method: str = "euler",
    enable_smart_init_splat_sigma: bool = False, smart_init_splat_sigma_px: float = 0.5,
    device: str = "cuda",
    # Exports
    export_png=True,
    export_npy=True,
    track_time=True,
    track_time_full=False,
    profile_trace=False,
    source_path=None,
    target_path=None,
    target_npy_path=None,
    timestamps_path=None,
    json_path=None,
    overwrite=True,
):
    """
    Dataset generation function for directory inference.
    
    Uses 'source' as input and generates 'target' and 'prompt.json'.
    Maintains the same directory hierarchy in source/ and target/.
    
    Output structure:
        dataset/
        ├── source/  (input images)
        ├── target/  (generated PNGs, same hierarchy as source/)
        └── prompt.json
    
    Exports PNGs only to target/ for dataset generation.
    Extended exports to paths: target_npy_path for NPY, timestamps_path for timing.
    """
    source_path = Path(source_path) if source_path else Path("./source")
    
    print(f"Initializing models on {device}...")
    fm, denoiser, control_net, conditioner = load_pipeline(
        control_ckpt_path, 
        grid_size, enable_gecco, enable_adaptive_gate_injection,
        smart_init_features, sdf_features, batch_coords_features, device
    )

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(source_path.rglob(ext))
        image_files.extend(source_path.rglob(ext.upper()))

    # Always process in deterministic sorted order.
    image_files = sorted(set(image_files), key=lambda p: str(p))

    print(f"Found {len(image_files)} images in {source_path}.")

    json_entries = []
    for i, img_path in enumerate(tqdm(image_files, total=len(image_files), desc="images"), 1):
        # Build JSON entry (always uses OT for dataset generation)
        source_rel_subpath = img_path.relative_to(source_path)
        target_rel_subpath = source_rel_subpath.with_suffix('.png')
        json_entries.append(
            {
                "source": f"source/{source_rel_subpath.as_posix()}",
                "target": f"target/{target_rel_subpath.as_posix()}",
                "prompt": "Stippling",
            }
        )

        # Check if output PNG already exists (dataset generation always exports PNG)
        target_png_path = target_path / f"{img_path.stem}.png"
        if not overwrite and target_png_path.exists():
            print(f"[{i}/{len(image_files)}] Skipping {img_path.name}: output PNG already exists")
            continue
        
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        process_single_image(
            image_path=img_path, 
            fm=fm, denoiser=denoiser, control_net=control_net, conditioner=conditioner,
            grid_size=grid_size, truncation_ratio=truncation_ratio,
            ode_steps=ode_steps, ode_method=ode_method,
            smart_init_features=smart_init_features, sdf_features=sdf_features,
            smart_init_seed=smart_init_seed, sdf_truncate_px=sdf_truncate_px,
            enable_smart_init_splat_sigma=enable_smart_init_splat_sigma,
            smart_init_splat_sigma_px=smart_init_splat_sigma_px,
            device=device, n_samples=1,
            # Exports
            output_dir=None,
            export_conditions=False, export_png=export_png, export_npy=export_npy, 
            track_time=track_time, track_time_full=track_time_full, profile_trace=profile_trace, profile_trace_chrome=False, 
            show_denoising=False,
            conditions_dir=None, png_dir=target_path, npy_dir=target_npy_path, timestamps_dir=timestamps_path, denoising_dir=None
        )

    # Write prompt.json at the root dataset level
    if json_path is not None:
        with json_path.open("w", encoding="utf-8") as f:
            for entry in json_entries:
                f.write(json.dumps(entry) + "\n")
    else:
        print("Warning: json_path not provided, skipping prompt.json export.")

    print("Directory processing complete.")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn-middle", dest="attn_middle", action=argparse.BooleanOptionalAction, default=ATTN_MIDDLE,
                        help="Must match the checkpoint: zero-init bottleneck attention")
    parser.add_argument("--attn-heads", dest="attn_heads", type=int, default=ATTN_HEADS)
    parser.add_argument("--use-ema", dest="use_ema", action=argparse.BooleanOptionalAction, default=USE_EMA,
                        help="Load the EMA weights from the checkpoint when present")
    parser.add_argument("--control_ckpt_path", default=CONTROL_CKPT_PATH)

    # Input routing (output path inferred dynamically)
    parser.add_argument("--image_path", default=INPUT_IMAGE_PATH, help="Path to the input image.")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Base directory where single-image exports will be saved.")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    
    # Model Flags
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    parser.add_argument("--enable_gecco", default=ENABLE_GECCO, action=argparse.BooleanOptionalAction)
    parser.add_argument("--enable_adaptive_gate_injection", default=ENABLE_ADAPTIVE_GATE_INJECTION, action=argparse.BooleanOptionalAction)
    parser.add_argument("--truncation_ratio", type=float, default=TRUNCATION_RATIO)
    parser.add_argument("--fm-coupling", dest="fm_coupling", choices=["gaussian", "smartinit"], default="gaussian",
                        help="FM source: gaussian (noise->data; SDEdit uses truncation_ratio) or smartinit (start ODE exactly at smart-init, t=1->0)")
    parser.add_argument("--conditioning", choices=["controlnet", "concat", "spade"], default="controlnet",
                        help="Conditioning architecture: 'controlnet' (dual-branch), 'concat' or 'spade' (single-stage, attention-free)")
    parser.add_argument("--concat-smart-init-grid", dest="concat_smart_init_grid", action=argparse.BooleanOptionalAction, default=False,
                        help="(concat/spade) also feed the smart-init occupancy grid as a condition channel")
    parser.add_argument("--smart_init_features", action=argparse.BooleanOptionalAction, default=SMART_INIT_FEATURES)
    parser.add_argument("--sdf_features", action=argparse.BooleanOptionalAction, default=SDF_FEATURES)
    parser.add_argument("--batch_coords_features", action=argparse.BooleanOptionalAction, default=BATCH_COORDS_FEATURES)
    parser.add_argument("--ode_steps", type=int, default=ODE_STEPS)
    parser.add_argument("--ode_method", choices=["euler", "heun"], default=ODE_METHOD)
    parser.add_argument("--strict-arch", dest="strict_arch", action=argparse.BooleanOptionalAction, default=True,
                        help="(when --no-arch-from-ckpt) refuse a checkpoint whose stamped architecture differs")
    parser.add_argument("--arch-from-ckpt", dest="arch_from_ckpt", action=argparse.BooleanOptionalAction, default=True,
                        help="Build the network from the checkpoint's architecture stamp (authoritative). "
                             "Use --no-arch-from-ckpt to build from CLI flags instead.")
    parser.add_argument("--eta", type=float, default=ETA,
                        help="Stochasticity: 0 = deterministic ODE (default), 1.0 = canonical reverse SDE. "
                             "Inference-time only; requires no retraining.")
    parser.add_argument("--fm_source_jitter_px", type=float, default=FM_SOURCE_JITTER_PX,
                        help="Source jitter the checkpoint was TRAINED with. Required when --eta > 0 "
                             "and --fm-coupling smartinit; ignored for gaussian coupling.")
    parser.add_argument("--sdf_truncate_px", type=float, default=SDF_TRUNCATE_PX)
    parser.add_argument("--smart_init_seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--smart_init_splat_sigma_px", type=float, default=SMART_INIT_SPLAT_SIGMA_PX)
    parser.add_argument("--enable_smart_init_splat_sigma", action=argparse.BooleanOptionalAction, default=ENABLE_SMART_INIT_SPLAT_SIGMA)
    parser.add_argument("--show_denoising_interval", type=int, default=SHOW_DENOISING_INTERVAL, help="Interval for saving denoising steps (every N steps)")
    parser.add_argument("--device", default=DEVICE)
    
     # Artifact Exports
    parser.add_argument("--export_conditions", action=argparse.BooleanOptionalAction, default=EXPORT_CONDITIONS)
    parser.add_argument("--export_png", action=argparse.BooleanOptionalAction, default=EXPORT_PNG)
    parser.add_argument("--export_npy", action=argparse.BooleanOptionalAction, default=EXPORT_NPY)
    parser.add_argument("--track_time", action=argparse.BooleanOptionalAction, default=TRACK_TIME, help="Export a _time.txt file tracking stage speeds")
    parser.add_argument("--track_time_full", action=argparse.BooleanOptionalAction, default=TRACK_TIME_FULL, help="Enable deep CUDA-event profiling for subcomponents")
    parser.add_argument("--profile_trace", action=argparse.BooleanOptionalAction, default=PROFILE_TRACE, help="Export a PyTorch profiler chrome trace alongside the timing files")
    parser.add_argument("--profile_trace_chrome", action=argparse.BooleanOptionalAction, default=PROFILE_TRACE_CHROME, help="Export PyTorch profiler trace in Chrome format (requires --profile_trace)")
    parser.add_argument("--show_denoising", action=argparse.BooleanOptionalAction, default=SHOW_DENOISING, help="Export denoising steps to denoising_steps/png and denoising_steps/npy")

    parser.add_argument("--conditions_dir", default=None, help="Directory to save condition tensors (overrides default)")
    parser.add_argument("--png_dir", default=None, help="Directory to save output PNGs (overrides default)")
    parser.add_argument("--npy_dir", default=None, help="Directory to save output NPYs (overrides default)")
    parser.add_argument("--timestamps_dir", default=None, help="Directory to save timing TXT files (overrides default)")
    parser.add_argument("--denoising_dir", default=None, help="Directory to save denoising step visualizations (overrides default)")

    args = parser.parse_args()
    run(args)


def run(args):
    image_path = Path(args.image_path)
    output_path = Path(args.output_dir)

    if image_path.is_dir():
        raise ValueError(
            "This CLI now runs a single input image only. Use run_inference_on_directory() "
            "directly if you need folder processing."
        )

    fm, denoiser, control_net, conditioner = load_pipeline(
        args.control_ckpt_path, args.grid_size, args.enable_gecco, args.enable_adaptive_gate_injection,
        args.smart_init_features, args.sdf_features, args.batch_coords_features, args.device, args.fm_coupling,
        conditioning=args.conditioning, concat_smart_init_grid=args.concat_smart_init_grid,
        attn_middle=args.attn_middle, attn_heads=args.attn_heads, use_ema=args.use_ema,
        strict_arch=args.strict_arch, arch_from_ckpt=args.arch_from_ckpt,
    )

    # Single-image behavior: write under sample_outputs/<image_stem>.
    single_output_dir = output_path / image_path.stem

    process_single_image(
        image_path=Path(image_path),
        fm=fm, denoiser=denoiser, control_net=control_net, conditioner=conditioner,
        grid_size=args.grid_size, truncation_ratio=args.truncation_ratio,
        smart_init_features=args.smart_init_features, sdf_features=args.sdf_features,
        smart_init_seed=args.smart_init_seed, sdf_truncate_px=args.sdf_truncate_px,
        enable_smart_init_splat_sigma=args.enable_smart_init_splat_sigma,
        smart_init_splat_sigma_px=args.smart_init_splat_sigma_px,
        device=args.device, n_samples=args.n_samples,
        ode_steps=args.ode_steps, ode_method=args.ode_method,
        eta=args.eta, fm_source_jitter_px=args.fm_source_jitter_px,
        show_denoising_interval=args.show_denoising_interval,
        # Exports
        output_dir=Path(single_output_dir),
        export_conditions=args.export_conditions, export_png=args.export_png, export_npy=args.export_npy, track_time=args.track_time, track_time_full=args.track_time_full, profile_trace=args.profile_trace, profile_trace_chrome=args.profile_trace_chrome, 
        show_denoising=args.show_denoising,
        conditions_dir=args.conditions_dir, png_dir=args.png_dir, npy_dir=args.npy_dir, timestamps_dir=args.timestamps_dir, denoising_dir=args.denoising_dir,
    )
    print("Done single image processing at:", single_output_dir)

if __name__ == "__main__":
    main()
