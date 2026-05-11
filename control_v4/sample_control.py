"""Generate stipple point sets using Dynamic ControlNet V4 (Truncated Control)."""

import argparse
import json
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

from control_v4.conditioning import build_condition_tensors_from_image
from control_v4.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from control_v4.smart_init import (
    add_noise_at_t,
    generate_smart_init_points_from_density,
    smart_init_points_to_offsets,
)
from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig

# Editable defaults 
CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT = "config/GBN/model.ckpt"
# CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep8120.pt"
CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"

INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/monkey/source/emoji-one_4_monkey.png"

# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/quadratic_V2/source/quadratic_density_gradient.png"

# INPUT_IMAGE_PATH = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/results/plant2/source/plant2_400x400.png"

OUTPUT_DIR = "control_v4/sample_outputs"

ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
TIMESTEPS = 1000
TRUNCATION_RATIO = 0.30
RESAMPLE_JUMPS = 2
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
T_START_STEP = -1

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


def _save_condition_debug_tensors(
    high_res, high_res_sdf, target_density, target_sdf, 
    smart_init_grid_raw, smart_init_grid_model, out_dir
):
    os.makedirs(out_dir, exist_ok=True)
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
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)

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
        plt.savefig(os.path.join(out_dir, "conditions_collage.png"), dpi=140, bbox_inches="tight")
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

def load_pipeline(config_path, base_ckpt, control_ckpt, grid_size, enable_gecco, smart_init_features, sdf_features, batch_coords_features, device):
    """Loads the U-Net and ControlNet into VRAM once."""
    diffusion = ParseSampleConfig(config_path)
    diffusion.load_state_dict(torch.load(base_ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    denoiser = diffusion.model
    denoiser.eval()

    control_net = DynamicControlNet(
        denoiser,
        grid_size=grid_size,
        enable_gecco=enable_gecco,
        smart_init_features=smart_init_features,
        sdf_features=sdf_features,
        batch_coords_features=batch_coords_features,
    ).to(device)
    
    state = torch.load(control_ckpt, map_location="cpu")
    control_net.safe_load_state_dict(state, strict=False)
    control_net.eval()
    
    return diffusion, control_net


def process_single_image(
    image_path, output_dir, timestamp_dir=None, diffusion=None, control_net=None,
    grid_size=32, timesteps=1000, truncation_ratio=0.30, t_start_step=-1, resample_jumps=2,
    smart_init_features=False, sdf_features=False, smart_init_seed=42, sdf_truncate_px=8.0,
    enable_smart_init_splat_sigma=False, smart_init_splat_sigma_px=0.5, no_ot=False,
    export_png=True, export_npy=True, export_conditions=True, track_time=False,
    device="cuda", n_samples=1, use_subdirs: bool = True
):
    """Runs the inference pipeline for a single image, with exact timing tracking.
    
    For dataset generation (e.g., via run_inference_on_directory), always uses OT.
    The no_ot flag is primarily for testing/debugging single-image inference.
    """
    out_dir = Path(output_dir)
    img_stem = Path(image_path).stem

    # When `use_subdirs` is True (legacy/sample mode) create `png/` and `npy/`
    # subfolders. For dataset generation we set `use_subdirs=False` so outputs
    # are written directly under the `target/` folder to preserve hierarchy.
    png_dir = out_dir / "png" if use_subdirs else out_dir
    npy_dir = out_dir / "npy" if use_subdirs else out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    if use_subdirs:
        if export_npy: npy_dir.mkdir(parents=True, exist_ok=True)
        if export_png: png_dir.mkdir(parents=True, exist_ok=True)
    
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
        from control_v4.smart_init import render_smart_init_grid
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
        cond_dir = out_dir / "conditions"
        _save_condition_debug_tensors(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid_raw, smart_init_grid, cond_dir)

    # Setup Denoiser.
    # If diffusion.model is already wrapped from a previous image, unwrap to the base denoiser
    # to avoid nested wrappers that break on the `controls` kwarg.
    base_denoiser = diffusion.model.locked if isinstance(diffusion.model, DynamicControlledDenoiser) else diffusion.model
    controlled = DynamicControlledDenoiser(base_denoiser, control_net)
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, smart_init_grid)
    diffusion.model = controlled
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    # Determine schedule start
    t_start = t_start_step if t_start_step >= 0 else int(timesteps * truncation_ratio)
    t_start = int(np.clip(t_start, 1, max(timesteps - 1, 1)))

    if truncation_ratio == 1.0 and t_start_step < 0:
        img = torch.randn((n_samples, 2, grid_size, grid_size), device=device)
        t_start = timesteps - 1
        time_si = 0.0 # Override because we aren't using the generated prior
    else:
        x_init = smart_init_offsets.expand(n_samples, -1, -1, -1).contiguous()
        alpha_t = diffusion.alphas_cumprod[t_start]
        img = add_noise_at_t(x_init, alpha_t)

    # 3. Denoising (TIMED)
    t_denoise_start = time.perf_counter()
    with torch.no_grad() if resample_jumps == 0 else torch.enable_grad():
        for i in tqdm(reversed(range(t_start)), total=t_start, desc=f"Denoising {img_stem}"):
            t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
            for u in range(resample_jumps + 1):
                with torch.no_grad():
                    img = diffusion.p_sample(img, cond=None, t=t_tensor, clip_denoised=diffusion.sample_clip, with_sampling=True)
                if u == resample_jumps or i == 0: break
                beta_i = diffusion.betas[i]
                noise = torch.randn_like(img)
                img = (1.0 - beta_i).sqrt() * img + beta_i.sqrt() * noise
    time_denoise = time.perf_counter() - t_denoise_start

    samples_raw = img.detach().cpu().numpy()

    # 4. Optimal Transport & Saving (TIMED)
    for idx, s in enumerate(samples_raw):
        suffix = f"_{idx + 1}" if n_samples > 1 else ""
        
        t_ot_start = time.perf_counter()
        if not no_ot:
            pts = to_pointset_optimal_transport(s)
            pts = pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T
        else:
            pts = s
        time_ot += (time.perf_counter() - t_ot_start)

        if export_npy:
            np.save(npy_dir / f"{img_stem}{suffix}.npy", pts if not no_ot else s)
        if export_png and not no_ot:
            save_sample_image(image_path, pts, png_dir / f"{img_stem}{suffix}.png")

    time_total = time.perf_counter() - t_total_start

    # Time Tracking Export
    if track_time:
        ts_dir = Path(timestamp_dir) if timestamp_dir else out_dir
        ts_dir.mkdir(parents=True, exist_ok=True)
        with open(ts_dir / f"{img_stem}_time.txt", "w") as f:
            f.write(f"Smart Init Time: {time_si:.4f} s\n")
            f.write(f"Denoising Time: {time_denoise:.4f} s\n")
            f.write(f"Optimal Transport Time: {time_ot:.4f} s\n")
            f.write(f"Total Inference Time: {time_total:.4f} s\n")
            
    return time_total


# ── Public API ────────────────────────────────────────────────────────────────

def run_inference_on_directory(
    input_dir: str, config_path: str, base_ckpt: str, control_ckpt: str,
    grid_size: int = 32, timesteps: int = 1000, enable_gecco: bool = True,
    smart_init_features: bool = False, sdf_features: bool = False, batch_coords_features: bool = False,
    truncation_ratio: float = 0.30, t_start_step: int = -1, smart_init_seed: int = 42,
    sdf_truncate_px: float = 8.0, resample_jumps: int = 2,
    enable_smart_init_splat_sigma: bool = False, smart_init_splat_sigma_px: float = 0.5,
    export_png: bool = True,
    export_conditions: bool = True, track_time: bool = True, device: str = "cuda",
    target_dir: str | None = None, timestamps_dir: str | None = None,
    json_path: str | None = None, overwrite: bool = False
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
    
    Always uses Optimal Transport (no_ot=False internally).
    Exports PNGs only to target/ for dataset generation.
    """
    in_path = Path(input_dir)
    target_root = Path(target_dir) if target_dir is not None else None
    timestamps_root = Path(timestamps_dir) if timestamps_dir is not None else None
    json_file = Path(json_path) if json_path is not None else None
    
    print(f"Initializing models on {device}...")
    diffusion, control_net = load_pipeline(
        config_path, base_ckpt, control_ckpt, grid_size, enable_gecco, 
        smart_init_features, sdf_features, batch_coords_features, device
    )

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(in_path.rglob(ext))
        image_files.extend(in_path.rglob(ext.upper()))

    # Always process in deterministic sorted order.
    image_files = sorted(set(image_files), key=lambda p: str(p))

    print(f"Found {len(image_files)} images in {input_dir}.")

    if target_root is not None:
        target_root.mkdir(parents=True, exist_ok=True)
    if timestamps_root is not None:
        timestamps_root.mkdir(parents=True, exist_ok=True)

    json_entries = []

    for i, img_path in enumerate(tqdm(image_files, total=len(image_files), desc="images"), 1):
        if target_root is None:
            if 'source' not in img_path.parts:
                print(f"[{i}/{len(image_files)}] Skipping {img_path.name}: Not located inside a 'source' folder.")
                continue

            # Find exactly where 'source' is in the path
            source_idx = img_path.parts.index('source')

            # Base directory containing the 'source' folder
            base_path = Path(*img_path.parts[:source_idx])

            # Extract the subpath strictly AFTER the 'source' folder (excluding the filename)
            # E.g., .../source/item_01/image.png -> subpath = "item_01"
            rel_subpath = Path(*img_path.parts[source_idx + 1:]).parent

            # Build sibling target and timestamps folders
            target_out_dir = base_path / "target" / rel_subpath
            timestamp_out_dir = base_path / "timestamps" / rel_subpath
        else:
            rel_subpath = img_path.relative_to(in_path).parent
            target_out_dir = target_root / rel_subpath
            timestamp_out_dir = timestamps_root / rel_subpath if timestamps_root is not None else None

        # Build JSON entry (always uses OT for dataset generation)
        target_rel_path = img_path.relative_to(in_path).with_suffix('.png')
        source_rel_path = img_path.relative_to(in_path)
        json_entries.append(
            {
                "source": f"source/{source_rel_path.as_posix()}",
                "target": f"target/{target_rel_path.as_posix()}",
                "prompt": "Stippling",
            }
        )

        # Check if output PNG already exists (dataset generation always exports PNG)
        target_png_path = Path(target_out_dir) / f"{img_path.stem}.png"

        if not overwrite and target_png_path.exists():
            print(f"[{i}/{len(image_files)}] Skipping {img_path.name}: output PNG already exists")
            continue
        
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        # For dataset generation, always use OT (no_ot=False)
        process_single_image(
            image_path=str(img_path), 
            output_dir=str(target_out_dir),
            timestamp_dir=str(timestamp_out_dir) if timestamp_out_dir is not None else None,
            diffusion=diffusion, control_net=control_net,
            grid_size=grid_size, timesteps=timesteps, truncation_ratio=truncation_ratio,
            t_start_step=t_start_step, resample_jumps=resample_jumps,
            smart_init_features=smart_init_features, sdf_features=sdf_features,
            smart_init_seed=smart_init_seed, sdf_truncate_px=sdf_truncate_px,
            enable_smart_init_splat_sigma=enable_smart_init_splat_sigma,
            smart_init_splat_sigma_px=smart_init_splat_sigma_px,
            no_ot=False, export_png=True, export_npy=False,
            export_conditions=export_conditions, track_time=track_time,
            device=device, n_samples=1, use_subdirs=False
        )

    # Write prompt.json at the root dataset level
    if json_file is not None:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with json_file.open("w", encoding="utf-8") as f:
            for entry in json_entries:
                f.write(json.dumps(entry) + "\n")
    else:
        # Default: write prompt.json at the dataset root
        if target_root is None:
            # Infer root from the base_path of the first image
            if image_files:
                first_img = image_files[0]
                if 'source' in first_img.parts:
                    source_idx = first_img.parts.index('source')
                    dataset_root = Path(*first_img.parts[:source_idx])
                    default_json_file = dataset_root / "prompt.json"
                    with open(default_json_file, "w", encoding="utf-8") as f:
                        for entry in json_entries:
                            f.write(json.dumps(entry) + "\n")
                    print(f"Wrote prompt.json to {default_json_file}")
        else:
            default_json_file = target_root.parent / "prompt.json"
            with open(default_json_file, "w", encoding="utf-8") as f:
                for entry in json_entries:
                    f.write(json.dumps(entry) + "\n")
            print(f"Wrote prompt.json to {default_json_file}")

    print("Directory processing complete.")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--base_ckpt", default=BASE_CKPT)
    parser.add_argument("--control_ckpt", default=CONTROL_CKPT)
    
    # Input routing (output path inferred dynamically)
    parser.add_argument("--input", default=INPUT_IMAGE_PATH, help="Path to the input image.")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="Base directory where single-image exports will be saved.")
    
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--grid_size", type=int, default=GRID_SIZE)
    parser.add_argument("--no_ot", action="store_true")
    
    # Artifact Exports
    parser.add_argument("--export-png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-npy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-conditions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track-time", action=argparse.BooleanOptionalAction, default=False, help="Export a _time.txt file tracking stage speeds")
    
    # Model Flags
    parser.add_argument("--enable-gecco", default=ENABLE_GECCO, action=argparse.BooleanOptionalAction)
    parser.add_argument("--smart-init-features", action=argparse.BooleanOptionalAction, default=SMART_INIT_FEATURES)
    parser.add_argument("--batch-coords-features", action=argparse.BooleanOptionalAction, default=BATCH_COORDS_FEATURES)
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    parser.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)
    parser.add_argument("--sdf-features", action=argparse.BooleanOptionalAction, default=SDF_FEATURES)
    parser.add_argument("--truncation-ratio", type=float, default=TRUNCATION_RATIO)
    parser.add_argument("--t-start-step", type=int, default=T_START_STEP)
    parser.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    parser.add_argument("--smart-init-splat-sigma-px", type=float, default=SMART_INIT_SPLAT_SIGMA_PX)
    parser.add_argument("--enable-smart-init-splat-sigma", action=argparse.BooleanOptionalAction, default=ENABLE_SMART_INIT_SPLAT_SIGMA)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out_dir)

    if input_path.is_dir():
        raise ValueError(
            "This CLI now runs a single input image only. Use run_inference_on_directory() "
            "directly if you need folder processing."
        )

    diffusion, control_net = load_pipeline(
        args.config, args.base_ckpt, args.control_ckpt, args.grid_size, args.enable_gecco,
        args.smart_init_features, args.sdf_features, args.batch_coords_features, args.device
    )

    # Single-image behavior: write under sample_outputs/<image_stem>.
    single_out_dir = out_path / input_path.stem

    process_single_image(
        image_path=str(input_path), output_dir=str(single_out_dir), timestamp_dir=None,
        diffusion=diffusion, control_net=control_net,
        grid_size=args.grid_size, timesteps=args.timesteps, truncation_ratio=args.truncation_ratio,
        t_start_step=args.t_start_step, resample_jumps=args.resample_jumps,
        smart_init_features=args.smart_init_features, sdf_features=args.sdf_features,
        smart_init_seed=args.smart_init_seed, sdf_truncate_px=args.sdf_truncate_px,
        enable_smart_init_splat_sigma=args.enable_smart_init_splat_sigma,
        smart_init_splat_sigma_px=args.smart_init_splat_sigma_px,
        no_ot=args.no_ot, export_png=args.export_png, export_npy=args.export_npy,
        export_conditions=args.export_conditions, track_time=args.track_time,
        device=args.device, n_samples=args.n_samples
    )
    print("Done single image processing.")

if __name__ == "__main__":
    main()
