"""teaser_demo_stage_1.py -- Part 1: run the truncated denoising on the demo image.

Our method uses TRUNCATED (SDEdit) denoising: it starts from the rejection prior at
t_start = eval_timesteps * truncation_ratio (= 500 for ratio 0.5) and denoises to the
end. This script runs that on a single condition image and snapshots the point state
at t = 500 (rejection prior), t = 750 (mid) and t = 1000 (final), saving each as:

    OUTPUT_DIR/steps/<t_label>.npy   (N,2) points -- used for the vectorized figure
    OUTPUT_DIR/steps/<t_label>.png   scatter preview

The condition image (source/Demo.png) is left in place for stage 2 to load.

Snapshot mapping (with truncation 0.5): figure-t = t_start + elapsed_denoise_steps, so
    t=500  -> elapsed 0   (denoising step _0000)
    t=750  -> elapsed 250 (denoising step _0250)
    t=1000 -> the final result
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import cv2  # only needed for the optional source/original preprocessing
except Exception:
    cv2 = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Transforms import to_pointset_optimal_transport
from control_v4.sample_control import load_pipeline, process_single_image, load_condition
from control_v4.smart_init import generate_smart_init_points_from_density, smart_init_points_to_offsets

# ── Config: OUR method = full weights + SDEdit truncation 0.5 ──────────────────
BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = ""   # full weights are from-scratch: load_pipeline restores the denoiser from the control ckpt
CONTROL_CKPT = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt"

INPUT_IMAGE = "experiments/outputs/teaser_demo_results/source/Demo.png"
OUTPUT_DIR = "experiments/outputs/teaser_demo_results"

# Optional preprocessing BEFORE inference: populate the source folder (parent of --image)
# from ORIGINAL_DIR. "" defaults to the sibling "original" folder of the source; missing/empty
# -> no-op. Grayscale-fill when source is empty, or GBN pipeline + overwrite when APPLY_PREPROCESS.
ORIGINAL_DIR = ""               # raw images folder; "" -> sibling "original" of the source dir
APPLY_PREPROCESS = False        # GBN preprocess original -> source (overwrites)
DISABLE_BG_SUPPRESSION = False  # skip the bg-suppression step of the preprocess
INVERT_IMAGE = False            # invert pixels after (pre)processing
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# GRID_SIZE = 24
# GRID_SIZE = 32
GRID_SIZE = 48
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
INFER_TRUNCATION_RATIO = 0.5
RESAMPLE_JUMPS = 0
EVAL_TIMESTEPS = 1000
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SPLAT_SIGMA_PX = 0.5
ENABLE_SMART_INIT_SPLAT_SIGMA = False
DEVICE = "cuda"

STEP_INTERVAL = 250   # saves elapsed 0 (t=500) and 250 (t=750); t=1000 comes from the final export
# (figure-t label, denoising 'elapsed' step or "final")
# SNAPSHOTS = [("t500", "prior"), ("t750", 250), ("t1000", "final")]
SNAPSHOTS = [("t500", 0), ("t750", 250), ("t1000", "final")]

DOT_SIZE = 4.0


# --- GBN preprocessing (ported from GaussianBlueNoise/scripts/image_preprocess.py) ---
def _percentile_stretch(gray, p_low=1.0, p_high=99.0):
    lo, hi = np.percentile(gray, [p_low, p_high])
    if hi - lo < 1e-6:
        return gray.copy()
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_clahe(gray, clip_limit=3.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def _unsharp_mask(gray, sigma=1.4, amount=1.5):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray.astype(np.float32), 1.0 + amount, blur.astype(np.float32), -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _suppress_background(gray):
    g = gray.copy()
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    h, w = bin_img.shape
    flood = bin_img.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)
    bg = flood == 128
    out = g.astype(np.float32)
    out[bg] = 0.30 * out[bg] + 0.70 * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_image(gray, do_bg_suppression=True):
    """GBN enhancement: stretch -> CLAHE -> unsharp -> stretch -> optional bg-suppress."""
    x = _percentile_stretch(gray, 1.0, 99.0)
    x = _apply_clahe(x, 3.0, 8)
    x = _unsharp_mask(x, 1.4, 1.5)
    x = _percentile_stretch(x, 0.8, 99.2)
    if do_bg_suppression:
        x = _suppress_background(x)
    return x


def _load_gray(image_path):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS) if folder.is_dir() else []


def ensure_source_from_original(original_dir, source_dir, apply_preprocess=False,
                                disable_bg_suppression=False, invert_image=False):
    """Populate the grayscale `source` folder from the `original` folder, BEFORE inference.

    - apply_preprocess=True -> GBN preprocess each original image and OVERWRITE the source image.
    - else, only when `source` is missing/empty -> plain grayscale conversion.
    - otherwise (source already populated) -> do nothing.
    """
    original_dir, source_dir = Path(original_dir), Path(source_dir)
    if not apply_preprocess and _list_images(source_dir):
        return 0
    originals = _list_images(original_dir)
    if not originals:
        if apply_preprocess:
            print(f"[preprocess] no images in {original_dir}; nothing to do")
        return 0
    if cv2 is None:
        raise RuntimeError("cv2 is required for source/original preprocessing (pip install opencv-python).")
    source_dir.mkdir(parents=True, exist_ok=True)
    mode = "GBN preprocess -> overwrite" if apply_preprocess else "grayscale (source was empty)"
    print(f"[source] {mode}: {len(originals)} image(s) {original_dir} -> {source_dir}")
    for ip in originals:
        gray = _load_gray(ip)
        if apply_preprocess:
            gray = preprocess_image(gray, do_bg_suppression=not disable_bg_suppression)
        if invert_image:
            gray = 255 - gray
        cv2.imwrite(str(source_dir / f"{ip.stem}.png"), gray)
    return len(originals)


def offsets_to_points(off_2gg):
    """(2,G,G) offset field -> (N,2) point set, matching sample_control's final export."""
    pts = to_pointset_optimal_transport(np.asarray(off_2gg, dtype=np.float64))
    return pts.reshape(pts.shape[0], -1).T


def save_scatter(pts, path, dot_size):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=dot_size, c="black", linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="Run truncated denoising on the demo image, export t=500/750/1000.")
    ap.add_argument("--image", default=INPUT_IMAGE)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--control-ckpt", default=CONTROL_CKPT)
    ap.add_argument("--dot-size", type=float, default=DOT_SIZE)
    ap.add_argument("--original-dir", default=ORIGINAL_DIR,
                    help="Raw images folder to (pre)process into the source dir before inference "
                         "(\"\" -> sibling 'original' of the --image folder).")
    ap.add_argument("--preprocess", action=argparse.BooleanOptionalAction, default=APPLY_PREPROCESS,
                    help="Run the GBN preprocess on original/ and OVERWRITE source/ (else only fills an empty source/).")
    ap.add_argument("--disable-bg-suppression", action="store_true", default=DISABLE_BG_SUPPRESSION,
                    help="Skip the background-suppression step of the preprocess.")
    ap.add_argument("--invert-image", action=argparse.BooleanOptionalAction, default=INVERT_IMAGE,
                    help="Invert pixels after (pre)processing.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    stem = Path(args.image).stem
    out = Path(args.output)
    final_npy_dir = out / "_final_npy"
    final_png_dir = out / "_final_png"
    denoise_dir = out / "_denoising"
    steps_dir = out / "steps"

    t_start = int(EVAL_TIMESTEPS * INFER_TRUNCATION_RATIO)
    print(f"Demo '{stem}': full weights, trunc={INFER_TRUNCATION_RATIO} -> t_start={t_start}")
    print(f"  snapshots {[ (lbl, w) for lbl, w in SNAPSHOTS ]}  interval={STEP_INTERVAL}")
    if args.dry_run:
        print(f"DRY -> {steps_dir}")
        return 0

    for d in (final_npy_dir, final_png_dir, denoise_dir, steps_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Preprocess BEFORE inference: fill/overwrite the grayscale source folder (parent of --image)
    # from the original folder, so process_single_image reads the (pre)processed condition.
    source_dir = Path(args.image).parent
    original_dir = Path(args.original_dir) if args.original_dir else source_dir.parent / "original"
    ensure_source_from_original(original_dir, source_dir, apply_preprocess=args.preprocess,
                                disable_bg_suppression=args.disable_bg_suppression,
                                invert_image=args.invert_image)

    diffusion, control_net = load_pipeline(
        base_config_path=BASE_CONFIG_PATH, base_ckpt_path=BASE_CKPT_PATH, control_ckpt_path=args.control_ckpt,
        grid_size=GRID_SIZE, enable_gecco=ENABLE_GECCO,
        enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        batch_coords_features=BATCH_COORDS_FEATURES, device=DEVICE,
    )

    process_single_image(
        image_path=Path(args.image), diffusion=diffusion, control_net=control_net,
        grid_size=GRID_SIZE, truncation_ratio=INFER_TRUNCATION_RATIO, eval_timesteps=EVAL_TIMESTEPS,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        resample_jumps=RESAMPLE_JUMPS, sdf_truncate_px=SDF_TRUNCATE_PX, t_start_step=-1,
        smart_init_seed=SMART_INIT_SEED, smart_init_splat_sigma_px=SMART_INIT_SPLAT_SIGMA_PX,
        enable_smart_init_splat_sigma=ENABLE_SMART_INIT_SPLAT_SIGMA,
        show_denoising_interval=STEP_INTERVAL, device=DEVICE,
        export_conditions=False, export_png=True, export_npy=True,
        track_time=False, show_denoising=True,   # <- intermediate frames
        conditions_dir=None, png_dir=final_png_dir, npy_dir=final_npy_dir,
        timestamps_dir=None, denoising_dir=denoise_dir,
    )

    # Clean rejection-sampling prior (smart init BEFORE noise) -- exactly what the model is
    # seeded with at t_start, reproduced via the same load_condition + seed as process_single_image.
    image_01 = load_condition(Path(args.image), GRID_SIZE, DEVICE,
                              sdf_features=SDF_FEATURES, sdf_truncate_px=SDF_TRUNCATE_PX)[0]
    smart_points = generate_smart_init_points_from_density(
        image_01, n_points=GRID_SIZE * GRID_SIZE, seed=SMART_INIT_SEED)
    prior_offsets = smart_init_points_to_offsets(smart_points)   # (2,G,G), un-noised

    # Assemble clean point snapshots (all as (N,2) points).
    for label, which in SNAPSHOTS:
        if which == "prior":
            pts = offsets_to_points(prior_offsets)   # clean rejection prior, no noise added
        elif which == "final":
            pts = np.load(final_npy_dir / f"{stem}.npy").astype(np.float64)   # already (N,2)
        else:
            step_npy = denoise_dir / "npy" / f"{stem}_step_{int(which):04d}.npy"
            if not step_npy.exists():
                print(f"  WARN: missing {step_npy} (check STEP_INTERVAL / truncation)"); continue
            off = np.load(step_npy).astype(np.float64)      # (n_samples, 2, G, G)
            pts = offsets_to_points(off[0])
        np.save(steps_dir / f"{label}.npy", pts)
        save_scatter(pts, steps_dir / f"{label}.png", args.dot_size)
        print(f"  saved {label}: {pts.shape[0]} pts -> {steps_dir / f'{label}.npy'}")

    print(f"\nStage 1 done -> {steps_dir}  (condition: {Path(args.image)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
