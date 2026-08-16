"""ablation_visual_compare_stage_1.py

Stage 1 of the visual ablation comparison. Same shape as ablation_advance_metrics_stage_1.py:
uncomment ONE method block below, set EPOCH, and run. It builds/reuses a shared validation
selection + manifest, then runs that method's EPOCH checkpoint over VALID_SAMPLES, exporting
the predicted point .npy (and a .png) into ONE shared folder. Stage 2 merges a (possibly
smaller) selection into a panel.

Depends only on control_v4.sample_control (load_pipeline + process_single_image).

Layout (OUTPUT_DIR = experiments/outputs/ablation_visual_results):
    OUTPUT_DIR/validation_data/{source,target}/<name>   (+ target/<stem>.npy = GT)
    OUTPUT_DIR/validation_manifest.json                 selected stems, in order
    OUTPUT_DIR/<RESULTS_DIR>/<stem>.npy   (+ <stem>.png)
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_v4.sample_control import load_pipeline, process_single_image


BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = "config/GBN/model.ckpt"
FREEZE_DENOISER = True  # informational only (load_pipeline auto-detects from the checkpoint)
GRID_SIZE = 32
ENABLE_GECCO = False
ENABLE_ADAPTIVE_GATE_INJECTION = False
EVAL_TIMESTEPS = 1000
INFER_TRUNCATION_RATIO = 1.0
RESAMPLE_JUMPS = 0
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False


# ── Uncomment ONE method block ────────────────────────────────────────────────

# Vanilla
WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_vanilla/checkpoints"
RESULTS_DIR = "vanilla"


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
# WEIGHTS_DIR = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints"
# RESULTS_DIR = "full"
# BASE_CKPT_PATH = ""  # NOTE
# FREEZE_DENOISER = False  # NOTE
# ENABLE_GECCO = True  # NOTE
# ENABLE_ADAPTIVE_GATE_INJECTION = True  # NOTE


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
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/target"
OUTPUT_DIR = "experiments/outputs/ablation_visual_results"

SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SPLAT_SIGMA_PX = 0.5

DEVICE = "cuda"
SPLIT_SEED = 42
VAL_SPLIT = 0.1

# Which validation samples to run, as indices into the seed-42 validation split order.
VALID_SAMPLES = [1, 2, 4, 7]
# Which checkpoint epoch to use (finds "*ep{EPOCH}.*" in WEIGHTS_DIR).
EPOCH = 5000

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def list_images(folder):
    p = Path(folder)
    return [str(f) for f in sorted(p.rglob("*")) if f.suffix.lower() in IMG_EXTS]


def select_validation_images(all_images, val_frac, seed):
    imgs = sorted(all_images)
    n = len(imgs)
    val_len = min(max(int(n * float(val_frac)), 0), max(n - 1, 0))
    order = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))).tolist()
    return [imgs[i] for i in order[n - val_len:]]


def pick_valid_samples(val_images, valid_samples):
    bad = [i for i in valid_samples if not (0 <= i < len(val_images))]
    if bad:
        raise IndexError(f"VALID_SAMPLES out of range for {len(val_images)} val images: {bad}")
    return [val_images[i] for i in valid_samples]


def backup_selected(selected_pairs, out_base, target_dir, val_split, split_seed):
    """Copy each selected (split_index, source) (+ target PNG and GT .npy) into validation_data/,
    and write selection.json mapping each split_index -> filename/stem (like eval_selection.json).
    """
    out_base = Path(out_base)
    vd = out_base / "validation_data"
    src_bk, tgt_bk = vd / "source", vd / "target"
    src_bk.mkdir(parents=True, exist_ok=True)
    tgt_bk.mkdir(parents=True, exist_ok=True)

    target_img_map, target_npy_map = {}, {}
    for tf in Path(target_dir).rglob("*"):
        if not tf.is_file():
            continue
        if tf.suffix.lower() in IMG_EXTS:
            target_img_map.setdefault(tf.name, tf)
        elif tf.suffix.lower() == ".npy":
            target_npy_map.setdefault(tf.stem, tf)

    selected_rows = []
    for split_index, img in tqdm(selected_pairs, desc="Backing up selected validation"):
        src = Path(img)
        if not (src_bk / src.name).exists():
            shutil.copy2(src, src_bk / src.name)
        if src.name in target_img_map and not (tgt_bk / src.name).exists():
            shutil.copy2(target_img_map[src.name], tgt_bk / src.name)
        if src.stem in target_npy_map and not (tgt_bk / f"{src.stem}.npy").exists():
            shutil.copy2(target_npy_map[src.stem], tgt_bk / f"{src.stem}.npy")  # GT points
        selected_rows.append({"split_index": int(split_index), "filename": src.name, "stem": src.stem})

    selection = {"val_split": val_split, "split_seed": split_seed,
                 "valid_samples": [r["split_index"] for r in selected_rows],
                 "selected_rows": selected_rows}
    (out_base / "selection.json").write_text(json.dumps(selection, indent=2))
    (out_base / "validation_manifest.json").write_text(   # plain list, for convenience
        json.dumps([r["filename"] for r in selected_rows], indent=2))
    return src_bk, selected_rows


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
    ap = argparse.ArgumentParser(description="Run the active method's EPOCH weights over VALID_SAMPLES.")
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--source", default=SOURCE_DIR)
    ap.add_argument("--target", default=TARGET_DIR)
    ap.add_argument("--val-split", type=float, default=VAL_SPLIT)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--valid-samples", default=json.dumps(VALID_SAMPLES),
                    help="JSON list of indices into the val split to run, e.g. '[1,2,4,7]'")
    ap.add_argument("--epoch", type=int, default=EPOCH, help="Checkpoint epoch to use (finds *ep{EPOCH}.*)")
    ap.add_argument("--overwrite", action="store_true", help="Re-run samples whose .npy already exists.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)
    valid_samples = json.loads(args.valid_samples)

    # Shared selection.json (split_index -> filename) + backups; built once, reused across methods.
    selection_path = out_base / "selection.json"
    src_bk = out_base / "validation_data" / "source"
    if selection_path.exists():
        selected_rows = json.loads(selection_path.read_text())["selected_rows"]
        print(f"Reusing selection.json: {len(selected_rows)} samples, "
              f"split_index {[r['split_index'] for r in selected_rows]}")
    else:
        all_images = list_images(args.source)
        if not all_images:
            print(f"No source images in {args.source}"); return 2
        val_images = select_validation_images(all_images, args.val_split, args.seed)
        bad = [i for i in valid_samples if not (0 <= i < len(val_images))]
        if bad:
            print(f"VALID_SAMPLES out of range for {len(val_images)} val images: {bad}"); return 2
        selected_pairs = [(i, val_images[i]) for i in valid_samples]
        src_bk, selected_rows = backup_selected(selected_pairs, out_base, args.target, args.val_split, args.seed)
        print(f"Selected + backed up {len(selected_rows)} samples -> {out_base/'validation_data'}")

    run_paths = [str(src_bk / r["filename"]) for r in selected_rows]

    ckpt = find_checkpoint_at_epoch(WEIGHTS_DIR, args.epoch)
    if ckpt is None:
        print(f"No checkpoint at epoch {args.epoch} in {WEIGHTS_DIR}"); return 2
    method_out = out_base / RESULTS_DIR
    method_out.mkdir(parents=True, exist_ok=True)
    print(f"Method '{RESULTS_DIR}' epoch {args.epoch}: {ckpt}")
    print(f"  gecco={ENABLE_GECCO} agi={ENABLE_ADAPTIVE_GATE_INJECTION} "
          f"base_ckpt={'<from-scratch>' if not BASE_CKPT_PATH else BASE_CKPT_PATH} trunc={INFER_TRUNCATION_RATIO}")

    if args.dry_run:
        print(f"DRY -> {method_out}  samples: {[Path(p).name for p in run_paths]}")
        return 0

    diffusion, control_net = load_pipeline(
        base_config_path=BASE_CONFIG_PATH, base_ckpt_path=BASE_CKPT_PATH, control_ckpt_path=ckpt,
        grid_size=GRID_SIZE, enable_gecco=ENABLE_GECCO, enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION,
        smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
        batch_coords_features=BATCH_COORDS_FEATURES, device=DEVICE,
    )
    for i, img in enumerate(run_paths, start=1):
        stem = Path(img).stem
        if (method_out / f"{stem}.npy").exists() and not args.overwrite:
            print(f"  [{i}/{len(run_paths)}] {stem}: exists, skip"); continue
        print(f"  [{i}/{len(run_paths)}] {stem}", flush=True)
        process_single_image(
            image_path=Path(img), diffusion=diffusion, control_net=control_net,
            grid_size=GRID_SIZE, truncation_ratio=INFER_TRUNCATION_RATIO, eval_timesteps=EVAL_TIMESTEPS,
            smart_init_features=SMART_INIT_FEATURES, sdf_features=SDF_FEATURES,
            resample_jumps=RESAMPLE_JUMPS, sdf_truncate_px=SDF_TRUNCATE_PX, t_start_step=-1,
            smart_init_seed=SMART_INIT_SEED, smart_init_splat_sigma_px=SMART_INIT_SPLAT_SIGMA_PX,
            enable_smart_init_splat_sigma=ENABLE_SMART_INIT_SPLAT_SIGMA,
            show_denoising_interval=50, device=DEVICE,
            export_conditions=False, export_png=True, export_npy=True,
            track_time=False, show_denoising=False,
            conditions_dir=None, png_dir=Path(method_out), npy_dir=Path(method_out),
            timestamps_dir=None, denoising_dir=None,
        )

    print("\nStage 1 done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
