"""ablation_visual_compare_stage_1.py

Stage 1 of the visual ablation comparison. Same shape as ablation_advance_metrics_stage_1.py:
uncomment ONE method block below, set EPOCH, and run. It runs that method's EPOCH checkpoint
over VALID_SAMPLES, exporting the predicted point .npy (and a .png) into ONE shared folder.
Stage 2 merges a (possibly smaller) selection into a panel.

VALID_SAMPLES are indices into the manifest staged by ablation_visual_compare_stage_0.py.
The manifest is in selection order, so index i is the same image the old seed-42 split
resolved to -- no split is re-derived here, and no selection.json / manifest is written:
the staged manifest plus VALID_SAMPLES already determines the selection completely.

Depends only on control_v4.sample_control (load_pipeline + process_single_image).

Layout (OUTPUT_DIR = experiments/outputs/ablation_visual_results):
    OUTPUT_DIR/resources/source/<name>                  (staged by stage 0)
    OUTPUT_DIR/resources/validation_manifest.json       (staged by stage 0)
    OUTPUT_DIR/<RESULTS_DIR>/<stem>.npy   (+ <stem>.png)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

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
OUTPUT_DIR = "experiments/outputs/ablation_visual_results"
MANIFEST_NAME = "validation_manifest.json"

SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0
SMART_INIT_SPLAT_SIGMA_PX = 0.5

DEVICE = "cuda"

# Which validation samples to run, as indices into the staged manifest (which is in the
# seed-42 validation split order, so these are the same split_index values as before).
VALID_SAMPLES = [1, 2, 4, 7]
# Which checkpoint epoch to use (finds "*ep{EPOCH}.*" in WEIGHTS_DIR).
EPOCH = 5000


def select_from_manifest(out_base, valid_samples):
    """Resolve VALID_SAMPLES indices against the manifest staged by stage 0.

        OUTPUT_DIR/resources/validation_manifest.json
        OUTPUT_DIR/resources/source/

    Returns [(split_index, image_path)]. The manifest is in selection order, so index i
    names exactly the image the old seed-42 split gave for split_index i.
    """
    res = Path(out_base) / "resources"
    manifest_path = res / MANIFEST_NAME
    source_dir = res / "source"
    missing = [str(p) for p in (manifest_path, source_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing staged resources: " + ", ".join(missing)
            + "  -- run experiments/ablation_visual_compare_stage_0.py first.")

    names = json.loads(manifest_path.read_text())
    bad = [i for i in valid_samples if not (0 <= i < len(names))]
    if bad:
        raise IndexError(f"VALID_SAMPLES out of range for {len(names)} manifest entries: {bad}")

    out = []
    for i in valid_samples:
        p = source_dir / names[i]
        if not p.exists():
            raise FileNotFoundError(f"manifest[{i}] = {names[i]} not found under {source_dir}")
        out.append((int(i), str(p)))
    return out


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
    ap.add_argument("--valid-samples", default=json.dumps(VALID_SAMPLES),
                    help="JSON list of manifest indices to run, e.g. '[1,2,4,7]'")
    ap.add_argument("--epoch", type=int, default=EPOCH, help="Checkpoint epoch to use (finds *ep{EPOCH}.*)")
    ap.add_argument("--overwrite", action="store_true", help="Re-run samples whose .npy already exists.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)
    valid_samples = json.loads(args.valid_samples)

    selected = select_from_manifest(out_base, valid_samples)
    run_paths = [p for _, p in selected]
    print(f"Selected {len(selected)} samples from the manifest: "
          + ", ".join(f"[{i}] {Path(p).name}" for i, p in selected))

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
