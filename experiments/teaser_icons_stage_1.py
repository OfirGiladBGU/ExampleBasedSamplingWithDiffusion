"""teaser_icons_stage_1.py

Stage 1 of the teaser figure. Modeled on control_v4/eval_dataset.py (which already
produces Target / Result / OT-Map artifacts and owns the GT GBN offsets) and on
experiments/ablation_visual_compare_stage_1.py (which owns the shared selection.json).

Runs the SDEdit inference -- FULL weights, truncation 0.5 -- over VALID_SAMPLES and
exports, per sample, into OUTPUT_DIR:

    target/<stem>.png              Target: the condition image the model saw
    result/<stem>.npy  (+ .png)    Result: predicted points (N,2) in [0,1], SDEdit
    ot_map/<stem>.npy  (+ .png)    OT Map: GT GBN offset field (2,G,G) + quiver preview
    selection.json                 split_index -> {filename, stem}, in selection order

Stage 2 (teaser_icons_stage_2.py) reads these and lays them out as the teaser panel.
VALID_SAMPLES may be flat ([0,1,2,3]) or nested ([[0,1,2,3],[4,5,6,7]]); stage 1 only
cares about the *set* of indices, so it flattens either form. The grouping/order is
used by stage 2 to build the big columns.

VALID_SAMPLES index into the manifest staged by teaser_icons_stage_0.py, which is stored
in selection order -- so index i is the same image the old seed-42 split gave. The split is
no longer re-derived here. The GBN offsets still come from the real source/offsets dirs
(they cannot be staged), so the dataset is still built; only the choice of which files form
the validation subset comes from the manifest.
"""

import argparse
import ast
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Transforms import to_pointset_optimal_transport
from control_v4.DynamicStippleDataset import DynamicStippleDataset
from control_v4.train_control import dynamic_collate, ensure_offsets_dir, sample_eval_batch
# Reuse eval_dataset's model loader + tiny render helpers (Target / Result / OT-Map previews).
from control_v4.eval_dataset import (
    _load_models,
    _save_condition_image,
    _save_point_scatter,
    _save_offset_quiver,
)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"

# The teaser uses the FULL ablation weights with the SDEdit inference config.
CONTROL_CKPT = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt"
SOURCE_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/source"
TARGET_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/target"
OFFSETS_DIR = ""  # empty -> ensure_offsets_dir builds/loads the GBN offsets from source+target
OUTPUT_DIR = "experiments/outputs/teaser_icons_results"
MANIFEST_NAME = "validation_manifest.json"

# Model components (FULL config) + SDEdit inference.
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
INFER_TRUNCATION_RATIO = 0.5      # SDEdit start level (the teaser trunc)
RESAMPLE_JUMPS = 0                # current approach: SDEdit truncation only (no RePaint resampling)
EVAL_TIMESTEPS = 1000
GRID_SIZE = 32
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
SMART_INIT_SEED = 42
SDF_TRUNCATE_PX = 8.0

DEVICE = "cuda"
OVERWRITE = True   # re-run samples even if their result .npy already exists

# Which validation samples to run: a single flat list of indices into the seed-42
# val split order. (A nested list is also tolerated -- it is flattened here; the
# grouping into big columns is stage 2's concern.)
VALID_SAMPLES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def flatten_samples(valid_samples):
    """Flatten flat/nested VALID_SAMPLES to a unique, order-preserving index list."""
    if valid_samples and isinstance(valid_samples[0], (list, tuple)):
        flat = [i for group in valid_samples for i in group]
    else:
        flat = list(valid_samples)
    seen, out = set(), []
    for i in flat:
        i = int(i)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def load_manifest(out_base):
    """Validation filenames, in selection order, staged by teaser_icons_stage_0.py."""
    mpath = os.path.join(out_base, MANIFEST_NAME)
    if not os.path.exists(mpath):
        raise FileNotFoundError(
            f"Missing {mpath}  -- run experiments/teaser_icons_stage_0.py first.")
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_val_dataset(args):
    """Validation dataset, with membership+order taken from the staged manifest.

    The base dataset is still constructed because it is what resolves each image to its
    path relative to the source root AND intersects with the offsets that actually exist;
    the manifest only decides which of those files are the validation set.
    """
    offsets_dir = ensure_offsets_dir(args.source, args.target, args.offsets, args.grid_size)
    base_dataset = DynamicStippleDataset(
        args.source, offsets_dir, grid_size=args.grid_size, sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=None, smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features, sdf_features=args.sdf_features, preload_ram=False,
    )
    if len(base_dataset) == 0:
        raise RuntimeError("Dataset is empty after matching source images and offsets")

    # The manifest holds flat basenames; the dataset keys on source-relative paths.
    by_base = {os.path.basename(f): f for f in base_dataset.filenames}
    manifest = load_manifest(args.out)
    val_filenames, missing = [], []
    for name in manifest:
        rel = by_base.get(os.path.basename(name))
        (val_filenames if rel else missing).append(rel if rel else name)
    if missing:
        print(f"  [warn] {len(missing)} manifest entries have no source+offsets pair, "
              f"e.g. {missing[:3]}")
    if not val_filenames:
        raise RuntimeError("No manifest entry matched the dataset (source/offsets mismatch?)")
    print(f"  validation set from manifest: {len(val_filenames)} images")

    val_dataset = DynamicStippleDataset(
        args.source, offsets_dir, grid_size=args.grid_size, sdf_truncate_px=args.sdf_truncate_px,
        cache_data_dir=None, smart_init_seed=args.smart_init_seed,
        smart_init_features=args.smart_init_features, sdf_features=args.sdf_features,
        filenames=val_filenames, preload_ram=False,
    )
    return val_dataset


def parse_args():
    ap = argparse.ArgumentParser(description="Teaser stage 1: SDEdit inference + GT OT-map export.")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--control-ckpt", default=CONTROL_CKPT)
    ap.add_argument("--source", default=SOURCE_DIR)
    ap.add_argument("--target", default=TARGET_DIR)
    ap.add_argument("--offsets", default=OFFSETS_DIR)
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--grid-size", type=int, default=GRID_SIZE)
    ap.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    ap.add_argument("--sdf-truncate-px", type=float, default=SDF_TRUNCATE_PX)
    ap.add_argument("--eval-timesteps", type=int, default=EVAL_TIMESTEPS)
    ap.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    ap.add_argument("--infer-truncation-ratio", type=float, default=INFER_TRUNCATION_RATIO)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--valid-samples", default=json.dumps(VALID_SAMPLES),
                    help="JSON list (flat) or list-of-lists (nested big columns).")
    ap.add_argument("--enable-gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    ap.add_argument("--enable-adaptive-gate-injection", action=argparse.BooleanOptionalAction,
                    default=ENABLE_ADAPTIVE_GATE_INJECTION)
    ap.add_argument("--smart-init-features", action=argparse.BooleanOptionalAction, default=SMART_INIT_FEATURES)
    ap.add_argument("--sdf-features", action=argparse.BooleanOptionalAction, default=SDF_FEATURES)
    ap.add_argument("--batch-coords-features", action=argparse.BooleanOptionalAction, default=BATCH_COORDS_FEATURES)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE,
                    help="Re-run samples whose result .npy exists (default: True; --no-overwrite skips existing).")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        valid_samples = flatten_samples(json.loads(args.valid_samples))
    except (ValueError, TypeError):
        valid_samples = flatten_samples(ast.literal_eval(args.valid_samples))
    if not valid_samples:
        print("VALID_SAMPLES is empty"); return 2

    out = args.out
    if not out:
        print("--out is empty"); return 2
    # Create the output tree up front. os.makedirs(..., exist_ok=True) also creates any
    # missing PARENTS, so a missing experiments/outputs/teaser_icons_results -- or even a
    # missing experiments/outputs -- is created here rather than crashing on first write.
    os.makedirs(out, exist_ok=True)
    for sub in ("target", "result", "ot_map"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    print(f"Output dir: {os.path.abspath(out)}")

    val_dataset = build_val_dataset(args)
    bad = [i for i in valid_samples if not (0 <= i < len(val_dataset))]
    if bad:
        print(f"VALID_SAMPLES out of range for {len(val_dataset)} val images: {bad}"); return 2

    selected, meta_rows = [], []
    for idx in valid_samples:
        selected.append(val_dataset[idx])
        filename = os.path.basename(str(val_dataset.filenames[idx]))
        stem = os.path.splitext(filename)[0]
        meta_rows.append({"split_index": int(idx), "filename": filename, "stem": stem})

    print(f"Teaser stage 1: {len(selected)} samples, split_index {valid_samples}")
    print(f"  control_ckpt={args.control_ckpt}")
    print(f"  gecco={args.enable_gecco} agi={args.enable_adaptive_gate_injection} "
          f"trunc={args.infer_truncation_ratio} resample={args.resample_jumps}")

    if args.dry_run:
        print(f"DRY -> {out}  stems: {[r['stem'] for r in meta_rows]}")
        return 0

    device = torch.device(args.device)
    batch = dynamic_collate(selected)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    diffusion, denoiser, control_net = _load_models(args, device)
    pred_raw = sample_eval_batch(
        diffusion, denoiser, control_net, batch, device,
        n_samples=batch["high_res"].shape[0], eval_timesteps=args.eval_timesteps,
        resample_jumps=args.resample_jumps, show_tqdm=True, tqdm_desc="teaser sampling",
        truncation_ratio=args.infer_truncation_ratio,
    )

    cond = batch["high_res"].detach().cpu().numpy()      # (B,1,H,W)
    gt_offsets = batch["offsets"].detach().cpu().numpy()  # (B,2,G,G)  GT GBN offsets
    pred_offsets = pred_raw.detach().cpu().numpy()        # (B,2,G,G)  SDEdit prediction

    for i, row in enumerate(meta_rows):
        stem = row["stem"]
        if os.path.exists(os.path.join(out, "result", f"{stem}.npy")) and not args.overwrite:
            print(f"  [{i+1}/{len(meta_rows)}] {stem}: exists, skip"); continue
        print(f"  [{i+1}/{len(meta_rows)}] {stem}", flush=True)

        # Target: the condition image the model conditioned on.
        _save_condition_image(os.path.join(out, "target", f"{stem}.png"), cond[i, 0])

        # Result: SDEdit predicted points (N,2) in [0,1], canonical x-then-y.
        pts = to_pointset_optimal_transport(pred_offsets[i]).reshape(2, -1).T
        np.save(os.path.join(out, "result", f"{stem}.npy"), np.asarray(pts, dtype=np.float64))
        _save_point_scatter(os.path.join(out, "result", f"{stem}.png"), pred_offsets[i])

        # OT Map: the GT GBN offset field (2,G,G) + quiver preview.
        np.save(os.path.join(out, "ot_map", f"{stem}.npy"), np.asarray(gt_offsets[i], dtype=np.float64))
        _save_offset_quiver(os.path.join(out, "ot_map", f"{stem}.png"), gt_offsets[i])

    selection = {
        "source": args.source, "grid_size": args.grid_size,
        "selection_from": MANIFEST_NAME,
        "valid_samples": valid_samples, "control_ckpt": args.control_ckpt,
        "infer_truncation_ratio": args.infer_truncation_ratio,
        "selected_rows": meta_rows,
    }
    with open(os.path.join(out, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    print(f"\nStage 1 done -> {out}")
    print(f"Saved selection.json ({len(meta_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
