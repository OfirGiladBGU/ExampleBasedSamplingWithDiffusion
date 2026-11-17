#!/usr/bin/env python3
"""Generalization diagnostics, stage 0: stage 200 training and 200 held-out Icons-50 images.

Why the learned sampler loses local regularity (CVT, capacity, rho_s) against its teacher:
  A. train vs test  -- does it reach the teacher on images it was trained on? (data-limited or not)
  B. from the exemplar -- start from the teacher's own stipple noised to step K and denoise: at
     which noise level is the regularity lost? (low-noise steps vs layout decisions)
  C. loss per noise level on train vs held-out images (diag_generalization_timestep_loss.py)

This stage writes, under OUTPUT_DIR (Ours-GBN, Icons-50, 1024 points):
    {train,test}_source/<stem>.png     symlinks to the condition images
    {train,test}_init/<stem>.npy       the teacher's exemplar as the OT offset grid (2, 32, 32), i.e.
                                       the exact training target (processed_offsets)
    CN-GBN-{train,test}_1024/GBN-exemplar/<stem>.npy   the same exemplar as (N, 2) points, so that
                                       few_step_sampling_stage_2.py scores it like a sampler

The held-out images are the trainer's validation split (z_validation_data manifest); the training
images are a seeded random draw from the other 9000.

    python experiments/diag_generalization_stage_0.py
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.Transforms import to_pointset_optimal_transport  # noqa: E402

DATA = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN"
TEST_MANIFEST = "experiments/outputs/z_validation_data/Icons-50_1024/validation_manifest.json"
TEST_POINTS = "experiments/outputs/rescore_current_metrics-Icons-50_1024/GBN"   # convention check only
OUTPUT_DIR = "experiments/outputs/diag_generalization"
COUNT = 200
SEED = 0
POINTS = 1024


def offsets_to_points(offsets):
    pts = to_pointset_optimal_transport(offsets)
    return pts.reshape(pts.shape[0], -1).T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--count", type=int, default=COUNT)
    args = ap.parse_args()
    out = Path(args.output)

    src = {}
    for root, _, files in os.walk(Path(DATA) / "source"):
        for f in files:
            if f.lower().endswith(".png"):
                src[Path(f).stem] = Path(root) / f
    off = {p.stem: p for p in (Path(DATA) / "processed_offsets").rglob("*.npy")}
    test_all = sorted(Path(n).stem for n in json.loads(Path(TEST_MANIFEST).read_text()))
    train_all = sorted(set(src) & set(off) - set(test_all))
    rng = np.random.RandomState(SEED)
    picks = {"train": sorted(rng.choice(train_all, args.count, replace=False)),
             "test": sorted(rng.choice(test_all, args.count, replace=False))}
    print(f"{len(train_all)} training images, {len(test_all)} held-out images; {args.count} of each staged")

    worst = 0.0
    for split, stems in picks.items():
        (out / f"{split}_source").mkdir(parents=True, exist_ok=True)
        (out / f"{split}_init").mkdir(parents=True, exist_ok=True)
        ex_dir = out / f"CN-GBN-{split}_{POINTS}" / "GBN-exemplar"
        ex_dir.mkdir(parents=True, exist_ok=True)
        for s in stems:
            link = out / f"{split}_source" / f"{s}.png"
            if not link.exists():
                link.symlink_to(src[s])
            offsets = np.load(off[s]).astype(np.float32)
            np.save(out / f"{split}_init" / f"{s}.npy", offsets)
            pts = offsets_to_points(offsets)
            np.save(ex_dir / f"{s}.npy", pts)
            if split == "test":  # same point set as the scored GBN targets, up to order?
                ref = np.load(Path(TEST_POINTS) / f"{s}.npy")
                a = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
                b = ref[np.lexsort((ref[:, 1], ref[:, 0]))]
                worst = max(worst, float(np.abs(a - b).max()))
    (out / "picks.json").write_text(json.dumps(picks, indent=1))
    print(f"convention check: max |exemplar-from-offsets - scored GBN points| over the test picks = {worst:.2e} "
          f"(unit square; ~0 means the offsets are the scored exemplars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
