"""Stage 0: build the uniform-density source images for the spectral analysis.

Why uniform images
------------------
A radially-averaged power spectrum assumes a STATIONARY point process. A capacity-constrained
stipple of an icon is non-stationary by construction -- the silhouette contributes exactly the
low-frequency energy whose absence is the blue-noise signature -- so a periodogram of those
point sets largely measures the conditioning image rather than the sampling quality. On a
constant target density the process IS stationary, which is the standard setting in which
blue-noise spectra are reported, and the one place our sampler can be compared to GBN on its
own ground.

Why several grey levels
-----------------------
Capacity constraints distribute points by RELATIVE density, so a black and a mid-grey uniform
image describe the same target distribution: the absolute level cancels under normalisation.
The grey level is still a different network input (it enters the conditioning image and the
density map), so sweeping it tests INVARIANCE -- that the model responds to relative density
rather than absolute intensity -- not four different targets. Pure white is excluded: it
carries no mass and there is nothing to distribute.

Why repeated copies
-------------------
A single point set's periodogram is essentially noise; the smooth spectra in the literature are
averaged over many independent realizations. The rest of this pipeline is one-point-set-per-
source-image, so we emit N identical copies per condition under distinct filenames. Each
sampler then produces N independent point sets with no changes to any sampling script.

Output
------
    experiments/outputs/spectral_analysis/source/uniform_g<level>_r<idx>.png
    experiments/outputs/spectral_analysis/manifest.json

Then run each sampler over that folder, writing to sibling folders:
    experiments/outputs/spectral_analysis/target_WVS_1024/<stem>.npy   (+ .png)
    experiments/outputs/spectral_analysis/target_BNOT_1024/...
    experiments/outputs/spectral_analysis/target_GBN_1024/...
    experiments/outputs/spectral_analysis/target_CN_1024/...
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = r"experiments/outputs/spectral_analysis"

# Grey levels to emit, as 8-bit pixel values. rho = 1 - value/255, so 0 is maximum ink.
# 255 (pure white) is deliberately absent: zero mass, nothing for the model to place.
GREY_LEVELS = [0, 64, 128, 192]

# Independent realizations per grey level. 25 is enough for a clean mean spectrum at 1024
# points; raise it if the radial profile still looks ragged.
REALIZATIONS = 25

# Source resolution, matching experiments/outputs/quantitative_advance_metrics/source.
RESOLUTION = 512

# Point budget these sources are intended to be sampled at. Only used to name the expected
# target folders in the manifest -- stage 0 does not sample anything itself.
POINT_BUDGET = 1024

METHODS = ["WVS", "BNOT", "GBN", "CN"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Create uniform-density source images for spectral analysis"
    )
    p.add_argument("--output", default=OUT_DIR,
                   help=f"Base output folder (default: {OUT_DIR})")
    p.add_argument("--levels", default=",".join(str(g) for g in GREY_LEVELS),
                   help=f"Comma-separated 8-bit grey levels (default: {GREY_LEVELS}). "
                        f"255 is rejected: it carries no mass.")
    p.add_argument("--realizations", type=int, default=REALIZATIONS,
                   help=f"Copies per grey level (default: {REALIZATIONS}). Each becomes one "
                        f"independent point set per sampler.")
    p.add_argument("--resolution", type=int, default=RESOLUTION,
                   help=f"Source image size in pixels (default: {RESOLUTION})")
    p.add_argument("--point-budget", type=int, default=POINT_BUDGET,
                   help=f"Point budget recorded in the manifest (default: {POINT_BUDGET})")
    p.add_argument("--clean", action="store_true",
                   help="Delete existing source images before writing")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    bad = [g for g in levels if not (0 <= g < 255)]
    if bad:
        print(f"ERROR: grey levels must be in [0, 255); pure white carries no mass. Got: {bad}")
        return 2

    out_base = Path(args.output)
    src_dir = out_base / "source"
    n_total = len(levels) * args.realizations

    print(f"Output base : {out_base}")
    print(f"Source dir  : {src_dir}")
    print(f"Grey levels : {levels}   (ink density rho = " +
          ", ".join(f"{1.0 - g / 255.0:.2f}" for g in levels) + ")")
    print(f"Realizations: {args.realizations} per level")
    print(f"Resolution  : {args.resolution}x{args.resolution}")
    print(f"Total images: {n_total}")

    if args.dry_run:
        print("\nDRY RUN: would write")
        for g in levels:
            print(f"  {src_dir / f'uniform_g{g:03d}_r00.png'} ... r{args.realizations - 1:02d}.png")
        return 0

    src_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = 0
        for f in src_dir.glob("uniform_g*_r*.png"):
            f.unlink()
            removed += 1
        if removed:
            print(f"Removed {removed} existing source image(s)")

    written = []
    for g in levels:
        img = np.full((args.resolution, args.resolution), g, dtype=np.uint8)
        for r in range(args.realizations):
            stem = f"uniform_g{g:03d}_r{r:02d}"
            path = src_dir / f"{stem}.png"
            if not cv2.imwrite(str(path), img):
                print(f"ERROR: failed to write {path}")
                return 3
            written.append({"stem": stem, "grey": g, "rho": 1.0 - g / 255.0, "realization": r})
        print(f"  grey {g:3d} (rho {1.0 - g / 255.0:.2f}): {args.realizations} images")

    # The manifest lets stage 1/2 group realizations by condition without parsing filenames.
    manifest = {
        "resolution": args.resolution,
        "point_budget": args.point_budget,
        "grey_levels": levels,
        "realizations": args.realizations,
        "methods": METHODS,
        "expected_target_dirs": [f"target_{m}_{args.point_budget}" for m in METHODS],
        "images": written,
    }
    (out_base / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {len(written)} source images")
    print(f"Manifest: {out_base / 'manifest.json'}")
    print("\nNext: run each sampler over the source folder, writing .npy point sets to")
    for m in METHODS:
        print(f"    {out_base / f'target_{m}_{args.point_budget}'}")
    print("then run spectral_analysis_stage_1.py per target folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
