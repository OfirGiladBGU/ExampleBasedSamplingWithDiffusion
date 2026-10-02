"""spectral_analysis_unconstrained_stage_0_old.py

Unconstrained spectral analysis, stage 0: create the input images.

The base GBN diffusion model takes no conditioning image, so it can only be compared with the
capacity-constrained samplers on a uniform FULL-BLACK target (rho = 1.0), where they are asked for
the same unconstrained blue-noise process. This writes REALIZATIONS copies of that image -- one per
independent realization -- plus a manifest:

    OUT_DIR/source/uniform_g000_r<k>.png
    OUT_DIR/manifest.json

Unlike spectral_analysis_stage_0.py, the image is FULLY black: no white margin (MARGIN_PX = 0), so
every conditioned method is asked for exactly the domain the base model fills, the whole unit
square. The margin was only needed for the patterned images, where WVS / GBN's per-image min-max
stretch would otherwise map the lightest grey to zero; on a constant image both samplers skip the
stretch and see a uniform density, and BNOT inverts black to a uniform density of 1.

Because these images differ from the original analysis' (which carry the margin), its WVS / BNOT /
GBN / Ours point sets do NOT apply here and stage 1 re-runs every method. CHECK_AGAINST (off by
default) compares the images with another source folder pixel by pixel, for a setup that does reuse
earlier results.

Run from the project root:
    python experiments/spectral_analysis_unconstrained_stage_0_old.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "experiments/outputs/spectral_analysis_unconstrained"

GREY_LEVEL = 0             # full black: rho = 1.0
REALIZATIONS = 25          # independent realizations, as in the original spectral analysis
RESOLUTION = 512
MARGIN_PX = 0              # no white margin: fully black (1 = the original analysis' margin)
POINT_BUDGET = 1024
METHODS = ["WVS", "BNOT", "GBN", "Base-GBN", "CN-WVS", "CN-GBN"]

# Source folder to compare against pixel by pixel (only meaningful when stage 1 copies results made
# from it, e.g. "experiments/outputs/spectral_analysis/source" with MARGIN_PX = 1). "" skips it.
CHECK_AGAINST = ""


def parse_args():
    p = argparse.ArgumentParser(description="Unconstrained spectral analysis, stage 0: full-black input images")
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--grey", type=int, default=GREY_LEVEL)
    p.add_argument("--realizations", type=int, default=REALIZATIONS)
    p.add_argument("--resolution", type=int, default=RESOLUTION)
    p.add_argument("--margin-px", type=int, default=MARGIN_PX)
    p.add_argument("--point-budget", type=int, default=POINT_BUDGET)
    p.add_argument("--check-against", default=CHECK_AGAINST,
                   help="Folder of original sources to verify against ('' to skip)")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.grey < 255):
        print(f"ERROR: grey level must be in [0, 255); pure white carries no mass. Got {args.grey}")
        return 2
    out = Path(args.output)
    src_dir = out / "source"
    src_dir.mkdir(parents=True, exist_ok=True)

    img = np.full((args.resolution, args.resolution), 255, dtype=np.uint8)
    m = args.margin_px
    img[m:args.resolution - m, m:args.resolution - m] = args.grey      # m = 0: the whole image

    images = []
    for r in range(args.realizations):
        stem = f"uniform_g{args.grey:03d}_r{r:02d}"
        if not cv2.imwrite(str(src_dir / f"{stem}.png"), img):
            print(f"ERROR: failed to write {stem}.png")
            return 3
        images.append({"stem": stem, "kind": "uniform", "grey": args.grey, "rho": 1.0 - args.grey / 255.0})

    manifest = {
        "resolution": args.resolution,
        "margin_px": int(args.margin_px),     # the analysis crops this many pixels off every edge
        "point_budget": args.point_budget,
        "realizations": args.realizations,
        "tests_1": True,
        "tests_2": False,
        "grey_levels": [args.grey],
        "patterns": [],
        "regions": {},
        "methods": METHODS,
        "expected_target_dirs": [f"target_{x}_{args.point_budget}" for x in METHODS],
        "images": images,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(images)} full-black images ({args.resolution}px, {m}px white margin, "
          f"grey {args.grey}, rho {1 - args.grey / 255:.2f}) to {src_dir}")
    print(f"Manifest: {out / 'manifest.json'}")

    if args.check_against:
        ref = Path(args.check_against)
        same = missing = 0
        for im in images:
            p = ref / f"{im['stem']}.png"
            if not p.exists():
                missing += 1
                continue
            same += int(np.array_equal(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), img))
        diff = len(images) - same - missing
        print(f"Check against {ref}: {same} identical, {diff} different, {missing} missing")
        if diff or missing:
            print("  WARNING: results made from that folder do NOT apply to these images; "
                  "re-run the samplers (stage 1) instead of copying.")
    print("\nNext: spectral_analysis_unconstrained_stage_1_old.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
