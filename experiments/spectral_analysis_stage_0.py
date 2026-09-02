"""Stage 0: build the source images for the spectral analysis.

Two independent experiments, selected by the flags below.

SPECTRAL_TESTS_1 -- uniform images, analysed with the power spectrum
--------------------------------------------------------------------
A radially averaged periodogram assumes a STATIONARY point process. A capacity-constrained
stipple of an icon is non-stationary by construction -- the silhouette contributes exactly the
low-frequency energy whose absence is the blue-noise signature -- so a periodogram of those
point sets largely measures the conditioning image rather than the sampling quality. On a
constant target density the process IS stationary, which is the standard setting in which
blue-noise spectra are reported and the one place our sampler is comparable to GBN on its own
ground.

Several grey levels are emitted because capacity constraints distribute points by RELATIVE
density: a black and a mid-grey uniform image describe the same target distribution, since the
absolute level cancels under normalisation. The level is still a different network input, so
sweeping it tests INVARIANCE -- that the model responds to relative density, not absolute
intensity. Pure white is excluded: it carries no mass and there is nothing to distribute.

SPECTRAL_TESTS_2 -- patterned images, analysed with the pair correlation function
--------------------------------------------------------------------------------
Tests whether blue-noise structure survives LOCALLY when the density varies within a single
image. This cannot be done with an FFT: any structured image is non-stationary, and cropping a
region applies a boxcar window whose sinc transform smears the very annulus being measured. The
pair correlation function has no such restriction -- it normalises distances by the local mean
spacing, which is what Oztireli and Gross designed it for -- so the analysis is done per region
with the PCF instead.

Region edges are handled properly in stage 1: only points in the INTERIOR of a region are used
as centres, with the neighbour radius capped by the margin, so every counted pair lies inside a
single density regime.

Output
------
    experiments/outputs/spectral_analysis/source/<stem>.png
    experiments/outputs/spectral_analysis/manifest.json

Then run each sampler over that folder, writing to sibling folders:
    experiments/outputs/spectral_analysis/target_{WVS,BNOT,GBN,CN}_<budget>/<stem>.npy
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

# ── which experiment to build ────────────────────────────────────────────────
SPECTRAL_TESTS_1 = True    # uniform images  -> power spectrum
SPECTRAL_TESTS_2 = True    # patterned images -> per-region PCF

# ── TEST 1: uniform grey levels ──────────────────────────────────────────────
# 8-bit values; rho = 1 - value/255, so 0 is maximum ink. 255 is rejected (no mass).
GREY_LEVELS = [0, 64, 128, 192]

# ── TEST 2: pattern ──────────────────────────────────────────────────────────
# "quad4"   2x2 with four DISTINCT non-zero levels. Recommended: every quadrant carries
#           points, so all four are analysable and one image spans the density range.
# "checker" 2x2 [black | white / white | grey]. Two quadrants are empty by construction
#           (white has no mass), so only two are analysable -- but it doubles as a capacity
#           figure, showing evacuation of white and the 2:1 ratio between black and grey.
# Both patterns are built by default; each yields its own source images and its own region
# definitions, so they can be analysed independently in stage 1.
PATTERNS = ["quad4", "checker"]
QUAD4_LEVELS = [0, 51, 102, 153]        # rho = 1.00, 0.80, 0.60, 0.40 -- balanced point counts
CHECKER_LEVELS = [0, 255, 255, 128]     # black | white / white | grey

# Independent realizations per source image. A single point set's statistics are noise; the
# smooth curves in the literature are averaged over many.
REALIZATIONS = 25

RESOLUTION = 512
POINT_BUDGET = 1024
METHODS = ["WVS", "BNOT", "GBN", "CN-WVS", "CN-GBN"]


def parse_args():
    p = argparse.ArgumentParser(description="Create source images for the spectral analysis")
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--tests-1", action=argparse.BooleanOptionalAction, default=SPECTRAL_TESTS_1,
                   help="Build uniform images for the power-spectrum experiment")
    p.add_argument("--tests-2", action=argparse.BooleanOptionalAction, default=SPECTRAL_TESTS_2,
                   help="Build patterned images for the per-region PCF experiment")
    p.add_argument("--patterns", default=",".join(PATTERNS),
                   help=f"Comma-separated patterns to build (default: {','.join(PATTERNS)}). "
                        f"Available: quad4, checker")
    p.add_argument("--levels", default=",".join(str(g) for g in GREY_LEVELS),
                   help="TEST 1 grey levels (255 is rejected: no mass)")
    p.add_argument("--realizations", type=int, default=REALIZATIONS)
    p.add_argument("--resolution", type=int, default=RESOLUTION)
    p.add_argument("--point-budget", type=int, default=POINT_BUDGET)
    p.add_argument("--clean", action="store_true", help="Delete existing sources first")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_pattern(pattern, res):
    """Return (image, regions). Each region is a quadrant with its grey level and box."""
    levels = QUAD4_LEVELS if pattern == "quad4" else CHECKER_LEVELS
    img = np.zeros((res, res), dtype=np.uint8)
    h = res // 2
    # quadrant order: top-left, top-right, bottom-left, bottom-right
    boxes = [(0, 0, h, h), (h, 0, res, h), (0, h, h, res), (h, h, res, res)]
    regions = []
    for (x0, y0, x1, y1), g in zip(boxes, levels):
        img[y0:y1, x0:x1] = g
        regions.append({
            "name": f"q{len(regions)}",
            "grey": int(g),
            "rho": 1.0 - g / 255.0,
            # normalised [x0, y0, x1, y1] so stage 1 can assign points without the image
            "box": [x0 / res, y0 / res, x1 / res, y1 / res],
            "analysable": g < 255,          # white carries no mass -> no points to analyse
        })
    return img, regions


def main():
    args = parse_args()
    out_base = Path(args.output)
    src_dir = out_base / "source"

    if not args.tests_1 and not args.tests_2:
        print("ERROR: both --no-tests-1 and --no-tests-2; nothing to build.")
        return 2

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    bad = [g for g in levels if not (0 <= g < 255)]
    if args.tests_1 and bad:
        print(f"ERROR: TEST 1 grey levels must be in [0, 255); pure white carries no mass. Got {bad}")
        return 2

    planned = []
    if args.tests_1:
        planned += [(f"uniform_g{g:03d}_r{r:02d}", "uniform", g)
                    for g in levels for r in range(args.realizations)]
    pattern_imgs, regions_by_pattern = {}, {}
    pats = [x.strip() for x in args.patterns.split(",") if x.strip()] if args.tests_2 else []
    bad_pat = [x for x in pats if x not in ("quad4", "checker")]
    if bad_pat:
        print(f"ERROR: unknown pattern(s) {bad_pat}; available: quad4, checker")
        return 2
    for pat in pats:
        img_p, regs = build_pattern(pat, args.resolution)
        pattern_imgs[pat] = img_p
        regions_by_pattern[pat] = regs
        planned += [(f"pattern_{pat}_r{r:02d}", "pattern", pat)
                    for r in range(args.realizations)]

    print(f"Output base : {out_base}")
    print(f"TEST 1 (uniform / power spectrum) : {'ON' if args.tests_1 else 'off'}"
          + (f"   levels {levels}" if args.tests_1 else ""))
    print(f"TEST 2 (pattern / per-region PCF) : {'ON' if args.tests_2 else 'off'}"
          + (f"   patterns {pats}" if args.tests_2 else ""))
    for pat in pats:
        regs = regions_by_pattern[pat]
        n_ok = sum(r["analysable"] for r in regs)
        print(f"  pattern '{pat}'  ({n_ok}/4 regions analysable)")
        for r in regs:
            note = "" if r["analysable"] else "   (white: no points, capacity demo only)"
            print(f"    {r['name']}  grey {r['grey']:3d}  rho {r['rho']:.2f}{note}")
    print(f"Realizations: {args.realizations} per image")
    print(f"Total images: {len(planned)}")

    if args.dry_run:
        print("\nDRY RUN: nothing written")
        return 0

    src_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        removed = sum(1 for f in src_dir.glob("*.png") if f.unlink() is None)
        if removed:
            print(f"Removed {removed} existing source image(s)")

    written = []
    for stem, kind, grey in planned:
        if kind == "uniform":
            img = np.full((args.resolution, args.resolution), grey, dtype=np.uint8)
            meta = {"stem": stem, "kind": "uniform", "grey": grey, "rho": 1.0 - grey / 255.0}
        else:
            img = pattern_imgs[grey]          # `grey` carries the pattern name for this kind
            meta = {"stem": stem, "kind": "pattern", "pattern": grey}
        if not cv2.imwrite(str(src_dir / f"{stem}.png"), img):
            print(f"ERROR: failed to write {stem}.png")
            return 3
        written.append(meta)

    manifest = {
        "resolution": args.resolution,
        "point_budget": args.point_budget,
        "realizations": args.realizations,
        "tests_1": bool(args.tests_1),
        "tests_2": bool(args.tests_2),
        "grey_levels": levels if args.tests_1 else [],
        "patterns": pats,
        # regions differ per pattern, so they are keyed by pattern name
        "regions": regions_by_pattern,
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
    print("then spectral_analysis_stage_1.py per target folder, then stage_2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
