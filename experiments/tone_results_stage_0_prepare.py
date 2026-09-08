"""tone_results_stage_0_prepare.py -- Part 0: stage the target images for the tone experiment.

Copies the condition images out of the shared validation folder and writes the manifest the
later stages read:

    experiments/outputs/z_validation_data/Icons-50_1024/source/
      -> experiments/outputs/tone_results/source/
      -> experiments/outputs/tone_results/manifest.json

Only `source/` is copied. The tone experiment compares a RENDERED stipple against a target
image and never references an exemplar point set, so the per-method `target_<METHOD>_<n>/`
folders are not needed here -- unlike the quantitative-metrics staging, which copies the
whole tree.

    python experiments/tone_results_stage_0_prepare.py
    python experiments/tone_results_stage_0_prepare.py --dry-run
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = "experiments/outputs/z_validation_data/Icons-50_1024"
OUT_DIR = "experiments/outputs/tone_results"
COUNT = 24

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# An image that is almost all background has very little ink to redistribute, so the tone
# correction has nothing to work with and the run is uninformative rather than wrong.
MIN_INK_FRACTION = 0.04


# --------------------------------------------------------------------------
# Defaults for every command-line argument. Edit here, not in parse_args.
# --------------------------------------------------------------------------
SEED = 0


def parse_args():
    ap = argparse.ArgumentParser(
        description="Copy the validation condition images and select targets for the tone experiment.")
    ap.add_argument("--src", default=SRC_DIR,
                    help="Shared validation folder (built by z_copy_validatation_data.py).")
    ap.add_argument("--output", default=OUT_DIR, help="Tone experiment output folder.")
    ap.add_argument("--count", type=int, default=COUNT, help="-1 keeps every eligible image.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-ink", type=float, default=MIN_INK_FRACTION)
    ap.add_argument("--no-copy", action="store_true",
                    help="Select from an already-populated <output>/source instead of copying.")
    ap.add_argument("--dry-run", action="store_true", help="Report what would happen; write nothing.")
    return ap.parse_args()


def ink_fraction(path):
    """Mean darkness in [0,1]. 0 is a blank page."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    return float((1.0 - img.astype(np.float64) / 255.0).mean()), img.shape


def main():
    args = parse_args()
    src_root = Path(args.src)
    out = Path(args.output)
    dst_src = out / "source"

    if not args.no_copy:
        src = src_root / "source"
        if not src.is_dir():
            print(f"Error: source folder not found: {src}")
            return 1
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            print(f"Error: no images under {src}")
            return 1
        print(f"src: {src}  ({len(files)} images)")
        print(f"dst: {dst_src}")
        if not args.dry_run:
            # Flatten: the later stages address images by stem, so a nested tree would let
            # two different icons collide on one manifest entry.
            dst_src.mkdir(parents=True, exist_ok=True)
            seen = {}
            for f in files:
                stem = f.stem
                if stem in seen:
                    print(f"  WARNING: duplicate stem {stem!r} "
                          f"({f} vs {seen[stem]}) -- keeping the first")
                    continue
                seen[stem] = f
                shutil.copy2(f, dst_src / f.name)
            print(f"copied {len(seen)} image(s)")
    else:
        if not dst_src.is_dir():
            print(f"Error: --no-copy given but {dst_src} does not exist")
            return 1

    scan = dst_src if not args.dry_run else (src_root / "source")
    files = sorted(p for p in scan.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    eligible = []
    for f in files:
        ink, shape = ink_fraction(f)
        if ink is None:
            print(f"  unreadable, skipped: {f.name}")
            continue
        if ink < args.min_ink:
            continue
        eligible.append({"path": str(dst_src / f.name), "stem": f.stem,
                         "ink_fraction": round(ink, 5),
                         "height": int(shape[0]), "width": int(shape[1])})

    print(f"{len(eligible)}/{len(files)} pass the >= {args.min_ink:.0%} ink filter")
    if not eligible:
        print("Error: nothing eligible; lower --min-ink or check the source folder")
        return 1

    if 0 <= args.count < len(eligible):
        random.Random(args.seed).shuffle(eligible)
        eligible = sorted(eligible[:args.count], key=lambda e: e["stem"])
    print(f"selected {len(eligible)} image(s)")

    if args.dry_run:
        for e in eligible[:10]:
            print(f"  {e['stem']:44s} ink={e['ink_fraction']:.3f}")
        print("\n(dry run: nothing written)")
        return 0

    manifest = {"src": str(src_root), "source": str(dst_src), "seed": args.seed,
                "min_ink": args.min_ink, "count": len(eligible), "images": eligible}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"-> {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
