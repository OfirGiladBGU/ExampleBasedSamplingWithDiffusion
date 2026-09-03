#!/usr/bin/env python3
"""Stage 0: build the four per-method runtime folders and stage their input images.

Fills each of outputs/icons_results_runtimes_{wvs,bnot,gbn,cn}/ with `source/` and
`original/`, nested as Icons-50/<category>/<image>.png. Each method's own stippling code
then reads its folder and writes target_<METHOD>_<GRID>/ + timestamps_<METHOD>_<GRID>/
back into it. Then:

    merge_runtime_results_stage_1.py   gathers every timestamps_* folder into one place
    merge_runtime_results_stage_2.py   aggregates them into runtimes_avg.json

IMAGES is an explicit PINNED list, not a manifest prefix or a seeded draw: runtime results
for these exact images already exist in the target_*/timestamps_* folders, so the set must
not drift. (Its original provenance is unknown -- it matches neither the seed-42 manifest
prefix nor the alphabetical first 10.)

SRC_DATASET points at the WVS-family source/, NOT the dataset's `original/`. Verified
against the icons_results_runtimes_*.zip backups: these folders held the 512x512 grayscale
from Icons-50_1024_{WVS,BNOT,DITHER,FS}/source, whereas `original/` is a 108x108 icon --
using it would change the input resolution and so the runtimes.

`source/` is seeded with the same file; BNOT and GBN overwrite their own `source/` with a
preprocessed version when they run, which is their pipeline, not this stage's job.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent / "outputs"

# The 512x512 grayscale inputs the existing runtime results were measured on. Identical in
# Icons-50_1024_{WVS,BNOT,DITHER,FS}/source -- see the module docstring.
SRC_DATASET = Path("/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_WVS/source")

METHOD_DIRS = [
    "icons_results_runtimes_wvs",
    "icons_results_runtimes_bnot",
    "icons_results_runtimes_gbn",
    "icons_results_runtimes_cn",
]

# Sub-folders each method folder gets, both seeded with the same file.
INPUT_SUBDIRS = ["source", "original"]

# PINNED -- see the module docstring. Paths are relative to SRC_DATASET and are kept
# nested, matching the layout the existing target_*/timestamps_* folders use.
IMAGES = [
    "Icons-50/arrow_directions/microsoft_4_up_down_arrow.png",
    "Icons-50/arrow_directions/twitter_0_arrow_pointing_rightwards_then_curving_upwards.png",
    "Icons-50/clock/samsung_3_watch.png",
    "Icons-50/emotion_face/microsoft_0_kissing_face.png",
    "Icons-50/family/microsoft_2_family_man_man_girl.png",
    "Icons-50/fast_train/microsoft_3_metro.png",
    "Icons-50/hand/emoji-one_3_waving_hand_sign.png",
    "Icons-50/japanese_ideograph/samsung_4_squared_cjk_unified_ideograph_7533.png",
    "Icons-50/mailbox/htc_0_closed_mailbox_with_raised_flag.png",
    "Icons-50/worker/twitter_0_female_factory_worker.png",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC_DATASET),
                    help="Dataset folder holding the pinned images (nested Icons-50/...).")
    ap.add_argument("--methods", nargs="+", default=METHOD_DIRS,
                    help="Per-method folders to build (relative to outputs/, or absolute).")
    ap.add_argument("--clean", action="store_true",
                    help="Also delete existing target_*/timestamps_* folders. Off by default: "
                         "they hold the measured results for these same images.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    return ap.parse_args()


def stale_result_dirs(method_path: Path):
    if not method_path.is_dir():
        return []
    return sorted(d for d in method_path.iterdir()
                  if d.is_dir() and (d.name.startswith("target_") or d.name.startswith("timestamps_")))


def main():
    args = parse_args()
    src = Path(args.src)
    if not src.is_dir():
        print(f"Error: source dataset not found: {src}")
        return 1

    missing = [r for r in IMAGES if not (src / r).exists()]
    if missing:
        print(f"Error: {len(missing)} pinned image(s) missing from {src}:")
        for r in missing:
            print(f"    {r}")
        return 1

    print(f"src    : {src}")
    print(f"images : {len(IMAGES)} pinned (matching the existing runtime results)")
    for i, r in enumerate(IMAGES):
        print(f"   [{i}] {r}")
    print()

    for method in args.methods:
        mp = Path(method) if Path(method).is_absolute() else BASE / method
        stale = stale_result_dirs(mp)
        action = "rebuild" if mp.is_dir() else "CREATE "
        print(f"  {action} {mp}")
        for sub in INPUT_SUBDIRS:
            print(f"      {sub}/  <- {len(IMAGES)} images (nested)")
        if stale:
            verb = "DELETE" if args.clean else "keep  "
            print(f"      {verb} {len(stale)} existing target_*/timestamps_* folder(s)")

        if args.dry_run:
            continue

        mp.mkdir(parents=True, exist_ok=True)
        for sub in INPUT_SUBDIRS:
            d = mp / sub
            if d.exists():
                shutil.rmtree(d)          # this stage owns source/ and original/
            for r in IMAGES:
                dst = d / r
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / r, dst)
        if args.clean:
            for dd in stale:
                shutil.rmtree(dd)

    if args.dry_run:
        print("\nDRY RUN: nothing written.")
        return 0

    print(f"\nStaged {len(IMAGES)} images into {len(args.methods)} method folder(s) "
          f"x {len(INPUT_SUBDIRS)} input dir(s).")
    print("The existing target_*/timestamps_* results were measured on these same images, "
          "so they stay valid;\nre-running a method overwrites its own files in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
