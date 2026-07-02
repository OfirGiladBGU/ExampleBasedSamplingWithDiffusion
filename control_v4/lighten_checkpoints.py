"""Lighten training checkpoints by clearing optimizer state.

Scans a training outputs folder (expects a `checkpoints/` subfolder)
and writes copies into `checkpoints_lighten/` where the `optimizer`
entry is set to `None` to reduce file size.

Example:
    python control_v4/lighten_checkpoints.py \
        --src /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_no_random

"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

# Global default parameters (edit here)
# SRC_FOLDER = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_agi"
# SRC_FOLDER = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full"
# SRC_FOLDER = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_gecco"
SRC_FOLDER = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_vanilla"
CHECKPOINTS_SUBDIR = "checkpoints"
OUT_SUBDIR = "checkpoints_lighten"
PATTERN = "*.pt"
DRY_RUN_DEFAULT = False

def human(x: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"


def process_checkpoint(in_path: str, out_path: str, dry_run: bool = False) -> tuple:
    import torch

    size_before = os.path.getsize(in_path)
    if dry_run:
        # estimate no-op: can't know exact size but report original
        return size_before, size_before

    state = torch.load(in_path, map_location="cpu")

    # Set optimizer entry to None if present
    if isinstance(state, dict) and "optimizer" in state:
        state["optimizer"] = None

    # Ensure output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(state, out_path)

    size_after = os.path.getsize(out_path)
    return size_before, size_after


def main():
    p = argparse.ArgumentParser(description="Lighten checkpoints by zeroing optimizer state")
    p.add_argument("--src", required=False, default=SRC_FOLDER,
                   help=f"Path to training outputs folder (contains '{CHECKPOINTS_SUBDIR}'), default={SRC_FOLDER}")
    p.add_argument("--checkpoints-subdir", default=CHECKPOINTS_SUBDIR,
                   help=f"Name of checkpoints subfolder (default: {CHECKPOINTS_SUBDIR})")
    p.add_argument("--out-subdir", default=OUT_SUBDIR,
                   help=f"Name of output subfolder to write lightened checkpoints (default: {OUT_SUBDIR})")
    p.add_argument("--pattern", default=PATTERN, help=f"Glob pattern for checkpoint files (default: {PATTERN})")
    p.add_argument("--dry-run", action="store_true", help="Do not write files; just report sizes")
    args = p.parse_args()

    src = Path(args.src)
    ckpt_dir = src / args.checkpoints_subdir
    out_dir = src / args.out_subdir

    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        print(f"Checkpoints dir not found: {ckpt_dir}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(str(ckpt_dir / args.pattern)))
    if not files:
        print(f"No checkpoint files found matching: {ckpt_dir / args.pattern}")
        return

    total_before = 0
    total_after = 0

    print(f"Found {len(files)} checkpoints. Writing to: {out_dir}")
    for f in files:
        fname = os.path.basename(f)
        out_path = str(out_dir / fname)
        try:
            before, after = process_checkpoint(f, out_path, dry_run=args.dry_run)
            total_before += before
            total_after += after
            print(f"{fname}: {human(before)} -> {human(after)}")
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("--- Summary ---")
    print(f"Total before: {human(total_before)}")
    print(f"Total after : {human(total_after)}")


if __name__ == '__main__':
    main()
