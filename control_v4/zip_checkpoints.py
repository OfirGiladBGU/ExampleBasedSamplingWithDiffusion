#!/usr/bin/env python
"""Archive checkpoints by epoch intervals and include latest backup."""

import re
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

# ── Configuration ────────────────────────────────────────────────────────────

# Vanilla
BASE_TRAIN_FOLDER = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_vanilla"
ZIP_NAME = "ablation_vanilla_10K_checkpoints.zip"

# Gecco
# BASE_TRAIN_FOLDER = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_gecco"
# ZIP_NAME = "ablation_gecco_10K_checkpoints.zip"

# AGI
# BASE_TRAIN_FOLDER = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_agi"
# ZIP_NAME = "ablation_agi_10K_checkpoints.zip"

# Full
# BASE_TRAIN_FOLDER = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_outputs_icons50_512_full"
# ZIP_NAME = "ablation_full_10K_checkpoints.zip"


EVERY_EPOCH_NUM = 500

# Subdirectories to search
CHECKPOINTS_LIGHTEN_DIR = "checkpoints_lighten"
CHECKPOINTS_DIR = "checkpoints"

# Pattern for checkpoint files
CHECKPOINT_PATTERN = r"dynamic_ep(\d+)\.ckpt"


# ── Main ─────────────────────────────────────────────────────────────────────

def extract_epoch_num(filename):
    """Extract epoch number from checkpoint filename."""
    match = re.search(CHECKPOINT_PATTERN, filename)
    return int(match.group(1)) if match else None


def find_checkpoints(directory, pattern=CHECKPOINT_PATTERN):
    """Find all checkpoint files in directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    checkpoints = []
    for file in directory.glob("*.ckpt"):
        epoch = extract_epoch_num(file.name)
        if epoch is not None:
            checkpoints.append((epoch, file))

    return sorted(checkpoints, key=lambda x: x[0])


def filter_by_interval(checkpoints, interval):
    """Filter checkpoints where epoch % interval == 0."""
    return [(epoch, ckpt) for epoch, ckpt in checkpoints if epoch % interval == 0]


def create_checkpoint_archive(base_folder, output_zip, every_n_epochs):
    """Create zip archive with filtered checkpoints."""
    base_path = Path(base_folder)

    if not base_path.exists():
        print(f"Error: Base folder not found: {base_path}")
        sys.exit(1)

    lighten_dir = base_path / CHECKPOINTS_LIGHTEN_DIR
    checkpoints_dir = base_path / CHECKPOINTS_DIR

    print(f"Base folder: {base_path}")
    print(f"Epoch interval: {every_n_epochs}")
    print(f"Output zip: {output_zip}\n")

    # Find checkpoints from lighten folder
    if lighten_dir.exists():
        all_lighten = find_checkpoints(lighten_dir)
        filtered_lighten = filter_by_interval(all_lighten, every_n_epochs)
        print(f"Found {len(all_lighten)} checkpoints in {CHECKPOINTS_LIGHTEN_DIR}")
        print(f"Filtered to {len(filtered_lighten)} checkpoints (every {every_n_epochs} epochs)\n")

        for epoch, path in filtered_lighten:
            print(f"  ✓ ep{epoch}: {path.name}")
    else:
        print(f"Warning: {CHECKPOINTS_LIGHTEN_DIR} not found")
        filtered_lighten = []

    # Find latest checkpoint from checkpoints folder for backup
    latest_backup = None
    if checkpoints_dir.exists():
        all_checkpoints = find_checkpoints(checkpoints_dir)
        if all_checkpoints:
            latest_epoch, latest_path = all_checkpoints[-1]
            latest_backup = latest_path
            print(f"\nBackup checkpoint (latest):")
            print(f"  ✓ ep{latest_epoch}: {latest_path.name}\n")
    else:
        print(f"Warning: {CHECKPOINTS_DIR} not found\n")

    # Create zip archive
    print(f"Creating archive: {output_zip}")
    with ZipFile(output_zip, 'w') as zf:
        # Add filtered lighten checkpoints
        for epoch, path in filtered_lighten:
            arcname = f"{CHECKPOINTS_LIGHTEN_DIR}/{path.name}"
            zf.write(path, arcname=arcname)
            print(f"  Added: {arcname}")

        # Add latest backup
        if latest_backup:
            arcname = f"{CHECKPOINTS_DIR}/{latest_backup.name}"
            zf.write(latest_backup, arcname=arcname)
            print(f"  Added: {arcname}")

    # Show final size
    zip_size_mb = Path(output_zip).stat().st_size / (1024 * 1024)
    print(f"\n✓ Archive created: {output_zip} ({zip_size_mb:.1f} MB)")


if __name__ == "__main__":
    create_checkpoint_archive(BASE_TRAIN_FOLDER, ZIP_NAME, EVERY_EPOCH_NUM)
