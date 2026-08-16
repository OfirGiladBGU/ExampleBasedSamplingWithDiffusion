#!/usr/bin/env python
"""Archive ablation checkpoints at a fixed epoch interval.

Updated for the new GBN ablations and the current weights layout.

Ablations (each has its own train_outputs_Icons-50_1024_GBN_<method> folder):
    vanilla, unfrozen, gecco, agi, full
`sdedit` is inference-only and reuses the `full` weights, so it has no folder
of its own -- archive `full` for it.

Weights layout: a single `checkpoints/` folder (the old `checkpoints_lighten/`
folder is gone) containing periodic snapshots `dynamic_ep{N}.ckpt` alongside
`best_dynamic_ep{N}_...ckpt`. Only the periodic snapshots on the every-N-epoch
interval are archived; the best_* checkpoints are ignored.
"""

import re
import sys
from pathlib import Path
from zipfile import ZipFile

# -- Configuration -----------------------------------------------------------
CONTROL_FOLDER = Path(__file__).resolve().parent

# Vanilla
BASE_TRAIN_FOLDER = CONTROL_FOLDER / "train_outputs_Icons-50_1024_GBN_vanilla"
ZIP_NAME = "ablation_vanilla_5K_checkpoints.zip"

# Unfrozen
# BASE_TRAIN_FOLDER = CONTROL_FOLDER / "train_outputs_Icons-50_1024_GBN_unfrozen"
# ZIP_NAME = "ablation_unfrozen_5K_checkpoints.zip"

# Gecco
# BASE_TRAIN_FOLDER = CONTROL_FOLDER / "train_outputs_Icons-50_1024_GBN_gecco"
# ZIP_NAME = "ablation_gecco_5K_checkpoints.zip"

# AGI
# BASE_TRAIN_FOLDER = CONTROL_FOLDER / "train_outputs_Icons-50_1024_GBN_agi"
# ZIP_NAME = "ablation_agi_5K_checkpoints.zip"

# Full  (also archive this one for the sdedit ablation)
# BASE_TRAIN_FOLDER = CONTROL_FOLDER / "train_outputs_Icons-50_1024_GBN_full"
# ZIP_NAME = "ablation_full_5K_checkpoints.zip"


EVERY_EPOCH_NUM = 500

# The single checkpoints folder in the new layout.
CHECKPOINTS_DIR = "checkpoints"

# Periodic snapshots only. The leading anchor makes this NOT match
# best_dynamic_ep..., so best checkpoints are excluded.
PERIODIC_PATTERN = r"^dynamic_ep(\d+)\.ckpt$"


# -- Helpers -----------------------------------------------------------------

def extract_epoch(name):
    """Epoch number if `name` is a periodic checkpoint, else None."""
    match = re.match(PERIODIC_PATTERN, name)
    return int(match.group(1)) if match else None


def find_periodic_checkpoints(directory):
    """All (epoch, path) periodic checkpoints in `directory`, sorted by epoch."""
    directory = Path(directory)
    if not directory.exists():
        return []
    found = []
    for file in directory.glob("*.ckpt"):
        epoch = extract_epoch(file.name)
        if epoch is not None:
            found.append((epoch, file))
    return sorted(found, key=lambda x: x[0])


def filter_by_interval(items, interval):
    """Keep (epoch, path) where epoch % interval == 0."""
    return [(epoch, path) for epoch, path in items if epoch % interval == 0]


# -- Main --------------------------------------------------------------------

def create_checkpoint_archive(base_folder, output_zip, every_n_epochs):
    """Create a zip of the periodic (every N epochs) checkpoints."""
    base_path = Path(base_folder)
    if not base_path.exists():
        print(f"Error: Base folder not found: {base_path}")
        sys.exit(1)

    checkpoints_dir = base_path / CHECKPOINTS_DIR
    if not checkpoints_dir.exists():
        print(f"Error: checkpoints folder not found: {checkpoints_dir}")
        sys.exit(1)

    print(f"Base folder: {base_path}")
    print(f"Epoch interval: {every_n_epochs}")
    print(f"Output zip: {output_zip}\n")

    periodic = find_periodic_checkpoints(checkpoints_dir)
    filtered = filter_by_interval(periodic, every_n_epochs)

    print(f"Found {len(periodic)} periodic checkpoints; "
          f"{len(filtered)} on the every-{every_n_epochs} interval.")
    for epoch, path in filtered:
        print(f"  ep{epoch}: {path.name}")

    if not filtered:
        print("\nError: nothing to archive (no periodic checkpoints on the interval).")
        sys.exit(1)

    print(f"\nCreating archive: {output_zip}")
    with ZipFile(output_zip, "w") as zf:
        for epoch, path in filtered:
            arcname = f"{CHECKPOINTS_DIR}/{path.name}"
            zf.write(path, arcname=arcname)
            print(f"  Added ep{epoch}: {arcname}")

    zip_size_mb = Path(output_zip).stat().st_size / (1024 * 1024)
    print(f"\nArchive created: {output_zip} "
          f"({len(filtered)} files, {zip_size_mb:.1f} MB)")


if __name__ == "__main__":
    create_checkpoint_archive(BASE_TRAIN_FOLDER, ZIP_NAME, EVERY_EPOCH_NUM)
