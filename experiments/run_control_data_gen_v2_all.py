#!/usr/bin/env python3
"""Run control_data_gen.py for all ICONS-TIMES-V2 grid sizes.

After each run, the output subfolders (target, target_npy, timestamps) are
renamed with a _CN_<GRID_SIZE> postfix. source/ is left untouched.
"""

import subprocess
import sys
from pathlib import Path

# -- Configuration -------------------------------------------------------------

SCRIPT_PATH = Path(__file__).parent / "control_data_gen.py"

DATA_PATH = Path(
    "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion"
    "/experiments/outputs/icons_results_runtimes"
)

# ICONS - TIMES - V2  (interpreted as grid_size^2 internally)
GRID_SIZES = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112]

# Subfolders produced by each run that should be renamed after completion
OUTPUT_SUBDIRS = ["target", "target_npy", "timestamps"]

# -- Main ----------------------------------------------------------------------

def main():
    if not SCRIPT_PATH.exists():
        print(f"ERROR: Script not found at {SCRIPT_PATH}")
        sys.exit(1)

    print(f"Running {len(GRID_SIZES)} grid sizes: {GRID_SIZES}")
    print(f"Script: {SCRIPT_PATH}\n")

    for grid_size in GRID_SIZES:
        print(f"\n{'='*70}")
        print(f"  GRID_SIZE = {grid_size}")
        print(f"{'='*70}\n")

        cmd = [sys.executable, str(SCRIPT_PATH), "--grid_size", str(grid_size)]
        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nWarning: grid_size={grid_size} exited with code {result.returncode}")

        for subdir in OUTPUT_SUBDIRS:
            src = DATA_PATH / subdir
            n_points = grid_size * grid_size
            dst = DATA_PATH / f"{subdir}_CN_{n_points}"
            if src.exists():
                src.rename(dst)
                print(f"[rename] {subdir}  ->  {dst.name}")

    print(f"\n{'='*70}")
    print("All grid sizes processed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
