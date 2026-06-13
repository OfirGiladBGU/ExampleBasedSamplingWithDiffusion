#!/usr/bin/env python
"""Run sample_control.py with different grid sizes."""

import subprocess
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

# Defaults
ADDITIONAL_ARGS = None
# Modify this list to change which grid sizes to test
GRID_SIZES = [16, 32, 48, 64, 80, 96, 112]
# Path to sample_control.py
SCRIPT_PATH = Path(__file__).parent / "sample_control.py"


# CPU tracking
ADDITIONAL_ARGS = [
    "--device", "cpu",
    "--track_time",
    "--no-track_time_full",
    "--no-profile_trace",
]
OUTPUT_DIR_FORMAT = "control_v4/sample_outputs_{}_cpu"


# # GPU 6000 tracking (no profiler trace)
# ADDITIONAL_ARGS = [
#     "--device", "cuda",
#     "--track_time",
#     "--track_time_full",
# ]
# OUTPUT_DIR_FORMAT = "control_v4/sample_outputs_{}_gpu_6000_reg"


# # GPU 3090 tracking (no profiler trace)
# ADDITIONAL_ARGS = [
#     "--device", "cuda",
#     "--track_time",
#     "--track_time_full",
# ]
# OUTPUT_DIR_FORMAT = "control_v4/sample_outputs_{}_gpu_3090_reg"


# # GPU 6000 tracking (with profiler trace)
# ADDITIONAL_ARGS = [
#     "--device", "cuda",
#     "--track_time",
#     "--track_time_full",
#     "--profile_trace",
# ]
# OUTPUT_DIR_FORMAT = "control_v4/sample_outputs_{}_gpu_6000"


# # GPU 3090 tracking (with profiler trace)
# ADDITIONAL_ARGS = [
#     "--device", "cuda",
#     "--track_time",
#     "--track_time_full",
#     "--profile_trace",
# ]
# OUTPUT_DIR_FORMAT = "control_v4/sample_outputs_{}_gpu_3090"


# ── Main ─────────────────────────────────────────────────────────────────────

def run_grid_sizes(script_path, grid_sizes, output_dir_format, additional_args=None):
    """Run sample_control.py for each grid size."""
    if additional_args is None:
        additional_args = []

    script_path = Path(script_path)
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    print(f"Running inference with grid sizes: {grid_sizes}")
    print(f"Script: {script_path}\n")

    for grid_size in grid_sizes:
        print(f"\n{'='*70}")
        print(f"Running with grid_size={grid_size}")
        print(f"{'='*70}\n")
        output_dir = output_dir_format.format(grid_size)

        cmd = [
            sys.executable,
            str(script_path),
            "--grid_size", str(grid_size),
            "--output_dir", output_dir
        ] + additional_args

        print(f"Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nWarning: grid_size={grid_size} exited with code {result.returncode}")
        else:
            print(f"\nCompleted: grid_size={grid_size}")

    print(f"\n{'='*70}")
    print("All grid sizes processed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_grid_sizes(SCRIPT_PATH, GRID_SIZES, OUTPUT_DIR_FORMAT, ADDITIONAL_ARGS)
