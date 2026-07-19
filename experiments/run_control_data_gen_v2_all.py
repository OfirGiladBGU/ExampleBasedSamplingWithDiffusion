#!/usr/bin/env python3
"""Run control_data_gen.py for all ICONS-TIMES-V2 grid sizes.

After each run, the output subfolders (target, target_npy, timestamps) are
renamed with a _CN_<GRID_SIZE> postfix. source/ is left untouched.

All run parameters are passed explicitly on the command line so this driver
does not depend on the defaults inside control_data_gen.py.
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

# Full ICONS - TIMES - V2 parameter set (passed explicitly; do not assume the
# underlying script defaults).
BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = "config/GBN/model.ckpt"
CONTROL_CKPT = "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt"

EVAL_TIMESTEPS = 1000
INFER_TRUNCATION_RATIO = 0.30
RESAMPLE_JUMPS = 2
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
SMART_INIT_FEATURES = False
SDF_FEATURES = False
BATCH_COORDS_FEATURES = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False
SMART_INIT_SEED = 42
SMART_INIT_SPLAT_SIGMA_PX = 0.5
SDF_TRUNCATE_PX = 8.0

DEVICE = "cuda"
OVERWRITE = True
EXPORT_PNG = True
EXPORT_NPY = True

TRACK_TIME = True  # NOTE
TRACK_TIME_FULL = False  # NOTE
TRACK_PROFILE_TRACE = False  # NOTE

# ICONS - TIMES - V2  (interpreted as grid_size^2 internally)
GRID_SIZES = [
    16,   # 256 
    24,   # 576
    32,   # 1024
    40,   # 1600
    48,   # 2304
    56,   # 3136
    64,   # 4096
    72,   # 5184
    80,   # 6400
    88,   # 7744
    96,   # 9216
    104,  # 10816
    112   # 12544
]

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

        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--data_path", str(DATA_PATH),
            "--base_config_path", BASE_CONFIG_PATH,
            "--base_ckpt_path", BASE_CKPT_PATH,
            "--control_ckpt_path", CONTROL_CKPT,
            "--eval_timesteps", str(EVAL_TIMESTEPS),
            "--grid_size", str(grid_size),
            "--infer-truncation-ratio", str(INFER_TRUNCATION_RATIO),
            "--resample_jumps", str(RESAMPLE_JUMPS),
            "--enable_gecco" if ENABLE_GECCO else "--no-enable_gecco",
            "--enable_adaptive_gate_injection" if ENABLE_ADAPTIVE_GATE_INJECTION else "--no-enable_adaptive_gate_injection",
            "--smart_init_features" if SMART_INIT_FEATURES else "--no-smart_init_features",
            "--sdf_features" if SDF_FEATURES else "--no-sdf_features",
            "--batch_coords_features" if BATCH_COORDS_FEATURES else "--no-batch_coords_features",
            "--enable_smart_init_splat_sigma" if ENABLE_SMART_INIT_SPLAT_SIGMA else "--no-enable_smart_init_splat_sigma",
            "--smart_init_seed", str(SMART_INIT_SEED),
            "--smart_init_splat_sigma_px", str(SMART_INIT_SPLAT_SIGMA_PX),
            "--sdf_truncate_px", str(SDF_TRUNCATE_PX),
            "--device", DEVICE,
            "--overwrite" if OVERWRITE else "--no-overwrite",
            "--export_png" if EXPORT_PNG else "--no-export_png",
            "--export_npy" if EXPORT_NPY else "--no-export_npy",
            "--track_time" if TRACK_TIME else "--no-track_time",
            "--track_time_full" if TRACK_TIME_FULL else "--no-track_time_full",
            "--profile_trace" if TRACK_PROFILE_TRACE else "--no-profile_trace",
        ]
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
