import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. Configuration ---

# CPU mode: List of timing txt files
DEVICE_MODE = "CPU"  # Set to "CPU" or "GPU"
TXT_FILES = [
    "control_v4/sample_outputs_16_cpu/emoji-one_4_monkey/timestamps/emoji-one_4_monkey.txt",
    "control_v4/sample_outputs_32_cpu/emoji-one_4_monkey/timestamps/emoji-one_4_monkey.txt",
    "control_v4/sample_outputs_48_cpu/emoji-one_4_monkey/timestamps/emoji-one_4_monkey.txt",
    "control_v4/sample_outputs_64_cpu/emoji-one_4_monkey/timestamps/emoji-one_4_monkey.txt",
    "control_v4/sample_outputs_80_cpu/emoji-one_4_monkey/timestamps/emoji-one_4_monkey.txt",
    "control_v4/sample_outputs_96_cpu/emoji-one_4_monkey/timestamps/emoji-one_4_monkey.txt",
    "control_v4/sample_outputs_112_cpu/emoji-one_4_monkey/timestamps/emoji-one_4_monkey.txt",
]
#sym:DEFAULT_X_AXIS_MODE
DEFAULT_X_AXIS_MODE = "grid"
PLOT_NAME = "control_v4/z_runtime_outputs_plots/cpu_denoising_time_grid.png"
# DEFAULT_X_AXIS_MODE = "points"
# PLOT_NAME = "control_v4/z_runtime_outputs_plots/cpu_denoising_time_points.png"


# GPU 6000 mode: List of profiler summary txt files
# DEVICE_MODE = "GPU"  # Set to "CPU" or "GPU"
# TXT_FILES = [
#     "control_v4/sample_outputs_16_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_32_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_48_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_64_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_80_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_96_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_112_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
# ]
# DEFAULT_X_AXIS_MODE = "grid"
# PLOT_NAME = "control_v4/z_runtime_outputs_plots/gpu_6000_profiler_times_grid.png"
# DEFAULT_X_AXIS_MODE = "points"
# PLOT_NAME = "control_v4/z_runtime_outputs_plots/gpu_6000_profiler_times_points.png"


# GPU 3090 mode: List of profiler summary txt files
# DEVICE_MODE = "GPU"  # Set to "CPU" or "GPU"
# TXT_FILES = [
#     "control_v4/sample_outputs_16_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_32_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_48_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_64_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_80_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_96_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
#     "control_v4/sample_outputs_112_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_profiler_summary.txt",
# ]
# DEFAULT_X_AXIS_MODE = "grid"
# PLOT_NAME = "control_v4/z_runtime_outputs_plots/gpu_3090_profiler_times_grid.png"
# DEFAULT_X_AXIS_MODE = "points"
# PLOT_NAME = "control_v4/z_runtime_outputs_plots/gpu_3090_profiler_times_points.png"


# --- 2. Parsing Functions ---

def parse_cpu_timing(txt_path):
    """Extract Denoising Time from CPU timing txt file."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")

    try:
        with open(txt_path, 'r') as f:
            content = f.read()

        match = re.search(r"Denoising Time:\s*([\d.]+)\s*s", content)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Could not find 'Denoising Time' in {txt_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading {txt_path}: {e}")


def parse_gpu_profiler(txt_path):
    """Extract CPU and CUDA times from GPU profiler summary txt file."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"File not found: {txt_path}")

    try:
        with open(txt_path, 'r') as f:
            content = f.read()

        cpu_match = re.search(r"Self CPU time total:\s*([\d.]+)s", content)
        cuda_match = re.search(r"Self CUDA time total:\s*([\d.]+)s", content)

        if not cpu_match:
            raise ValueError(f"Could not find 'Self CPU time total' in {txt_path}")
        if not cuda_match:
            raise ValueError(f"Could not find 'Self CUDA time total' in {txt_path}")

        return float(cpu_match.group(1)), float(cuda_match.group(1))
    except Exception as e:
        raise RuntimeError(f"Error reading {txt_path}: {e}")


def extract_grid_size(txt_path):
    """Extract grid size from file path (e.g., sample_outputs_32_cpu -> 32)."""
    match = re.search(r"sample_outputs_(\d+)", str(txt_path))
    if match:
        return int(match.group(1))
    return None


# --- 3. Plotting Functions ---

def generate_cpu_plot(txt_paths, plot_name, x_mode="grid"):
    """Generate plot for CPU mode."""
    grid_sizes = []
    labels = []
    denoise_times = []

    for path in txt_paths:
        grid_size = extract_grid_size(path)
        if grid_size is None:
            continue

        denoise_time = parse_cpu_timing(path)
        grid_sizes.append(grid_size)
        labels.append(f"{grid_size}x{grid_size}\n({grid_size**2} pts)")
        denoise_times.append(denoise_time)

    if not grid_sizes:
        print("Error: No valid data found for CPU mode")
        return

    # --- Setup Plotting ---
    base_height = 6

    if x_mode == "grid":
        fig_width = max(10, len(grid_sizes) * 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, base_height), dpi=150)
        x = grid_sizes
        ax.set_xlabel('Grid Size Resolution', fontsize=12, fontweight='bold')
        ax.set_title('CPU Denoising Time by Grid Size', fontsize=15, fontweight='bold')
        rotation = 0
        ha = 'center'

    elif x_mode == "points":
        fig_width = max(10, len(grid_sizes) * 3.5)
        fig, ax = plt.subplots(figsize=(fig_width, base_height), dpi=150)
        x = [g**2 for g in grid_sizes]
        ax.set_xlabel('Number of Points (Grid Size × Grid Size)', fontsize=12, fontweight='bold')
        ax.set_title('CPU Denoising Time by Point Count', fontsize=15, fontweight='bold')
        rotation = 15
        ha = 'right'

    else:
        raise ValueError("x_mode must be 'grid' or 'points'")

    ax.set_ylabel('Denoising Time (Seconds)', fontsize=12, fontweight='bold')

    # Plot
    ax.plot(x, denoise_times, marker='o', markersize=8, linewidth=3,
            linestyle='-', alpha=1.0, label='CPU Denoising Time', color='#1f77b4')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=rotation, ha=ha)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_name, dpi=300)
    print(f"Saved plot to {plot_name}")


def generate_gpu_plot(txt_paths, plot_name, x_mode="grid"):
    """Generate plot for GPU mode."""
    grid_sizes = []
    labels = []
    cpu_times = []
    cuda_times = []

    for path in txt_paths:
        grid_size = extract_grid_size(path)
        if grid_size is None:
            continue

        cpu_time, cuda_time = parse_gpu_profiler(path)
        grid_sizes.append(grid_size)
        labels.append(f"{grid_size}x{grid_size}\n({grid_size**2} pts)")
        cpu_times.append(cpu_time)
        cuda_times.append(cuda_time)

    if not grid_sizes:
        print("Error: No valid data found for GPU mode")
        return

    # --- Setup Plotting ---
    base_height = 6

    if x_mode == "grid":
        fig_width = max(10, len(grid_sizes) * 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, base_height), dpi=150)
        x = grid_sizes
        ax.set_xlabel('Grid Size Resolution', fontsize=12, fontweight='bold')
        ax.set_title('GPU Profiler Times by Grid Size', fontsize=15, fontweight='bold')
        rotation = 0
        ha = 'center'

    elif x_mode == "points":
        fig_width = max(10, len(grid_sizes) * 3.5)
        fig, ax = plt.subplots(figsize=(fig_width, base_height), dpi=150)
        x = [g**2 for g in grid_sizes]
        ax.set_xlabel('Number of Points (Grid Size × Grid Size)', fontsize=12, fontweight='bold')
        ax.set_title('GPU Profiler Times by Point Count', fontsize=15, fontweight='bold')
        rotation = 15
        ha = 'right'

    else:
        raise ValueError("x_mode must be 'grid' or 'points'")

    ax.set_ylabel('Time (Seconds)', fontsize=12, fontweight='bold')

    # Plot both CPU and CUDA
    ax.plot(x, cpu_times, marker='o', markersize=8, linewidth=3,
            linestyle='-', alpha=1.0, label='CPU Time', color='#1f77b4')
    ax.plot(x, cuda_times, marker='s', markersize=8, linewidth=3,
            linestyle='-', alpha=1.0, label='CUDA Time', color='#ff7f0e')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=rotation, ha=ha)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_name, dpi=300)
    print(f"Saved plot to {plot_name}")


# --- 4. Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate timing plots from txt files (CPU or GPU mode).")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["CPU", "GPU"],
        default=DEVICE_MODE,
        help="Select 'CPU' or 'GPU' mode."
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        choices=["grid", "points"],
        default=DEFAULT_X_AXIS_MODE,
        help="Select 'grid' for linear spacing or 'points' for quadratic point-count spacing."
    )
    args = parser.parse_args()

    if args.mode == "CPU":
        print(f"Generating CPU plots (x_axis={args.x_axis})...")
        generate_cpu_plot(TXT_FILES, PLOT_NAME, x_mode=args.x_axis)
    elif args.mode == "GPU":
        print(f"Generating GPU plots (x_axis={args.x_axis})...")
        generate_gpu_plot(TXT_FILES, PLOT_NAME, x_mode=args.x_axis)
