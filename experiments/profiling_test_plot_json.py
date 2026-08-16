import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


# Toggle this between "grid" and "points"
DEFAULT_X_AXIS_MODE = "grid" 

# --- 1. Define your JSON files here (Ensure order matches: 16 -> 32 -> 48 -> 64 -> 80 -> 96 -> 112) ---

# GPU 6000 mode (no profiler trace)
JSON_FILES = [
    "outputs/profiling_test/sample_outputs_16_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "outputs/profiling_test/sample_outputs_32_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "outputs/profiling_test/sample_outputs_48_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "outputs/profiling_test/sample_outputs_64_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "outputs/profiling_test/sample_outputs_80_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "outputs/profiling_test/sample_outputs_96_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "outputs/profiling_test/sample_outputs_112_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json"
]
DEFAULT_X_AXIS_MODE = "grid" 
PLOT_NAME = "outputs/profiling_test/z_runtime_outputs_plots/gpu_6000_reg_scaling_by_grid.png"  # Output plot filename
# DEFAULT_X_AXIS_MODE = "points" 
# PLOT_NAME = "outputs/profiling_test/z_runtime_outputs_plots/gpu_6000_reg_scaling_by_points.png"  # Output plot filename


# GPU 3090 mode (no profiler trace)
# JSON_FILES = [
#     "outputs/profiling_test/sample_outputs_16_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_32_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_48_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_64_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_80_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_96_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_112_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json"
# ]
# DEFAULT_X_AXIS_MODE = "grid" 
# PLOT_NAME = "outputs/profiling_test/z_runtime_outputs_plots/gpu_3090_reg_scaling_by_grid.png"  # Output plot filename
# DEFAULT_X_AXIS_MODE = "points" 
# PLOT_NAME = "outputs/profiling_test/z_runtime_outputs_plots/gpu_3090_reg_scaling_by_points.png"  # Output plot filename


# GPU 6000 mode (with profiler trace)
# JSON_FILES = [
#     "outputs/profiling_test/sample_outputs_16_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_32_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_48_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_64_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_80_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_96_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_112_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json"
# ]
# DEFAULT_X_AXIS_MODE = "grid" 
# PLOT_NAME = "outputs/profiling_test/z_runtime_outputs_plots/gpu_6000_scaling_by_grid.png"  # Output plot filename
# DEFAULT_X_AXIS_MODE = "points" 
# PLOT_NAME = "outputs/profiling_test/z_runtime_outputs_plots/gpu_6000_scaling_by_points.png"  # Output plot filename


# GPU 3090 mode (with profiler trace)
# JSON_FILES = [
#     "outputs/profiling_test/sample_outputs_16_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_32_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_48_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_64_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_80_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_96_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "outputs/profiling_test/sample_outputs_112_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json"
# ]
# DEFAULT_X_AXIS_MODE = "grid" 
# PLOT_NAME = "outputs/profiling_test/z_runtime_outputs_plots/gpu_3090_scaling_by_grid.png"  # Output plot filename
# DEFAULT_X_AXIS_MODE = "points" 
# PLOT_NAME = "outputs/profiling_test/z_runtime_outputs_plots/gpu_3090_scaling_by_points.png"  # Output plot filename


def normalize_key(key):
    """Harmonizes variations in key names across different run versions."""
    key = key.lower()
    if "controlnet" in key and "forward" in key:
        return "controlnet.forward_total"
    if "unet" in key and "forward" in key:
        return "unet.forward_total"
    return key

def generate_scaling_plot(json_paths, plot_name, x_mode="grid"):
    grid_sizes = []
    labels = []
    
    # Store times for each normalized key across all sizes
    data_history = defaultdict(list)
    
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        
        grid = data["grid_size"]
        grid_sizes.append(grid)
        labels.append(f"{grid}x{grid}\n({grid**2} pts)")
        
        comps = data.get("components", {})
        
        current_file_data = {}
        for category, contents in comps.items():
            if isinstance(contents, dict):
                for k, v in contents.items():
                    current_file_data[normalize_key(k)] = v
            else:
                current_file_data[normalize_key(category)] = contents
        
        if len(grid_sizes) == 1:
            for k, v in current_file_data.items():
                data_history[k].append(v)
        else:
            for k in list(data_history.keys()):
                val = current_file_data.get(k, 0.0)
                data_history[k].append(val)

    # --- 2. Setup Plotting Mode & Dynamic Figsize ---
    base_height = 9
    
    if x_mode == "grid":
        # Standard categorical spacing needs less horizontal room
        fig_width = max(10, len(grid_sizes) * 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, base_height), dpi=150)
        
        x = grid_sizes  
        ax.set_xlabel('Grid Size Resolution', fontsize=12, fontweight='bold')
        ax.set_title('Component Scaling by Grid Size', fontsize=15, fontweight='bold')
        rotation = 0
        ha = 'center'
        
    elif x_mode == "points":
        # Proportional spacing requires a wider canvas to stretch the lower values apart
        fig_width = max(10, len(grid_sizes) * 3.5)
        fig, ax = plt.subplots(figsize=(fig_width, base_height), dpi=150)
        
        x = [g**2 for g in grid_sizes]  
        ax.set_xlabel('Number of Points (Grid Size × Grid Size)', fontsize=12, fontweight='bold')
        ax.set_title('Component Scaling by Point Count', fontsize=15, fontweight='bold')
        rotation = 15  # Angle the text to prevent bounding box collision
        ha = 'right'   # Align to the tick mark
        
    else:
        raise ValueError("x_mode must be 'grid' or 'points'")

    ax.set_ylabel('Inference Time (Seconds)', fontsize=12, fontweight='bold')

    # --- 3. Plotting Logic ---
    sorted_keys = sorted(data_history.keys())
    
    for key in sorted_keys:
        if key == "p_mean_variance.model_forward":
            continue  # Skip this key if it exists, as it's not a direct forward time
        
        y_values = data_history[key]
        
        if len(y_values) != len(grid_sizes) or sum(y_values) <= 1e-6:
            continue
            
        is_macro = "total" in key or key in ["model_forward", "unet_forward"]
        linewidth = 3.5 if is_macro else 1.5
        linestyle = '--' if is_macro else '-'
        alpha = 1.0 if is_macro else 0.75
        
        ax.plot(x, y_values, marker='o', markersize=6, 
                linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=key)

    # --- 4. Formatting ---
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=rotation, ha=ha)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=2)
    
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    
    plt.tight_layout()
    Path(plot_name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_name, dpi=300)
    print(f"Saved plot to {plot_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate component scaling plots from JSON logs.")
    parser.add_argument(
        "--x_axis", 
        type=str, 
        choices=["grid", "points"], 
        default=DEFAULT_X_AXIS_MODE,
        help="Select 'grid' for linear spacing or 'points' for quadratic point-count spacing."
    )
    args = parser.parse_args()
    
    generate_scaling_plot(JSON_FILES, PLOT_NAME, x_mode=args.x_axis)
