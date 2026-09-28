import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


# Compact figure size (inches): the same as the txt (profiler) plots, so text renders at the
# same size when the plots are placed side by side. The legend is inside the axes now.
FIG_WIDTH = 9.0
FIG_HEIGHT = 4.0
# Components whose time stays below this (seconds) at every grid size are merged into a
# single summed "Others" line, so only the interesting high-time model blocks stand out.
MERGE_THRESHOLD = 1.0

# Titles / axis labels / legend
SHOW_TITLE = False             # the file name already says what the plot is
X_LABEL = "Points Budget"
LEGEND_LOC = "upper left"      # legend inset inside the axes
LEGEND_NCOL = 1                # one entry per row
LEGEND_FONTSIZE = 11           # same as the txt (profiler) plots
LEGEND_GAP = 0.04              # min gap (fraction of axes height) between the legend box and the lines under it


# Toggle this between "grid" and "points"
DEFAULT_X_AXIS_MODE = "grid" 

# --- 1. Define your JSON files here (Ensure order matches: 16 -> 32 -> 48 -> 64 -> 80 -> 96 -> 112) ---

# GPU 6000 mode (no profiler trace)
JSON_FILES = [
    "experiments/outputs/profiling_test/sample_outputs_16_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "experiments/outputs/profiling_test/sample_outputs_32_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "experiments/outputs/profiling_test/sample_outputs_48_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "experiments/outputs/profiling_test/sample_outputs_64_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "experiments/outputs/profiling_test/sample_outputs_80_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "experiments/outputs/profiling_test/sample_outputs_96_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
    "experiments/outputs/profiling_test/sample_outputs_112_gpu_6000_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json"
]
DEFAULT_X_AXIS_MODE = "grid" 
PLOT_NAME = "experiments/outputs/profiling_test/z_runtime_outputs_plots/gpu_6000_reg_scaling_by_grid.png"  # Output plot filename
# DEFAULT_X_AXIS_MODE = "points" 
# PLOT_NAME = "experiments/outputs/profiling_test/z_runtime_outputs_plots/gpu_6000_reg_scaling_by_points.png"  # Output plot filename


# GPU 3090 mode (no profiler trace)
# JSON_FILES = [
#     "experiments/outputs/profiling_test/sample_outputs_16_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_32_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_48_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_64_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_80_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_96_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_112_gpu_3090_reg/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json"
# ]
# DEFAULT_X_AXIS_MODE = "grid" 
# PLOT_NAME = "experiments/outputs/profiling_test/z_runtime_outputs_plots/gpu_3090_reg_scaling_by_grid.png"  # Output plot filename
# DEFAULT_X_AXIS_MODE = "points" 
# PLOT_NAME = "experiments/outputs/profiling_test/z_runtime_outputs_plots/gpu_3090_reg_scaling_by_points.png"  # Output plot filename


# GPU 6000 mode (with profiler trace)
# JSON_FILES = [
#     "experiments/outputs/profiling_test/sample_outputs_16_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_32_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_48_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_64_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_80_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_96_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_112_gpu_6000/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json"
# ]
# DEFAULT_X_AXIS_MODE = "grid" 
# PLOT_NAME = "experiments/outputs/profiling_test/z_runtime_outputs_plots/gpu_6000_scaling_by_grid.png"  # Output plot filename
# DEFAULT_X_AXIS_MODE = "points" 
# PLOT_NAME = "experiments/outputs/profiling_test/z_runtime_outputs_plots/gpu_6000_scaling_by_points.png"  # Output plot filename


# GPU 3090 mode (with profiler trace)
# JSON_FILES = [
#     "experiments/outputs/profiling_test/sample_outputs_16_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_32_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_48_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_64_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_80_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_96_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json",
#     "experiments/outputs/profiling_test/sample_outputs_112_gpu_3090/emoji-one_4_monkey/timestamps/emoji-one_4_monkey_full.json"
# ]
# DEFAULT_X_AXIS_MODE = "grid" 
# PLOT_NAME = "experiments/outputs/profiling_test/z_runtime_outputs_plots/gpu_3090_scaling_by_grid.png"  # Output plot filename
# DEFAULT_X_AXIS_MODE = "points" 
# PLOT_NAME = "experiments/outputs/profiling_test/z_runtime_outputs_plots/gpu_3090_scaling_by_points.png"  # Output plot filename


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
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=150)
        
        x = grid_sizes
        ax.set_xlabel(X_LABEL, fontsize=12, fontweight='bold')
        if SHOW_TITLE:
            ax.set_title('Component Scaling by Grid Size', fontsize=15, fontweight='bold')
        rotation = 0
        ha = 'center'
        
    elif x_mode == "points":
        # Proportional spacing requires a wider canvas to stretch the lower values apart
        fig_width = max(10, len(grid_sizes) * 3.5)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=150)
        
        x = [g**2 for g in grid_sizes]
        ax.set_xlabel(X_LABEL, fontsize=12, fontweight='bold')
        if SHOW_TITLE:
            ax.set_title('Component Scaling by Point Count', fontsize=15, fontweight='bold')
        rotation = 15  # Angle the text to prevent bounding box collision
        ha = 'right'   # Align to the tick mark
        
    else:
        raise ValueError("x_mode must be 'grid' or 'points'")

    ax.set_ylabel('Inference Time (Seconds)', fontsize=12, fontweight='bold')

    # --- 3. Plotting Logic ---
    sorted_keys = sorted(data_history.keys())
    
    # Merge the many small, near-constant components (max < MERGE_THRESHOLD s) into ONE
    # summed "Others" line, so only the high-time blocks (the model blocks) stand out.
    others = None
    others_count = 0
    significant = []
    for key in sorted_keys:
        if key == "p_mean_variance.model_forward":
            continue  # not a direct forward time
        y_values = data_history[key]
        if len(y_values) != len(grid_sizes) or sum(y_values) <= 1e-6:
            continue
        if max(y_values) < MERGE_THRESHOLD:
            others = list(y_values) if others is None else [a + b for a, b in zip(others, y_values)]
            others_count += 1
        else:
            significant.append((key, y_values))

    for key, y_values in significant:
        is_macro = "total" in key or key in ["model_forward", "unet_forward"]
        ax.plot(x, y_values, marker='o', markersize=6,
                linewidth=3.5 if is_macro else 1.5,
                linestyle='--' if is_macro else '-',
                alpha=1.0 if is_macro else 0.75, label=key)

    if others is not None:
        ax.plot(x, others, marker='s', markersize=6, linewidth=1.8, linestyle=':',
                alpha=0.9, color='gray',
                # label=f"others (< {MERGE_THRESHOLD:g}s each, sum of {others_count})")
                label=f"others (< {MERGE_THRESHOLD:g}s each)")

    # --- 4. Formatting ---
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
    
    # Legend inset in the plot. Its real box is measured and the y-axis top is raised just
    # enough that no line under the box reaches it (the box size depends on the number of
    # entries, the font and the figure size, so it is measured rather than guessed).
    series = [v for _, v in significant] + ([others] if others is not None else [])
    ymax = max(max(v) for v in series)
    y0 = ax.get_ylim()[0]
    ax.set_ylim(bottom=y0, top=ymax * 1.05)
    leg = ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONTSIZE, ncol=LEGEND_NCOL, framealpha=0.9)
    plt.tight_layout()                      # final axes size, so the measured box matches the saved figure
    fig.canvas.draw()
    box = leg.get_window_extent().transformed(ax.transAxes.inverted())   # axes fractions
    x_lo, x_hi = ax.get_xlim()
    x_right = x_lo + box.x1 * (x_hi - x_lo)                               # data x under the box
    under = [yv for v in series for xv, yv in zip(x, v) if xv <= x_right] or [y0]
    free = box.y0 - LEGEND_GAP                                            # usable fraction below the box
    top = y0 + (max(under) - y0) / max(free, 0.05)
    ax.set_ylim(bottom=y0, top=max(top, ymax * 1.05))
    
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    
    plt.tight_layout()
    Path(plot_name).parent.mkdir(parents=True, exist_ok=True)
    pdf_name = str(Path(plot_name).with_suffix(".pdf"))
    plt.savefig(plot_name, dpi=300)
    plt.savefig(pdf_name)  # vector PDF (scalable, high quality)
    print(f"Saved plot to {plot_name}")
    print(f"Saved plot to {pdf_name}")

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
