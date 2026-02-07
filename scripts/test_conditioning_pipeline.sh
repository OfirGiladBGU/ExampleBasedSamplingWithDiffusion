#!/bin/bash

# Test pipeline: Build balanced dataset, train, and evaluate
# Tests the model on 200 samples per gradient type (1800 total)

set -e  # Exit on error

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SUBSET_DIR="/tmp/gradient_balanced_subset"
DATASET_FILE="$ROOT_DIR/data/datasets/gradient_dataset_balanced.hdf5"
MODEL_DIR="$ROOT_DIR/outputs/models/gradient_models_balanced"
RESULTS_DIR="$ROOT_DIR/outputs/results_balanced"

SAMPLES_PER_TYPE=${1:-200}
TRAINING_STEPS=${2:-10000}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     TESTING CONDITIONING: Balanced Dataset Pipeline        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Samples per type: $SAMPLES_PER_TYPE"
echo "  Total samples: ~$((SAMPLES_PER_TYPE * 9))"
echo "  Training steps: $TRAINING_STEPS"
echo ""

# Step 1: Create balanced subset
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Creating balanced subset..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n qmcdiffusion python "$ROOT_DIR/scripts/build_balanced_subset.py" \
    --source /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target \
    --output "$SUBSET_DIR" \
    --samples-per-type $SAMPLES_PER_TYPE

echo "✅ Balanced subset created in: $SUBSET_DIR"
echo ""

# Step 2: Build HDF5 dataset with conditioning
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Building HDF5 dataset with conditioning..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n qmcdiffusion python "$ROOT_DIR/build_gradient_dataset.py" \
    --source "$SUBSET_DIR" \
    --output "$DATASET_FILE"

echo "✅ Dataset built: $DATASET_FILE"

# Verify conditioning
echo ""
echo "Verifying conditioning..."
conda run -n qmcdiffusion python "$ROOT_DIR/scripts/verify_conditioning.py" --file "$DATASET_FILE"
echo ""

# Step 3: Train model
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Training model ($TRAINING_STEPS steps)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update config to use balanced dataset
cp "$ROOT_DIR/config_gradient.json" "$ROOT_DIR/config_gradient_balanced.json"
sed -i "s|\"dataset\": \".*\"|\"dataset\": \"$DATASET_FILE\"|" "$ROOT_DIR/config_gradient_balanced.json"
sed -i "s|\"path\": \".*\"|\"path\": \"$MODEL_DIR/\"|" "$ROOT_DIR/config_gradient_balanced.json"

echo "Training (this may take 10-30 minutes on GPU)..."
conda run -n qmcdiffusion python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/config_gradient_balanced.json" \
    --its $TRAINING_STEPS \
    --tqdm True

echo "✅ Training complete!"
echo ""

# Step 4: Find latest checkpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Generating samples for each gradient type..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$MODEL_DIR/model.ckpt" ]; then
    LATEST_MODEL="$MODEL_DIR/model.ckpt"
else
    LATEST_MODEL=$(ls -t "$MODEL_DIR"/model_*.ckpt 2>/dev/null | head -1)
fi

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ Error: No model checkpoint found in $MODEL_DIR"
    exit 1
fi

echo "Using model: $LATEST_MODEL"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Generate samples for each type
TYPES=(
    "Radial_Cosine_Gradient"
    "Noise"
    "Wave"
    "Radial_Wave"
    "Cosine_Gradient"
    "Sinusoidal_Gradient"
    "Combined_Shape"
    "Linear_Gradient"
    "Radial_Sinusoidal_Gradient"
)

CONDITIONS=(
    "1 0 0 0 0 0 0 0 0"
    "0 1 0 0 0 0 0 0 0"
    "0 0 1 0 0 0 0 0 0"
    "0 0 0 1 0 0 0 0 0"
    "0 0 0 0 1 0 0 0 0"
    "0 0 0 0 0 1 0 0 0"
    "0 0 0 0 0 0 1 0 0"
    "0 0 0 0 0 0 0 1 0"
    "0 0 0 0 0 0 0 0 1"
)

echo "Generating 10 samples per type..."
echo ""

for i in "${!TYPES[@]}"; do
    TYPE_NAME="${TYPES[$i]}"
    COND="${CONDITIONS[$i]}"
    OUTPUT_FILE="$RESULTS_DIR/type_${i}_${TYPE_NAME}.npy"
    
    echo -n "  [$((i+1))/9] $TYPE_NAME ... "
    
    conda run -n qmcdiffusion python "$ROOT_DIR/sample.py" \
        --config "$ROOT_DIR/config_gradient_balanced.json" \
        --model "$LATEST_MODEL" \
        --shape 10 2 32 32 \
        --cond $COND \
        --timesteps 100 \
        --output "$OUTPUT_FILE" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Saved"
    else
        echo "❌ Failed"
    fi
done

echo ""
echo "✅ Sample generation complete!"
echo ""

# Step 5: Visualize results
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Creating visualizations..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create summary visualization showing all types
python << PYTHONEOF
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path(r"${RESULTS_DIR}")

# Load all samples
types_data = {}
type_names = [
    "Radial_Cosine_Gradient",
    "Noise",
    "Wave",
    "Radial_Wave",
    "Cosine_Gradient",
    "Sinusoidal_Gradient",
    "Combined_Shape",
    "Linear_Gradient",
    "Radial_Sinusoidal_Gradient"
]

for i, name in enumerate(type_names):
    file = results_dir / f"type_{i}_{name}.npy"
    if file.exists():
        samples = np.load(file, allow_pickle=True)
        types_data[name] = samples
        print(f"✓ Loaded {name}: {len(samples)} samples")
    else:
        print(f"✗ Missing {name}")

print(f"\nLoaded {len(types_data)} types")

# Create visualization: 3 rows x 3 columns for 9 types
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Generated Samples per Gradient Type\n(10 samples each)', 
             fontsize=16, fontweight='bold')

for idx, (type_name, samples) in enumerate(types_data.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Plot first sample of this type
    if len(samples) > 0:
        sample = samples[0]
        ax.scatter(sample[:, 0], sample[:, 1], s=3, alpha=0.7, c='darkblue')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(type_name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(results_dir / 'summary_all_types.png', dpi=200, bbox_inches='tight')
print("\n✓ Summary visualization saved:", results_dir / 'summary_all_types.png')
plt.close()

# Create per-type detail visualizations
for type_name, samples in types_data.items():
    if len(samples) == 0:
        continue
    
    # Show 4 samples from this type
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'{type_name}\n(4 of {len(samples)} samples)', 
                 fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(samples):
            sample = samples[idx]
            ax.scatter(sample[:, 0], sample[:, 1], s=5, alpha=0.7, c='darkblue')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Sample {idx+1}', fontsize=10)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    safe_name = type_name.replace(' ', '_')
    plt.savefig(results_dir / f'detailed_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("✓ Per-type visualizations created")

PYTHONEOF

echo ""
echo "✅ Visualizations created!"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Dataset:"
echo "   Files: $DATASET_FILE"
echo "   Samples: $((SAMPLES_PER_TYPE * 9)) total"
echo "   Per type: $SAMPLES_PER_TYPE"
echo ""
echo "✅ Model:"
echo "   Path: $MODEL_DIR"
echo "   Latest: $LATEST_MODEL"
echo "   Training steps: $TRAINING_STEPS"
echo ""
echo "✅ Results:"
echo "   Samples: $RESULTS_DIR/ (9 types × 10 samples)"
echo "   Visualizations:"
echo "     - $RESULTS_DIR/summary_all_types.png"
echo "     - $RESULTS_DIR/detailed_*.png"
echo ""

# Analyze results
echo "═══════════════════════════════════════════════════════════════"
echo "ANALYSIS: Does each type generate correctly?"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Review the visualizations:"
echo "  1. summary_all_types.png - Quick overview of all 9 types"
echo "  2. detailed_*.png - 4 samples per type for detailed analysis"
echo ""
echo "Questions to answer:"
echo "  ✓ Does each type look visually distinct?"
echo "  ✓ Do Linear gradients show lines?"
echo "  ✓ Do Radial patterns show circles?"
echo "  ✓ Does Noise look random?"
echo "  ✓ Do Wave patterns show undulations?"
echo ""
echo "If YES to all: Conditioning is working! ✨"
echo "If NO: May need more training steps"
echo ""

echo "═══════════════════════════════════════════════════════════════"
