#!/bin/bash

# Generate samples for all 9 gradient types
# Usage: bash generate_all_types.sh <model_checkpoint> [output_dir]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL=${1:-$ROOT_DIR/outputs/models/gradient_models_balanced/model.ckpt}
OUTPUT_DIR=${2:-$ROOT_DIR/outputs/results_by_type}

# Gradient types (must match build_gradient_dataset.py)
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

# Conditioning vectors (one-hot, 9 dimensions)
CONDITIONS=(
    "1 0 0 0 0 0 0 0 0"  # 0: Radial_Cosine_Gradient
    "0 1 0 0 0 0 0 0 0"  # 1: Noise
    "0 0 1 0 0 0 0 0 0"  # 2: Wave
    "0 0 0 1 0 0 0 0 0"  # 3: Radial_Wave
    "0 0 0 0 1 0 0 0 0"  # 4: Cosine_Gradient
    "0 0 0 0 0 1 0 0 0"  # 5: Sinusoidal_Gradient
    "0 0 0 0 0 0 1 0 0"  # 6: Combined_Shape
    "0 0 0 0 0 0 0 1 0"  # 7: Linear_Gradient
    "0 0 0 0 0 0 0 0 1"  # 8: Radial_Sinusoidal_Gradient
)

echo "=========================================="
echo "🎨 Generating samples for all 9 types"
echo "=========================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    echo "Available models:"
    ls -lh "$ROOT_DIR/outputs/models"/*/*.ckpt 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Generate for each type
for i in "${!TYPES[@]}"; do
    TYPE_NAME="${TYPES[$i]}"
    COND="${CONDITIONS[$i]}"
    OUTPUT_FILE="$OUTPUT_DIR/type_${i}_${TYPE_NAME}_samples.npy"
    
    echo "[$((i+1))/9] Generating $TYPE_NAME..."
    
    conda run -n qmcdiffusion python "$ROOT_DIR/sample.py" \
        --config "$ROOT_DIR/config_gradient.json" \
        --model "$MODEL" \
        --shape 5 2 32 32 \
        --cond $COND \
        --timesteps 100 \
        --output "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "      ✅ Saved to: $OUTPUT_FILE"
    else
        echo "      ❌ Failed to generate samples"
    fi
done

echo ""
echo "=========================================="
echo "✅ Generation complete!"
echo "=========================================="
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.npy 2>/dev/null
echo ""
echo "Visualize with:"
echo "  conda run -n qmcdiffusion python $ROOT_DIR/scripts/visualize_generated_results.py --samples <file>"
