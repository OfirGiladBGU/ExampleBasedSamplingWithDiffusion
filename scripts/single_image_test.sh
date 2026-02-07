#!/bin/bash

# Single-image sanity check: train on one repeated image and compare outputs.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_PATH="$1"
REPEATS=${2:-100}
TRAINING_STEPS=${3:-5000}
TIMESTEPS=${4:-1000}
POINT_SIZE=${5:-1024}

if [ -z "$IMAGE_PATH" ]; then
    echo "Usage: $0 /path/to/image.png [repeats] [training_steps] [timesteps] [point_size]"
    exit 1
fi

DATASET_FILE="$ROOT_DIR/data/datasets/single_image_dataset.hdf5"
CONFIG_FILE="$ROOT_DIR/config_single_image.json"
MODEL_DIR="$ROOT_DIR/outputs/models/single_image_models"
RESULTS_DIR="$ROOT_DIR/outputs/results_single_image"

mkdir -p "$MODEL_DIR" "$RESULTS_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Build single-image dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n qmcdiffusion python "$ROOT_DIR/scripts/build_single_image_dataset.py" \
    --image "$IMAGE_PATH" \
    --output "$DATASET_FILE" \
    --repeats "$REPEATS" \
    --point-sizes "$POINT_SIZE"

GRID_SIZE=$(python - <<PY
import math
print(int(math.sqrt($POINT_SIZE)))
PY
)

if [ $((GRID_SIZE * GRID_SIZE)) -ne $POINT_SIZE ]; then
    echo "Point size must be a perfect square. Got $POINT_SIZE."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Prepare training config"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python - <<PY
import json
import os

config_path = "config_gradient.json"
output_path = "$CONFIG_FILE"

with open(config_path, "r") as f:
    config = json.load(f)

config["path"] = "$MODEL_DIR/"
config["train"]["dataset"] = "$DATASET_FILE"
config["train"]["samplers"] = ["single_image"]
config["model"]["use_conditioning"] = False
config["model"]["cond_size"] = 0
config["train"]["batch_size"] = min(config["train"]["batch_size"], $REPEATS)

with open(output_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Wrote {output_path}")
PY

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Train model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n qmcdiffusion python "$ROOT_DIR/train.py" \
    --config "$CONFIG_FILE" \
    --its "$TRAINING_STEPS" \
    --tqdm True

MODEL_PATH="$MODEL_DIR/model.ckpt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ No checkpoint found at $MODEL_PATH"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Generate samples"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n qmcdiffusion python "$ROOT_DIR/sample.py" \
    --config "$CONFIG_FILE" \
    --model "$MODEL_PATH" \
    --shape 5 2 "$GRID_SIZE" "$GRID_SIZE" \
    --timesteps "$TIMESTEPS" \
    --output "$RESULTS_DIR/single_image_samples.npy"


echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Compare training vs generated"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

conda run -n qmcdiffusion python "$ROOT_DIR/scripts/compare_training_vs_generated.py" \
    --dataset "$DATASET_FILE" \
    --generated "$RESULTS_DIR/single_image_samples.npy" \
    --samples 5 \
    --output "$RESULTS_DIR/single_image_compare.png"

if [ -f "$RESULTS_DIR/single_image_compare.png" ]; then
    echo "✅ Saved comparison to $RESULTS_DIR/single_image_compare.png"
else
    echo "⚠️  Comparison image not found."
fi
