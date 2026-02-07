#!/bin/bash
#
# Quick training script for gradient dataset
# This script:
# 1. Builds HDF5 database from gradient images (1000 samples for testing)
# 2. Trains the model for 2000 steps
# 3. Generates sample point sets
#

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /home/ofirgila/.bashrc
conda activate qmcdiffusion

echo "===== Gradient Point Set Training ====="
echo ""

# Configuration
DATASET_SIZE=1000
CONFIG_FILE="$ROOT_DIR/config_gradient.json"
DATABASE_FILE="$ROOT_DIR/data/datasets/gradient_dataset.hdf5"
TRAIN_STEPS=2000
TRAIN_TIME=30  # minutes
OUTPUT_DIR="$ROOT_DIR/outputs/results_gradients"
MODEL_DIR="$OUTPUT_DIR/models"
EVAL_DIR="$OUTPUT_DIR/eval"
RESULTS_DIR="$OUTPUT_DIR/samples"

echo "[1/4] Creating output directories..."
mkdir -p $MODEL_DIR
mkdir -p $EVAL_DIR
mkdir -p $RESULTS_DIR

echo "[2/4] Building HDF5 dataset from gradient images..."
echo "      (extracting points from 1000 target images)"
python "$ROOT_DIR/build_gradient_dataset.py" \
    --source /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target \
    --output $DATABASE_FILE \
    --max-samples $DATASET_SIZE \
    --threshold 128

echo ""
echo "[3/4] Training diffusion model..."
echo "      (2000 steps or 30 minutes, whichever comes first)"

# Temporarily update config to output to our directory
CONFIG_TMP="$OUTPUT_DIR/config_gradient_temp.json"
python -c "
import json
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
cfg['path'] = '$MODEL_DIR/'
cfg['eval']['directory'] = '$EVAL_DIR/'
import os
os.makedirs(os.path.dirname('$CONFIG_TMP'), exist_ok=True)
with open('$CONFIG_TMP', 'w') as f:
    json.dump(cfg, f, indent=4)
"

python "$ROOT_DIR/train.py" \
    --config $CONFIG_TMP \
    --its $TRAIN_STEPS \
    --time $TRAIN_TIME \
    --tqdm True

echo ""
echo "[4/4] Sampling from trained model..."
# Find the latest checkpoint
LATEST_CKPT=$(ls -t $MODEL_DIR/model*.ckpt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $MODEL_DIR"
    echo "Files in $MODEL_DIR:"
    ls -la $MODEL_DIR/
    exit 1
fi

echo "      Using checkpoint: $LATEST_CKPT"
python "$ROOT_DIR/sample.py" \
    --config $CONFIG_TMP \
    --model $LATEST_CKPT \
    --shape 5 2 32 32 \
    --timesteps 100 \
    --output $RESULTS_DIR/gradient_samples.npy

echo ""
echo "===== Training Complete ====="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_DIR/"
echo "  ├── models/          - Trained model checkpoints"
echo "  ├── eval/            - Evaluation logs"
echo "  └── samples/         - Generated point set samples"
echo ""
echo "Main output:"
echo "  - Trained model: $LATEST_CKPT"
echo "  - Sample output: $RESULTS_DIR/gradient_samples.npy"
echo ""
echo "To visualize samples:"
echo "  python -c \"import numpy as np; import matplotlib.pyplot as plt; samples = np.load('$RESULTS_DIR/gradient_samples.npy'); plt.gca().set_aspect('equal'); [plt.scatter(*s.T, s=1, alpha=0.5) for s in samples]; plt.show()\""
