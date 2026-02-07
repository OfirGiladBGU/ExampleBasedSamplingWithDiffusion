# Training on Gradient Point Set Data

## Overview

This guide explains how to use the **ExampleBasedSamplingWithDiffusion** model to train on gradient point pattern data from `/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048`.

The data contains:
- **40,000 PNG images** (512×512 pixels)
- **Target images**: Black point patterns on white background
- **Point count**: ~2048 points per image (normalized to 2025 = 45²)

## Quick Start

### 1. One-Command Training (Recommended)

```bash
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
bash train_gradient_quick.sh
```

This script automatically:
1. Builds HDF5 database from 1000 gradient images
2. Trains for 2000 steps (or 30 minutes)
3. Generates sample point sets
4. Visualizes results

### 2. Manual Steps

#### Step 1: Build Dataset

```bash
conda run -n qmcdiffusion python build_gradient_dataset.py \
    --source /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target \
    --output gradient_dataset.hdf5 \
    --max-samples 1000 \
    --threshold 128
```

**Arguments:**
- `--source`: Input directory with PNG images (required)
- `--output`: Output HDF5 file (default: `gradient_dataset.hdf5`)
- `--max-samples`: Limit number of samples to process (default: all)
- `--threshold`: Pixel threshold for detecting black points (0-255)
- `--point-sizes`: Valid point set sizes, must be perfect squares for 2D (default: `[1024, 2025]`)

**Output:**
```
gradient_dataset.hdf5
└── gradients/
    └── scale_2025/
        ├── data       (10 samples × 2025 points × 2 dims)
        ├── data_t     (10 samples × 45 × 45 × 2 for OT transform)
        └── prop       (properties/conditioning values)
```

#### Step 2: Train Model

```bash
conda run -n qmcdiffusion python train.py \
    --config config_gradient.json \
    --its 2000 \
    --time 30 \
    --tqdm True
```

**Arguments:**
- `--config`: Config file (default: `config_gradient.json`)
- `--its`: Number of optimization steps
- `--time`: Time budget in minutes
- `--tqdm`: Show progress bar

**Training Configuration** (`config_gradient.json`):
```json
{
    "path": "gradient_models/",
    "model": {
        "num_channels": 2,
        "ch": 64,
        "out_ch": 2,
        "ch_mult": [1, 2, 2],
        "batch_size": 32,
        "lr": 1e-4
    }
}
```

#### Step 3: Sample from Model

```bash
conda run -n qmcdiffusion python sample.py \
    --config config_gradient.json \
    --model gradient_models/model_latest.ckpt \
    --shape 5 2 45 45 \
    --timesteps 100 \
    --output results/gradient_samples.npy
```

**Arguments:**
- `--shape`: Output shape `[batch, dim, n1, n2, ...]` (5th samples of 2025 points each)
- `--timesteps`: Number of diffusion steps (more = better quality, slower)
- `--output`: Output .npy file

## Data Preparation Details

### Point Extraction

The `build_gradient_dataset.py` script:

1. **Loads images** as 512×512 PNG (1-bit or 8-bit grayscale)
2. **Detects black points** using threshold (default: pixel_value < 128)
3. **Normalizes coordinates** to [0, 1] range based on image dimensions
4. **Groups by point count** for efficient batching
5. **Normalizes point counts** to perfect squares (2048 → 2025 = 45²)
6. **Applies optimal transport** transform for model input

### Valid Point Counts (Perfect Squares)

For 2D inputs, valid sizes are perfect squares:
- 1024 = 32²
- 2025 = 45²  ← Used for gradient data
- 4096 = 64²
- 40000 = 200² (very large, slow)

To use different sizes, modify `--point-sizes`:
```bash
python build_gradient_dataset.py --point-sizes 1024 4096
```

## Configuration Files

### `config_gradient.json`

Key settings for gradient data:
```json
{
    "model": {
        "ch": 64          // Base channel count
        "ch_mult": [1, 2, 2]  // Channel multipliers (determines depth)
        "dropout": 0.1    // Dropout probability
    },
    "diffusion": {
        "betas": {        // Beta schedule for noise
            "min": 1e-4,
            "max": 1e-2,
            "count": 1000
        }
    },
    "train": {
        "batch_size": 32,
        "lr": 1e-4,       // Learning rate
        "ema_decay": 0.995    // Exponential moving average
    }
}
```

## Monitoring Training

### During Training

```bash
# Watch training logs
tail -f gradient_models/log.txt

# Monitor GPU usage
nvidia-smi
```

### Evaluation

Samples are saved during training in `eval/` directory:
```
eval/
├── step_1000/
│   ├── sample_shape_45_45.npy
│   └── ...
└── step_2000/
    └── ...
```

### Visualize Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Load samples
samples = np.load('results/gradient_samples.npy')

# Plot first sample
plt.figure(figsize=(8, 8))
plt.gca().set_aspect('equal')
plt.scatter(*samples[0].T, s=2, alpha=0.6)
plt.title('Generated Point Pattern')
plt.show()
```

## Common Issues & Solutions

### Issue: `ValueError: Can't perform OT: N != n ** D`

**Cause**: Point count is not a perfect square.

**Solution**: Use `--point-sizes` to specify valid sizes:
```bash
python build_gradient_dataset.py --point-sizes 1024 2025 4096
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size in config:
```json
"train": {
    "batch_size": 16  // Reduce from 32
}
```

### Issue: Training too slow

**Solution**: Use GPU allocation:
```bash
salloc --partition=gpu_nodes --gpus=rtx_6000:1 --time=2-00:00:00 --mem=100G --cpus-per-task=8
```

Then run training in allocated node.

## Full Training on All 40,000 Images

For production training:

```bash
# Build full dataset (takes ~5-10 minutes)
conda run -n qmcdiffusion python build_gradient_dataset.py \
    --source /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target \
    --output gradient_dataset_full.hdf5

# Create production config
cp config_gradient.json config_gradient_full.json

# Update config_gradient_full.json with:
# - "path": "gradient_models_full/"
# - "train": {"batch_size": 64, "lr": 1e-4}

# Request long GPU allocation
bash /home/ofirgila/scripts/pycharm_rtx_6000_32h.sh

# Train in allocated node
conda run -n qmcdiffusion python train.py \
    --config config_gradient_full.json \
    --its 1000000 \
    --time 1920  # 32 hours
```

## Key Parameters to Tune

| Parameter | Impact | Default | Range |
|-----------|--------|---------|-------|
| `ch` | Model capacity | 64 | 32-256 |
| `batch_size` | Training stability | 32 | 8-128 |
| `lr` | Learning speed | 1e-4 | 1e-5 to 1e-3 |
| `ema_decay` | EMA smoothness | 0.995 | 0.99-0.9995 |
| `timesteps` | Sampling quality | 100 | 10-1000 |

## Files Created

```
ExampleBasedSamplingWithDiffusion/
├── build_gradient_dataset.py     # Dataset builder
├── config_gradient.json          # Training config
├── train_gradient_quick.sh       # Quick training script
├── gradient_dataset.hdf5         # Built database (auto-created)
├── gradient_models/              # Model checkpoints (auto-created)
│   ├── model_0.ckpt
│   ├── model_100.ckpt
│   └── ...
├── eval/                         # Evaluation samples (auto-created)
├── results/                      # Output samples
│   └── gradient_samples.npy
└── test_gradient_dataset.hdf5    # Test dataset
```

## Next Steps

1. Run quick training to validate: `bash train_gradient_quick.sh`
2. Adjust hyperparameters based on results
3. Train on full dataset (40,000 images) for production
4. Fine-tune for specific use cases

## Environment

- **Python**: 3.8+
- **PyTorch**: GPU-enabled
- **Other**: POT (Optimal Transport), h5py, tensorboardX, tqdm

Activate environment:
```bash
conda activate qmcdiffusion
```
