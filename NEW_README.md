# Organized Guide: ExampleBasedSamplingWithDiffusion

This document consolidates the gradient-data workflow, conditioning setup, visualization, sanity checks, and troubleshooting. It reflects the reorganized repo layout (scripts, outputs, datasets) so the legacy docs can be removed safely.

## Repo Layout (Current)

- Core training/sampling: `build_gradient_dataset.py`, `train.py`, `sample.py`
- Helper utilities: `scripts/`
- Datasets: `data/datasets/`
- Outputs (results, figures, checkpoints): `outputs/`

## Gradient Data Overview

Dataset source: `/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048`
- 40,000 PNG images (512x512)
- Black point patterns on white background
- Raw points ~2048; normalized to 1024 or 2025 (perfect squares)

Valid point counts (perfect squares for 2D):
- 1024 = 32x32
- 2025 = 45x45
- 4096 = 64x64

## Quick Start: Gradient Dataset (Unconditional)

1) Build dataset
```bash
conda run -n qmcdiffusion python build_gradient_dataset.py \
  --source /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target \
  --output data/datasets/gradient_dataset.hdf5 \
  --max-samples 1000
```

2) Train
```bash
conda run -n qmcdiffusion python train.py \
  --config config_gradient.json \
  --its 2000 \
  --tqdm True
```

3) Sample
```bash
conda run -n qmcdiffusion python sample.py \
  --config config_gradient.json \
  --model outputs/models/gradient_models/model.ckpt \
  --shape 5 2 32 32 \
  --timesteps 100 \
  --output outputs/results_gradients/samples/gradient_samples.npy
```

4) Visualize
```bash
conda run -n qmcdiffusion python scripts/visualize_generated_results.py \
  --samples outputs/results_gradients/samples/gradient_samples.npy \
  --output outputs/generated_results_visualization.png
```

## Type-Based Conditioning (9 Gradient Types)

Why: without conditioning, the model averages all types into uniform noise. Conditioning lets each type learn its own distribution.

Type order (one-hot):
```
0: Radial_Cosine_Gradient
1: Noise
2: Wave
3: Radial_Wave
4: Cosine_Gradient
5: Sinusoidal_Gradient
6: Combined_Shape
7: Linear_Gradient
8: Radial_Sinusoidal_Gradient
```

How conditioning works:
- Filename parsed: `gen_gray_<TYPE>_<N>_<N>.png`
- Type -> one-hot vector stored as HDF5 `prop` (shape: N x 9)
- Model receives `cond` and learns type-specific patterns

### Build Conditioned Dataset
```bash
conda run -n qmcdiffusion python build_gradient_dataset.py \
  --source /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target \
  --output data/datasets/gradient_dataset_conditioned.hdf5
```

### Train (Conditioned)
```bash
conda run -n qmcdiffusion python train.py \
  --config config_gradient.json \
  --its 50000 \
  --tqdm True
```

### Generate Per-Type Samples
```bash
conda run -n qmcdiffusion python scripts/generate_samples_per_type.py
```

### Compare GT vs Generated Per Type
```bash
conda run -n qmcdiffusion python scripts/compare_per_type_training_vs_generated.py \
  --dataset data/datasets/gradient_dataset_balanced.hdf5 \
  --config outputs/models/gradient_models_balanced/config.json \
  --model outputs/models/gradient_models_balanced/model.ckpt \
  --output outputs/results_balanced/compare_per_type
```

### Verify Conditioning Vectors
```bash
conda run -n qmcdiffusion python scripts/verify_conditioning.py \
  --file data/datasets/gradient_dataset_conditioned_test.hdf5
```

Expected output:
- `prop` shape is (N, 9)
- One-hot vectors map to the correct type

## Single-Image Sanity Check

Why: some gradient types include rotated/shifted variants, which still average out. This test proves the architecture works by training on a single repeated image.

```bash
scripts/single_image_test.sh /path/to/one_image.png 100 5000 1000 1024
```

Outputs:
- `outputs/results_single_image/single_image_compare.png`

## Visualization

Recommended scripts:
- `scripts/visualize_generated_results.py` (summary + individual samples)
- `scripts/compare_training_vs_generated.py` (GT vs generated)
- `scripts/visualize_gradient_samples.py` (raw vs OT-transformed)

Custom plot (example):
```python
import numpy as np
import matplotlib.pyplot as plt

samples = np.load('outputs/results_gradients/samples/gradient_samples.npy', allow_pickle=True)
points = samples[0]

plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.7, c='darkblue')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('Generated Sample 1')
plt.grid(True, alpha=0.2)
plt.show()
```

## Training Monitoring

```bash
tail -f outputs/models/gradient_models/log.txt
nvidia-smi
```

## Common Issues

- `ValueError: Can't perform OT: N != n ** D`
  - Use perfect squares for point counts (1024, 2025, 4096).

- CUDA OOM
  - Reduce batch size in config or use 32x32 instead of 45x45.

- Model outputs uniform noise
  - Train longer, or run the single-image sanity check to verify the pipeline.

## Helper Scripts

- `scripts/train_gradient_quick.sh`: one-shot gradient pipeline
- `scripts/test_gradient_integration.sh`: small integration test
- `scripts/test_conditioning_pipeline.sh`: balanced subset -> train -> per-type samples
- `scripts/generate_all_types.sh`: generate 9 types from a checkpoint
- `scripts/verify_conditioning.py`: validate HDF5 conditioning

## Outputs Location

All generated artifacts live under `outputs/`:
- `outputs/results_gradients/`, `outputs/results_balanced/`, `outputs/results_single_image/`
- `outputs/models/` for checkpoints
- `outputs/*.png` for summary figures

## Status Checklist

- Conditioning implemented and verified
- Per-type comparisons generated
- Single-image sanity check passes
- Repo reorganized (scripts/, outputs/, data/datasets/)
