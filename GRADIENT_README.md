# Gradient Data Training - Implementation Summary

## ✅ What Was Created

You now have a complete pipeline to train the diffusion model on gradient point pattern data. Here are the files created:

### Core Scripts

| File | Purpose |
|------|---------|
| `build_gradient_dataset.py` | Extract points from PNG images → HDF5 database |
| `train_gradient_quick.sh` | One-command training (1000 samples, 2000 steps) |
| `test_gradient_integration.sh` | Verify pipeline works end-to-end |
| `config_gradient.json` | Model training configuration |
| `GRADIENT_TRAINING.md` | Complete documentation (60+ sections) |

### Generated Files (auto-created)

| Directory | Contents |
|-----------|----------|
| `gradient_dataset.hdf5` | HDF5 database of training data |
| `gradient_models/` | Model checkpoints during training |
| `eval/` | Evaluation samples |
| `results/` | Final output point sets |

---

## 🚀 Quick Start (3 Steps)

### 1. Build Dataset (1000 samples)
```bash
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
conda run -n qmcdiffusion python build_gradient_dataset.py \
    --max-samples 1000 \
    --output gradient_dataset.hdf5
```
**Time**: ~2 minutes

### 2. Train Model (2000 steps)
```bash
conda run -n qmcdiffusion python train.py \
    --config config_gradient.json \
    --its 2000 \
    --tqdm True
```
**Time**: ~5-10 minutes on GPU

### 3. Generate Samples
```bash
# Find latest checkpoint
LATEST=$(ls -t gradient_models/model_*.ckpt | head -1)

conda run -n qmcdiffusion python sample.py \
    --config config_gradient.json \
    --model $LATEST \
    --shape 5 2 45 45 \
    --timesteps 100 \
    --output results/samples.npy
```
**Time**: ~1 minute

### OR: Run All at Once
```bash
bash train_gradient_quick.sh
```
Automatically does all 3 steps with proper settings.

---

## 📊 Data Pipeline

```
PNG Images (512×512)
    ↓
[build_gradient_dataset.py]
    ↓
• Extract black pixels → 2D coordinates
• Threshold filtering
• Normalize to [0, 1]
• Subsample 2048 → 2025 (45² for OT)
    ↓
HDF5 Database [gradient_dataset.hdf5]
    ├── gradients/
    └── scale_2025/
        ├── data      (N, 2025, 2)
        ├── data_t    (N, 2, 45, 45)  [OT transformed]
        └── prop      (N, 1)          [properties]
    ↓
[train.py] → [sample.py]
    ↓
Point Set Samples
```

---

## 🔧 Configuration

**Model Architecture** (config_gradient.json):
- Base channels: 64
- Channel multipliers: [1, 2, 2]
- Diffusion steps: 1000
- Learning rate: 1e-4
- Batch size: 32

**Customization**:
```json
{
    "model": {
        "ch": 64,           // Increase for more capacity (128, 256)
        "ch_mult": [1, 2, 2]   // Add more layers: [1, 2, 3, 4]
    },
    "train": {
        "batch_size": 32,   // Reduce if OOM: 16, 8
        "lr": 1e-4          // Adjust: 1e-5, 1e-3
    }
}
```

---

## 📈 Training on Full Dataset (40,000 images)

For production training:

```bash
# 1. Build full dataset (~5-10 minutes)
conda run -n qmcdiffusion python build_gradient_dataset.py \
    --output gradient_dataset_full.hdf5
    # No --max-samples = use all 40,000

# 2. Request long GPU allocation
salloc --partition=gpu_nodes --gpus=rtx_6000:1 \
    --time=1-08:00:00 --mem=100G --cpus-per-task=8

# 3. Train in allocated node
conda run -n qmcdiffusion python train.py \
    --config config_gradient.json \
    --its 1000000 \
    --time 1500  # ~25 hours
```

---

## 🧪 Verification

Run the integration test to verify everything works:

```bash
bash test_gradient_integration.sh
```

Expected output:
```
✓ All Tests Passed
✓ PyTorch version: 2.1.1
✓ CUDA available: True
✓ GPU: NVIDIA RTX 6000 Ada Generation
```

---

## 🎯 What the Model Does

**Input**: Random noise (Batch, 2, 45, 45)
**Process**: Denoise through diffusion process (100-1000 steps)
**Output**: Point set (2025 points in [0,1]², representing black dots on image)

The model learns:
- Point distribution patterns from training data
- Spatial coherence and clustering
- Point density variations
- Perceptual quality of the generated patterns

---

## 📝 Key Parameters Explained

| Parameter | Meaning | Effect |
|-----------|---------|--------|
| `--max-samples` | Limit dataset size | Smaller = faster testing, larger = better model |
| `--threshold` | Pixel value cutoff | Lower = more points detected (try 100-150) |
| `--timesteps` (sampling) | Diffusion steps | More = better quality, slower sampling |
| `--its` | Optimization steps | More = better training, longer training time |
| `batch_size` | Samples per step | Larger = faster training, more GPU memory |
| `lr` | Learning rate | Lower = slower but stable, higher = faster but unstable |

---

## ⚠️ Troubleshooting

### Issue: CUDA out of memory
```bash
# Reduce batch size in config: 32 → 16 → 8
# Or reduce image size: 45×45 → 32×32
```

### Issue: Point count error
```bash
# If you get "Can't perform OT: N != n**D"
# The image points don't match a perfect square
# Solution: --point-sizes 1024 2025 4096
```

### Issue: Training is slow
```bash
# Use GPU timing
salloc --partition=gpu_nodes --gpus=rtx_6000:1
# Then run inside allocated node
```

---

## 📚 Full Documentation

For detailed information, see:
```bash
cat GRADIENT_TRAINING.md
```

Sections include:
- Data preparation details
- Configuration options
- Monitoring & visualization
- Hyperparameter tuning
- Common issues & fixes

---

## 🎓 Learning Resources

This implementation uses:
1. **Diffusion Models**: Generative process by gradually denoising
2. **Optimal Transport**: Maps point sets to regular grids
3. **UNet Architecture**: Denoising network with residual connections
4. **EMA**: Exponential moving average for model stability

---

## ✨ Next Steps

1. ✅ Run quick training: `bash train_gradient_quick.sh`
2. Visualize samples: See GRADIENT_TRAINING.md
3. Tune hyperparameters based on results
4. Train on full 40,000 images for production model
5. Fine-tune for specific applications

---

## 📞 Environment

```bash
# Activate training environment
conda activate qmcdiffusion

# Check it works
conda run -n qmcdiffusion python -c "import torch; print(torch.__version__)"
```

All required dependencies installed:
- PyTorch 2.1.1
- h5py, tqdm, tensorboardX
- POT (optimal transport)
- matplotlib, NumPy

---

**Status**: ✅ Ready to train!

Start with: `bash train_gradient_quick.sh`
