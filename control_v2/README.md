# Dynamic ControlNet V2 for Stipple Generation

This is the second-generation control module for grayscale-conditioned stipple generation. It replaces the static conditioning of V1 with a **dynamic density feedback loop** that recomputes the conditioning signal at every denoising step.

## Key Differences from V1

| Aspect | V1 (ControlNet) | V2 (Dynamic ControlNet) |
|--------|-----------------|------------------------|
| Condition input | 1ch grayscale (32x32) | 4ch: offsets(2) + target density(1) + dynamic density(1) |
| Density computation | Static resize at init | Dynamic `grid_sample` from high-res at each step |
| Skip injection | ZeroConv2d (additive) | AdaptiveGateInjection (sigmoid-gated) |
| Image resolution | Downsampled to 32x32 | Full resolution (512x512) preserved |

## Architecture

```
              [offsets_t (2ch)]
                    |
                    v
        ┌── grid_centers + offsets_t / 32 ──┐
        |           = positions              |
        |               |                   |
        |    F.grid_sample(high_res_img)     |
        |               |                   |
        |    dynamic_local_density (1ch)     |
        └───────────────┬───────────────────┘
                        |
    cat([offsets_t, target_density, dynamic_density]) = 4ch
                        |
                  Hint Encoder (4ch -> 2ch)
                        |
                  offsets_t + hint
                        |
              Trainable Control Encoder
              (deepcopy of frozen encoder)
                        |
              AdaptiveGateInjection (x7)
                        |
            ┌───── gated addition ──────┐
            |                           |
      Frozen UNet Encoder → Middle → Decoder → noise prediction
```

### AdaptiveGateInjection

Instead of standard zero convolutions (`skip + zero_conv(ctrl)`), V2 uses sigmoid-gated injection:

```
output = skip + sigmoid(gate(ctrl)) * transform(ctrl)
```

The gate bias is initialized to -4.0 (sigmoid ≈ 0.018), providing the same near-zero startup stability as zero convolutions while allowing the network to adaptively scale control injection per spatial location during training.

## Prerequisites

Same as V1 -- use the `qmcdiffusion` conda environment plus OpenCV:

```bash
conda activate qmcdiffusion
pip install opencv-python
```

## Workflow

### 1. Prepare training data

Same OT pipeline as V1 -- the offset representation is unchanged:

```bash
python control_v2/prepare_data.py
```

### 2. Train the Dynamic ControlNet

```bash
python control_v2/train_control.py \
    --config  config/GBN/config.json \
    --ckpt    config/GBN/model.ckpt \
    --source  /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/source \
    --offsets /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets \
    --epochs  100 \
    --batch_size 16 \
    --lr 1e-4 \
    --out control_v2/control_out
```

Note: the `--source` argument points to the full-resolution images (not downsampled), since V2 uses the high-res image for dynamic density sampling.

### 3. Generate conditioned stipple patterns

```bash
python control_v2/sample_control.py \
    --config       config/GBN/config.json \
    --base_ckpt    config/GBN/model.ckpt \
    --control_ckpt control_v2/control_out/dynamic_controlnet_ep100.pt \
    --image        my_photo.png \
    --batch        16 \
    --timesteps    1000 \
    --output       stippled.npy
```

### 4. Quick overfit test (single example)

```bash
python control_v2/test_overfit.py --steps 10000 --sample-index 0
```

## Files

| File | Purpose |
|------|---------|
| `DynamicControlNet.py` | `AdaptiveGateInjection`, `DynamicControlNet`, `DynamicControlledDenoiser` |
| `DynamicStippleDataset.py` | Dataset returning (high_res, target_density, offsets) |
| `prepare_data.py` | Convert stippled images to OT offset tensors |
| `train_control.py` | Training script with dynamic conditioning loop |
| `sample_control.py` | Conditioned inference script |
| `test_overfit.py` | Single-example overfit debugging |
| `run_overfit.sh` | SLURM job script for overfit test |

## Dataset Differences

V2's `DynamicStippleDataset` returns three items per sample instead of V1's two:

- **high_res_tensor** `(1, 512, 512)` -- full-resolution source image (kept at native resolution for `grid_sample`)
- **target_density_map** `(1, 32, 32)` -- created on-the-fly via area interpolation (average pooling)
- **offset_tensor** `(2, 32, 32)` -- same OT offsets as V1
