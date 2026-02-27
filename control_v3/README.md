# Dynamic ControlNet V3.2 for Stipple Generation

This is the third-generation control module for grayscale-conditioned stipple generation. It evolves from V2 through two refinement stages:

- **V3.1 "Static Anchor"** -- removed the GECCO-style dynamic feature sampling (`F.grid_sample` with a small CNN) that produced chaotic features for high-frequency images, causing the optimizer to shut the gates and collapse to uniform grids.
- **V3.2 "Shock the System"** -- replaced the zero-initialized `AdaptiveGateInjection` with `StandardInjection` (Kaiming-initialized 1×1 conv). Zero output at step 1 had allowed the frozen U-Net to learn a uniform-grid shortcut before the ControlNet could exert any influence.

## Key Differences from V2

| Aspect | V2 (Dynamic ControlNet) | V3.2 (Static Anchor + Standard Init) |
|--------|------------------------|---------------------------------------|
| Condition input | 4ch: offsets(2) + target density(1) + dynamic density(1) | 5ch: offsets(2) + target density(1) + coord grid(2) |
| Dynamic sampling | `grid_sample` from high-res image every step | None -- static conditioning only |
| Feature extractor | None (samples raw 1ch pixel values) | None (removed; high-res image unused internally) |
| Hint encoder | 4ch -> 2ch (added before conv1) | 5ch -> 32 -> 64 -> 128ch with SiLU (added after conv1) |
| Skip injection | AdaptiveGateInjection (sigmoid-gated, bias=-4.0) | StandardInjection (1×1 conv, Kaiming init) |
| Spatial awareness | None | +2ch static [-1,1] coordinate meshgrid |
| Default overfit LR | 1e-4 | 5e-4 |

## Architecture

```
              [offsets_t (2ch)]
                    |
        ┌───────── | ─────────────────────┐
        |          v                      |
        |   cat([offsets_t,               |
        |        target_density,   (5ch)  |
        |        coord_grid])             |
        |          |                      |
        |   Hint Encoder                  |
        |   (5ch -> 32 -> 64 -> 128ch)    |
        |   with SiLU activations         |
        |          |                      |
        |        hint (128ch)             |
        └──────────┬──────────────────────┘
                   |
            ctrl_conv1(offsets_t) + hint
                   |
            Trainable Control Encoder
            (deepcopy of frozen encoder)
            [128, 256, 384] channels
                   |
            StandardInjection (×7)
            (Kaiming-init 1×1 conv)
                   |
          ┌── additive injection ──┐
          |                        |
    Frozen UNet Encoder → Middle → Decoder → noise prediction
```

### StandardInjection

V3.2 replaces the sigmoid-gated injection with a plain 1×1 convolution:

```
output = skip + inject_conv(ctrl)
```

The `inject_conv` uses PyTorch's default Kaiming initialization, producing a non-zero control signal from step 1. This forces the frozen U-Net to immediately account for the image condition, preventing it from settling into the uniform-grid local minimum that plagued earlier versions.

## Prerequisites

Same as V1/V2 -- use the `qmcdiffusion` conda environment plus OpenCV:

```bash
conda activate qmcdiffusion
pip install opencv-python
```

## Workflow

### 1. Prepare training data

Same OT pipeline as V1/V2 -- the offset representation is unchanged:

```bash
python control_v3/prepare_data.py \
    --gt_images /path/to/target \
    --output    /path/to/processed_offsets \
    --grid_size 32
```

### 2. Train the Dynamic ControlNet

```bash
python control_v3/train_control.py \
    --config  config/GBN/config.json \
    --ckpt    config/GBN/model.ckpt \
    --source  /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/source \
    --offsets /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets \
    --epochs  100 \
    --batch_size 16 \
    --lr 1e-4 \
    --out control_v3/control_out
```

Note: the `--source` argument points to the full-resolution images. V3.2 passes them through for API compatibility but does not use them internally (no dynamic sampling).

### 3. Generate conditioned stipple patterns

```bash
python control_v3/sample_control.py \
    --config       config/GBN/config.json \
    --base_ckpt    config/GBN/model.ckpt \
    --control_ckpt control_v3/control_out/dynamic_controlnet_v3_ep100.pt \
    --image        my_photo.png \
    --batch        16 \
    --timesteps    1000 \
    --output       stippled.npy
```

### 4. Quick overfit test (single example)

```bash
python control_v3/test_overfit.py --steps 5000 --sample-index 0
```

Default learning rate is `5e-4` (higher than V2's `1e-4` to help the Kaiming-initialized injections escape local minima faster).

## Files

| File | Purpose |
|------|---------|
| `DynamicControlNet.py` | `StandardInjection`, `DynamicControlNet`, `DynamicControlledDenoiser` |
| `DynamicStippleDataset.py` | Dataset returning (high_res, target_density, offsets) |
| `prepare_data.py` | Convert stippled images to OT offset tensors |
| `train_control.py` | Training script with static conditioning loop |
| `sample_control.py` | Conditioned inference script |
| `test_overfit.py` | Single-example overfit debugging |
| `Self_README.md` | Compact technical reference for the V3.2 data flow |

## Dataset

V3 uses the same `DynamicStippleDataset` as V2, returning three items per sample:

- **high_res_tensor** `(1, H, W)` -- full-resolution source image (kept for call-signature compatibility; not used internally by V3.2)
- **target_density_map** `(1, 32, 32)` -- created on-the-fly via area interpolation (average pooling)
- **offset_tensor** `(2, 32, 32)` -- same OT offsets as V1/V2

## Version History

| Version | Codename | Change |
|---------|----------|--------|
| V3.0 | -- | Added GECCO CNN feature extractor, 21ch hint encoder, AdaptiveGateInjection (bias=0.0), coordinate grid |
| V3.1 | Static Anchor | Removed GECCO CNN and `F.grid_sample`; 5ch hint encoder (static conditioning only) |
| V3.2 | Shock the System | Replaced AdaptiveGateInjection with StandardInjection (Kaiming init); bumped default LR to 5e-4 |
