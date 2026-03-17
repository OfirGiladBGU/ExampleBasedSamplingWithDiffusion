# Dynamic ControlNet V3.7 (Ablation) for Stipple Generation

This is the third-generation control module for grayscale-conditioned stipple generation.
Current active code path is **V3.7 ablation**:
- Reverted SDF conditioning from V3.4
- Removed Min-SNR-gamma weighting from overfit
- Removed target-density binarization in overfit
- Kept full resampling workflow (`--resample-jumps`) in sampling
- Re-added optional GECCO dynamic feature sampling for retests (`--enable-gecco` in overfit)
- Reverted V3.2 injection change: using `AdaptiveGateInjection` (not `StandardInjection`)

- **V3.1 "Static Anchor"** -- removed the GECCO-style dynamic feature sampling (`F.grid_sample` with a small CNN) that produced chaotic features for high-frequency images, causing the optimizer to shut the gates and collapse to uniform grids.
- **V3.2 "Shock the System"** -- replaced the zero-initialized `AdaptiveGateInjection` with `StandardInjection` (Kaiming-initialized 1×1 conv). Zero output at step 1 had allowed the frozen U-Net to learn a uniform-grid shortcut before the ControlNet could exert any influence.
- **V3.3 "CFG Override" (retired)** -- tested and removed for coordinate diffusion due to unstable extrapolation artifacts on large offsets.
- **V3.4 "Stable Geometry + Training"** -- baseline that introduced:
    - SDF channel conditioning (**SDF was first introduced in V3.4**)
    - 3-layer dilated hint encoder (dilation 2 and 4)
    - Min-SNR-gamma weighted denoising loss (overfit)
    - binary target-density conditioning
    - full-schedule RePaint-style resampling (sampling scripts)
- **V3.5 "Spatial-Temporal Loss Stacking" (retired)** -- tested additive spatial SDF weighting on top of Min-SNR.
- **V3.6 "Spatial Cross-Attention Routing" (retired)** -- tested optional bottleneck cross-attention (with zero-init output projection) and CLI toggle.
- **V3.7 "Resampling-Only Ablation" (active)** -- reverts condition representation to pre-SDF form, removes Min-SNR and binarization from overfit, and keeps full resampling.

## Key Differences from V2

| Aspect | V2 (Dynamic ControlNet) | V3.7 (active ablation) |
|--------|------------------------|---------------|
| Condition input | 4ch: offsets(2) + target density(1) + dynamic density(1) | 5ch: offsets(2) + target density(1) + coord grid(2) |
| Dynamic sampling | `grid_sample` from high-res image every step | Optional GECCO retest path in overfit (`--enable-gecco`) |
| Hint encoder | 4ch -> 2ch | 5ch -> 32 -> 64(d=2) -> 128(d=4) |
| Skip injection | AdaptiveGateInjection | AdaptiveGateInjection (V3.2 revert active) |
| Min-SNR (overfit) | No | No |
| Resample-jumps (sampling) | No | Yes |
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
        |   (5ch -> 32 -> 64(d2) -> 128(d4))
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
            AdaptiveGateInjection (×7)
            (sigmoid-gated 1×1 projection)
                   |
          ┌── additive injection ──┐
          |                        |
    Frozen UNet Encoder → Middle → Decoder → noise prediction
```

### AdaptiveGateInjection

Current active path uses sigmoid-gated 1×1 injection:

```
output = skip + sigmoid(gate(ctrl)) * transform(ctrl)
```

For historical context: V3.2 tested replacing this with `StandardInjection` (Kaiming 1×1 conv), but this branch currently runs the V3.2 revert requested for retesting.

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
python control_v3/test_overfit.py \
    --steps 5000 \
    --sample-index 0 \
    --enable-gecco
```
python control_v3/test_overfit.py --steps 10000 --sample-index 0

Default learning rate is `5e-4` (higher than V2's `1e-4` to help the Kaiming-initialized injections escape local minima faster).

V3.7 overfit uses plain denoising MSE and non-binarized target density while ablating SDF from the control representation.

## Tested Versions Archive (for quick jump/re-implementation)

| Version | Status | What was introduced | How to re-implement quickly |
|---------|--------|---------------------|-----------------------------|
| V3.4 | Historical baseline | SDF channel, 6ch condition, 3-layer dilated hint encoder, Min-SNR + binarization, resample-jumps | Add SDF computation in scripts; pass `sdf_map` through `DynamicControlledDenoiser.set_condition` and `DynamicControlNet.forward`; use 6ch hint input |
| V3.5 | Retired | Spatial SDF loss stacking on top of Min-SNR (overfit loss only) | In `test_overfit.py`, multiply unreduced MSE by `(1 + alpha*sdf)`, reduce to per-sample, then apply Min-SNR weighting |
| V3.6 | Retired | Bottleneck spatial cross-attention (Q=middle features, K/V=hint), zero-init final projection, CLI toggle | In `DynamicControlNet.py`, add `SpatialCrossAttention`; add `enable_spatial_attn` constructor arg; wire in middle block; add `--enable-spatial-attn` to `test_overfit.py` and `sample_control.py` |
| V3.7 | Active | Resampling-only ablation: 5ch condition representation, plain MSE + non-binarized density, full resampling kept | Current code |

Notes:
- SDF first appeared in V3.4.
- Min-SNR + binarization are part of the V3.4 winning baseline in this repo.

## Files

| File | Purpose |
|------|---------|
| `DynamicControlNet.py` | `StandardInjection`, `DynamicControlNet`, `DynamicControlledDenoiser` |
| `DynamicStippleDataset.py` | Dataset returning (high_res, target_density, offsets) |
| `prepare_data.py` | Convert stippled images to OT offset tensors |
| `train_control.py` | Training script with static conditioning loop |
| `sample_control.py` | Conditioned inference script |
| `test_overfit.py` | Single-example overfit debugging |
| `Self_README.md` | Compact technical reference for the V3.x data flow |

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
| V3.3 | CFG Override (retired) | Experimental wrapper-level CFG tested then removed for coordinate-diffusion instability |
| V3.4 | Stable Geometry + Training | Added SDF conditioning, dilated hint encoder, Min-SNR + binarization, and full resampling workflow |
| V3.5 | Spatial-Temporal Loss Stacking (retired) | Experimental additive spatial weighting variant |
| V3.6 | Spatial Cross-Attention Routing (retired) | Experimental bottleneck cross-attention with zero-init output projection and CLI toggle |
| V3.7 | Resampling-Only Ablation (active) | Reverted SDF conditioning from representation; removed Min-SNR and binarization; kept full resampling |
