# Control V5: Lightweight Image-GECCO Early Fusion

Control V5 is a lightweight alternative to V4's ControlNet branch.
Instead of running a full parallel U-Net encoder, a tiny two-layer CNN extracts features from the conditioning image and injects them directly into the denoiser's first convolution via GECCO feature sampling.

Key properties:
- no parallel ControlNet branch — ~10× fewer trainable parameters vs V4
- no Smart Init, no SDF channel, no truncation, no resample jumps
- full diffusion schedule used in both training and inference
- resolution-agnostic at inference time (train 32×32, run 48×48)
- jointly trains image extractor + widened denoiser first conv together

## Core idea

GECCO (Geometry-Aware Feature Conditioning with Offsets) samples image features at the **current noisy-offset positions** on every denoiser call.
The sampled features are concatenated to the noisy offsets and fed through the U-Net's widened first convolution.
This is *early fusion*: the image information enters the model at the very first layer rather than via injected skip controls.

```
conditioning image (B, 1, H, W)
          │
   gecco_extractor
   Conv2d(1 → 8, 3×3, SiLU)
   Conv2d(8 → gecco_ch, 3×3, SiLU)
          │
  gecco_map (B, gecco_ch, H, W)
          │
     F.grid_sample at noisy-offset positions
          │
  gecco (B, gecco_ch, G, G)
          │
  cat([offsets_t (B, 2, G, G), gecco])
          │
  widened conv1 (2 + gecco_ch → 128)
          │
     U-Net denoiser
          │
  predicted noise (B, 2, G, G)
```

## Architecture

### ImageGECCOWrapper (`LightweightAdapter.py`)

The wrapper holds:

1. **`gecco_extractor`** — two `Conv2d` layers with `SiLU` activation:
   ```
   (B, 1, H, W)  →  (B, 8, H, W)  →  (B, gecco_ch, H, W)
   ```
   Default `gecco_ch = 8`.

2. **Widened `denoiser.conv1`** — on construction, the denoiser's first convolution is widened **in-place** from `Conv2d(2, 128)` to `Conv2d(2 + gecco_ch, 128)`.
   - Pretrained weights for the original 2 channels are copied unchanged.
   - New channels are zero-initialised so the model starts as a near-identity correction.

3. **`compute_gecco(offsets_t, image)`** — derives grid-cell centers dynamically from `offsets_t.shape[-1]` (G), adds the noisy offsets as sub-cell displacements, and calls `F.grid_sample` with bilinear interpolation:
   ```
   positions = grid_centers + offsets_t / G   # absolute image coordinates ∈ [0, 1]
   sample_coords = positions * 2 − 1          # normalised to [-1, 1] for grid_sample
   ```

4. **`forward(x, t)`** — concatenates `gecco` to `x` along the channel axis and calls the wrapped denoiser once:
   ```
   x_aug = cat([x, gecco], dim=1)   # (B, 2 + gecco_ch, G, G)
   return self.denoiser(x_aug, t)
   ```

### API

```python
wrapper = ImageGECCOWrapper(denoiser, gecco_ch=8)
wrapper.set_condition(high_res_image)   # (B, 1, H, W) float32 in [0, 1]
noise_pred = wrapper(offsets_t, t)      # (B, 2, G, G)
```

`set_condition` must be called once per batch before `forward`.
The cached image is broadcast automatically if `B=1` during inference.

## Key differences from V4

| Aspect | V4 | V5 |
|--------|----|----|
| Control mechanism | Parallel ControlNet encoder + AdaptiveGateInjection | Early fusion via GECCO channel concat |
| Trainable branch | Separate ControlNet (frozen denoiser) | Image extractor + widened conv1 (joint training) |
| Smart Init | Yes — GPU-rendered raster grid, jitter augmentation | No |
| SDF channel | Yes | No |
| Hint input channels | offsets(2) + density(1) + sdf(1) + smart_init(1) + coord(2) = 7 | offsets(2) + GECCO(gecco_ch) |
| Truncated training | Yes — last `truncation_ratio` fraction of schedule | No — full schedule |
| Sample start | SDEdit from noised Smart Init | Pure Gaussian noise |
| Resample jumps | Optional | None |
| Approximate param count | ~millions (full parallel U-Net encoder) | ~thousands (two Conv2d layers + widened conv1) |

## Training

```bash
python control_v5/train_control.py \
    --config config/GBN/config.json \
    --ckpt config/GBN/model.ckpt \
    --source /path/to/source \
    --offsets /path/to/processed_offsets \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --gecco-ch 8 \
    --out control_v5/train_outputs
```

Checkpoints are saved as `gecco_wrapper_ep{N}.pt` with keys `wrapper` and `gecco_ch`.
Best-geometry checkpoint is saved separately with CV and clumped-% in the filename.

## Inference

```bash
python control_v5/sample_control.py \
    --config config/GBN/config.json \
    --base_ckpt config/GBN/model.ckpt \
    --wrapper_ckpt control_v5/train_outputs/best_gecco_wrapper.pt \
    --source /path/to/source \
    --out control_v5/sample_outputs
```

Sampling uses the full DDPM reverse loop (`p_sample_loop`) from pure Gaussian noise with no resampling jumps.

## Files

| File | Purpose |
|------|---------|
| `LightweightAdapter.py` | `ImageGECCOWrapper` — the core model |
| `DynamicStippleDataset.py` | Dataset returning `(high_res, offsets)` pairs |
| `train_control.py` | Training loop with W&B, geometry metrics, checkpointing |
| `sample_control.py` | Batch inference script |
| `test_overfit.py` | Single-example overfit sanity test |
| `__init__.py` | Package export (`ImageGECCOWrapper`) |
