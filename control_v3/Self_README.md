# Control V3.6 Flow (V3.4 Baseline + Optional Attention)

Current behavior is intentionally split into two modes:

- Baseline mode (default): exact V3.4 path.
- Attention mode (opt-in): V3.6 spatial cross-attention at bottleneck.

Use `--enable-spatial-attn` in `test_overfit.py` and `sample_control.py` to activate V3.6.

## Baseline (V3.4)

Condition channels (6 total):
- offsets_t: 2ch
- target_density: 1ch (binarized)
- sdf: 1ch
- coord_grid: 2ch

Hint path:
- `cat([offsets_t, target_density, sdf, coord_grid])`
- 3-layer dilated hint encoder:
  - 6 -> 32 (k3, p1)
  - 32 -> 64 (k3, p2, d2)
  - 64 -> 128 (k3, p4, d4)

Control path:
- `x = ctrl_conv1(offsets_t) + hint`
- trainable copied encoder + middle
- StandardInjection 1x1 conv per control output

Training loss in overfit:
- Min-SNR-gamma weighted denoising MSE
- binary target density + SDF conditioning

Sampling:
- Full reverse diffusion
- Optional RePaint-style micro-loops (`--resample-jumps`)

## V3.6 Spatial Cross-Attention

Location:
- Applied after the control middle block output.

Mechanism:
- Query (Q): deep middle feature map.
- Key/Value (K/V): hint tensor.
- Spatial flattening to sequences (`HW` tokens), multi-head attention, reshape back.
- Residual output: `x + attn_out`.

Safety constraint (critical):
- Final attention projection is zero-initialized (weights and bias = 0.0).
- This guarantees identity behavior at step 1 when enabling attention.

## Checkpoint compatibility

When attention is enabled but loading an older checkpoint:
- `sample_control.py` uses `strict=False` and reports missing/unexpected key counts.
- This is expected because V3.6 adds attention parameters.

## Practical toggle guide

- Stable baseline run:
  - omit `--enable-spatial-attn`
- V3.6 run:
  - include `--enable-spatial-attn`
- Recommended first experiment:
  - overfit edge cases (rings/rectangles) with full training horizon.
