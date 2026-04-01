# Dynamic ControlNet V4: Truncated Control + Smart Init

Control V4 is a standalone architecture for grayscale-conditioned stipple generation.
It keeps the frozen base denoiser plus a trainable ControlNet branch, but changes the problem formulation:

- training is truncated to the final fraction of the diffusion schedule
- inference starts from a Smart Init instead of pure Gaussian noise
- the hint path includes an explicit Smart Init grid channel
- train, sample, and overfit workflows are preserved
- W&B reporting, geometry metrics, periodic checkpoints, and best-geometry checkpoints are preserved

## Core idea

Instead of forcing the model to solve global routing from pure noise at the start of the schedule, V4 uses a classical Smart Init as a structural prior.
The model is then trained to refine and un-clump that initialization during the late denoising regime.

## Active V4 stack

- frozen base denoiser from the original diffusion model
- trainable V4 ControlNet branch
- AdaptiveGateInjection skip control
- optional GECCO dynamic feature sampling from high-res `[image, sdf]`
- late-timestep truncated training via `--truncation-ratio`
- Smart Init points + Smart Init raster grid
- SDEdit-style inference start from noised Smart Init
- CV / clumped% geometry scoring
- periodic epoch checkpoints via `--save_every`
- best-geometry checkpoint saving with score, CV, and clumped% in filename

## Key differences from V3

| Aspect | V3.8 | V4 |
|--------|------|----|
| Initial sampling state | Pure Gaussian noise | Smart Init noised to `t_start` |
| Training timestep range | Full schedule | Truncated late schedule only |
| Static hint channels | density + sdf | density + sdf + smart_init_grid |
| Hint input | offsets(2) + density(1) + sdf(1) + coord(2) | offsets(2) + density(1) + sdf(1) + smart_init(1) + coord(2) |
| Control objective | Global + local correction | Mostly local refinement / unclumping |
| Sample start policy | `noise_fn(shape)` | explicit `sqrt(alpha)*x_init + sqrt(1-alpha)*noise` |
| Smart Init export | none | yes |

## Architecture

```text
              [offsets_t (2ch)]
                    |
        cat([
            offsets_t,
            target_density,
            target_sdf,
            smart_init_grid,
            coord_grid
        ])
                    |
              Hint Encoder
         (7ch -> 32 -> 64(d2) -> 128(d4))
                    |
         ctrl_conv1(offsets_t) + hint
                    |
         Trainable Control Encoder
                    |
         AdaptiveGateInjection skips
                    |
          Frozen U-Net noise prediction
```

When GECCO is enabled, dynamic high-resolution features from `[image, sdf]` are appended to the hint path at runtime.

## Smart Init

V4 uses an in-repo Smart Init implementation in:

- `control_v4/smart_init.py`

It provides:

- density-based point initialization from the source image
- conversion of Smart Init points into OT-style offset grids
- rasterized `smart_init_grid` for hint conditioning
- explicit forward-noise helper for SDEdit-style start
- debug export of Smart Init points / offsets / grid

Training is designed to cache Smart Init grids to disk for throughput.
Sampling and overfit generate Smart Init artifacts directly for debugging.

## Truncated training

V4 trains only on the last fraction of scheduler timesteps:

```python
truncation_cutoff = int(num_timesteps * truncation_ratio)
t = torch.randint(0, truncation_cutoff, (batch_size,), device=device)
```

The default truncation ratio is `0.30`.

This means the model learns to refine slightly noised, already-structured point sets rather than route points from pure noise across the whole image.

## Inference

V4 sampling does not start from pure Gaussian noise.

1. Build Smart Init points and Smart Init offsets.
2. Choose `t_start` from `--truncation-ratio` or `--t-start-step`.
3. Apply explicit forward noise:

```python
x_noisy = sqrt(alpha_cumprod[t_start]) * x_init + sqrt(1 - alpha_cumprod[t_start]) * noise
```

4. Reverse denoise only from `t_start - 1` down to `0`.

This is a coordinate-space SDEdit-style initialization.

## Scripts

### 1. Train

```bash
python control_v4/train_control.py \
    --config config/GBN/config.json \
    --ckpt config/GBN/model.ckpt \
    --source /path/to/source \
    --offsets /path/to/processed_offsets \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --truncation-ratio 0.30 \
    --out control_v4/train_outputs
```

Important V4 flags:

- `--truncation-ratio`
- `--smart-init-cache-dir`
- `--smart-init-seed`
- `--save_every`
- `--best-max-cv`
- `--best-max-clumped-pct`

### 2. Sample

```bash
python control_v4/sample_control.py \
    --config config/GBN/config.json \
    --base_ckpt config/GBN/model.ckpt \
    --control_ckpt control_v4/train_outputs/dynamic_controlnet_v3_ep1.pt \
    --image my_photo.png \
    --timesteps 1000 \
    --truncation-ratio 0.30 \
    --out-dir control_v4/sample_outputs
```

Important V4 flags:

- `--truncation-ratio`
- `--t-start-step`
- `--show-denoising`
- `--denoise-interval`
- `--use-sdf`

### 3. Overfit

```bash
python control_v4/test_overfit.py --steps 10000 --sample-index 0
```

The overfit path keeps:

- W&B logging
- periodic visualization exports
- geometry scoring (CV / clumped % / score)
- best checkpoint naming with geometry values

## Output structure

### Sampling

```text
control_v4/sample_outputs/
  image-name/
    conditions/
      high_res.npy
      high_res_sdf.npy
      target_density.npy
      target_sdf.npy
      smart_init_grid.npy
      conditions_collage.png
    smart_init/
      smart_init_points.npy
      smart_init_offsets.npy
      smart_init_grid.npy
      smart_init_collage.png
    denoising_steps/
      step_0000.png
      step_0050.png
      ...
    npy/
      image-name_1.npy
    png/
      image-name_1.png
```

### Training

```text
control_v4/train_outputs/
  dynamic_controlnet_v3_epN.pt
  checkpoints/
    best_controlnet_epXXXX_score..._cv..._clumped....pt
```

## W&B reporting

V4 preserves the reporting style used in V3 and adds a first-batch condition debugger.

Metrics include:

- `step_loss/total`
- `step_loss/denoise`
- `epoch_loss/train`
- `epoch_loss/valid`
- `epoch_loss/compare`
- `geom/cv`
- `geom/clumped_pct`
- `geom/score`
- `debug/hint_channels`

The first batch logs a visual debug image of:

- target density
- target sdf
- smart_init_grid

## Checkpoints

V4 keeps two checkpoint families:

1. Periodic epoch checkpoints:
- `dynamic_controlnet_v3_epN.pt`
- controlled by `--save_every`

2. Best geometry checkpoints:
- `best_controlnet_epXXXX_score..._cv..._clumped....pt`
- gated by geometry thresholds and best score improvement

Checkpoint payloads include geometry metadata such as:

- `best_geom_score`
- `cv_score`
- `clumped_score`
- `current_geom_score`

## Files

| File | Purpose |
|------|---------|
| `smart_init.py` | Smart Init generation, rasterization, debug export, explicit forward noising |
| `DynamicControlNet.py` | V4 ControlNet and controlled denoiser wrapper with Smart Init condition support |
| `DynamicStippleDataset.py` | Dataset returning image, density, sdf, offsets, and cached Smart Init grid |
| `conditioning.py` | Density + SDF preprocessing |
| `train_control.py` | V4 training with truncated timesteps and W&B logging |
| `sample_control.py` | V4 sampling with Smart Init SDEdit-style start |
| `test_overfit.py` | Single-example V4 overfit workflow |
| `Self_README.md` | Compact technical V4 flow summary |

## Notes

- V4 is intended to be standalone.
- It should not depend on legacy `control_v1`, `control_v2`, `control_v3`, or any `train_free_*` implementation logic at runtime.
- Smart Init is implemented directly inside `control_v4`.
- Auxiliary x0 losses were tested during V4 development (`mse`, `chamfer`, `swd`) and did not improve quality in our runs, so they were removed from active train/overfit codepaths.
