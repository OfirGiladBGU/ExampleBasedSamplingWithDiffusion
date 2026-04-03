# Control V4 (Truncated Control + Smart Init) flow

Active V4 components:
- truncated late-timestep training
- Smart Init coordinate start for sampling and overfit
- GPU-rendered `smart_init_grid` (Gaussian soft splatting) as an extra hint channel
- GPU Smart Init micro-jitter augmentation (pixel-unit strength)
- optional GECCO dynamic feature sampling from high-res `[image, sdf]`
- AdaptiveGateInjection control skips
- DDPM sampling path
- geometry metrics, visual exports, and checkpointing

## Smart Init flow

- Input: source grayscale image in `[0,1]`, where black means ink / target mass.
- Smart Init points are sampled from dark regions using density-weighted rejection sampling.
- Smart Init points are converted to an OT-style offset tensor.
- Smart Init points are also rasterized into a low-res occupancy-style grid.

Outputs:
- `smart_init_points` -> `(N, 2)` in `[0,1]`
- `smart_init_offsets` -> `(2, 32, 32)`
- `smart_init_grid` -> `(1, 32, 32)`

Augmentation policy:
- training: optional GPU micro-jitter via `--smart-init-jitter-px`
- training: GPU Gaussian soft-splat rendering via `--smart-init-splat-sigma-px`
- validation: no jitter (deterministic Smart Init)
- overfit: same GPU controls (`--smart-init-jitter-px`, `--smart-init-splat-sigma-px`)

## Condition flow

Static condition inputs in V4:
- `offsets_t` -> `(B, 2, 32, 32)`
- `target_density` -> `(B, 1, 32, 32)`
- `target_sdf` -> `(B, 1, 32, 32)`
- `smart_init_grid` -> `(B, 1, 32, 32)`
- `coord_grid` -> `(B, 2, 32, 32)`

Concatenation:
- GECCO off:
  - `cat([offsets_t, target_density, target_sdf, smart_init_grid, coord_grid])`
  - total = `7` channels
- GECCO on:
  - append dynamic high-res sampled features on top of those 7 channels

Hint encoder:
- `7ch -> 32 -> 64(d=2) -> 128(d=4)` before GECCO channels

Fusion:
- `x = ctrl_conv1(offsets_t) + hint`

## GECCO path

When enabled:
- high-res `[image, sdf]` is passed through the GECCO CNN
- features are sampled at current noisy point positions using `grid_sample`
- sampled features are appended to the static hint input

## Training

Frozen:
- base denoiser

Trainable:
- V4 ControlNet branch

Timestep sampling:
- `truncation_cutoff = int(num_timesteps * truncation_ratio)`
- `t ~ Uniform(0, truncation_cutoff)`

So V4 learns only on late-stage denoising.

Loss:
- Min-SNR weighted denoising MSE

Dataset returns:
- `high_res`
- `target_density`
- `high_res_sdf`
- `target_sdf`
- `offsets`
- `smart_init_grid`

A first-batch debug image is logged to W&B showing:
- `target_density`
- `target_sdf`
- `smart_init_grid`

## Sampling

V4 sampling is SDEdit-style in coordinate space.

1. Build `smart_init_offsets`
2. Choose `t_start`
3. Apply explicit forward noise:

```python
x_noisy = sqrt(alpha_t) * x_init + sqrt(1 - alpha_t) * noise
```

4. Reverse denoise only from `t_start - 1` down to `0`

This means V4 does not start from pure Gaussian noise at the top of the schedule.

## Inference backend

- active V4 inference uses DDPM-only sampling paths

## Overfit

V4 overfit keeps the same debugging philosophy as V3:
- periodic sampling visualization
- CV / clumped % / geometry score reporting
- best checkpoint naming with score, CV, clumped %
- W&B logging when enabled

## Checkpoints

Periodic checkpoints:
- `dynamic_controlnet_v3_epN.pt`
- controlled by `--save_every`

Best geometry checkpoints:
- `best_controlnet_epXXXX_score..._cv..._clumped....pt`

Stored geometry metadata:
- `best_geom_score`
- `cv_score`
- `clumped_score`
- `current_geom_score`

## Sampling outputs

Per-image output folder contains:
- `conditions/`
- `smart_init/`
- `denoising_steps/`
- `npy/`
- `png/`

## Important constraints

- V4 is standalone.
- Do not depend on `train_free_*` folders.
- Do not depend on legacy `control_v1`, `control_v2`, or `control_v3` runtime codepaths.
- Smart Init logic lives inside `control_v4/smart_init.py`.
