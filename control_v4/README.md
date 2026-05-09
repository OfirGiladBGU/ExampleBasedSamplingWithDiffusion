# Dynamic ControlNet V4: Truncated Control + Smart Init

Control V4 is a standalone architecture for grayscale-conditioned stipple generation.
It keeps the frozen base denoiser plus a trainable ControlNet branch, but changes the problem formulation:

- training is truncated to the final fraction of the diffusion schedule
- inference starts from a Smart Init instead of pure Gaussian noise
- inference utilizes RePaint-style resampling jumps for local refinement
- train, sample, and overfit workflows are preserved
- W&B reporting, geometry metrics, periodic checkpoints, and best-geometry checkpoints are preserved

## Core idea

Instead of forcing the model to solve global routing from pure noise at the start of the schedule, V4 uses a classical Smart Init as a structural prior.
The model is then trained to refine and un-clump that initialization during the late denoising regime using optimal transport and resampling.

## Active V4 stack

- frozen base denoiser from the original diffusion model
- trainable V4 ControlNet branch
- `AdaptiveGateInjection` skip control (implementing dynamically learned "Gated Zero Convolutions")
- GECCO dynamic feature sampling from high-res `[image]` directly at point coordinates
- late-timestep truncated training via `--truncation-ratio`
- SDEdit-style inference start from noised Smart Init
- RePaint-style forward-backward resampling via `--resample-jumps`
- CV / clumped% geometry scoring
- periodic epoch checkpoints via `--save_every`
- best-geometry checkpoint saving with score, CV, and clumped% in filename

### Current Paper Configuration (Feature Flags)
While the codebase supports extensive conditioning channels, the final paper configuration explicitly **disables** several features to prevent structural leakage and isolate the GECCO boundary-awareness. The current defaults are:
```python
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
SMART_INIT_FEATURES = False      # Grid channel disabled (points still used for SDEdit start)
SDF_FEATURES = False             # Disabled in final model (used only in stress tests)
BATCH_COORDS_FEATURES = False    # Disabled in final model
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False

```

## Key differences from V3

| Aspect | V3.8 | V4 |
| --- | --- | --- |
| Initial sampling state | Pure Gaussian noise | Smart Init noised to `t_start` |
| Training timestep range | Full schedule | Truncated late schedule only |
| Static hint channels | density + sdf | density (SDF & Smart Init grids disabled by default) |
| Control objective | Global + local correction | Strict local refinement / unclumping |
| Sample start policy | `noise_fn(shape)` | explicit `sqrt(alpha)*x_init + sqrt(1-alpha)*noise` |
| Resampling | None | Forward-backward loops (`jumps=2`) |

## Architecture

```text
              [offsets_t (2ch)]
                    |
        cat([
            offsets_t,
            target_density,
            (target_sdf),        <-- Disabled by default
            (smart_init_grid),   <-- Disabled by default
            (coord_grid)         <-- Disabled by default
        ])
                    |
              Hint Encoder
         (3ch -> 32 -> 64(d2) -> 128(d4))
                    |
         ctrl_conv1(offsets_t) + hint + [GECCO Features]
                    |
         Trainable Control Encoder
                    |
         AdaptiveGateInjection skips (Transform Gates)
                    |
          Frozen U-Net noise prediction

```

When GECCO is enabled, dynamic high-resolution features from the image are extracted via `grid_sample` at the exact floating-point coordinates of the current point set and appended to the hint path.

## Smart Init

V4 uses an in-repo Smart Init implementation in:

* `control_v4/smart_init.py`

It provides:

* density-based point initialization from the source image via rejection sampling
* conversion of Smart Init points into OT-style offset grids
* explicit forward-noise helper for SDEdit-style start
* debug export of Smart Init points / offsets / grid

*Note: While the network supports rasterizing the Smart Init points into a GPU-rendered hint channel (Gaussian soft splatting), this channel (`SMART_INIT_FEATURES`) is disabled in the final paper configuration. The Smart Init is solely used as the physical starting state $x_{init}$ for the truncated SDEdit schedule.*

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
2. Choose `t_start` from `--truncation-ratio` (default `0.30`).
3. Apply explicit forward noise:

```python
x_noisy = sqrt(alpha_cumprod[t_start]) * x_init + sqrt(1 - alpha_cumprod[t_start]) * noise

```

4. Reverse denoise from `t_start - 1` down to `0`.
5. **(New in V4)** Apply RePaint-style forward-backward resampling (`jumps=2`) during this local refinement window to give the model multiple opportunities to enforce capacity constraints and unclump points.

## DPM++ status (tried, currently disabled)

We tested a DPM-Solver++ inference branch in V4 (`sample_control.py` and `test_overfit.py`) and observed worse visual/geometric quality than the DDPM truncated loop for our data.

Current policy:

* V4 uses DDPM-only sampling paths in active scripts.
* DPM++ code paths and CLI flags were removed from the runnable flow.

How DPM++ had been implemented (for future rollback):

* Build a `DDPMScheduler` from the model betas.
* Create `DPMSolverMultistepScheduler.from_config(..., algorithm_type="dpmsolver++", solver_order=2)`.
* Call `set_timesteps(inference_steps)`.
* Keep truncation by taking the tail subset of timesteps.
* Re-noise `smart_init_offsets` at the DPM start step using scheduler `alphas_cumprod`.
* Run iterative `scheduler.step(noise_pred, t, x).prev_sample` updates.

If you revisit DPM++ later, re-introduce it behind an explicit optional flag and compare against DDPM on the same checkpoints with both visual and geometry metrics.

## Post-GECCO point attention status (tried, removed)

We also tested an overfit-only point self-attention block immediately after GECCO feature sampling.

What was tested:

* take GECCO sampled features after `grid_sample`
* reshape from `(B, C, H, W)` to `(B, N, C)` with `N = H * W`
* run a lightweight pre-norm Transformer block
* reshape back to `(B, C, H, W)` and continue through the existing hint path

The block design that was tested:

* `LayerNorm -> MultiheadAttention(batch_first=True) -> residual`
* `LayerNorm -> MLP(Linear, GELU, Linear) -> residual`
* zero-init on `mha.out_proj` and the final MLP linear layer so the block starts as near-identity

Observed result:

* small positive boost in the 32x32 overfit setting
* not enough improvement to justify carrying the extra point-mixing path in the active V4 baseline
* concern that this branch may become a limiting architectural fork for future larger-grid tests, so it was removed from runnable code

If we restore it later:

* keep it overfit-only at first
* add it right after `DynamicControlNet.compute_gecco_features()` returns the sampled GECCO map
* flatten sampled features to `(B, N, C)`, apply attention, then reshape back before concatenating into the hint input
* expose it behind explicit flags such as `--point-attn` and `--point-attn-heads`
* compare 32x32 and 64x64 separately rather than assuming the same conclusion holds across scales

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

* `--truncation-ratio`
* `--save_every`
* `--best-max-cv`
* `--best-max-clumped-pct`
* *(Note: Smart Init / SDF channel flags are safely defaulted to False in the code)*

### 2. Sample

```bash
python control_v4/sample_control.py \
    --config config/GBN/config.json \
    --base_ckpt config/GBN/model.ckpt \
    --control_ckpt control_v4/train_outputs/dynamic_controlnet_v4_ep8120.pt \
    --image my_photo.png \
    --timesteps 1000 \
    --truncation-ratio 0.30 \
    --resample-jumps 2 \
    --out-dir control_v4/sample_outputs

```

Important V4 flags:

* `--truncation-ratio`
* `--resample-jumps`
* `--t-start-step`
* `--show-denoising`
* `--denoise-interval`

### 3. Overfit

```bash
python control_v4/test_overfit.py --steps 10000 --sample-index 0

```

The overfit path keeps:

* W&B logging
* periodic visualization exports
* geometry scoring (CV / clumped % / score)
* best checkpoint naming with geometry values

## Output structure

### Sampling

```text
control_v4/sample_outputs/
  image-name/
    conditions/
      high_res.npy
      target_density.npy
      conditions_collage.png
    smart_init/
      smart_init_points.npy
      smart_init_offsets.npy
      smart_init_collage.png
    denoising_steps/
      step_0000.png
      step_0050.png
      ...
    npy/
      image-name_1.npy
    png/
      image-name_1.png
    metrics/
      results_panel.png

```

### Training

```text
control_v4/train_outputs/
  dynamic_controlnet_v4_epN.pt
  checkpoints/
    best_controlnet_epXXXX_score..._cv..._clumped....pt

```

## W&B reporting

V4 preserves the reporting style used in V3 and adds a first-batch condition debugger.

Metrics include:

* `step_loss/total`
* `step_loss/denoise`
* `epoch_loss/train`
* `epoch_loss/valid`
* `epoch_loss/compare`
* `geom/cv`
* `geom/clumped_pct`
* `geom/score`
* `debug/hint_channels`

The first batch logs a visual debug image of the active conditioning channels.

## Checkpoints

V4 keeps two checkpoint families:

1. Periodic epoch checkpoints:

* `dynamic_controlnet_v4_epN.pt`
* controlled by `--save_every`

2. Best geometry checkpoints:

* `best_controlnet_epXXXX_score..._cv..._clumped....pt`
* gated by geometry thresholds and best score improvement

Checkpoint payloads include geometry metadata such as:

* `best_geom_score`
* `cv_score`
* `clumped_score`
* `current_geom_score`

## Files

| File | Purpose |
| --- | --- |
| `smart_init.py` | Smart Init generation, rasterization, debug export, explicit forward noising |
| `DynamicControlNet.py` | V4 ControlNet and controlled denoiser wrapper with Adaptive Gates and GECCO |
| `DynamicStippleDataset.py` | Dataset returning image, density, sdf, offsets, and cached Smart Init grid |
| `conditioning.py` | Density + SDF preprocessing |
| `train_control.py` | V4 training with truncated timesteps and W&B logging |
| `sample_control.py` | V4 sampling with Smart Init SDEdit-style start and resampling |
| `test_overfit.py` | Single-example V4 overfit workflow |
| `train_control_ablation.py` | Pre-configured script for generating isolated ablation checkpoints |
| `Self_README.md` | Compact technical V4 flow summary |

## Notes

* V4 is intended to be standalone.
* It should not depend on legacy `control_v1`, `control_v2`, `control_v3`, or any `train_free_*` implementation logic at runtime.
* Smart Init is implemented directly inside `control_v4`.
* Auxiliary x0 losses were tested during V4 development (`mse`, `chamfer`, `swd`) and did not improve quality in our runs, so they were removed from active train/overfit codepaths.
