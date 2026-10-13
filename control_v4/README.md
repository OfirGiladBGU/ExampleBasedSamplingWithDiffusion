# Dynamic ControlNet V4: Smart Init + Truncated SDEdit Sampling

This is the control module for grayscale-conditioned stipple generation. Given a grayscale
photograph or tone map, it generates a 2D point set (stippling) whose local density follows the
tonal values of the input image.

It pairs the point-set diffusion U-Net (OT offset grid) with a ControlNet branch, and changes the
problem formulation:

- the base U-Net and the control branch are randomly initialized and **trained jointly from scratch**
  (the original ControlNet recipe of a frozen pretrained base is kept as an ablation)
- inference starts from a Smart Init instead of pure Gaussian noise and runs only the last part of
  the schedule (truncation ratio `0.5`, i.e. `t_start = 500` of `T = 1000`)
- the loss is a Min-SNR-weighted noise-prediction MSE plus a KDE density loss
- RePaint-style resampling jumps are supported but off by default
- train, sample, overfit, ablation, and dataset-evaluation workflows are supported
- W&B reporting, geometry metrics, periodic checkpoints, and optional best-geometry checkpoints are
  supported

> **Note on naming:** the code still tags its artifacts with a `v4` marker (W&B runs are named
> `v4-train-...` / `v4-overfit-...`). The approaches that were tried and removed are documented
> below and in [Self_README.md](Self_README.md).

## Core idea

Instead of forcing the model to solve global routing from pure noise at the start of the schedule,
this approach uses a classical Smart Init as a structural prior. Because the Smart Init already
follows the target density, the reverse process can start from a partially noised version of it and
only has to refine and un-clump the points. This halves inference time relative to sampling the same
model from pure noise, at matched quality.

## Active stack

- base U-Net trained jointly with the control branch (`--no-freeze-denoiser --base_ckpt_path ""`)
- ControlNet branch
- `AdaptiveGateInjection` skip control (implementing dynamically learned "Gated Zero Convolutions")
- GECCO dynamic feature sampling from high-res `[image]` directly at point coordinates
- Min-SNR-gamma weighted denoising loss (`--min-snr-gamma`, default `5.0`)
- KDE density loss on the one-step `x_0` decode (`--density-loss-weight`, default `0.8`)
- SDEdit-style inference start from noised Smart Init (`--infer-truncation-ratio`, default `0.5`)
- optional RePaint-style forward-backward resampling (`--resample_jumps`, default `0`)
- CV / clumped% geometry scoring
- periodic epoch checkpoints via `--save_every`, pruned via `--keep-every`
- optional best-geometry checkpoint saving (`--save-best`, off by default) with score, CV, and
  clumped% in filename

### Current Paper Configuration (Feature Flags)

While the codebase supports extensive conditioning channels, the final paper configuration
explicitly **disables** several features to prevent structural leakage and isolate the GECCO
boundary-awareness. The current defaults (`train_control.py`, `sample_control.py`) are:

```python
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
SMART_INIT_FEATURES = False      # Grid channel disabled (points still used for SDEdit start)
SDF_FEATURES = False             # Disabled in final model (used only in stress tests)
BATCH_COORDS_FEATURES = False    # Disabled in final model
ENABLE_SMART_INIT_JITTER = False
ENABLE_SMART_INIT_SPLAT_SIGMA = False
```

The paper models were trained with `--base_ckpt_path "" --no-freeze-denoiser` and the remaining
defaults; each run records its exact arguments in `run_params.json` inside its output folder.

## Key differences from V3

| Aspect | V3.8 | V4 |
| --- | --- | --- |
| Base U-Net | Frozen, pretrained | Trained jointly with the control branch from scratch |
| Initial sampling state | Pure Gaussian noise | Smart Init noised to `t_start` (`0.5 * T`) |
| Training timestep range | Full schedule | Full schedule (`--train-truncation-ratio 1.0`) |
| Loss | Denoising MSE | Min-SNR-weighted MSE + KDE density loss |
| Static hint channels | density + sdf | density (SDF & Smart Init grids disabled by default) |
| Control objective | Global + local correction | Strict local refinement / unclumping |
| Sample start policy | `noise_fn(shape)` | explicit `sqrt(alpha)*x_init + sqrt(1-alpha)*noise` |
| Resampling | None | Optional forward-backward loops (`--resample_jumps`, off by default) |

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
         Control Encoder
                    |
         AdaptiveGateInjection skips (Transform Gates)
                    |
          U-Net noise prediction (trained jointly)
```

When GECCO is enabled, dynamic high-resolution features from the image are extracted via
`grid_sample` at the exact floating-point coordinates of the current point set and appended to the
hint path.

## Smart Init

Smart Init is implemented in-repo in:

* `smart_init.py`

It provides:

* density-based point initialization from the source image via rejection sampling
* conversion of Smart Init points into OT-style offset grids
* explicit forward-noise helper for SDEdit-style start
* debug export of Smart Init points / offsets / grid

*Note: While the network supports rasterizing the Smart Init points into a GPU-rendered hint channel
(Gaussian soft splatting), this channel (`SMART_INIT_FEATURES`) is disabled in the final paper
configuration. The Smart Init is solely used as the physical starting state x_init for the
truncated SDEdit schedule.*

## Training

Training timesteps are drawn from the first `train_truncation_ratio` fraction of the schedule:

```python
truncation_cutoff = max(1, int(eval_timesteps * train_truncation_ratio))
t = torch.randint(0, truncation_cutoff, (batch_size,), device=device)
```

`--train-truncation-ratio` defaults to `1.0`, so the paper models are trained over the **full**
schedule; the truncation is applied at inference only. `--t-sampling logit_normal` concentrates the
draws on low-t (final-spacing) steps instead of sampling uniformly.

The loss is

```text
min(SNR(t), gamma) / SNR(t) * ||eps_theta(x_t, t, c) - eps||^2  +  lambda_KDE * L_KDE(x0_hat, rho)
```

The KDE term compares a Gaussian KDE of the one-step `x_0` decode with the target density on a
`--density-kde-grid` map (default 32). It is applied only where `t < --density-loss-t-frac * T`
(default `0.4`) and is ramped in over `--density-loss-warmup-epochs` (default 5).

Ground-truth points come from the exact `<stem>.npy` coordinates next to each target
(`--points-source npy`, falling back to PNG centroid detection). Points on pure-white background are
dropped and replaced by duplicates of surviving points (`--drop-white-points`, `--white-threshold
255`). The points are then matched to the OT grid and cached as offsets in `processed_offsets/` next
to the target folder.

## Inference

Sampling does not start from pure Gaussian noise.

1. Build Smart Init points and Smart Init offsets.
2. Choose `t_start = --infer-truncation-ratio * T` (default `0.5 * 1000 = 500`), or set it directly
   with `--t_start_step`.
3. Apply explicit forward noise:

```python
x_noisy = sqrt(alpha_cumprod[t_start]) * x_init + sqrt(1 - alpha_cumprod[t_start]) * noise
```

4. Reverse denoise from `t_start - 1` down to `0`.
5. Optionally apply RePaint-style forward-backward resampling (`--resample_jumps N`) within this
   refinement window.

With `--infer-truncation-ratio 1.0` the Smart Init is skipped and the model samples from pure noise
over the full schedule. Active inference uses DDPM-only sampling paths.

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
    --source /path/to/training/Icons-50_1024_GBN/source \
    --target /path/to/training/Icons-50_1024_GBN/target \
    --out control_v4/train_outputs_Icons-50_1024_GBN \
    --base_ckpt_path "" --no-freeze-denoiser \
    --grid-size 32
```

Use `--grid-size 40` for 1600-point datasets. Defaults: 5000 epochs, batch size 16, learning rate
`1e-4`, `--val-split 0.1` (split with seed 42). Training resumes from the latest checkpoint in
`--out` (`--resume-latest`).

Important training flags:

* `--base_ckpt_path`, `--freeze-denoiser` / `--no-freeze-denoiser`
* `--grid-size`
* `--train-truncation-ratio`, `--t-sampling`
* `--density-loss-weight`, `--density-kde-grid`, `--density-loss-t-frac`
* `--min-snr-gamma`
* `--points-source`, `--drop-white-points`, `--white-threshold`
* `--save_every`, `--keep-every`
* `--save-best` / `--no-save-best` (default: no), `--best-max-cv`, `--best-max-clumped-pct`
* *(Note: Smart Init / SDF channel flags are safely defaulted to False in the code)*

Training logs to Weights & Biases (project `Stipple-ControlNet`) and reads `WANDB_API_KEY` from a
`.env` file. Set `WANDB_ACTIVE = False` at the top of `train_control.py` to train offline.

### 2. Sample

```bash
python control_v4/sample_control.py \
    --control_ckpt_path control_v4/train_outputs_Icons-50_1024_GBN/checkpoints/dynamic_ep5000.ckpt \
    --image_path my_photo.png \
    --grid_size 32 \
    --output_dir control_v4/sample_outputs
```

The checkpoint holds both the control branch and the jointly trained base denoiser, so
`--base_ckpt_path` stays empty (the default). It is only needed for frozen-base checkpoints, which
store no denoiser weights.

Important sampling flags:

* `--grid_size` (output has `grid_size^2` points; any square budget works at inference)
* `--infer-truncation-ratio`
* `--t_start_step`
* `--eval_timesteps`
* `--resample_jumps`
* `--n_samples`
* `--show_denoising`, `--show_denoising_interval`
* `--track_time`, `--track_time_full`, `--profile_trace`

`sample_control_advance.py` is an extended variant of the sampler used for batch / dataset runs and
additional diagnostics; it shares the same feature flags and Smart Init / resampling logic. To run a
checkpoint over a whole dataset folder, use `experiments/control_data_gen.py`.

### 3. Overfit

```bash
python control_v4/test_overfit.py --steps 10000 --sample-index 0
```

The overfit path keeps:

* W&B logging
* periodic visualization exports
* geometry scoring (CV / clumped % / score)
* Min-SNR weighted denoising loss (`--min-snr-gamma`, default `5.0`)
* best checkpoint naming with geometry values

Note that `test_overfit.py` still defaults to the older frozen-base setup
(`config/GBN/model.ckpt`, train truncation `0.30`).

### 4. Ablations

`train_control_ablation.py` is a pre-configured training entry point for the architecture ablations
on Icons-50 (GBN). Each variant is a block in its configuration section, writing to
`control_v4/train_outputs_Icons-50_1024_GBN_<variant>`:

| Variant | Base | GECCO | Adaptive gates |
| --- | --- | --- | --- |
| `vanilla` | frozen, pretrained (`config/GBN/model.ckpt`) | off | off |
| `unfrozen` | trained jointly from scratch | off | off |
| `gecco` | frozen, pretrained | on | off |
| `agi` | frozen, pretrained | off | on |
| `full` | trained jointly from scratch | on | on |

The ablation models are trained over the full schedule, and their in-training evaluation samples
start from pure noise (both truncation ratios `1.0`). The *SDEdit* variant is inference-only: it reuses the `full` weights with a Smart Init start
(`--infer-truncation-ratio < 1`).

### 5. Dataset evaluation

`eval_dataset.py` reproduces the exact train/valid split of `train_control.py` and exports the
train-style panel (Condition | GT | Predict | GT Offset Quiver) for selected samples. Checkpoint
tools:

* `lighten_checkpoints.py` -- copies checkpoints with the optimizer state cleared, to save space
* `zip_checkpoints.py` -- archives the ablation checkpoints at a fixed epoch interval

Grid-size runs and timing plots live in `experiments/` (`profiling_test_stage_*.py`,
`grid_size_results_stage_*.py`).

## Output structure

### Sampling

```text
control_v4/sample_outputs/
  image-name/
    conditions/
      image-name_high_res.npy
      image-name_target_density.npy
      ...
      image-name_conditions_collage.png
    png/
      image-name.png                  # final stipple
    npy/
      image-name.npy                  # final points, (N, 2)
    timestamps/
      image-name.txt                  # per-stage timing (--track_time)
      image-name_full.json            # detailed timing (--track_time_full)
    denoising_steps/                  # only with --show_denoising
      png/image-name_step_0000.png
      npy/image-name_step_0000.npy
      ...
```

With `--n_samples > 1`, each output file gets a per-sample suffix.

### Training

```text
control_v4/train_outputs_<Dataset>_<N>_<SAMPLER>/
  run_params.json                     # the exact training arguments
  losses_log.json                     # per-epoch losses
  train_panel_ep<N>.png / val_panel_ep<N>.png
  checkpoints/
    dynamic_ep<N>.ckpt                # periodic checkpoints
    best_dynamic_ep<N>_score..._cv..._clumped....ckpt   # only with --save-best
```

## W&B reporting

Training logs per-epoch metrics and visual panels against `epoch`:

* `metrics/train_loss`
* `metrics/train_density_loss`
* `metrics/valid_loss`
* `metrics/geo_cv`
* `metrics/geo_clumped_pct`
* `metrics/geo_score`
* `visual/train`, `visual/valid` (panels), `visual/compare` (train-vs-valid loss chart), and
  `visual/hint_channels` (the active conditioning channels of the first batch)

## Checkpoints

Two checkpoint families exist:

1. Periodic epoch checkpoints:

* `checkpoints/dynamic_ep<N>.ckpt`
* saved every `--save_every` epochs (default 10); once an epoch is a multiple of `--keep-every`
  (default 100), the earlier ones that are not multiples of it are deleted together with their panels

2. Best geometry checkpoints:

* `checkpoints/best_dynamic_ep<N>_score..._cv..._clumped....ckpt`
* written only with `--save-best` (off by default: the final periodic checkpoint is the one used,
  and the best score is still tracked in every checkpoint's `best_geom_score`)
* gated by geometry thresholds and best score improvement; never pruned

Checkpoint payloads contain:

* `control_net` -- control branch weights
* `denoiser` -- base U-Net weights (`None` when the base was frozen)
* `freeze_denoiser`, `optimizer`, `epoch`, `global_step`
* geometry metadata: `best_geom_score`, `cv_score`, `clumped_score`, `current_geom_score`

## Files

| File | Purpose |
| --- | --- |
| `smart_init.py` | Smart Init generation, rasterization, debug export, explicit forward noising |
| `DynamicControlNet.py` | V4 ControlNet and controlled denoiser wrapper with Adaptive Gates and GECCO |
| `DynamicStippleDataset.py` | Dataset returning image, density, sdf, offsets, and cached Smart Init grid |
| `conditioning.py` | Density + SDF preprocessing |
| `train_control.py` | Training (joint base + control, Min-SNR + KDE loss) with W&B logging |
| `train_control_ablation.py` | Pre-configured script for the architecture ablation checkpoints |
| `train_control_old_configs.py` | Backup of the training configurations used for the submitted runs |
| `sample_control.py` | Sampling with Smart Init SDEdit-style start and optional resampling |
| `sample_control_advance.py` | Extended sampler for batch / dataset runs and diagnostics |
| `test_overfit.py` | Single-example overfit workflow |
| `eval_dataset.py` | Train/valid sample evaluation and panel generation |
| `lighten_checkpoints.py`, `zip_checkpoints.py` | Checkpoint helpers (lighten / zip) |
| `prepare_data.py` | Legacy offline target-to-offsets converter (offsets are now exported by `train_control.py`) |
| `Self_README.md` | Compact technical V4 flow summary |

The geometry metrics (CV / clumping / score) live in `utils/stippling_metrics.py` and
`utils/stippling_metrics_advance.py`.

## Notes

* V4 is intended to be standalone.
* It should not depend on legacy `control_v1`, `control_v2`, `control_v3`, or any `train_free_*` implementation logic at runtime.
* Smart Init is implemented directly inside `control_v4`.
* Auxiliary x0 losses were tested during V4 development (`mse`, `chamfer`, `swd`) and did not improve quality in our runs, so they were removed from active train/overfit codepaths.
