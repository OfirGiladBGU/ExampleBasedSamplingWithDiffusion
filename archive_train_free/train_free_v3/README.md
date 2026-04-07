# train_free_v3

Version: v3.2

In-loop CCVT/Lloyd guidance for frozen diffusion.

Status: experimental and currently not working reliably.

This folder is being kept in the repository as a documented research branch.
At the moment, `train_free_v3` does not consistently achieve both correct density routing and stable isotropic blue-noise structure, so it should not be treated as a working final method.

## Version Notes

### v3.2 (current)

This README and implementation describe v3.2.

Key updates in v3.2:
- Reused the existing `DifferentiableLloydStep` implementation as the barycenter engine via a wrapper API.
- Enforced continuous coordinate conversion between diffusion space and image space to avoid discrete snapping artifacts.
- Kept the VRAM-safe flow by detaching `x0_hat` before geometry guidance operations.
- Reduced default guidance aggressiveness (`--lambda_scale=0.05`) to avoid tearing points out of the diffusion manifold.
- Added optional RePaint-style resampling jumps (`--resample-jumps`, `--jump-length`) to give points more migration time without increasing lambda.
- Restored spring-physics guidance magnitude (`g = x - b`) by removing vector normalization.
- Updated run commands and tuning guidance for safe-first execution.

Current interpretation of v3.2:
- better grounded mathematically than v3.1
- easier to debug than earlier variants
- still not reliable enough to recommend as a successful pipeline
- preserved mainly for reproducibility and future investigation

### v3.1 (failed behavior)

v3.1 was a partial prototype that failed qualitatively for the following reasons:
- Guidance scale was too aggressive by default (high lambda), which caused manifold tearing and unstable point motion.
- Barycenter guidance was not sufficiently constrained to continuous image-domain behavior, increasing risk of quantization/grid-like artifacts.
- Early runs showed “shattered” or clumped point structures instead of stable isotropic blue-noise neighborhoods.

In short: v3.1 proved the integration path, but its default dynamics were too strong and not robust enough for first-run stability.

## Current Recommendation

- Keep `train_free_v3` as an archived experimental branch.
- Use it only for controlled comparisons and debugging.
- Do not present it as a working solution.
- Prefer `train_free_v4` as the more promising branch for continued development.

## What it does

At each reverse diffusion step:
1. Predicts `x0_hat` from `x_t` using frozen U-Net.
2. Detaches from U-Net graph (VRAM-safe).
3. Converts offsets to absolute points `[0,1]^2`.
4. Computes continuous differentiable barycenters using `DifferentiableLloydStep`.
5. Applies guidance `g = x - b` to the DDPM posterior mean.

Supported guidance modes:
- `lloyd`
- `ccvt`
- `repulsion`

These modes are still useful for experiments, but none of them are currently validated as robust on the target images we care about.

## Run

From repo root:

```bash
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
```

### Case A: balanced-force test (low lambda, no jumps)

Use this as a diagnostic run, not as a recommended production configuration.

```bash
python train_free_v3/sample_ccvt.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --base_ckpt config/GBN/model.ckpt \
  --config config/GBN/config.json \
  --ccvt_mode ccvt \
  --lambda_scale 0.5 \
  --ccvt_grid_size 64 \
  --tau 0.01 \
  --num_steps 1 \
  --timesteps 1000 \
  --debug_guidance \
  --debug_every 100 \
  --output_dir train_free_v3/sample_outputs
```

### Case B: low lambda + resample jumps (more travel time)

Use this to test whether extra denoise/re-noise cycles help migration. This is still experimental and not known to fix the method.

```bash
python train_free_v3/sample_ccvt.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --base_ckpt config/GBN/model.ckpt \
  --config config/GBN/config.json \
  --ccvt_mode ccvt \
  --lambda_scale 0.5 \
  --ccvt_grid_size 64 \
  --tau 0.01 \
  --num_steps 1 \
  --resample-jumps 2 \
  --jump-length 10 \
  --timesteps 1000 \
  --debug_guidance \
  --debug_every 100 \
  --output_dir train_free_v3/sample_outputs
```

### Lloyd mode

```bash
python train_free_v3/sample_ccvt.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --ccvt_mode lloyd \
  --lambda_scale 0.05 \
  --timesteps 1000
```

### Repulsion mode

```bash
python train_free_v3/sample_ccvt.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --ccvt_mode repulsion \
  --repulsion_strength 0.2 \
  --repulsion_radius 0.03 \
  --lambda_scale 0.05 \
  --timesteps 1000
```

## Useful tuning notes

- Start with `ccvt` and `lambda_scale=0.5`.
- If output tears or over-collapses, reduce to `0.1`.
- If output ignores density and stays too uniform, increase to `1.0`.
- Enable `--resample-jumps 2 --jump-length 10` before raising lambda aggressively.
- Keep `--grad_clip 0` unless you see numerical explosions.
- Increase `tau` for smoother, less aggressive assignments.
- Increase `num_steps` to refine barycenter targets more per diffusion step.
- Use `repulsion` if local collapse appears.

Important: these are diagnostic tuning suggestions, not a known-good recipe.

## Known Failure Modes

- Uniform-grid fallback:
  guidance is too weak relative to the frozen U-Net, so the output stays close to unconditional sampling.

- Central collapse / fuzzy blob:
  guidance is too strong or too globally coherent, so points collapse toward dense regions without preserving blue-noise spacing.

- Manifold tearing:
  large guidance steps pull points out of the learned diffusion manifold and destroy local point structure.

- Resample jumps:
  helpful for experimentation, but not yet demonstrated to make v3 reliably work.

## Outputs

Timestamped outputs in `train_free_v3/sample_outputs`:
- `*.npy` points
- `*_points.png`
- `*_comparison.png`

Treat these outputs as experiment artifacts, not final-quality results.
