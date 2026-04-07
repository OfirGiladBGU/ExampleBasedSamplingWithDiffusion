# train_free_v4

Stable unconditional blue-noise generation followed by a post-hoc transport warp.

## Version Summary

- `v4.1`: unconditional diffusion + separable 2D CDF warp.
- `v4.2/v4.3`: unconditional diffusion + joint 2D Sinkhorn grid OT warp. **Abandoned.** Sinkhorn diverged at sharp epsilon, and the de-shrink fix only partially helped. Axis-aligned corrugation was unacceptable.
- `v4.4` (current): unconditional diffusion + CDF warp + Lloyd/CCVT relaxation.

## How v4.4 Works

The core insight: the CDF warp does 95% of the work (global density routing) instantly. We only need local isotropic re-spacing, which is exactly what Lloyd relaxation does.

**Three-stage pipeline:**

1. **Unconditional U-Net** — Run the frozen diffusion model `cond=None` for the full reverse schedule. Because there is no in-loop guidance fighting the prior, the model produces its cleanest possible isotropic blue-noise sample.
2. **CDF warp** — Routes each uniform point to the dark regions of the target image by inverting the marginal and conditional CDFs of the target density. Fast (no iterations). Produces the correct density profile but introduces axis-aligned stretching.
3. **Lloyd relaxation** — Runs the CDF-warped points through `DifferentiableLloydStep` (from `sinkhorn_lloyd_losses.py`) for `lloyd_relax_steps` iterations. Each step moves each point toward its density-weighted Voronoi centroid, eliminating the axis-separable stretching while preserving the target density.

Why this works where v3 failed: Lloyd relaxation is run **outside** the diffusion loop on a complete point set, so it is never fighting Langevin noise. It converges smoothly to the isotropic equilibrium.

## Backends

- `none`: identity baseline, no density warp
- `cdf`: CDF warp only (v4.1 baseline, fast, shows some stretching)

The Lloyd relaxation is controlled via `--lloyd_relax_steps` and is independent of the backend choice.

## Run

From repo root:

```bash
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
```

### Baseline (no warp, no relaxation)

```bash
/home/ofirgila/.conda/envs/qmcdiffusion/bin/python train_free_v4/sample_ot_warp.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --base_ckpt config/GBN/model.ckpt \
  --config config/GBN/config.json \
  --backend none \
  --lloyd_relax_steps 0 \
  --timesteps 1000 \
  --output_dir train_free_v4/sample_outputs
```

### v4.1: CDF warp only

```bash
/home/ofirgila/.conda/envs/qmcdiffusion/bin/python train_free_v4/sample_ot_warp.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --base_ckpt config/GBN/model.ckpt \
  --config config/GBN/config.json \
  --backend cdf \
  --warp_grid_size 256 \
  --density_mode dark \
  --density_gamma 2.0 \
  --lloyd_relax_steps 0 \
  --timesteps 1000 \
  --output_dir train_free_v4/sample_outputs
```

### v4.4: CDF warp + Lloyd relaxation (recommended)

```bash
/home/ofirgila/.conda/envs/qmcdiffusion/bin/python train_free_v4/sample_ot_warp.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --base_ckpt config/GBN/model.ckpt \
  --config config/GBN/config.json \
  --backend cdf \
  --warp_grid_size 256 \
  --density_mode dark \
  --density_gamma 2.0 \
  --lloyd_relax_steps 15 \
  --lloyd_tau 0.005 \
  --lloyd_grid_size 64 \
  --timesteps 1000 \
  --output_dir train_free_v4/sample_outputs
```

## Useful Flags

- `--density_mode {dark,light}`: which image intensities attract points.
- `--density_gamma`: sharpen density peaks (try `1.0`, `2.0`, `3.0`).
- `--warp_grid_size`: CDF solver resolution (higher = sharper density match).
- `--lloyd_relax_steps`: number of Lloyd iterations post-warp (default `15`, set `0` to disable).
- `--lloyd_tau`: Lloyd softmax temperature (lower = sharper Voronoi boundaries, default `0.005`).
- `--lloyd_grid_size`: Lloyd solver grid resolution (default `64`).
- `--timesteps`: reduce for speed, keep `1000` for the cleanest base sample.

## Outputs

Files are written with a timestamp prefix:

- `*_uniform.npy`
- `*_warped.npy`
- `*_uniform.png`
- `*_warped.png`
- `*_comparison.png`
