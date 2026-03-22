# train_free_v4

Unconditional blue-noise generation followed by post-hoc OT/CDF warp.

## What it does

1. Runs the frozen diffusion model unconditionally (`cond=None`) to generate uniform blue-noise points.
2. Converts offsets to points in `[0,1]^2`.
3. Applies a warp backend:
   - `none`: identity (baseline)
   - `cdf`: separable 2D CDF inverse warp

## Run

From repo root:

```bash
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
```

### Baseline (`none`)

```bash
python train_free_v4/sample_ot_warp.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --base_ckpt config/GBN/model.ckpt \
  --config config/GBN/config.json \
  --backend none \
  --timesteps 1000 \
  --n_samples 1 \
  --output_dir train_free_v4/sample_outputs
```

### CDF warp

```bash
python train_free_v4/sample_ot_warp.py \
  --image /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/train_free_v2/examples/taksim-circle.pgm.png \
  --base_ckpt config/GBN/model.ckpt \
  --config config/GBN/config.json \
  --backend cdf \
  --warp_grid_size 256 \
  --density_mode dark \
  --density_gamma 2.0 \
  --timesteps 1000 \
  --n_samples 1 \
  --output_dir train_free_v4/sample_outputs
```

## Useful flags

- `--density_mode {dark,light}`: which intensities attract points.
- `--density_gamma`: density sharpening, e.g. `1.0`, `2.0`, `3.0`.
- `--disable_cdf_interpolation`: debug mode (expect more grid artifacts).
- `--timesteps`: reduce for speed (`200`), increase quality (`1000`).

## Outputs

Files are written with a timestamp prefix:
- `*_uniform.npy`
- `*_warped.npy`
- `*_uniform.png`
- `*_warped.png`
- `*_comparison.png`
