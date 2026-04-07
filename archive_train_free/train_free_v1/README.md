# Training-Free Stipple Generation V1

Training-free, density-adaptive stipple generation using a **quadtree + guided diffusion** pipeline. No ControlNet, no adapter training -- the frozen unconditional diffusion model is used as-is.

## Motivation

ControlNet approaches (V1--V3) attempted to teach a neural network to map image density to point offsets. This failed for complex images because the OT grid representation scrambles local neighbourhoods under non-uniform density, destroying the CNN's spatial inductive bias. The frozen U-Net learns a uniform-grid shortcut and ignores the control signal.

This approach sidesteps the problem entirely: density is encoded geometrically by the **quadtree cell sizes**, and the frozen model only generates uniform blue noise inside each cell -- the one thing it was trained to do perfectly.

## Key Differences from ControlNet Approaches

| Aspect | ControlNet (V1--V3) | Train-Free V1 |
|--------|---------------------|---------------|
| Training required | Yes (adapter network) | None |
| Density encoding | Neural network learns from image | Quadtree cell sizes (geometric) |
| Model usage | Conditioned via control signals | Unconditional, frozen, per-cell |
| Grid size | 32x32 (1024 pts, single global grid) | 8x8 (64 pts per cell, many cells) |
| Boundary handling | N/A (single grid) | x0-hat repulsion guidance |
| Failure mode | Uniform-grid local minimum | Boundary artifacts (known V1 limitation) |

## Architecture

```
              ┌──────────────────────────┐
              │  Grayscale Image (HxW)   │
              └────────────┬─────────────┘
                           │
                    Quadtree Splitting
              (capacity-constrained, offline)
                           │
              ┌────────────┴─────────────┐
              │  K leaf cells             │
              │  each: (x, y, w, budget)  │
              │  + neighbour adjacency    │
              └────────────┬─────────────┘
                           │
                  Guided Diffusion Loop
                  (T reverse steps)
              ┌────────────┴─────────────┐
              │ For each step t:          │
              │  1. noise_pred = UNet(y_t)│
              │  2. x0_hat = predict(y_t) │
              │  3. global = affine(x0)   │
              │  4. E = repulsion(global) │
              │  5. grad = d(E)/d(y_t)    │
              │  6. clamp grad [-1, 1]    │
              │  7. y_{t-1} = DDPM - λ·g  │
              └────────────┬─────────────┘
                           │
                  Post-hoc Culling
              (subsample to cell budget)
                           │
              ┌────────────┴─────────────┐
              │  Final point set (N, 2)   │
              └──────────────────────────┘
```

### Quadtree Splitting

- Recursively divides the image into square cells based on pixel darkness
- `cell_budget = total_budget * (cell_darkness / total_darkness)`
- Splits when `budget > 64` (the model's native 8x8 = 64 points)
- Skips cells with `budget < 0.5` (white/near-white regions)
- Enforces a minimum cell size to prevent infinite recursion
- Tracks neighbour adjacency for the boundary energy computation

### x0-hat Boundary Guidance

At each reverse-diffusion step, the predicted clean offsets (x0_hat) are mapped to global image coordinates via a differentiable affine transform. A repulsion energy penalises cross-cell point overlap:

```
E = Σ max(0, r - dist)²     over all neighbour-cell point pairs
```

The repulsion radius `r` is adaptive per boundary: `r = 0.5 * (spacing_A + spacing_B)`, where `spacing = cell_width / 8`. This handles multi-resolution boundaries where adjacent cells differ in size.

The gradient `d(E)/d(y_t)` is computed via `torch.autograd.grad`, clamped to `[-1, 1]`, and subtracted from the standard DDPM step:

```
y_{t-1} = DDPM_step(y_t) - λ_t * clamp(grad)
```

### Post-hoc Culling

The model always generates 64 points per cell. Cells with `budget < 64` have their points randomly subsampled after the diffusion loop completes. Blue noise's uniform spatial frequencies make random subsampling safe.

## Prerequisites

Same conda environment as the base model:

```bash
conda activate qmcdiffusion
pip install opencv-python
```

## Usage

```bash
python train_free_v1/sample_trainfree.py \
    --image  path/to/grayscale.png \
    --budget 1024 \
    --timesteps 200 \
    --lambda-scale 50 \
    --seed 42
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | (required) | Path to grayscale source image |
| `--budget` | 1024 | Total number of stipple points |
| `--timesteps` | 200 | Diffusion reverse steps |
| `--lambda-scale` | 50.0 | Guidance strength multiplier |
| `--grad-clip` | 1.0 | Hard clamp on guidance gradient |
| `--chunk-size` | 256 | Max cells per U-Net forward pass |
| `--max-points` | 64 | Points per cell (8x8 grid) |
| `--min-cell-pixels` | 4 | Minimum cell size in pixels |
| `--skip-threshold` | 0.5 | Skip cells below this budget |
| `--seed` | 42 | Random seed |
| `--output` | auto | Output directory |

## Files

| File | Purpose |
|------|---------|
| `quadtree.py` | Capacity-constrained quadtree with neighbour tracking |
| `guided_sampling.py` | Custom guided diffusion loop with boundary repulsion |
| `sample_trainfree.py` | Main entry point: image to stipple points |
| `README.md` | This file |

## Outputs

Running the script produces:

- `points.npy` -- final point set, shape `(N, 2)` in `[0, 1]^2`
- `result.png` -- three-panel visualisation (source, points, overlay)
- `quadtree.png` -- quadtree cell boundaries with per-cell budgets
- `config.json` -- all parameters and timing info

## Known Limitations (V1)

1. **Boundary gaps**: repulsion-only energy prevents overlap but cannot fill dead-zone strips between cells. A future version could add mild attraction toward expected spacing.
2. **Density quantisation**: density is encoded by cell size, so it cannot vary smoothly within a single cell.
3. **Square image only**: the quadtree assumes a square image. Rectangular images should be padded or cropped first.
4. **No post-assembly relaxation**: a few iterations of Lloyd's algorithm on the global point set would smooth residual boundary artifacts.
