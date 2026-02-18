# ControlNet Extension for Example-Based Sampling with Diffusion

This extension adds a **ControlNet adapter** to the base diffusion model, enabling grayscale-image-conditioned stipple generation. Given a grayscale photograph or tone map, the model generates a 2D point set (stippling) whose local density follows the tonal values of the input image.

The pretrained diffusion model is kept frozen. Only the lightweight control branch is trained, so the original unconditional generation capability is fully preserved.

## Architecture

```
                           ┌────────────────────────────────┐
  grayscale image ───────► │  Hint Encoder (1ch → 2ch)      │
                           └──────────────┬─────────────────┘
                                          │  + x_noisy
                                          ▼
                           ┌────────────────────────────────┐
                           │  Trainable Control Encoder      │
                           │  (deepcopy of pretrained        │
                           │   encoder + middle block)       │
                           └──────────┬─────────────────────┘
                                      │
                              7 x ZeroConv (init = 0)
                                      │
                  ┌───────────────────┼───────────────────────┐
                  │  Frozen UNet      │                        │
                  │  ┌────────┐       ▼                        │
  x_noisy ──────►│  │Encoder │──► Middle ──► Decoder ──► out  │
                  │  └────────┘   (+ ctrl)   (+ ctrl skips)    │
                  └────────────────────────────────────────────┘
```

- **16M trainable parameters** (control branch only)
- **27M frozen parameters** (locked base UNet)
- Channel progression: `[128, 256, 384]` with 2 ResBlocks per level
- 7 zero convolutions ensure the control branch starts as a no-op

## Prerequisites

Use the `qmcdiffusion` conda environment (same as the base repo), plus OpenCV for the data preparation step:

```bash
conda activate qmcdiffusion
pip install opencv-python
```

## Dataset

Training uses the **Wave** subset (8,903 pairs) extracted from `data_grads_v3_1024`:

```
/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/
├── source/           # 8,903 grayscale condition images
├── target/           # 8,903 corresponding stippled dot images
├── processed_offsets/  # (created by step 1 below)
└── prompt.json       # JSONL metadata
```

## Workflow

### 1. Prepare training data

The preparation script extracts dot centroids from the stippled target images and converts them to the `(2, 32, 32)` offset grid format used by the diffusion model, using the repo's existing optimal-transport mapping.

All paths default to the Wave dataset, so you can simply run:

```bash
python prepare_data.py
```

Or override explicitly:

```bash
python prepare_data.py \
    --gt_images /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/target \
    --output    /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets \
    --grid_size 32
```

### 2. Train the ControlNet

All paths default to the Wave dataset and the GBN pretrained model:

```bash
python train_control.py
```

Or with explicit arguments:

```bash
python train_control.py \
    --config     config/GBN/config.json \
    --ckpt       config/GBN/model.ckpt \
    --cond       /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/source \
    --offsets    /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_wave_1024/processed_offsets \
    --epochs     100 \
    --batch_size 16 \
    --lr         1e-4 \
    --out        control_out
```

| Argument       | Description                                      |
|----------------|--------------------------------------------------|
| `--config`     | Base model config (defines UNet architecture)    |
| `--ckpt`       | Pretrained diffusion model checkpoint            |
| `--cond`       | Directory of grayscale condition images           |
| `--offsets`    | Directory of `.npy` offset files from step 1     |
| `--epochs`     | Number of training epochs                        |
| `--lr`         | Learning rate (default `1e-4`)                   |
| `--save_every` | Save checkpoint every N epochs (default `10`)    |
| `--out`        | Output directory for checkpoints and TensorBoard |

Monitor training with TensorBoard:

```bash
tensorboard --logdir control_out
```

### 3. Generate conditioned stipple patterns

```bash
python sample_control.py \
    --config       config/GBN/config.json \
    --base_ckpt    config/GBN/model.ckpt \
    --control_ckpt control_out/controlnet_ep100.pt \
    --image        my_photo.png \
    --batch        16 \
    --timesteps    1000 \
    --output       stippled.npy
```

| Argument         | Description                                     |
|------------------|-------------------------------------------------|
| `--config`       | Base model config                               |
| `--base_ckpt`    | Pretrained diffusion model checkpoint           |
| `--control_ckpt` | Trained ControlNet checkpoint from step 2       |
| `--image`        | Input grayscale image (condition)               |
| `--batch`        | Number of point sets to generate                |
| `--timesteps`    | Diffusion sampling steps (default `1000`)       |
| `--no_ot`        | Skip inverse OT (save raw offset grids)         |
| `--output`       | Output `.npy` file                              |

The output `.npy` contains point sets of shape `(batch, 1024, 2)` with coordinates in `[0, 1]^2`.

## Files added / modified

| File                      | Type     | Purpose                                          |
|---------------------------|----------|--------------------------------------------------|
| `models/Denoiser.py`      | Modified | Added `controls` parameter to `forward()` (backward-compatible) |
| `models/ControlNet.py`    | New      | `ControlNet`, `ControlledDenoiser`, `ZeroConv2d` |
| `data/StippleDataset.py`  | New      | PyTorch Dataset for (condition, offset) pairs    |
| `prepare_data.py`         | New      | Convert stippled images to offset tensors        |
| `train_control.py`        | New      | ControlNet training script                       |
| `sample_control.py`       | New      | Conditioned inference script                     |

## Notes

- The base model's unconditional sampling (`sample.py`) is completely unaffected by the changes.
- The `ControlledDenoiser` wrapper is designed to be a drop-in replacement inside the existing `DiffusionModel` sampling loop -- no changes to `Diffusion.py` were needed.
- Zero convolutions ensure that at initialization, the ControlNet produces zero output, so the frozen model behaves identically to the original until training begins to adjust the control branch.
