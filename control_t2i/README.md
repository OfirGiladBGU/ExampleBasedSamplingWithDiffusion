# control_t2i — Lightweight T2I Adapter for Stipple Diffusion

Compact adapter-style control module that conditions the GBN diffusion denoiser
on a high-resolution grayscale image and a target density map, without SDF,
Smart Init hint channels, or heavy copied encoder blocks.

---

## Design

| Property | Value |
|---|---|
| Conditioning inputs | `high_res_image (1,H,W)` + `target_density (1,G,G)` |
| SDF / Smart Init hints | None |
| Adapter params (~27M denoiser) | ~6M (gecco=True) |
| Training timestep range | `t ∈ [0, truncation_cutoff)` where `cutoff = ratio × 1000` |
| Default truncation ratio | `0.30` (steps 0–299) |
| Inference start | Pure noise at `t_start = truncation_cutoff` (SDEdit) |
| Optimizer | AdamW split: adapter @ `5e-4`, denoiser @ `1e-5` |
| Denoiser training | Joint (not frozen); denoiser lr matches GBN baseline |

The adapter is a lean encoder mirror of the denoiser's `conv1` + `encoder_layers`,
with `ZeroConv2d` injections at each level so training starts at zero perturbation.
Optional GECCO dynamic feature sampling adds ~6K extra parameters.

---

## Files

| File | Purpose |
|---|---|
| `LightweightAdapter.py` | `DynamicLightweightAdapter`, `LightweightControlledDenoiser` |
| `DynamicStippleDataset.py` | Dataset: returns `(high_res, target_density, offsets)` |
| `train_control.py` | Full training script with wandb, val panels, geometry checkpointing |
| `test_overfit.py` | Single-pair overfit sanity check with SDEdit inference |
| `sample_control.py` | Inference script: load checkpoint → generate stipple point set |
| `smart_init.py` | Rejection-sampling Smart Init (used only for SDEdit start point) |

---

## Training

```bash
# from project root
python control_t2i/train_control.py \
    --config  config/GBN/config.json \
    --source  /path/to/source \
    --offsets /path/to/processed_offsets \
    --epochs  2000 \
    --batch_size 16 \
    --lr 5e-4 \
    --denoiser-lr 1e-5 \
    --truncation-ratio 0.30 \
    --out control_t2i/train_outputs
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--lr` | `5e-4` | Adapter learning rate |
| `--denoiser-lr` | `1e-5` | Denoiser lr — keep at baseline to avoid NaN |
| `--truncation-ratio` | `0.30` | Fraction of timestep range to train on |
| `--enable-gecco` / `--no-enable-gecco` | `True` | Dynamic high-res feature sampling |
| `--min-snr-gamma` | `5.0` | Min-SNR loss weighting (`0` disables) |
| `--resume-latest` | `True` | Auto-resume from latest checkpoint in `--out` |
| `--save_every` | `10` | Checkpoint + panel export frequency (epochs) |

---

## Overfit test (single image)

```bash
python control_t2i/test_overfit.py --steps 5000
python control_t2i/test_overfit.py --steps 5000 --sample-index 42 --vis-every 200
```

Edit `DATA_ROOT` at the top of `test_overfit.py` to point at a `source/` + `target/` folder.
No pretrained weights are loaded — trains from scratch on one image pair.

---

## Inference

```bash
python control_t2i/sample_control.py \
    --control_ckpt control_t2i/train_outputs/checkpoints/t2i_ep100.pt \
    --input-image  /path/to/input.png \
    --out-dir      control_t2i/sample_outputs \
    --timesteps    1000 \
    --truncation-ratio 0.30
```

Inference uses SDEdit-style truncation: Smart Init offsets (rejection-sampled dark pixels)
are noised to `t_start` and denoised for `t_start` steps — not from pure noise.

---

## NaN prevention

The denoiser diverges if trained at a high lr (e.g. `5e-4`). Always keep
`--denoiser-lr` at `1e-5` (the GBN baseline). The split-optimizer design makes
this explicit and separate from the adapter lr.

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/DynamicControlNet.py)

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_control.py#1-1), lines 1 to 320

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_control.py#321-321), lines 321 to 700

I’ve loaded the core architecture and most of training; next I’m reading the model-construction/checkpoint sections to wire the new adapter without breaking resume, eval, and saving behavior.

Searched for regex `DynamicControlNet|DynamicControlledDenoiser|control_net|freeze|AdamW|torch.save|state_dict|resume|checkpoint|trainable` (`**/ExampleBasedSamplingWithDiffusion/control_v4/train_control.py`), 92 results

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_control.py#700-700), lines 700 to 1300

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/test_overfit.py#1-1), lines 1 to 260

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/__init__.py)

I confirmed where resume/saving/gradient logic lives; now I’m reading the final checkpoint block so I can switch keys and filenames cleanly for v5.

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v4/train_control.py#1280-1280), lines 1280 to 1418

I’m now implementing the lightweight adapter itself, and I’m quickly checking the base denoiser block structure so the adapter outputs exactly match the expected control shapes.

Searched for regex `class Denoiser|encoder_layers|downsamp_layers|def forward\(self, x, t, cond=None, controls=None\)` (`**/ExampleBasedSamplingWithDiffusion/models/**/*.py`), 13 results

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/models/Denoiser.py)

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/models/Layers.py)

I’m creating a clean control_v5 scaffold by copying only the core runnable scripts from v4, then I’ll patch them to use the new lightweight adapter and v5 checkpoint conventions.

Ran terminal command:  set -e
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
mkdir -p control_v5
cp control_v4/train_control.py control_v5/train_control.py
cp control_v4/sample_control.py control_v5/sample_control.py
cp control_v4/conditioning.py control_v5/conditioning.py
cp control_v4/DynamicStippleDataset.py control_v5/DynamicStippleDataset.py
cp control_v4/smart_init.py control_v5/smart_init.py
cp control_v4/cleanup_train_outputs.sh control_v5/cleanup_train_outputs.sh
: > control_v5/__init__.py
ls -1 control_v5

Ran terminal command:  cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion && ls -1 control_v5

Created [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/LightweightAdapter.py)

Progress update: the new adapter module is in place. I’m now wiring control_v5/train_control.py to use it end-to-end (imports, optimizer, resume/save keys, and mandatory joint denoiser training).

Searched for regex `freeze_denoiser|FREEZE_DENOISER|control_net|DynamicControlNet|DynamicControlledDenoiser|dynamic_controlnet_v4_ep|best_controlnet|"control_net"|v4-train` (`**/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py`), 46 results

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py#1-1), lines 1 to 120

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py#430-430), lines 430 to 520

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py#840-840), lines 840 to 1045

Ran terminal command:  cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion && \
perl -0pi -e 's/from control_v4\.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser/from control_v5.LightweightAdapter import DynamicLightweightAdapter, LightweightControlledDenoiser/g; s/from control_v4\.DynamicStippleDataset import DynamicStippleDataset/from control_v5.DynamicStippleDataset import DynamicStippleDataset/g; s/from control_v4\.smart_init import add_noise_at_t/from control_v5.smart_init import add_noise_at_t/g; s/control_v4/train_outputs/control_v5/train_outputs/g; s/python control_v4/train_control\.py/python control_v5/train_control.py/g; s/--out control_v4/control_out/--out control_v5/control_out/g; s/v4-train-/v5-train-/g; s/DynamicControlNet/DynamicLightweightAdapter/g; s/DynamicControlledDenoiser/LightweightControlledDenoiser/g; s/control_net/adapter/g; s/dynamic_controlnet_v4_ep/lightweight_adapter_v5_ep/g; s/best_controlnet/best_adapter/g; s/"control_net"/"adapter"/g' control_v5/train_control.py

Searched for regex `freeze_denoiser|FREEZE_DENOISER|DynamicLightweightAdapter|LightweightControlledDenoiser|adapter|lightweight_adapter_v5_ep|"adapter"|control_v4` (`**/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py`), 45 results

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py#1-1), lines 1 to 120

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py#840-840), lines 840 to 1045

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py#1320-1320), lines 1320 to 1418

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py#740-740), lines 740 to 815

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py#228-228), lines 228 to 256

Generating patch (96 lines) in [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py)


```

```


Searched for regex `freeze_denoiser|FREEZE_DENOISER|ControlNet V4|Trainable ControlNet|control \+ denoiser` (`**/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py`), no results

The training migration is mostly done; I’m now wiring control_v5/sample_control.py so inference loads adapter checkpoints and wrapper classes with backward-compatible key handling.

Searched for regex `control_v4|DynamicControlNet|DynamicControlledDenoiser|CONTROL_CKPT|dynamic_controlnet_v3|_extract_control_state_dict|control_net` (`**/ExampleBasedSamplingWithDiffusion/control_v5/sample_control.py`), 25 results

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/sample_control.py#1-1), lines 1 to 130

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/sample_control.py#330-330), lines 330 to 460

Searched for regex `control_v4\.|DynamicControlNet|DynamicControlledDenoiser|dynamic_controlnet_v|"control_net"|FREEZE_DENOISER|freeze-denoiser` (`**/ExampleBasedSamplingWithDiffusion/control_v5/**/*.py`), 16 results

Generating patch (49 lines) in [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/sample_control.py)


```

```


Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/DynamicStippleDataset.py#1-1), lines 1 to 70

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/DynamicStippleDataset.py#176-176), lines 176 to 225

Generating patch (21 lines) in [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/DynamicStippleDataset.py)


```

```
I’ve finished the core wiring edits; next I’m running diagnostics across the new v5 files to catch any syntax or reference issues before you run training.

Checked [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/LightweightAdapter.py), [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py), [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/sample_control.py), [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/DynamicStippleDataset.py), no problems found

Searched for regex `control_v4\.|DynamicControlNet|DynamicControlledDenoiser|dynamic_controlnet_v|"control_net"` (`**/ExampleBasedSamplingWithDiffusion/control_v5/**/*.py`), 4 results

Read [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/config/GBN/config.json)

Read changed files in [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion)

Generating patch (13 lines) in [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/cleanup_train_outputs.sh)


```

```


Generating patch (10 lines) in [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/__init__.py)


```

```


Generating patch (6 lines) in [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/DynamicStippleDataset.py)


```

```


Generating patch (6 lines) in [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/sample_control.py)


```

```
I’ve applied final naming and compatibility cleanup; I’m doing one last diagnostics pass to confirm the new v5 Python modules are clean.

Checked [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/LightweightAdapter.py), [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/train_control.py), [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/sample_control.py), [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/DynamicStippleDataset.py), [](file:///groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/control_v5/__init__.py), no problems found

Implemented. I created a new control_v5 stack and wired it for a lightweight adapter with joint denoiser training, as requested.

What I added

1. New lightweight adapter module with zero-conv injections and drop-in controls output
- LightweightAdapter.py
- Includes:
  - DynamicLightweightAdapter
  - LightweightControlledDenoiser
  - Zero-initialized 1x1 projections per encoder control and middle control
  - GECCO + static hint processing path matching v4 conditioning inputs
  - Programmatic channel extraction from denoiser encoder blocks

2. New control_v5 training pipeline migrated from v4 and integrated to adapter
- train_control.py
- Key changes:
  - Uses DynamicLightweightAdapter and LightweightControlledDenoiser imports
  - Joint training is enforced (denoiser always trainable)
  - Optimizer always tracks adapter + denoiser params
  - Checkpoint key switched to adapter
  - Resume/save naming switched to lightweight_adapter_v5_epN.pt
  - Resume supports backward compatibility via fallback to control_net key if present

3. New control_v5 inference/sample pipeline migrated to adapter
- sample_control.py
- Key changes:
  - Uses control_v5 modules
  - Loads adapter checkpoints (with fallback support for control_net/model_state_dict/state_dict)
  - Uses LightweightControlledDenoiser wrapper

4. Supporting control_v5 package files
- DynamicStippleDataset.py
- conditioning.py
- smart_init.py
- __init__.py
- cleanup_train_outputs.sh

Validation status

1. No Python diagnostics reported in:
- LightweightAdapter.py
- train_control.py
- sample_control.py
- DynamicStippleDataset.py
- __init__.py

Channel multiplier answer

For your current base U-Net config:
- ch_mult in config: [1, 2, 3]
- base ch: 128 from config.json
- effective encoder channels: [128, 256, 384]

Notes

1. I noticed unrelated existing changes in the repo (new images under control_v4/sample_test) and did not touch/revert them.
2. I did not run a full training job yet, only static diagnostics.

Natural next steps

1. Run a short smoke train in control_v5 for 50-200 steps and verify trainable parameter counts and loss trend.
2. Run one sampling pass with a saved v5 checkpoint through sample_control.py.
3. If you want, I can also clone and migrate test_overfit.py into control_v5 with the same adapter interface.

Made changes.