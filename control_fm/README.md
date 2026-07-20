# control_fm — Flow-Matching ControlNet for stippling

`control_fm` is the **Flow-Matching (conditional-OT / rectified-flow)** counterpart of
`control_v4` (which is DDPM-based). It generates OT offset grids conditioned on a grayscale
density image, then decodes them to stipple point sets.

There is **no diffusion process here**: no betas, no `alphas_cumprod`, no ancestral sampler,
no RePaint resampling, no DPM++. The generative process is an ODE (optionally an SDE), and
the network regresses a *velocity field* rather than noise.

---

## 1. The Flow-Matching formulation

```
t in [0, 1],    t = 0 -> data,    t = 1 -> source
x_t = (1 - t) * x_data + t * eps          (linear / OT interpolant)
v   = dx_t/dt = eps - x_data              (constant target velocity)
```

The network predicts `v(x_t, t, cond)`. Sampling integrates `dx/dt = v` from `t = 1` down to
`t = 0`. Because the OT path is linear the target velocity is constant along a trajectory, so
an Euler step is exact when the model is perfect.

`net_t` rescales continuous `t` by `--t-scale` (default 1000) before the sinusoidal timestep
embedding, which was designed for integer DDPM steps (~0..1000).

Exact one-step decode of the clean endpoint: `x0 = x_t - t * v`.

## 2. Couplings — what the source endpoint is

| `--fm-coupling` | source at `t = 1` | meaning |
| --- | --- | --- |
| `gaussian` (default) | `N(0, I)` | standard FM: generate from pure noise |
| `smartinit` | smart-init offsets (+ `--fm-source-jitter-px` noise) | OT bridge: *transport* the clumpy smart-init to the blue-noise target |

Both share the same interpolant and velocity target; only `eps` changes.

**Constraints:** min-SNR weighting is only defined for `gaussian` (see §4), and stochastic
sampling (`--eta > 0`) needs a Gaussian component in the source — `smartinit` with
`--fm-source-jitter-px 0` has none, and the sampler raises rather than silently pushing the
state onto off-path regions the field never saw.

## 3. Two training modes

### (a) Unified / from scratch — `--base-ckpt-path ""` (default)

Base velocity U-Net **and** control branch are trained jointly from random init under the FM
loss, and stored together in the control_fm checkpoint.

### (b) Two-stage frozen base — the control_v4 analogue

Load a **trained** FM base, freeze it, and train only the ControlNet branch on top: the
classic ControlNet paradigm (Zhang et al.) with the generative process swapped DDPM → FM.

```bash
python -m control_fm.train_control \
    --base-ckpt-path config_trained/GBN_FM/model.ckpt \
    --freeze-denoiser --fm-coupling gaussian --conditioning controlnet
```

Mechanics (`flow_matching.load_denoiser_base_weights`):

* The baseline is produced by the repo-root `train.py` / `train_stress.py` with
  `config/GBN/config_fm.json` (a `"flow"` block instead of `"diffusion"`).
* That checkpoint is `{"diffu": <EMA state_dict>, "raw": <raw state_dict>}`; keys carry a
  `model.` prefix from the `FlowMatchingModel` wrapper, which the loader strips.
  `--base-use-raw` selects the non-EMA weights.
* The load is **strict** by default — any missing/unexpected key raises, so a "frozen" base
  can never be silently half-random. `--no-base-strict` overrides.
* Base weights load **before** `DynamicControlNet` is constructed, so the control branch's
  deep copies (`ctrl_conv1`, `ctrl_encoder_layers`, `ctrl_middle`, …) warm-start from the
  trained base rather than from noise.
* Guards: requires `--conditioning controlnet` (the concat/spade builders change input
  channels, so base weights would not align) and `--fm-coupling gaussian` (the base was
  trained noise→data).

A healthy load prints `matched N / N tensors; missing=0 unexpected=0`.

## 4. Loss

Per-sample velocity MSE, optionally reweighted by **min-SNR** (`--min-snr-gamma`, 0 disables):

```
SNR(t) = (1 - t)^2 / t^2                 (linear interpolant, Gaussian source)
x0 = x_t - t*v   =>   MSE_x0 = t^2 * MSE_v
w(t) = t^2 * min(SNR, gamma) = min( (1 - t)^2 , gamma * t^2 )
```

Derived for *this* interpolant — do not copy the DDPM epsilon weighting. Weights are
normalized to batch-mean 1 so enabling this does not move the effective learning rate. Only
defined for `gaussian` coupling; the constructor raises otherwise.
**Validation loss is deliberately unweighted** so val curves stay comparable across gammas.

Timestep distribution (`--t-dist`):

* `uniform` — `t ~ U(eps_t, 1 - eps_t)`
* `logitnormal` (default) — `u ~ N(m, s)`, `t = sigmoid(u)` (SD3); concentrates gradient on
  the harder mid-path. Partly redundant with min-SNR — ablate them separately.

## 5. Conditioning architectures (`--conditioning`)

| value | description |
| --- | --- |
| `controlnet` | dual-branch `DynamicControlNet` + `AdaptiveGateInjection` skips + GECCO. The only mode that supports a frozen pretrained base. |
| `concat` | single-stage: condition channels concatenated to the input (Palette / SR3 style) |
| `spade` | single-stage: SPADE spatially-adaptive normalization |

GECCO (`--enable-gecco`) samples high-res image features via `grid_sample` at the current
point coordinates — the only point-coordinate-aware operator in the stack.

## 6. Sampling

**Deterministic vs stochastic** — same trained field, selected at inference only:

* `--eta 0` (default): probability-flow ODE, `--ode-method euler|heun`
* `--eta > 0`: reverse SDE via Euler–Maruyama (`eta = 1` is canonical). The score is recovered
  in closed form from the predicted velocity, so no extra network is needed. `--ode-method`
  is ignored on this branch.

**Truncation / SDEdit start (`--truncation_ratio`, inference only).** With `gaussian`
coupling and `truncation_ratio < 1`:

```python
x_start    = fm.interpolate(smart_init_offsets, noise, t=truncation_ratio)
t_schedule = linspace(truncation_ratio, 0.0, n_steps + 1)
```

The ODE starts from a *noisy smart-init* at `t = truncation_ratio` and integrates to 0 — the
FM analogue of DDPM SDEdit. Because `t` is simultaneously the interpolation level and the ODE
start time, the "start step vs input noise level" mismatch DDPM required care about is
structurally impossible here.

With `smartinit` coupling truncation is **ignored**: the ODE already starts exactly at the
smart-init at `t = 1`, with no noise blend.

> **Note:** `--truncation-ratio` in `train_control.py` is validated but **not used during
> training** — training always draws `t` over the full range. Truncation is purely a decoding
> choice. Training-time eval panels likewise sample from pure noise at `--t-start`
> (default 1.0) and do not apply the SDEdit start.

## 7. Scripts

### Train

```bash
python -m control_fm.train_control \
    --source /path/to/source \
    --target /path/to/target \
    --offsets /path/to/processed_offsets \
    --conditioning controlnet --fm-coupling gaussian \
    --min-snr-gamma 5.0 --t-dist logitnormal \
    --ode-method euler --ode-steps 50 \
    --epochs 2000 --batch_size 16 --lr 1e-4 \
    --out control_fm/train_outputs
```

Key flags: `--base-ckpt-path` `--base-use-raw` `--base-strict` `--freeze-denoiser`
`--conditioning` `--fm-coupling` `--fm-source-jitter-px` `--min-snr-gamma` `--t-dist`
`--t-logitnorm-m` `--t-logitnorm-s` `--ode-method` `--ode-steps` `--eta` `--t-scale`
`--t-start` `--grid-size` `--enable-gecco` `--enable-adaptive-gate-injection` `--save_every`
`--resume-latest` `--best-max-cv` `--best-max-clumped-pct`

Stability (from the overfit blow-up fix): `--spike-factor` `--spike-ema-beta` `--grad-clip`
`--lr-schedule` `--lr-warmup` `--lr-min`.

### Sample

```bash
python -m control_fm.sample_control \
    --control_ckpt_path control_fm/train_outputs/checkpoints/dynamic_controlnet_fm_ep1000.pt \
    --image_path my_photo.png \
    --conditioning controlnet --fm-coupling gaussian \
    --truncation_ratio 0.30 --ode_steps 50 --ode_method euler \
    --output_dir control_fm/sample_outputs
```

This script mixes `-` and `_` in flag names (e.g. `--fm-coupling` but `--truncation_ratio`,
`--ode_method`) — use exactly what `--help` lists.

### Evaluate a dataset split

```bash
python -m control_fm.eval_dataset \
    --source ... --target ... --offsets ... \
    --conditioning controlnet --fm-coupling gaussian \
    --truncation-ratio 0.30 --control-ckpt <ckpt> --out <dir>
```

### Overfit sanity

```bash
python -m control_fm.test_overfit
```

`--base_config_path` defaults to **`config/GBN/config_fm.json`**. Only its `model` block is
read (the `flow` / `diffusion` blocks are ignored here), so it is interchangeable with
`config/GBN/config.json` as long as the `model` blocks match — pointing at the FM config
keeps the two files free to diverge.

## 8. Checkpoints

Periodic: `checkpoints/dynamic_controlnet_fm_epN.pt` (`--save_every`).
Best-geometry: `checkpoints/best_controlnet_epXXXX_score..._cv..._clumped....pt`.

Payload keys: `control_net`, `denoiser` (base weights, saved even when frozen so the
checkpoint is self-contained for sampling), `optimizer`, `epoch`, `global_step`,
`best_geom_score`, `cv_score`, `clumped_score`, `current_geom_score`.

> Checkpoints written before the denoiser-saving fix lack `"denoiser"`; resuming from one
> prints a warning and restarts the base U-Net from random init.

## 9. Differences from control_v4 (DDPM)

| Aspect | control_v4 | control_fm |
| --- | --- | --- |
| Process | DDPM (betas, ancestral sampling) | Flow Matching (ODE / optional SDE) |
| Network target | noise `eps` | velocity `v = eps - x_data` |
| Base model | frozen `config/GBN/model.ckpt` | from scratch **or** frozen FM base via `--base-ckpt-path` |
| Truncated training | yes (late timesteps only) | **no** — full `t` range always |
| Truncation | training + inference | inference only |
| Resampling | RePaint jumps (`--resample-jumps`) | **removed** |
| DPM++ | tested, removed | n/a |
| Solver | DDPM loop | Euler / Heun ODE, or reverse SDE (`--eta`) |
| Conditioning | ControlNet only | `controlnet` / `concat` / `spade` |

## 10. Files

| File | Purpose |
| --- | --- |
| `flow_matching.py` | `FlowMatching` (interpolant, couplings, `sample_t`, `loss_weight`, score, ODE/SDE sampler), `build_velocity_network`, `load_denoiser_base_weights` |
| `DynamicControlNet.py` | ControlNet branch + controlled denoiser wrapper (Adaptive Gates, GECCO) |
| `single_stage.py` | `concat` / `spade` single-stage conditioners and their builders |
| `DynamicStippleDataset.py` | Dataset: image, density, sdf, offsets, cached smart-init grid/offsets |
| `smart_init.py` | Density-based smart-init generation, rasterization, offset conversion |
| `conditioning.py` | Density + SDF preprocessing |
| `train_control.py` | Training (unified or two-stage frozen base) + W&B |
| `sample_control.py` | Single-image sampling, truncation/SDEdit, timing harness |
| `eval_dataset.py` | Batch evaluation over a dataset split |
| `prepare_data.py`, `build_smart_init_cache.py`, `migrate_cache_names.py` | Data / cache utilities |

## 11. Known gaps

* **No weight EMA on the control branch.** The only EMA here is a scalar *loss* EMA used for
  spike detection. Weight EMA exists in the repo baseline trainer (`utils/Trainer.py`, used by
  `train.py`) and in `control_fm_v2`, but not in this training loop. In the two-stage setup
  the frozen base is already EMA-smoothed; only the control branch trains without it.
* **No bottleneck attention and no geometry loss.** Both live in `control_fm_v2`.
* `--truncation-ratio` is accepted by `train_control.py` but unused (see §6).
