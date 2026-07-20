# control_fm_v2 — Flow Matching stippling (config-free, Tier 1 + Tier 2)

`control_fm_v2` generates stipple point sets from a grayscale image using **Flow Matching**
(rectified flow / conditional-OT), not diffusion. It is the successor to `control_fm` (v1),
adding the Tier-1/Tier-2 upgrades from `FM_tier1_tier2_implementation_guide.md`, an auxiliary
geometry loss, minibatch-OT coupling, and architecture stamping.

There is **no DDPM anywhere in this package** — no betas, no noise schedule, no ancestral
sampler, no frozen `model.ckpt` prior. The base velocity net and the control branch are
Flow-Matching weights trained together from scratch.

---

## Core convention

```
t in [0, 1],    t = 0 -> data,    t = 1 -> source
x_t = (1 - t) * x_0 + t * eps          (linear / OT interpolant)
v   = dx_t/dt = eps - x_0              (constant target velocity)
```

The network predicts the **velocity** `v(x_t, t, cond)`. Sampling integrates the ODE from
`t = 1` down to `t = 0`. Because the path is linear, an Euler step is exact when the model is
perfect — which is why the overfit sanity check converges in very few steps.

The state is a **2-channel OT offset field** `(B, 2, G, G)`; a point's coordinate is
`grid_center + offset / G`.

---

## What changed vs `control_fm` (v1)

| | v1 | v2 |
|---|---|---|
| Architecture source | `config/GBN/config.json` | **`MODEL_CONFIG`** in `flow_matching.py` (no config file) |
| `t` sampling | uniform | uniform **or logit-normal** (Tier 1.2) |
| Loss weighting | plain velocity MSE | **min-SNR** derived for this interpolant (Tier 1.3) |
| Weight EMA | none | **`WeightEMA`**, actually used for eval/geometry/checkpoints (Tier 1.1) |
| Bottleneck attention | none | **zero-init, position-encoding-free** block (Tier 2) |
| Coupling | gaussian / smartinit | + **`otbatch`** (minibatch OT-CFM) |
| Geometry objective | none | optional **density-warped spacing loss** on one-step-decoded points |
| Checkpoint identity | untracked | **`arch` stamp**; samplers rebuild from it |

---

## Architecture (`MODEL_CONFIG`)

Defined once in `flow_matching.py`; there is no config file to keep in sync.

```python
MODEL_CONFIG = {
    "num_channels": 2,      # the (dx, dy) offset field; `concat` widens this to 2 + n_cond
    "out_ch": 2,            # predicted velocity has the state's shape
    "ch": 128,
    "ch_mult": [1, 2, 3],   # 128 -> 256 -> 384; 3 levels, 2 downsamples (G=32 -> 8)
    "num_res": 2,
    "dropout": 0.1,
    "resamp_with_conv": True,
    "attn_middle": True,    # v2's OWN bottleneck attention (NOT DenoiserModel's)
    "attn_heads": 4,
}
```

`DenoiserModel` also requires `attn_layers` and its own `attn_middle`. Both are **hard-wired
inert** inside `_denoiser_kwargs()` and deliberately kept out of `MODEL_CONFIG`:

* `attn_layers = []` — high-res attention is out of scope (quadratic in token count, and it
  would break grid transfer).
* `attn_middle = False` — `DenoiserModel`'s own block is single-head and its output projection
  is **not** zero-initialised. v2 attaches its own instead.

---

## Couplings (`--fm-coupling`)

The coupling picks the **source endpoint at `t = 1`** and, for `otbatch`, how it is paired with
the data during training.

| value | source at t=1 | notes |
|---|---|---|
| `gaussian` | `N(0, I)` | standard FM, noise -> data. Independent pairing. |
| `smartinit` | smart-init offsets (+ `--fm-source-jitter-px`) | OT bridge: learns to *move* the clumpy smart-init to the target rather than denoise. |
| `otbatch` | `N(0, I)`, **reordered by minibatch OT** | OT-CFM (Tong et al. 2023). Training-only change. |

### Why `otbatch` exists

With independent pairing, a given `x_t` is consistent with many `(x_0, eps)` pairs, so the
regressed velocity target is a **conditional mean** over all of them. Averaging many valid
blue-noise displacements yields a displacement that produces *no* blue noise — the smoothed /
mushy layouts FM tends to produce. `otbatch` permutes the noise within the batch to minimize
`sum_i ||x_0[i] - eps[pi(i)]||^2` (exact Hungarian assignment), so paths cross far less and the
target becomes near-deterministic.

Marginally `eps` is unchanged (a relabelled set of `N(0, I)` samples), so **SNR, score, SDE and
the inference noise-start are identical to `gaussian`** — only the training pairing differs. It
is a no-op for batch < 2, so it does nothing in the single-example overfit test. Larger batches
give a stronger OT effect. Requires `scipy`.

---

## Tier 1 upgrades

**1.1 — Weight EMA** (`ema.py`). `WeightEMA(modules, decay=0.9999, warmup_steps=500)` keeps a
shadow copy and exposes an `averaged()` context manager. Critically, the EMA weights are the
ones **actually used** for eval sampling, geometry scoring and checkpointing — the classic bug
is maintaining an EMA and then sampling from the raw weights. Checkpoints carry
`control_net_ema` / `denoiser_ema` / `ema` so a run can resume without re-warming.

**1.2 — logit-normal `t`** (`--t-dist logitnormal`). `u ~ N(m, s)`, `t = sigmoid(u)`, clamped to
`(eps_t, 1 - eps_t)`. Concentrates training mass on the harder mid-path (SD3).

**1.3 — min-SNR loss weighting** (`--min-snr-gamma`, 0 disables). Derived for *this* interpolant
— do **not** copy the DDPM epsilon weighting. Since `x_0 = x_t - t*v` exactly, a velocity error
`dv` maps to `dx_0 = -t*dv`, so `MSE_x0 = t^2 * MSE_v` and the weight on the velocity MSE is:

```
w(t) = t^2 * min(SNR(t), gamma) = min( (1 - t)^2 , gamma * t^2 )
```

Weights are renormalised to batch-mean 1 so enabling this does not silently change the LR.
**Only valid when the source is marginally `N(0, I)`** — `gaussian` or `otbatch`; it raises for
`smartinit`, where "signal-to-noise" has no meaning. Note `w(t) -> 0` as `t -> 1`, i.e. exactly
where gaussian sampling *starts*, so this is the least certain Tier-1 change — ablate it alone.

---

## Tier 2 — bottleneck attention (`attention.py`)

A multi-head self-attention block inserted **between the two middle ResnetBlocks**
(`ResBlock -> Attn -> ResBlock`), giving the network a global receptive field where conv is
local. Two non-negotiable properties:

* **Zero-initialised output projection** — the block starts as exact identity, so adding it
  cannot destabilise training; it only contributes once gradients flow.
* **No positional encoding of any kind** — no learned absolute position table. This is what
  preserves **resolution transferability**: train at `G = 32`, sample at `G = 8…112`.

Bottleneck-only keeps it cheap (at `G = 96` the bottleneck is ~24x24 tokens).

> **Ordering constraint.** `build_velocity_network()` attaches attention to `denoiser.middle`,
> and `DynamicControlNet` **deep-copies** that middle. Always build the velocity network
> *before* the control branch, or the control branch silently lacks attention.

---

## Geometry loss (`geometry_loss.py`) — optional

Velocity-MSE is a per-point regression error: two point configurations can share a velocity
loss and differ wildly in spacing quality, so the model can drive v-MSE down while still
clumping. Nothing in the velocity loss says *"these two points are too close together."*

This module supplies that signal cheaply — **no ODE solve, no trajectory unroll**:

```
x0_pred = x_t - t * v_pred            # exact one-step decode of the interpolant
points  = offsets_to_coords(x0_pred)
L_geo   = lambda * w(t) * spacing(points)   [+ lambda_cap * w(t) * capacity(points)]
```

The spacing term is **target-free**: it warps pairwise distances by local density
(Wei & Wang / metric M5) so the criterion is capacity-aware, then applies a one-sided hinge on
warped mass below `r_pack = sqrt(2/sqrt(3)) ~ 1.0746`. No `gbn_bar.pt` reference set needed.

**`w(t)` must concentrate on low `t`** — `x0_pred` is a rough guess near the noise end and
accurate near data. Default is a hard mask (`t < GEO_T_MAX`). The module documents why a hard
mask beats a smooth ramp here: since `dL/dv = dL/dx0 * (-t)`, the geometry gradient scales with
`t` and vanishes at `t = 0`, so a `(1-t)^k` ramp puts its largest weight exactly where the
gradient is smallest.

Enabled via `GEO_WEIGHT > 0` (`--geo-weight`); a warmup + linear ramp avoids a step-change in
the total loss that the spike detector would otherwise read as a bad batch. It is
**training-only and deliberately not part of `ARCH_KEYS`** — it changes no weight shape, so a
checkpoint trained with it stays loadable by a sampler that knows nothing about it.

> Validate on the **hard** scipy/FFT validators in `utils/stippling_metrics_advance.py`, never
> on the soft surrogate — otherwise you are optimising a proxy.

---

## Conditioning architectures (`--conditioning`)

| value | mechanism |
|---|---|
| `controlnet` | Dual-branch `DynamicControlNet` + `AdaptiveGateInjection` skips + **GECCO** dynamic feature sampling (`grid_sample` of high-res image features at the *current* point coordinates). |
| `concat` | Single-stage: density map concatenated onto the input channels at `conv1` (Palette/SR3 style). |
| `spade` | Single-stage: density modulates every ResnetBlock's normalisation via spatially-adaptive `gamma/beta`. |

GECCO exists **only** on the `controlnet` path — it is a hint-encoder input feature, and the
single-stage conditioners have no hint encoder. SPADE in particular routes all conditioning
through normalisation from a static grid-aligned map, so a dynamic per-point feature has
nowhere to enter.

---

## Sampling

Integrate `dx/dt = v` from `t_start` down to 0. Euler or Heun (`--ode-method`).

**Deterministic vs stochastic** (`--eta`, inference-time only, no retraining):

* `eta = 0` — probability-flow ODE (deterministic).
* `eta = 1` — canonical reverse SDE. The score is recovered in closed form from the predicted
  velocity, so no extra network is needed.

Stochastic sampling requires a Gaussian component in the source (`sigma > 0`); it raises for
`smartinit` with `--fm-source-jitter-px 0`, where the score does not exist.

**Truncation / SDEdit** (`--truncation-ratio`). With `gaussian`/`otbatch` and a ratio `< 1`,
sampling starts from a **noised smart-init** at `t = ratio` — `x_start = (1-r)*smart_init +
r*noise` — and integrates `r -> 0`. Because `t` is simultaneously the interpolation level and
the ODE start time, the start point and the noise level *cannot* desync (unlike DDPM, where they
were two separate quantities). With `smartinit` coupling truncation is ignored: the ODE already
starts exactly at the smart-init at `t = 1`.

> **Truncation is inference-only in FM.** It is validated but never read by the training loop —
> the model always trains on the full `t` range. This differs from `control_v4` (DDPM), where the
> ratio also restricted training timesteps.

---

## Architecture stamping (`arch.py`)

Checkpoints embed an `arch` dict so a sampler cannot silently rebuild a mismatching network
(this has bitten the codebase twice — a coupling default and an attention flag).

```python
ARCH_KEYS = ("conditioning", "fm_coupling", "attn_middle", "attn_heads", "enable_gecco",
             "enable_adaptive_gate_injection", "smart_init_features", "sdf_features",
             "batch_coords_features", "concat_smart_init_grid")
```

Samplers call `resolve_arch(...)`: the checkpoint stamp is authoritative and anything overridden
is printed. `--no-arch-from-ckpt` / `--strict-arch` are the escape hatches. **`grid_size` is
deliberately excluded** — sampling at a different grid than training must stay legal.

---

## Checkpoints and disk hygiene

Two families, written to `<out>/checkpoints/`:

1. **Periodic** — `dynamic_controlnet_fm_ep{N}.pt`, every `--save_every` epochs.
2. **Best-geometry** — `best_controlnet_ep{N}_score..._cv..._clumped....pt`, gated on the
   CV / clumped% score improving.

Payloads carry `arch`, `control_net`, `denoiser`, the EMA shadows, optimizer, `global_step`, and
geometry metadata.

**`KEEP_EVERY` pruning.** Once an epoch is a multiple of `KEEP_EVERY` (default 100), earlier
periodic checkpoints *and their train/val panels* whose epoch is **not** a multiple of
`KEEP_EVERY` are deleted. With `SAVE_EVERY=10, KEEP_EVERY=100`, epochs 10..90 are pruned once
epoch 100 lands, keeping 100, 200, … `best_*` checkpoints are **never** touched. Set
`--keep-every 0` to disable; it must be 0 or a multiple of `--save_every`.

On startup the pruner also re-runs at the highest keep-boundary already on disk, so a deletion
interrupted mid-way (process killed) is finished on the next launch rather than lingering.

---

## Scripts

### Train

```bash
python -m control_fm_v2.train_control \
    --source /path/to/source \
    --target /path/to/target \
    --offsets /path/to/processed_offsets \
    --out control_fm_v2/train_outputs_run \
    --conditioning controlnet \
    --fm-coupling gaussian \
    --epochs 2000 --batch_size 16 --lr 1e-4
```

Most knobs are also editable as constants at the top of the file (the config-block style used
throughout this repo). Useful flags: `--fm-coupling`, `--t-dist`, `--min-snr-gamma`,
`--use-ema/--no-use-ema`, `--attn-middle/--no-attn-middle`, `--geo-weight`, `--keep-every`,
`--freeze-denoiser`, `--truncation-ratio`, `--eta`, `--ode-method`.

### Sample / evaluate / overfit

```bash
python -m control_fm_v2.sample_control  --control_ckpt <ckpt> --image my.png
python -m control_fm_v2.eval_dataset    --control-ckpt <ckpt>
python -m control_fm_v2.test_overfit    --steps 10000 --sample-index 0
```

---

## Files

| File | Purpose |
| --- | --- |
| `flow_matching.py` | `MODEL_CONFIG`, `FlowMatching` (interpolant, couplings incl. `otbatch`, `t` sampling, min-SNR, score/SDE, ODE sampler), `build_velocity_network` |
| `attention.py` | Tier-2 zero-init, position-encoding-free `BottleneckAttention` + `attach_bottleneck_attention` |
| `ema.py` | `WeightEMA` with warmup and an `averaged()` swap-in context manager |
| `geometry_loss.py` | Auxiliary density-warped spacing (and optional capacity) loss on one-step-decoded points |
| `arch.py` | `ARCH_KEYS` stamping, `arch_from`, `assert_arch_matches`, `resolve_arch` |
| `train_control.py` | Training loop, EMA/geometry wiring, panels, geometry scoring, checkpointing + `KEEP_EVERY` pruning |
| `sample_control.py` | Sampling (ODE/SDE, truncated SDEdit start), arch-driven rebuild |
| `eval_dataset.py` | Dataset-wide evaluation / metrics |
| `test_overfit.py` | Single-example overfit sanity check |
| `single_stage.py` | `concat` and `spade` conditioners + their velocity-net builders |
| `prepare_data.py`, `build_smart_init_cache.py`, `migrate_cache_names.py` | Data prep and cache utilities |

### Shared with v1 (imported, not duplicated)

`DynamicControlNet`, `DynamicStippleDataset`, `smart_init` and `conditioning` are imported from
**`control_fm`** — they are unchanged between versions.

> ⚠️ `control_fm_v2/` also contains *copies* of those four files
> (`DynamicControlNet.py`, `DynamicStippleDataset.py`, `smart_init.py`, `conditioning.py`).
> They have **zero importers** and are dead. Edit the `control_fm/` originals; the local copies
> are safe to delete.

---

## Gotchas

* **Build order.** `build_velocity_network()` before `DynamicControlNet(...)` — the control
  branch deep-copies the denoiser's middle (and, in the two-stage path, its weights).
* **min-SNR needs a Gaussian-marginal source.** `gaussian` or `otbatch` only. Under `otbatch`
  the derivation is approximate (`eps|x_0` is OT-biased), but it is kept enabled so an OT run
  changes exactly one variable vs the gaussian baseline.
* **`otbatch` is training-only.** Sampling is identical to `gaussian`; both start from noise.
* **Truncation does nothing during training** (see Sampling).
* **No learned absolute positional embeddings anywhere** — this is what keeps grid transfer
  legal. Do not add any.
* **Geometry loss is a surrogate.** Report the hard validators, not `geo/*`.
