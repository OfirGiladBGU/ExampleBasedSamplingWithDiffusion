# Phase 2 ? WVS<->GBN style-conditioned model: run guide

Gate 0 passed: WVS<->GBN differ in **spacing-regularity at fixed density** (non-decomposable,
density-controlled `norm_nn_cv`, visible as GBN contour-banding vs WVS uniform fill). Phase 2
trains ONE model conditioned on a continuous style scalar `s` (per-sample normalized `norm_nn_cv`),
jointly on both oracles, injected additively into the timestep embedding (zero-init, position-free).

Run everything from the PROJECT ROOT (the dir containing control_v4/, control_v4_mix/, config/,
models/, utils/), same convention as control_v4.

## Files (control_v4_mix/)
- `spacing_regularity.py`      -- the s descriptor (from Gate 0)
- `precompute_style_s.py`      -- computes per-icon normalized s -> style_s.json
- `StyleStippleDataset.py`     -- (image, oracle) samples with high_res, target_density, offsets, s
- `DynamicControlNetStyle.py`  -- control net + style MLP added to temb (+ inference wrapper)
- `train_control_style.py`     -- fork of control_v4/train_control.py, style-conditioned
- `verify_style_model.py`      -- zero-init identity / wiring / gradient checks (no data needed)

## Step 1 ? precompute the style values s
```
python control_v4_mix/precompute_style_s.py
```
Writes `control_v4_mix/style_s.json` (+ `_hist.png`). It auto-exports offsets for both oracle roots
if missing (reusing the training exporter). SANITY: WVS s-mean ~ 0, GBN s-mean ~ 1, and the
"WVS above GBN-median fraction" ~ 0 (two clean clusters). If not, stop ? the conditioning signal is
wrong. Use `--limit N` for a quick check first.

## Step 2 ? verify the style branch wiring (no training data)
```
python control_v4_mix/verify_style_model.py
```
Must print PASS for all three: [1] zero-init identity (s has no effect at init -> starts at the
unconditioned baseline), [2] wiring (s changes the control after perturbation), [3] gradient reaches
the style MLP.

## Step 3 ? train (matches the UNIFIED recipe: UNFROZEN base)
```
python control_v4_mix/train_control_style.py --no-freeze-denoiser \
    --out control_v4_mix/train_outputs_style_wvs_gbn
```
Data paths default to WVS `icons-50_512_WVS` + GBN `icons-50_512_GBN` (override with
`--wvs-root/--gbn-root`, `--style-s-json`). The dataset emits 2xN samples (both oracles per icon);
the split is by SOURCE icon so an icon's WVS and GBN never straddle train/val. Everything else
(truncated-t, min-SNR, EMA/checkpoints, W&B, geometry scoring) is inherited from control_v4.

Smoke first (few epochs / small subset via a tiny --style-s-json, or just watch the first epoch):
confirm the "Style dataset: ... per-oracle counts = {'WVS': .., 'GBN': ..}" line, that loss
decreases, and that a checkpoint saves.

## Step 4 ? re-verify G-transfer after adding the style branch
Run the existing G in {32,48,64} inference check on a style checkpoint (the style MLP is a global
scalar->vector add to temb and introduces NO positional dependence, but re-verify per the plan).

## Deferred (next phase, once a checkpoint exists)
Fork the sampler/eval to accept `--style-s` (the inference wrapper `DynamicControlledStyleDenoiser`
already threads it) and render `s in {0, 0.25, 0.5, 0.75, 1.0}`. Validate:
- **Endpoint fidelity** vs the single-teacher baselines (s=0 ~ WVS, s=1 ~ GBN).
- **Interpolation monotonicity**: measured `norm_nn_cv` of the output tracks requested s (plot).
- **Capacity preserved**: density/tone + spacing quality stay at endpoint level across the range.
- **Visual coherence**: intermediates read as between uniform-fill and contour-banding.
Watch the three failure modes: snapping (step, not ramp), muddy mixture (capacity degrades mid-range),
invisible interpolation (metric moves but eye cannot see it).
