# FM Upgrade Plan — Tier 1 (free wins) + Tier 2 (bottleneck attention)

Target repo: `control_fm/` (Flow-Matching stippling). This plan is for an agent
editing real code. **Diagnose before editing** — locate the actual code path, confirm
it matches the description, then change one thing. Do NOT assume file contents from
this doc; verify against the source.

## GLOBAL RULES (read first)
1. **One change at a time.** After each change, run a short train + sample and eyeball
   output quality. Do not stack all changes blindly.
2. **Do not touch GECCO, Tier 3 (depth/width), or Tier 4 (point-space).** Out of scope.
3. **Preserve resolution-transferability.** The model trains at G=32 and must still
   sample at G=8…112. Any change that adds *learned absolute positional embeddings* is
   forbidden — it breaks this property. (See analysis §7.1.)
4. **Measure three things after Tier 2**, not just quality: (a) sample quality, (b) wall
   time at G=32, (c) grid-transfer at G=64 and G=96 (does it still run + stay clean).
5. Keep the active config as-is otherwise: `conditioning=controlnet`,
   `fm_coupling=gaussian`, `truncation_ratio=0.30`, `ode_method=heun`, `eta=0.0`,
   `grid_size=32`, `FREEZE_DENOISER=False` (unified training stays unified).

---

# TIER 1 — Free wins (do these FIRST, in this order)

Each is a known quality lever that is currently disabled. Expectation: these may close
much of the FM-vs-DDPM gap with zero architecture change. Validate after each.

## 1.1 — Weight EMA (highest priority)

**Problem (analysis §8):** config specifies `ema_decay: 0.995` but `control_fm` never
applies weight EMA — the only EMA is on the *loss* for spike detection. A weight EMA
implementation exists in `utils/Trainer.py` but the control training loop doesn't use it.

**Steps:**
1. Locate the training loop in `train_control.py` and the existing weight-EMA helper in
   `utils/Trainer.py`. Confirm whether the helper is importable/reusable as-is.
2. Maintain an EMA copy of the **trainable** params (control branch + denoiser, since
   both train jointly). Update after each optimizer step:
   `ema_param = decay * ema_param + (1 - decay) * param`, with `decay = 0.9999`
   (start higher than the config's 0.995 for late-stage smoothing; make it a config key).
3. **CRITICAL — actually USE the EMA weights for sampling/eval and checkpointing.** The
   common bug is maintaining EMA weights but sampling from the raw weights. All eval
   panels, geometry metrics, and saved "best" checkpoints must be generated from the EMA
   weights. Add a context-manager or swap-in/swap-out around eval + final save.
4. Save both raw and EMA state dicts in the checkpoint payload (the payload already lists
   `control_net`, `denoiser`, etc. — add `control_net_ema`, `denoiser_ema`).
5. Consider an EMA warmup (don't start EMA until a few hundred steps in) so early-random
   weights don't poison the average.

**Validate:** sample the same fixed held-out images from raw vs EMA weights at equal
step count. EMA should look cleaner / less noisy. If EMA is worse, decay is too high or
warmup missing.

## 1.2 — logit-normal t sampling

**Problem (analysis §8):** training samples `t ~ U(1e-4, 1-1e-4)`. SD3's ablation favors
logit-normal `t`, concentrating training on the harder mid-range.

**Steps:**
1. Find where `t` is drawn in the FM training step (analysis §2: `t ~ U(...)` feeding the
   interpolant `x_t = (1-t)·x0 + t·ε`).
2. Replace uniform draw with logit-normal: sample `u ~ N(m, s)`, then `t = sigmoid(u)`.
   Start `m=0.0, s=1.0`. Clamp to `[1e-4, 1-1e-4]` to match current guard.
3. Make `m`, `s`, and a boolean `t_dist: {uniform|logitnormal}` config keys so it's
   ablatable and reversible.

**Validate:** train short, compare quality vs the EMA-only run. Keep if better or neutral.

## 1.3 — SNR / min-snr loss weighting

**Problem (analysis §8):** `min_snr_gamma` is parsed but never applied; loss is plain
per-sample velocity MSE.

**Steps:**
1. Locate the velocity-MSE loss in the FM training step.
2. Apply a per-sample weight derived from the FM interpolant's SNR at `t`. For the linear
   interpolant `x_t=(1-t)x0+tε`, define the weight consistently with min-SNR (clamp the
   effective SNR by `gamma`, default `gamma=5.0`). **Confirm the exact SNR→weight mapping
   for the rectified-flow interpolant** — do not copy a DDPM epsilon-loss weighting
   blindly; the interpolant differs (analysis §2). If unsure, implement it as an
   explicitly-documented weighting function and expose `min_snr_gamma` as the config knob
   that is currently dead.
3. Wire `min_snr_gamma` from config into the loss (currently parsed, unused).

**Validate:** short train, compare. This one is the least certain of the three — if it
destabilizes or doesn't help, revert and proceed; EMA + logit-normal are the safer wins.

### Tier 1 checkpoint
Run all three together, sample held-out set, compare to the pre-Tier-1 FM baseline. Record
the quality delta. This is the new, fair FM baseline before any architecture change.

---

# TIER 2 — Bottleneck attention (position-encoding-free)

**Goal:** give the network a global receptive field at the bottleneck so it can coordinate
large-offset point routing (analysis §7.4: conv is local in grid-space, repulsion is local
in point-space; they diverge at large offsets — FM's regime). Bottleneck-only keeps it
cheap and grid-safe.

**Non-negotiable constraints:**
- Attention goes at the **middle/bottleneck ONLY** (`attn_middle`-equivalent). Do NOT add
  attention at the high-res encoder/decoder levels — that scales quadratically in the
  full token count and destroys speed at large grids.
- **NO learned absolute positional embeddings.** Start with **no positional encoding** in
  the attention (the conv trunk already carries position; the middle attention only needs
  global mixing). This guarantees grid-transfer. (If later you find you need position
  info, use *relative* bias or RoPE — never a learned absolute table.)

**Steps:**
1. In `models/Denoiser.py` (`DenoiserModel`), locate the `middle` block (analysis §5:
   `ResnetBlock → ResnetBlock`, currently no attention because `attn_middle=false`).
2. Implement a self-attention module that operates on the bottleneck feature map
   `(B, C, h, w)`:
   - reshape to tokens `(B, h*w, C)`,
   - multi-head self-attention (start 4 heads), **no positional encoding**,
   - reshape back to `(B, C, h, w)`,
   - residual add, with the **output projection zero-initialised** (so the block starts as
     identity and training is stable — consistent with the codebase's zero-init warm-start
     philosophy in analysis §7.2).
3. Insert it between the two middle ResnetBlocks: `ResBlock → Attn → ResBlock`.
4. Gate it on the existing `attn_middle` config flag (currently inert per analysis §5).
   Set `attn_middle: true`. Confirm `DenoiserModel.__init__` still accepts `attn_layers`
   (leave `[]` — no high-res attention).
5. **The control branch (`DynamicControlNet`) deep-copies the denoiser's `middle`**
   (analysis §4c). Confirm whether the copied middle should also get attention. Likely yes
   for symmetry — verify the deep-copy picks up the new module, or replicate it in the
   control middle. Check this explicitly; a mismatch here is a silent bug.

**Why zero-init output:** the model + all its injections currently start as near-identity
(analysis §7.2). A zero-init attention output preserves that, so adding attention can't
destabilize the existing trained behavior — it only helps once gradients flow.

### Tier 2 validation (MANDATORY — all three)
1. **Quality:** sample held-out set, compare to the Tier-1 baseline. Expect improvement on
   images with large offsets / tight dark structures (where §7.4 tension bites).
2. **Grid-transfer:** train at G=32, then sample at **G=64 and G=96**. Confirm it (a) runs
   without shape errors (it will, if no learned pos-enc), and (b) quality does not collapse
   at unseen sizes. If quality collapses at G=96 but runs, the attention learned a
   size-specific pattern — investigate before proceeding.
3. **Speed:** measure wall-time at G=32 and G=96, compare to the Tier-1 baseline. Bottleneck
   attention token count stays small (G=96 → ~24×24 bottleneck), so the added cost should
   be minor. If large-grid time explodes, confirm the attention is really bottleneck-only
   and not accidentally applied at a high-res level.

---

# ORDER OF OPERATIONS (summary)
1. 1.1 Weight EMA (+ actually sample from EMA) → validate
2. 1.2 logit-normal t → validate
3. 1.3 SNR weighting → validate (revert if it destabilizes)
4. Establish Tier-1 FM baseline (fair comparison point)
5. 2.x Bottleneck attention, no pos-enc, zero-init output → validate quality + grid-transfer + speed

# CONFIG KEYS TO ADD/ACTIVATE
- `train.ema_decay` (wire it in — currently ignored), `train.ema_warmup_steps`
- `train.t_dist` (`uniform`|`logitnormal`), `train.t_logitnorm_m`, `train.t_logitnorm_s`
- `diffusion.min_snr_gamma` (wire the already-parsed value into the loss)
- `model.attn_middle: true` (currently inert; make it actually add bottleneck attention)

# DO NOT
- Do not add high-res attention.
- Do not add learned absolute positional embeddings.
- Do not touch GECCO, ch_mult, ch (Tier 3/4 — out of scope).
- Do not change coupling/sampler/truncation.
- Do not stack all changes before validating each.
