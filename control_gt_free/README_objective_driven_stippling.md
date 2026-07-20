# Objective-Driven (GT-Free) Stippling — Implementation Plan

## Goal

Replace **imitation training** (learn to copy GBN/WVS outputs, which caps quality at the
teacher) with **objective-driven training**: the diffusion/flow sampler descends a
*differentiable quality loss computed directly on its own output points*, with **no
per-sample ground-truth target**. This is meant to break the student–master ceiling —
the model optimizes the same objective the analytic methods optimize, amortized into one
sampler.

**Thesis to test:** a single learned sampler, trained GT-free, matches or beats
GBN/WVS/BNOT on the *hard* validation metrics while staying visually clean (no clumping).

**Fallback that is still publishable:** even if it doesn't beat the analytic methods, a
differentiable amortized sampler that comes close *without a teacher* demonstrates a
capability analytic methods lack (differentiability, single-checkpoint amortization).

---

## NON-NEGOTIABLE GUARDRAILS (read before writing any code)

1. **Soft losses are TRAINING-ONLY.** The differentiable surrogates (soft-Voronoi,
   soft-histogram PCF) are optimization signals. They are **never** reported as results.
2. **Hard metrics are the ONLY source of truth.** All reported numbers come from the
   existing numpy/scipy validators in `stippling_metrics_advance.py`
   (`compute_m1_cvt_energy`, `compute_m2_capacity_constraint`, `compute_m5_spatial_measure`,
   `plot_visual_m5_spectrum`). Do not modify these functions — they are the gold standard.
3. **Visual check is mandatory at every evaluation.** Metric wins mean nothing until
   ~20 outputs are rendered and inspected for clumping. (A prior metric bug already
   produced a false "we win" — do not trust numbers alone.)
4. **The kill-test (Phase 1) must pass before any full training run (Phase 2).**
5. Do **not** refactor, rename, or "improve" the existing metric/GBN code. Add new files;
   leave validators untouched.

---

## Repo anchors (existing code to reuse — do not rewrite)

| Purpose | Location |
|---|---|
| Hard capacity metric (validator) | `stippling_metrics_advance.py:157` `compute_m2_capacity_constraint` |
| Hard CVT energy (validator) | `stippling_metrics_advance.py:338` `compute_m1_cvt_energy` |
| Hard spatial measure / warp def | `stippling_metrics_advance.py:281` `compute_m5_spatial_measure` (warp at :304–322) |
| Hard radial spectrum (validator) | `stippling_metrics_advance.py:1280` `plot_visual_m5_spectrum` |
| Differentiable pairwise distances | GBN block `torch.cdist` at `:454` / `:483` (wrapped in `@torch.no_grad()` — drop that in the NEW copy) |
| Density warp factor `D_x` | reuse formula from M5 (`D_x = N * local_rho/mean_rho`) |

> Note: the numpy M5/spectrum functions are NN/FFT-based and **cannot** be reused as the
> differentiable loss. They stay as validators only.

---

## New files to create

```
losses/
  soft_membership.py     # soft-Voronoi weights w_i(x); shared by cap + cvt
  loss_capacity.py       # L_cap_soft  (differentiable delta_c surrogate)
  loss_cvt.py            # L_cvt_soft  (reuses w_i)
  loss_pcf.py            # soft_bin() + density-warped PCF + L_pcf_soft
  composite.py           # L_total = weighted sum; returns dict of components
train/
  objective_train.py     # GT-free training loop (from-noise primary)
  tau_schedule.py        # softmax temperature anneal
eval/
  validate_hard.py       # runs scipy/FFT validators + dumps visual grid
  precompute_gbn_bar.py  # GBN/BNOT composite-loss + PCF target, computed ONCE
config/
  objective.yaml         # all knobs (weights, tau schedule, grid res, pcf source)
```

---

## Loss specifications

### Soft-Voronoi membership (shared)
```
# over density-grid points x (possibly downsampled), points s_i
w_i(x) = softmax_i( -||x - s_i||^2 / tau )
```
Compute ONCE per step; feed both capacity and CVT.

### L_cap_soft  (differentiable delta_c; validator = :157)
```
c_i   = sum_x rho(x) * w_i(x)
L_cap = mean( (c_i / mean(c_i) - 1)^2 )
```

### L_cvt_soft  (validator = :338)
```
L_cvt = sum_i sum_x rho(x) * w_i(x) * ||x - s_i||^2
```

### L_pcf_soft  (validator = :281 / :1280) — the one genuinely new piece
```
D       = torch.cdist(pts, pts)              # copy from GBN block, NO no_grad
D_x     = N * local_rho / mean_rho           # M5 warp; START stop-grad on this factor
D_warp  = D * sqrt(D_x)
pcf     = soft_bin(D_warp, edges, kernel=Gaussian)   # kernel-splat, NOT np.histogram
L_pcf   = || pcf - pcf_target ||             # target: EMPIRICAL from GBN (default)
```
`soft_bin`: each pairwise distance contributes a Gaussian bump across radial bins
(differentiable). Exclude self-distance (diagonal). This function is also the PCF
**metric**, so write it once, clean.

### Composite
```
L_total = w_cap*L_cap + w_cvt*L_cvt + w_pcf*L_pcf
```
Return each component separately for logging.

---

## Key design decisions (defaults set; flip in config)

- **`pcf_target` = EMPIRICAL from GBN** (default). Compute the density-warped PCF of a
  GBN reference set ONCE (`precompute_gbn_bar.py`), average, freeze as target. This is
  "reference not comparison" — model matches an *aggregate statistical signature*, never a
  per-sample target, so no student–master cap. (Alt: analytic blue-noise curve.)
- **Loss grid = DOWNSAMPLED** (default). Soft-Voronoi is O(N x G) memory; do not use full
  density resolution for the loss. Cap grid (e.g. 32x32 or 48x48) for gradients; hard
  validator still runs at full res.
- **Init = FROM NOISE** (default, primary). Prior flow-matching-from-smart-init runs showed
  a train/val generalization gap (init-coupling overfit). Smart-init is an **ablation**,
  re-tested under the GT-free loss (which removes the coupling that caused the overfit).
- **Warp gradient = STOP-GRAD** on `D_x` initially. Gradient flows through `torch.cdist`
  only. Add `grid_sample` differentiable rho-lookup ONLY if the warp needs to move.

---

## tau (softmax temperature) schedule — this is the whole ballgame

- Too large: mushy membership, capacity under-enforced, points clump.
- Too small: hard assignment, vanishing gradients, training stalls.
- **Anneal:** start warm (stable global signal), decay toward hard as training progresses
  so the final solution matches the scipy validator.
- **Log the soft-vs-hard capacity gap every eval.** If they diverge, tau is too large and
  the loss is optimizing something the validator won't credit.

---

## Phase 1 — KILL-TEST (do this FIRST, ~1 afternoon, gate for everything else)

Purpose: collapse most of the uncertainty cheaply before committing to full training.

1. Load current imitation-trained checkpoint **A**. Keep inference (rejection prior +
   truncated schedule + resample) **identical**.
2. `precompute_gbn_bar.py`: compute GBN's HARD composite-metric values and the empirical
   PCF target on held-out data. This is the **bar to beat**.
3. Short GT-free fine-tune of the refinement steps against `L_total`, **from noise**.
4. `validate_hard.py`: run scipy/FFT validators (M1/M2/M5 + spectrum) on **held-out** data
   for checkpoint A vs the fine-tuned checkpoint vs the GBN bar. Render ~20 output grids.

**Decision rule:**
- HARD validators improve over A **and** outputs visually clean → thesis alive, go Phase 2.
- Soft loss drops but HARD validators do **not** move → surrogate gap: fix tau schedule /
  PCF target BEFORE scaling. Do not proceed.
- Clumping appears → raise `w_pcf`, anneal tau faster, re-run kill-test.

---

## Phase 2 — Full training (only if Phase 1 passes)

Three checkpoints, identical architecture + inference, for the paper's core table:
- **A** — current imitation-trained (now the honest baseline).
- **B** — objective-trained **from noise** (primary).
- **C** — objective-trained **from smart init** (ablation; re-test since GT-free removes
  the supervised coupling that caused the earlier val gap).

Report all three on the **hard** validators vs the GBN/WVS/BNOT bar. Headline claim lives
in this table; verify every cell against a visual grid.

---

## Phase 3 — Rigorous validation (answers R4-style scrutiny, non-circular)

- **Do NOT** report a raw radial spectrum on capacity-constrained output — it assumes a
  stationary/uniform process and is meaningless under varying density. (This is the point
  the reviewer got wrong; state it, then give the correct diagnostic below.)
- **Correct blue-noise diagnostic = density-warped (differential-domain) PCF**, following
  the Wei & Wang warp already used by M5. Blue noise → PCF dip near 0 then settles to 1;
  clumping → peak near 0. This is capacity-aware and scale-invariant.
- **Non-circularity:** trained on the *soft* PCF surrogate; validated on the *hard*
  numpy/FFT PCF/spectrum and on a held-out statistic. Different code paths → not circular.

---

## Logging / anti-self-deception checklist (every eval step)

- [ ] soft-loss components (cap / cvt / pcf) — training only
- [ ] HARD scipy M1, M2, M5 on held-out — the real numbers
- [ ] soft-vs-hard capacity gap (tau health)
- [ ] current tau value
- [ ] 20-image visual grid saved to disk
- [ ] comparison vs GBN bar (precomputed)

If HARD metrics and visuals disagree with the soft loss, TRUST THE HARD METRICS + VISUALS.

---

## Explicit non-goals (do not let the agent drift here)

- No CLIP / text / multimodal work in this plan.
- No new architecture; reuse the existing sampler + inference path.
- No editing of the validator or GBN metric functions.
- No reporting of soft-loss numbers as results.
- No raw spectral plot on constrained output as a "blue-noise proof."

---

## Open items for the human (decide before Phase 2, not blocking Phase 1)

1. Loss-grid resolution actually affordable at your typical training budget.
2. tau start value + decay curve (tune from kill-test behavior).
3. Loss weights `w_cap : w_cvt : w_pcf` — start at comparable gradient magnitudes, tune.
4. Whether smart-init (C) behaves differently now that supervised coupling is gone.
