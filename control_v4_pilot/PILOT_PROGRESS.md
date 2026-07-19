# Anisotropy Teacher Pilot — Progress & Next Steps

Companion to `anisotropy_teacher_pilot.md` (the plan). This file records **what was
actually run, what the numbers were, what we learned, and what to do next.**

Last updated: 2026-07-19

---

## TL;DR

| Gate | Question | Result |
|---|---|---|
| **M0** Legibility | Is anisotropy visible in pure dots? | ✅ **PASS** — legible for κ ≳ 1.4 |
| **M1** Oracle achievability | Is anisotropy compatible with capacity? | ✅ **PASS** — capacity invariant to κ |
| **M2** Constructive sampler | Can a sampler hit all constraints at once? | ✅ **PASS** — 0 violations, exact N, orientation 1–4° |
| **M3** Multimodality | Is the task one-to-many (diffusion justified)? | ✅ **PASS** — inter-seed W₂ ≈ 15× jitter floor |
| **M4a** Conditioning mechanism | Can a control branch steer the frozen model? | ✅ **PROVEN** — ON vs OFF = 15–40×, orientation 1.5–4.3° |
| **M4b** Generalization | Does it transfer to unseen images? | ❌ **NOT YET** — works in-domain, fails on unseen icons |

**All four data-side gates passed.** The model-side mechanism is proven. The single
open problem is **generalization to unseen images**.

---

## Part 1 — Data-side gates (M0–M3), all PASSED

Self-contained numpy/scipy pilots. Each milestone's scripts + outputs were moved
into its own folder (`m0_test/`, `m1_test/`, `m2_test/`, `m3_test/`).

### M0 — Legibility (`m0_test/`)
Files: `aniso_pilot.py`, `run_m0.py`, `run_kappa_ladder.py`

- Strength floor **0.013** at κ=1 → **0.220** at κ=2, monotone.
- Orientation recovered to **<5°** at θ = 0/45/90/135.
- Objective liftoff ≈ **κ 1.2**; perceptual floor ≈ **κ 1.3–1.4**; obvious by κ ≥ 1.6.
- **Usable control band: κ ∈ [1.4, 2.0].**

### M1 — Oracle achievability (`m1_test/`)
Files: `aniso_m1.py`, `run_m1.py`

Oracle = area-preserving diffeomorphism of clean blue noise on a torus (exact N,
density preserved by construction).

- **GATE A (capacity — the spec blocker): PASS at every κ.** Density-based capacity
  (regional count, CCVT carrier CoV, KDE) ratios **0.90–1.06** vs the identity
  baseline, including at κ≈2. Confirms the determinant-metric mechanism:
  area-preserving ⇒ density invariant ⇒ capacity invariant.
- **GATE B (hard-core, diagnostic):** clean smooth-field ceiling **κ_med ≤ 1.52
  (κ_max ≤ 1.80)**, qmin ≥ 0.985. At κ≈2 qmin dips to 0.975 but the 1st percentile
  is ≥ 0.982 — only the single worst pair, i.e. the local-metric validity edge.
- Naive isotropic CVT trap is **mild** below κ≈2 (≤ ×1.03), contrary to expectation.

### M2 — Constructive sampler (`m2_test/`)
Files: `aniso_m2.py`, `run_m2.py`

Li-style anisotropic dart throwing; spacing is an **acceptance invariant**, never a penalty.

- Every condition (κ = 1.25/1.5/2.0 at θ=30°, plus a smooth rotating field):
  **0 hard-core violations, qmin = 1.000, exact N = 1024**, capacity **0.93–1.03**
  vs the κ=1 sampler, orientation bias **1–4°**.
- Realized anisotropy `align` rises monotonically 0.06 → 0.12 → 0.21 with κ.

### M3 — Multimodality (`m3_test/`)
File: `run_m3.py`

- **D_points / jitter floor ≈ 15** at every κ (gate was 3).
- D_points ≈ **1.5× mean spacing** — seeds are about as different as two independent
  samples, i.e. maximal diversity.
- **Collapse ratio ≈ 1.0 through κ=2** — anisotropy does not shrink the valid set.
- D_style ≈ 0.04 — all seeds hit the same target statistics.

**⇒ The task is genuinely one-to-many: a diffusion model is justified over regression.**

---

## Part 2 — M4: model conditioning (current work, this folder)

### Architecture (locked)

```
frozen : base GBN denoiser            (config/GBN/model.ckpt)
frozen : density control branch       (ep10000, icons-50)
TRAIN  : AnisoControlNet  <-- the ONLY trainable part
         hint = [offsets, density, (m·cos2θ, m·sin2θ, m)],  m = log κ
         conv-only, zero-init injections, residuals SUMMED into the same U-Net
```

Design constraints honoured (from the spec): control injected **internally** (never
as an edit to the density map), density stays a **hard** condition, control is a
**residual**, no absolute positional encoding on query coords (grid-transfer safe).

Density-branch checkpoints:
- `train_outputs_icons50_512_no_random` — TRUNCATION_RATIO **0.30**
- `train_outputs_icons50_512_full` — TRUNCATION_RATIO **1.0** ← use this one
  (valid at all timesteps; lets train/sample truncation be decoupled)

Both: `GECCO=True, ADAPTIVE_GATE=True, SDF/SMART_INIT/BATCH_COORDS=False, G=32`.

### M4a — Overfit mechanism test: ✅ PROVEN

`run_m4_overfit.py` + `m4_teacher.py` — flat black canvas, identical density in every
condition, only the control field differs.

**Decisive ablation** (teacher-init, control ON vs OFF from the *same* start):

| condition | control ON | control OFF |
|---|---|---|
| κ=1 | 0.010 | 0.020 (both floor ✓) |
| κ=2 / θ0 | **0.117 @ 2.3°** | 0.003 @ 84.8° |
| κ=2 / θ90 | **0.132 @ 1.5°** | 0.003 @ 10.2° |
| κ=1.6 / θ45 | **0.105 @ 4.3°** | 0.007 @ 40.9° |

Zeroing the control collapses anisotropy **15–40×** and randomizes orientation.
Branch instruments: injection ratio **0.59–0.73**, control sensitivity (L2) **0.57–0.65**.

**Truncation study** (teacher-init strength; teacher reference 0.151–0.194):

| train / sample | strength |
|---|---|
| 0.30 / 0.30 | **0.105–0.132** (best) |
| 1.00 / 0.30 | 0.069–0.096 |
| 1.00 / 1.00 | 0.012–0.028 (floor) |

Sampling truncation dominates; training truncation has a secondary (dilution) effect.

### M4b — Generalization test: ❌ NOT YET

`m4_primitives.py` + `run_m4_generalize.py` — trained on **4 primitives only**
(flat/disk/square/stripe), 400 samples, **continuous** θ∈[0,180°), κ∈[1,2], 40k steps.
Evaluated with the **upgrade protocol**: frozen model's own isotropic stipple →
re-noise → denoise with control ON vs OFF.

| upgrade_trunc | in-domain (primitives) | unseen icons |
|---|---|---|
| 0.30 | ON **0.030** vs OFF 0.017, oe **10.2°** | ON 0.014 vs OFF 0.012, oe 29.3° |
| 0.45 | ON **0.029** vs OFF 0.016, oe **17.4°** | ON 0.012 vs OFF 0.013, oe 43.0° |
| 0.60 | ON **0.032** vs OFF 0.014, oe **13.8°** | ON 0.014 vs OFF 0.016, oe 28.1° |

**Two conclusions:**

1. **Nucleation WORKS in-domain.** Starting from an isotropic baseline with *no*
   anisotropic seed, the control creates correctly-oriented structure (ON ≈ 2× OFF,
   orientation 10–17°, inside the 20° gate, reproducible across truncations).
   This *reverses* the earlier "nucleation gap" reading.
2. **No transfer to unseen icons** (ON ≈ OFF, orientation random).
   ⇒ **The blocker is a GENERALIZATION gap, not a nucleation gap.**

**Truncation is settled** — 0.30/0.45/0.60 are equivalent. Stop sweeping it.

**Magnitude caveat:** even in-domain, the upgrade protocol recovers only ~20% of
teacher strength (0.030 vs 0.15–0.19), vs 70–80% under teacher-init.

---

## Part 3 — Hard-won lessons (do not rediscover these)

**Measurement**
1. **k-NN anisotropy estimators are broken here.** Selecting the k nearest neighbours
   fills a round disk, so covariance/double-angle read isotropic regardless of truth.
   Use `global_near_field_anisotropy` (closeness-weighted + coherent global sum) and
   cross-check with `nn_vector_anisotropy`.
2. **Per-point orientation is far too noisy on varying fields** — its median floors
   near 30° regardless of the truth. Use the **de-rotated coherent** readout:
   `D = Σ w·exp(2i(φ_ij − θ(x_i))) / Σw`; bias = ½·arg(D). This fixed a bogus 33.7° → 2.4°.
3. **Never measure anisotropy on a spatially irregular subset** (e.g. "interior of
   ink"). Missing neighbours near edges *fabricate* orientation aligned with the
   stroke. Gate on all points.
4. **Magnitude-only control sensitivity is misleading** — use relative **L2** between
   branch outputs with vs without the control (0.073 vs 0.568 on the same run).
5. **W₂ jitter floor must use matched indices** (same points, jittered), else
   independent subsamples inflate the floor into a false failure.

**Training / setup**
6. **Canvas must be BLACK (0.0), not mid-gray.** `target_density` is the RAW grayscale
   and metrics use `1 − image`, so **dark = ink = points**. 0.5 is ambiguous *and* OOD.
7. **For an overfit test use `n_per_cond = 1`.** Multiple targets sharing one control
   field make the task unlearnable-by-memorization; the loss plateaus at the ambiguity
   floor and looks like "converged/broken" when it is a setup bug.
8. **Zero-init branches need tens of thousands of steps** to gain authority (400 steps
   was ~nothing; injection ratio grew 0.22 → 0.73 between 8k and 60k).
9. **Never compare losses across different t-ranges** — full-range training includes
   easy high-t steps and looks "better" while being worse.

**Conventions**
10. **The two conventions are opposite. Do not mix them.**
    - M0 warp `p → M·p`: spacing elongates **ALONG θ**; NN concentrate at **θ+90°**.
    - M2/M4 metric `J = metric_M(θ,κ)`: exclusion extends **perpendicular**, so spacing
      elongates perpendicular and NN concentrate **ALONG θ**. Compare measured axis to **θ**.

**Environment**
11. `R:\` is SSHFS: Write/Edit tools fail EPERM. PowerShell here-strings were also
    blocked by a content hook. **Robust path: Write to the local scratchpad, then
    `Copy-Item` onto R:.** Pure-stdlib python patch scripts run fine.

---

## Part 4 — Strength calibration (how to read the numbers)

Anisotropy is a **local, sub-cell** property — invisible at full-field scale. Run
`python visual_primer.py` for zoomed reference figures.

| strength | source | appearance |
|---|---|---|
| ~0.19 | teacher targets (κ=2) | obvious when zoomed |
| ~0.10 | model, teacher-init (proven mechanism) | visible when zoomed |
| ~0.03 | model, in-domain upgrade path | faint — needs the zoom |
| ~0.01 | isotropic floor | nothing to see |

Convention reminder: at θ=0° dots pack into **horizontal chains**; at θ=90°, vertical.

---

## Part 5 — File inventory (this folder)

| file | role |
|---|---|
| `anisotropy_teacher_pilot.md` | the original plan / spec |
| `PILOT_PROGRESS.md` | this file |
| `aniso_pilot.py` | M0 core: blue noise, `metric_M`, warps, dots renderer, **estimators** |
| `aniso_m1.py` | M1 core: toroidal blue noise, div-free flows + deformation gradient, capacity metrics |
| `aniso_m2.py` | M2 core: **anisotropic dart thrower** (uniform density) |
| `aniso_density.py` | **density-aware** sampler `J = √λ·A` for non-uniform ρ |
| `aniso_control.py` | `AnisoControlNet` (2nd branch) + `DualControlledDenoiser` |
| `m4_teacher.py` | flat-density teacher: points → OT offset grid + control map |
| `m4_primitives.py` | primitives dataset (continuous θ,κ) → `.npz` |
| `run_m4_overfit.py` | M4a mechanism test (+ `--eval_only`) |
| `run_m4_generalize.py` | M4b primitives → unseen icons (+ `--include_primitives`) |
| `visual_primer.py` | zoomed reference figures for the eye |
| `inspect_ckpt.py` | report a control_v4 checkpoint's baked-in config |
| `m0_test/ m1_test/ m2_test/ m3_test/` | completed milestone scripts + outputs |
| `m4_out*/ m4_gen*/` | M4 run outputs |

### Reproduce

```bash
# data gates (from each milestone folder)
python run_m0.py --out m0_out ; python run_kappa_ladder.py --out ladder_out
python run_m1.py --out m1_out
python run_m2.py --out m2_out
python run_m3.py --out m3_out

# M4a — mechanism (from repo root)
python control_v4_pilot/run_m4_overfit.py --steps 60000 --out control_v4_pilot/m4_out

# M4b — generalization
cd control_v4_pilot && python m4_primitives.py --n_samples 400
cd .. && python control_v4_pilot/run_m4_generalize.py --steps 40000
python control_v4_pilot/run_m4_generalize.py --eval_only --include_primitives \
    --out control_v4_pilot/m4_gen_out
```

---

## Part 6 — NEXT STEPS (ranked)

### 1. Train on REAL ICONS (main path)
The primitives → icons domain shift is the thing that failed. `aniso_density.py` now
lets us build teacher targets on actual icons, removing the shift entirely.

- Generate teacher for ~60–100 icons × continuous (θ,κ) (~4 s/sample ⇒ ~25 min).
- **Held-out icon test set** — the question is transfer, so the split matters.
- Reuse the upgrade protocol + ON/OFF ablation; gate on all points.
- *To write:* `m4_icons_teacher.py` (mirror of `m4_primitives.py`, reads real icons,
  train/test split).

### 2. `include_density=False` ablation (cheap, diagnostic)
The branch currently receives the density map, so with only 4 shapes it could key on
*those shapes* instead of learning a local rule. Denying it density **forces** a
content-independent rule. If transfer improves, the entanglement diagnosis is confirmed
and it is a one-line architectural win. Run alongside step 1.

### 3. More shape diversity (if staying with primitives)
Random blobs, rings, multi-disc, polygons, and **thin strokes** — the local contexts
icons actually contain.

### 4. Only if 1–3 fail: joint / unfrozen training
Unfreeze the **density branch first**, keeping the base denoiser locked. Costs the
clean "frozen model + plug-in control" story and risks degrading stipple quality, so
it is a genuine last resort — but it is the right lever if the frozen prior is simply
too strong to bend.

### Still outstanding (deferred, not blocking)
- **Grid-transfer validation at G=64/96** — required by the spec, but only meaningful
  once transfer works. (Architecture is grid-safe by construction: conv-only, no
  absolute positional encoding.)
- **M1 stage-2 formal write-up** for non-uniform ρ — the sampler is validated
  (capacity error 0.088 at κ=1 vs 0.098 at κ=2, same feasible r0) but was not run as a
  formal gate.

### Realistic expectation
The mechanism is proven and nucleation works in-domain, so this is a **data/diversity**
problem, not an architectural dead end. Failing to generalize from four primitives to
arbitrary icons is an unremarkable result; training on the target domain is the
standard fix and has not yet been tried.
