# Anisotropy Teacher Pilot — Progress & Next Steps

Companion to `anisotropy_teacher_pilot.md` (the plan). This file records **what was
actually run, what the numbers were, what we learned, and what to do next.**

Last updated: 2026-07-27

---

## TL;DR

| Gate | Question | Result |
|---|---|---|
| **M0** Legibility | Is anisotropy visible in pure dots? | ✅ **PASS** — legible for κ ≳ 1.4 |
| **M1** Oracle achievability | Is anisotropy compatible with capacity? | ✅ **PASS** — capacity invariant to κ |
| **M2** Constructive sampler | Can a sampler hit all constraints at once? | ✅ **PASS** — 0 violations, exact N, orientation 1–4° |
| **M3** Multimodality | Is the task one-to-many (diffusion justified)? | ✅ **PASS** — inter-seed W₂ ≈ 15× jitter floor |
| **M4a** Conditioning mechanism | Can a control branch steer the frozen model? | ✅ **PROVEN** — ON vs OFF = 15–40×, orientation 1.5–4.3° |
| **M4b** Generalization | Does it transfer to unseen images? | 🟡 **PARTIAL** — direction transfers (ON ~28° vs OFF ~41°) but strength capped at **~0.02 on unseen icons regardless of init / trunc / gain** — the branch's *generalized authority ceiling*. Teacher strength (0.10+) only in-domain. Fork: more data vs unfreeze. |

**All four data-side gates passed** and the mechanism is proven (M4a). Density-blinding the
aniso branch made **orientation direction transfer** to unseen icons. The open problem is
**strength magnitude**: on unseen icons the branch outputs ~0.02 no matter the init, trunc, or
residual gain — its *generalized authority ceiling*. Teacher strength (0.10+) is reached only
in-domain. Current fork: **scale up training data** vs **unfreeze the density branch**; a cheap
in-domain diagnostic (`run_indomain_check.sh`) decides which. See Part 6.

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

### M4b density ablation — hide density from the aniso branch: ✅ ENTANGLEMENT CONFIRMED

Same upgrade protocol at trunc 0.45, aniso branch built `include_density=False`. The
**frozen density branch still receives ρ**, so density stays a hard condition — only the
*orientation* branch is blinded to content.

| set | with density (baseline) | density hidden |
|---|---|---|
| in-domain primitives | ON 0.029 / OFF 0.016, oe 17.4° | ON 0.028 / OFF 0.021, oe 13.8° |
| **unseen icons** | ON 0.012 / OFF 0.013, oe **43.0°** | ON 0.026 / OFF 0.012, oe **13.1°** |

On unseen icons, orientation error collapsed **43°→13°** (inside the 20° gate) and ON went
from tied with OFF to **2.2×**. **Diagnosis: with only 4 shapes the branch used ρ to identify
the shape and memorised a per-shape answer.** Blinding it forces a content-independent local
rule that transfers. `include_density=False` is now the **default** in `run_m4_generalize.py`
(opt back in with `--density`). The hard-coded strength gate still prints "no transfer"
because ON−OFF=0.014 misses the 0.03 margin — a magnitude issue, not orientation; the
protocol's ~20% strength ceiling makes 0.03 barely reachable.

### The inference flow we've settled on (canonical recipe)

Two branches feed the one frozen denoiser; injections are **summed** into the U-Net:
- **frozen density branch** (control_v4) ← image + ρ : count/placement (density = hard condition)
- **aniso branch** (trainable, `include_density=False`) ← (θ,κ) field : local orientation

Upgrade protocol (also the product story):
1. **baseline**: frozen model + density branch, aniso field = 0, smart-init → denoise at
   base_trunc 0.30 → ordinary **isotropic** stipple P₀.
2. **control field**: from desired (θ,κ), `ctl = [m·cos2θ, m·sin2θ, m]`, m = log κ.
3. **upgrade**: re-noise P₀ to **upgrade_trunc 0.45**, denoise back down with BOTH branches
   ON → same icon, same ρ, same N, now **anisotropic**.

ON-vs-OFF is only the measurement instrument; production runs the ON path only.

> ⚠️ This recipe is the *architecture / mechanism*, and it works IN-DOMAIN. On UNSEEN icons
> the anisotropy magnitude collapses to ~0.02 (see the magnitude sections below) — the flow is
> correct, but the branch's generalized authority is the open blocker.

### M4b on REAL ICONS (60 train / 10 held-out, density-blind): partial transfer, NO trunc optimum

Trained the density-blind branch on 60 real icons (continuous θ,κ, 8 samples each via
`m4_icons_teacher.py`), evaluated on 10 **disjoint held-out** icons.

- **Training longer does NOT help** (40k vs 100k: aggregate ON oe 28°→32°, slightly *worse*).
  The branch overfits the train icons; authority is not the limiter. Undertraining ruled out.
- **upgrade_trunc sweep** (eval-only, same 100k ckpt, **single seed**):

  | trunc | mean ON oe | t00 | t45 | t90 |
  |---|---|---|---|---|
  | 0.45 | 32.3° | 24.7 | 31.0 | 41.3 |
  | 0.55 | 32.6° | 38.4 | 22.2 | 37.3 |
  | **0.60** | **20.3°** | 19.5 | 15.1 | 26.4 |
  | 0.65 | 33.2° | 31.1 | 27.3 | 41.1 |
  | 0.75 | 32.0° | 36.0 | 25.2 | 34.6 |

  Mean ON oe sits at ~32° across the WHOLE sweep except a single sharp dip to 20° at 0.60,
  flanked by ~33° at both 0.55 and 0.65. That isolated spike is **not** a smooth optimum:
  at ON strength ~0.02 on only 10 held-out icons, per-sample orientation is noisy, so 0.60
  is most likely a **lucky seed**. ON does beat OFF (~45–55°) at every trunc, so orientation
  transfers *weakly and consistently*, but it does NOT robustly clear the 20° gate.
- **Multi-seed check settles it: 0.60 was a lucky seed.** Re-running trunc 0.60 at three
  fresh seeds (`--eval_seed`) gives 24.2 / 30.5 / 30.6 → **mean 28.4°, std 3.0°**. The 20.3°
  was the outlier. With ~3–4° seed noise the whole sweep (32.3 / 32.6 / ~28 / 33.2 / 32.0) is
  **one flat band — there is no trunc optimum**, and orientation clears 20° nowhere.
- **What IS robust across every seed:** ON ~28° vs OFF ~41°, ON str 0.020–0.024 vs OFF
  0.015–0.018. The density-blind branch genuinely steers orientation on unseen real icons —
  the *direction* transfers; the *magnitude* does not.

### The controlling insight: orientation error is DOWNSTREAM of magnitude

Every achieved strength in the project, against its orientation error:

| achieved strength | source | orientation error |
|---|---|---|
| 0.15–0.19 | teacher targets | **1–4°** |
| 0.10–0.13 | model, teacher-init (M4a) | **1.5–4.3°** |
| 0.030 | model, in-domain upgrade | **10–17°** |
| 0.020–0.024 | model, icons upgrade | **~28°** |

Orientation error is a **monotone function of achieved strength**. This is not a steering
failure: at strength ~0.02 the point set is barely anisotropic, so its axis is intrinsically
ill-defined and no estimator can read it precisely. **The orientation gate is downstream of
magnitude — so magnitude is the single blocker.** No amount of truncation tuning or extra
training can fix orientation while strength sits at 0.02, which is exactly what was observed
empirically twice (100k steps made it worse; the trunc sweep is one flat band).

**Lever order for magnitude** — results:
1. `--control_gain` — inference-time scaling of the aniso **residual**. ❌ **TESTED, FAILED.**
   Strength saturates at ~0.031 (gain 2.5 ≈ 4.0) and orientation *degrades* with gain
   (oe 27° → 45°, back to OFF-level ~43°). Scaling the residual injects **incoherent**
   displacement — the strength estimator reads the disorder as magnitude, but it has no
   coherent axis. Confirms a hard **~0.03 ceiling** on the from-isotropic path.
2. **Anisotropy-aware init** (`--init aniso --init_kappa 1.2`) — seed the upgrade from a weak
   low-κ classical sample and let the branch STRENGTHEN it toward the target κ. This is the
   teacher-init regime — the ONLY thing shown to exceed 0.03 (reached 0.10–0.13 @ 1.5–4.3°).
   **Implemented; now the main lever.** Reframes the deliverable: *refine a weak anisotropic
   stipple to the commanded (θ,κ) at fixed density*, not *create anisotropy from isotropic*.

**All story-preserving from-isotropic levers are now exhausted** (trunc = flat band; more
training = worse; residual gain = saturates + degrades). The ~0.03 coherent-anisotropy ceiling
is a property of nucleating from an isotropic seed, and only a non-isotropic init breaks it.

### The model ERASES anisotropy — it does not lack a strong seed

Raw classical-seed strength, measured before the model (local, `icons_ds` test imgs):
κ=1.2 → **0.059**, κ=1.5 → **0.134**, κ=2.0 → **0.241**. Seeds are strong.

The first `--init aniso` run fed the 0.134 seed through **upgrade_trunc 0.45** and got **0.020**
out (OFF even dropped to 0.015). So the frozen model **destroyed a 0.134 seed down to the
floor** — the blocker is the prior isotropising during denoise, NOT a weak seed.

**That run used the wrong trunc.** Teacher-init works (0.10–0.13) at **trunc 0.30** with a
strong seed, so the corrected test mirrored it: strong classical seed (κ 1.5–2.0) at **LOW
trunc (0.20–0.30)**, on unseen icons (`run_init_lowtrunc.sh`).

### Corrected low-trunc aniso-init: ❌ ALSO NULL — the ceiling is the BRANCH, not init/trunc/prior

Even at trunc 0.20 with the 0.241 seed, output is ~0.023 (2 seeds):

| trunc | init | ON str | keep% | ON oe |
|---|---|---|---|---|
| 0.20 | iso | 0.023 | — | 30.5 |
| 0.20 | aniso κ=1.5 (raw 0.134) | 0.023 | 17% | 27.8 |
| 0.20 | aniso κ=2.0 (raw 0.241) | 0.024 | 10% | 27.5 |
| 0.30 | iso | 0.023 | — | 30.8 |
| 0.30 | aniso κ=1.5 | 0.021 | 16% | 27.5 |
| 0.30 | aniso κ=2.0 | 0.023 | 9% | 25.9 |

**Decisive: the output is ~0.023 regardless of the init.** iso, κ=1.5 (0.134), κ=2.0 (0.241)
all land at 0.023 — the init is **fully erased** and the branch regenerates its own fixed ~0.02,
keeping only 10–17% of even the strongest seed. So init / trunc / gain are ALL downstream of one
number:

> **On unseen icons the icons-trained branch outputs ~0.02 anisotropy no matter what.** That is
> the branch's *generalized authority ceiling*; nothing at inference time exceeds it.

**Teacher-init's 0.10 was IN-DOMAIN memorization, not generalization.** Its init points *were*
the branch's own memorized flat-canvas target — it only had to not-move a memorized answer
(injection 0.59–0.73). The generalized push on unseen content is ~0.02. And **more training made
it worse** (100k overfit vs 40k) → the branch is not under-trained, it is **under-data'd**
(60 icons × 8 samples memorises; it does not generalise).

### Where it stands — the fork (see Part 6)

The architecture CAN inject strongly (the overfit reached 0.10+), so this is a **learning/data**
limit, not architectural. Two remaining levers:
- **Scale up data** (many more icons × samples, ~40k steps — NOT more steps) — keeps the frozen
  plug-in story; the principled bet, since more steps overfit → data is the suspect.
- **Unfreeze** the density branch (base still frozen) — original step 4; loses the clean plug-in
  story; the lever if the prior is simply too strong.

**In progress:** cheap in-domain diagnostic (`run_indomain_check.sh` + `--eval_train_n`) — does
the branch preserve strongly on TRAINING icons (keep% ≫ 17% ⇒ data gap ⇒ scale up) or is it weak
even in-domain (keep% ≈ 17% ⇒ weak branch ⇒ unfreeze)?

### Reframed deliverable (target framing — NOT yet achieved on unseen icons)

The frozen model cannot generate/amplify anisotropy from isotropic (proven every way: trunc,
training, gain, weak seed, strong-seed-low-trunc — all cap at ~0.02–0.03). What it can do —
proven only IN-DOMAIN — is **preserve** commanded anisotropy through a refinement denoise the
base model alone would isotropise:

> cheap classical anisotropic seed → light re-noise → the **control branch holds** (θ,κ) while
> the frozen model cleans it onto its blue-noise manifold at fixed density.

This becomes the honest product IF the branch can be made to preserve on UNSEEN content — exactly
what the data-scale-up / unfreeze fork will settle.

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
6. **Never trust a single-seed mean at this noise floor.** 10 held-out icons × 3 conditions
   at strength ~0.02 carries **±3–4° seed noise** on the mean orientation error. A single
   run produced an apparent 12° "optimum" at trunc 0.60 that vanished under three fresh
   seeds. Replicate across `--eval_seed` before believing any difference under ~8°.

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
| ~0.02 | model, UNSEEN-icon upgrade (branch ceiling) | ~invisible |
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
| `m4_icons_teacher.py` | **real-icons** dataset + held-out split (continuous θ,κ) → `.npz` |
| `run_m4_overfit.py` | M4a mechanism test (+ `--eval_only`) |
| `run_m4_generalize.py` | M4b eval/train; flags `--init iso/aniso` `--init_kappa` `--control_gain` `--eval_seed` `--eval_train_n` `--density` `--include_primitives` |
| `run_gain_sweep.sh` | residual-gain sweep (FAILED lever) |
| `run_init_compare.sh` | iso vs aniso-init at fixed trunc |
| `run_init_lowtrunc.sh` | strong-seed low-trunc aniso-init (NULL) |
| `run_indomain_check.sh` | in-domain vs held-out keep% diagnostic (the fork decider) |
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

### Status: real-icons training DONE; magnitude is the blocker; at a data-vs-unfreeze fork.

M4a and the real-icons M4b training are done. Density-blinding fixed orientation *direction* on
unseen icons. Every inference-time magnitude lever — trunc, more steps, residual gain,
aniso-init at any trunc — is exhausted; all cap at the branch's **~0.02 generalized-authority
ceiling** on unseen icons. Ranked from here:

### 1. In-domain diagnostic ← DOING NOW (cheap; decides 2 vs 3)
`run_indomain_check.sh` (`--eval_train_n`). Does the branch preserve strongly on TRAINING icons?
- keep% ≫ 17% (ON str → 0.08+) → **generalization / data gap** → step 2.
- keep% ≈ 17% (still ~0.02) → **weak branch everywhere** → step 3.

### 2. Scale up training data (if step 1 = data gap)
Many more icons × samples (e.g. 150–200 icons × 16–24 samples), retrain at **~40k steps, NOT
more** (100k overfit 480 samples). Keeps the frozen plug-in story.
`m4_icons_teacher.py --n_train_icons 180 --samples_per_icon 20` → `run_m4_generalize.py
--steps 40000` → re-run the ON/OFF + low-trunc eval.

### 3. Unfreeze the density branch (if step 1 = weak branch, or step 2 plateaus)
Unfreeze **density branch first**, base denoiser still frozen. Costs the clean plug-in story and
risks stipple quality — genuine last resort — but the right lever if the frozen prior is simply
too strong to bend. (Original spec step 4.)

### Still outstanding (deferred, not blocking)
- **Grid-transfer validation at G ∈ {32, 48, 64}** — required by the spec, only meaningful once
  transfer works. Architecture grid-safe by construction. **G = 96 dropped** (beyond current
  model controllability).
- **M1 stage-2 formal write-up** for non-uniform ρ — sampler validated (cap err 0.088 @ κ=1 vs
  0.098 @ κ=2, same feasible r0), not run as a formal gate.

### Honest expectation
The mechanism is proven and the architecture can inject strongly (overfit reached 0.10+), so a
real result is reachable — but so far only orientation *direction* transfers; *magnitude* does
not, and it is the whole blocker. Whether **data** or **unfreezing** fixes it is the open
question the in-domain diagnostic + step 2/3 will answer. This is **not yet** a solved control on
unseen content.
