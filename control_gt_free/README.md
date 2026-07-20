# control_gt_free — Objective-Refined Imitation for Stipple Generation

> **Naming caveat, read first.** The folder is called `gt_free`, but the method that survived
> is **not** GT-free. The pure objective-driven (no-teacher) direction was fully implemented,
> tested, and **closed with a negative result** — see [Status](#status). What ships and trains
> here is the **hybrid**: imitation (velocity) supervision as the blue-noise *spacing anchor*,
> plus a differentiable geometry objective that *refines* on top.

---

## TL;DR — run it

```bash
# from the repo root
python control_gt_free/train_control.py
```

No flags needed. It is hybrid by default, **starts clean** (no warm-start), and **auto-builds
the PCF target** into `<out>/gbn_bar.pt` on first run.

---

## Status

Three modes exist; only one is the method.

| Mode | Flag | Loss | Verdict |
|---|---|---|---|
| **Hybrid** | *(default)* | velocity MSE **+** geometry objective on the one-step x0 decode | **THE METHOD** |
| Pure imitation | `--no-gt-free` | velocity MSE only | Honest baseline (A) |
| Pure objective (GT-free) | `--gt-free-mode unroll` | objective on the model's own unrolled points, no GT | **DEAD** — see below |

### Why GT-free was closed

`L_capacity` and `L_cvt` are both **attractive/allocative** — they say *"put mass where density
is high."* Strong edges are high-density, so points pile onto them. **Neither term rewards
spacing.** The density-warped PCF is the only *repulsive* term.

The direction got two honest attempts:

1. **First run** — PCF was silently disabled (no `gbn_bar.pt` ⇒ `w_pcf` forced to 0), leaving a
   capacity-only objective. Clumped **by construction**. That run was *invalid*, not a finding.
2. **Second run** — transfer probe with the PCF repulsion term genuinely active, warm-started
   from a converged hybrid checkpoint (`overfit1` → `overfit2`). Improved, but points **still
   collapsed onto edges**.

**Conclusion:** a differentiable spacing objective cannot maintain blue-noise structure against
the attractive capacity term, even starting from a checkpoint that demonstrably had it.
**Blue noise here must be supervised, not optimized.** This is exactly why the hybrid keeps the
velocity loss — that term is load-bearing, not vestigial.

> Caveat for write-up: per-term **gradient-magnitude logging was never added**, so the honest
> claim is *"collapsed with a properly-weighted PCF term present"*, not a certified
> `grad(PCF) >= grad(capacity)` proof. That log is the one thing that would make it airtight.

---

## The hybrid loss

```
L_total = L_velocity(v_pred, v_target)              # primary: spacing prior from GT/GBN targets
        + lambda * w(t) * L_geometry(x0_pred)       # auxiliary: refines on top

x0_pred = offsets_t - t * v_pred                    # ONE-STEP decode, algebraic (no ODE solve)
w(t)    = 1 if t < geo_t_max else 0                 # hard low-t mask
```

* **One-step decode.** Since `x_t = (1-t)*x_0 + t*eps` and `v = eps - x_0`, then `x_0 = x_t - t*v`.
  Cheap and exact — no sampler unroll during training.
* **Low-`t` mask.** `x0_pred` is a rough guess near noise and reliable near data, so the geometry
  term is applied only where `t < geo_t_max` (default `0.4`). Applying it uniformly poisons the
  signal with high-`t` noise.
* **Warmup.** Geometry stays off for `geo_warmup` steps (default 200) while `x0_pred` is garbage.
* **Velocity stays dominant.** `lambda` (`--geo-weight`, default `0.1`) is deliberately small.

### The geometry objective

`L_geometry = w_cap*L_cap + w_cvt*L_cvt + w_pcf*L_pcf` (`losses/composite.py`), all computed on
the decoded **points**:

| Term | What it does | Hard validator it mirrors |
|---|---|---|
| `L_cap` | soft-Voronoi capacity, squared CV (`delta_c`) — *attractive* | `compute_m2_capacity_constraint` |
| `L_cvt` | soft weighted CVT energy — *attractive* | `compute_m1_cvt_energy` |
| `L_pcf` | **density-warped PCF — the only repulsive term** | `compute_m5_spatial_measure` / spectrum |

* Soft-Voronoi membership `w_i(x) = softmax_i(-||x - s_i||^2 / tau)` is computed **once** and
  shared by the capacity and CVT terms. `tau` anneals warm → hard (`train/tau_schedule.py`) so the
  surrogate converges toward what the scipy validator actually measures.
* `L_pcf` uses a **soft-binned** (Gaussian kernel-splat) radial histogram of density-warped
  pairwise distances — differentiable, unlike `np.histogram`. The warp is the Wei & Wang / M5
  factor `sqrt(D_x)`, `D_x = N * local_rho / mean_rho` (stop-grad by default). Blue noise ⇒ dip
  near 0; clumping ⇒ peak near 0.
* `pcf_target` is **empirical and aggregate**: the averaged density-warped PCF of GBN reference
  sets. It is a *statistical signature*, never a per-sample target — "reference, not comparison".

---

## Running

### Train (flag-free)

```bash
python control_gt_free/train_control.py
```

On startup it will: resolve/export offsets → **auto-build `<out>/gbn_bar.pt`** (PCF target, ~64
reference samples, cached) → train the hybrid from scratch.

Override anything as usual:

```bash
python control_gt_free/train_control.py \
    --source <source_dir> --target <target_dir> \
    --out control_gt_free/train_outputs_hybrid \
    --geo-weight 0.1 --w-pcf 1.0 --w-cap 1.0 --w-cvt 1.0
```

### Validate on the HARD metrics

```bash
python -m control_gt_free.eval.validate_hard \
    --ckpt <checkpoint.pt> --source <src> --offsets <offs> \
    --gbn-bar control_gt_free/gbn_bar.pt --out control_gt_free/val_out
```

Runs the untouched scipy/FFT validators, writes `hard_report.json`, and dumps a 20-image visual
grid. Accepts train_control checkpoints (`control_net` key + embedded `gtfree_config`).

### Single-image overfit (sanity check)

```bash
python control_gt_free/test_overfit.py --steps 2000 --sample-index 0
```

Same three modes. Defaults to `smartinit` coupling (starts from the smart-init cloud, which
converges far faster than from noise on a single image).

---

## Package layout

```
train_control.py            MAIN entry — hybrid training (flag-free, auto-bar, clean start)
test_overfit.py             single-image overfit; --data-root, --init-ckpt (weights-only warm-start)

losses/
  soft_membership.py        soft-Voronoi w_i(x), point/grid geometry, rho lookup
  loss_capacity.py          L_cap  (soft delta_c)
  loss_cvt.py               L_cvt  (soft CVT energy)
  loss_pcf.py               soft_bin() + density-warped PCF + L_pcf   <- the new piece
  composite.py              L_geometry = weighted sum; shares membership; returns components
train/
  tau_schedule.py           softmax-temperature anneal (warm -> hard)
  unroll.py                 grad-enabled ODE unroll (used only by the dead `unroll` mode)
eval/
  precompute_gbn_bar.py     PCF target + GBN hard "bar"; build_bar_from_dataset() is what
                            train_control calls automatically; also --from-overfit-dir
  validate_hard.py          scipy/FFT validators + visual grid (THE source of truth)
config/objective.yaml       knob reference / intent (documentation, not auto-loaded)

overfit1.py, overfit2.py    DEAD — the GT-free transfer probe, kept as the recorded negative result
```

`*.pre_gtfree.bak` files hold the originals from before this package was adapted.

---

## Key defaults

| Knob | Default | Note |
|---|---|---|
| `--gt-free` / `--gt-free-mode` | on / `hybrid` | the method |
| `--geo-weight` (lambda) | `0.1` | keep well below the velocity term |
| `--geo-t-max` | `0.4` | hard low-`t` mask |
| `--geo-warmup` | `200` | velocity-only steps first |
| `--w-cap` / `--w-cvt` / `--w-pcf` | `1.0` / `1.0` / `1.0` | PCF is the term that fights clumping |
| `--tau-start` → `--tau-end` | `0.02` → `0.002` | cosine anneal |
| `--loss-grid` | `32` | soft-Voronoi is O(N x G) memory |
| `--gbn-bar` | auto-built | to `<out>/gbn_bar.pt` if not given |
| conditioning / coupling | `concat` / `gaussian` | `spade` + `smartinit` available |
| resume | **off** | hybrid trains clean; `--resume-latest` to opt in |

---

## Guardrails (non-negotiable)

1. **Soft losses are TRAINING-ONLY.** The soft-Voronoi and soft-binned PCF are optimization
   signals. They are **never** reported as results.
2. **Hard metrics are the only source of truth** — `utils/stippling_metrics_advance.py`
   (`compute_m1_cvt_energy`, `compute_m2_capacity_constraint`, `compute_m5_spatial_measure`,
   `plot_visual_m5_spectrum`). **Do not modify those validators.**
3. **Visual check is mandatory.** Metric wins mean nothing until ~20 outputs are rendered and
   inspected for clumping. A prior metric bug already produced a false "we win".
4. **Non-circular by construction:** trained on the *soft* surrogate, validated on the *hard*
   numpy/FFT path. Different code paths.
5. **No raw radial spectrum** as a blue-noise proof on capacity-constrained output — it assumes a
   stationary/uniform process. Use the density-warped (differential-domain) PCF.

---

## Known gaps

* **No per-term gradient-magnitude logging.** This is the instrument that distinguishes
  "the repulsion term fought and lost" (a finding) from "it was out-gunned by capacity"
  (a mis-weighted, invalid run). Worth adding before making strong claims about the objective.
* `config/objective.yaml` is documentation only — it is not read at runtime; pass CLI flags.
* Module imports still resolve model/dataset code from `control_fm` (identical copies); only the
  *output paths* were moved to `control_gt_free/`.

---

## History

`README_objective_driven_stippling.md` is the original plan for the pure GT-free method. It is
kept for provenance; where it and this file disagree, **this file is current** — notably, the
plan's `train/objective_train.py` and `overfit_objective.py` were folded into `train_control.py`
and `test_overfit.py`, and the pure objective-driven direction was closed.
