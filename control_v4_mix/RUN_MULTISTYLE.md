# Multi-Style Stippling: WVS / GBN / DITHER (and beyond)

How the current multi-oracle style-conditioning approach works, end to end. This generalizes the
scalar WVS<->GBN model (see `RUN_STYLE.md`) to **K oracles** driven by a **style vector**.

Run everything from the PROJECT ROOT (the dir with `control_v4/`, `control_v4_mix/`, `config/`,
`models/`, `utils/`).

---

## 1. The idea

Each "oracle" is a stippling algorithm (WVS = Weighted Voronoi / Lloyd, GBN = Gaussian Blue Noise,
DITHER = dithering, ...). They all place ~1024 points on the SAME source image, so they share the
same density field rho and differ only in **local point arrangement** (a non-decomposable
point-process property -- see `style_axis_wvs_gbn_plan.md` and the Gate-0 analysis).

We train ONE control model conditioned on a K-dim **style vector** `s`:

* Training uses only the simplex **vertices** (one-hot): `WVS=[1,0,0]`, `GBN=[0,1,0]`,
  `DITHER=[0,0,1]`.
* At inference we can ask for any point in the simplex -- e.g. `[0,0.5,0.5]` = a GBN+DITHER **mix**.
  The middle of the simplex is the (unsupervised) interpolation hope: does the model blend styles it
  was never explicitly trained to blend?

**Backward compatible.** For K=2 (WVS,GBN) a single scalar `s in [0,1]` maps to `[1-s, s]`
(`style_scalar_to_vec`), so `s=0`=WVS, `s=1`=GBN, `s=0.5`=even blend -- exactly the original
single-value interface.

---

## 2. How the conditioning is wired

`DynamicControlNetMultiStyle` (subclass of control_v4's `DynamicControlNet`) adds ONE thing:

```
style_mlp:  Linear(K, ch*4) -> SiLU -> Linear(ch*4, ch*4)     # last layer ZERO-INIT
forward:    temb = temb + style_mlp(style_vec)                # added to the timestep embedding
```

* Every control resblock already consumes `temb`, so this conditions the whole control branch.
* **Position-free**: `s` is a global vector, no positional embedding -> G-transfer (train 32, infer
  48/64) is preserved (re-verify after training).
* **Zero-init** last layer: at step 0 `style_mlp(s)=0`, so the model is bit-identical to the
  unconditioned baseline and *learns* to use `s`. (The control INJECTION layers are also zero-init,
  so nothing conditions the output until training moves them -- see `verify_multistyle_model.py`.)

Only `forward()` is overridden; GECCO / hint / injection paths are untouched. control_v4 stays frozen.

---

## 3. Data, oracles config, and the split

**Oracle registry** -- `control_v4_mix/oracles_config.py` maps NAME -> data root, so the command
line only needs names:

```
--oracles "WVS;GBN;DITHER"          # names; paths come from oracles_config.py
--oracles "WVS;GBN;MINE:/my/root"   # inline override still allowed
```
List ORDER fixes the one-hot index (WVS=0, GBN=1, DITHER=2). Keep it consistent between train & eval.
Each root must contain `source/`, `target/` (rendered stipple, ~1024 dots), `processed_offsets/`
(auto-exported from `target/` on first run).

**Shared source (rho).** All oracles share the first oracle's `source/` as the density condition;
only their `target` offsets differ. One `(image, oracle)` pair per sample -> the dataset yields
`K x N` samples with a one-hot `style_vec`.

**Train/val split** -- `control_v4_mix/data_split.py`:
* Default: an explicit **manifest** of val basenames, `control_v4_mix/validation_manifest.json`
  (`split_from_manifest`), which reproduces the reference control_v4 val set EXACTLY (matched by
  basename), independent of source-folder structure. Watch the startup line
  `Val set from manifest ...: 1000 val icons` -- 1000 = a clean match.
* Fallback (no manifest): `source_train_val_split` -- control_v4's method
  (`randperm(N, seed=42)`, val=tail) keyed on the source-folder listing, oracle-independent.

---

## 4. Files (`control_v4_mix/`)

| File | Role |
| --- | --- |
| `oracles_config.py` | NAME->path registry + `resolve_oracles` |
| `MultiStyleStippleDataset.py` | (image, oracle) samples with one-hot `style_vec` |
| `DynamicControlNetMultiStyle.py` | control net + style MLP into temb; `style_scalar_to_vec`; inference wrapper |
| `data_split.py` | manifest / source-folder train-val split |
| `train_control_multistyle.py` | training (fork of control_v4 recipe; unfrozen base) |
| `verify_multistyle_model.py` | data-free wiring check (zero-init / wiring / gradient) |
| `eval_multistyle_interpolation.py` | samples the simplex, metrics + dot-render montage |
| `validation_manifest.json` | the 1000 held-out val basenames (reference split) |

Launchers at project root: `background_running_multistyle.py` (K=2 `WVS;GBN`),
`background_running_multistyle_k3.py` (K=3 `WVS;GBN;DITHER`).

---

## 5. Run order

```
# (a) data-free wiring check -- 3x PASS; use --K 2 for the WVS/GBN case
python control_v4_mix/verify_multistyle_model.py --K 3

# (b) train (background). Paths are in oracles_config.py; only names on the CLI.
python background_running_multistyle_k3.py
#   -> tail logs/train_control_multistyle.py_k3_*.log
#   -> confirm: "Multi-style dataset: ... K=3 oracles=['WVS','GBN','DITHER']; per-oracle counts=..."
#              "Val set from manifest ...: 1000 val icons"

# (c) interpolation eval on a checkpoint
python control_v4_mix/eval_multistyle_interpolation.py \
    --control-ckpt control_v4_mix/train_outputs_multistyle_wvs_gbn_dither/checkpoints/dynamic_controlnet_v4_ep1000.pt \
    --oracles "WVS;GBN;DITHER"
```

**Eval knobs**: `--style-vecs "0,0.5,0.5;0.33,0.33,0.33"` (explicit mixes; default = vertices +
pairwise midpoints + centroid), `--s-values 0,0.5,1` (K=2 scalar mode), `--icons "a.png,b.png"`
(specific icons), `--val-manifest` (same held-out set as training), `--n-samples`, `--n-montage`,
`--truncation` (1.0 = full sampling from noise, matching train_truncation=1.0).

Outputs in `--out`: `interp_metrics.json/.csv`, an interpolation curve, and
`interp_montage.pdf` -- **oracle teachers | model at each style vector** as dot renders.

---

## 6. Reading the eval

Per style vector the eval reports `norm_nn_cv` (the Gate-0 regularity descriptor), clumping, and
Chamfer distance to EVERY oracle teacher.

* **Endpoint fidelity**: a vertex (e.g. `[0,1,0]`) should match that oracle's teacher (low Chamfer to
  it, teacher-level `norm_nn_cv`).
* **Mixture blend**: for `[0,0.5,0.5]`, Chamfer should be LOW to the two blended oracles and HIGHER
  to the excluded one; `norm_nn_cv` should sit BETWEEN the two teachers.
* **Visual**: the montage is the real test -- does a mixture look like a genuine blend, and can you
  SEE it (not just move a metric)?

Three failure modes to watch: **snapping** (a mix collapses to one vertex), **muddy mixture**
(capacity/spacing degrades mid-simplex), **invisible interpolation** (metric moves, eye cannot tell).

---

## 7. DITHER / BNOT caveats (read before trusting a K>=3 run)

DITHER lives at `Icons-50_1024_DITHER` (1024px) -- a different folder/resolution from the 512 WVS/GBN.
Because all oracles share ONE source (WVS 512) for rho, two things MUST hold or DITHER silently
contributes 0 samples / crashes offset export:

1. **Stem match**: DITHER's `processed_offsets` stems must equal the WVS source stems (same
   names/subfolders). `Icons-50/...` vs `icons-50/...` would break it.
2. **Offset export**: DITHER `target/` must yield ~1024 dot centroids (a perfect square for the
   32x32 grid). If Floyd-Steinberg output is a halftone BITMAP rather than dots, export misbehaves
   (BNOT previously hit an Inf/NaN in export).

Quick check: the startup `per-oracle counts = {...}` line -- a near-zero DITHER count means the
format/stems do not line up. BNOT is a commented 4th oracle in `oracles_config.py`.

---

## 8. Capacity vs optimization (practical note)

For a handful of oracles that share the density backbone, this is almost certainly
**optimization-time-limited, not parameter-limited**: the control branch is a full copy of the
denoiser (millions of params); `style_mlp` is tiny (~263K) and barely grows with K; the oracles
differ only in local arrangement, so marginal capacity per style is small. The scalar model already
proved the mechanism (a coherent WVS<->GBN blend by ep1000, just soft endpoints = undertrained).
Biggest speedup for K>=3: **warm-start the control branch + denoiser** from a converged checkpoint
(the `style_mlp` input dim changes with K so it re-inits, but the backbone already knows how to place
density-following points). Diagnose which regime you are in by watching per-oracle endpoint fidelity:
plateaus BELOW single-oracle baseline -> capacity/interference; just SLOWER to reach it ->
optimization.
