# `train_control_descriptor.py` vs `train_control_descriptor_consistency.py`

**One difference: the consistency variant adds a descriptor-consistency loss term. Everything else
is identical.** They are a control/treatment pair for a single question, not two different models.

Both are forks of `control_v4/train_control.py`, and both are descriptor-conditioned.

---

## The question they answer together

The model is conditioned on a measured descriptor field via spatial FiLM
(`DynamicControlNetDescriptor`). Nothing forces it to *use* that conditioning — it could ignore the
descriptor and still drive the denoising loss down, because ρ already predicts most of the target.
That is the plan's named failure mode: *metric response without visual change*.

| | how the model learns descriptor → output |
| --- | --- |
| `train_control_descriptor.py` | **implicitly.** Paired `(descriptor, target)` data and the diffusion loss alone. The association forms only because matching the target requires it. |
| `train_control_descriptor_consistency.py` | **explicitly, in addition.** Measures the descriptor the model ACTUALLY produced from its decoded `x0` and penalises the difference from the one it was asked for. |

`train_control_descriptor.py` is the **control**. Run both; the comparison is the experiment.

Early evidence (one epoch, FiLM-only): `film |scale|` grew `0 → 6–10e-2` and `mean|gate|`
`0 → ~3e-2`. So the implicit path *does* work. The open question is whether closing the loop makes
it stronger or faster — not whether it works at all.

---

## Identical in both

Everything that defines the model and the data:

- `DynamicControlNetDescriptor` with spatial FiLM, zero-init gates, `--descriptor-inject film|hint|both`
- `OracleStippleDataset` — one `(image, oracle)` sample per pair, 7 oracles
- Split **by source image**, reproducing `train_control.py`'s `randperm(seed=42)` exactly
- The descriptor-aware eval wrapper, so validation panels are sampled *conditioned*
- `film |scale|` / `mean|gate|` logged every epoch
- Checkpoint naming via `WEIGTHS_FILENAME_FORMAT` / `BEST_WEIGHTS_FILENAME` (ported from
  `train_control.py`, which also switched the extension to `.ckpt`)
- The adopted V4 baseline, below

### The V4 baseline (both trainers)

Per `control_v4/ddpm_imitation_improvements.md`, *Currently adopted in V4 (2026-07-29)*:

| | setting | why |
| --- | --- | --- |
| base | **from scratch**, `BASE_CKPT_PATH = ""`, denoiser unfrozen | matched pretrained-init on every sampling metric; a model carrying no GBN weights defuses "you're just reusing GBN" |
| C2 density loss | `weight 0.8`, `kde-grid 32`, low-t masked `t < 0.4·T`, `warmup 5` | consistent clump reduction toward the teacher, no cost to dmatch/chamfer |
| schedule | full 1000 steps, `TRAIN_TRUNCATION_RATIO = 1.0` | keeps every inference truncation supported; operating point ~0.5 |
| C1 logit-normal t | **rejected** | no gain at 0.5, breaks half the icons at trunc 1.0 |
| EMA | decided, not yet in code | cannot be retrofitted to existing checkpoints |

Both trainers carried `DENSITY_LOSS_WEIGHT = 0.0` until 2026-08-09. Runs started before that used
the **pre-C2clean** baseline and are not comparable to anything trained after.

A caveat inherited straight from the V4 doc: from-scratch was validated *without* the density loss
(clean isolation), and C2clean was validated *pretrained*+density. **"From-scratch + C2 density
loss" has never been trained.** M2 is the first run of that combination, so a surprising result may
belong to the baseline rather than to descriptor conditioning — which is precisely what the
FiLM-only control is there to separate.

### Background points are removed from the targets

`prepare_offsets.py` and `precompute_descriptors.py` both default to `--drop-white-points` at
`--white-threshold 255`: a ground-truth point whose source pixel is at least that bright is dropped
and replaced by a duplicate of a survivor, so the count stays at G·G. The reason is a training
dynamic rather than fidelity — shown a few background points, the model learns the background is
permissible territory and then puts far more there than the teacher ever did.

**The two must always agree.** Filtering the offsets but not the descriptors would tell the model
that a filtered point set carries the descriptor of the unfiltered one: a silent label error, not a
crash. They agree without coordinating an RNG — the keep mask is a pure function of
`(points, gray, threshold)`, and only the choice of which survivor to duplicate is random, which the
descriptors never see because `drop_exact_duplicates` runs before measurement.

Two consequences worth knowing: `--reuse` in `prepare_offsets.py` is force-disabled while filtering
(the cached offsets were built by `train_control` under its own setting, which cannot be verified
from here), and **M0's separation numbers predate this** — they were measured on unfiltered points.

---

## Only in the consistency variant

| flag | default | meaning |
| --- | --- | --- |
| `--descriptor-loss-weight` | `0.0` | **0 disables the term**, making the file behave exactly like the control |
| `--descriptor-loss-keys` | `nn_cv,aniso` | which descriptors the loss optimises |
| `--descriptor-loss-t-frac` | `0.4` | apply only where `t < 0.4·T` |
| `--descriptor-loss-warmup-epochs` | `5` | linear ramp, matching C2's convention |
| `--descriptor-pcf-sigma` | `0.08` | soft-binning width, only used if `pcf_peak` is in the keys |
| `--descriptor-loss-grad-log-every` | `2000` | `[DESC grad]` gradient-ratio logging |
| `--descriptor-loss-probe` | `off` | measure the ratio but **do not apply** the term |

At `--descriptor-loss-weight 0.0` the two files are functionally the same program. The default is 0
precisely so the variant can't be run "by accident" with an uncalibrated weight.

### Why only `nn_cv,aniso`

`calibrate_soft_descriptors.py` measured each differentiable descriptor against the exact numpy one:

| descriptor | agreement | in the loss? |
| --- | --- | --- |
| `nn_cv` | r = 1.00000, rank MATCH | **yes** |
| `aniso` | r = 0.99863, rank MATCH | **yes** |
| `edge_align` | r = 1.00000, rank MATCH | opt-in — also needs ρ and a Sobel map whose construction must match `precompute_descriptors` exactly |
| `pcf_peak` | r = 0.99963 at σ=0.08 | **no** — faithful *values*, pathological *gradients* |
| `cap_cv` | not implemented | needs a Monte-Carlo argmin; the one descriptor where the surrogate could genuinely diverge |

`pcf_peak` is the interesting exclusion: its per-neighbour normaliser divides by ~1e-9 at the σ
required for fidelity, amplifying gradients ~3e8. It stays a **conditioning channel** in both
trainers — only the loss term omits it.

---

## Running them

```bash
python background_running_descriptor_train.py         # control  (FiLM only)
python background_running_descriptor_consistency.py   # treatment (FiLM + consistency loss)
```

Re-calibrate the weight if anything about the descriptors or batch size changes — probe first, never
guess:

```bash
python control_v4_mix_metrics/train_control_descriptor_consistency.py \
  --source .../Icons-50_1024_Oracles/source --no-freeze-denoiser \
  --descriptor-loss-weight 1.0 --descriptor-loss-probe \
  --descriptor-loss-warmup-epochs 0 --descriptor-loss-grad-log-every 50 --epochs 1 \
  --out control_v4_mix_metrics/train_outputs_desc_probe
```

Then `weight = 0.1 / median_ratio`. Three probes gave a median ratio ≈ 1.2 → **0.08**.

---

## What to compare

1. **`film |scale|` and `mean|gate|` per epoch.** Both start at exactly 0 by construction. Does the
   treatment grow faster or further? If either run flatlines near 0, that run's model is ignoring the
   descriptor.
2. **Validation panels.** The model column should track the GT column's style oracle by oracle.
3. **`[DESC grad] ... ratio=`** in the treatment: should sit in 0.05–0.20 and must never print `nan`.
   NaN there was a real bug — `atan2(0,0)` on coincident neighbours, found with
   `torch.autograd.detect_anomaly` and fixed; `debug_soft_nan.py` is the regression harness.

---

## Data state (2026-08-09)

All seven oracles now have exact `.npy` coordinates for all 10,000 icons. The earlier caveats are
resolved:

- BNOT's 36 missing icons are generated. The crash was an out-of-bounds read in the solver's line
  search (`collect_visible_*` builds visible-only arrays, applied with `hidden=false`, which walks
  every vertex); a second bug left the optimiser stuck once a vertex was hidden. Both fixed in
  `BNOT_new/ibnot_cli`. 9,999 targets hold exactly 1024 points; `emoji-one_4_clinking_beer_mugs`
  holds 1023 and is padded by one duplicate.
- GBN / WVS / BNOT no longer derive from PNG centroids. `LINK_DATASETS` pointed at the PNG-only
  `icons-50_512_*` datasets, which is why `target_{gbn,wvs,bnot}` had 10,000 png and 0 npy; it now
  points at `Icons-50_1024_*`.

Rebuild after any relink — `--force` is required, and `--no-reuse` is implied while filtering:

```bash
python control_v4_mix_metrics/build_oracles_dataset.py --stage link --link-existing gbn,wvs,bnot --force
python control_v4_mix_metrics/build_oracles_dataset.py --stage verify
python control_v4_mix_metrics/precompute_descriptors.py --stage all --force --workers 32
python control_v4_mix_metrics/prepare_offsets.py --no-reuse --force --workers 32
```

### Known, quantified: the source images are not identical across oracles

`Icons-50_1024_WVS` and `Icons-50_1024_BNOT` were solved against a source variant differing from
`Icons-50_1024_GBN` (which is byte-identical to the Oracles source) by **at most 7/255, mean
0.28/255, over ~25% of pixels** — a resampling difference, not different artwork. Their points
therefore follow a ρ about 0.1% away from the one the model is conditioned on.

Far below the oracle separation M0 measured (|d| > 4), so it does not threaten the result. But it is
an asymmetry *between oracles*, and M2's premise is comparing oracles on a shared ρ, so it belongs in
any writeup. Removing it means re-running both stipplers against the Oracles source.
