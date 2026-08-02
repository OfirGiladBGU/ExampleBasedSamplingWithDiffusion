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
- Base trains **from scratch** (`BASE_CKPT_PATH = ""`), denoiser unfrozen
- The descriptor-aware eval wrapper, so validation panels are sampled *conditioned*
- `film |scale|` / `mean|gate|` logged every epoch
- Every inherited `train_control.py` flag: C2 density loss, min-SNR, truncation, smart-init, …

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

## Caveats on the current data

Both runs share the same dataset, so the A/B is valid. The absolute numbers are not yet clean:

- GBN / WVS / BNOT descriptors and offsets derive from **PNG** (pixel-quantised) until those
  generators finish re-running with `--export_npy`. Only `fs`/`ordered`/`white`/`jitgrid` are exact.
- BNOT is short 36 icons, pending the CGAL rebuild in `BNOT_new/ibnot_cli/build`.

When both land:

```bash
python control_v4_mix_metrics/precompute_descriptors.py --stage all --force --workers 32
python control_v4_mix_metrics/prepare_offsets.py --no-reuse --force --workers 32
```
