# Milestone 0 — Multi-oracle descriptor separation (BLOCKING gate)

Status: **PASSED.** 396 class-stratified icons, **21/21 oracle pairs separated**, visibility
confirmed by inspection. Conditioning stack settled at **5 descriptors**
(`descriptor_fields.CONDITIONING_KEYS`).

Plan: `descriptor_control_plan.md`. Baseline: `control_v4` (the adopted **C2clean** recipe —
unfrozen denoiser, C2 density-match KDE loss, full-schedule training, sigmoid
`AdaptiveGateInjection`; see `control_v4/ddpm_imitation_improvements.md`, "Currently adopted in V4").

## Files

| File | Purpose |
| --- | --- |
| `oracles.py` | Floyd–Steinberg, ordered/Bayer dither, white noise, jittered grid at matched N. Torch-free. |
| `descriptor_fields.py` | The five plan descriptors as **local `(G,G)` fields**, not per-image scalars. |
| `descriptor_selftest.py` | Data-free math check. Run first; must print `SELF-TEST PASSED`. |
| `point_io.py` | Stem enumeration, centroid extraction, lattice quantisation. |
| `gen_oracles.py` | Generates the new oracles over the icon set. Resumable. |
| `m0_run.py` | The gate: separation + non-redundancy + visibility panels + two artefact controls. |
| `stems_m0.txt` | 400 stems, 8 per icon class — the stratified M0 sample. |

## How to run

```bash
python control_v4_mix_metrics/descriptor_selftest.py          # must PASS before anything else

python control_v4_mix_metrics/gen_oracles.py \
    --source <train>/icons-50_512_GBN/source \
    --out <gen_root> --workers 16                             # add --stems-file for a subset

python control_v4_mix_metrics/m0_run.py \
    --gen-root <gen_root> --out control_v4_mix_metrics/m0_outputs --workers 16
```

`--stems-file` avoids an `os.walk` over 10k files per directory, which on the SSHFS mount cost more
than the entire descriptor computation. Pass it whenever you are not doing a full run.

## Results (396 icons, 7 oracles, G=32, window=5)

| descriptor | gbn | wvs | bnot | fs | ordered | white | jitgrid | pairs sep. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nn_cv` (primary) | 0.183 | 0.098 | 0.137 | 0.178 | 0.158 | 0.483 | 0.356 | 17/21 |
| `cap_cv` | 0.756 | 0.816 | 0.859 | 0.911 | 0.887 | 1.078 | 0.866 | 12/21 |
| `edge_align` | 0.590 | 0.876 | 0.864 | 0.922 | 0.988 | 0.997 | 0.947 | 6/21 |
| `aniso` | 0.019 | 0.004 | 0.005 | 0.006 | 0.005 | 0.100 | 0.013 | 6/21 |
| `pcf_peak` | 2.112 | 2.993 | 1.952 | 2.982 | **4.425** | 1.870 | 1.909 | 15/21 |

(Raw-view counts. The gate itself judges cross-family pairs on the common raster — see Controls.)

**21 / 21 oracle pairs are separated by at least one descriptor** (|d| ≥ 0.8 and paired agreement
≥ 0.90). Coverage of those 21 pairs under the gate's mixed view: `nn_cv` 20, `pcf_pos` 17,
`pcf_peak` 12, `cap_cv` 9, `edge_align` 6, `aniso` 6. **No descriptor has any UNIQUE coverage** —
every pair is separated by at least two. That redundancy is robustness, not waste: no single
descriptor is a single point of failure for any pair.

Cross-check against the earlier single-axis Gate 0: `nn_cv` gives gbn 0.183 vs wvs 0.098, d = +3.79,
agreement 1.00 — WVS more regular than GBN, the same direction and a comparable effect size to the
`norm_nn_cv` result that cleared the WVS↔GBN gate. The scalar→field rewrite did not drift from it.

The descriptors carry genuinely different work, which is the multi-descriptor claim doing real labour
rather than several dials on one axis: `nn_cv` orders the regularity axis but puts gbn, fs and
ordered in the same place (0.16–0.18); `edge_align` is what separates gbn from all six others
(d ≈ −2); `aniso` isolates white noise; `pcf_peak` isolates ordered dither. No single descriptor
separates more than 20/21.

### Non-redundancy

`nn_mean` vs `nn_cv` correlate at **r = −0.957**, above the 0.90 threshold — one axis wearing two
hats, as the plan warned. `nn_cv` is kept (it is the measure with prior validation) and `nn_mean`
dropped. `pcf_pos` survives the threshold (r = 0.835 with `nn_cv`, the highest surviving pair) but
adds **no unique coverage** — all 21 pairs stay covered without it — so it is kept as a reported
diagnostic and left out of the conditioning stack. Everything else is below 0.75.

Critically, **no descriptor tracks rho**: |r| ≤ 0.19 against `rho_mean`, so none is a tone readout in
disguise — the failure that killed the first WVS↔GBN attempt.

So the conditioning stack is **5 descriptors** — `nn_cv`, `cap_cv`, `edge_align`, `aniso`,
`pcf_peak` — mapping one-to-one onto the plan's descriptors 1–5. M1/M2 must read
`CONDITIONING_KEYS`, not `FIELD_KEYS`.

### Controls

|  | `nn_cv` | `cap_cv` | `edge_align` | `aniso` |
| --- | --- | --- | --- | --- |
| quantisation (GBN → dither raster) | **+0.086** | +0.087 | +0.115 | −0.000 |
| round-trip (render → re-extract) | −0.006 | −0.004 | +0.023 | −0.001 |

The **round-trip control passes cleanly**: measuring through centroid detection instead of exact
coordinates moves everything by ≤ 0.023, an order of magnitude below the oracle gaps. Comparing the
disk oracles against the generated ones is therefore valid.

The **quantisation control does not**, and it is why `m0_run.py` computes every descriptor twice.
Snapping GBN onto the halftone raster shifts `nn_cv` by +0.086 — larger than the raw gbn|fs gap
(0.005), gbn|ordered (0.026) or fs|ordered (0.020). Any continuous-vs-dither `nn_cv` comparison in
the raw view is measuring the lattice, not the oracle. So every oracle is also measured on a
**common raster**, where the lattice is shared and cancels, and cross-family pairs are judged only
there. Same-family pairs stay raw, because quantising two continuous oracles discards real
information and would under-report a genuine difference.

## Resolved: `fs|ordered` needed descriptor 5

Descriptors 1–4 separated 20/21 pairs and left Floyd–Steinberg tied with ordered dither (best
d = 0.72). That was **not** an FS bug: on constant tone FS produces serpentine banding and ordered an
exact square lattice, plainly different by eye (`fs_worm_diagnostic.png`). What they share is a
similar NN-distance CV and near-zero anisotropy. Their real difference is **periodicity**, which
nothing in 1–4 measures — exactly the trigger the plan names for descriptor 5.

Adding the local PCF first-peak resolved it: **fs|ordered d = −4.46, agreement 1.00,
non-overlapping**, ordered at 4.43 (sharp lattice peak) against fs at 2.98. 20/21 → 21/21.

`pcf_peak` needed three non-obvious decisions, each forced by a measurement:

**Pooled per window, not per point or per image.** A single point's 32 neighbour distances cannot
locate a peak, but a per-image PCF is what the plan forbids — a global number cannot say *where*. So
each point's neighbour-distance histogram is splatted into its own cell and box-filtered over the
window, giving ~600 distances per cell while staying local.

**Distances normalised by each point's own local spacing**, for the same reason `nn_cv` is —
otherwise the peak just tracks rho.

**Smoothing along u is OFF, deliberately.** A 3-bin smooth suppresses the max-of-noise bias but
crushed the lattice peak from 5.26 to 1.77, destroying the sharpness that is the entire point. The
bias is common to all oracles at matched N, so it offsets every column equally and cancels in the
paired comparison — the only comparison made.

One honest limitation, documented in the code: `pcf_peak` is a **lattice-periodicity detector, not a
monotone "amount of structure" ranking**. On synthetic fixtures an exact lattice reads 5.26 against
1.75 (jittered), 1.92 (white noise), 2.32 (lines). White noise sitting slightly above a jittered
lattice is not a bug — self-normalising by a locally-estimated spacing is what buys
density-invariance, but for a clustered process that estimate fluctuates too, and a ratio of two
fluctuating quantities concentrates into a spurious peak. Raising `k_pcf` (32→48) or the window
(5→9) changes it by nothing, which is how we know it is this and not sample noise.

## Visibility (the check anisotropy died on)

`m0_outputs/visibility/*_blind.png` — full size, dots only, uniform black, unlabeled, no titles.
Thumbnails over-ink to solid black and are useless for this judgement, so only full-size columns are
written; `KEY.txt` holds the column order separately so the panel can be judged blind first.

Confirmed distinct by inspection: **GBN** (contour-following bands), **white noise** (obvious clumps
and voids), **ordered** (hard periodic lattice), **jittered grid** (loose lattice). WVS and BNOT both
read as uniform fill and are the hardest pair by eye, though they separate metrically (`nn_cv`
d = −3.93). The pair the plan cares most about — **GBN vs Floyd–Steinberg — is clearly distinct**.
This is the check the anisotropy branch failed, and it passes here.

## Findings worth not re-deriving

**The dither raster must be derived from the point budget, not chosen.** At native 512/1024 the ink
fill is ~0.1%; ordered dither collapsed to a bare lattice that ignored mid-greys entirely (tone
corr 0.48) and FS discharged accumulated error into white background (corr 0.77). Both violated the
plan's requirement that oracles respect the density map, and at 0.1% fill FS emits isolated dots with
no worm structure at all — high resolution destroys the character it is there to provide.
`halftone_raster()` targets ~25% fill over the inked region; every oracle then reproduces tone at
corr ≥ 0.95.

**The k-NN double-angle statistic is biased, and the bias reads as signal.** The naive
`|mean exp(2iφ)|` over k neighbours — what `control_v4_pilot/aniso_pilot.py:252` implements and marks
BROKEN — gives white noise 0.316 where √(π/4k) at k=8 is 0.313. Every isotropic oracle would have
scored ~0.31 and the descriptor would have looked informative while carrying nothing. Rayleigh's
correction `(k|R|² − 1)/(k − 1)` fixes it exactly: white noise now reads 0.07, a line process 0.63.
Pool the squared statistic and take the root *after* the window mean, or clipping reintroduces the
bias.

**`edge_align` is undefined, not large, inside flat regions.** A solid black icon interior has zero
image gradient, so "alignment relative to structure" has no referent there. Guarding the division
with an epsilon produced pooled means around 1e10 and the descriptor separated 0/21 pairs. Cells
whose window has ink but essentially no gradient must be NaN, with the floor set as a fraction of the
image's own rho-weighted mean gradient — an absolute floor, since a relative one cannot tell "no
structure" from "weak structure".

**Never pad point sets with uniform random to reach N when measuring.** `train_control.py:312-317`
does this to force a fixed tensor shape, which is correct for a diffusion target but fatal here: it
injects uniform noise into a point set whose regularity is about to be measured, manufacturing the
difference under test. `point_io.extract_centroids` subsamples but never pads, and reports
under-count to the caller instead.
