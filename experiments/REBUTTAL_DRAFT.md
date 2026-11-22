# StippleDiffusion (paper1182): rebuttal draft and evidence bank

Started 2026-10-28, last synced with the results 2026-11-18. This file collects every result we can
use in the rebuttal ("ammo"), with its status, plus draft answers to the concerns we expect. Reviews
arrive Nov 23; the rebuttal is due Nov 30. The experiment plan is in [REBUTTAL_PLAN.md](REBUTTAL_PLAN.md)
and the run log in [CONTEXT.md](CONTEXT.md).

**Paper files to update once the experiments are done** (not before; see CONTEXT.md ground rules):
`paper_rebuttal/main_EG_rebuttal_11.10.tex`, `paper_rebuttal/supplemental_EG_rebuttal_11.10.tex` and
`paper_rebuttal/tables_rebuttal/` on the server (copies of the user's local files, 2026-10-28).

Status tags: **FINAL** (numbers verified, can be quoted) · **CHECK** (computed, needs one more
verification) · **PENDING** (experiment running or not started).

**Sampler decision (user, 2026-10-30, confirmed 2026-11-18):** the few-step sampler is stochastic
DDIM with 25 steps (`ddim-25-eta1`), same SDEdit start (t = 500), same trained weights. It goes into
the **supplement** as a faster sampling variant found after submission (a sampler, not a new model:
no retraining), with its curve and a short paragraph in the paper's runtime section. The paper's
tables keep the submitted configuration (DDPM-500).

---

## 0. Watch-outs before writing anything

1. **Tone correction (A12).** The paper's Table 3 (+0.83 dB, 24/24) was scored on the one-step
   proxy decode with the optimization's own fixed noise, not on the real sampler. On the real
   sampler, that one-step field gives +0.09 dB (n.s.). With gradients through the full DDIM-25 chain
   it is +0.39 dB (18/24, p = 0.002). Never quote +0.83 dB or "all 24 targets" as a property of the
   sampler's output.
   **Updated 2026-11-21:** quote the paper's 500-step sampler, as the full tex version does:
   +0.38 dB (17/24, p = 0.004); +0.39 dB (18/24) is the 25-step output. **Do not claim the field
   beats the tone curve:** optimized the same way, the curve reaches the same PSNR on the real
   sampler (field − curve +0.01 dB, better on 11/24, p = 1), and within 0.2 dB at every gain of the
   sweep. The field's own advantages: lowest MAE, and +0.23 dB over gradient-free feedback.
   "Above GBN's tone" holds clearly only for the 25-step output (16.95 vs 16.81 dB); on the
   500-step sampler it is 16.89 vs 16.81, so do not lead with it.
2. **Point budgets are visual only (A9).** The models are trained at one budget. Do not report
   metrics at other budgets; answer with Fig. 4 and the runtime graph.
3. **"Ours-BNOT" contradicts the paper's claim that BNOT exemplars are impractical (A11).** Correct
   it in the rebuttal before a reviewer points it out.
4. **The aspect ratio does not explain the capacity gap (A4).** Do not offer it as the reason.
5. **Do not say the model "does not memorize" (A19).** It reproduces training exemplars more
   closely than held-out ones (0.15 vs 0.23 cells; denoising error 20–24× lower on training images).
   The claim that holds: its quality metrics are the same on training and held-out images.
6. **BNOT runtimes (2026-11-18). The NUMBERS in this item are superseded by watch-out 7; the solver
   story still holds.** Our BNOT port
   used a slow linear solver (Eigen `SparseQR` instead of the original's SuiteSparse QR). With the
   fix BNOT takes 5.2 / 5.8 / 7.1 / 8.5 / 9.5 / 20.9 / 12.9 / 15.3 / 17.2 s at 256 / 576 / 1024 /
   1600 / 2304 / 3136 / 4096 / 5184 / 6400 points (old, wrong: 7.2 / 8.8 / 13.6 / 25.5 / 59.2 / 333 /
   393 / 872 / 1802 s). Never write "BNOT grows by more than 100x", "1802 s at 6400" or "13.6 s per
   image". A1, A3, A11 and the W1 / W2 / Q1 rows below carry the new numbers; `runtimes_avg.json` and
   the runtime graphs are rebuilt. The BNOT point sets and quality numbers are unchanged. Cautions:
   - **The paper's 500-step sampler is no longer clearly faster than BNOT.** Fig. 7 values: ours
     5.75 / 5.80 / 5.88 / 6.02 s against BNOT 5.2 / 5.8 / 7.1 / 8.5 s at 256 / 576 / 1024 / 1600
     points, i.e. BNOT is faster at 256, equal at 576, and we are 1.2x / 1.4x faster at the training
     budgets. Do not claim a speed advantage over BNOT for DDPM-500; the 25-step sampler has it
     (0.44 / 0.61 s, 16x / 14x at 1024 / 1600).
   - The 3136 mean (20.9 s) is carried by two slow images (88 s and 39 s; the other eight 7–15 s).
   - BNOT was re-timed on an AMD EPYC 7702P node (1 thread); WVS, GBN and the paper's own curve are
     the stored measurements from the original, unrecorded machine.
   See the resolved TODO in CONTEXT.md and `BNOT_V2/CONTEXT.md`.
7. **Runtimes of record (2026-11-20): `runtime_v10`, all four methods re-timed; quote only these.**
   10 Icons-50 validation images (the first 10, sorted; the old pinned images made BNOT at its true
   defaults blow up), BNOT at its true defaults (500 / 500 iterations, epsilon 1.0, fixed build).
   BNOT / WVS: one CPU-only EPYC 7702P node, 1 thread. GBN / ours: one job on a dedicated RTX 6000
   Ada with 16 reserved EPYC 7343 cores (node shared with other users' jobs, our runs never
   overlapping). Ours: median of 3 repetitions per image; exact OT below 3136 points,
   `bisection-blocks16` from 3136. **Median over the 10 images** (user decision), at 256 / 576 / 1024 /
   1600 / 2304 / 3136 / 4096 / 5184 / 6400 points:
   - BNOT 3.6 / 3.9 / 6.1 / 5.8 / 7.0 / 8.3 / 9.4 / 10.8 / 12.8 s
   - WVS 1.7 / 3.8 / 6.8 / 11.0 / 15.9 / 21.5 / 32.9 / 37.8 / 50.8 s
   - GBN 9.3 / 9.0 / 9.5 / 9.5 / 9.6 / 9.7 / 9.9 / 10.0 / 10.1 s
   - Ours DDPM-500 5.3 / 5.4 / 5.5 / 5.9 / 6.1 / 5.9 / 6.3 / 6.5 / 7.0 s
   - Ours DDIM-25 0.27 / 0.32 / 0.42 / 0.60 / 1.01 / 0.70 / 0.88 / 1.06 / 1.33 s
   Stored in `outputs/icons_results_runtimes/runtimes_avg.json` (provenance `RUNTIMES_SOURCE.txt`
   there); raw data and mean / exact-OT variants in `outputs/runtime_v10/`. Readings: DDPM-500 is on
   par with BNOT at the training budgets (5.5 / 5.9 s against 6.1 / 5.8 s) and slower at 256 / 576;
   it beats GBN at every budget; DDIM-25 is 14x / 10x faster than BNOT at 1024 / 1600. Paper figures:
   `paper_rebuttal/figures_rebuttal/` (the full version's Fig. 7, and the supplement graph with DDIM-25 for
   both versions). The non-full version keeps the submitted Fig. 7 and adds DDIM-25 (mean, same
   run) to it in the supplement.

---

## 1. Evidence bank

### A1. Few-step sampling: equal or better on every metric, much faster (E1) — FINAL

Same SDEdit start as the paper (Smart Init, forward-noised to t = 500), then stochastic DDIM (η = 1)
instead of 500 DDPM steps. Icons-50, 1000 test images, paired Wilcoxon tests.
Source: `experiments/outputs/few_step_sampling-Icons-50_1024/summary.md`.

| | denoise time / image | CVT | HOT | cap δc | centroid D | EMD | Sinkhorn | ρs |
|---|---|---|---|---|---|---|---|---|
| Ours-GBN, DDPM-500 (paper) | 5.42 s | 2.456 | 4.082 | 0.0178 | 0.0059 | 0.0132 | 0.0925 | 0.494 |
| Ours-GBN, DDIM-25, η=1 | 0.27 s | −0.5%* | −0.6%* | **−8.4%*** | −1.2%* | −0.1% | 0.0% | **+1.5%*** |
| Ours-GBN, DDIM-10, η=1 | 0.11 s | −0.4%* | −0.4%* | **−5.1%*** | −1.1%* | −0.1% | 0.0% | **+1.5%*** |
| Ours-BNOT, DDPM-500 (paper) | 3.23 s† | 2.422 | 2.855 | 0.0194 | 0.0032 | 0.0110 | 0.0922 | 0.509 |
| Ours-BNOT, DDIM-25, η=1 | 0.30 s | −0.6%* | −0.7%* | **−8.7%*** | −4.7%* | −0.4%* | 0.0% | **+1.6%*** |
| Ours-BNOT, DDIM-10, η=1 | 0.12 s | −0.6%* | −0.1%* | **−4.6%*** | −3.3%* | −0.1%* | +0.01%* | **+1.8%*** |

\* p < 0.05. CVT and HOT are ×10⁻⁵. † Different GPU (see CONTEXT.md, E1 open question 3).

- **Total time per stipple, runtime of record (watch-out 7, 2026-11-20):** DDIM-25 **0.42 s at 1024
  and 0.60 s at 1600 points**, against 5.5 / 5.9 s for DDPM-500 in the same runs; WVS 6.8 / 11.0 s,
  GBN 9.5 / 9.5 s, BNOT 6.1 / 5.8 s; at 6400 points 1.33 s (blocks16) against WVS 50.8, GBN 10.1,
  BNOT 12.8 s (medians over 10 images). The clean-run bullet below is superseded.
- *(Superseded 2026-11-20)* **Total time per stipple, clean run** (job 23786262: exclusive node, RTX 6000 Ada, exact OT,
  10 pinned runtime images, 3 repetitions, `outputs/runtime_clean-Icons-50/summary.md`): DDIM-25
  **0.44 s at 1024 and 0.61 s at 1600 points**, against 6.29 / 6.53 s for DDPM-500 re-timed on the
  same node. Baselines at 1024 / 1600: WVS 5.82 / 10.04 s, GBN 8.10 / 8.19 s (stored paper
  measurements), BNOT 7.08 / 8.49 s (fixed build, re-timed 2026-11-18; watch-out 6). With
  `bisection-blocks16` above 2304 points (visual-only budgets, Fig. 4 unchanged): 1.38 s at 6400
  (`outputs/runtime_clean_blocks16-Icons-50/summary.md`), against WVS 45.7 s, GBN 10.2 s, BNOT
  17.2 s. (The older `runtime_fewstep` numbers, 0.64 s, are superseded.)
- **Throughput** at batch 32, 1024 points: 60 stipples/s (DDIM-25), 149 (DDIM-10), 3.3 (DDPM-500).
- Denoising time is flat in the point budget (0.36 s for DDIM-25); the CPU Smart Init (mostly the
  OT assignment) dominates from about 1600 points.
- **The only significantly worse cells** are Ours-BNOT's Sinkhorn (+0.010% at 10 steps, +0.001% at
  25), which are negligible. Say "match or improve", not "improve".
- **On the other datasets and variants** (Icons-50 re-scored with the current metric code, the new
  variant against the paper's own point sets): δc −8.8% (Ours-BNOT), −9.2% (Ours-GBN), −2.9%
  (Ours-WVS), ρs +1.5–1.6%, every other metric equal or better. The full four-dataset table is E13.

#### A1b. Why fewer steps help: the variance check — FINAL

Ours-GBN, Icons-50, paired against DDPM-500 (1000 images): ddim-500-eta1 δc −1.9%, ρs +0.4%;
ddim-250 −3.0%; ddim-100 −4.4%; ddim-50 −6.4%; **ddim-25 −8.4%**; ddim-10 −5.1%.
So switching from DDPM's fixed variance β to DDIM's smaller posterior variance gives a small gain
by itself, and shortening the chain gives most of it, with the best result at 25 steps.
Wording: "replacing the DDPM variance with the DDIM (η = 1) posterior variance, and shortening the
chain to 25 steps, improves capacity by 8–9% and ρs by 1.5%."

### A2. The learned variants keep their samplers' ranking (E2) — FINAL

All six methods re-scored with one metric code on all 4 datasets; 84 pairs (4 datasets × 7 metrics
× 3 sampler pairs). Source: `outputs/rank_preservation/summary.md`.
- **Paired Wilcoxon (p < 0.05) decides the order: 78 agree, 4 flip, 2 ties**, so 78 of the 82
  decided pairs (95%).
- On means at table rounding: 73 agree, 2 flip, 9 ties (73 of 75). The earlier "73 of 74" came from
  stored JSONs with mixed metric code; do not quote it.
- Kendall's τ per dataset × metric: mean 0.90, τ = 1 in 24 of 28.
- Three of the four flips are capacity, BNOT vs GBN (Icons, CelebA, ShapeNet2D), where the two
  samplers are within 1–5% of each other; the fourth is ShapeNet2D CVT, BNOT vs WVS.
- Wording: "where the exemplar samplers differ significantly, the learned variants reproduce their
  order in 78 of 82 cases; the exceptions are near-ties between BNOT and GBN on capacity."

### A3. Ours-BNOT is the strongest learned sampler — FINAL (paper numbers, DDPM-500)

| dataset | HOT: BNOT / Ours-BNOT / GBN / WVS | centroid D | EMD |
|---|---|---|---|
| Icons-50 | 2.421 / **2.850** / 5.204 / 10.536 | 0.001 / **0.003** / 0.007 / 0.014 | 0.010 / **0.011** / 0.015 / 0.020 |
| CelebA | 6.118 / **7.045** / 12.864 / 21.807 | 0.001 / **0.003** / 0.010 / 0.016 | 0.013 / **0.014** / 0.018 / 0.024 |
| ShapeNet2D | 3.206 / **4.329** / 7.055 / 11.432 | 0.001 / **0.004** / 0.007 / 0.012 | 0.010 / **0.012** / 0.015 / 0.019 |
| Anime Chibi | 1.300 / **1.977** / 3.262 / 4.655 | 0.002 / **0.005** / 0.010 / 0.011 | 0.009 / **0.012** / 0.016 / 0.017 |

Ours-BNOT beats GBN and WVS on HOT, centroid displacement and EMD on all 4 datasets at the training
budget, and comes close to BNOT itself. Speed (watch-out 7): BNOT takes 6.1 s at 1024 points
and 12.8 s at 6400; a learned variant takes 5.5 s / 7.0 s with the paper's 500-step sampler (the
cost does not depend on the exemplar sampler), so no real advantage there, and 0.42 s / 1.33 s
with the 25-step sampler (A1).

### A4. The capacity gap comes from the model, not the exemplars; not from the aspect ratio — FINAL

- On each dataset the three learned variants have nearly the same capacity deviation, whatever
  their sampler. Anime Chibi: learned 0.110 / 0.129 / 0.107 (BNOT / WVS / GBN variants), samplers
  0.040 / 0.061 / 0.030. ShapeNet2D: learned 0.039 / 0.059 / 0.038, samplers 0.013 / 0.043 / 0.013.
- **E3: the aspect ratio is not the cause** (`outputs/aspect_ratio_capacity/summary.md`). On images
  that are already square (|log(w/h)| < 0.05) the learned variants still have 2–4× the samplers' δc
  (Chibi Ours-GBN 0.094 vs GBN 0.024, 107 images; ShapeNet2D 0.035 vs 0.012, 163 images), and the
  per-image gap barely correlates with anisotropy (Spearman ≤ 0.17).
- Gap ratio (learned / sampler δc, GBN): Icons 1.8× (10k training images, 1024 points), CelebA
  1.7× (5k, 1024), ShapeNet2D 3.0× (4k, 1600), Chibi 3.6× (2.6k, 1600). Halving the training set from
  Icons to CelebA did not change it, so the point budget (or the domain) is the more likely factor;
  the datasets do not separate them.

### A5. The SDEdit start at t = 500 is the right design — FINAL (Ours-GBN, Icons-50)

- Starting the reverse process later is worse at every start: HOT +19% (t = 300) to +111% (t = 50).
- Skipping the forward noise (starting from the clean Smart Init) is much worse: capacity +6745%
  at t = 500. The model expects the noise level of its timestep.
- **Starting from pure noise (no Smart Init) is worse too** (1000 images, vs DDPM-500 with SDEdit,
  `outputs/few_step_noise-Icons-50_1024/summary.md`): DDIM-25/50/100 and even DDPM-1000 from noise
  are 8–16% worse on HOT, centroid and EMD. So the Smart Init + SDEdit start is a quality component,
  not only a speed-up.

### A6. Per-step noise is what produces blue noise — FINAL

Deterministic samplers (DDIM η = 0, DPM-Solver++) lose 13–17% on HOT and 19–48% on centroid
displacement at any step count, while stochastic DDIM matches or beats DDPM. This is why fast
deterministic ODE solvers are not the answer here, a likely reviewer suggestion.

### A7. Table 2 subset — FINAL (fact for E0)

Table 2 (ablation) used the first 50 images of the Icons-50 validation manifest, a seed-42 random
order. On those 50 images Table 1's Ours-GBN has capacity 0.0268, against Table 2's SDEdit 0.0255,
so the Table 1 / Table 2 mismatch comes from the subset. Fix the caption and text in the revision.

### A8. Cross-dataset generalization (E5) — FINAL

Source: `outputs/cross_dataset_summary.md`; paired against the in-domain result on the same images,
DDPM-500, current metric code.
- **Icons → CelebA works:** δc +22% (GBN variant) / +8% (BNOT variant), HOT +31% / +18%. Ours-GBN
  trained on Icons still beats GBN itself on CelebA in HOT (10.8 vs 12.9) and EMD (0.0169 vs 0.0183).
- ShapeNet2D → Chibi: δc +66% / +51%; Chibi → ShapeNet2D: +54% / +44%. HOT stays at or below GBN.
- **CelebA → Icons fails on capacity:** δc ×5.7 / ×6.1 (0.10 / 0.12 vs 0.018 / 0.019).
- ddim-25-eta1 is slightly better than DDPM-500 in every cross-domain cell.
- Reading: transport quality transfers across domains, capacity does not; the diverse Icons set
  transfers to faces, faces do not transfer to icons. Report it as a limitation with this nuance.

### A9. Point budgets: visual only (E4 archived, user decision 2026-11-01)

- The models are trained at one budget (1024 or 1600 points) and never on others, so we show other
  budgets visually and do not evaluate them metrically.
- Fig. 4 (point-count generalization), regenerated with the exact OT assignment (A10):
  `grid_size_results/pointcount_generalization_exactOT_{ddpm-500,ddim-25-eta1}`.
- The runtime graph, updated with the new variant (the re-timing runs still need a summary).
- The metric runs are archived in `experiments/outputs/a_archive/E4_budget_generalization/`. Do not quote them.

### A10. OT-assignment bug at large budgets, found and fixed (E18) — FINAL

- From 3136 points on, POT's `ot.emd` with its default iteration limit returned a non-optimal grid
  assignment (6400 points: 33–76% above the optimal cost). Training (1024 / 1600 points) and every
  paper table are unaffected; Fig. 4's 4096-point panel and the large-budget timings are affected.
- With the exact assignment, the stray points outside the shape at 5184–6400 points mostly
  disappear (visual check). Smart Init time rises from 2.5 s to 5.9 s at 6400. An internal metric
  check found no metric worse; per A9 it is not quoted.
- Fig. 4 is already regenerated with the exact assignment
  (`grid_size_results/pointcount_generalization_exactOT_*`).
- Use only if a reviewer questions the large-budget results; otherwise just fix it in the revision.

### A11. Cost of BNOT exemplars and of training (E6 / E10) — FINAL (BNOT re-timed with the fixed build, 2026-11-18)

- BNOT per image, fixed build: 7.1 s at 1024 points, 8.5 s at 1600 (paper driver caps, old images).
  Exemplar generation at that speed ≈ 20 h (Icons), 10 h (CelebA), 9 h (ShapeNet2D), 6 h (Chibi) of
  single-image BNOT time, about the same as GBN's cost (8.1 / 8.2 s per image). With the runtimes of
  record (watch-out 7, true defaults): 6.1 / 5.8 s per image against GBN 9.5 / 9.5 s, so in the
  paper write "a few seconds per image on one CPU thread, comparable to GBN". **This contradicts the paper's
  "impractical" wording.**
- Our exemplars were generated with the slow build (13.6 / 25.5 s per image: 38 / 19 / 28 / 19 h).
  The point sets are the same (the 10 runtime images agree to < 1e-5 of a point spacing), so quote
  the fixed-build cost; the 38 h figure is only what we happened to spend.
- Training: about 31–95 h per checkpoint for 5000 epochs (upper bounds from checkpoint times).

### A12. Tone correction: what holds on the real sampler (E7, E16, D1, D3) — FINAL, with a caveat

- **The submitted Table 3 is the one-step proxy.** It scores the decode at t = 120 with the
  optimization's own fixed noise (15.76 → 16.59 dB). On the real sampler (fresh seeds 0/1/2), the
  uncorrected stipple is already 16.51 dB (DDPM-500) / 16.55 dB (DDIM-25), and that one-step field
  adds only +0.09 dB (10/24 and 11/24 images, n.s.). The one-step tone *curve* still adds +0.28 dB
  (p ≤ 4e-4).
- **E16, gradients through the full DDIM-25 chain**, scored on the real sampler with fresh seeds:
  field 16.95 dB, **+0.39 dB (18/24, p = 0.002)**; curve +0.36 dB (20/24). On DDPM-500: +0.38 dB
  (17/24). The corrected 16.95 dB is above GBN's own rendered tone (16.81 dB on the same targets).
- Cost: 0.56 s per optimizer step and 5.1 GiB (≈ 160 s per image for 300 steps), against 0.027 s
  and 2.5 GiB for the one-step path.
- **E7, direct point optimization** (μ = repulsion weight; `direct_points_fine`): μ = 0 reaches
  19.0 dB but destroys the distribution (δc 0.36, 20% of points clumped); μ = 0.02 keeps spacing
  but renders at 16.1 dB with δc 0.16 (ours: 0.049); larger μ is worse on both. No setting matches
  both our tone and our distribution. Quote δc and clumping, not ρs (the repulsion term inflates ρs).
- **D1, gradient-free iterative feedback** (the "rho <- rho + eta (rho0 - R(S(rho)))" baseline;
  multiplicative fixed point, 6 updates, 7–8 sampler calls per image, `tone_results/fixed_point/
  summary.md`): with **GBN** as the stippler +0.24 dB (19/24); with our sampler +0.15 dB (DDPM-500)
  / +0.18 dB (DDIM-25). Our full-chain gradient correction: +0.36–0.39 dB. So the gradient beats
  feedback, also feedback with a classical sampler. Claim quality, not cost: our optimization takes
  ~160 s per image, comparable to 8 GBN calls.
- **Per image on DDPM-500, fresh seeds 0/1/2** (2026-11-20, `tone_results/plots_fullchain/summary.md`,
  stage 9): uncorrected 16.51, feedback 16.66, curve 16.88, field 16.89 dB. field − curve +0.01 dB
  (11/24, p = 1); field − feedback +0.23 dB (16/24, p = 0.06); curve − feedback +0.22 dB (18/24,
  p = 0.02). The gains vary strongly between targets (−0.4 to +2.7 dB for the field). Hero image of
  the ToneMechanism figure (lg_2_woman_with_bunny_ears): curve +1.29 dB > field +0.80 dB (the
  figure does not show the curve). Image 19 (whatsapp_0_woman_doing_cartwheel_type_5) is where the
  field clearly wins (+1.92 vs +1.23) if a different example is wanted.
- **Ink-gain sweep on the real sampler** (2026-11-21, job 24362421,
  `tone_results/gain_sweep_ddim25/summary.md`; submitted size: 12 targets, 150 steps, DDIM-25
  gradient, DDPM-500 scoring, seeds 0/1/2, rendered at each gain). field − none: +0.03, +0.04,
  +0.25 (p = 0.06), +0.82 (p = 0.003), +1.36 dB (11/12, p = 0.001) at ×0.25 … ×4. field − curve:
  −0.09 (p = 0.02, curve better), −0.14, −0.17, +0.12, +0.07 dB (n.s.). The submitted sweep (one-step
  proxy) had the field ahead of the curve at every gain; that no longer holds.
- **Full tex version** now uses all of this: Table 3 and text (DDPM-500), ToneMechanism and
  ToneGraph regenerated (`figures_rebuttal/tone_*_fullchain.pdf`), supplement gain sweep
  (`figures_rebuttal/tone_gain_sweep_fullchain.pdf`). The non-full version keeps the submitted
  one-step figures.

### A13. Rendered-tone fidelity of every method (E17) — FINAL, weak

GBN has the best rendered tone on all 4 datasets; the learned variants are 0.3–1 dB below their
samplers (Icons: GBN 16.77, Ours-GBN 16.44, Ours-GBN DDIM-25 16.50 dB), and keep the samplers'
order on Icons. Use only to support rank preservation if asked.

### A14. The student beats GBN on transport metrics: paired, and not from the init — FINAL

Icons-50, 1000 test images, paired Wilcoxon against GBN with the same (current) metric code
(`rescore_current_metrics-Icons-50_1024/GBN_json`; computed 2026-11-15):
- Ours-GBN (paper points, DDPM-500): **HOT −22% (better on 64% of images, p = 8e-29), centroid
  −21% (p = 2e-28), EMD −13% (p = 4e-40)**; worse on capacity (+82%), CVT (+3%), ρs (−4%).
- DDIM-25 (new variant): HOT −22%, centroid −21%, EMD −13%, capacity +65%.
- **From pure noise, no Smart Init** (DDPM-1000): still HOT −15% (p = 1e-18), centroid −13%; with
  DDIM-25 from noise HOT −12%. So the initialization contributes about a third of the HOT gain and
  is not its cause.
- Not the KDE loss: noKDE vs full shows no difference (A15).
- Caveat: on the 50-image ablation subset the HOT difference to GBN is not significant (heavy tails);
  always quote the 1000-image test.
- Explanation to offer: the model outputs a displacement field from the OT grid and the denoiser
  regresses towards the conditional mean of the exemplars, which favors smooth, low-transport-cost
  layouts (HOT, centroid, EMD measure transport) at the expense of local regularity (capacity, CVT,
  ρs), the pattern seen in every table.

### A15. KDE-loss ablation (E8) — FINAL, null result

Epoch 5000, 50 paired images, same noise, Table 2 protocol (`ablation_nokde_e500_b50_1024/summary.md`):
no metric differs (all p > 0.08): δc 0.0232 vs 0.0264 (−12%, p = 0.64), HOT +2.7%, D +3.8%, EMD
+1.4%. The noKDE run is slower early (δc 0.033 at epochs 500–1500 vs 0.024–0.029). Say the KDE loss
speeds up early training but does not change the final checkpoint; do not claim more.

### A16. Test split and near-duplicates (W6) — FINAL

- The Icons-50 test set **is** the trainer's held-out 10% (seed-42 randperm over the sorted file
  list): reproduced 2026-11-15, identical to `validation_manifest.json`, zero overlap with the
  9000 training icons.
- But 997 of the 1000 test icons have the same emoji from another vendor or version in training
  (only 3 have an unseen concept), so the reviewer's point is factually right.
- **They do not drive the results** (`outputs/near_duplicate_check/summary.md`, job 24174333): per
  test icon, look-alike score = max Pearson correlation (128x128) with a same-concept training icon
  (median 0.74, quartiles 0.59 / 0.89), against the learned/GBN metric ratio on that icon. All
  correlations are tiny (|Spearman| <= 0.14). HOT and centroid ratios are slightly *worse* on icons
  with close look-alikes (HOT 1.06 lowest quartile vs 1.18 highest, ρ = +0.08), the opposite of
  memorization; capacity and CVT lean slightly better (δc ratio 1.94 vs 1.78, ρ = −0.08; CVT
  ρ = −0.14), a few percent at most. EMD shows nothing (ρ = +0.03, p = 0.31).
- Also cite cross-dataset (A8): trained on Icons, Ours-GBN still beats GBN on HOT and EMD on CelebA,
  where no near-duplicates exist.

### A17. Spectral anisotropy (W7) — FINAL (paper sampler and E15)

Stored spectra (uniform densities, 1024 points, 25 realizations per grey level, mean anisotropy in
dB, lower = more isotropic; `outputs/spectral_analysis/anisotropy_summary.md` once E15 finishes):

| method | low (< 0.75) | peak (0.75–1.5) | high (1.5–3) | low-freq power (r < 0.5) |
|---|---|---|---|---|
| GBN | +5.4 | −9.6 | −15.6 | 0.0016 |
| Ours-GBN (DDPM-500) | −5.5 | −8.4 | −18.0 | 0.0057 |
| Ours-GBN (DDIM-25) | −2.0 | −7.8 | −16.4 | 0.0035 |
| WVS | −11.8 | −14.2 | −20.9 | 0.0209 |
| Ours-WVS (DDPM-500) | −12.1 | −9.7 | −13.8 | 0.0114 |
| Ours-WVS (DDIM-25) | −10.0 | −8.5 | −12.4 | 0.0097 |
| BNOT | −15.6 | −15.2 | −21.2 | 0.0010 |

The reviewer is partly right: Ours-WVS is 4.5 dB more anisotropic than WVS around the peak and
7 dB at high frequencies; Ours-GBN is within ~1 dB of GBN around the peak and more isotropic
elsewhere. The new sampler (E15, job 24174335) lowers the low-frequency power (bluer: 0.0057 →
0.0035 for Ours-GBN, 0.0114 → 0.0097 for Ours-WVS, both below WVS's 0.0209) but is ~1 dB more
anisotropic around the peak. Concede the anisotropy with numbers; do not claim isotropy.

**Ours-BNOT in this analysis (added 2026-11-22, paper sampler, job 24390607), and the BNOT column
at epsilon 1.0 (user decision, same day).**
- *What was wrong:* the BNOT column of the submitted spectral figures is `target_BNOT_1024`,
  generated at epsilon 0.01 (`outputs/spectral_analysis/FINDINGS.md`), while Table 1's BNOT and
  the Ours-BNOT training exemplars use the default epsilon 1.0. Against that column Ours-BNOT
  looked as if it had not learned BNOT (spectral peak 2.28 vs 3.64, PCF peak 1.7-2.0 vs 3.8-4.0).
- *Now:* both paper figures use `target_BNOT_1024_eps1p0` (150 sources, 25 distinct realizations
  per case, 2-4 iterations) → `plots_full_bnot_eps1/`. BNOT (eps 1.0) vs Ours-BNOT at rho = 1.00:
  spectral peak 1.54 vs 2.28, peak frequency 1.14 vs 1.02, low-frequency power 0.0101 vs 0.0099,
  tail deviation 0.032 vs 0.066; quad4 PCF peak 1.68-1.75 vs 1.73-1.96, exclusion leak up to
  0.0005 vs 0.0013. Anisotropy (dB, low / peak / high band): BNOT eps 1.0 -15.96 / -22.04 /
  -25.62, Ours-BNOT -12.30 / -14.86 / -24.32. So Ours-BNOT reproduces its exemplar sampler, with
  a somewhat stronger ring, and has far less four-fold grid modulation than Ours-WVS / Ours-GBN.
- *Exposure:* BNOT at epsilon 1.0 is now the least structured spectrum in the figure, and the
  paper calls BNOT the reference on the transport metrics. If a reviewer asks why BNOT's ring is
  weak: (1) it is BNOT with its default stopping criteria, the setting used for every BNOT
  result in the paper and for the exemplars; (2) the tighter runs exist and can be reported:
  epsilon 0.01 gives peak 3.64 and low-frequency power 0.0011 (`target_BNOT_1024`, 25-iteration
  budget; converged: `target_BNOT_1024_eps0p01`), i.e. BNOT's spectrum depends strongly on the
  tolerance (epsilon sweep in FINDINGS.md section 2); (3) user's note: the same was checked for
  GBN, whose own unconstrained sampler gives a different spectrum from the image-conditioned
  one: peak 1.55 (bounded) / 1.64 (toroidal) against 2.91, tail deviation 0.057 / 0.063 against
  0.191 (`outputs/spectral_analysis_unconstrained/plots_gbn_variants/spectral_summary.csv`), so
  a low peak is not by itself a sign of a failed sampler.
- Not done: no DDIM-25 spectral run for Ours-BNOT; Table 1 and the other figures were not
  re-checked for the BNOT tolerance (CONTEXT.md, 2026-11-18, records them as epsilon 1.0).

### A18. E13: the new variant on all 4 test sets — FINAL

`outputs/e13_ddim25_summary.md` (`e13_summary.py`; ddim-25-eta1, SDEdit start, exact OT; all 12
Table 1 checkpoints; current metric code for every column). Against the paper's own point sets
(DDPM-500), paired over the full test sets: **capacity improves significantly in all 12 cases
(−1.0% to −9.2%), ρs in all 12 (+1.1% to +2.3%)**; of the 84 cells, **none is significantly worse**
in the paired sense. (One cell, ShapeNet2D Ours-WVS CVT, has a mean change of +0.1% with p < 0.05,
but its median per-image change is −0.4%: the test is paired, so it is an improvement. The summary
now prints mean / median for every cell.) The six-method table keeps the
paper's picture: Ours-BNOT is the best learned sampler on HOT, centroid, W2 and Sinkhorn on every
dataset, and every learned variant beats the GBN sampler on HOT and W2 on every dataset. With A1's
0.42 s per stipple, this answers W1 on all four datasets, not only Icons. LaTeX version (mean ± std,
supplement style): `paper_rebuttal/tables_rebuttal/quantitative_comparison_ddim25_table.tex`
(`experiments/e13_ddim25_tex_table.py`).

### A19. Generalization diagnostics: the capacity gap is there on training images too — FINAL (numbers); the last bullet is a reading

`outputs/diag_generalization/summary.md` and `timestep_loss.md` (`diag_generalization_*.py`,
2026-11-16). Ours-GBN, Icons-50, 1024 points; 200 training images (seeded random draw from the 9000)
and 200 held-out test images. Ratios are learned / the teacher's own exemplar of the same image,
averaged over images (1.00 = the teacher; ρs is higher-is-better). Distances are in grid-cell
widths (1 cell = 1/32 of the image side).

| sampler | metric | training | held-out | p (Mann-Whitney) |
|---|---|---|---|---|
| DDPM-500 | CVT | 1.035 | 1.041 | 0.049 |
| DDPM-500 | capacity δc | 1.88 | 2.03 | 0.27 |
| DDPM-500 | ρs | 0.957 | 0.959 | 0.99 |
| DDPM-500 | HOT | 1.24 | 1.13 | 0.34 |
| DDPM-500 | centroid | 1.55 | 1.42 | 0.087 |
| DDPM-500 | W2 | 0.965 | 0.941 | 0.091 |
| DDPM-500 | distance to the exemplar's points | 0.149 | 0.235 | |
| DDIM-25 | CVT | 1.029 | 1.034 | 0.0003 |
| DDIM-25 | capacity δc | 1.61 | 1.83 | 0.02 |
| DDIM-25 | ρs | 0.976 | 0.974 | 0.12 |
| DDIM-25 | HOT | 1.21 | 1.12 | 0.55 |
| DDIM-25 | centroid | 1.50 | 1.39 | 0.21 |
| DDIM-25 | W2 | 0.956 | 0.937 | 0.18 |
| DDIM-25 | distance to the exemplar's points | 0.143 | 0.234 | |

Start from the teacher's exemplar noised to step K, then DDPM from K (training / held-out):

| K | noise std (cells) | CVT | capacity δc | ρs | moved from the exemplar (cells) |
|---|---|---|---|---|---|
| 10 | 0.041 | 1.011 / 1.011 | 1.37 / 1.38 | 0.970 / 0.969 | 0.055 / 0.066 |
| 20 | 0.065 | 1.016 / 1.018 | 1.50 / 1.55 | 0.965 / 0.961 | 0.069 / 0.094 |
| 50 | 0.133 | 1.022 / 1.029 | 1.64 / 1.78 | 0.960 / 0.954 | 0.088 / 0.159 |
| 100 | 0.242 | 1.024 / 1.034 | 1.71 / 1.91 | 0.959 / 0.957 | 0.098 / 0.214 |
| 500 | 0.852 | 1.026 / 1.038 | 1.78 / 1.96 | 0.959 / 0.960 | 0.108 / 0.234 |

- **Quality does not depend on having seen the image.** With the paper sampler the only ratio that
  differs between training and held-out images is CVT (1.035 vs 1.041, p = 0.049). With DDIM-25,
  capacity is better on training images (1.61 vs 1.83, p = 0.02) and CVT by 0.5%. HOT, centroid and
  W2 lean better on the held-out images (not significant).
- **The capacity gap is not a data gap.** On images the model was trained on, δc is still 1.6–1.9×
  the teacher's (2.0× / 1.8× held-out). More training data of the same kind would not close it.
- **The model does fit its training exemplars more closely; say so if asked.** Its points land
  0.15 cells from the exemplar's points on training images against 0.23 on held-out images, and
  from the noised exemplar at K = 500 it returns to within 0.11 cells (0.23 held-out). The denoising
  error (eps MSE, the training objective) is 20–24× larger on held-out images at t = 150–500 (x0 RMS
  error at t = 300: 0.07 vs 0.34 cells) and 1.3–2.2× larger at t <= 20. This closer fit does not
  turn into better metrics (first bullet).
- **The capacity is lost in the low-noise steps.** Ten DDPM steps from the teacher's exemplar
  (noise std 0.04 cells) already give δc 1.37×, on training images too; starting from K = 500 it is
  1.78× / 1.96×. At t <= 20 the eps MSE is 0.33–0.81 on training images (predicting zero gives 1.0), and
  these steps have a Min-SNR weight of 0.002–0.02 in training (γ = 5).
- **Reading (not tested):** capacity deviation reacts to position errors of a few hundredths of a
  cell, which is the model's accuracy floor at low noise (x0 RMS error 0.02–0.04 cells at t <= 20,
  training images). A remedy would be on the training side (more weight on the low-noise steps),
  not more data. Offer it as future work only.
- Use for: W2 (capacity), W6 (generalization, near-duplicates), the capacity limitation (A4).

### A20. Gated-injection ablation (W5) — FINAL

The paper's "full" model retrained without gated injection (plain zero convolutions; everything else
identical: jointly trained, GECCO features, KDE loss, 5000 epochs), evaluated like E8: Table 2
"Full + SDEdit" settings, the same 50 validation images and per-image noise, paired against the
full checkpoint (`ablation_nokde_e500_b50_1024/summary_noGate.md`, jobs 24174334 / 24257553–660).

| metric | median per-image change without gating | worse without gating on | p (Wilcoxon) |
|---|---|---|---|
| CVT | +0.8% | 78% of images | 2e-5 |
| capacity δc | +7.7% | 82% | 5e-5 |
| ρs | −0.7% | 76% | 4e-6 |
| HOT | +0.8% | 54% | 0.40 |
| centroid D | −0.4% | 46% | 0.46 |
| power-cell CV | +0.9% | 64% | 0.14 |
| EMD (W2) | +0.5% | 56% | 0.54 |
| Sinkhorn | 0.0% | 54% | 0.79 |

- **Gated injection gives a small, consistent gain in local regularity** (CVT, capacity, ρs) and
  no measurable change in the transport metrics. This is the jointly trained ablation the reviewer
  asked for; it supports gating as a contribution, with a modest effect size.
- **Do not quote the mean capacity change.** The mean δc is 4.1% *lower* without gating (0.0253 vs
  0.0264) because of a few outlier images, while 82% of the images are worse. Quote the paired
  numbers above. (The first version of `summary_noGate.md` labelled δc "better" from the mean;
  `ablation_nokde_stage_3.py` now prints the median change, the share of images and a paired verdict.)
- 50 images, one training run per variant.

---

### A21. Seed robustness (E14) — FINAL

`outputs/e14_seeds-Icons-50_1024/summary.md`: Ours-GBN, 1000 Icons-50 test images, three independent
runs (Smart Init seeds 42 / 43 / 44, different noise), both samplers.
- **Spread of the dataset mean across runs** (std, % of the mean): CVT 0.05%, ρs 0.04%, Sinkhorn
  0.07%; capacity 1.5–1.9%; W2 2.4%; **HOT 6–7%, centroid 5%**. HOT and centroid are heavy-tailed: a
  third of the HOT difference between two runs comes from 10 of the 1000 images.
- **What holds in every run** (paired per image): DDIM-25 vs DDPM-500 capacity −10.0 / −10.4 / −10.2%
  (median), ρs +1.5%, CVT −0.5%, all p < 1e-70. The gap to GBN: HOT mean −21 / −12 / −20% (median
  per image −12 / −8 / −11%, better on 59–62% of images, p <= 1e-11), centroid −20 / −13 / −20%, W2
  −13 / −9 / −12%.
- **What does not:** DDIM-25 vs DDPM-500 in HOT, centroid and W2 (−0.4% to −1.9%, p from 4e-4 to
  0.2 depending on the run). Do not claim a transport-metric gain for the few-step sampler.
- Consequence: quote the gap to GBN as a range ("HOT 12–21% lower in three runs"), not "−22%".
- One training run per variant; this is sampling noise only.

### A22. Fig. 3 strip percentages over realizations — FINAL

`outputs/fig3_strip_realizations/summary.md` (targets 1.7 / 11.0 / 29.7 / 57.6%). The figure's
numbers are one stored realization per method. Mean ± std over 25 realizations (new Smart Init and
noise seeds), DDPM-500: **Ours-GBN 1.4 ± 0.2 / 10.7 ± 0.4 / 30.4 ± 0.4 / 57.4 ± 0.8** (largest error
of the mean 0.8 points; worst single realization 2.0); **Ours-WVS 1.2 ± 0.3 / 12.7 ± 0.7 /
30.4 ± 0.6 / 55.7 ± 1.0** (1.9 points; worst 3.7). DDIM-25 is the same within 0.2 points. Stored
single realizations: BNOT and GBN within 0.2 points, WVS 3.6.
**Ours-BNOT** (added to the figure on 2026-11-22; one stored realization, default seeds, no
realizations study): 0.5 / 11.0 / 29.8 / 58.7, largest error 1.2 points, in the sparsest strip
(0.5% where 1.7% is due, about 5 points instead of 17).
- **The paper's "within about one percentage point" (twice in the Fig. 3 paragraph, both paper
  versions) holds for Ours-GBN only.** For Ours-WVS it is about two points on average. Fix the
  sentence; Ours-WVS is still closer than WVS itself (3.6 points in its stored realization).

### A23. Model size — FINAL

`outputs/model_stats.md`: 44.1 M parameters (base denoiser 27.3 M, conditioning branch 16.7 M).
Training about 2.6 days per model on one RTX 6000-class GPU (5000 epochs; noKDE / noGate runs).

### A24. Ablation with the few-step sampler (for the supplement) — FINAL, read the caution

`outputs/ablation_ddim25_e5000_b50_1024/summary.md`: the five Table 2 variants plus noGate and
noKDE at epoch 5000, 50 Table 2 images, SDEdit start for every row, same Smart Init and noise.
- Against Full, paired (DDIM-25): Vanilla, Unfrozen, GECCO-only and Gated-only are significantly
  worse on capacity (+51% to +179% median) and on CVT (+0.9% to +6.9%); the single components help
  far less than their combination, as in Table 2.
- **Second evaluation of the gating ablation (independent of A20's noise):** without gating, CVT
  +0.3% (DDIM-25) / +0.4% (DDPM-500) and capacity +4.5% / +5.9% median, both significantly worse;
  ρs −0.2% (worse, DDPM-500 only); transport metrics no difference. Same direction as A20 (CVT
  +0.8%, capacity +7.7%, ρs −0.7%), smaller size. Quote the range and call the effect small.
- noKDE: no difference on any metric, with both samplers (as A15).
- **Caution for any table of means:** Full does not have the best mean capacity among these rows
  (0.0266 against 0.0230 without gating and 0.0224 without the KDE loss, DDIM-25), because a few
  outlier images raise its mean, while per image it is better than noGate on 64% of images. A means
  table that adds the noGate / noKDE rows will show Full not in bold for capacity; give the paired
  columns next to it or keep those two rows in text.

### A25. Low-noise fine-tune, final trend — FINAL (model study)

`outputs/diag_generalization/ft_lownoise_trend_g0.md`: no Min-SNR, epochs 5100–6000, 200 held-out
images, paired vs the paper checkpoint, DDPM-500. Capacity −5.5% (5100), −15.0% (5200), −12.8%
(5300), −9.3% (5600), −18.1% (6000), always significant (better on 69–92% of images); ρs +1.5% to
+1.8%; CVT −0.4% to −0.8%; transport metrics unchanged. The gain appears within 200 epochs and
then fluctuates between checkpoints; it does not keep growing. Capacity stays 1.5–1.6× the
teacher's. One optional sentence in the rebuttal at most.

## 2. Draft answers to the concerns we expect

Each answer quotes a number from section 1 and commits a change to the revision.

**"Too slow: 500 diffusion steps."** With stochastic DDIM, 25 steps match or improve all seven
metrics against our 500-step sampler on 1000 Icons-50 images (capacity −8%, ρs +1.5%, p < 0.05),
and cut the time per stipple from 5.5 s to 0.42 s at 1024 points on the same GPU (median over 10
images, watch-out 7); batched, the sampler produces 60 stipples/s. *We will present the 25-step
sampler in the supplement, with its table and runtime graph.* (A1, A1b)

**"BNOT dominates; why not train on BNOT?"** We did: Ours-BNOT beats GBN and WVS on HOT, centroid
displacement and EMD on all four datasets and comes close to BNOT (A3), at a fixed cost per
stipple. *We will add Ours-BNOT to Table 1 and correct the statement that BNOT exemplars are
impractical: they cost about the same as GBN's (A11).*

**"Is it really example-based?"** Where the exemplar samplers differ significantly, the learned
variants reproduce their order in 78 of 82 cases (paired tests over 4 datasets × 7 metrics; Kendall
τ 0.90); the exceptions are near-ties between BNOT and GBN on capacity (A2). *We will add this
statistic.*

**"The model never beats its teacher."** By design: the contribution is amortization and
differentiability, not exceeding a per-target optimizer. At the training budget, where the
teacher does not optimize the transport energy (GBN, WVS), the learned variant improves on it in
HOT, D and EMD; where it does (BNOT), the variant approaches it (A3).

**"Worse capacity on the non-square datasets."** The gap is the same for all three samplers, so it
comes from the model (A4). It is not the aspect ratio: on the square images of those datasets the
gap remains (A4). It tracks the point budget and dataset; *we will state this as a limitation.*

**"Point budgets are only shown qualitatively."** The model is trained at a single budget, and
Fig. 4 shows how it carries over to other budgets (A9). *We will regenerate Fig. 4 with an exact OT
assignment (A10) and update the runtime graph with the new sampler.*

**"Does it generalize to new image domains?"** A8: Icons → CelebA keeps transport quality and loses
22% capacity; CelebA → Icons fails on capacity. *We will report this in the supplement.*

**"Tone correction uses a one-step gradient / small gains / why not optimize the points directly?"**
With gradients through the full 25-step sampler, the correction improves the output of the paper's
500-step sampler by +0.38 dB (17/24 targets, p = 0.004), against +0.15 dB for gradient-free
feedback (A12). A global tone curve optimized the same way reaches the same PSNR; the field gives
the lowest MAE. The gain grows with the dot gain (+1.36 dB at ×4). Optimizing the points
directly reaches higher PSNR only by breaking the distribution (δc 0.36 vs 0.05); with enough
repulsion to keep the spacing, it renders worse than our corrected sampler (A12). *We will replace
Table 3, the two tone figures and the gain sweep with the full-chain results on the real sampler
and describe the protocol exactly.*

**"Why t = 500? Why diffusion rather than the prior alone?"** A5.

**"Why not a fast deterministic solver?"** A6.

**"Table 1 and Table 2 disagree."** A7: *we will state the 50-image subset in the caption.*

---

## 3. Paper changes to make in the revision (collect here; apply to the `_rebuttal_11.10` files)

**Applied 2026-11-20 (user decision: two versions, all in `paper_rebuttal/`, compiled without errors):**
- `*_rebuttal_11.10.tex` (non-full): the submitted text and numbers unchanged; new results only. A
  supplement section "Faster Sampling" (DDIM-25, fast OT assignment, six-method DDIM-25 table, the
  submitted Fig. 7 curves + DDIM-25 in a new graph) and one pointer sentence in the main runtime
  paragraph.
- `*_rebuttal_full_11.10.tex` (full): all of the non-full version plus every fix: items 2, 3, 4, 5,
  6 and 8 below (Fig. 7 re-timed, watch-out 7; Fig. 4 exact OT; BNOT exemplar / checkpoint /
  abstract wording; A2 statistic; A4 capacity paragraph; ablation caption; tone Table 3 and text on
  the real sampler; conclusion). Item 7 (cross-dataset) is not in either version. ~~Still open in
  the full version: the tone figures and the ink-gain sweep show the one-step proxy.~~ **Done
  2026-11-21:** ToneMechanism, ToneGraph and the supplement's gain sweep regenerated on the real
  sampler (A12); no tone TODOs remain in the full version.
- Originals backed up in `paper_rebuttal/archive/rebuttal_bak_20261120/`; new figures in
  `paper_rebuttal/figures_rebuttal/`, tables in `paper_rebuttal/tables_rebuttal/`.

1. The few-step sampler (`ddim-25-eta1`): in the paper, or in the supplement with E13's full table;
   updated Fig. 7 and runtime text (A1, E13).
2. Ours-BNOT in Table 1 and the corrected BNOT-exemplar statement (A3, A11).
3. The rank-preservation statistic (A2).
4. Table 2 caption and text: the 50-image subset (A7).
5. **Table 3 and the tone text: E16 numbers on the real sampler, exact protocol, "17 of 24"
   (DDPM-500) instead of "all 24"; the tone curve ties the field; figures and gain sweep
   regenerated (A12).** Done in the full version.
6. Fig. 4 with the exact OT assignment (A10); runtime graph with the new variant (A9).
   **Fig. 7's BNOT curve from the fixed build**, and every sentence built on the old curve
   ("1802 s at 6400 points", "grows by more than 100x", BNOT slower than ours at every budget):
   see watch-out 6. State the correction in the rebuttal (REBUTTAL_TEXT.md Part 2, Q1).
7. Cross-dataset results in the supplement (A8).
8. Capacity-gap limitation: not the aspect ratio; budget or domain (A4). It is also there on
   training images and arises in the low-noise steps (A19).

---

## 4. Status of the experiments

| ID | What | Status |
|---|---|---|
| E0 | Numbers audit, Table 2 subset | fact found (A7); text fix pending (paper edits on hold) |
| E1 | Few-step sampling | **done** (A1, A1b) |
| E2 | Rank preservation, Wilcoxon | **done** (A2) |
| E3 | Aspect ratio vs capacity | **done, negative** (A4); steps 2–3 not run |
| E4 | Quantitative budget generalization | **archived** (user decision; budgets visual only, A9) |
| E5 | Cross-dataset generalization | **done** (A8) |
| E6 / E10 | BNOT exemplar and training cost | **done** (A11) |
| E7 | Direct point-optimization baseline (tone) | **done** (A12) |
| E8 | KDE-loss ablation | **done, null**: no metric differs at epoch 5000 (A15) |
| E13 | Four-dataset table with ddim-25-eta1 | **done** (A18): `outputs/e13_ddim25_summary.md` |
| Ablation, new sampler | Table 2 variants (+ noGate, noKDE) with ddim-25-eta1 for the supplement | **done** (A24): `outputs/ablation_ddim25_e5000_b50_1024/summary.md` |
| E14 | Seed robustness, 3 independent runs (W5) | **done** (A21): `outputs/e14_seeds-Icons-50_1024/summary.md` |
| Fig. 3 strips | Strip percentages over 25 realizations; parameter counts | **done** (A22, A23) |
| E15 | Spectra / anisotropy with the new sampler | **done** (A17) |
| Gating | Joint GECCO without gated injection (W5) | **done** (A20): gating helps CVT, capacity and ρs slightly but consistently |
| Near-dup | Same-concept look-alikes vs quality (W6) | **done**: no memorization effect (A16) |
| Gen. diag | Training vs held-out images; start from the exemplar; loss per noise level | **done** (A19): `outputs/diag_generalization/` |
| Pure noise | DDIM / DDPM from pure noise, no Smart Init | **done** (A5) |
| Fast OT | Fast Smart Init assignment | **done**: only `bisection-blocks16` passes; used above 2304 points (A1) |
| E16 | Tone through the full sampler | **done** (A12) |
| E17 | Rendered-tone fidelity | **done** (A13) |
| E18 | OT-assignment bug | **found and fixed** (A10) |
| D1 | Gradient-free fixed point (tone) | **done** (A12) |
| Runtime, new variant | clean 3-rep re-timing | **superseded** by runtime_v10 (watch-out 7) |
| Runtime, all methods | runtime_v10: all four methods re-timed on 10 new images, BNOT true defaults | **done 2026-11-20** (watch-out 7): `outputs/runtime_v10/` |

---

## 5. Mock review R1 (received 2026-11-15, "borderline, leaning reject, 4/10") — answers

The full review text is in `experiments/reviews/mock_review_R1.md`. Strength of our answer:
**strong** (finished evidence), **partial**, **none** (concede or a job is running).

| Point | Strength | Answer (evidence) |
|---|---|---|
| W1 speed "≈ WVS at 1024–1600" | strong | 25-step stochastic DDIM: 0.42 s at 1024, 0.60 s at 1600 incl. exact OT Smart Init, vs WVS 6.8 / 11.0, GBN 9.5 / 9.5 (GPU), BNOT 6.1 / 5.8 s (true defaults; medians, watch-out 7); equal or better on all 7 metrics than our DDPM-500 (A1, A1b). 60 stipples/s batched. |
| W1 "unoptimized WVS" | partial | It is Rougier's NumPy/SciPy reference implementation; say so. Even a 10x faster WVS only ties us at 1024 and grows with N (50.8 s at 6400 vs our 1.33 s). All methods were re-timed on one protocol (watch-out 7); WVS is still the reference implementation. |
| W1 tiles / lattices | none | Related-work paragraph: not example-based, not differentiable. |
| W1 mottling at 4096 | partial | The large-budget panels used a truncated OT solver (A10); Fig. 4 is regenerated with the exact assignment. Budgets stay visual only (A9). |
| W2 BNOT not trained | strong | Ours-BNOT on all 4 datasets (A3); exemplar cost ~20 h for Icons with the fixed BNOT build, about the same as GBN's (A11): agree with the reviewer and correct the "impractical" sentence. |
| W2 title "capacity-constrained" | weak | Capacity stays 1.7–3.6x GBN's even for Ours-BNOT (A4). Offer a retitle (e.g. "capacity-aware") or define the term as "trained on capacity-constrained exemplars". The gap is a model limit, not a data gap: 1.6–1.9x on training images too (A19). |
| W3 one-step gradient, Q5 | moderate, with a correction | Q5: no, Table 3 was the one-step proxy; replace it by the full-chain result on the 500-step sampler: +0.38 dB (17/24, p = 0.004). The tone curve optimized the same way ties it (A12); say so. |
| W3 iterative feedback baseline | strong | We ran it (D1): with GBN +0.24 dB, with our sampler +0.15/+0.18 dB, vs our gradient +0.38 dB on DDPM-500 (field − feedback +0.23 dB, 16/24). Direct point optimization only wins PSNR by clumping (A12, E7). |
| W4 / Q3 student beats teacher | strong | Paired tests on 1000 images in three independent runs: HOT 12–21% lower (p <= 1e-11 in each), centroid 13–20%, W2 9–13%; still lower from pure noise; not the KDE loss (A14, A15, A21). Quote the range, not −22%. |
| W4 Sinkhorn uninformative | concede | Move to the supplement. |
| W5 steps / sampler / truncation | strong | Full sweep (A1b, A5, A6); pure noise is 8–16% worse (A5). |
| W5 KDE loss | strong (null) | E8: no effect at epoch 5000, faster early training (A15). |
| W5 gating without the rest | strong | We trained it: the full model without gated injection is worse on CVT (+0.8%, 78% of images), capacity (median +7.7%, 82%) and ρs (−0.7%, 76%), all p < 1e-4 paired; transport metrics unchanged (A20). Effect is modest; say so. |
| W5 single runs / noise | partial | E14 (A21): dataset means vary < 0.1% (CVT, ρs), ~2% (capacity, W2), 5–7% (HOT, centroid) between runs; the capacity and ρs gains of the few-step sampler and the gap to GBN hold in every run. One training run per variant: concede. |
| W6 cross-dataset | strong | Icons→CelebA keeps transport quality and beats GBN on HOT/EMD; CelebA→Icons fails on capacity, stated as a limitation (A8). |
| W6 near-duplicates in the split | strong | True that 997/1000 test icons have a same-concept training icon, but the learned/teacher ratio does not improve with look-alike similarity (all \|ρ\| <= 0.14; HOT slightly worse on close look-alikes) (A16). The learned/teacher ratios are also the same on training and held-out images (A19). |
| W6 perfect-square n | concede | Limitation of the OT-grid representation. |
| W7 anisotropy | partial / concede | A17: Ours-WVS more anisotropic than WVS around the peak, Ours-GBN within ~1 dB of GBN; the new sampler is bluer at low frequency but ~1 dB more anisotropic. |
| W8 novelty | argument | Ours-BNOT, 78/82 paired rank preservation (A2), full-chain differentiability (A12), the SDEdit start as a quality component (A5). |
| Q1 implementations | facts | WVS: Rougier NumPy/SciPy, CPU. BNOT: the authors' C++ code in our headless port, CPU, 1 thread; the port's linear solver was slow and is fixed (with BNOT's true defaults 6.1 s at 1024, 12.8 s at 6400 instead of 13.6 / 1802 s; same point sets; watch-outs 6 and 7), and Fig. 7 is re-timed for all methods. **GBN: authors' CUDA binary on an RTX-class GPU.** Our time includes the Smart Init OT (exact network simplex, 0.14 s at 1024, 6.4 s at 6400; 1.07 s at 6400 with bisection-blocks16). |
| Q2 aspect ratio | strong | Not given to the network (condition resized to the grid, per-axis normalization), but not the cause: on already-square images the gap remains 2–4x (A4, E3). |
| Q4 50–100 steps, deterministic | strong | 50 / 100 steps also beat DDPM-500 (δc −6.4% / −4.4%); deterministic solvers lose 13–17% HOT (A1b, A6). |
| Minor: EMD is W1 | fact | Our M3 is the 2-Wasserstein distance (sqrt of the OT cost with squared distances, `compute_m3_emd`); rename it "W2" in the revision. |
| Minor: Fig. 3 strip % | fact + fix | One realization per method. Over 25 realizations Ours-GBN is within 0.8 points on average, Ours-WVS within 1.9 (A22); the paper's "within about one percentage point" must be corrected for Ours-WVS. |
| Minor: refs, Fig. 8 size, training time | concede | Add the four references; enlarge Fig. 8; training about 2.6 days per model, 44.1 M parameters (A23), code release. |

**Raise-the-score list:** BNOT variant (have it), quality-vs-time over step counts (have it; WVS
stays the reference implementation), 4K–10K quantitative (decline: visual only, models trained at
one budget), tone vs iterative feedback with a classical sampler (have it, D1).

---

## 6. What is still open before the reviews (written 2026-11-18; reviews Nov 23, rebuttal due Nov 30)

Every planned experiment is done or running. What remains is mostly decisions and writing.

**Running (results land without action):**
- ~~E14 seed robustness, Fig. 3 strip averages, parameter counts, ablation with the few-step sampler~~ all done 2026-11-21 (A21–A24).
- The no-Min-SNR fine-tune to epoch 6000 (model study; at most one sentence in the rebuttal). Early
  trend: the gain is complete by epoch 5200 (capacity −15% at 5200, −13% at 5300; ρs +1.5% from 5100).
- ~~BNOT runtime investigation~~ **done 2026-11-18** (watch-out 6): fixed build, `runtimes_avg.json`
  and the three runtime graphs rebuilt and checked; this file and REBUTTAL_TEXT.md carry the new
  numbers. Still open: the paper's Fig. 7 and its text, and REBUTTAL_PLAN.md's "two orders of
  magnitude" sentence.

**Decisions only the user can make:**
1. ~~Default sampler in the revision~~ **decided 2026-11-18:** DDIM-25 goes into the supplement as
   a faster sampling variant, with its curve and a paragraph in the paper's runtime section; the
   paper's tables keep DDPM-500.
2. The title: keep "Capacity-Constrained", or retitle / define the term (W2 is our weakest answer).
3. ~~When to start editing the paper files~~ **started 2026-11-20** (section 3, "Applied").
4. Whether to mention the low-noise fine-tune result at all.

**Writing that can be prepared now (no compute):**
- The one-page rebuttal text from section 2 and section 5 (PLAN section 4 has the skeleton); the real
  reviews will reorder it, but the numbered answers can be drafted.
- Related-work paragraph on tile / lattice methods and the four missing references (W1, minor).
- ~~Replacement Table 3~~ (done 2026-11-20/21 with its two figures and the gain sweep, A12), six-column Table 1 with Ours-BNOT (A3 / A18),
  Table 2 caption (50-image subset, A7), the gating and KDE rows (A15, A20).
- ~~Updated runtime figure~~ **done 2026-11-20:** all curves re-timed (runtime_v10), median, blocks16
  from 3136 points (watch-out 7); in `paper_rebuttal/figures_rebuttal/` and the full tex version.
- Limitations paragraph: capacity gap (A4, A19), perfect-square n, anisotropy (A17), CelebA→Icons.

**Answers that stay weak (prepare the concession, do not spend compute):**
- "Unoptimized WVS baseline" (reference NumPy implementation; not re-timed).
- No comparison with tile / lattice samplers.
- No quantitative results at 4K–10K points (models trained at one budget; Fig. 4 and runtime only).
- Training-seed variance: one training run per variant (E14 covers sampling noise only).
- Break-even, if asked: exemplars + training ≈ 80 h per model against ≈ 9.1 s saved per stipple vs
  GBN (9.5 s against 0.42 s at 1024 points, watch-out 7), i.e. tens of thousands of stipples. Do not volunteer it; the case rests on amortized use,
  batching (60 stipples/s) and differentiability.

<!-- bnot-times-fixed-20261118: BNOT runtimes replaced by the fixed-build measurements -->
