# StippleDiffusion (EG 2027, paper1182) — Rebuttal Plan

Hand this file to Claude Code from the repo root. Section 0 is context; Section 2 is the experiment queue in priority order.

---

## 0. Context and constraints

### EG 2027 timeline (from the official CfP)

| Date | Milestone |
|---|---|
| **Nov 23, 2026** | Reviews released |
| **Nov 30, 2026** | Rebuttal due (one week) |
| Dec 18, 2026 | Conditional acceptance / rejection |
| Jan 26, 2027 | Revised version due (second review round) |
| Feb 9, 2027 | Final notification |
| Feb 23, 2027 | Camera-ready |

**Implication: run all experiments now, before Nov 23.** The rebuttal week should be spent choosing results and writing, not computing.

### What can change, and where

- **Rebuttal.** The CfP calls it a "brief document to address any factual errors or clarify any issues raised by reviews."
  - Keep new numbers to small, targeted tables, and only for issues a reviewer actually raised.
  - Each new number should come with a sentence like "we will add X to Table Y."
- **Revised version (Jan 26).** This is a real second round, not just a camera-ready.
  - The main paper can change, including tables, figures and claims. The primary reviewer checks that the required changes from the conditional acceptance were made.
  - Changes the rebuttal promised are effectively pre-approved. Large unrequested changes (a new method, new headline claims) are risky.
- **Title and abstract.** Minor edits are explicitly allowed.
- **Author list.** Frozen.

**Strategy.** Use the rebuttal to *promise* main-paper changes, such as "Ours-BNOT becomes a column of Table 1" or "the 'impractical' claim on l.207–211 is corrected." Then deliver them in the January revision.

---

## 1. Key messages for the rebuttal

1. **The model really is example-based (new evidence).** With three teachers (BNOT, WVS, GBN), the learned variants preserve the teachers' pairwise ordering in **73 of 74** non-tied teacher pairs across 4 datasets × 7 metrics. The single flip is CVT BNOT-vs-WVS on ShapeNet2D. *Already computed from the rebuttal supplement table; recompute in the repo to confirm.*
2. **Ours-BNOT is the strongest learned sampler.**
   - It comes close to BNOT on HOT, centroid and EMD. For example, on Icons HOT is 2.85 vs 2.42, centroid 0.003 vs 0.001 and EMD 0.011 vs 0.010.
   - It beats both GBN and WVS on those metrics on all 4 datasets.
   - Its inference cost is fixed, while BNOT's cost grows by about two orders of magnitude over the budget range.
3. **The contribution is amortization plus differentiability, not beating the teacher.** BNOT and GBN are quality upper bounds. Reframe the claims around this.
4. **The capacity gap comes from the model, not the teacher.**
   - On each dataset, all three learned variants have nearly the same capacity deviation.
   - On Anime Chibi it is 0.110 / 0.129 / 0.107, while the teachers are at 0.040 / 0.061 / 0.030.
   - So the gap comes from the representation (aspect ratio, OT grid). Experiment E3 tests that directly.
5. **Learned variants beat GBN and WVS on HOT, centroid and EMD, but never beat BNOT.** This fits an inductive bias toward transport-optimal layouts (from the OT grid and the KDE loss): it helps when the teacher doesn't optimize HOT and can't exceed a teacher that does. Say this before a reviewer asks why a student beats its teacher.

---

## 2. Experiment queue (do in order)

Each item lists: **why** (the likely reviewer concern), **how**, **output**, and **what to say** in each outcome.

### P0 — no retraining, highest value

#### E0. Numbers audit and a single source of truth (do first)

- **Why.** The submission has visible inconsistencies:
  - **Table 2 vs Table 1 (cause known).** The ablation was evaluated on a *partial* Icons-50 test subset, not the full 1000 images. So Table 2's SDEdit column (HOT 4.978, capacity 0.0255) doesn't match Table 1's Ours-GBN (HOT 4.059, capacity 0.018), and the caption doesn't say a subset was used.
  - The rebuttal supplement differs from the submission in the third decimal: WVS CVT 2.193 vs 2.194, Ours-GBN CVT 2.449 vs 2.450, BNOT ρs 0.525 vs 0.526.
- **How. Keep the Table 2 results as they are; fix only the text.**
  1. **Record the subset details:** its size k and how it was drawn (random with seed s, or first k images).
  2. **Table 2 caption.** Add something like: "Evaluated on a fixed [k]-image subset of the Icons-50 test set (random, seed s); absolute values therefore differ from Table 1, which uses all 1000 test images."
  3. **Ablation text (l.476).** One clause: "...on a [k]-image subset of the Icons-50 test set."
  4. **Abstract (the "0.063 to 0.026" sentence) and contribution bullet (l.50–53).** These quote Table 2. Add "in the ablation" / "on the ablation subset," so a reader doesn't compare them with Table 1's 0.018.
  5. **Optional, cheap.** Add ± std to Table 2 from the existing per-image results, as already done for Table 1 in the supplement.
  6. Fix the third-decimal mismatches between the submission and the rebuttal supplement: rebuild both tables from the same saved point sets and rounding.
- **Output.** Edited caption, l.476, abstract and l.50–53 text; subset size and seed noted.
- **Say (only if a reviewer notices the mismatch, otherwise just fix it in the revision).** "Table 2 was evaluated on a fixed [k]-image subset of the Icons-50 test set, which explains the difference from Table 1; we will state this in the caption and text."
- **Have one sentence ready** in case a reviewer asks why not the full set. For example: it was a fixed subset chosen to keep the cost manageable across six variants, and the comparison is paired across variants on the same images.

#### E1. Few-step sampling (the most important speed result)

- **Why.** 500 DDPM steps at about 6 s for 1024 points is close to WVS on a CPU. Reviewers will ask about DDIM or distillation, and about real-time GPU stipplers.
- **How.**
  - Use the existing Ours-GBN and Ours-BNOT checkpoints with the same SDEdit start (t = 500 from the rejection-sampled prior).
  - Compare samplers: DDPM-500 (baseline), DDIM (η = 0) with 10 / 25 / 50 / 100 steps, and DPM-Solver++(2M) with 10 / 20 steps (diffusers schedulers work with a custom eps-model wrapper).
  - Data: the full Icons-50 test set (1000 images) at 1024 points. Also time 4096 and 6400 points.
  - Report per-image wall time at batch 1. Also report **throughput at batch 32/64** (stipples per second), which is the real amortization story.
  - Optionally try `torch.compile`, CUDA graphs and fp16/bf16, since the supplement shows wall time dominated by kernel-launch overhead.
- **Output.** A table of steps × {all 7 metrics, time/image, throughput}, and an updated Fig. 7 with a "Ours (DDIM-25)" curve.
- **Say.**
  - If DDIM-25 is within about 5% on the metrics: "Our sampler runs in X s (Y stipples/s batched), Zx faster than WVS/GBN/BNOT."
  - If quality drops: report the quality/speed tradeoff honestly and mention distillation as future work.

#### E2. Rank-preservation statistic

- **Why.** This is the quantitative backing for "example-based."
- **How.**
  - Over 4 datasets × 7 metrics × 3 teacher pairs, count pairs where sign(teacher_i − teacher_j) equals sign(learned_i − learned_j).
  - Ties: values equal after rounding to the table's precision. Better, use per-image paired tests (Wilcoxon signed-rank) instead of rounded means.
  - Report the count and Kendall's τ.
- **Output.** One sentence for the rebuttal, and one paragraph plus a small table for the revision.

#### E3. Aspect-ratio confound (fills the gap admitted on l.426)

- **Why.** Per-axis normalization to [0,1]² makes the layout anisotropic in image space for non-square images. That may explain the capacity drop on ShapeNet2D and Anime Chibi.
- **How, in three steps of increasing cost:**
  1. **Free.** On the existing outputs, scatter per-image capacity δc (and spatial ρs) against aspect ratio for ShapeNet2D and Chibi. Report the Spearman correlation. Also check the BNOT outliers on Chibi (capacity 0.040 ± 0.104).
  2. **Cheap.** Pad the Chibi and ShapeNet2D test images to square with white (zero density), keeping the same point budget. Run the existing 1600-point checkpoints. Evaluate inside the original image rectangle, and run the teachers under the same padding.
  3. **Cheap, and the best fix.** Use an *aspect-matched rectangular OT grid* at inference, e.g. 32×50 for a 1:1.56 image. Normalize coordinates isotropically so cells stay square. The U-Net is fully convolutional, so this may need no retraining. Check that grid dimensions are divisible by 2^(number of downsampling levels). This also partly fixes the perfect-square-only limitation (n = N·M).
- **Output.** A table for Chibi and ShapeNet2D: {original, padded, rect-grid} × {capacity, HOT, ρs}.
- **Say.**
  - If capacity recovers: "The gap on non-square datasets comes from per-axis normalization; an aspect-matched grid reduces δc from X to Y without retraining."
  - If it doesn't: "Aspect ratio is not the cause; [budget / training-set size] is."

#### E4. Quantitative budget generalization — ARCHIVED (user decision, 2026-11-01)

- Not followed. The models are trained at one budget and never on others, so other budgets are
  shown visually only: Fig. 4 (regenerated with the exact OT assignment, E18) and the runtime graph
  (updated with the new variant). No budget metrics in the paper or the rebuttal.
- The runs that were made are archived in `experiments/outputs/a_archive/E4_budget_generalization/` (scripts: `experiments/archive/E4_budget_generalization_scripts/`).

#### E5. Cross-dataset generalization

- **How.**
  - Icons → CelebA and CelebA → Icons at 1024 points.
  - ShapeNet2D → Chibi and Chibi → ShapeNet2D at 1600 points.
  - Use Ours-GBN and Ours-BNOT, compared with the in-domain checkpoint and the teacher.
- **Output.** A small table: in-domain vs. cross-domain vs. teacher.
- **Say.** "The model generalizes across image domains with a drop of X," or name the limitation honestly.

### P1 — moderate cost

#### E6. Ours-BNOT documentation (results already exist)

- Record the BNOT exemplar generation cost: CPU-hours and per-image time at 1024 and 1600 points.
- This **contradicts the l.207–211 claim that BNOT is impractical for exemplars**. The rebuttal must correct it before a reviewer notices.
- **Decide the ranking convention for the revised Table 1:** either rank all six columns, or rank the learned variants only with the teachers as references. Recommended: rank all six. Ours-BNOT then wins many second places, which reads well and isn't cherry-picked.
- Fix the supplement caption wording "not trained on BNOT exemplars due to its performance," which reads as a quality judgment.

#### E7. Direct point-optimization baseline for tone correction

- **Why.** Reviewers will ask why you differentiate through the sampler instead of optimizing the points directly.
- **How.**
  - Start from the GBN and BNOT outputs on the same 24 Icons targets. Minimize ‖R(points) − ρ₀‖² + μ·E_GBN(points) with Adam, using the same 300 steps.
  - Sweep μ over about 4 values.
  - Report PSNR, SSIM and MAE **plus** the blue-noise metrics (ρs, capacity, spectrum) of the corrected point sets.
- **Expected story.** Direct optimization can reach high PSNR but degrades the point distribution (clustering, worse ρs), while optimizing ρ keeps the outputs on the sampler's distribution. If direct optimization wins on both counts, reframe the tone correction as a capability demonstration, not a state-of-the-art claim.
- **Output.** Extra rows in Table 3 with distribution-quality columns.

#### E8. KDE-loss ablation

- Ours-GBN on Icons with λ_KDE = 0 vs. 0.8. If 5000 epochs is too long, compare both at the same shorter schedule and say so.
- One more row in Table 2.

### P2 — only if time permits or a reviewer asks

- **E9. Fast-baseline runtime.** Time one GPU Lloyd/WVS implementation (e.g., jump-flooding Voronoi) and one tile-based importance sampler, if a public implementation exists. Otherwise add a related-work paragraph saying these are faster but are not capacity-constrained and are not differentiable from the density.
- **E10. Training cost and break-even.** GPU-hours per checkpoint plus exemplar generation, divided by the per-stipple saving against each teacher, gives the number of stipples at which training pays off.
- **E11. Video for the camera-ready.** Denoising animation, budget sweep, tone-correction loop.
- **E12. Related work to add.**
  - Learned halftoning/dithering.
  - Weighted Linde-Buzo-Gray stippling (Deussen et al. 2017).
  - The "stippling by example" line (Kim et al. 2009; Martín et al.). This one matters because of the shared "example-based" term.
  - Verify the exact bibliographic entries before citing.

---

## 3. Reviewer concerns → which experiment answers them

| Likely concern | Answer |
|---|---|
| Incremental: ControlNet + GECCO + SDEdit | Msg 1–3, E2, E3 (rect grid), ablations E8 |
| Doesn't beat its teacher; title says capacity-constrained | Msg 3–4, E3, E6; optional title tweak |
| Not fast; 500 steps; GPU vs CPU | E1, E9, E10 |
| BNOT dominates, and why not train on BNOT? | Ours-BNOT (E6), Msg 2, Msg 5 |
| Non-square datasets worse; confound admitted | E3 |
| Budget generalization only qualitative | Fig. 4 (exact OT) + runtime graph; E4 archived |
| Generalization to unseen domains | E5 |
| Tone correction: weak baseline, small gains | E7 + existing ink-gain sweep (Supp. Fig. 1) |
| Table inconsistencies | E0 |
| Perfect-square n, grid artifacts | E3 step 3 (rectangular grid) |
| Missing related work | E12 |

---

## 4. Rebuttal document skeleton (fill in after Nov 23)

Aim for about one page. Open with what is common to all reviews, then go reviewer by reviewer. Lead with whatever moves the most negative but persuadable reviewer.

```
We thank the reviewers for ... [one line naming what they liked: R1 "...", R3 "..."]

**Common concerns**
(C1) Positioning/novelty: [Msg 3 + Msg 1 rank preservation 73/74]
(C2) Ours-BNOT: [compact 4-row table: dataset × {BNOT, GBN, Ours-BNOT} for HOT/δc/D/EMD]
     Correction: l.207-211 overstated BNOT cost; generating 10K exemplars took X CPU-h.
(C3) Runtime: [E1 table: DDPM-500 vs DDIM-25 time + δc/HOT]
(C4) Capacity on non-square data: [E3 result]

**R1** ...
**R2** ...

**Committed revisions:** (1) Ours-BNOT column in Table 1; (2) few-step sampling + updated Fig. 7;
(3) cross-dataset results in supp., Fig. 4 with exact OT; (4) corrected l.207-211 and abstract
wording on capacity; (5) related work: [list]; (6) code + checkpoints release.
```

Writing rules:

- Quote the reviewer's concern in a few words, then answer with a number.
- Never argue with a factual criticism you can fix; say "we will fix it."
- Put every new number next to the sentence that commits it to the revision.

---

## 5. Suggested calendar

| Dates | Work |
|---|---|
| Oct 27 – Nov 2 | E0, E2, E3 step 1, start E1 |
| Nov 3 – Nov 9 | E1 done, E3 steps 2–3 |
| Nov 10 – Nov 16 | E5, E6, E7 |
| Nov 17 – Nov 22 | E8, buffer, prepare result tables and figure snippets |
| Nov 23 – Nov 30 | Read reviews, pick results, write rebuttal |
| Dec 18 → Jan 26 | Revision: apply all committed changes to main paper + supp. |


---

## 6. Additional ideas (added 2026-10-28)

These build on the E1 result: with the 10-step sampler, every sampling experiment is about 50×
cheaper, so several items that were "P2" before are now cheap. Ablation-style items run on Ours-GBN
only; comparisons run on all three learned variants (the project's protocol).

#### E13. Full re-run of Table 1 with the chosen few-step sampler (P0, requested by the user)

- **Why.** If the paper switches to the 10-step sampler, every table must come from it, and a
  complete, consistent table is the strongest single piece of evidence.
- **How.** After the variance check (E1, A1b in REBUTTAL_DRAFT.md) fixes the sampler: run
  Ours-WVS, Ours-GBN and Ours-BNOT on all 4 test sets (1000 / 500 / 400 / 263 images, 6489 stipples),
  with `few_step_sampling_stage_1.py` (it takes `--source`, `--grid_size`, `--label`), score with
  stage 2, and re-score the samplers' stored point sets with the **same metric code** (CONTEXT.md:
  never mix fresh metrics with the stored Table 1 JSONs). All 12 epoch-5000 checkpoints exist.
- **Cost.** Sampling: minutes per dataset. Scoring: about 36 CPU-hours, a few hours as a sharded array.
- **Output.** New Table 1 (6 columns, Ours-BNOT included) and the supplement's full table with std.

#### E14. Seed robustness (P1)

- **Why.** The sampler is stochastic; a reviewer may ask how much results vary between runs.
- **How.** Ours-GBN, Icons-50, the chosen sampler, 3 noise seeds. Report the std of the means
  across seeds next to the differences between methods.

#### E15. Spectra and pair correlations with the new sampler (P0 if the default changes)

- **Why.** Fig. 6 and the supplement's pair correlations were made with DDPM-500. If the default
  sampler changes, they must be regenerated, and the per-step variance may change the spectrum.
- **How.** Re-run the spectral pipeline (uniform target, 25 realizations, 1024 points) and the
  quad4 pair correlations for Ours-WVS and Ours-GBN with the new sampler.

#### E16. Tone correction through the full reverse process (P1)

- **Why.** The paper differentiates through a single denoising step and names full
  back-propagation as future work. With 10 steps, full back-propagation fits in memory.
- **How.** Same 24 targets and objective; gradients through all 10 steps vs the one-step estimate.
  Report PSNR / SSIM / MAE and the cost. Turns a stated limitation into a result.

#### E17. Rendered-tone fidelity for every method (P1)

- **Why.** Reviewers may ask for a perceptual number, not only point-set metrics.
- **How.** Render every stipple of Table 1 with the tone experiment's renderer (fixed ink gain) and
  report PSNR / SSIM against the target image for all six methods. CPU only, no sampling.

#### E18. Smart Init on the GPU (P2)

- **Why.** At 10 steps, the CPU rejection sampling (0.13–0.18 s) takes longer than the denoising
  (0.11 s). Moving it to the GPU would bring the total close to the denoising time.
