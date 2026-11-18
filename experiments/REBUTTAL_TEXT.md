# StippleDiffusion (paper1182): rebuttal text, draft 1 (2026-11-18)

Written before the real reviews (due Nov 23) from the evidence bank in
[REBUTTAL_DRAFT.md](REBUTTAL_DRAFT.md); the A-numbers point to it. Part 1 is the one-page text,
ordered for the concerns of the mock review; reorder and cut to the real reviews. Part 2 holds
shorter answers to use only if asked. Part 3 lists what must be settled before sending.

Rules kept: every number is a FINAL entry of the evidence bank; nothing about budgets other than
the training budget is quantified; no large-budget BNOT time is quoted.

`[...]` marks a choice or a number that is still open (Part 3).

---

## Part 1. The one-page text (about 770 words without the bracketed options)

We thank the reviewers for their careful reading. All new results below use the submitted
checkpoints; none needed retraining unless stated.

**Speed (few-step sampling).** Replacing the 500 DDPM steps by 25 stochastic DDIM steps from the same
start gives 0.44 s per stipple at 1024 points and 0.61 s at 1600, including the CPU initialization,
against 6.3 s for our 500-step sampler on the same GPU, 5.8 / 10.0 s for WVS and 8.1 / 8.2 s for GBN
(the authors' CUDA implementation). Quality does not drop: across all 12 checkpoints and 4 test
sets, capacity deviation improves by 1–9% and ρs by 1.1–2.3% (paired tests, p < 0.05), and no metric
is significantly worse. Batched, the sampler produces 60 stipples per second. Deterministic solvers
(DDIM η = 0, DPM-Solver++) are not an alternative: they lose 13–17% in HOT energy. *We will present
the 25-step sampler in the supplement (four-dataset table, step sweep, ablation, spectra) and add
its curve and a short paragraph to the runtime section of the paper; the paper's tables keep the
submitted 500-step sampler.*

**Training on BNOT.** We trained it: Ours-BNOT beats the GBN and WVS samplers in HOT energy, centroid
displacement and W2 on all four datasets, and comes close to BNOT (Icons-50 HOT ×10⁵: BNOT 2.42,
Ours-BNOT 2.85, GBN 5.20, WVS 10.5). The reviewer is right that BNOT exemplars are affordable.
*We will add Ours-BNOT to Table 1 and correct the sentence that calls BNOT exemplars impractical.*

**Is the method example-based, and why does a student beat its teacher?** With three exemplar samplers
on 4 datasets and 7 metrics, wherever two samplers differ significantly, the learned variants
reproduce their order in 78 of 82 cases (Kendall τ = 0.90). The gain over GBN is significant in a
paired test on 1000 Icons-50 images: HOT −22% (p < 1e-28), centroid −21%, W2 −13%. It does not come
from the initialization (from pure noise HOT is still 15% lower) or from the KDE loss (see below).
The model predicts a displacement field from the OT grid, which favors low transport cost and
costs local regularity; this matches its losses in capacity, CVT and ρs. *We will add the paired
tests and this explanation.*

**Capacity.** The capacity gap is the same whichever sampler provides the exemplars, so it is a limit
of the model. It is not the aspect ratio: on already-square images of the two non-square datasets
it remains 2–4×. It is not a generalization gap either: on training images capacity deviation is
still 1.6–1.9× the teacher's. It arises in the last low-noise denoising steps. *We will state this
limitation precisely [and change the title to "…" / and define "capacity-constrained" as the
property of the exemplars].*

**Tone correction.** We must correct the paper here. Table 3 was scored on the single-step decode
used for the gradient, not on the sampler's output. Measured on the real sampler with fresh noise,
optimizing through the full 25-step chain improves PSNR by 0.39 dB (18 of 24 targets, p = 0.002),
and the result renders above GBN's own tone (16.95 vs 16.81 dB). The gradient-free feedback
suggested by the reviewer gives +0.24 dB with GBN as the stippler and +0.15 to +0.18 dB with ours.
Optimizing the points directly reaches higher PSNR only by clumping them (capacity deviation 0.36
against 0.05). *We will replace Table 3 by the full-chain results and describe the protocol exactly.*

**Ablations.** (i) Gated injection: the full model retrained without it is worse in CVT (+0.8%),
capacity (worse on 82% of the images) and ρs (−0.7%), all p < 1e-4 paired; transport metrics do not
change. (ii) KDE loss: without it the final model does not differ on any metric; it only speeds up
early training, and we will say so. (iii) Start of sampling: later starts are worse (HOT +19% to
+111%), and so is pure noise (8–16% in HOT, centroid and W2), so the initialization is a quality
component. (iv) Steps: see above.

**Generalization.** Trained on Icons-50 and applied to CelebA, the model keeps its transport quality
and still beats GBN in HOT and W2; the reverse direction fails in capacity, which we will report as
a limitation. The test icons are the held-out 10% of a seeded split. Most of them do have a
same-emoji icon from another vendor in training, as the reviewer suspected, but quality relative to
the teacher does not improve with the similarity of that icon (|ρ| ≤ 0.14 for every metric).

**Committed revisions.** Ours-BNOT in Table 1; the 25-step sampler in the supplement (same
weights, no retraining) with its tables, and its curve in the runtime figure; Table 3 replaced; the gating and KDE ablations; paired tests and rank
preservation; cross-dataset and anisotropy results in the supplement; Fig. 4 regenerated (see
Part 2); corrected BNOT-cost and capacity wording; the missing references; code and checkpoints.

---

## Part 2. Shorter answers, to use only if the point is raised

**Implementations and what is timed (Q1).** WVS is Rougier's NumPy / SciPy reference implementation
(CPU), BNOT the authors' C++ code (CPU), GBN the authors' CUDA binary. Our time includes the
initialization: rejection sampling plus an exact assignment of the points to the grid, 0.14 s at
1024 points. *We will state the implementations and the breakdown in the paper.*
`[BNOT times: wait for the runtime check before quoting any.]`

**"The WVS baseline is unoptimized."** We agree that it is a reference implementation and will say
so. A WVS ten times faster would tie our 25-step sampler at 1024 points; the comparison that does
not depend on implementation is GBN on the same GPU class (8.1 s against 0.44 s).

**Tile- and lattice-based samplers.** They are faster, but they are not example-based and are not
differentiable with respect to the density, which our tone correction needs. *We will add this
discussion and the references to related work.*

**Mottling at 4096 points in Fig. 4.** The large-budget panels used an assignment solver that stopped
before optimality from about 3000 points on (training and all tables are unaffected). *We
regenerated Fig. 4 with the exact assignment.* The model is trained at one budget; we show other
budgets qualitatively and state that quality is best near the training budget.

**Quantitative results at 4K–10K points.** The models are trained at 1024 or 1600 points only, so
we do not claim quantitative quality at other budgets. *We will make this limit explicit.*

**Aspect ratio (Q2).** It is not given to the network; coordinates are normalized per axis. This is
not the cause of the capacity gap (see Capacity).

**50–100 steps (Q4).** 50 and 100 stochastic DDIM steps also improve on 500 DDPM steps (capacity
−6.4% and −4.4%); 25 is the best.

**Spectral anisotropy (W7).** Measured as the angular variance of the power spectrum around the
blue-noise peak (25 realizations, lower is more isotropic): GBN −9.6 dB, Ours-GBN −8.4 dB; WVS
−14.2 dB, Ours-WVS −9.7 dB; BNOT −15.2 dB. Ours-GBN is close to its teacher, Ours-WVS is clearly
more anisotropic. *We will report these numbers next to the radial averages and state the
limitation.*

**Does the model memorize? (W6)** Its metrics are the same on training and held-out images (for
example CVT 1.035× against 1.041× the teacher's). It does reproduce training exemplars more closely
(0.15 against 0.23 cell widths from the exemplar's points), which does not turn into better metrics.

**"EMD" (minor).** Our metric is the 2-Wasserstein distance (the square root of the optimal
transport cost with squared distances). *We will call it W2.*

**Fig. 3 strip percentages (minor).** They come from one realization per method. *We will report the
mean and standard deviation over 25 realizations: `[numbers from fig3_strip_realizations]`.*

**Variation between runs (W5).** `[E14: spread of the dataset means over three independent runs,
compared with the reported differences.]` Each variant was trained once; we will say so.

**Training cost, model size (minor).** About 2.6 days per model on one RTX 6000-class GPU (5000 epochs);
`[parameter count from model_stats]`. Validation metrics are flat from about epoch 2500.

**Perfect-square point counts.** A limitation of the grid representation; we will state it.

**Sinkhorn rows.** We agree that they do not separate the methods and will move them to the supplement.

**Optional, only if capacity is pressed:** the capacity loss is tied to the loss weighting of the
last denoising steps. A 300-epoch fine-tune of the submitted checkpoint without that weighting
lowers capacity deviation by 13% on 92% of held-out images, with no metric worse, while a control
fine-tune with the original weighting does not change it. We would present this as an analysis in
the revision, not as a new method.

---

## Part 3. To settle before sending

1. **Sampler: decided (user, 2026-11-18).** The 25-step sampler goes into the supplement as a
   faster sampling variant of the same trained weights (call it a sampler, not a new model), with
   its curve and a short paragraph in the paper's runtime section. The paper's tables keep DDPM-500.
   Part 1 is written this way.
2. **Title.** Choose one of the two bracketed endings of the Capacity paragraph.
3. **BNOT runtime.** No BNOT time is quoted in Part 1. If the investigation confirms the stored
   times at 1024 / 1600 points (13.6 / 25.5 s), add them to the Speed paragraph and add the exemplar
   cost (about 38 h for Icons-50, 1.7× GBN's) to the BNOT paragraph.
4. **Pending numbers:** E14 spread, Fig. 3 means, parameter count (all running, 2026-11-18).
5. **Length.** Part 1 is about 770 words; cut the paragraphs no reviewer asks about.
6. **Check against the real reviews** that no answer promises something a reviewer did not ask for
   (EG: large unrequested changes are risky).
