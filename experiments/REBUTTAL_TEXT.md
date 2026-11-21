# StippleDiffusion (paper1182): rebuttal text, draft 1 (2026-11-18, numbers updated 2026-11-21)

Written before the real reviews (due Nov 23) from the evidence bank in
[REBUTTAL_DRAFT.md](REBUTTAL_DRAFT.md); the A-numbers point to it. Part 1 is the one-page text,
ordered for the concerns of the mock review; reorder and cut to the real reviews. Part 2 holds
shorter answers to use only if asked. Part 3 lists what must be settled before sending.

Rules kept: every number is a FINAL entry of the evidence bank; nothing about budgets other than
the training budget is quantified; BNOT times are the fixed-build ones (REBUTTAL_DRAFT.md,
watch-out 6), never the submitted 13.6 s / 1802 s.

`[...]` marks a choice or a number that is still open (Part 3).

---

## Part 1. The one-page text (about 790 words without the bracketed options)

We thank the reviewers for their careful reading. All new results below use the submitted
checkpoints; none needed retraining unless stated.

**Speed (few-step sampling).** Replacing the 500 DDPM steps by 25 stochastic DDIM steps from the same
start gives 0.42 s per stipple at 1024 points and 0.60 s at 1600, including the CPU initialization,
against 5.5 / 5.9 s for our 500-step sampler on the same GPU, 6.8 / 11.0 s for WVS, 9.5 / 9.5 s for GBN
(the authors' CUDA implementation) and 6.1 / 5.8 s for BNOT (medians over 10 targets, all methods
re-timed on one protocol). Quality does not drop: across all 12 checkpoints and 4 test
sets, capacity deviation improves by 1–9% and ρs by 1.1–2.3% (paired tests, p < 0.05), and no metric
is significantly worse. Batched, the sampler produces 60 stipples per second. Deterministic solvers
(DDIM η = 0, DPM-Solver++) are not an alternative: they lose 13–17% in HOT energy. *We will present
the 25-step sampler in the supplement (four-dataset table, runtime graph, step sweep, ablation,
spectra) and point to it from the runtime section of the paper; the paper's tables keep the
submitted 500-step sampler.*

**Training on BNOT.** We trained it: Ours-BNOT beats the GBN and WVS samplers in HOT energy, centroid
displacement and W2 on all four datasets, and comes close to BNOT (Icons-50 HOT ×10⁵: BNOT 2.42,
Ours-BNOT 2.85, GBN 5.20, WVS 10.5). The reviewer is right that BNOT exemplars are affordable:
about 6 s per 1024-point image on one CPU thread, less than GBN's 9.5 s on a GPU.
*We will add Ours-BNOT to Table 1 and correct the sentence that calls BNOT exemplars impractical.*

**Is the method example-based, and why does a student beat its teacher?** With three exemplar samplers
on 4 datasets and 7 metrics, wherever two samplers differ significantly, the learned variants
reproduce their order in 78 of 82 cases (Kendall τ = 0.90). The gain over GBN is significant in paired
tests on 1000 Icons-50 images in each of three independent runs: HOT 12–21% lower, centroid
13–20%, W2 9–13% (p < 1e-10 in every run). It does not come
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
used for the gradient, not on the sampler's output. We now back-propagate through the full 25-step
chain and score the paper's 500-step sampler on fresh noise: the optimized density improves PSNR by
0.38 dB (17 of 24 targets, p = 0.004), against +0.15 dB for the gradient-free feedback suggested by
the reviewer (+0.24 dB with GBN as the stippler). A global tone curve optimized through the sampler
in the same way reaches the same PSNR, and our spatially varying density gives the lowest MAE; we
will state this. The gain over the uncorrected stipple grows with the strength of the dot gain
(+1.4 dB at four times the calibrated ink gain). Optimizing the points directly reaches higher PSNR
only by clumping them (capacity deviation 0.36 against 0.05). *We will replace Table 3, the two
tone figures and the supplement's ink-gain sweep by the full-chain results and describe the
protocol exactly.*

**Ablations.** (i) Gated injection: the full model retrained without it is worse in CVT (+0.3% to
+0.8%) and capacity (median +4.5% to +7.7%, worse on 60–82% of the images), significant in both of
two independent evaluations; transport metrics do not change. The effect is small. (ii) KDE loss: without it the final model does not differ on any metric; it only speeds up
early training, and we will say so. (iii) Start of sampling: later starts are worse (HOT +19% to
+111%), and so is pure noise (8–16% in HOT, centroid and W2), so the initialization is a quality
component. (iv) Steps: see above.

**Generalization.** Trained on Icons-50 and applied to CelebA, the model keeps its transport quality
and still beats GBN in HOT and W2; the reverse direction fails in capacity, which we will report as
a limitation. The test icons are the held-out 10% of a seeded split. Most of them do have a
same-emoji icon from another vendor in training, as the reviewer suspected, but quality relative to
the teacher does not improve with the similarity of that icon (|ρ| ≤ 0.14 for every metric).

**Committed revisions.** Ours-BNOT in Table 1; the 25-step sampler in the supplement (same
weights, no retraining) with its tables and runtime graph; Fig. 7 re-timed; Table 3, the tone figures and the ink-gain sweep replaced; the gating and KDE ablations; paired tests and rank
preservation; cross-dataset and anisotropy results in the supplement; Fig. 4 regenerated (see
Part 2); corrected BNOT-cost and capacity wording; the missing references; code and checkpoints.

---

## Part 2. Shorter answers, to use only if the point is raised

**Implementations and what is timed (Q1).** WVS is Rougier's NumPy / SciPy reference implementation
(CPU), BNOT the authors' C++ code (CPU), GBN the authors' CUDA binary. Our time includes the
initialization: rejection sampling plus an exact assignment of the points to the grid, 0.14 s at
1024 points. *We will state the implementations and the breakdown in the paper.*
The BNOT times in the submission were inflated by our own build: our headless port had replaced
the original's sparse QR solver by a slower one. With the original solver and BNOT's default convergence
criteria it takes 6.1 s at 1024 points and 12.8 s at 6400 (median over 10 targets; submitted:
13.6 s and 1802 s). The point sets agree with the earlier ones to within 10⁻⁵ of a point spacing,
so every BNOT exemplar and quality number is unchanged. *We re-timed all four methods on one
protocol and will replace Fig. 7 and the text built on it.*

**"The WVS baseline is unoptimized."** We agree that it is a reference implementation and will say
so. A WVS ten times faster would tie our 25-step sampler at 1024 points; the comparison that does
not depend on implementation is GBN on the same GPU (9.5 s against 0.42 s).

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

**Fig. 3 strip percentages (minor).** They come from one realization per method. Over 25
realizations (targets 1.7 / 11.0 / 29.7 / 57.6%): Ours-GBN 1.4 ± 0.2 / 10.7 ± 0.4 / 30.4 ± 0.4 /
57.4 ± 0.8%, Ours-WVS 1.2 ± 0.3 / 12.7 ± 0.7 / 30.4 ± 0.6 / 55.7 ± 1.0%. *We will report these means and
correct "within about one percentage point": it holds for Ours-GBN (0.8 points on average) and is
about two points for Ours-WVS.*

**Variation between runs (W5).** In three independent runs of Ours-GBN on the 1000 Icons-50 test
images (new initialization and noise seeds), the dataset means vary by under 0.1% for CVT, ρs and
Sinkhorn, about 2% for capacity and W2, and 5–7% for HOT and centroid displacement, which are
heavy-tailed. The effects we rely on are larger and hold in every run: the 25-step sampler lowers
capacity deviation by 10% and raises ρs by 1.5% (median per image), and the gap to GBN in HOT is
12–21%. Differences between the two samplers in HOT, centroid and W2 are within this noise, and we
do not claim them. Each variant was trained once; we will say so.

**Training cost, model size (minor).** About 2.6 days per model on one RTX 6000-class GPU (5000 epochs);
44.1 M parameters (27.3 M in the denoiser, 16.7 M in the conditioning branch). Validation metrics are flat from about epoch 2500.

**Perfect-square point counts.** A limitation of the grid representation; we will state it.

**Sinkhorn rows.** We agree that they do not separate the methods and will move them to the supplement.

**Optional, only if capacity is pressed:** the capacity loss is tied to the loss weighting of the
last denoising steps. A 300-epoch fine-tune of the submitted checkpoint without that weighting
lowers capacity deviation by 13% on 92% of held-out images (9–18% across later checkpoints), with
no metric worse, while a control
fine-tune with the original weighting does not change it. We would present this as an analysis in
the revision, not as a new method.

---

## Part 3. To settle before sending

1. **Sampler: decided (user, 2026-11-18).** The 25-step sampler goes into the supplement as a
   faster sampling variant of the same trained weights (call it a sampler, not a new model), with
   its curve and a short paragraph in the paper's runtime section. The paper's tables keep DDPM-500.
   Part 1 is written this way.
2. **Title.** Choose one of the two bracketed endings of the Capacity paragraph.
3. **BNOT runtime: resolved 2026-11-18.** The submitted times came from a slow solver in our BNOT
   port. **Updated 2026-11-20:** all methods re-timed (runtime_v10, medians over 10 new targets,
   BNOT at its true defaults; REBUTTAL_DRAFT.md watch-out 7). The new times (BNOT 6.1 / 5.8 s at
   1024 / 1600 points) are in the Speed paragraph, the exemplar cost in the BNOT paragraph, and the
   correction itself in Part 2 (Q1). **Decide whether the correction moves to Part 1:** Fig. 7
   changes visibly (BNOT no longer climbs to 1802 s), and the paper's 500-step sampler is then on
   par with BNOT at the training budgets (5.5 / 5.9 s against 6.1 / 5.8 s) and slower at 256 and 576
   points. The speed argument against BNOT rests on the 25-step sampler.
4. **Pending numbers: all in (2026-11-21).** E14 spread, Fig. 3 means and the parameter count are
   filled in above (REBUTTAL_DRAFT.md A21–A23). Two statements were softened because of them: the
   gap to GBN is now a range over three runs, and the gating effect a range over two evaluations.
5. **Length.** Part 1 is about 790 words; cut the paragraphs no reviewer asks about.
6. **Check against the real reviews** that no answer promises something a reviewer did not ask for
   (EG: large unrequested changes are risky).

<!-- bnot-times-fixed-20261118: BNOT runtimes replaced by the fixed-build measurements -->
