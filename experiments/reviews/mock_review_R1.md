# Review: paper1182, "StippleDiffusion: Capacity-Constrained Stippling using Controlled Diffusion" (EG 2027)

(Mock / example review the user received on 2026-11-15. Answers: REBUTTAL_DRAFT.md section 5.)

Reviewed from the main PDF and the supplemental PDF (version compiled 15 Nov 2026). The video was not seen and is not factored in.

**Recommendation: Borderline, leaning reject (4/10).**
**Reviewer confidence: 4/5** (familiar with diffusion models and blue-noise sampling; metrics were not re-derived).

## Summary

The paper trains an image-conditioned point-diffusion model to imitate a classical stippler (WVS or GBN). It builds on the OT-grid representation of Doignies et al. 2023 and adds a jointly trained ControlNet-style branch with GECCO per-point image features and gated injection, a KDE density loss, and SDEdit-style truncated sampling from a density-weighted rejection-sampled initialization. It is evaluated on four datasets with seven metrics, a spectral analysis, runtimes, an ablation, and a dot-gain correction experiment that uses the sampler's differentiability.

## Strengths

S1. To my knowledge this is the first image-conditioned point-diffusion sampler for stippling. It is a sensible extension of DBC*23 that removes its one-model-per-density limitation.

S2. The evaluation is broad and unusually candid: seven metrics, a BNOT reference, standard deviations in the supplement, and text that states where the method loses.

S3. Budget generalization from one checkpoint (Fig. 4) and the "student inherits the teacher's ranking" result (27 of 28 pairs) are good evidence that the model learns a sampler's style.

S4. The tone-correction experiment shows a capability beyond imitation.

S5. The paper is clearly written and the pipeline figure is easy to follow.

## Weaknesses

W1. The practical value is unclear. The learned sampler is worse than its teacher on CVT energy, spatial measure and capacity deviation, so it has to win on speed. At 1024 and 1600 points, the only budgets with quantitative quality results, Fig. 7 puts it at about the same time as WVS (roughly 6 s), with the proposed method on an RTX 6000 and WVS on a CPU. The speed advantage appears only from about 2304 points, where no quality metrics are reported and where Fig. 4 already shows mottling at 4096. Several seconds for 50 Lloyd iterations on 1K points also suggests an unoptimized baseline. The fast deterministic methods cited in related work (tiles, lattices) are not compared.

W2. "Capacity-constrained" in the title is not supported. Nothing enforces capacity, and capacity deviation is 1.7 to 3.6 times worse than GBN on every dataset (0.107 against 0.030 on Anime Chibi) and far from BNOT, which is the actual capacity-constrained method. The stated reason for not training on BNOT is cost, but Fig. 7 puts BNOT at roughly 13 s for 1024 points. That is about 30 CPU-hours for the 9K Icons-50 exemplars, a one-off and parallel cost that is small next to 5000 epochs for each of eight checkpoints.

W3. Differentiability is the main distinguishing claim and its demonstration is thin. Gradients go through a single denoising step at t = 120, not through the sampler. The finite-difference check validates autograd on this surrogate, not agreement with the full 500-step sampler. The gain is small (15.76 to 16.59 dB, about 0.6 dB over a global tone curve) at low absolute PSNR and SSIM. The obvious baseline is missing: gradient-free iterative feedback, rho <- rho + eta (rho0 - R(S(rho))), as in model-based halftoning, which works with any black-box stippler including WVS. Random search over 4096 dimensions with 600 evaluations is a weak comparison.

W4. The student beating its teacher on HOT energy, centroid displacement and EMD is not explained. The per-image standard deviations are as large as the differences (4.06 +/- 3.06 against 5.20 +/- 4.86 on Icons-50) and no paired test is reported. The effect may come from the KDE loss or the initialization, and no ablation isolates it. The Sinkhorn rows are uninformative (all within 0.002).

W5. The ablation has gaps. Gating alone brings almost nothing (capacity 0.0609 against 0.0626, CVT worse), and there is no jointly-trained variant with GECCO and without gating, so gated injection, a listed contribution, is not supported. There is no ablation of the KDE loss, of the truncation point (only 50%), or of the number of steps or a faster sampler, which is what the runtime claim depends on. Each variant is a single training run, and several differences (CVT 2.399 against 2.412) look within noise.

W6. Scale and generality are limited. 1024 to 1600 points is small for stippling, n must be a perfect square, and each dataset and sampler needs its own model. There is no cross-dataset test, so "generalizes to unseen images" holds in-domain only. A random split of Icons-50 also places near-identical icons (the same emoji in different vendor styles) in both train and test.

W7. Spectral quality. The 2D spectra of the learned variants in Fig. 5 are visibly anisotropic and grid-aligned, with a higher low-frequency floor, and only a uniform density is analysed. This is acknowledged, but isotropy is central to blue noise and the radial average hides it.

W8. Novelty lies mostly in the combination: OT grid, ControlNet, GECCO features, highway gates, SDEdit and Min-SNR all exist. That is acceptable when the result is compelling; together with W1 it reads as incremental.

## Questions for the rebuttal

Q1. In Fig. 7, how are WVS and BNOT implemented (language, threads)? Is the OT assignment of the initial points included in your time, which solver is used, and how does it scale with n?

Q2. Is the aspect ratio given to the network? With per-axis normalization, isotropic spacing in image space becomes anisotropic in the unit square. Could this explain the larger capacity gap on the two non-square datasets?

Q3. Why does the learned sampler improve HOT energy over its teacher?

Q4. What is the quality with 50 to 100 steps, or with a deterministic sampler?

Q5. In the tone correction, is the reported PSNR measured on the full sampler with the initialization re-drawn from the optimized density?

## Missing references

Deussen et al. 2017 (Weighted Linde-Buzo-Gray Stippling); Kim et al. 2009 (Stippling by Example), relevant to the "example-based" framing; Martin et al. 2017 (A Survey of Digital Stippling); Huang et al. 2023 (Patternshop).

## Minor points

- EMD is described as the 2-Wasserstein distance; EMD usually denotes W1.
- Supplemental Figs. 2 to 4 have no axis units, and their 16 to 25 s wall times next to 5.65 s in the paper are confusing even with the profiling caveat.
- The introduction motivates rendering and object placement, but only stippling is evaluated.
- Fig. 8 is too small to judge the ablation visually.
- Training time, parameter count and code release are not stated.
- It is unclear whether the strip percentages in Fig. 3 come from one realization or an average.

## What would raise the score

A BNOT-trained variant (or a retitled paper), a quality-against-time curve over step counts with a properly optimized WVS baseline, quantitative results at 4K to 10K points, and a tone-correction comparison against iterative feedback with a classical sampler.
