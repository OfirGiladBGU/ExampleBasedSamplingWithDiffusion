# Tone-correction experiment — demonstrating differentiability

Answers reviewer item #5: *"the paper advertises an end-to-end differentiable sampler, but
the differentiability is never demonstrated."*

## What the experiment is

Rendered stipples use finite-radius dots. Where dots overlap, ink saturates, so the tone you
**perceive** is not the density you **asked for** — the classical dot-gain problem, normally
patched with a hand-tuned transfer curve. Because our sampler is a network, we can instead
*solve* for the pre-compensated density:

```
rho* = argmin_rho  || Render(Sampler(rho)) - I ||^2 + lambda * R(rho)
```

Gradients run: loss → rendered image → point positions → denoiser → `rho`.

**No baseline can do this.** WVS, BNOT and GBN are iterative solvers; putting one in this loop
would need gradients through Lloyd iterations or a Newton solve on a weighted Laplacian. That
is the point — made concretely rather than asserted.

Three configurations are compared, all fitted by the same optimizer against the same objective:

| config | conditioning | role |
| --- | --- | --- |
| `none` | the target image itself | what you get today; evaluated, not optimized |
| `curve` | one global monotone transfer curve | the classical fix, and the honest baseline |
| `field` | a spatially varying correction | ours; only reachable through the sampler |

## What data is needed

**None that you do not already have, and no new generation runs.**

The objective compares a *rendered stipple* against a *target image*. It never references an
exemplar point set, so the GBN / WVS / BNOT datasets are **not** required here. The only inputs
are grayscale target images — reuse any existing `source/` directory, e.g.
`experiments/outputs/quantitative_advance_metrics/source`.

You also need one trained checkpoint. Use the same one the paper reports; both variants work,
and running both is a reasonable extra if time allows.

## Running it

```bash
# 0. pick targets (writes a manifest; no images are generated)
python experiments/tone_results_stage_0_prepare.py \
    --source experiments/outputs/quantitative_advance_metrics/source --count 24

# 1. optimize the conditioning, per image
python experiments/tone_results_stage_1_optimize.py \
    --control_ckpt_path control_v4/train_outputs_Icons-50_1024_GBN_full/dynamic_ep5000.ckpt \
    --steps 300 --mode onestep

# 2. figures, captions and the summary table -- no arguments needed; this writes
#    exactly the three figures the paper uses into tone_results/plots/
python experiments/tone_results_stage_3_plots.py

# optional: the exploratory figures as well. They go to tone_results/plots-backup/,
# so plots/ always holds exactly the paper set.
python experiments/tone_results_stage_3_plots.py --extras
```

Every default is repo-root-relative, so run these from the project root and no path needs
passing. Nothing outside `experiments/outputs/tone_results/` is written -- in particular
these scripts never touch `paper/`.

Before the real run, check the plumbing without a GPU or a checkpoint:

```bash
python experiments/tone_results_stage_1_optimize.py --self-test --steps 30
```

This substitutes a stub sampler and one synthetic ramp. It validates the renderer, the
optimizer, the metrics and the figure code end to end. It says **nothing** about the trained
model — the stub is not the model.

## Run order

Numbered stages run in order; the numbering *is* the dependency. The one `extra` script is
genuinely order-free.

| script | needs | GPU + ckpt |
|---|---|---|
| `tone_results_utils.py` | *not a stage* -- the shared renderer, `DensityField`, the model wrapper and the metrics. Imported by stages 1 and extra | n/a |
| `tone_results_stage_0_prepare.py` | a `source/` dir of grayscale images | no |
| `tone_results_stage_1_optimize.py` | stage 0 | yes |
| `tone_results_stage_2_gain_sweep.py` | stage 1 -- reads the calibrated ink gain from `main/summary.json` | yes |
| `tone_results_stage_3_plots.py` | stage 1, and stage 2 for the second graph | **no** |
| `tone_results_stage_extra_gradcheck.py` | stage 0 only -- nothing reads its output, its numbers go straight into the paper text, so run it whenever | yes |

Stage 2 must precede stage 3 because stage 3 draws the paper's second graph from
`gain_sweep/gain_sweep.csv`.

Only stage 3 is cheap and repeatable -- it reads the `.npz` files stage 1 wrote and never
touches the model, which is why it takes no arguments and runs in seconds.

## Gradient modes

`--mode onestep` (default) noises the fixed initialization to one low timestep, runs the
denoiser once, and takes the closed-form `x0` estimate. This is exactly the path the
Component-2 KDE loss in `train_control.py` already differentiates through, so it is known to
give usable gradients. One forward and one backward per step.

`--mode unrolled --unroll-steps K` runs the truncated reverse process and keeps gradients for
the final `K` steps only. Faithful to what inference actually does, at proportionally more
memory. Start with `onestep`; escalate only if a reviewer would notice.

## What is differentiated, and what is not

Stated plainly, because the paper should state it too.

**Differentiated:** the density channels of the conditioning (the high-resolution image and
the area-pooled target density), through the ControlNet and the locked denoiser, to the point
coordinates, through the splat, to the rendered image.

**Held fixed, deliberately:**

- **the SDF channels** — they encode the silhouette, and this experiment changes tone, not
  shape, so freezing them stops the optimizer quietly redrawing the object. They also come
  from a thresholded distance transform, which carries no useful gradient.
- **the Smart-Init point set** — it comes from rejection sampling, which is not
  differentiable. It plays the role of the noise seed here, and is drawn once from the
  original image so every configuration starts identically.

The claim is that gradients flow *through the sampler*, which they do — not that every stage
of the pipeline is differentiable.

## The rendering model

`RENDER_RES`, `DOT_SIGMA_PX` and `INK_GAIN` in `tone_results_stage_1_optimize.py` define the "printer"
being compensated for: dot radius and how fast overlapping ink saturates
(`darkness = 1 - exp(-gain * ink)`, Beer-Lambert). These are the experiment's physical model,
not tuning knobs — they are recorded in `summary.json` so the run is reproducible. If a
reviewer objects that the effect was manufactured by an extreme choice, re-run with a smaller
`--ink-gain`: the gap narrows, and it should.

## Failure modes the code refuses to hide

- a **zero gradient** at step 0 aborts with an explicit message. It means nothing connects
  `rho` to the rendered image — most likely `set_condition` is not receiving the tensor being
  optimized.
- a **non-finite gradient** aborts rather than producing a plausible-looking result from a
  broken path.

Neither is a warning. A silently wrong differentiability demonstration is worse than none.

## Outputs

```
experiments/outputs/tone_results/
    README.md              the index: headline numbers, layout, caveats
    manifest.json          the selected targets
    source/                staged condition images
    main/                  <stem>.npz, tone_summary.csv, summary.json
    gradcheck/             gradcheck.json
    gain_sweep/            one folder per ink-gain multiplier + gain_sweep.csv
    plots/
    plots/                 THE PAPER SET (default run), plus tone_table.md, captions_paper.md
        tone_improvement.*          graph 1: per-image gain, all three strategies
        tone_gain_sweep_compact.*   graph 2: advantage vs ink gain
        tone_explain_compact_<stem>.*  the visual: densities, stipples, error maps
    plots-backup/          everything else, for the rebuttal: tone_hero_*, tone_correction_*,
                           tone_explain_* (6-panel), tone_convergence, tone_scatter,
                           tone_gain_sweep (2-panel), captions.md
```

## Honest caveats to carry into the paper

- This demonstrates differentiability **on a task we chose**. It shows gradients are usable,
  not that they are useful for every downstream objective.
- `onestep` differentiates a one-step estimate, not the full reverse trajectory. If the
  headline claim is "end-to-end through inference", that needs an `unrolled` run at a
  scale that supports it -- ~285x the cost per step. The claim made instead is the one the
  reviewer asked for: a concrete task that exploits the gradients.
- The dot-gain model is a stand-in for a real printer, not a calibrated one.
