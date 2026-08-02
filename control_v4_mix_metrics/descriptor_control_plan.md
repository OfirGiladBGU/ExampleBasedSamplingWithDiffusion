# Multi-Oracle Descriptor Control — Test Plan

Milestone-gated. No calendar. A blocking milestone that fails stops downstream work or redirects to
the named fallback.

---

## The claim (stated narrowly — do not overclaim)

**Wrong claim:** "only a learned model can control point-distribution metrics."
Classical samplers control metrics routinely — Lloyd's controls CVT energy, Poisson disk controls
minimum spacing, Heck et al. 2013 targets a specified PCF. Name any *single* metric and there is
usually an optimizer that hits it. A reviewer will say this immediately.

**Correct claim:**

> Learned, **spatially-varying**, **multi-descriptor** control in a **single feed-forward pass**,
> with capacity preserved **across style transitions**.

No classical method accepts "be GBN-like here, Floyd–Steinberg-like there, blend across the
boundary, keep capacity valid throughout." Each optimizes its own objective globally.

**Why the naive workaround fails (this is the proof).** Run two oracles and composite the regions:
neither point set knows about the other, so at the boundary you get double-density or gaps, and
capacity breaks exactly where the styles meet. That failure is demonstrable in a figure and it is
the core non-decomposability argument.

---

## Why this axis and not anisotropy

Anisotropy passed M0–M3 (oracle achievable, clean capacity-respecting targets, multimodality holds)
— it is technically validated, not broken. It is set aside because **anisotropy is a directional
property and the icon dataset is flat**: contours and uniform regions, no hair/fur/grain/flow for
the property to bite on. Demonstrating it would require a different dataset in a domain not
currently available. Recorded as "set aside for lack of suitable data," not "failed."

The descriptor axis is chosen because on flat icons the oracles **are** visibly different — GBN
follows contours, WVS fills regions, and Floyd–Steinberg is expected to be dramatically distinct
from both. Visibility is the thing anisotropy lacked here.

---

## Oracles

Chosen for **descriptor spread and visual distinctness**, not for quality.

**Essential four:**
| Oracle | Character | Role |
|---|---|---|
| GBN | contour-following, high regularity | quality anchor, one end of alignment axis |
| WVS | region-filling, looser | other end of alignment axis |
| Floyd–Steinberg error diffusion | serpentine "worm" structures | maximally distinct, cheap |
| White noise (or large-radius Poisson disk) | clumpy / loose | low-regularity extreme |

**Add if cheap:** ordered/Bayer dithering (hard grid — anchors the maximum-structure corner),
Lloyd/CVT to convergence (near-hexagonal), jittered grid.

**On the "bad" oracles.** Floyd–Steinberg and white noise are not included because they are good —
they are included because they **stretch descriptor space**. With only WVS and GBN the axis is short
and interpolation is subtle. With the extremes in, the space is wide and the middle is visible.
Frame them in the paper as legitimate halftone aesthetics with real print history, not as degraded
blue noise.

All oracles must respect the density map (tone reproduction is halftoning's job) and run at
**matched point count N** per image.

---

## Descriptors

**Hard requirement: every descriptor must be computable on a LOCAL WINDOW.** A single global number
per image permits only global conditioning, and "region A in one style, region B in another" becomes
unreachable — there is no way to say *where*. Each descriptor is emitted as a **spatial field** at
the same resolution as the density map, computed in a sliding window sized to a few expected
spacings.

1. **Local NN statistics** — mean nearest-neighbour distance normalized by expected spacing
   `sqrt(1/lambda)`, plus NN-distance **variance**. Most direct regularity measure; separates Lloyd
   (low variance) from white noise (high). Cheap and robust.
2. **Local capacity CV** — Voronoi (or power) cell area variance.
   **Use POWER cells, not Voronoi, for any comparison against BNOT** — R4's objection. Fix it here
   rather than inheriting the bug.
3. **Edge / structure alignment** — mean image-gradient magnitude sampled at point locations,
   **normalized against rho** so it measures alignment rather than "edges are darker anyway."
   This is the WVS↔GBN axis and the one confirmed visible.
4. **Local anisotropy** (double-angle statistic, already built and validated) — optional but free.
   On flat icons most oracles read ~0, which is fine; it should catch Floyd–Steinberg's directional
   worms. Include if it costs nothing; drop if it adds no separation.
5. **Local PCF first-peak position/height** — richer spectral character, more expensive. Add ONLY
   if 1–4 fail to separate the oracles.

Normalize each descriptor to roughly [0,1] across the dataset so requested values are interpretable.

---

## Milestone 0 — Descriptor separation (BLOCKING, no training)

Cheapest kill-shot. Data prep only.

Compute all descriptors on all oracle outputs across the icon set. Then check:

- **Separation:** do the oracles occupy *distinct, non-overlapping* regions of descriptor space —
  not merely different means, but separated distributions across icons?
- **Non-redundancy:** drop any descriptor correlating > ~0.9 with another descriptor, or with rho.
  Two descriptors at 0.95 correlation are one axis wearing two hats.
- **Visibility sanity check:** render the oracle outputs dots-only, unlabeled. Confirm by eye that
  the endpoints are obviously distinct (especially GBN vs Floyd–Steinberg). This is the check
  anisotropy failed to satisfy on this dataset — do not skip it.

**Blocker:** if oracles do not separate in descriptor space, conditioning on these descriptors
cannot control anything. STOP and re-identify the descriptors before any training.

**Blocker:** if descriptors separate but the outputs are not visually distinguishable, this repeats
the anisotropy situation — metric-real, visually invisible. Re-select oracles toward the extremes
(add Floyd–Steinberg / ordered dither / white noise if not already in) before proceeding.

---

## Milestone 1 — Dataset construction

Once M0 passes:

1. For every source image, run **all** oracles at matched N. Same rho, same budget, same everything
   except the oracle.
2. Compute the descriptor **fields** on each output.
3. Training tuples: `(rho, D, P)` — density map, descriptor field stack, target point set.
4. **Split images into train / val.** See "Validation" below — this is non-negotiable.

Conditioning signal is the **measured descriptor field**, never a one-hot oracle label. A one-hot is
an arbitrary tag with no semantics; nothing between `[1,0,0,0]` and `[0,1,0,0]` means anything, and
the model could only ever emit the trained styles. Measured descriptors make intermediate values
meaningful and make interpolation a real request rather than a latent walk.

---

## Milestone 2 — Conditioning and training

**Injection: internal, alongside existing conditioning.** The descriptor field enters as a learned
spatial conditioning input at the conditioning sites — NOT as a modification to rho. Density stays a
hard condition; style is a residual control. This is the same Path-B requirement as the
capacity-reallocation work: an edit to the input density map would be separable and would destroy
the contribution.

**Architectural constraint (unchanged):** no learned absolute positional embeddings anywhere. The
descriptor branch must not smuggle position dependence in. Re-verify G-transfer at
G ∈ {32, 48, 64} after adding it.

One model, all oracles, conditioned on `(rho, D)`. Everything else follows the imitation guide
(logit-normal t, density-match loss, EMA/unfreeze as adopted) — the descriptor conditioning is the
only new variable.

Do **not** train one checkpoint per oracle. Per-oracle checkpoints are imitators of existing tools
and invite "just run GBN directly." One conditioned model spanning the space is the contribution.

---

## Validation (train/val split — do NOT evaluate only on training images)

There are **two independent generalization axes** and they must not be conflated:

1. **Descriptor generalization** — does the model respond correctly to descriptor values never seen?
   Every intermediate/mixed descriptor is unseen regardless of which image is used.
2. **Image generalization** — does it work on images never seen?

Evaluating only on training images covers axis 1 and silently drops axis 2 — and axis 2 is where a
specific failure hides: on a training image the model may have largely **memorized** the target
point set, so "changing the descriptor changes the metrics" could be interpolation between memorized
outputs rather than a learned controllable manifold. These look identical on train data and entirely
different on new data. Training-set descriptor sweeps are fine as a **debugging** tool to confirm the
control is wired up at all — they are not the reported evidence.

Report everything below on **held-out images**.

**(a) Fidelity (supervised).** Given oracle X's descriptor field, does the output match oracle X?
Necessary but NOT sufficient — this alone is multi-imitation, not a contribution.

**(b) Controllability (the core quantitative figure).** Request unseen descriptor values across the
range. Plot **requested vs measured**. Should be monotonic and near-linear.

**(c) Independence.** Vary one descriptor, confirm the others stay put. If they all move together
you have one entangled axis, not multi-metric control. Worth knowing early.

**(d) Spatial mixing.** Two-region descriptor field, different styles per region. Check both regions
hit their targets **and** capacity/spacing holds through the transition zone.

**(e) THE MONEY FIGURE.** Naive composite (run two oracles separately, stitch the regions) side by
side with the model on the same spatially-varying request. Show the seam breaking in the composite —
double-density or gaps — and holding in the model's output. This single figure carries the
non-decomposability argument better than any prose.

**Honest limitation to state in the paper:** there is no ground truth for mixed/intermediate
descriptor fields. Evidence for those is monotonicity (b), independence (c), capacity preservation
(d), and visual coherence (e) — not comparison against a target. Generalizing beyond the discrete
oracle supervision is itself the interesting claim.

---

## Failure modes — recognize these fast, do not tune against them for months

1. **Snapping.** Mid-range descriptor produces output that looks like one endpoint, not between.
   Diagnostic: requested-vs-measured is a step function, not a ramp.
2. **Muddy averaging.** Mid-range looks like neither endpoint and worse than both — conditional-mean
   averaging appearing in conditioning space. There is prior form for this failure in this project.
   Diagnostic: capacity/spacing metrics degrade mid-range while both endpoints are fine.
3. **Metric response without visual change.** Descriptors move monotonically but nothing looks
   different. This is the anisotropy outcome. Guard: ensure at least one descriptor whose variation
   is **confirmed visible** (edge-alignment WVS↔GBN, and the Floyd–Steinberg extreme) is in the set.
   Do not build a control panel where every dial is invisible.
4. **Entangled descriptors.** Independence test (c) fails — one effective axis, not several.
   Fallback: report single-axis control honestly rather than claiming multi-metric.

**Stop condition:** if intermediates snap, muddy, or are invisible on held-out images, the axis does
not carry a contribution. Report as a negative (descriptor conditioning does not induce a continuous
controllable style manifold) and redirect.

---

## Order

1. **M0** — descriptors + oracles, separation and visibility check. No training. BLOCKING.
2. **M1** — build `(rho, D, P)` dataset with train/val image split.
3. **M2** — add descriptor conditioning (internal), train one model on all oracles.
4. Validate (a) fidelity, then (b) controllability, (c) independence, (d) spatial mixing,
   (e) composite-vs-model seam figure — all on held-out images.
5. Re-verify G-transfer at G ∈ {32, 48, 64} after adding the descriptor branch.

Start with M0. It is an afternoon of data prep, needs no training, and it tells you whether the whole
idea has legs.
