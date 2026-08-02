# WVS↔GBN Style Axis — Dataset & Experiment Plan

Runs in parallel with the anisotropy branch. Different risk profile: no new oracle to build, uses
data that already exists, and the property is **visible** (which is exactly what anisotropy lacks).

---

## What this axis is (and what it is NOT)

**The observed difference:** GBN produces crisp edges / points that follow contours. WVS fills
regions more evenly. This is visually distinguishable — that is the whole reason this axis is worth
pursuing.

**This is NOT a spectral axis.** Pair correlation functions and radial power spectra describe how
points space relative to *each other*, isotropically averaged, ignoring image content. "Follows
contours" vs "fills interiors" is about where points sit relative to **image structure**. Two point
sets can have nearly identical PCFs while one hugs contours and the other doesn't. Using a PCF /
spectral descriptor here would measure the wrong thing, find WVS and GBN nearly identical, and kill
a direction that is actually visible.

**Name it correctly: edge / structure alignment.** The descriptor must measure point position
relative to image structure, not point position relative to other points.

**Why it is non-decomposable.** Fix the density map ρ, vary the style control, get different point
sets. A grayscale ρ encodes local intensity ("how many points here") — it has no channel for "and
place them along contours rather than spread through the interior." Arrangement-relative-to-structure
is orthogonal to ρ, so no ρ-edit + single-oracle pipeline reproduces it. Decomposition test passes.

---

## Hard constraints (violating either kills the contribution)

1. **BNOT is dropped.** It is hard to distinguish from WVS. A third label adds a class without
   adding visible range, and it compresses the axis. Two maximally-separated endpoints give
   interpolation the most room to be visible.

2. **NEVER manufacture intermediate data by editing the input image.** Sharpening edges / boosting
   contrast and re-running an oracle produces intermediates that a reviewer reproduces in one line
   ("edit the image, run GBN"). That is Path A / separable and it destroys the exact property that
   makes this a contribution. The control must be injected internally, never as a ρ edit.

3. **No intermediate ground truth exists.** Neither GBN nor WVS exposes a parameter controlling
   point distribution over the image (only iteration count and point count — iterations control
   *convergence*, not style; an under-converged GBN is a worse GBN, not "halfway to WVS"). So
   the middle of the axis is unsupervised. This is a known, accepted limitation — see Validation.

---

## Gate 0 — Endpoint separation (BLOCKING, do before anything else)

Cheapest possible kill-shot. Uses data you already have. No training.

**Build an edge-alignment descriptor.** Candidates (pick 1–2, keep it small):
- Distribution of distance from each point to the nearest image edge (edge map from a Sobel /
  Canny / gradient-magnitude threshold on the source image).
- Fraction of points within k pixels of an edge, for a few k.
- Mean gradient magnitude of the source image sampled at point locations.
- Ratio of point density in high-gradient vs low-gradient regions, normalized by the density map so
  it measures *alignment*, not tone.

Normalize against ρ where possible so the measure captures arrangement-relative-to-structure and
not simply "edges are darker so more points go there."

**Run it on existing WVS and GBN outputs across the icon set.**

**Pass:** WVS and GBN separate cleanly and consistently on this measure — ideally non-overlapping
distributions across icons, not just a difference in means.

**Blocker:** if they do not separate, the descriptor is measuring the wrong thing. The visible
difference is something else (possibly detail preservation or clumping rather than alignment).
**STOP and re-identify the visual property before training anything.** Do not proceed on a
descriptor that does not distinguish the endpoints — everything downstream depends on it.

This descriptor is also the evaluation metric for the whole experiment, so it has to be right.

---

## Dataset construction

Once Gate 0 passes:

1. For every source image in the icon set, generate **both** oracle outputs at matched point count:
   `P_WVS` and `P_GBN`. Same ρ, same N, same everything except the oracle.
2. Compute the edge-alignment descriptor `s` on each output.
3. Training pairs: `(ρ, s, P)` — density map, measured style descriptor, target point set.

**Conditioning signal: use the measured descriptor, not a one-hot label.**
- A one-hot `[1,0]` / `[0,1]` is an arbitrary label with no meaning; nothing "between" two one-hot
  vectors means anything, and the model can only ever produce the two trained styles.
- The measured descriptor `s` is a continuous, *meaningful* axis. WVS lands at one value, GBN at
  another, and values in between specify a real, well-defined alignment strength. Same two training
  values, but interpolation now has semantics.
- Normalize `s` to roughly [0,1] across the dataset (WVS ≈ 0, GBN ≈ 1) so intermediate conditioning
  values are interpretable.

**Injection: internal, alongside existing conditioning.** Feed `s` as a learned conditioning vector
(FiLM / additive embedding at the conditioning sites), NOT as a modification to ρ. This is the same
Path-B requirement as the capacity-reallocation work. Density stays a hard condition; style is a
residual control.

**Architectural constraint (unchanged):** no learned absolute positional embeddings anywhere. The
style branch must not sneak position dependence in — re-verify G-transfer after adding it.

---

## Training

One model, conditioned on `(ρ, s)`. Train on both oracles' outputs jointly. Everything else follows
the imitation guide (logit-normal t, density-match loss, EMA/unfreeze as adopted) — the style
conditioning is the only new variable.

Do not train three checkpoints. Three checkpoints = three imitators of existing tools, and invites
R3's "just use GBN directly." One conditioned model that spans the axis is the contribution.

---

## Validation

**Two separate claims, validated differently.**

### 1. Endpoint fidelity (supervised — has ground truth)
Given `s = s_WVS`, does the output match WVS? Given `s = s_GBN`, does it match GBN?
- Measure with the edge-alignment descriptor + your standard capacity/spacing metrics.
- Compare against the single-teacher baselines you already have.
- **This alone is NOT a contribution** — it is three imitators in one network.

### 2. Interpolation (unsupervised — no ground truth, this is the contribution)
Feed `s` values the model never saw (0.25, 0.5, 0.75). With no intermediate oracle there is nothing
to compare against, so the evidence is:
- **Monotonicity:** the *measured* descriptor of the output tracks the *requested* `s`. Plot
  requested vs measured; it should be monotonic and roughly linear.
- **Capacity preserved:** density/tone reproduction and spacing quality stay at endpoint level
  across the whole range. Style should change; capacity should not.
- **Visual coherence:** intermediates look like plausible stipples that read as *between* "fills
  regions" and "hugs contours" — not like one endpoint, not like a degraded mixture.

State this limitation honestly in the paper: the middle of the axis is validated by monotonicity,
capacity preservation, and visual coherence, not against a target. That is weaker than supervised
validation but it is what the oracle situation allows — and generalizing beyond two discrete
supervision points is itself the interesting claim.

---

## Failure modes — know these in advance

The model must learn a **continuous axis from only two discrete supervision points**. That may not
happen. Three specific ways it fails:

1. **Snapping.** `s = 0.5` produces something that looks like WVS or like GBN, not between. The
   model treated the descriptor as a switch. Diagnostic: measured descriptor vs requested is a step
   function, not a ramp.
2. **Muddy mixture.** `s = 0.5` looks like neither and worse than both — an incoherent average.
   This is conditional-mean averaging appearing in conditioning space; there is prior form for this
   failure in this project. Diagnostic: capacity/spacing metrics degrade in the middle of the range
   while both endpoints are fine.
3. **Invisible interpolation.** Metrics move monotonically but a human cannot see the difference.
   Interpolation is strictly *less* separated than the endpoints, so if the endpoint gap is small
   the intermediate gap is smaller. Gate 0 partly guards against this; also eyeball intermediates
   directly.

**Stop condition:** if intermediates snap, muddy, or are invisible, this axis does not carry a
contribution. Report it as a negative (two-endpoint conditioning does not induce a continuous style
manifold) and redirect.

---

## Suggested order

1. **Gate 0** — build the edge-alignment descriptor, verify WVS/GBN separate. No training.
2. Generate matched-N paired outputs, compute `s`, assemble `(ρ, s, P)` dataset.
3. Add style conditioning (internal injection), train one model on both oracles.
4. Validate endpoint fidelity first (must match single-teacher baselines).
5. Validate interpolation — monotonicity, capacity preservation, visual coherence.
6. Re-verify G-transfer at G ∈ {32, 48, 64} after adding the style branch.

---

## Relationship to the anisotropy branch

These are two instances of the same principle: **controlling a point-process property that ρ cannot
encode.** Anisotropy controls local spacing *orientation*; style controls arrangement *relative to
image structure*. Anisotropy passed M1–M3 (oracle achievable, clean targets, multimodality holds)
but failed M0 (not legible in dot-only renders) — that is a *representation* problem, not a broken
oracle, and the open question there is output representation (variable dot size/elongation), not
more sampler work.

The style axis has the opposite profile: no new oracle needed, visibly distinguishable, but no
intermediate ground truth. If the style axis is visible and anisotropy is metric-verifiable, a paper
can plausibly carry both as two instances of one principle.
