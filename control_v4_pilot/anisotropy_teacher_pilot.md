# Density-Invariant Point-Process Editing — Teacher Data Pilot

## Purpose

Decide, before any model training, whether we can **generate ground-truth data** for a
non-decomposable stippling control. The whole project rests on one property:

> Same image, same tonal density, same point budget — but a controllable higher-order
> point structure that a grayscale density map cannot encode and that GBN/WVS cannot vary.

If we cannot *generate* clean supervised targets for such a control, we do not build the model.
Data feasibility is the gate. Architecture comes only after.

## The core principle (the decomposition test)

Every candidate control must pass this test:

> After reducing the conditioned input to its density map ρ, do two different prompts still
> produce different point sets?

- **No** → the control is separable ("edit the image / density and run GBN"). **Discard it.**
  This covers density contrast, "remove background," darken a region, dot size, tone edits.
- **Yes** → it is a genuine point-distribution control. **Keep it.**

The surviving axes are higher-order point-process properties: **anisotropy / oriented spacing**,
and (secondarily) **regularity / spectrum**. First attribute under test: **anisotropy under
fixed density**.

## Why anisotropy is the chosen first attribute

- Not encoded by a grayscale density map → non-decomposable by construction.
- Has an existing constructive sampler (Li-style anisotropic dart throwing).
- Density and orientation can be decoupled **algebraically** via the determinant of the
  local metric — so there is no attractive-capacity vs repulsive-spacing gradient fight.
  This is what avoids the collapse observed in the earlier GT-free experiments.
- Hard spacing can be enforced as an **acceptance invariant**, not a loss term.
- Retains multimodality (many valid configs per condition) → justifies diffusion.
- Has a strong classical baseline → honest comparison later.

## Governing constraint (do not violate)

Blue-noise / spacing structure must be **inherited from a clean exemplar** or **enforced by a
constructive hard-core sampler** — never optimized as a soft penalty inside a scalar energy.
A scalar objective lets the optimizer "purchase" lower capacity/alignment loss by destroying
spacing. That failure mode is already confirmed. No gradient balancing, no GradNorm, no
weighting schedule will fix it. Spacing is a hard invariant or it is nothing.

Concretely, every teacher stage must map the feasible set to itself:
- Candidate insertion rejected if spacing fails.
- Any point update backtracked if spacing fails.
- Locked points never moved.
- Displacement bounds applied as projections, not penalties.
- Capacity failure → **reject the target**, never raise an attraction weight.

## The determinant-metric mechanism (why this doesn't collapse)

Desired intensity: `λ(x) = N · ρ(x) / ∫ρ`.

Local metric: `G_u(x) = λ(x) · A_u(x)`, with `A_u(x) ≻ 0`, `det A_u(x) = 1`.
Then `√det G(x) = λ(x)` in 2D.

- `det G` controls **how many** points belong locally (density).
- The determinant-one `A_u` controls **how they are arranged** locally (anisotropy).
- The exclusion ellipse `(p−q)ᵀ G(x) (p−q) < r₀²` has area `π r₀² / √det G = π r₀² / λ(x)`,
  which is **independent of the anisotropy ratio κ**.

So changing anisotropy changes the *shape* of the exclusion zone but not its *area* → it cannot
add or steal points → no capacity gradient competes with spacing. This is the key.

Anisotropic Jacobian field:

```
J(x) = √λ(x) · R_θ(x) · diag( √κ(x), 1/√κ(x) ) · R_θ(x)ᵀ
```

Hard-core acceptance test for a candidate x against each accepted neighbor pᵢ (both directed
checks, because the metric is asymmetric):

```
|| J(x)  · (x − pᵢ) || ≥ r₀
|| J(pᵢ) · (x − pᵢ) || ≥ r₀
```

---

## Milestones and blockers

Milestones are ordered by dependency. **Blocking** = failure stops all downstream work (or
redirects to the named fallback). **Diagnostic** = failure informs but does not halt. No calendar
— progress is gated by milestones, not time.

### Milestone 0 — Legibility (BLOCKING, do first)

The cheapest kill-shot for the highest risk. Anisotropic *point positions alone* may be
perceptually invisible (Li et al. render anisotropy with oriented sticks for this reason). Our
output is dots.

- Generate a handful of clean anisotropic exemplars via density-preserving warp (see M1).
- Render **dots only** — no sticks, no size/shape cues.
- Blind check: can a human distinguish κ = 1 / 1.5 / 2, and read the orientation field?

**Blocker:** if oriented spacing is not visible in pure dots, anisotropy is not a viable control
for this output format. **STOP the anisotropy track.** Fall back to **regularity/spectrum control**
as the first attribute (loose vs strongly-regular spacing is guaranteed visible in dots), or
revisit output representation (variable dot shape/size) before continuing.

Rationale for going first: needs no new sampler, and decides which attribute is the headline.

### Milestone 1 — Oracle achievability (BLOCKING)

Build a guaranteed-clean oracle to test whether the requested capacity + anisotropy are jointly
achievable **at all**, independent of sampler quality.

- Start from a clean isotropic WVS/GBN set `P₀`.
- Apply analytic **density-preserving diffeomorphic warps**:
  - constant area-preserving shear (`det DΦ = 1`),
  - smooth periodic shear,
  - swirl from a divergence-free flow,
  - one field with an isotropic transition region.
- For nonuniform density use the weighted divergence-free flow `v_ψ = ρ_ε⁻¹ ∇^⊥ ψ`.
- Output `P_oracle = Φ(P₀)`: exact N, no collisions (Φ is a diffeomorphism), preserved measure,
  exact correspondence to `P₀`.

**Blocker:** if this guaranteed-clean warped set already fails the capacity threshold, the
requested anisotropy strength and the capacity definition are **incompatible** — the requirement
is wrong, not the implementation. Reduce anisotropy range or revise the capacity metric before
proceeding. Do not attempt to fix this with the sampler.

### Milestone 2 — Constructive sampler (BLOCKING)

Implement Li-style anisotropic dart throwing against the metric `J(x)` above.

```
S = empty
while |S| < N:
    draw x from the metric-volume distribution
    valid = true
    for p in nearby accepted points:
        if || J(x) · (x − p) || < r0: valid = false
        if || J(p) · (x − p) || < r0: valid = false
    if valid: add x to S
```

First-pass settings: periodic square domain (removes boundary effects), N = 1024,
κ ∈ {1, 1.25, 1.5, 2.0}, density clamped away from zero, spatial hash sized by the smallest
Euclidean exclusion radius, r₀ tuned by bisection or a target-count stopping rule, 16–32 seeds
per condition.

Optional capacity refinement (if spacing is good but capacity is short) — CCVT-style balanced
assignment, **not** a capacity force:
1. draw a dense carrier set `Q ∼ ρ`,
2. assign exactly `|Q|/N` carriers to each point under the anisotropic metric,
3. propose moving each point to its assigned centroid,
4. backtrack any move that breaks a hard-core constraint,
5. stop after a few iterations.
All sites get disjoint equal quotas, so they cannot all chase the same edge.

**Blocker:** if the sampler cannot simultaneously satisfy hard-core spacing (zero violations),
exact N, bounded capacity penalty, and commanded orientation on smooth fields, the teacher family
is not buildable — **STOP.**

### Milestone 3 — Multimodality (BLOCKING for the diffusion justification)

For each fixed `(ρ, θ, κ)`, generate K = 8–32 seeds. Condition only on coarse
`(θ(x), log κ(x))` fields and a few statistics — never on absolute phase or which neighbor lies
in which direction.

- `D_style = std_z m(P_z)` should be **small** (targets hit the requested statistics).
- `D_points = median_{z_a≠z_b} W₂(P_{z_a}, P_{z_b})` should be **clearly above the jitter floor**.

**Blocker:** if imposing the anisotropy target collapses the valid set to essentially one
configuration (`D_points → 0`), the conditional task is deterministic — a regression baseline
suffices and the diffusion justification evaporates. Measure this; do not assume it.

### Milestone 4 — Build conditioning and train (only after 0–3 pass)

Everything above is data-side and needs no model change. Cross into training only when M0–M3 all
hold. Then:
- add the region + `(θ, log κ)` control signal injected **internally** (never as an edit to the
  input density map — that would be separable),
- keep density as a hard condition and text/control as a residual,
- text is an **inference-time selector** (prompt → control field), never a training loss on the
  geometry (no CLIP-on-stipple loss),
- **re-validate grid-transfer at G = 64/96 after adding the control branch** — the
  budget-generalization claim depends on full resolution-transferability, and a control branch
  that sneaks in learned absolute positional encodings on the query coordinates will break it
  silently. This is a required post-training validation gate.

---

## Validation metrics (for M1–M3)

**Hard-spacing safety.** For each neighbor pair,
`q_ij = min( ||J(pᵢ)(pⱼ−pᵢ)||, ||J(pⱼ)(pᵢ−pⱼ)|| ) / r₀`; require `q_min ≥ 0.98` (tolerance for
numerical precision only). Also report the 1st percentile of Euclidean normalized NN distance.

**Capacity.** Voronoi mass coefficient of variation; EMD or Sinkhorn to ρ; low-pass KDE error vs
ρ; regional point-count error. Do **not** use isotropic CVT energy as the primary acceptance
metric — intentional anisotropy inflates it even when correct. Initial gate: anisotropy raises
capacity error by ≤ ~10–15% vs the same sampler at κ=1; ultimately set the threshold relative to
WVS/GBN variance on the eval densities.

**Orientation / anisotropy.** Double-angle local statistic
`a_i = | (1/|N_i|) Σ_j w_ij e^{2i(φ_ij − θ_i)} |`. Report median orientation error, anisotropy
strength, monotonicity vs commanded κ, error near field singularities. Initial gate: median
orientation error < 10° on constant/shear fields, < 15° on smooth curved fields; measured
anisotropy monotonic across all four κ levels. Also check `z_ij = J(pᵢ)(pⱼ−pᵢ)` recovers
approximately isotropic blue noise — this confirms the raw anisotropy is the requested metric
deformation and not accidental clumping.

**Multimodality.** As Milestone 3.

**Visual legibility.** As Milestone 0 — the blocking gate.

---

## Scope decisions (locked for v1)

- **Clustering: excluded from v1.** Strict fine-scale capacity and true clustering are
  structurally opposed; clustering would force redefining "density-invariant" (capacity only
  above the cluster scale). Do not let it re-enter as a "nice extra prompt category" — it changes
  the evaluation metric, not just the attribute list.
- **Anisotropy claim scope.** The sampler produces oriented *local pair statistics* — "points
  locally form elongated spacing along the direction," **not** "points trace individual curves /
  streamlines." Curve-tracing is a different, harder problem (curve attraction) and is out of
  scope. Keep the mental model as "locally elongated spacing."
- **Field smoothness.** The local-metric approximation is valid for smooth fields and bounded
  ratio: require `||∇J|| · r_local ≪ ||J||`; smooth `(cos 2θ, sin 2θ)` rather than θ; κ ≤ 2 to
  start; drive κ → 1 where orientation confidence is low; blend toward isotropy across several
  point spacings near singularities.

## Go / no-go summary

Proceed to model training only when all hold:
1. Dots-only output visibly expresses the control (M0).
2. Guaranteed-clean oracle meets the capacity threshold (M1).
3. Exact point count, zero hard-core violations, capacity close to isotropic teacher,
   requested anisotropy quantitatively recovered (M2).
4. Different seeds remain coordinate-wise diverse (M3).

If the warped oracle passes but the dart-thrower fails → target family is feasible, the sampler
needs work. If the oracle itself fails → reduce anisotropy range or revise the capacity
definition. If M0 fails → switch first attribute to regularity/spectrum.

## References (for implementation lookup)

- Li et al. 2010, anisotropic blue noise / anisotropic sampling (HKUST) — the sampler.
- BNOT (Caltech) — capacity-constrained blue noise via optimal transport (note: power cells,
  not Voronoi — relevant to the separate capacity-metric fix).
- Heck et al. 2013 — blue-noise sampling from target pair-correlation functions.
- Yuksel — weighted sample elimination (exact-count Poisson-disk alternative).
