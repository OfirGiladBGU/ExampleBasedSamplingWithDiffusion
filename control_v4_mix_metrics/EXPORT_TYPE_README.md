# `--export_png` / `--export_npy` for the GBN / WVS / BNOT generators, and the BNOT crash fix

Two independent changes:

1. **`--export_png` / `--export_npy`** on all three dataset generators, so targets can be written as
   exact continuous coordinates as well as (or instead of) a rasterised PNG. Two independent
   booleans, matching the convention in `experiments/control_data_gen.py`.
2. **Four finiteness fixes in the BNOT C++ solver**, which is what made 36/10000 icons fail.

**Both default to `True`**: every re-run writes the rasterised `target/.../<stem>.png` and the exact
`target/.../<stem>.npy` side by side in the same folder. `.npy` is outside every consumer's
`VALID_EXT` set (including `train_control.py:222`), so the extra files are invisible to the existing
image walks.

---

## 1. `--export_type`

| generator | file |
| --- | --- |
| GBN | `GaussianBlueNoise/scripts/gbn_data_gen.py` |
| WVS | `Rougier-2017/src/stippler_data_gen.py` |
| BNOT | `BNOT_new/bnot_data_gen.py` |

**Each generator is self-contained.** There is no shared module: every one defines its own
`points_to_canonical()` and `save_points_npy()`, and declares `--export_png` / `--export_npy` in its
own argument parser (via `argparse.BooleanOptionalAction`, so `--no-export_png` works) alongside its
other flags. The three repos stay independent -- editing one stippler cannot affect another.

```bash
python scripts/gbn_data_gen.py                       # png + npy, by default
python src/stippler_data_gen.py  --no-export_npy     # PNG only (previous behaviour)
python bnot_data_gen.py          --no-export_png     # coordinates only
```

Output lands beside the target with the same stem and tree: `target/Icons-50/<class>/<stem>.npy`.

### The coordinate convention

`(N, 2) float64`, **x then y**, in **[0, 1]**, **y increasing downward**.

This is exactly what `control_v4/train_control.py:extract_points_from_target` returns (`[cx/w, cy/h]`),
so an `.npy` is a drop-in replacement for centroid detection with no per-oracle special-casing.

**Coordinates, not 32×32 offsets** — deliberately. The offsets grid is *derived*:
`data/Transforms.to_image_optimal_transport` assigns N points to G·G cells and stores a per-cell
displacement, so the representation depends on G. Storing offsets would bake G=32 into the dataset and
force a full regeneration to change grid size. `train_control.ensure_offsets_dir` already does that
conversion once.

Each solver works in its own space, and each already contains the mapping to pixels inside its
rasteriser. The `to_canonical_*` helpers reuse **that same mapping minus the rounding** — the rounding
being the only lossy step:

| solver | native space | conversion |
| --- | --- | --- |
| GBN | `[0,1]`, y **up** | flip y |
| WVS | density-array pixels, y **up** (density was row-flipped), `1/zoom` scale | scale, divide by W/H, flip y |
| BNOT | centred `[-dx,dx]×[-dy,dy]`, y **up** | `0.5(x+dx)/dx`, `0.5(dy−y)/dy` |

Verified against each rasteriser on 1024 random points: GBN and BNOT reproduce the original pixel
indices **exactly** (max delta 0). WVS differs by **≤1 px**, because its rasteriser uses `rint(x)` as
the pixel index while the canonical form uses `x/W` — a half-pixel convention difference. The `.npy`
is the more correct of the two there, and it is the one that agrees with `extract_points_from_target`.

Each generator's `save_points_npy` **asserts** the point count rather than repairing it. A short export means the solver
did not place the requested number of points, and silently padding it — which the training loader does,
with *uniform random* points ([train_control.py:315-317](ExampleBasedSamplingWithDiffusion/control_v4/train_control.py#L315-L317)) —
would inject white noise into a target whose point statistics are the entire object of study.

### When npy actually matters (measured)

Storing via a 512² one-pixel-dot PNG and recovering by centroid detection costs, at N=1024:
`nn_cv` −1.9%, `cap_cv` −0.2%, `aniso` −3.7% for the clumpiest point process, and ≤1% for a regular
one. Against between-oracle differences of tens to hundreds of percent, that is negligible — **PNG is
adequate at N=1024.** It stops being adequate as N grows, because expected pixel coincidences scale as
N²/(2·512²) and relative quantisation error as √N/512:

| N | spacing @512 | expected coincidences |
| --- | --- | --- |
| 1024 | 16.0 px | ~1 |
| 3136 | 9.1 px | ~10 |
| 12544 | 4.6 px | ~154 |

So `npy` matters increasingly with N, and both exports are on by default because the `.npy` costs one
small file per icon and removes the question entirely.

### On stray points in white areas

Measured over 150 icons, points landing on near-white pixels (of 1024): ρ<0.02 — gbn 5.8, wvs 1.8,
bnot 2.2; but ρ<0.25 — gbn 177, wvs 217, bnot 200. The second column is **not** error: sources are
LANCZOS-resized to 512, so edges carry a real anti-aliased ρ gradient and a density-following sampler
belongs there. Only the ~0.4% at ρ<0.02 looks like a genuine stray.

They are deliberately **kept**. N is forced to G² = 1024 by the offsets tensor — there is no
empty-cell encoding — so dropping points does not lower the budget, it triggers
`train_control.py:315-317`, which pads back to 1024 with points drawn **uniformly over the whole
canvas**. That would replace a few mildly-misplaced points with maximally-misplaced ones, hand the OT
assignment huge displacements, and inject white-noise clumping into the exact statistics this project
measures. Keeping them costs ~0.4% imitation error and keeps N exact.

---

## 2. BNOT crash fix (36/10000 icons)

All 36 failures were identical:

```
ibnot_new_cli error: CGAL ERROR: assertion violation!
Expr: dexp != 2047      File: CGAL/Mpzf.h
Explanation: Creating an Mpzf from infinity or NaN.
```

`dexp == 2047` is the all-ones IEEE double exponent — a **non-finite double reached CGAL's exact
predicates**. Not duplicate sites: `sites.cpp:21 generate_random_sites_based_on_image` uses continuous
rejection sampling and weights start at 0.

Two unguarded paths produced it, and two more places let it propagate:

| # | file | problem |
| --- | --- | --- |
| 1 | `primitives.h` `compute_centroid` | `if (sum_area == 0.0)` does **not** protect the division. A degenerate sliver power cell gives `sum_area` a *denormal* nonzero value (~1e-320) and `sum_vector / sum_area` overflows to ±Inf. |
| 2 | `convex_polygon.h` `compute_centroid` | same `== 0.0` guard, same overflow. |
| 3 | `optimizer.cpp` `solve_linear_system` | SparseQR can report success and still return Inf/NaN for a singular Laplacian; `x` was never checked. |
| 4 | `init.cpp` `update_positions` / `update_weights` | neither validated its input. `m_domain.clamp()` **propagates** NaN (all comparisons against NaN are false), and `update_weights` subtracts `compute_mean(weights)` — so **one** non-finite weight poisons **every** weight, which is why failure was abrupt and total rather than localised. |

Fixes: replace both `== 0.0` guards with `!(sum_area > 1e-30)` plus an explicit `std::isfinite`
check on the result; reject a non-finite solve in `solve_linear_system` (callers already treat
`false` as "leave weights untouched"); and refuse to store non-finite positions/weights in
`init.cpp`. `<cmath>` added to all four files.

### Rebuild

```bash
cd BNOT_new/ibnot_cli/build
cmake --build . -j            # or: cmake .. && make -j
```

Then re-run only the 36 failures. If any still fail, they are a genuinely degenerate configuration
rather than a numerical accident — retry those with a different `--seed`
(`main_cli.cpp:258 std::srand(options.seed)` fully determines the initial site placement, so a new
seed re-rolls the whole trajectory).

### Backups

Originals are preserved next to each patched file as `*.bak_preexport`.
