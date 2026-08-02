# Gate 0 - WVS <-> GBN edge-alignment descriptor (BLOCKING kill-shot)

This is step 1 of the style-axis experiment in `style_axis_wvs_gbn_plan.md`. Before any
model/dataset/training work, we must prove that WVS ("fills regions") and GBN ("hugs contours")
**separate cleanly** on an *edge/structure-alignment* descriptor. If they do not separate, the
visible difference is something else and the axis carries no contribution -- STOP and
re-identify the property. This uses data that already exists and needs no training.

The descriptor built here is also the **evaluation metric** for the whole downstream experiment
(endpoint fidelity + interpolation monotonicity), which is why it lives in a clean, reusable,
torch-free module.

## Files

| File | Purpose |
| --- | --- |
| `edge_alignment.py` | Reusable descriptor library (numpy/scipy/PIL only, no torch). |
| `gate0_selftest.py` | Data-free synthetic sanity check of the descriptor math. |
| `gate0_run.py` | Runs the descriptor on the real WVS/GBN icon set and prints the verdict. |
| `GATE0_README.md` | This file. |

## What it measures (and what it deliberately does NOT)

Point position **relative to image structure**, normalized against the tone density
`rho = clip(1 - image, 0, 1)` (the project's canonical ink probability, `smart_init.py:18`) so
it captures *arrangement*, not *tone*. It is **not** a spectral / pair-correlation measure --
those are rho-blind and isotropically averaged, would find WVS ~= GBN, and would wrongly kill a
visible axis.

Primary descriptor `s_raw` (rho-normalized edge over-representation):
- `expected_frac` = fraction of rho mass inside the high-gradient edge band (what tone predicts).
- `observed_frac` = fraction of oracle points landing in the edge band.
- `s_raw = observed_frac / expected_frac`. GBN >> 1 (crowds edges beyond tone); WVS ~= 1.

Corroborating descriptors: distance-to-nearest-edge (mean/median px), `frac_within_k` px for
k in {1,2,4,8}, and mean gradient magnitude sampled at points.

## Data layout (key fact)

WVS and GBN share the **same source images** (same rho); only the target stipple PNGs differ:
```
<wvs-root>/source/<stem>.png   <wvs-root>/target/<stem>.png     (WVS oracle)
<gbn-root>/source/<stem>.png   <gbn-root>/target/<stem>.png     (GBN oracle)
```
Defaults mirror `control_v4/train_control.py`:
`icons-50_512_WVS` (WVS) and `icons-50_512_GBN` (GBN). Point locations are recovered from the target
PNGs by dot-centroid detection (same method as `train_control.py:extract_points_from_target`).

## How to run

1) Local math check (no cluster data required):
```
python control_v4_mix/gate0_selftest.py
```
Expect `SELF-TEST PASSED` (synthetic contour set reads as high-alignment, interior-fill set as
rho-baseline).

2) Real verdict (run on the cluster where the icon data lives):
```
python control_v4_mix/gate0_run.py --point-source both \
    --out control_v4_mix/gate0_outputs
```
Override data locations if needed:
`--wvs-root <dir> --gbn-root <dir>` (each must contain `source/` and `target/`), or point at the
individual dirs with `--wvs-source/--wvs-target/--gbn-source/--gbn-target`. For the gridded
variant, offsets default to `<root>/processed_offsets` (override with `--wvs-offsets/--gbn-offsets`).

Useful flags: `--edge sobel|canny`, `--edge-quantile 0.80`, `--min-area N` (drop noise blobs),
`--limit N` (quick smoke test on N icons), `--n-overlays N` (how many overlay panels to render).

## Outputs (under `--out`, default `control_v4_mix/gate0_outputs/`)

- `per_icon.csv` - one row per (stem, oracle, point_source) with every descriptor + point count.
- `summary.json` - per-descriptor WVS/GBN means+std, aligned Cohen's d, paired GBN>WVS agreement,
  a near-non-overlap flag, and the `s` normalization (WVS median -> 0, GBN median -> 1).
- `plots/` - paired per-icon scatter + distribution histograms (for `s_raw` and `dist_mean`),
  and overlay panels (source + edge band + points) for a few icons, WVS vs GBN side by side.

## Reading the verdict

The console ends with a `GATE 0 VERDICT` block for the raw (centroid) point source, using the
primary descriptor `s_raw`:
- **PASS** = paired GBN>WVS agreement >= 0.90 AND aligned Cohen's d >= 0.80. Ideally the
  distributions are also near-non-overlapping (`clean/non-overlapping = True`), not just a mean
  gap. -> green-light Phase 2 (paired `(rho, s, P)` dataset, then internal style conditioning).
- **BLOCKER** = weak/inconsistent separation. STOP. Check whether *any* descriptor in the table
  separated, eyeball the overlay plots, and re-identify the visual property before training.

Also inspect the **raw-vs-gridded** comparison (from `--point-source both`): if raw separates but
the 32x32 OT-gridded points collapse, that is an early **representation-ceiling** warning (the
anisotropy-M0 failure mode) - the axis may be real yet not expressible at grid resolution. Surface
this now, before any training.

## Next (only if Gate 0 PASSES)

Phase 2+: paired `(rho, s, P)` dataset reading both oracles; internal `s` injection
(FiLM/additive at the conditioning sites, **not** a rho edit) in a forked `DynamicControlNet`;
joint training on both oracles; endpoint-fidelity + interpolation-monotonicity validation;
G-transfer re-verification. All deferred behind this verdict.
