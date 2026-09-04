"""Stage 1: analyse one method's point sets.

Run once per target folder (uncomment the TARGET_DIR you want), mirroring
quantitative_advance_metrics_stage_1.py. Both experiments built by stage 0 are handled; which
one runs is decided per image by its filename prefix.

TEST 1 -- uniform sources -> POWER SPECTRUM
-------------------------------------------
The spectrum is evaluated DIRECTLY from the point coordinates,

    S(f) = sum_j exp(-2*pi*i * f . x_j),      P(f) = |S(f)|^2 / n

rather than by rasterising onto a pixel grid and running an FFT. Rasterising snaps every point
to a cell centre, injecting quantisation noise into exactly the high-frequency band under
study, and collisions silently drop points. (utils.stippling_metrics_advance.plot_visual_m5_-
spectrum does rasterise, and also omits an fftshift so its radial bins mix unrelated
frequencies; it is deliberately not used here.)

Reported: radially averaged power -- which should dip to ~0 at low frequency, rise through a
peak near the principal frequency and settle at 1 -- and anisotropy in dB, the angular variance
within each radial band, where low values mean a clean isotropic annulus.

TEST 2 -- patterned sources -> PER-REGION PAIR CORRELATION
----------------------------------------------------------
An FFT cannot be used here: a patterned image is non-stationary, and cropping a region applies
a boxcar window whose sinc transform smears the annulus being measured. The pair correlation
function normalises distances by the local mean spacing and so remains valid under varying
density.

Region borders are handled by the standard correction: only points in the INTERIOR of a region
act as centres, and the neighbour radius is capped by the margin, so every counted pair lies
within one density regime. Without this, points near a boundary would be paired with neighbours
across it at a different density, biasing g(r).

Output
------
    <target>_spectral/spectral_g<level>.npz      TEST 1, per grey level
    <target>_spectral/pcf_<pattern>_<region>.npz TEST 2, per region
    <target>_spectral/summary.json               scalar descriptors
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = r"experiments/outputs/spectral_analysis"

# Default input folders
SOURCE_DIR = r"experiments/outputs/spectral_analysis/source"
TARGET_DIR = r"experiments/outputs/spectral_analysis/target_WVS_1024"
# TARGET_DIR = r"experiments/outputs/spectral_analysis/target_BNOT_1024"
# TARGET_DIR = r"experiments/outputs/spectral_analysis/target_GBN_1024"
# TARGET_DIR = r"experiments/outputs/spectral_analysis/target_CN-WVS_1024"
# TARGET_DIR = r"experiments/outputs/spectral_analysis/target_CN-GBN_1024"

# TEST 1 -- half-width of the integer frequency grid. The principal frequency of n points in
# the unit square is sqrt(n) (32 for n=1024), so 64 covers ~2x that per axis.
FREQ_HALF_WIDTH = 64
RADIAL_BINS = 96
RADIAL_MAX = 3.0
ANGULAR_SECTORS = 16

# TEST 2 -- PCF range in units of the local mean spacing, and bin count.
PCF_MAX = 3.0
PCF_BINS = 60

UNIFORM_RE = re.compile(r"uniform_g(\d+)_r(\d+)")
PATTERN_RE = re.compile(r"pattern_([A-Za-z0-9]+)_r(\d+)")


def parse_args():
    p = argparse.ArgumentParser(description="Analyse one method's point sets on the stage-0 sources")
    p.add_argument("target", nargs="?", default=TARGET_DIR)
    p.add_argument("--source", default=SOURCE_DIR)
    p.add_argument("--base", default=BASE_DIR, help="Folder holding manifest.json")
    p.add_argument("--output", default=None, help="Default: {target}_spectral")
    p.add_argument("--freq-half-width", type=int, default=FREQ_HALF_WIDTH)
    p.add_argument("--radial-bins", type=int, default=RADIAL_BINS)
    p.add_argument("--radial-max", type=float, default=RADIAL_MAX)
    p.add_argument("--sectors", type=int, default=ANGULAR_SECTORS)
    p.add_argument("--pcf-max", type=float, default=PCF_MAX)
    p.add_argument("--pcf-bins", type=int, default=PCF_BINS)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_points(npy_path, margin_frac=0.0):
    """Points in [0,1]^2, with the source image's white margin cropped away.

    Stage 0 insets the pattern by a white margin so that WVS/GBN's per-image min-max
    density stretch is the identity (see its module docstring). That margin carries no
    mass, so a correct sampler puts nothing there; any stray point in it is dropped, and
    the rest are renormalised onto the content square so the unit-square assumption behind
    the spectrum and PCF maths stays exact.
    """
    pts = np.load(npy_path).astype(np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points in {npy_path}, got {pts.shape}")
    if margin_frac > 0.0:
        lo, hi = margin_frac, 1.0 - margin_frac
        keep = ((pts[:, 0] >= lo) & (pts[:, 0] < hi)
                & (pts[:, 1] >= lo) & (pts[:, 1] < hi))
        pts = (pts[keep] - lo) / (hi - lo)
    return pts


# ── TEST 1: power spectrum ───────────────────────────────────────────────────

def power_spectrum(pts, F):
    """|sum_j exp(-2 pi i f . x_j)|^2 / n on [-F, F]^2, DC removed.

    S = A @ B.T with A[a,j] = exp(-2 pi i fx_a x_j), B[b,j] = exp(-2 pi i fy_b y_j), since
    sum_j A[a,j] B[b,j] is exactly the 2-D sum -- one small matrix product instead of an
    O(n (2F+1)^2) sweep.
    """
    n = len(pts)
    f = np.arange(-F, F + 1, dtype=np.float64)
    A = np.exp(-2j * np.pi * np.outer(f, pts[:, 0]))
    B = np.exp(-2j * np.pi * np.outer(f, pts[:, 1]))
    P = (np.abs(A @ B.T) ** 2) / max(n, 1)
    P[F, F] = 0.0
    return P


def radial_and_anisotropy(P, F, n, n_bins, r_max, sectors):
    f = np.arange(-F, F + 1, dtype=np.float64)
    fx, fy = np.meshgrid(f, f, indexing="ij")
    r = np.sqrt(fx ** 2 + fy ** 2) / np.sqrt(max(n, 1))
    theta = np.arctan2(fy, fx) % np.pi           # the spectrum is centrally symmetric
    sect = np.minimum((theta / np.pi * sectors).astype(int), sectors - 1)

    edges = np.linspace(0, r_max, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    radial = np.full(n_bins, np.nan)
    aniso = np.full(n_bins, np.nan)
    idx = np.digitize(r.ravel(), edges) - 1
    Pf, sf = P.ravel(), sect.ravel()
    for b in range(n_bins):
        m = idx == b
        if m.sum() < sectors:
            continue
        vals = Pf[m]
        radial[b] = vals.mean()
        sm = np.array([vals[sf[m] == s].mean() if (sf[m] == s).any() else np.nan
                       for s in range(sectors)])
        sm = sm[np.isfinite(sm)]
        if len(sm) > 1 and radial[b] > 1e-12:
            aniso[b] = 10.0 * np.log10(max(sm.var(ddof=1) / (radial[b] ** 2), 1e-12))
    return centres, radial, aniso


def spectrum_descriptors(centres, radial):
    ok = np.isfinite(radial)
    if ok.sum() < 4:
        return {}
    c, v = centres[ok], radial[ok]
    low, tail = c < 0.5, c > 2.0
    i = int(np.argmax(v))
    return {"low_freq_power": float(v[low].mean()) if low.any() else float("nan"),
            "peak_power": float(v[i]), "peak_freq": float(c[i]),
            "tail_deviation": float(np.abs(v[tail] - 1.0).mean()) if tail.any() else float("nan")}


# ── TEST 2: per-region pair correlation ──────────────────────────────────────

def region_pcf(pts, box, r_max, n_bins):
    """g(r) inside one constant-density region, with a border correction.

    Only points at least `margin` inside the region act as centres, and the neighbour radius is
    capped at that margin, so every counted pair lies within the region and therefore within a
    single density regime. Distances are expressed in units of the region's mean spacing
    s = sqrt(area / n), in which the Poisson expectation per centre over an annulus is simply
    pi((u+du)^2 - u^2) -- independent of the density, so regions of different greys are directly
    comparable.
    """
    from scipy.spatial import cKDTree
    x0, y0, x1, y1 = box
    inside = ((pts[:, 0] >= x0) & (pts[:, 0] < x1) &
              (pts[:, 1] >= y0) & (pts[:, 1] < y1))
    sub = pts[inside]
    n = len(sub)
    area = (x1 - x0) * (y1 - y0)
    if n < 16:
        return None, None, n
    spacing = np.sqrt(area / n)
    margin = r_max * spacing
    if margin >= 0.5 * min(x1 - x0, y1 - y0):
        return None, None, n              # region too small for an unbiased estimate

    core = ((sub[:, 0] >= x0 + margin) & (sub[:, 0] < x1 - margin) &
            (sub[:, 1] >= y0 + margin) & (sub[:, 1] < y1 - margin))
    centres_pts = sub[core]
    if len(centres_pts) < 8:
        return None, None, n

    edges = np.linspace(0, r_max, n_bins + 1)
    tree = cKDTree(sub)
    hist = np.zeros(n_bins)
    for p in centres_pts:
        d = np.linalg.norm(sub[tree.query_ball_point(p, margin)] - p, axis=1) / spacing
        hist += np.histogram(d[d > 1e-12], bins=edges)[0]
    expected = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)      # Poisson, per centre
    g = hist / max(len(centres_pts), 1) / np.maximum(expected, 1e-12)
    return 0.5 * (edges[1:] + edges[:-1]), g, n


def pcf_descriptors(centres, g):
    ok = np.isfinite(g)
    if ok.sum() < 4:
        return {}
    c, v = centres[ok], g[ok]
    near, tail = c < 0.6, c > 1.8
    return {"exclusion_leak": float(v[near].mean()) if near.any() else float("nan"),
            "peak": float(v.max()), "peak_r": float(c[int(np.argmax(v))]),
            "tail_deviation": float(np.abs(v[tail] - 1.0).mean()) if tail.any() else float("nan")}


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    target = Path(args.target)
    out_dir = Path(args.output) if args.output else Path(f"{args.target}_spectral")

    npys = sorted(target.glob("*.npy"))
    if not npys:
        print(f"ERROR: no .npy point sets found in {target}")
        return 2

    manifest = {}
    mpath = Path(args.base) / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
    # Regions are keyed by pattern, since quad4 and checker define different quadrants.
    # A plain list is accepted too, for manifests written before multi-pattern support.
    # Stage 0 pads every source image with a white margin (so WVS/GBN's per-image density
    # stretch is the identity); crop it back off here. Absent/0 -> no cropping, which keeps
    # manifests written before the margin was introduced working unchanged.
    _res = int(manifest.get("resolution", 0) or 0)
    _margin_px = int(manifest.get("margin_px", 0) or 0)
    margin_frac = (_margin_px / _res) if (_res > 0 and _margin_px > 0) else 0.0
    if margin_frac > 0.0:
        print(f"cropping a {_margin_px}px white margin ({margin_frac:.5f} of the image) "
              f"and renormalising")

    regions_raw = manifest.get("regions", {})
    regions_by_pattern = regions_raw if isinstance(regions_raw, dict) else {}
    legacy_regions = regions_raw if isinstance(regions_raw, list) else []

    uniform, pattern = {}, {}
    for f in npys:
        m = UNIFORM_RE.match(f.stem)
        if m:
            uniform.setdefault(int(m.group(1)), []).append(f)
            continue
        m = PATTERN_RE.match(f.stem)
        if m:
            pattern.setdefault(m.group(1), []).append(f)

    print(f"Target: {target}")
    print(f"Output: {out_dir}")
    if uniform:
        print(f"TEST 1 uniform grey levels: " +
              ", ".join(f"{g}:{len(v)}" for g, v in sorted(uniform.items())))
    if pattern:
        print(f"TEST 2 patterns: " + ", ".join(f"{k}:{len(v)}" for k, v in pattern.items()))
        if not regions_by_pattern and not legacy_regions:
            print("  WARNING: manifest.json has no region definitions; TEST 2 will be skipped.")
    if not uniform and not pattern:
        print("ERROR: no recognised filenames (uniform_g*/pattern_*) in the target folder")
        return 2
    if args.dry_run:
        print("DRY RUN: nothing computed")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ── TEST 1 ──
    F = args.freq_half_width
    for grey in sorted(uniform):
        files = sorted(uniform[grey])
        acc, n_pts = None, None
        for i, f in enumerate(files, 1):
            pts = load_points(f, margin_frac)
            n_pts = len(pts)
            P = power_spectrum(pts, F)
            acc = P if acc is None else acc + P
            print(f"  [T1] grey {grey:3d} [{i}/{len(files)}] {f.stem}  n={n_pts}", flush=True)
        mean_P = acc / len(files)
        centres, radial, aniso = radial_and_anisotropy(
            mean_P, F, n_pts, args.radial_bins, args.radial_max, args.sectors)
        d = spectrum_descriptors(centres, radial)
        np.savez_compressed(out_dir / f"spectral_g{grey:03d}.npz",
                            mean_spectrum=mean_P.astype(np.float32),
                            radial_freq=centres, radial_power=radial, anisotropy_db=aniso,
                            grey=grey, n_points=n_pts, realizations=len(files),
                            freq_half_width=F)
        summary[f"spectrum_g{grey:03d}"] = {"test": 1, "grey": grey, "rho": 1.0 - grey / 255.0,
                                            "n_points": int(n_pts), "realizations": len(files), **d}
        print(f"  -> grey {grey:3d}: " + "  ".join(f"{k}={v:.4f}" for k, v in d.items()))

    # ── TEST 2 ──
    for pat, files in pattern.items():
        regions = regions_by_pattern.get(pat, legacy_regions)
        if not regions:
            print(f"  [T2] {pat}: no region definitions in the manifest -- skipped")
            continue
        for reg in regions:
            if not reg.get("analysable", True):
                continue
            acc, cent, counts, used = None, None, [], 0
            for f in sorted(files):
                pts = load_points(f, margin_frac)
                c, g, n_in = region_pcf(pts, reg["box"], args.pcf_max, args.pcf_bins)
                counts.append(n_in)
                if g is None:
                    continue
                acc = g if acc is None else acc + g
                cent = c
                used += 1
            if acc is None:
                print(f"  [T2] {pat}/{reg['name']}: too few points "
                      f"(mean {np.mean(counts):.0f}) -- skipped")
                continue
            mean_g = acc / max(used, 1)
            d = pcf_descriptors(cent, mean_g)
            np.savez_compressed(out_dir / f"pcf_{pat}_{reg['name']}.npz",
                                r=cent, g=mean_g, grey=reg["grey"], rho=reg["rho"],
                                region=reg["name"], pattern=pat,
                                mean_points=float(np.mean(counts)), realizations=used)
            summary[f"pcf_{pat}_{reg['name']}"] = {
                "test": 2, "pattern": pat, "region": reg["name"], "grey": reg["grey"],
                "rho": reg["rho"], "mean_points": float(np.mean(counts)),
                "realizations": used, **d}
            print(f"  -> [T2] {pat}/{reg['name']} (rho {reg['rho']:.2f}, "
                  f"{np.mean(counts):.0f} pts): " + "  ".join(f"{k}={v:.4f}" for k, v in d.items()))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)} result(s) + summary.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
