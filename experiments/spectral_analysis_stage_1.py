"""Stage 1: compute the power spectrum of one method's point sets.

Run once per target folder (uncomment the TARGET_DIR you want), mirroring
quantitative_advance_metrics_stage_1.py.

What is computed
----------------
The spectrum is evaluated DIRECTLY from the point coordinates,

    S(f) = sum_j exp(-2*pi*i * f . x_j),      P(f) = |S(f)|^2 / n

rather than by rasterising the points onto a pixel grid and running an FFT. Rasterising snaps
every point to a cell centre, which injects quantisation noise into exactly the high-frequency
band the analysis is about, and collisions silently drop points. (Note in passing that
utils.stippling_metrics_advance.plot_visual_m5_spectrum does rasterise, and additionally omits
an fftshift, so its radial bins mix unrelated frequencies -- it is not used here.)

From P(f) we report the two standard blue-noise diagnostics:

  * radially averaged power -- should dip to ~0 at low frequency (no clumping at large scales),
    rise through a peak near the principal frequency, and settle at 1.
  * anisotropy, in dB -- the angular variance within each radial band. Low values mean the
    spectrum is a clean isotropic annulus rather than a structured/directional pattern.

Averaging
---------
A single point set's periodogram is essentially noise. Spectra are averaged over the repeated
realizations that stage 0 emitted per grey level, which is what produces the smooth profiles
reported in the literature.

Output
------
    <target>_spectral/spectral_g<level>.npz     mean 2-D spectrum, radial profile, anisotropy
    <target>_spectral/summary.json              scalar descriptors per grey level
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
# TARGET_DIR = r"experiments/outputs/spectral_analysis/target_CN_1024"

# Half-width of the integer frequency grid. The principal frequency of n points in the unit
# square is sqrt(n) (32 for n=1024), so 64 covers ~2x that in each axis -- enough to show the
# low-frequency dip, the peak, and the flat tail.
FREQ_HALF_WIDTH = 64

# Radial bins over |f|, in units of the principal frequency sqrt(n).
RADIAL_BINS = 96
RADIAL_MAX = 3.0

# Angular sectors used for the anisotropy estimate.
ANGULAR_SECTORS = 16

STEM_RE = re.compile(r"uniform_g(\d+)_r(\d+)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute power spectra for one method's point sets on uniform densities"
    )
    p.add_argument("target", nargs="?", default=TARGET_DIR,
                   help=f"Folder of <stem>.npy point sets (default: {TARGET_DIR})")
    p.add_argument("--source", default=SOURCE_DIR,
                   help=f"Folder of uniform source images (default: {SOURCE_DIR})")
    p.add_argument("--output", default=None,
                   help="Output folder (default: {target}_spectral)")
    p.add_argument("--freq-half-width", type=int, default=FREQ_HALF_WIDTH)
    p.add_argument("--radial-bins", type=int, default=RADIAL_BINS)
    p.add_argument("--radial-max", type=float, default=RADIAL_MAX)
    p.add_argument("--sectors", type=int, default=ANGULAR_SECTORS)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_points(npy_path):
    pts = np.load(npy_path).astype(np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points in {npy_path}, got {pts.shape}")
    return pts


def power_spectrum(pts, F):
    """|sum_j exp(-2 pi i f . x_j)|^2 / n on the integer grid [-F, F]^2, DC removed.

    Uses S = A @ B.T with A[a,j] = exp(-2 pi i fx_a x_j) and B[b,j] = exp(-2 pi i fy_b y_j),
    since sum_j A[a,j] B[b,j] is exactly the 2-D sum. That keeps an otherwise O(n * (2F+1)^2)
    evaluation to a single small matrix product.
    """
    n = len(pts)
    f = np.arange(-F, F + 1, dtype=np.float64)
    A = np.exp(-2j * np.pi * np.outer(f, pts[:, 0]))
    B = np.exp(-2j * np.pi * np.outer(f, pts[:, 1]))
    S = A @ B.T
    P = (np.abs(S) ** 2) / max(n, 1)
    P[F, F] = 0.0                      # drop DC: it only encodes the point count
    return P


def radial_and_anisotropy(P, F, n, n_bins, r_max, sectors):
    """Radially averaged power and per-band anisotropy (dB), in units of sqrt(n)."""
    f = np.arange(-F, F + 1, dtype=np.float64)
    fx, fy = np.meshgrid(f, f, indexing="ij")
    principal = np.sqrt(max(n, 1))
    r = np.sqrt(fx ** 2 + fy ** 2) / principal
    theta = np.arctan2(fy, fx) % np.pi                 # spectrum is centrally symmetric
    sect = np.minimum((theta / np.pi * sectors).astype(int), sectors - 1)

    edges = np.linspace(0, r_max, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    radial = np.full(n_bins, np.nan)
    aniso = np.full(n_bins, np.nan)

    idx = np.digitize(r.ravel(), edges) - 1
    Pf, sf = P.ravel(), sect.ravel()
    for b in range(n_bins):
        m = idx == b
        if m.sum() < sectors:                           # too few cells to speak of an average
            continue
        vals = Pf[m]
        radial[b] = vals.mean()
        # anisotropy: variance of the per-sector means, normalised by the band mean
        sm = np.array([vals[sf[m] == s].mean() if (sf[m] == s).any() else np.nan
                       for s in range(sectors)])
        sm = sm[np.isfinite(sm)]
        if len(sm) > 1 and radial[b] > 1e-12:
            aniso[b] = 10.0 * np.log10(max(sm.var(ddof=1) / (radial[b] ** 2), 1e-12))
    return centres, radial, aniso


def descriptors(centres, radial):
    """Scalar summaries of the radial profile, for the stage-2 table."""
    ok = np.isfinite(radial)
    if ok.sum() < 4:
        return {}
    c, v = centres[ok], radial[ok]
    low = c < 0.5                                       # the low-frequency dip
    tail = c > 2.0                                      # should have settled at 1
    peak_i = int(np.argmax(v))
    return {
        "low_freq_power": float(v[low].mean()) if low.any() else float("nan"),
        "peak_power": float(v[peak_i]),
        "peak_freq": float(c[peak_i]),
        "tail_deviation": float(np.abs(v[tail] - 1.0).mean()) if tail.any() else float("nan"),
    }


def main():
    args = parse_args()
    target = Path(args.target)
    out_dir = Path(args.output) if args.output else Path(f"{args.target}_spectral")

    npys = sorted(target.glob("*.npy"))
    if not npys:
        print(f"ERROR: no .npy point sets found in {target}")
        return 2

    groups = {}
    for f in npys:
        m = STEM_RE.match(f.stem)
        if not m:
            print(f"  skipping unrecognised stem: {f.name}")
            continue
        groups.setdefault(int(m.group(1)), []).append(f)

    if not groups:
        print(f"ERROR: no files matching uniform_g<level>_r<idx> in {target}")
        return 2

    print(f"Target: {target}")
    print(f"Output: {out_dir}")
    print(f"Grey levels found: {sorted(groups)}  "
          f"({', '.join(f'{g}:{len(v)}' for g, v in sorted(groups.items()))} realizations)")

    if args.dry_run:
        print("DRY RUN: no spectra computed")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    F = args.freq_half_width
    summary = {}

    for grey in sorted(groups):
        files = sorted(groups[grey])
        acc, n_pts = None, None
        for i, f in enumerate(files, 1):
            pts = load_points(f)
            n_pts = len(pts)
            P = power_spectrum(pts, F)
            acc = P if acc is None else acc + P
            print(f"  grey {grey:3d} [{i}/{len(files)}] {f.stem}  n={n_pts}", flush=True)
        mean_P = acc / len(files)

        centres, radial, aniso = radial_and_anisotropy(
            mean_P, F, n_pts, args.radial_bins, args.radial_max, args.sectors)
        d = descriptors(centres, radial)

        np.savez_compressed(
            out_dir / f"spectral_g{grey:03d}.npz",
            mean_spectrum=mean_P.astype(np.float32),
            radial_freq=centres, radial_power=radial, anisotropy_db=aniso,
            grey=grey, n_points=n_pts, realizations=len(files),
            freq_half_width=F,
        )
        summary[f"g{grey:03d}"] = {
            "grey": grey, "rho": 1.0 - grey / 255.0, "n_points": int(n_pts),
            "realizations": len(files), **d,
        }
        print(f"  -> grey {grey:3d}: " + "  ".join(f"{k}={v:.4f}" for k, v in d.items()))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)} spectra + summary.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
