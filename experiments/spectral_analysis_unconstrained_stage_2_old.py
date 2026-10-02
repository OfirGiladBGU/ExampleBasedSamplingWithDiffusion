"""spectral_analysis_unconstrained_stage_2_old.py

Unconstrained spectral analysis, stage 2: spectra and figures for the point sets built by
spectral_analysis_unconstrained_stage_1_old.py (full-black target, rho = 1.0, 25 realizations).

For every method in METHODS the power spectrum is computed exactly as in the original spectral
analysis (spectral_analysis_stage_1.py: direct evaluation from the point coordinates, mean over
the realizations, radial average), written to <folder>_spectral/, and drawn with
spectral_analysis_stage_2.py's figure code, so these figures match the original ones in style:

    plots/spectral_comparison_g000.pdf/.png   point set / mean 2-D spectrum / radial profile,
                                              one column per method:
                                              WVS | GBN | Base-GBN | Ours-WVS | Ours-GBN
    plots/spectral_radial_all.pdf/.png        radial profiles of every method, overlaid
    plots/spectral_summary.csv                low-frequency power, peak, tail deviation
    plots/captions.md

Run from the project root (CPU is enough):
    python experiments/spectral_analysis_unconstrained_stage_2_old.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import spectral_analysis_stage_1 as s1
import spectral_analysis_stage_2 as s2

BASE_DIR = "experiments/outputs/spectral_analysis_unconstrained"
GREY_LEVEL = 0

# (folder, display label), in plotting order = the columns of the comparison figure.
METHODS = [
    ("target_WVS_1024", "WVS"),
    # ("target_BNOT_1024", "BNOT"),
    ("target_GBN_1024", "GBN"),
    ("target_Base-GBN_1024", "Base-GBN"),
    ("target_CN-WVS_1024", "Ours-WVS"),
    ("target_CN-GBN_1024", "Ours-GBN"),
]

# Spectrum settings: the original analysis' defaults, so the numbers are directly comparable.
FREQ_HALF_WIDTH = s1.FREQ_HALF_WIDTH
RADIAL_BINS = s1.RADIAL_BINS
RADIAL_MAX = s1.RADIAL_MAX
ANGULAR_SECTORS = s1.ANGULAR_SECTORS

# What to draw (passed on to spectral_analysis_stage_2's figure code; same meaning as there).
SHOW_LEGENDS = False
SHOW_AXIS_LABELS = False
SHOW_CONDITION_THUMB = False


def parse_args():
    p = argparse.ArgumentParser(description="Unconstrained spectral analysis, stage 2: spectra + figures")
    p.add_argument("--base", default=BASE_DIR)
    p.add_argument("--output", default=None, help="Default: <base>/plots")
    p.add_argument("--grey", type=int, default=GREY_LEVEL)
    p.add_argument("--methods", default=None, help="Comma-separated folder:label pairs (replaces METHODS)")
    p.add_argument("--legends", action=argparse.BooleanOptionalAction, default=SHOW_LEGENDS)
    p.add_argument("--axis-labels", action=argparse.BooleanOptionalAction, default=SHOW_AXIS_LABELS)
    p.add_argument("--condition-thumb", action=argparse.BooleanOptionalAction, default=SHOW_CONDITION_THUMB)
    return p.parse_args()


def compute_spectrum(base, folder, grey, margin_frac):
    """Mean spectrum of one method over its realizations, saved like spectral_analysis_stage_1."""
    files = sorted((base / folder).glob(f"uniform_g{grey:03d}_r*.npy"))
    if not files:
        return None
    acc, n_pts, F = None, None, None
    for f in files:
        pts = s1.load_points(f, margin_frac)
        n_pts = len(pts)
        if F is None:
            F = s1.resolve_freq_half_width(FREQ_HALF_WIDTH, n_pts, RADIAL_MAX)
        P = s1.power_spectrum(pts, F)
        acc = P if acc is None else acc + P
    mean_P = acc / len(files)
    centres, radial, aniso = s1.radial_and_anisotropy(mean_P, F, n_pts, RADIAL_BINS, RADIAL_MAX,
                                                      ANGULAR_SECTORS)
    d = s1.spectrum_descriptors(centres, radial)
    out_dir = base / f"{folder}_spectral"
    out_dir.mkdir(parents=True, exist_ok=True)
    z = dict(mean_spectrum=mean_P.astype(np.float32), radial_freq=centres, radial_power=radial,
             anisotropy_db=aniso, grey=grey, n_points=n_pts, realizations=len(files),
             freq_half_width=F)
    np.savez_compressed(out_dir / f"spectral_g{grey:03d}.npz", **z)
    summary = {f"spectrum_g{grey:03d}": {"test": 1, "grey": grey, "rho": 1.0 - grey / 255.0,
                                         "n_points": int(n_pts), "realizations": len(files), **d}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {k: np.asarray(v) for k, v in z.items()}, summary


def main():
    args = parse_args()
    base = Path(args.base)
    out_dir = Path(args.output) if args.output else base / "plots"
    methods = METHODS if not args.methods else [tuple(x.split(":", 1)) for x in args.methods.split(",")]
    s2.SHOW_LEGENDS, s2.SHOW_AXIS_LABELS = args.legends, args.axis_labels
    s2.SHOW_CONDITION_THUMB = args.condition_thumb

    manifest = json.loads((base / "manifest.json").read_text())
    res, margin_px = int(manifest.get("resolution", 0) or 0), int(manifest.get("margin_px", 0) or 0)
    margin_frac = margin_px / res if res > 0 and margin_px > 0 else 0.0

    spectra, rows, have = {}, [], []
    for folder, label in methods:
        r = compute_spectrum(base, folder, args.grey, margin_frac)
        if r is None:
            print(f"  [skip] {label}: no point sets in {base / folder} (run stage 1)")
            continue
        z, summary = r
        spectra[folder] = {args.grey: z}
        have.append((folder, label))
        rows += [{"method": label, **d} for d in summary.values()]
        d = next(iter(summary.values()))
        print(f"  {label:9s} n={d['n_points']}  realizations={d['realizations']}  "
              f"low-freq={d['low_freq_power']:.4f}  peak={d['peak_power']:.3f} @ {d['peak_freq']:.3f}  "
              f"tail dev={d['tail_deviation']:.4f}")
    if not have:
        print("ERROR: nothing to plot.")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    s = s2.panel_figure(base, have, args.grey, spectra, out_dir)
    if s:
        written.append(s)
        z0 = spectra[have[0][0]][args.grey]
        s2.record_caption(s, (
            f"Power spectra on the uniform full-black target ($\\rho$ = 1.0), "
            f"{int(z0['n_points'])} points, averaged over {int(z0['realizations'])} independent "
            f"realizations. Base-GBN is the unconditioned GBN diffusion model: it takes no capacity "
            f"input, and a uniform target is the one setting in which it can be compared with the "
            f"capacity-constrained samplers. Top: one representative point set per method. Middle: "
            f"the mean power spectrum. Bottom: the radially averaged power. Blue noise shows a "
            f"low-frequency dip towards zero, a single peak near the principal frequency, and a "
            f"flat tail at one."))
    for key, ylab, name, hl, cap in (
            ("radial_power", "radial power", "spectral_radial_all", 1.0,
             "Radially averaged power spectrum of every method on the uniform full-black target "
             "($\\rho$ = 1.0), including the unconditioned base GBN model."),
            # no anisotropy figure in this experiment: the radial profile is the comparison
    ):
        s = s2.overlay_figure(base, have, spectra, out_dir, key, ylab, name, hline=hl)
        if s:
            written.append(s)
            s2.record_caption(s, cap)

    s2.write_csv(out_dir / "spectral_summary.csv", rows,
                 ["method", "grey", "rho", "n_points", "realizations",
                  "low_freq_power", "peak_power", "peak_freq", "tail_deviation"])
    lines = ["# Figure captions", "",
             "Captions for the unconstrained spectral-analysis figures. The figures carry no",
             "baked-in titles, so each entry below is the caption to use in the paper.", ""]
    for stem in sorted(s2.CAPTIONS):
        lines += [f"## `{stem}`", "", s2.CAPTIONS[stem], ""]
    (out_dir / "captions.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {len(written)} figure(s), spectral_summary.csv and captions.md to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
