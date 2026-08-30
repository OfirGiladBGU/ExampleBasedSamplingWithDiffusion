"""Stage 2: merge the per-method spectra from stage 1 into comparison figures.

Produces, under <base>/plots/:

    spectral_comparison_g<level>.pdf/.png   3 rows x N methods, one per grey level:
                                            point set / mean 2-D spectrum / radial profile
    spectral_radial_all.pdf/.png            radial profiles of every method, overlaid
    spectral_anisotropy_all.pdf/.png        anisotropy (dB) of every method, overlaid
    spectral_invariance_<METHOD>.pdf/.png   one method across grey levels -- the model should
                                            be invariant, since a uniform target specifies the
                                            same RELATIVE density at any intensity
    spectral_summary.csv                    scalar descriptors, all methods x grey levels

Reading the plots
-----------------
Blue noise shows a low-frequency dip approaching 0, a single moderate peak near the principal
frequency (radial frequency 1 in these units), and a flat tail at 1. A high peak with a spiky
tail means lattice-like regularity; a flat profile at 1 everywhere is white noise.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = r"experiments/outputs/spectral_analysis"

# Methods to include, in plotting order: (folder suffix, display label).
METHODS = [
    ("target_WVS_1024", "WVS"),
    ("target_BNOT_1024", "BNOT"),
    ("target_GBN_1024", "GBN"),
    ("target_CN_1024", "Ours"),
]
# Reference processes from the validation pass; uncomment to include as controls.
# METHODS = [
#     ("target_WHITE_1024", "White noise"),
#     ("target_JITTER_1024", "Jittered grid"),
#     ("target_LATTICE_1024", "Lattice"),
# ]

# Which method to use for the grey-level invariance figure.
INVARIANCE_METHOD = "target_CN_1024"

FIG_DPI = 130
POINT_MS = 1.6
SPEC_CLIP = 4.0          # display ceiling for the 2-D spectrum, in units of the flat level


def parse_args():
    p = argparse.ArgumentParser(description="Merge stage-1 spectra into comparison figures")
    p.add_argument("--base", default=BASE_DIR, help=f"Base folder (default: {BASE_DIR})")
    p.add_argument("--output", default=None, help="Plot folder (default: <base>/plots)")
    p.add_argument("--methods", default=None,
                   help="Comma-separated folder:label pairs, overriding METHODS")
    p.add_argument("--invariance-method", default=INVARIANCE_METHOD)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_method(base, folder):
    """Return {grey: npz_dict} for one method, or {} if stage 1 has not been run for it."""
    d = Path(base) / f"{folder}_spectral"
    if not d.is_dir():
        return {}
    out = {}
    for f in sorted(d.glob("spectral_g*.npz")):
        z = np.load(f)
        out[int(z["grey"])] = {k: z[k] for k in z.files}
    return out


def load_points(base, folder, grey):
    """One representative point set, for the top row."""
    for r in range(100):
        p = Path(base) / folder / f"uniform_g{grey:03d}_r{r:02d}.npy"
        if p.exists():
            return np.load(p).astype(np.float64)
    return None


def panel_figure(base, methods, grey, data, out_dir):
    cols = [(f, l) for f, l in methods if grey in data.get(f, {})]
    if not cols:
        return None
    fig, axes = plt.subplots(3, len(cols), figsize=(2.7 * len(cols), 8.2),
                             gridspec_kw={"height_ratios": [1, 1, 0.85]})
    if len(cols) == 1:
        axes = axes.reshape(3, 1)

    for j, (folder, label) in enumerate(cols):
        z = data[folder][grey]

        ax = axes[0, j]
        pts = load_points(base, folder, grey)
        if pts is not None:
            ax.plot(pts[:, 0], pts[:, 1], "k.", ms=POINT_MS)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=11)
        if j == 0:
            ax.set_ylabel("point set", fontsize=9)

        ax = axes[1, j]
        P = z["mean_spectrum"].astype(np.float64)
        ax.imshow(np.clip(P, 0, SPEC_CLIP), cmap="gray", origin="lower",
                  vmin=0, vmax=SPEC_CLIP, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if j == 0:
            ax.set_ylabel("mean power spectrum", fontsize=9)

        ax = axes[2, j]
        c, v = z["radial_freq"], z["radial_power"]
        ok = np.isfinite(v)
        ax.plot(c[ok], v[ok], color="crimson", lw=1.3)
        ax.axhline(1.0, color="k", ls=":", lw=0.7)
        ax.set_xlim(0, float(c.max())); ax.set_ylim(0, max(2.6, float(np.nanmax(v[ok])) * 1.1))
        ax.set_xlabel("radial freq / $\\sqrt{n}$", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("radial power", fontsize=9)

    rho = 1.0 - grey / 255.0
    n = int(data[cols[0][0]][grey]["n_points"])
    reals = int(data[cols[0][0]][grey]["realizations"])
    fig.suptitle(f"Uniform density $\\rho$ = {rho:.2f} (grey {grey}),  "
                 f"n = {n} points,  mean of {reals} realizations", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    stem = out_dir / f"spectral_comparison_g{grey:03d}"
    fig.savefig(f"{stem}.png", dpi=FIG_DPI); fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    return stem


def overlay_figure(methods, data, out_dir, key, ylabel, fname, hline=None, ylim=None):
    greys = sorted({g for f, _ in methods for g in data.get(f, {})})
    if not greys:
        return None
    fig, axes = plt.subplots(1, len(greys), figsize=(3.5 * len(greys), 3.2), squeeze=False)
    for i, grey in enumerate(greys):
        ax = axes[0, i]
        for folder, label in methods:
            z = data.get(folder, {}).get(grey)
            if z is None:
                continue
            c, v = z["radial_freq"], z[key]
            ok = np.isfinite(v)
            ax.plot(c[ok], v[ok], lw=1.3, label=label)
        if hline is not None:
            ax.axhline(hline, color="k", ls=":", lw=0.7)
        ax.set_title(f"$\\rho$ = {1.0 - grey / 255.0:.2f}", fontsize=10)
        ax.set_xlabel("radial freq / $\\sqrt{n}$", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    stem = out_dir / fname
    fig.savefig(f"{stem}.png", dpi=FIG_DPI); fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    return stem


def invariance_figure(folder, label, data, out_dir):
    d = data.get(folder, {})
    if len(d) < 2:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for grey in sorted(d):
        z = d[grey]
        c, v, a = z["radial_freq"], z["radial_power"], z["anisotropy_db"]
        ok = np.isfinite(v)
        ax[0].plot(c[ok], v[ok], lw=1.3, label=f"$\\rho$={1.0 - grey / 255.0:.2f}")
        ok2 = np.isfinite(a)
        ax[1].plot(c[ok2], a[ok2], lw=1.3, label=f"$\\rho$={1.0 - grey / 255.0:.2f}")
    ax[0].axhline(1.0, color="k", ls=":", lw=0.7)
    ax[0].set_xlabel("radial freq / $\\sqrt{n}$"); ax[0].set_ylabel("radial power")
    ax[0].set_title(f"{label}: radial power across grey levels", fontsize=10)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("radial freq / $\\sqrt{n}$"); ax[1].set_ylabel("anisotropy (dB)")
    ax[1].set_title(f"{label}: anisotropy across grey levels", fontsize=10)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    stem = out_dir / f"spectral_invariance_{label.replace(' ', '_')}"
    fig.savefig(f"{stem}.png", dpi=FIG_DPI); fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    return stem


def main():
    args = parse_args()
    base = Path(args.base)
    out_dir = Path(args.output) if args.output else base / "plots"

    methods = METHODS
    if args.methods:
        methods = [tuple(x.split(":", 1)) for x in args.methods.split(",")]

    data = {f: load_method(base, f) for f, _ in methods}
    have = [(f, l) for f, l in methods if data[f]]
    missing = [l for (f, l) in methods if not data[f]]
    if missing:
        print(f"No stage-1 output for: {', '.join(missing)}  (run spectral_analysis_stage_1.py)")
    if not have:
        print("ERROR: nothing to plot.")
        return 2

    greys = sorted({g for f, _ in have for g in data[f]})
    print(f"Base   : {base}")
    print(f"Methods: {', '.join(l for _, l in have)}")
    print(f"Greys  : {greys}")
    if args.dry_run:
        print("DRY RUN: no figures written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for grey in greys:
        s = panel_figure(base, have, grey, data, out_dir)
        if s:
            written.append(s)
    s = overlay_figure(have, data, out_dir, "radial_power", "radial power",
                       "spectral_radial_all", hline=1.0)
    if s:
        written.append(s)
    s = overlay_figure(have, data, out_dir, "anisotropy_db", "anisotropy (dB)",
                       "spectral_anisotropy_all")
    if s:
        written.append(s)

    inv = args.invariance_method
    lbl = dict(have).get(inv, inv)
    s = invariance_figure(inv, lbl, data, out_dir)
    if s:
        written.append(s)

    rows = []
    for folder, label in have:
        sm_path = base / f"{folder}_spectral" / "summary.json"
        if not sm_path.exists():
            continue
        for key, d in json.loads(sm_path.read_text()).items():
            rows.append({"method": label, **d})
    if rows:
        cols = ["method", "grey", "rho", "n_points", "realizations",
                "low_freq_power", "peak_power", "peak_freq", "tail_deviation"]
        with open(out_dir / "spectral_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n{'method':14s}{'rho':>6s}{'low-freq':>11s}{'peak':>8s}{'peak f':>9s}{'tail dev':>10s}")
        print("-" * 58)
        for r in rows:
            print(f"{r['method']:14s}{r['rho']:6.2f}{r.get('low_freq_power', float('nan')):11.4f}"
                  f"{r.get('peak_power', float('nan')):8.3f}{r.get('peak_freq', float('nan')):9.3f}"
                  f"{r.get('tail_deviation', float('nan')):10.4f}")

    print(f"\nWrote {len(written)} figure(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
