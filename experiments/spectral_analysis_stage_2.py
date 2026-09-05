"""Stage 2: merge the per-method stage-1 results into comparison figures.

Handles both experiments. Under <base>/plots/:

TEST 1 (uniform sources, power spectrum)
    spectral_comparison_g<level>.pdf/.png   3 rows x N methods:
                                            point set / mean 2-D spectrum / radial profile
    spectral_radial_all.pdf/.png            radial profiles of every method, overlaid
    spectral_anisotropy_all.pdf/.png        anisotropy (dB), overlaid
    spectral_invariance_<METHOD>.pdf/.png   one per model variant, across grey levels; the model
                                            should be
                                            invariant, since a uniform target specifies the same
                                            RELATIVE density at any intensity
    spectral_summary.csv

TEST 2 (patterned sources, per-region PCF)
    pcf_<pattern>_by_region.pdf/.png        one panel per region, all methods overlaid
    pcf_<pattern>_by_method.pdf/.png        one panel per method, all regions overlaid -- flat
                                            agreement between regions means the model keeps its
                                            correlation structure as the density changes
    pcf_summary.csv

Reading the plots
-----------------
Power spectrum: blue noise dips towards 0 at low frequency, peaks near radial frequency 1, and
settles at 1. A tall peak with a spiky tail is lattice-like; flat at 1 everywhere is white noise.

PCF: blue noise has g(r) ~ 0 inside the exclusion zone, one moderate peak near r = 1 (one mean
spacing), then settles at 1. Poisson noise is flat at 1 with no exclusion zone.
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

# (folder suffix, display label), in plotting order.
METHODS = [
    ("target_WVS_1024", "WVS"),
    ("target_BNOT_1024", "BNOT"),
    ("target_GBN_1024", "GBN"),
    ("target_CN-WVS_1024", "Ours-WVS"),
    ("target_CN-GBN_1024", "Ours-GBN"),
]
# Reference processes with known answers; uncomment to include as controls. Highly recommended
# for a figure -- they give the reader the two extremes to anchor against.
# METHODS = [
#     ("target_WHITE_1024", "White noise"),
#     ("target_RELAXED_1024", "Relaxed"),
# ]

# One invariance figure per model variant. Only our own model is expected to be invariant,
# so the baselines are not plotted here.
INVARIANCE_METHODS = ["target_CN-WVS_1024", "target_CN-GBN_1024"]
FIG_DPI = 130
POINT_MS = 1.6
SPEC_CLIP = 4.0          # display ceiling for the 2-D spectrum, in units of the flat level


def parse_args():
    p = argparse.ArgumentParser(description="Merge stage-1 results into comparison figures")
    p.add_argument("--base", default=BASE_DIR)
    p.add_argument("--output", default=None, help="Default: <base>/plots")
    p.add_argument("--methods", default=None, help="Comma-separated folder:label pairs")
    p.add_argument("--invariance-methods", default=",".join(INVARIANCE_METHODS),
                   help="Comma-separated folder suffixes to draw invariance figures for")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_method(base, folder):
    """Return (spectra_by_grey, pcf_by_key, summary) for one method."""
    d = Path(base) / f"{folder}_spectral"
    if not d.is_dir():
        return {}, {}, {}
    spectra, pcf = {}, {}
    for f in sorted(d.glob("spectral_g*.npz")):
        z = np.load(f)
        spectra[int(z["grey"])] = {k: z[k] for k in z.files}
    for f in sorted(d.glob("pcf_*.npz")):
        z = np.load(f)
        pcf[f.stem] = {k: z[k] for k in z.files}
    summary = {}
    sp = d / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text())
    return spectra, pcf, summary


def load_points(base, folder, stem):
    p = Path(base) / folder / f"{stem}.npy"
    return np.load(p).astype(np.float64) if p.exists() else None


def load_source(base, stem):
    """The conditioning image a point set was generated from, for the thumbnails."""
    p = Path(base) / "source" / f"{stem}.png"
    if not p.exists():
        return None
    img = plt.imread(str(p))
    return img[..., 0] if img.ndim == 3 else img


def draw_thumb(ax, img, box=None, title=None):
    """Render one conditioning image into its own axes.

    Drawn with extent [0, 1, 1, 0] so the vertical axis runs downwards, matching both the image
    rows and the y-down convention of the stored point sets, which keeps an outlined region in
    the same place it occupies in the source.
    """
    ax.set_xticks([]); ax.set_yticks([])
    if img is None:
        ax.axis("off")
        return
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=[0, 1, 1, 0], interpolation="nearest")
    if box is not None:
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="crimson", lw=1.4))
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_edgecolor("0.4")
        sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def thumb_row(fig, gs_row, n_cols, imgs, boxes=None, titles=None, width=0.42):
    """A row of centred thumbnails sharing one gridspec row, one per column."""
    boxes = boxes or [None] * n_cols
    titles = titles or [None] * n_cols
    for i in range(n_cols):
        cell = gs_row[i].get_position(fig)
        h = cell.height
        w = h * fig.get_figheight() / fig.get_figwidth()      # keep it square on the page
        ax = fig.add_axes([cell.x0 + (cell.width - w) / 2, cell.y0, w, h])
        draw_thumb(ax, imgs[i], boxes[i], titles[i])


def stipple_row(fig, gs_row, stipples, pad=0.012, gap=0.006):
    """A strip of point-set panels placed just above the axes, spanning the figure width.

    Used where the curve panels overlay several methods: a single thumbnail cannot stand for
    all of them, so every method's stippling is shown once across the top. Placed outside the
    canvas (y > 1) like figure_thumb, so it cannot collide with the row of axes below; save()
    writes with bbox_inches="tight", which grows the page to include it.
    """
    if not stipples:
        return
    n = len(stipples)
    y0 = 1.0 + pad
    cell = (1.0 - gap * (n - 1)) / n
    w = min(cell, 0.16)
    h = w * fig.get_figwidth() / fig.get_figheight()
    for i, (label, pts) in enumerate(stipples):
        x = i * (cell + gap) + (cell - w) / 2
        ax = fig.add_axes([x, y0, w, h])
        draw_points(ax, pts, title=label)


def figure_thumb(fig, img, title=None, size=0.075, pad=0.012):
    """One thumbnail ABOVE the axes, for figures whose panels share a condition.

    Anchored with its bottom edge at y = 1 + pad, i.e. entirely outside the canvas, so it
    cannot overlap the topmost row of axes no matter how tight_layout packs them. save()
    writes with bbox_inches="tight", which expands the bounding box to enclose every artist,
    so the thumbnail is still captured -- the page simply grows to make room for it.
    """
    if img is None:
        return
    w = size * fig.get_figheight() / fig.get_figwidth()
    ax = fig.add_axes([0.0, 1.0 + pad, w, size])
    draw_thumb(ax, img, title=title)


# Figure captions, collected as figures are built and written to captions.md. Kept out of
# the images themselves so the paper controls the wording and styling.
CAPTIONS = {}


def record_caption(stem, text):
    CAPTIONS[Path(stem).name] = " ".join(text.split())


def save(fig, stem):
    """Write both formats with a tight bounding box.

    bbox_inches="tight" is not cosmetic: without it a narrow figure clips its outermost
    artists in the PDF -- the y-axis labels and the conditioning thumbnails sit close to the
    edge. Expanding the bounding box to enclose every artist prevents that.
    """
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return stem


# ── TEST 1 figures ───────────────────────────────────────────────────────────

def add_legend(ax, fontsize=7):
    """Curve legend inside the axes, top right.

    Every curve panel gets one: these figures put one method (or one density) per curve and
    several panels side by side, so a legend on only the first panel forces the reader to
    match colours across panels.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(fontsize=fontsize, loc="upper right", framealpha=0.85,
              borderpad=0.3, handlelength=1.6, labelspacing=0.3)


def panel_figure(base, methods, grey, spectra, out_dir):
    cols = [(f, l) for f, l in methods if grey in spectra.get(f, {})]
    if not cols:
        return None
    fig, axes = plt.subplots(3, len(cols), figsize=(2.7 * len(cols), 8.2),
                             gridspec_kw={"height_ratios": [1, 1, 0.85]}, squeeze=False)
    for j, (folder, label) in enumerate(cols):
        z = spectra[folder][grey]
        ax = axes[0, j]
        pts = load_points(base, folder, f"uniform_g{grey:03d}_r00")
        if pts is not None:
            ax.plot(pts[:, 0], pts[:, 1], "k.", ms=POINT_MS)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(label, fontsize=11)
        if j == 0:
            ax.set_ylabel("point set", fontsize=9)

        ax = axes[1, j]
        ax.imshow(np.clip(z["mean_spectrum"].astype(np.float64), 0, SPEC_CLIP), cmap="gray",
                  origin="lower", vmin=0, vmax=SPEC_CLIP, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        if j == 0:
            ax.set_ylabel("mean power spectrum", fontsize=9)

        ax = axes[2, j]
        c, v = z["radial_freq"], z["radial_power"]
        ok = np.isfinite(v)
        ax.plot(c[ok], v[ok], color="crimson", lw=1.3)
        ax.axhline(1.0, color="k", ls=":", lw=0.7)
        ax.set_xlim(0, float(c.max()))
        ax.set_ylim(0, max(2.6, float(np.nanmax(v[ok])) * 1.1))
        ax.set_xlabel("radial freq / $\\sqrt{n}$", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("radial power", fontsize=9)

    z0 = spectra[cols[0][0]][grey]
    record_caption(
        out_dir / f"spectral_comparison_g{grey:03d}",
        f"Power spectra on a uniform target density of $\\rho$ = {1.0 - grey / 255.0:.2f} "
        f"(grey level {grey}), {int(z0['n_points'])} points, averaged over "
        f"{int(z0['realizations'])} independent realizations. Top: one representative point "
        f"set per method. Middle: the mean power spectrum. Bottom: the radially averaged "
        f"power. Blue noise shows a low-frequency dip towards zero, a single peak near the "
        f"principal frequency, and a flat tail at one; a flat profile throughout indicates "
        f"white noise. The inset at the upper left is the conditioning image.")
    fig.tight_layout()
    figure_thumb(fig, load_source(base, f"uniform_g{grey:03d}_r00"), title="condition")
    return save(fig, out_dir / f"spectral_comparison_g{grey:03d}")


def overlay_figure(base, methods, spectra, out_dir, key, ylabel, fname, hline=None):
    greys = sorted({g for f, _ in methods for g in spectra.get(f, {})})
    if not greys:
        return None
    # an extra thin row holds the conditioning images, so they sit above the plots rather
    # than over them
    fig, axes = plt.subplots(2, len(greys), figsize=(3.5 * len(greys), 3.9),
                             gridspec_kw={"height_ratios": [0.30, 1.0]}, squeeze=False)
    for a in axes[0]:
        a.axis("off")
    for i, grey in enumerate(greys):
        ax = axes[1, i]
        for folder, label in methods:
            z = spectra.get(folder, {}).get(grey)
            if z is None:
                continue
            c, v = z["radial_freq"], z[key]
            ok = np.isfinite(v)
            ax.plot(c[ok], v[ok], lw=1.3, label=label)
        if hline is not None:
            ax.axhline(hline, color="k", ls=":", lw=0.7)
        ax.set_title(f"$\\rho$ = {1.0 - grey / 255.0:.2f}", fontsize=10)
        ax.set_xlabel("radial freq / $\\sqrt{n}$", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        add_legend(ax)
    fig.tight_layout()
    thumb_row(fig, axes[0], len(greys),
              [load_source(base, f"uniform_g{g:03d}_r00") for g in greys],
              titles=[f"$\\rho$ = {1.0 - g / 255.0:.2f}" for g in greys])
    return save(fig, out_dir / fname)


def invariance_figure(folder, label, spectra, out_dir):
    d = spectra.get(folder, {})
    if len(d) < 2:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for grey in sorted(d):
        z = d[grey]
        lab = f"$\\rho$={1.0 - grey / 255.0:.2f}"
        for k, a in (("radial_power", ax[0]), ("anisotropy_db", ax[1])):
            v = z[k]
            ok = np.isfinite(v)
            a.plot(z["radial_freq"][ok], v[ok], lw=1.3, label=lab)
    ax[0].axhline(1.0, color="k", ls=":", lw=0.7)
    ax[0].set_xlabel("radial freq / $\\sqrt{n}$"); ax[0].set_ylabel("radial power")
    ax[0].set_title(f"{label}: radial power across grey levels", fontsize=10)
    ax[1].set_xlabel("radial freq / $\\sqrt{n}$"); ax[1].set_ylabel("anisotropy (dB)")
    ax[1].set_title(f"{label}: anisotropy across grey levels", fontsize=10)
    for a in ax:
        add_legend(a)
    fig.tight_layout()
    return save(fig, out_dir / f"spectral_invariance_{label.replace(' ', '_')}")


# ── TEST 2 figures ───────────────────────────────────────────────────────────

def draw_points(ax, pts, title=None, box=None):
    """One point set as a square scatter, in the same orientation as the source image.

    The stored point sets use the y-down image convention (see draw_thumb), so the y axis is
    inverted here -- otherwise the stippling appears mirrored relative to the conditioning
    image printed beside it, and the reader cannot match a dense region to its quadrant.
    """
    ax.set_xticks([]); ax.set_yticks([])
    if pts is None:
        ax.axis("off")
        return
    ax.plot(pts[:, 0], pts[:, 1], "k.", ms=POINT_MS)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
    if box is not None:
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="crimson", lw=1.2))
    for sp in ax.spines.values():
        sp.set_edgecolor("0.4"); sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=9, pad=3)



def pcf_figures(base, methods, pcfs, out_dir, regions_by_name=None):
    keys = sorted({k for f, _ in methods for k in pcfs.get(f, {})})
    if not keys:
        return []
    regions_by_name = regions_by_name or {}
    patterns = sorted({pcfs[f][k]["pattern"].item() if hasattr(pcfs[f][k]["pattern"], "item")
                       else str(pcfs[f][k]["pattern"])
                       for f, _ in methods for k in pcfs.get(f, {})})
    written = []
    for pat in patterns:
        pkeys = [k for k in keys if k.startswith(f"pcf_{pat}_")]
        if not pkeys:
            continue

        # one panel per region, methods overlaid
        thumb = load_source(base, f"pattern_{pat}_r00")
        # A row of stipplings above the region thumbnails: the panels below overlay every
        # method, so all of them are shown, each with the panel's region outlined. Ragged
        # counts (more methods than regions) are handled by a separate axes grid.
        stipples = [(l, load_points(base, f, f"pattern_{pat}_r00")) for f, l in methods]
        stipples = [(l, p) for l, p in stipples if p is not None]
        fig, axes = plt.subplots(2, len(pkeys), figsize=(3.4 * len(pkeys), 3.9),
                                 gridspec_kw={"height_ratios": [0.30, 1.0]}, squeeze=False)
        for a in axes[0]:
            a.axis("off")
        for i, k in enumerate(pkeys):
            ax = axes[1, i]
            rho = None
            for folder, label in methods:
                z = pcfs.get(folder, {}).get(k)
                if z is None:
                    continue
                ax.plot(z["r"], z["g"], lw=1.3, label=label)
                rho = float(z["rho"])
            ax.axhline(1.0, color="k", ls=":", lw=0.7)
            ax.set_title(f"{k.split('_')[-1]}   $\\rho$ = {rho:.2f}" if rho is not None else k,
                         fontsize=10)
            ax.set_xlabel("$r$ / mean spacing", fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_ylabel("$g(r)$", fontsize=9)
            add_legend(ax)
        record_caption(
            out_dir / f"pcf_{pat}_by_region",
            f"Pair correlation function measured separately in each constant-density region of "
            f"the '{pat}' pattern, averaged over realizations. Each panel shows one region; the "
            f"thumbnail above it outlines that region within the conditioning image, and the "
            f"top row shows one representative stippling per method. Distances "
            f"are normalised by the local mean spacing, so regions of different density are "
            f"directly comparable. Only points in the interior of a region act as centres, with "
            f"the neighbour radius capped by the margin, so every counted pair lies within a "
            f"single density regime. Blue noise shows $g(r) \\approx 0$ inside the exclusion "
            f"zone and a single peak near $r = 1$; Poisson noise is flat at one throughout.")
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        # each panel describes one region, so each thumbnail outlines the quadrant it refers to
        thumb_row(fig, axes[0], len(pkeys), [thumb] * len(pkeys),
                  boxes=[regions_by_name.get((pat, k.split("_")[-1])) for k in pkeys],
                  titles=[k.split("_")[-1] for k in pkeys])
        stipple_row(fig, axes[0], stipples)
        written.append(save(fig, out_dir / f"pcf_{pat}_by_region"))

        # one panel per method, regions overlaid: curves lying on top of each other means the
        # correlation structure is preserved as the local density changes
        have = [(f, l) for f, l in methods if any(k in pcfs.get(f, {}) for k in pkeys)]
        if have:
            # top row: the stippling each curve was measured from, so the g(r) panel below can
            # be read against the point set that produced it
            fig, axes = plt.subplots(2, len(have), figsize=(3.4 * len(have), 6.4),
                                     gridspec_kw={"height_ratios": [1.0, 0.95]}, squeeze=False)
            for i, (folder, label) in enumerate(have):
                draw_points(axes[0, i], load_points(base, folder, f"pattern_{pat}_r00"),
                            title=label)
                ax = axes[1, i]
                for k in pkeys:
                    z = pcfs.get(folder, {}).get(k)
                    if z is None:
                        continue
                    ax.plot(z["r"], z["g"], lw=1.2, label=f"$\\rho$={float(z['rho']):.2f}")
                ax.axhline(1.0, color="k", ls=":", lw=0.7)
                ax.set_xlabel("$r$ / mean spacing", fontsize=9)
                ax.tick_params(labelsize=7)
                if i == 0:
                    ax.set_ylabel("$g(r)$", fontsize=9)
                add_legend(ax)
            record_caption(
                out_dir / f"pcf_{pat}_by_method",
                f"Density invariance of the correlation structure on the '{pat}' pattern. Each "
                f"panel shows one method with every region overlaid. Curves that coincide mean "
                f"the sampler preserves its blue-noise correlation structure as the local "
                f"density changes; curves that separate mean the structure degrades at some "
                f"densities. The top row is one representative stippling per method, in the "
                f"orientation of the conditioning image; the inset at the upper left is the "
                f"conditioning image itself.")
            fig.tight_layout()
            figure_thumb(fig, thumb, title="condition", size=0.16)
            written.append(save(fig, out_dir / f"pcf_{pat}_by_method"))
    return written


# ── driver ───────────────────────────────────────────────────────────────────

def write_csv(path, rows, cols):
    if not rows:
        return None
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    args = parse_args()
    base = Path(args.base)
    out_dir = Path(args.output) if args.output else base / "plots"

    methods = METHODS
    if args.methods:
        methods = [tuple(x.split(":", 1)) for x in args.methods.split(",")]

    spectra, pcfs, summaries = {}, {}, {}
    for f, _ in methods:
        spectra[f], pcfs[f], summaries[f] = load_method(base, f)

    have_t1 = [(f, l) for f, l in methods if spectra[f]]
    have_t2 = [(f, l) for f, l in methods if pcfs[f]]
    missing = [l for f, l in methods if not spectra[f] and not pcfs[f]]
    if missing:
        print(f"No stage-1 output for: {', '.join(missing)}  (run spectral_analysis_stage_1.py)")
    if not have_t1 and not have_t2:
        print("ERROR: nothing to plot.")
        return 2

    print(f"Base   : {base}")
    print(f"TEST 1 : {', '.join(l for _, l in have_t1) or 'none'}")
    print(f"TEST 2 : {', '.join(l for _, l in have_t2) or 'none'}")
    if args.dry_run:
        print("DRY RUN: no figures written")
        return 0

    # region boxes, so each by-region panel can outline the quadrant it describes
    regions_by_name = {}
    mpath = base / "manifest.json"
    if mpath.exists():
        raw = json.loads(mpath.read_text()).get("regions", {})
        if isinstance(raw, dict):
            for pat_name, regs in raw.items():
                for r in regs:
                    regions_by_name[(pat_name, r["name"])] = r["box"]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if have_t1:
        for grey in sorted({g for f, _ in have_t1 for g in spectra[f]}):
            s = panel_figure(base, have_t1, grey, spectra, out_dir)
            if s:
                written.append(s)
        for key, ylab, name, hl in (("radial_power", "radial power", "spectral_radial_all", 1.0),
                                    ("anisotropy_db", "anisotropy (dB)", "spectral_anisotropy_all", None)):
            s = overlay_figure(base, have_t1, spectra, out_dir, key, ylab, name, hline=hl)
            if s:
                written.append(s)
                record_caption(s, (
                    "Radially averaged power spectrum for every method, one panel per uniform "
                    "target density; the thumbnail above each panel is the conditioning image. "
                    "Blue noise dips towards zero at low frequency, peaks near the principal "
                    "frequency, and settles at one."
                ) if key == "radial_power" else (
                    "Spectral anisotropy in decibels for every method, one panel per uniform "
                    "target density. Low values indicate a clean isotropic annulus rather than "
                    "a directional or structured spectrum."))
        for inv in [x.strip() for x in args.invariance_methods.split(",") if x.strip()]:
            label = dict(have_t1).get(inv, inv)
            s = invariance_figure(inv, label, spectra, out_dir)
            if not s:
                continue
            written.append(s)
            record_caption(s,
                f"Radial power and anisotropy for {label} across uniform "
                f"target densities. A capacity constraint distributes points by RELATIVE "
                f"density, so every uniform level specifies the same target distribution and "
                f"the curves should coincide; separation would indicate a dependence on "
                f"absolute intensity.")

    if have_t2:
        written += pcf_figures(base, have_t2, pcfs, out_dir, regions_by_name)

    rows_t1, rows_t2 = [], []
    for folder, label in methods:
        for key, d in summaries.get(folder, {}).items():
            (rows_t1 if d.get("test") == 1 else rows_t2).append({"method": label, **d})
    write_csv(out_dir / "spectral_summary.csv", rows_t1,
              ["method", "grey", "rho", "n_points", "realizations",
               "low_freq_power", "peak_power", "peak_freq", "tail_deviation"])
    write_csv(out_dir / "pcf_summary.csv", rows_t2,
              ["method", "pattern", "region", "grey", "rho", "mean_points", "realizations",
               "exclusion_leak", "peak", "peak_r", "tail_deviation"])

    if rows_t1:
        print(f"\nTEST 1  {'method':14s}{'rho':>6s}{'low-freq':>11s}{'peak':>8s}{'tail dev':>10s}")
        for r in rows_t1:
            print(f"        {r['method']:14s}{r['rho']:6.2f}{r.get('low_freq_power', float('nan')):11.4f}"
                  f"{r.get('peak_power', float('nan')):8.3f}{r.get('tail_deviation', float('nan')):10.4f}")
    if rows_t2:
        print(f"\nTEST 2  {'method':14s}{'region':>8s}{'rho':>6s}{'excl':>9s}{'peak':>8s}{'tail dev':>10s}")
        for r in rows_t2:
            print(f"        {r['method']:14s}{r['region']:>8s}{r['rho']:6.2f}"
                  f"{r.get('exclusion_leak', float('nan')):9.4f}{r.get('peak', float('nan')):8.3f}"
                  f"{r.get('tail_deviation', float('nan')):10.4f}")

    if CAPTIONS:
        lines = ["# Figure captions", "",
                 "Captions for the spectral-analysis figures. The figures carry no baked-in",
                 "titles, so each entry below is the caption to use in the paper.", ""]
        for stem in sorted(CAPTIONS):
            lines += [f"## `{stem}`", "", CAPTIONS[stem], ""]
        (out_dir / "captions.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Captions for {len(CAPTIONS)} figure(s) -> {out_dir / 'captions.md'}")

    print(f"\nWrote {len(written)} figure(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
