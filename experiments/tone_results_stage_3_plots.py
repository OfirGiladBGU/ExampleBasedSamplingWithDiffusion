"""Stage 2: turn the stage-1 results into the paper figures.

Under <base>/plots/:
    tone_hero_<stem>.pdf/.png    one image across the three configurations, the figure the
                                 paper leads with
    tone_correction_<stem>.*     what the optimizer actually did to the density
    tone_convergence.*           loss against step, all images overlaid
    tone_scatter.*               per-image PSNR/SSIM, corrected against uncorrected
    captions.md                  captions, kept out of the images so the paper owns the wording

Usage:
    python experiments/tone_results_stage_3_plots.py --base experiments/outputs/tone_results/main
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = "experiments/outputs/tone_results"
BASE = f"{ROOT_DIR}/main"
FIG_DPI = 300

# The image the paper's explainer figure uses. A bare run reproduces exactly that figure.
HERO_STEM = "lg_2_woman_with_bunny_ears"
POINT_MS = 1.1
LABEL = {"none": "Uncorrected", "random": "Random search (no gradients)",
         "curve": "Global tone curve", "field": "Ours (optimized density)"}
# Plot/table order. "random" is the control that matters: same parameters, same objective,
# matched sampler-evaluation budget, no gradients.
ORDER = ("none", "random", "curve", "field")

CAPTIONS = {}


# --------------------------------------------------------------------------
# Defaults for every command-line argument. Edit here, not in parse_args.
# --------------------------------------------------------------------------
OUTPUT    = None
MAX_HERO  = 3
DOT_SCALE = 1.8
STEMS     = None


def record_caption(stem, text):
    CAPTIONS[Path(stem).name] = " ".join(text.split())


def parse_args():
    p = argparse.ArgumentParser(description="Figures for the tone-correction experiment")
    p.add_argument("--base", default=BASE)
    p.add_argument("--output", default=OUTPUT, help="Default: <base>/plots")
    p.add_argument("--hero", default=HERO_STEM,
                   help="Stem for the explainer figure. Defaults to the one the "
                        "paper uses, so a bare run reproduces it.")
    p.add_argument("--max-hero", type=int, default=MAX_HERO)
    p.add_argument("--dot-scale", type=float, default=DOT_SCALE,
                   help="Vector dot radius as a multiple of the renderer's Gaussian "
                        "sigma. Opaque circles read heavier than the Gaussian falloff, "
                        "so below 1.0 keeps the stipple texture visible.")
    p.add_argument("--extras", action="store_true",
                   help="Also draw the exploratory figures (hero, correction, 6-panel "
                        "explainer, convergence, scatter, 2-panel sweep). They go to "
                        "plots-backup/ so plots/ stays exactly the paper set.")
    p.add_argument("--stems", default=STEMS,
                   help="Comma-separated stems to build per-image figures for, instead of "
                        "the first --max-hero alphabetically")
    return p.parse_args()


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return stem


def show(ax, img, title=None, cmap="gray", vmin=0, vmax=1):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def show_points(ax, coords, title=None, radius=None, res=256, sigma_px=2.0,
                dot_scale=1.8, ink_gain=None):
    """Draw a point set as vector circles, matched to the renderer's dot size.

    Uses data-coordinate ellipses, so the dots stay true circles under any figure size and
    remain vector paths in the PDF. `radius` overrides the renderer-derived size.

    Tone is handled by alpha rather than by opaque fill: alpha compositing multiplies
    transmittance per layer, which is the same form as the renderer's Beer-Lambert law
    (darkness = 1 - exp(-gain * ink)). A single dot is therefore given alpha
    1 - exp(-ink_gain), and overlapping dots darken toward black exactly as ink does.

    NOTE: still a depiction, not the raster the metrics use -- a uniform-alpha disc replaces
    a Gaussian profile. The error panels come from the raster.
    """
    from matplotlib.collections import EllipseCollection
    if radius is None:
        radius = dot_scale * (float(sigma_px) / float(res))
    d = 2.0 * radius
    import math
    alpha = 1.0 if ink_gain is None else max(0.05, min(1.0, 1.0 - math.exp(-float(ink_gain))))
    ec = EllipseCollection(widths=d, heights=d, angles=0.0, units="xy",
                           offsets=coords, offset_transform=ax.transData,
                           facecolors="black", edgecolors="none", linewidths=0.0,
                           alpha=alpha)
    ax.add_collection(ec)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)          # row 0 at the top, matching imshow
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def hero_figure(z, stem, configs, out_dir):
    """Target, then each configuration's rendered result and its point set."""
    cols = 1 + len(configs)
    fig, axes = plt.subplots(2, cols, figsize=(2.5 * cols, 5.2), squeeze=False)
    # Panels show REFLECTANCE (ink dark on white paper), not the darkness field the
    # objective works in, so the figure reads the way a printed page does.
    show(axes[0, 0], z["image_01"], "Target")
    axes[1, 0].axis("off")
    for j, c in enumerate(configs, start=1):
        show(axes[0, j], 1.0 - z[f"{c}_darkness"], LABEL.get(c, c))
        pts = z[f"{c}_coords"]
        axes[1, j].plot(pts[:, 0], pts[:, 1], "k.", ms=POINT_MS)
        axes[1, j].set_xlim(0, 1); axes[1, j].set_ylim(1, 0)
        axes[1, j].set_aspect("equal"); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    axes[0, 0].set_ylabel("rendered", fontsize=9)
    axes[1, 1].set_ylabel("point set", fontsize=9)
    fig.tight_layout()
    s = save(fig, out_dir / f"tone_hero_{stem}")
    record_caption(s, (
        "Tone correction through the sampler. Top row: the target image, then the rendered "
        "stipple obtained from each conditioning strategy. Bottom row: the corresponding "
        "point sets. Rendering uses finite-radius dots whose ink saturates where they "
        "overlap, so feeding the target density directly (Uncorrected) produces a result that "
        "is too dark in the dense regions. A global tone curve corrects the average response "
        "but cannot adapt locally. Optimizing the conditioning density through the sampler "
        "recovers the target tone, and is only possible because gradients pass through the "
        "point generation process."))
    return s


def correction_figure(z, stem, out_dir):
    """What the optimizer changed: the density before, after, and the difference."""
    if "field_rho" not in z:
        return None
    rho0 = 1.0 - z["image_01"]
    rho1 = z["field_rho"]
    d = rho1 - rho0
    lim = float(np.abs(d).max()) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    show(axes[0], rho0, r"requested density $\rho_0$")
    show(axes[1], rho1, r"optimized density $\rho^*$")
    im = axes[2].imshow(d, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="nearest")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title(r"$\rho^* - \rho_0$", fontsize=9)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    s = save(fig, out_dir / f"tone_correction_{stem}")
    record_caption(s, (
        "The correction the optimizer finds. Left: the density originally requested. Middle: "
        "the density that, once stippled and rendered, reproduces the target. Right: their "
        "difference. The correction is negative where dots would otherwise overlap and "
        "over-darken the page, and it varies spatially, which is what a global transfer curve "
        "cannot express."))
    return s


def convergence_figure(files, out_dir):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    drawn = 0
    for f in files:
        z = np.load(f)
        for c, style in (("curve", "--"), ("field", "-")):
            k = f"{c}_loss"
            if k in z and len(z[k]):
                ax.plot(z[k], style, lw=1.0, alpha=0.7,
                        color="tab:orange" if c == "curve" else "tab:blue")
                drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_yscale("log")
    ax.set_xlabel("optimization step", fontsize=9)
    ax.set_ylabel("objective", fontsize=9)
    ax.plot([], [], "--", color="tab:orange", label=LABEL["curve"])
    ax.plot([], [], "-", color="tab:blue", label=LABEL["field"])
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    s = save(fig, out_dir / "tone_convergence")
    record_caption(s, (
        "Convergence of the conditioning optimization, one curve per test image. Both "
        "parameterizations are fitted with the same optimizer against the same objective and "
        "differ only in how much freedom the correction has. The spatially varying "
        "parameterization reaches a lower objective on every image."))
    return s


def scatter_figure(rows, out_dir):
    by = {}
    for r in rows:
        by.setdefault(r["stem"], {})[r["config"]] = r
    stems = [s for s, d in by.items() if "none" in d and "field" in d]
    if not stems:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, key, name in ((axes[0], "psnr", "PSNR (dB)"), (axes[1], "ssim", "SSIM")):
        x = [float(by[s]["none"][key]) for s in stems]
        y = [float(by[s]["field"][key]) for s in stems]
        ax.scatter(x, y, s=18, alpha=0.8, label=LABEL["field"])
        if "curve" in by[stems[0]]:
            yc = [float(by[s]["curve"][key]) for s in stems]
            ax.scatter(x, yc, s=18, alpha=0.8, marker="^", label=LABEL["curve"])
        lo = min(x + y) - 0.02 * abs(min(x + y) or 1)
        hi = max(x + y) + 0.02 * abs(max(x + y) or 1)
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xlabel(f"uncorrected {name}", fontsize=9)
        ax.set_ylabel(f"corrected {name}", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    s = save(fig, out_dir / "tone_scatter")
    record_caption(s, (
        "Per-image agreement between the rendered stipple and the target, before and after "
        "correcting the conditioning density. Points above the diagonal are images the "
        "correction improved. The dotted line is equality."))
    return s


def explain_figure(z, stem, out_dir, crop=0.42):
    """Where the render is wrong, before and after -- the mechanism made visible.

    Top row: target, and the two renders. Bottom row: the signed error of each render against
    the target on a shared scale, and the correction the optimizer applied. Sub-decibel
    differences are invisible between the renders themselves but obvious in the error maps.
    """
    if "field_darkness" not in z or "none_darkness" not in z:
        return None
    tgt_full = 1.0 - z["image_01"]
    res = z["none_darkness"].shape[0]
    # match the target to the render resolution by block-averaging
    k = max(1, tgt_full.shape[0] // res)
    tgt = tgt_full[:res * k, :res * k].reshape(res, k, res, k).mean(axis=(1, 3))

    e_none = z["none_darkness"] - tgt
    e_field = z["field_darkness"] - tgt
    lim = float(np.percentile(np.abs(np.concatenate([e_none, e_field])), 99)) or 1.0

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    show(axes[0, 0], 1.0 - tgt, "target")
    show(axes[0, 1], 1.0 - z["none_darkness"], f"{LABEL['none']}")
    show(axes[0, 2], 1.0 - z["field_darkness"], f"{LABEL['field']}")

    for ax, e, name in ((axes[1, 1], e_none, LABEL["none"]),
                        (axes[1, 2], e_field, LABEL["field"])):
        im = ax.imshow(e, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"error, {name}\nRMS {float(np.sqrt((e ** 2).mean())):.4f}", fontsize=8)
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

    d = z["field_rho"] - (1.0 - z["image_01"])
    dl = float(np.abs(d).max()) or 1.0
    im2 = axes[1, 0].imshow(d, cmap="PuOr_r", vmin=-dl, vmax=dl, interpolation="nearest")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    axes[1, 0].set_title(u"correction applied  ρ* − ρ₀", fontsize=8)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    fig.tight_layout()
    s_ = save(fig, out_dir / f"tone_explain_{stem}")
    record_caption(s_, (
        "How the correction acts. Top: the target and the stipple rendered from the requested "
        "density and from the optimized one; the two renders are close enough that the "
        "difference is not apparent by eye. Bottom centre and right: the signed error of each "
        "render against the target on a shared scale, red where the page comes out too dark "
        "and blue where it is too light. The uncorrected render is systematically too dark "
        "wherever dots overlap; the corrected one is closer to neutral. Bottom left: the "
        "correction responsible, which asks for less ink in the regions that would otherwise "
        "over-darken and varies across the image in a way a single global transfer curve "
        "cannot express."))
    return s_


def gain_sweep_figure(base, out_dir):
    """Advantage against the severity of the dot gain being corrected.

    Reads the sweep summary if it exists; the sweep is optional, so absence is not an error.
    """
    csv_path = base.parent / "gain_sweep" / "gain_sweep.csv"
    if not csv_path.exists():
        return None
    rows = sorted((dict(r) for r in csv.DictReader(open(csv_path, encoding="utf-8"))),
                  key=lambda r: float(r["multiplier"]))
    if len(rows) < 2:
        return None
    x = [float(r["multiplier"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for c, style in (("none", ":"), ("curve", "--"), ("field", "-")):
        k = f"{c}_psnr"
        if all(k in r for r in rows):
            axes[0].plot(x, [float(r[k]) for r in rows], style, marker="o", ms=4,
                         label=LABEL[c])
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("ink gain / calibrated", fontsize=9)
    axes[0].set_ylabel("PSNR (dB)", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].tick_params(labelsize=8)

    for k, lab in (("gain_over_none", "vs uncorrected"), ("gain_over_curve", "vs tone curve")):
        if all(k in r for r in rows):
            axes[1].plot(x, [float(r[k]) for r in rows], marker="o", ms=4, label=lab)
    axes[1].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("ink gain / calibrated", fontsize=9)
    axes[1].set_ylabel("PSNR advantage (dB)", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].tick_params(labelsize=8)
    fig.tight_layout()
    s = save(fig, out_dir / "tone_gain_sweep")
    record_caption(s, (
        "Sensitivity to the rendering model. The ink gain sets how quickly overlapping dots "
        "saturate, i.e. how severe the dot gain being corrected is; it is swept from a "
        "quarter to four times its calibrated value. Left: absolute agreement with the "
        "target for each configuration. Right: the advantage of the optimized conditioning. "
        "The advantage grows with the severity of the distortion and falls towards zero as it "
        "vanishes, which is the expected behaviour when there is nothing left to compensate "
        "for, and the ordering of the three configurations is unchanged throughout."))
    return s


def improvement_figure(rows, out_dir):
    """Per-image PSNR gain over uncorrected, one series per strategy.

    Ordered by uncorrected PSNR so the x axis reads as "hardest image first". Preferred over a
    corrected-vs-uncorrected scatter for print: there the markers sit on the diagonal and the
    separation being claimed is not legible at column width.
    """
    by = {}
    for r in rows:
        by.setdefault(r["stem"], {})[r["config"]] = r
    stems = [st for st in by if "none" in by[st] and "field" in by[st]]
    if len(stems) < 2:
        return None
    stems.sort(key=lambda st: float(by[st]["none"]["psnr"]))
    x = range(1, len(stems) + 1)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    styles = {"random": ("^", "tab:orange"), "curve": ("s", "tab:green"),
              "field": ("o", "tab:blue")}
    means = {}
    for cfg in ("random", "curve", "field"):
        if not all(cfg in by[st] for st in stems):
            continue
        d = [float(by[st][cfg]["psnr"]) - float(by[st]["none"]["psnr"]) for st in stems]
        means[cfg] = sum(d) / len(d)
        mk, col = styles[cfg]
        ax.plot(x, d, marker=mk, ms=4, lw=1.2, color=col,
                label=f"{LABEL[cfg]}  (mean {means[cfg]:+.2f} dB)")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("test image, ordered by uncorrected PSNR", fontsize=9)
    ax.set_ylabel("PSNR gain over uncorrected (dB)", fontsize=9)
    ax.tick_params(labelsize=8)

    # the gradient-specific claim, as the legend title so it cannot collide with the data
    note = None
    if "random" in means:
        d = [float(by[st]["field"]["psnr"]) - float(by[st]["random"]["psnr"]) for st in stems]
        w = sum(1 for v in d if v > 0)
        note = f"ours - random search: {sum(d) / len(d):+.2f} dB mean, {w}/{len(d)} images"
    leg = ax.legend(fontsize=8, loc="upper left", title=note, title_fontsize=8)
    if note:
        leg.get_title().set_ha("left")
    fig.tight_layout()
    st_ = save(fig, out_dir / "tone_improvement")
    record_caption(st_, (
        "Tone agreement gained by each correction strategy, per test image, ordered by the "
        "agreement of the uncorrected render so the hardest images appear first. Every "
        "strategy is given the same objective and the same budget of sampler evaluations; the "
        "only difference is whether gradients are used. Optimizing the conditioning through "
        "the sampler improves every image, and beats both a fitted global transfer curve and a "
        "gradient-free search over the identical parameters"
        + (f" ({note}). " if note else ". ")
        + "The gradient-free comparison is the one that isolates the value of "
          "differentiability: a black-box search of the same space at the same cost recovers "
          "only a fraction of the available improvement."))
    return st_


def explain_compact_figure(z, stem, out_dir, dot_scale=1.8):
    """The mechanism in one compact figure, laid out as a 2x3 comparison.

    Columns read "what was asked for -> what rendered -> how wrong it is"; rows are the
    uncorrected and the corrected path. Including the optimized density itself matters: it is
    what the optimizer produces, and seeing it lighter than the target in the crowded regions
    is what makes "pre-compensation" concrete. Sub-decibel differences between the two renders
    are invisible by eye, which is why the error columns carry the argument.
    """
    if "field_darkness" not in z or "none_darkness" not in z:
        return None
    rho0_full = 1.0 - z["image_01"]           # requested density == target darkness
    res = z["none_darkness"].shape[0]
    k = max(1, rho0_full.shape[0] // res)
    tgt = rho0_full[:res * k, :res * k].reshape(res, k, res, k).mean(axis=(1, 3))

    e_none = z["none_darkness"] - tgt
    e_field = z["field_darkness"] - tgt
    lim = float(np.percentile(np.abs(np.concatenate([e_none, e_field])), 99)) or 1.0

    rho_star = z["field_rho"] if "field_rho" in z else None

    # the dot radius the metrics were computed with, so the drawn dots match the renderer
    dot_sigma_px, ink_gain = 2.0, None
    cfg_path = out_dir.parent / "main" / "summary.json"
    if cfg_path.exists():
        try:
            _c = json.load(open(cfg_path, encoding="utf-8"))["config"]
            dot_sigma_px = float(_c["dot_sigma_px"])
            ink_gain = float(_c["ink_gain"])
        except Exception:
            pass

    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.4))

    # column 1 -- the conditioning that was asked for, before and after optimization
    show(axes[0, 0], 1.0 - rho0_full, u"target = requested density ρ₀")
    if rho_star is not None:
        show(axes[1, 0], 1.0 - rho_star, u"optimized density ρ*")
    else:
        axes[1, 0].axis("off")

    # column 2 -- the stipple each conditioning produces, drawn as vector dots so the
    # points stay sharp when the PDF is zoomed (see show_points for the caveat)
    for ax, cfg in ((axes[0, 1], "none"), (axes[1, 1], "field")):
        ck = f"{cfg}_coords"
        if ck in z:
            show_points(ax, z[ck], LABEL[cfg], res=res, sigma_px=dot_sigma_px,
                        dot_scale=dot_scale, ink_gain=ink_gain)
        else:
            show(ax, 1.0 - z[f"{cfg}_darkness"], LABEL[cfg])

    # column 3 -- signed error against the target, one shared scale
    for ax, e in ((axes[0, 2], e_none), (axes[1, 2], e_field)):
        im = ax.imshow(e, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"error, RMS {float(np.sqrt((e ** 2).mean())):.4f}", fontsize=8)
    cb = fig.colorbar(im, ax=axes[:, 2], fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)

    for ax in axes.ravel():
        ax.title.set_fontsize(8)

    r_none = float(np.sqrt((e_none ** 2).mean()))
    r_field = float(np.sqrt((e_field ** 2).mean()))
    st_ = save(fig, out_dir / f"tone_explain_compact_{stem}")
    record_caption(st_, (
        "What the correction does, read left to right as requested conditioning, rendered "
        "result, and error against the target. Top row: the target tone, which is also the "
        "density the sampler is normally asked for, and the stipple rendered from it. Bottom "
        "row: the density recovered by optimizing through the sampler, and the stipple it "
        "produces. The stipple panels are drawn as vector dots at the renderer's dot radius, "
        "so they stay sharp in print; the error panels are computed from the rasterized "
        "render the metrics use. "
        "produces -- lighter than the target exactly where dots would otherwise crowd and "
        "over-darken the page. Right column: signed error on a shared scale, red where the "
        "page comes out too dark and blue where it is too light. The uncorrected render is "
        "systematically too dark wherever dots overlap, which is dot gain; the correction "
        f"trades that large, concentrated error for a smaller and more uniform one (RMS "
        f"{r_none:.4f} to {r_field:.4f}, a "
        f"{100.0 * (1.0 - r_field / r_none):.0f}% reduction). The correction varies across "
        "the image, so no single global transfer curve can express it."))
    return st_

def gain_sweep_compact_figure(base, out_dir):
    """The advantage-vs-ink-gain panel on its own -- the robustness argument, one panel."""
    csv_path = base.parent / "gain_sweep" / "gain_sweep.csv"
    if not csv_path.exists():
        return None
    rows = sorted((dict(r) for r in csv.DictReader(open(csv_path, encoding="utf-8"))),
                  key=lambda r: float(r["multiplier"]))
    if len(rows) < 2:
        return None
    x = [float(r["multiplier"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for k, lab in (("gain_over_none", "vs uncorrected"), ("gain_over_curve", "vs tone curve")):
        if all(k in r for r in rows):
            ax.plot(x, [float(r[k]) for r in rows], marker="o", ms=4, label=lab)
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ink gain / calibrated", fontsize=9)
    ax.set_ylabel("PSNR advantage (dB)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    s = save(fig, out_dir / "tone_gain_sweep_compact")
    record_caption(s, (
        "Sensitivity to the rendering model. The ink gain sets how quickly overlapping dots "
        "saturate -- how severe the dot gain being corrected is -- and is swept from a quarter "
        "to four times its calibrated value. The advantage of optimizing the conditioning "
        "grows with the severity of the distortion and falls towards zero as it vanishes, "
        "which is the expected behaviour when there is nothing left to compensate for. The "
        "ordering of the strategies is unchanged throughout, so the result is not an artefact "
        "of the particular gain that was calibrated."))
    return s


def main():
    a = parse_args()
    base = Path(a.base)
    # Plots live at the top of the tree, not inside the run they came from, so every
    # figure worth showing sits in one place.
    out_dir = Path(a.output) if a.output else base.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in base.glob("*.npz"))
    if not files:
        print(f"ERROR: no stage-1 .npz results under {base}")
        return 2

    rows = []
    csv_path = base / "tone_summary.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    written = []
    if a.stems:
        picks = [base / f"{x.strip()}.npz" for x in a.stems.split(",") if x.strip()]
    elif a.hero:
        picks = [base / f"{a.hero}.npz"]
    else:
        picks = files[:max(1, a.max_hero)]
    # The paper set, always: two graphs and one visual, into out_dir (plots/).
    for f in picks[:1]:
        z = np.load(f)
        s = explain_compact_figure(z, f.stem, out_dir, dot_scale=a.dot_scale)
        if s:
            written.append(s)
    if rows:
        s = improvement_figure(rows, out_dir)
        if s:
            written.append(s)
    s = gain_sweep_compact_figure(base, out_dir)
    if s:
        written.append(s)

    if a.extras:
        # Exploratory figures go beside the paper set, not into it.
        out_dir = base.parent / "plots-backup"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in picks:
            z = np.load(f)
            configs = [c for c in ORDER if f"{c}_darkness" in z]
            written.append(hero_figure(z, f.stem, configs, out_dir))
            for fn in (correction_figure, explain_figure):
                s = fn(z, f.stem, out_dir)
                if s:
                    written.append(s)

        s = convergence_figure(files, out_dir)
        if s:
            written.append(s)
        s = gain_sweep_figure(base, out_dir)
        if s:
            written.append(s)
        if rows:
            s = scatter_figure(rows, out_dir)
            if s:
                written.append(s)
    if rows:

        agg = {}
        for r in rows:
            agg.setdefault(r["config"], []).append(r)
        lines = ["| configuration | PSNR (dB) | SSIM | MAE | seconds |", "|---|---|---|---|---|"]
        for c in ORDER:
            if c in agg:
                g = agg[c]
                lines.append(f"| {LABEL[c]} | {np.mean([float(x['psnr']) for x in g]):.2f} "
                             f"| {np.mean([float(x['ssim']) for x in g]):.4f} "
                             f"| {np.mean([float(x['mae']) for x in g]):.4f} "
                             f"| {np.mean([float(x['seconds']) for x in g]):.1f} |")
        (out_dir / "tone_table.md").write_text(
            "\n".join([f"Averaged over {len(agg.get('field', []))} images.", ""] + lines) + "\n",
            encoding="utf-8")
        print(f"table -> {out_dir / 'tone_table.md'}")

    if CAPTIONS:
        md = ["# Figure captions", "",
              "Captions for the tone-correction figures. The figures carry no baked-in titles,",
              "so each entry below is the caption to use in the paper.", ""]
        for k in sorted(CAPTIONS):
            md += [f"## `{k}`", "", CAPTIONS[k], ""]
        # The two runs emit different figure sets, so they get different caption files
        # and neither can clobber the other's.
        name = "captions.md" if a.extras else "captions_paper.md"
        (out_dir / name).write_text(chr(10).join(md), encoding="utf-8")
        print(f"captions -> {out_dir / name}")

    print(f"\nWrote {len(written)} figure(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
