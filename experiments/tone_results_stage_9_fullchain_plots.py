"""Stage 9: the paper's two tone figures, rebuilt from the full-chain results (rebuttal, E16).

The submitted figures (stage 3) came from the one-step proxy of stage 1. Here every stipple is the
output of the real sampler (default: the paper's ddpm-500) on fresh noise seeds:

    requested densities   E16 stage-1 run through the DDIM-25 chain (<stage1>/<stem>.npz:
                          image_01, curve_rho, field_rho)
    our stipples          stage 5 samples, <samples>/<config>-seed<s>_1024/<sampler>/<stem>.npy
    feedback stipples     stage 7 (D1) fresh re-evaluation,
                          <feedback>/fresh-seed<s>/fp_1024/<sampler>/<stem>.npy

Everything is rendered with the stage-1 renderer and scored against the original target, so the
numbers match tone_results/full_sampler_ddim25/summary.md and tone_results/fixed_point/summary.md.

Writes to <output>/:
    tone_improvement.{pdf,png}                  per-image PSNR gain over uncorrected (mean over seeds)
    tone_explain_compact_<hero>.{pdf,png}       the six-panel mechanism figure, real-sampler stipples
    per_image.csv, summary.md                   the numbers behind both figures

Usage (from the project root):
    python experiments/tone_results_stage_9_fullchain_plots.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tone_results_stage_3_plots as s3                     # noqa: E402
from tone_results_utils import psnr, render_darkness, ssim  # noqa: E402

ROOT_DIR = "experiments/outputs/tone_results"

# --------------------------------------------------------------------------
# Defaults for every command-line argument. Edit here, not in parse_args.
# --------------------------------------------------------------------------
STAGE1_DIR = f"{ROOT_DIR}/ddim25"
SAMPLES_DIR = f"{ROOT_DIR}/full_sampler_ddim25/samples"
FEEDBACK_DIR = f"{ROOT_DIR}/fixed_point/ddpm-500"
OUTPUT = f"{ROOT_DIR}/plots_fullchain"
SAMPLER = "ddpm-500"
SEEDS = "0,1,2"
POINTS = 1024
HERO_STEM = "lg_2_woman_with_bunny_ears"
HERO_SEED = 0
DOT_SCALE = 1.8
DEVICE = "cpu"

REPLOT = False          # True: read <output>/per_image.csv instead of re-rendering every stipple

# tone_improvement, drawn here in stage 3's "lines" style (markers per image, dotted drop lines,
# images ordered by uncorrected PSNR) but with the legend in one row above the axes, so it
# cannot cover the largest gains.
IMPROVEMENT_SERIES = (("feedback", "Feedback (no gradients)", "^", "tab:orange"),
                      ("curve", "Global tone curve", "s", "tab:green"),
                      ("field", "Ours (optimized density)", "o", "tab:blue"))
IMPROVEMENT_FIGSIZE = (6.4, 3.2)
IMPROVEMENT_ZERO_LINE = True


def parse_args():
    p = argparse.ArgumentParser(description="Tone figures from the full-chain results")
    p.add_argument("--stage1", default=STAGE1_DIR)
    p.add_argument("--samples", default=SAMPLES_DIR)
    p.add_argument("--feedback", default=FEEDBACK_DIR)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--sampler", default=SAMPLER)
    p.add_argument("--seeds", default=SEEDS)
    p.add_argument("--points", type=int, default=POINTS)
    p.add_argument("--hero", default=HERO_STEM)
    p.add_argument("--hero-seed", type=int, default=HERO_SEED)
    p.add_argument("--dot-scale", type=float, default=DOT_SCALE)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--replot", action=argparse.BooleanOptionalAction, default=REPLOT)
    return p.parse_args()


def improvement_figure(per, stems, out_dir):
    """Per-image PSNR gain over uncorrected, one marker series per strategy."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, NullFormatter
    stems = sorted(stems, key=lambda s: per["none"][s][0])
    x = np.arange(1, len(stems) + 1)
    fig, ax = plt.subplots(figsize=IMPROVEMENT_FIGSIZE)
    top = None
    for c, label, mk, col in IMPROVEMENT_SERIES:
        d = np.array([per[c][s][0] - per["none"][s][0] for s in stems])
        ax.plot(x, d, ls="none", marker=mk, ms=4, color=col, label=label, zorder=3)
        top = d if top is None else np.maximum(top, d)
    if IMPROVEMENT_ZERO_LINE:
        ax.axhline(0.0, color="k", lw=0.8, ls=":", zorder=1)
    lo = ax.get_ylim()[0]
    ax.vlines(x, lo, np.maximum(top, 0.0), colors="0.65", linestyles=":", lw=0.8, zorder=1)
    ax.set_ylim(bottom=lo)
    ax.set_xlim(0.4, len(stems) + 0.6)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(IMPROVEMENT_SERIES),
              frameon=False, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    return s3.save(fig, out_dir / "tone_improvement")


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((Path(a.stage1) / "summary.json").read_text())["config"]
    res, sig, gain = cfg["render_res"], cfg["dot_sigma_px"], cfg["ink_gain"]
    seeds = [int(s) for s in a.seeds.split(",")]
    stems = sorted(f.stem for f in Path(a.stage1).glob("*.npz"))

    def pts_path(config, seed, stem):
        if config == "feedback":
            return Path(a.feedback) / f"fresh-seed{seed}" / f"fp_{a.points}" / a.sampler / f"{stem}.npy"
        return Path(a.samples) / f"{config}-seed{seed}_{a.points}" / a.sampler / f"{stem}.npy"

    def render(p):
        x = torch.as_tensor(np.load(p), dtype=torch.float32, device=a.device)[None]
        with torch.no_grad():
            return render_darkness(x, res, sig, gain).cpu().numpy()[0, 0]

    configs = ("none", "feedback", "curve", "field")
    rows, per = [], {}                          # per[config][stem] = (psnr, ssim, mae), seed mean
    if a.replot:
        for r in csv.DictReader(open(out / "per_image.csv", encoding="utf-8")):
            per.setdefault(r["config"], {})[r["stem"]] = np.array(
                [float(r["psnr"]), float(r["ssim"]), float(r["mae"])])
            rows.append(r)
    for stem in ([] if a.replot else stems):
        z = np.load(Path(a.stage1) / f"{stem}.npz")
        t = torch.as_tensor(1.0 - z["image_01"].astype(np.float32))[None, None]
        t = torch.nn.functional.interpolate(t, size=(res, res), mode="area").numpy()[0, 0]
        for c in configs:
            sc = []
            for s in seeds:
                f = pts_path(c, s, stem)
                if not f.exists():
                    raise FileNotFoundError(f)
                d = render(f)
                sc.append((psnr(d, t), ssim(d, t), float(np.abs(d - t).mean())))
            m = np.mean(sc, axis=0)
            per.setdefault(c, {})[stem] = m
            rows.append({"stem": stem, "config": c,
                         "psnr": f"{m[0]:.6f}", "ssim": f"{m[1]:.6f}", "mae": f"{m[2]:.6f}"})

    if not a.replot:
        with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["stem", "config", "psnr", "ssim", "mae"])
            w.writeheader()
            w.writerows(rows)

    lines = ["# Tone figures from the full-chain results", "",
             f"{len(stems)} targets, sampler {a.sampler}, seeds {seeds} (per image: mean over seeds). "
             f"Renderer: res {res}, sigma {sig} px, ink gain {gain:.4f}.", "",
             "| strategy | PSNR | SSIM | MAE | gain vs none (improved, p) |", "|---|---|---|---|---|"]
    base = np.array([per["none"][s][0] for s in stems])
    for c in configs:
        m = np.array([per[c][s] for s in stems])
        g = ""
        if c != "none":
            d = m[:, 0] - base
            g = f"{d.mean():+.2f} dB ({int((d > 0).sum())}/{len(d)}, p={wilcoxon(d).pvalue:.1g})"
        lines.append(f"| {c} | {m[:, 0].mean():.2f} | {m[:, 1].mean():.4f} | {m[:, 2].mean():.4f} | {g} |")
    for x, y in (("field", "curve"), ("field", "feedback"), ("curve", "feedback")):
        d = np.array([per[x][s][0] - per[y][s][0] for s in stems])
        lines.append(f"\n{x} - {y}: {d.mean():+.2f} dB mean, better on {int((d > 0).sum())}/{len(d)} "
                     f"(p={wilcoxon(d).pvalue:.1g})")

    improvement_figure(per, stems, out)

    # tone_explain_compact: the hero target with real-sampler stipples of one seed
    z = np.load(Path(a.stage1) / f"{a.hero}.npz")
    hero = {"image_01": z["image_01"], "field_rho": z["field_rho"]}
    for c in ("none", "field"):
        f = pts_path(c, a.hero_seed, a.hero)
        hero[f"{c}_coords"] = np.load(f)
        hero[f"{c}_darkness"] = render(f)
    s3.explain_compact_figure(hero, a.hero, out, dot_scale=a.dot_scale)
    hp = {c: per[c][a.hero][0] - per["none"][a.hero][0] for c in configs if c != "none"}
    lines.append(f"\nHero {a.hero}, seed {a.hero_seed} in the figure. Seed-mean PSNR gain on this "
                 "target: " + ", ".join(f"{c} {v:+.2f} dB" for c, v in hp.items()))
    lines += ["", "Figure caption (as stage 3 records it):", "",
              s3.CAPTIONS.get(f"tone_explain_compact_{a.hero}", "")]

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nfigures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
