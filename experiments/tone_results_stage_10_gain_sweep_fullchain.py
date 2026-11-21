"""Stage 10: the ink-gain sweep on the real sampler (rebuttal; full-chain version of stage 2).

Stage 2 with --mode ddim re-runs stage 1 at several multiples of the calibrated ink gain, with the
gradient taken through the DDIM-25 chain: <sweep>/gain_<m>x/<stem>.npz. This stage then

    prepare    writes each requested density as an 8-bit condition image (intensity = 1 - rho):
               <sweep>/gain_<m>x/conditions/<config>/<stem>.png
    (sample)   few_step_sampling_stage_1.py on every condition folder with the paper's ddpm-500 and
               fresh noise seeds -> <sweep>/gain_<m>x/samples/<config>-seed<s>_1024/ddpm-500/<stem>.npy
    summarize  renders every stipple with the renderer AT THAT GAIN, scores it against the original
               target, and writes <sweep>/gain_sweep_real_sampler.csv (stage-2 columns plus paired statistics),
               summary.md, and the compact figure tone_gain_sweep_compact.{pdf,png}

Usage (from the project root; experiments/tone_gain_sweep_ddim25.sbatch runs all of it):
    python experiments/tone_results_stage_2_gain_sweep.py --mode ddim --output <sweep>
    python experiments/tone_results_stage_10_gain_sweep_fullchain.py --prepare
    (sampling)
    python experiments/tone_results_stage_10_gain_sweep_fullchain.py --summarize
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tone_results_utils import psnr, render_darkness, ssim  # noqa: E402

ROOT_DIR = "experiments/outputs/tone_results"

# --------------------------------------------------------------------------
# Defaults for every command-line argument. Edit here, not in parse_args.
# --------------------------------------------------------------------------
SWEEP_DIR = f"{ROOT_DIR}/gain_sweep_ddim25"
CONFIGS = "none,curve,field"
SAMPLER = "ddpm-500"
SEEDS = "0,1,2"
POINTS = 1024
DEVICE = "cpu"

# Figure, matching stage 3's tone_gain_sweep_compact (same size, series, log-2 x axis).
FIG_DPI = 300
FIGSIZE = (4.4, 3.2)
AXIS_LABELS = False
ZERO_LINE = True


def parse_args():
    p = argparse.ArgumentParser(description="Ink-gain sweep scored on the real sampler")
    p.add_argument("--sweep", default=SWEEP_DIR)
    p.add_argument("--configs", default=CONFIGS)
    p.add_argument("--sampler", default=SAMPLER)
    p.add_argument("--seeds", default=SEEDS)
    p.add_argument("--points", type=int, default=POINTS)
    p.add_argument("--device", default=DEVICE)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    return p.parse_args()


def gain_dirs(sweep):
    ds = [d for d in Path(sweep).glob("gain_*x") if (d / "summary.json").exists()]
    return sorted(ds, key=lambda d: float(d.name[len("gain_"):-1]))


def prepare(a):
    for d in gain_dirs(a.sweep):
        stems = sorted(f.stem for f in d.glob("*.npz"))
        for c in a.configs.split(","):
            out = d / "conditions" / c
            out.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                rho = np.load(d / f"{stem}.npz")[f"{c}_rho"]
                img = np.clip(np.rint((1.0 - rho) * 255.0), 0, 255).astype(np.uint8)
                Image.fromarray(img, mode="L").save(out / f"{stem}.png")
        print(f"{d.name}: {len(stems)} images x {a.configs} -> {d / 'conditions'}")


def summarize(a):
    configs = a.configs.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    rows, lines = [], []
    for d in gain_dirs(a.sweep):
        m = float(d.name[len("gain_"):-1])
        cfg = json.loads((d / "summary.json").read_text())["config"]
        res, sig, gain = cfg["render_res"], cfg["dot_sigma_px"], cfg["ink_gain"]
        per = {}                                          # config -> stem -> seed-mean (psnr, ssim, mae)
        for f in sorted(d.glob("*.npz")):
            z = np.load(f)
            t = torch.as_tensor(1.0 - z["image_01"].astype(np.float32))[None, None]
            t = torch.nn.functional.interpolate(t, size=(res, res), mode="area").numpy()[0, 0]
            for c in configs:
                sc = []
                for s in seeds:
                    p = d / "samples" / f"{c}-seed{s}_{a.points}" / a.sampler / f"{f.stem}.npy"
                    x = torch.as_tensor(np.load(p), dtype=torch.float32, device=a.device)[None]
                    with torch.no_grad():
                        r = render_darkness(x, res, sig, gain).cpu().numpy()[0, 0]
                    sc.append((psnr(r, t), ssim(r, t), float(np.abs(r - t).mean())))
                per.setdefault(c, {})[f.stem] = np.mean(sc, axis=0)
        stems = sorted(per["none"])
        ps = {c: np.array([per[c][s][0] for s in stems]) for c in configs}
        row = {"multiplier": m, "gain": gain, "images": len(stems)}
        for c in configs:
            row[f"{c}_psnr"] = float(ps[c].mean())
        for name, x, y in (("over_none", "field", "none"), ("over_curve", "field", "curve"),
                           ("curve_over_none", "curve", "none")):
            dd = ps[x] - ps[y]
            key = f"gain_{name}" if name != "curve_over_none" else "curve_gain_over_none"
            row[key] = float(dd.mean())
            row[f"{key}_improved"] = int((dd > 0).sum())
            row[f"{key}_p"] = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
        rows.append(row)
        lines.append(f"| x{m:g} | {gain:.3f} | {row['none_psnr']:.2f} | {row['curve_psnr']:.2f} | "
                     f"{row['field_psnr']:.2f} | {row['gain_over_none']:+.2f} "
                     f"({row['gain_over_none_improved']}/{len(stems)}, p={row['gain_over_none_p']:.1g}) | "
                     f"{row['gain_over_curve']:+.2f} ({row['gain_over_curve_improved']}/{len(stems)}, "
                     f"p={row['gain_over_curve_p']:.1g}) |")

    out = Path(a.sweep)
    keys = sorted({k for r in rows for k in r})
    with open(out / "gain_sweep_real_sampler.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = ["# Ink-gain sweep on the real sampler", "",
          f"Densities optimized through the DDIM-25 chain at each gain; stipples from {a.sampler} on "
          f"seeds {seeds} (per image: mean over seeds), rendered at the same gain, scored against the "
          "original target. PSNR in dB.", "",
          "| gain | value | none | curve | field | field - none | field - curve |",
          "|---|---|---|---|---|---|---|"] + lines
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["multiplier"] for r in rows]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for k, lab in (("gain_over_none", "vs uncorrected"), ("gain_over_curve", "vs tone curve")):
        ax.plot(x, [r[k] for r in rows], marker="o", ms=4, label=lab)
    if ZERO_LINE:
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    if AXIS_LABELS:
        ax.set_xlabel("ink gain / calibrated", fontsize=9)
        ax.set_ylabel("PSNR advantage (dB)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"tone_gain_sweep_compact.{ext}", dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"\n-> {out / 'gain_sweep_real_sampler.csv'}, {out / 'tone_gain_sweep_compact.pdf'}")


def main():
    a = parse_args()
    if a.prepare:
        prepare(a)
    else:
        summarize(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
