#!/usr/bin/env python3
"""Low-noise fine-tune: trend over fine-tune epochs (is it still improving?).

For one variant (default g0 = no Min-SNR), compares every evaluated epoch with the paper checkpoint
(epoch 5000) on the 200 held-out diagnostic images, paired per image. Run folders under
experiments/outputs/diag_generalization: CN-GBN-test-ft-<tag>-ep<E>_1024 (epoch 5300 is
CN-GBN-test-ft-<tag>_1024), the matching -fromGT folders, and timestep_loss_ft-<tag>[-ep<E>].json.

Per epoch and sampler: mean / the teacher exemplar's mean, the median per-image change against the
paper checkpoint, and the share of images where the fine-tuned model is better (* = paired
Wilcoxon p < 0.05; read the direction from the median and the share, not from the means).

Writes <output>/ft_lownoise_trend_<tag>.md.

    python experiments/ft_lownoise_trend.py
    python experiments/ft_lownoise_trend.py --tag g0 --epochs 5100 5200 5300 5600 6000
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path("experiments/outputs/diag_generalization")
N = 1024
EPOCHS = [5100, 5200, 5300, 5600, 6000]
METRICS = [("M2_v1_capacity_delta_c", "cap dc", True), ("M5_spatial_measure_rho_mean", "rho_s", False),
           ("M1_v1_cvt_energy", "CVT", True), ("M1_v2_hot_energy", "HOT", True),
           ("M2_v2_centroid_displacement", "centroid", True), ("M3_emd_distance", "W2", True)]


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def run_name(tag, epoch, suffix=""):
    ep = "" if epoch == 5300 else f"-ep{epoch}"
    return f"CN-GBN-test-ft-{tag}{ep}{suffix}_{N}"


def cell(base, var, ex, key, lower):
    keys = sorted(set(base) & set(var) & set(ex))
    b = np.array([base[k][key] for k in keys]); v = np.array([var[k][key] for k in keys])
    e = np.array([ex[k][key] for k in keys])
    p = wilcoxon(v, b).pvalue if np.any(v != b) else 1.0
    better = float(((v < b) if lower else (v > b)).mean())
    return f"{v.mean() / e.mean():.3f} ({100 * (np.median(v / b) - 1):+.1f}%, {100 * better:.0f}%{'*' if p < 0.05 else ''})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="g0")
    ap.add_argument("--epochs", type=int, nargs="+", default=EPOCHS)
    a = ap.parse_args()
    ex = load(OUT / f"CN-GBN-test_{N}" / "GBN-exemplar_json")
    lines = [f"# Low-noise fine-tune ft-{a.tag}: trend over epochs (200 held-out images, paired vs epoch 5000)", "",
             "Cells: mean / teacher's mean (median per-image change vs the paper checkpoint, share of images",
             "better; * = paired Wilcoxon p < 0.05). rho_s is higher-is-better, the others lower-is-better.", ""]
    for title, suffix, samplers in (("Normal sampling", "", ["ddpm-500", "ddim-25-eta1"]),
                                    ("Start from the teacher's exemplar noised to step K", "-fromGT",
                                     ["ddpm-10", "ddpm-50", "ddpm-500"])):
        base_run = OUT / f"CN-GBN-test{suffix}_{N}"
        for s in samplers:
            base = load(base_run / f"{s}_json")
            if not base:
                continue
            keys = sorted(set(base) & set(ex))
            lines += [f"## {title}, {s}", "", "| epoch | " + " | ".join(l for _, l, _ in METRICS) + " |",
                      "|---|" + "---|" * len(METRICS),
                      "| 5000 (paper) | " + " | ".join(
                          f"{np.mean([base[k][m] for k in keys]) / np.mean([ex[k][m] for k in keys]):.3f}"
                          for m, _, _ in METRICS) + " |"]
            for ep in a.epochs:
                var = load(OUT / run_name(a.tag, ep, suffix) / f"{s}_json")
                if len(var) < len(keys):
                    continue
                lines.append(f"| {ep} | " + " | ".join(cell(base, var, ex, m, lo) for m, _, lo in METRICS) + " |")
            lines.append("")

    tabs = {}
    if (OUT / "timestep_loss.json").exists():
        tabs["5000 (paper)"] = json.loads((OUT / "timestep_loss.json").read_text())
    for ep in a.epochs:
        f = OUT / (f"timestep_loss_ft-{a.tag}.json" if ep == 5300 else f"timestep_loss_ft-{a.tag}-ep{ep}.json")
        if f.exists():
            tabs[str(ep)] = json.loads(f.read_text())
    if len(tabs) > 1:
        names = list(tabs)
        lines += ["## x0 RMS error in cells, held-out (training) images", "",
                  "| t | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for i, row in enumerate(next(iter(tabs.values()))):
            if row["t"] in (2, 10, 20, 50, 100, 300, 500, 800):
                lines.append(f"| {row['t']} | " + " | ".join(
                    f"{tabs[n][i]['x0_rms_test']:.3f} ({tabs[n][i]['x0_rms_train']:.3f})" for n in names) + " |")
        lines.append("")
    out = OUT / f"ft_lownoise_trend_{a.tag}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
