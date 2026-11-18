#!/usr/bin/env python3
"""Fig. 3 strip percentages over realizations (reviewer: "one realization or an average?").

The paper's top row uses rho(x) = x^2 + 1/500 on the unit square, split into four equal-width
vertical strips with target point masses 1.7 / 11.0 / 29.7 / 57.6 %. The numbers printed under the
panels come from ONE stored point set per method
(outputs/images_results_metrics/quadratic/target_<METHOD>_1024/quadratic_density_gradient.npy).

This script reports, for Ours-GBN and Ours-WVS, the mean and standard deviation of each strip's
share over the realizations sampled by rebuttal_seeds.sbatch (different Smart Init and noise seeds),
next to the stored single realization of every method.

Writes experiments/outputs/fig3_strip_realizations/summary.md.

    python experiments/fig3_strip_realizations.py
"""
from pathlib import Path

import numpy as np

OUT = Path("experiments/outputs/fig3_strip_realizations")
STORED = Path("experiments/outputs/images_results_metrics/quadratic")
STEM = "quadratic_density_gradient"
N = 1024
STRIPS = 4


def target_shares():
    edges = np.linspace(0, 1, STRIPS + 1)
    mass = (edges[1:] ** 3 - edges[:-1] ** 3) / 3 + (edges[1:] - edges[:-1]) / 500
    return 100 * mass / mass.sum()


def shares(points):
    x = np.clip(points[:, 0], 0, 1 - 1e-9)
    return 100 * np.bincount((x * STRIPS).astype(int), minlength=STRIPS) / len(points)


def main():
    tgt = target_shares()
    lines = ["# Fig. 3 strip percentages (quadratic density gradient, 1024 points, 4 vertical strips)", "",
             "Target shares: " + " / ".join(f"{v:.1f}%" for v in tgt), "",
             "## Stored point sets (one realization each; the numbers printed in the paper's figure)", "",
             "| method | " + " | ".join(f"strip {i + 1}" for i in range(STRIPS)) + " | max abs error (pp) |",
             "|---|" + "---|" * (STRIPS + 1)]
    for m in ("BNOT", "WVS", "GBN", "CN-WVS", "CN-GBN"):
        f = STORED / f"target_{m}_{N}" / f"{STEM}.npy"
        if f.exists():
            s = shares(np.load(f))
            lines.append(f"| {m.replace('CN-', 'Ours-')} | " + " | ".join(f"{v:.1f}" for v in s)
                         + f" | {np.abs(s - tgt).max():.1f} |")
    lines += ["", "## Ours, over realizations (mean ± std; new Smart Init and noise seeds per realization)", "",
              "| variant | sampler | realizations | " + " | ".join(f"strip {i + 1}" for i in range(STRIPS))
              + " | max abs error of the mean (pp) | worst single realization (pp) |",
              "|---|---|---|" + "---|" * (STRIPS + 2)]
    for m in ("GBN", "WVS"):
        for sampler in ("ddpm-500", "ddim-25-eta1"):
            rows = [shares(np.load(f)) for f in sorted(OUT.glob(f"CN-{m}-s*_{N}/{sampler}/{STEM}.npy"))]
            if not rows:
                continue
            a = np.array(rows)
            mean, std = a.mean(0), a.std(0, ddof=1) if len(a) > 1 else np.zeros(STRIPS)
            lines.append(f"| Ours-{m} | {sampler} | {len(a)} | "
                         + " | ".join(f"{u:.1f} ± {v:.1f}" for u, v in zip(mean, std))
                         + f" | {np.abs(mean - tgt).max():.1f} | {np.abs(a - tgt).max():.1f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
