#!/usr/bin/env python3
"""Ablation table with the few-step sampler (supplement): summary.

Reads experiments/outputs/ablation_ddim25_e5000_b50_1024/CN-<variant>_1024/<sampler>_json (variants
sampled by ablation_ddim25.sbatch at epoch 5000 from the Smart Init + SDEdit start, 50 Table 2
images, same Smart Init and noise for every variant).

Per sampler: the mean of every metric per variant (the table), and each variant against the full
model, paired per image: median per-image change, share of images where the variant is worse, and
a verdict from the paired Wilcoxon test (direction from the paired columns, not from the means).

Writes <output>/summary.md.

    python experiments/ablation_ddim25_summary.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path("experiments/outputs/ablation_ddim25_e5000_b50_1024")
N = 1024
VARIANTS = [("vanilla", "Vanilla (frozen base, plain control)"), ("unfrozen", "Unfrozen base"),
            ("gecco", "GECCO features"), ("agi", "Gated injection"),
            ("noGate", "Full without gated injection"), ("noKDE", "Full without the KDE loss"),
            ("full", "Full")]
METRICS = [("M1_v1_cvt_energy", "CVT x1e5", 1e5, True), ("M1_v2_hot_energy", "HOT x1e5", 1e5, True),
           ("M2_v1_capacity_delta_c", "cap dc", 1, True), ("M2_v2_centroid_displacement", "centroid D", 1, True),
           ("M3_emd_distance", "W2", 1, True), ("M4_sinkhorn_ot_cost", "Sinkhorn", 1, True),
           ("M5_spatial_measure_rho_mean", "rho_s", 1, False)]


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def main():
    lines = ["# Ablation with the few-step sampler (epoch 5000, 50 Table 2 images, SDEdit start)", "",
             "Every variant uses the same Smart Init and noise per image. Unlike the paper's Table 2, all",
             "rows start from the Smart Init (SDEdit); the paper samples its architecture rows from pure noise.", ""]
    for sampler in ("ddim-25-eta1", "ddpm-500"):
        data = {v: load(OUT / f"CN-{v}_{N}" / f"{sampler}_json") for v, _ in VARIANTS}
        data = {v: d for v, d in data.items() if d}
        if "full" not in data:
            continue
        keys = sorted(set.intersection(*(set(d) for d in data.values())))
        lines += [f"## {sampler} ({len(keys)} images)", "", "### Means", "",
                  "| variant | " + " | ".join(l for _, l, _, _ in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for v, name in VARIANTS:
            if v in data:
                lines.append(f"| {name} | " + " | ".join(
                    f"{s * np.mean([data[v][k][m] for k in keys]):.4g}" for m, _, s, _ in METRICS) + " |")
        lines += ["", "### Each variant against the full model (paired): median per-image change, "
                  "share of images worse, verdict (p < 0.05)", "",
                  "| variant | " + " | ".join(l for _, l, _, _ in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for v, name in VARIANTS:
            if v == "full" or v not in data:
                continue
            cells = []
            for m, _, _, lower in METRICS:
                a = np.array([data[v][k][m] for k in keys]); b = np.array([data["full"][k][m] for k in keys])
                p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                worse = float(((a > b) if lower else (a < b)).mean())
                verdict = "=" if p >= 0.05 else ("worse" if worse > 0.5 else "better")
                cells.append(f"{100 * (np.median(a / b) - 1):+.1f}%, {100 * worse:.0f}%, {verdict}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
