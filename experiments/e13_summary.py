#!/usr/bin/env python3
"""E13 summary: Table 1 with the new variant (ddim-25-eta1, exact OT) for all 4 test sets.

Per dataset and metric: means over the test images of the three samplers (BNOT, WVS, GBN) and the
three learned variants sampled with ddim-25-eta1 (outputs/e13_ddim25-<dataset>), all scored with
the current metric code (samplers: rescore_current_metrics-<dataset>/<S>_json). Second table: the
change of each learned variant against its paper point sets (CN-<S>-paper_json, DDPM-500), paired
Wilcoxon over the same images.

Writes experiments/outputs/e13_ddim25_summary.md.

    python experiments/e13_summary.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

O = Path("experiments/outputs")
DATASETS = ["Icons-50_1024", "CelebA-5K_1024", "ShapeNet2D-4K_1600", "Anime-Chibi_1600"]
SAMPLERS = ["BNOT", "WVS", "GBN"]
METRICS = [("M1_v1_cvt_energy", "CVT x1e5", 1e5, True), ("M1_v2_hot_energy", "HOT x1e5", 1e5, True),
           ("M2_v1_capacity_delta_c", "cap dc", 1, True), ("M2_v2_centroid_displacement", "centroid D", 1, True),
           ("M3_emd_distance", "W2 (EMD)", 1, True), ("M4_sinkhorn_ot_cost", "Sinkhorn", 1, True),
           ("M5_spatial_measure_rho_mean", "rho_s", 1, False)]


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def main():
    lines = ["# E13: Table 1 with the new variant (ddim-25-eta1, SDEdit start, exact OT)", "",
             "Means over each test set, current metric code. Best of the six in bold, best learned in italics.", ""]
    change = ["## Change vs the paper's point sets (DDPM-500), paired Wilcoxon (* p < 0.05)", "",
              "| dataset | variant | images | " + " | ".join(m[1] for m in METRICS) + " |",
              "|---|---|---|" + "---|" * len(METRICS)]
    for ds in DATASETS:
        ref = {s: load(O / f"rescore_current_metrics-{ds}/{s}_json") for s in SAMPLERS}
        n = int(ds.rsplit("_", 1)[1])
        new = {s: load(O / f"e13_ddim25-{ds}/CN-{s}_{n}/ddim-25-eta1_json") for s in SAMPLERS}
        paper = {s: load(O / f"rescore_current_metrics-{ds}/CN-{s}-paper_json") for s in SAMPLERS}
        cols = SAMPLERS + [f"Ours-{s}" for s in SAMPLERS]
        data = {**ref, **{f"Ours-{s}": new[s] for s in SAMPLERS}}
        common = sorted(set.intersection(*(set(v) for v in data.values())))
        lines += [f"## {ds} ({len(common)} images)", "", "| metric | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for key, lab, scale, lower in METRICS:
            means = {c: scale * np.mean([data[c][k][key] for k in common]) for c in cols}
            pick = min if lower else max
            best = pick(means.values())
            best_l = pick(means[c] for c in cols[3:])
            cells = []
            for c in cols:
                s = f"{means[c]:.4g}"
                s = f"**{s}**" if means[c] == best else (f"*{s}*" if c in cols[3:] and means[c] == best_l else s)
                cells.append(s)
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
        lines.append("")
        for s in SAMPLERS:
            ks = sorted(set(new[s]) & set(paper[s]))
            cells = []
            for key, lab, scale, lower in METRICS:
                a = np.array([new[s][k][key] for k in ks]); b = np.array([paper[s][k][key] for k in ks])
                p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                cells.append(f"{100 * (a.mean() / b.mean() - 1):+.1f}%{'*' if p < 0.05 else ''}")
            change.append(f"| {ds} | Ours-{s} | {len(ks)} | " + " | ".join(cells) + " |")
    out = O / "e13_ddim25_summary.md"
    out.write_text("\n".join(lines + [""] + change) + "\n")
    print("\n".join(lines + [""] + change))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
