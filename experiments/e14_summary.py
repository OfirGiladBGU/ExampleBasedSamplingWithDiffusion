#!/usr/bin/env python3
"""E14: seed robustness of Ours-GBN on the Icons-50 test set (1000 images).

Three independent runs per sampler (different Smart Init seed and noise seeds):
    seed 0   outputs/few_step_sampling-Icons-50_1024/CN-GBN_1024          (the E1 run)
    seed 1,2 outputs/e14_seeds-Icons-50_1024/CN-GBN-seed<k>_1024           (rebuttal_seeds.sbatch)

Per metric: the dataset mean of each run, the standard deviation of that mean across the runs (as
a percentage of the mean), and for scale the effects the paper and the rebuttal rely on, computed
from the seed-0 run with the same metric code:
    DDIM-25 vs DDPM-500, the gap to the GBN teacher, and the gap between Ours-GBN and Ours-WVS.
An effect is safe from sampling noise when it is many times the seed-to-seed spread.

Writes experiments/outputs/e14_seeds-Icons-50_1024/summary.md.

    python experiments/e14_summary.py
"""
import json
from pathlib import Path

import numpy as np

O = Path("experiments/outputs")
RUNS = [O / "few_step_sampling-Icons-50_1024/CN-GBN_1024",
        O / "e14_seeds-Icons-50_1024/CN-GBN-seed1_1024",
        O / "e14_seeds-Icons-50_1024/CN-GBN-seed2_1024"]
TEACHER = O / "rescore_current_metrics-Icons-50_1024/GBN_json"
OURS_WVS = O / "rescore_current_metrics-Icons-50_1024/CN-WVS-paper_json"
METRICS = [("M1_v1_cvt_energy", "CVT"), ("M1_v2_hot_energy", "HOT"), ("M2_v1_capacity_delta_c", "cap dc"),
           ("M2_v2_centroid_displacement", "centroid"), ("M3_emd_distance", "W2"),
           ("M4_sinkhorn_ot_cost", "Sinkhorn"), ("M5_spatial_measure_rho_mean", "rho_s")]


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def main():
    teacher, wvs = load(TEACHER), load(OURS_WVS)
    data = {s: [load(r / f"{s}_json") for r in RUNS] for s in ("ddpm-500", "ddim-25-eta1")}
    lines = ["# E14: seed robustness (Ours-GBN, Icons-50, 1000 test images, 3 independent runs)", "",
             "Spread = standard deviation of the dataset mean across the 3 runs, in % of the mean.", ""]
    for s, runs in data.items():
        keys = sorted(set.intersection(*(set(r) for r in runs)) & set(teacher) & set(wvs))
        other = data["ddpm-500" if s == "ddim-25-eta1" else "ddim-25-eta1"][0]
        lines += [f"## {s} ({len(keys)} images scored in all runs)", "",
                  "| metric | run 0 | run 1 | run 2 | spread | "
                  + ("DDIM-25 vs DDPM-500" if s == "ddim-25-eta1" else "DDPM-500 vs DDIM-25")
                  + " | gap to the GBN teacher | Ours-GBN vs Ours-WVS | smallest effect / spread |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m, lab in METRICS:
            means = np.array([np.mean([r[k][m] for k in keys]) for r in runs])
            spread = 100 * means.std(ddof=1) / means.mean()
            eff = [100 * (means[0] / np.mean([d[k][m] for k in keys]) - 1) for d in (other, teacher, wvs)]
            ratio = min(abs(e) for e in eff) / spread if spread > 0 else float("inf")
            lines.append(f"| {lab} | " + " | ".join(f"{v:.5g}" for v in means) + f" | {spread:.2f}% | "
                         + " | ".join(f"{e:+.1f}%" for e in eff) + f" | {ratio:.0f}x |")
        lines.append("")
    out = O / "e14_seeds-Icons-50_1024/summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
