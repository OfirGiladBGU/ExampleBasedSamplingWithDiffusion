#!/usr/bin/env python3
"""Do the learned variants keep the ranking of their exemplar samplers? (rebuttal E2)

For every dataset, metric and pair of exemplar samplers (BNOT, GBN, WVS), the order of the two
samplers is compared with the order of the two matching learned variants (Ours-BNOT, Ours-GBN,
Ours-WVS). The order of a pair is decided per image with a paired Wilcoxon signed-rank test:
"a < b" or "a > b" when p < alpha, a tie otherwise. Pairs are then counted as
    agree       both orders decided, same direction
    flip        both orders decided, opposite directions
    tie         at least one of the two orders is a statistical tie
The means-based count (sign of the difference of the means, ties at the table's rounding) is
reported too, as is Kendall's tau between the sampler and learned-variant means over all pairs.

All six methods must be scored by the same metric code, so this reads the re-scored folders
outputs/rescore_current_metrics-<Dataset>/<METHOD>_json (paper point sets, current metric code).

Writes outputs/rank_preservation/summary.md and summary.json.

Run from the project root:
    python experiments/rank_preservation.py
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, wilcoxon

ROOT = "experiments/outputs"
OUTPUT_DIR = "experiments/outputs/rank_preservation"
DATASETS = ["Icons-50_1024", "CelebA-5K_1024", "ShapeNet2D-4K_1600", "Anime-Chibi_1600"]
SAMPLERS = ["BNOT", "GBN", "WVS"]
LEARNED = "CN-{}-paper"
METRICS = {"M1_v1_cvt_energy": ("CVT", 1e5, 3), "M1_v2_hot_energy": ("HOT", 1e5, 3),
           "M2_v1_capacity_delta_c": ("cap dc", 1, 3), "M2_v2_centroid_displacement": ("D", 1, 3),
           "M3_emd_distance": ("EMD", 1, 3), "M4_sinkhorn_ot_cost": ("Sinkhorn", 1, 3),
           "M5_spatial_measure_rho_mean": ("rho_s", 1, 3)}
ALPHA = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--alpha", type=float, default=ALPHA)
    return p.parse_args()


def load(folder):
    return {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}


def order(a, b, alpha):
    """-1 if a < b, +1 if a > b (paired Wilcoxon p < alpha), 0 for a tie."""
    d = a - b
    if not np.any(d != 0):
        return 0, 1.0
    p = float(wilcoxon(a, b).pvalue)
    return (int(np.sign(d.mean())) if p < alpha else 0), p


def mean_order(a, b, scale, digits):
    ra, rb = round(float(a.mean()) * scale, digits), round(float(b.mean()) * scale, digits)
    return int(np.sign(ra - rb))


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, sampler_means, learned_means = [], [], []
    for ds in args.datasets:
        base = Path(args.root) / f"rescore_current_metrics-{ds}"
        per = {m: load(base / f"{m}_json") for m in SAMPLERS}
        per.update({LEARNED.format(m): load(base / f"{LEARNED.format(m)}_json") for m in SAMPLERS})
        common = sorted(set.intersection(*(set(v) for v in per.values())))
        for key, (short, scale, digits) in METRICS.items():
            vals = {m: np.array([per[m][s][key] for s in common]) for m in per}
            for m in SAMPLERS:
                sampler_means.append(vals[m].mean())
                learned_means.append(vals[LEARNED.format(m)].mean())
            for i, j in itertools.combinations(SAMPLERS, 2):
                ts, tp = order(vals[i], vals[j], args.alpha)
                ls, lp = order(vals[LEARNED.format(i)], vals[LEARNED.format(j)], args.alpha)
                tm = mean_order(vals[i], vals[j], scale, digits)
                lm = mean_order(vals[LEARNED.format(i)], vals[LEARNED.format(j)], scale, digits)
                rows.append({"dataset": ds, "images": len(common), "metric": short, "pair": f"{i}-{j}",
                             "sampler_order": ts, "sampler_p": tp, "learned_order": ls, "learned_p": lp,
                             "sampler_mean_order": tm, "learned_mean_order": lm,
                             "sampler_means": [float(vals[i].mean()), float(vals[j].mean())],
                             "learned_means": [float(vals[LEARNED.format(i)].mean()),
                                               float(vals[LEARNED.format(j)].mean())]})

    def count(key_s, key_l):
        decided = [r for r in rows if r[key_s] != 0 and r[key_l] != 0]
        agree = sum(r[key_s] == r[key_l] for r in decided)
        return agree, len(decided) - agree, len(rows) - len(decided)

    wa, wf, wt = count("sampler_order", "learned_order")
    ma, mf, mt = count("sampler_mean_order", "learned_mean_order")
    # Kendall's tau between the 3 sampler means and the 3 learned-variant means, per (dataset, metric)
    taus = [float(kendalltau(sampler_means[k:k + 3], learned_means[k:k + 3]).statistic)
            for k in range(0, len(sampler_means), 3)]
    lines = ["# Rank preservation: exemplar samplers vs learned variants (E2)", "",
             f"{len(rows)} pairs = {len(args.datasets)} datasets x {len(METRICS)} metrics x 3 sampler pairs.",
             f"Kendall's tau of the 3 sampler means vs the 3 learned means, per (dataset, metric): "
             f"mean {np.mean(taus):.2f}, tau = 1 in {sum(t == 1 for t in taus)} of {len(taus)}.", "",
             "| decision rule | agree | flip | tie (either side) |", "|---|---|---|---|",
             f"| paired Wilcoxon, p < {args.alpha} | {wa} | {wf} | {wt} |",
             f"| sign of the means at table rounding | {ma} | {mf} | {mt} |", "",
             "Flips and ties (Wilcoxon rule):", "",
             "| dataset | metric | pair | samplers (means, p) | learned (means, p) | result |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["sampler_order"] != 0 and r["sampler_order"] == r["learned_order"]:
            continue
        res = "flip" if r["sampler_order"] * r["learned_order"] == -1 else "tie"
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['pair']} | "
                     f"{r['sampler_means'][0]:.4g} / {r['sampler_means'][1]:.4g} (p={r['sampler_p']:.2g}) | "
                     f"{r['learned_means'][0]:.4g} / {r['learned_means'][1]:.4g} (p={r['learned_p']:.2g}) | {res} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({"wilcoxon": {"agree": wa, "flip": wf, "tie": wt},
                                                      "means": {"agree": ma, "flip": mf, "tie": mt},
                                                      "kendall_tau_per_dataset_metric": taus,
                                                      "rows": rows}, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
