#!/usr/bin/env python3
"""Cross-dataset generalization summary (rebuttal E5).

A checkpoint trained on one dataset stipples the test set of another (rebuttal_sample.sbatch,
outputs/cross_dataset-<target>/CN-<M>-from<source>_<N>/<sampler>). For each target dataset and
variant this compares, on the same images:
    cross    the cross-domain checkpoint (ddpm-500, the paper sampler, and ddim-25-eta1)
    in       the in-domain paper point set (rescore_current_metrics-<target>/CN-<M>-paper, ddpm-500)
    sampler  the exemplar sampler itself (rescore_current_metrics-<target>/<M>)
All scored with the same (current) metric code. Cells: mean, and for "cross" the change vs the
in-domain result with a paired Wilcoxon test.

Writes outputs/cross_dataset_summary.md / .json.

Run from the project root:
    python experiments/cross_dataset_stage_3.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = "experiments/outputs"
OUTPUT = "experiments/outputs/cross_dataset_summary"
SAMPLERS = ["ddpm-500", "ddim-25-eta1"]
METRICS = {"M1_v2_hot_energy": ("HOT", 1e5), "M2_v1_capacity_delta_c": ("cap dc", 1),
           "M2_v2_centroid_displacement": ("D", 1), "M3_emd_distance": ("EMD", 1),
           "M5_spatial_measure_rho_mean": ("rho_s", 1)}
P_THRESHOLD = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--output", default=OUTPUT)
    return p.parse_args()


def load(folder):
    return {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}


def main():
    args = parse_args()
    root = Path(args.root)
    res, lines = {}, ["# Cross-dataset generalization (E5)", "",
                      "cross = checkpoint trained on another dataset; in = the in-domain paper result; "
                      "sampler = the exemplar sampler. HOT x1e5. Cross cells: mean (change vs in-domain, "
                      f"* = paired Wilcoxon p < {P_THRESHOLD}).", ""]
    for target_dir in sorted(root.glob("cross_dataset-*")):
        if not target_dir.is_dir():
            continue
        target = target_dir.name.split("-", 1)[1]
        rescore = root / f"rescore_current_metrics-{target}"
        lines += [f"## Target: {target}", "",
                  "| variant | trained on | sampler | images | " + " | ".join(s for s, _ in METRICS.values()) + " |",
                  "|" + "---|" * (4 + len(METRICS))]
        for run in sorted(d for d in target_dir.iterdir() if d.is_dir()):
            label = run.name.rsplit("_", 1)[0]                 # CN-GBN-fromIcons
            variant, source = label.split("-from")
            method = variant.split("-", 1)[1]                  # GBN
            ref_in = load(rescore / f"{variant}-paper_json")
            ref_s = load(rescore / f"{method}_json")
            for s in SAMPLERS:
                cross = load(run / f"{s}_json")
                common = sorted(set(cross) & set(ref_in) & set(ref_s))
                if not common:
                    continue
                entry = {}
                for name, data in (("sampler", ref_s), ("in", ref_in)):
                    if s != SAMPLERS[0]:
                        continue
                    cells = [variant if name == "in" else method, target if name == "in" else "-",
                             "ddpm-500" if name == "in" else "-", str(len(common))]
                    for key, (_, scale) in METRICS.items():
                        cells.append(f"{np.mean([data[k][key] for k in common]) * scale:.4g}")
                    lines.append("| " + " | ".join(cells) + " |")
                cells = [variant, source, s, str(len(common))]
                for key, (short, scale) in METRICS.items():
                    c = np.array([cross[k][key] for k in common])
                    r = np.array([ref_in[k][key] for k in common])
                    p = float(wilcoxon(c, r).pvalue) if np.any(c != r) else 1.0
                    rel = float(c.mean() / r.mean() - 1.0)
                    entry[short] = {"cross": float(c.mean()), "in": float(r.mean()),
                                    "sampler": float(np.mean([ref_s[k][key] for k in common])),
                                    "rel_change": rel, "p": p}
                    cells.append(f"{c.mean() * scale:.4g} ({rel * 100:+.1f}%{'*' if p < P_THRESHOLD else ''})")
                lines.append("| " + " | ".join(cells) + " |")
                res.setdefault(target, {}).setdefault(variant, {})[f"{source}/{s}"] = entry
        lines.append("")
    out = Path(args.output)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
