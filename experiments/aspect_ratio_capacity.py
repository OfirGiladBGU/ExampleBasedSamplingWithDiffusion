#!/usr/bin/env python3
"""Aspect ratio vs capacity on the non-square test sets (rebuttal E3, step 1).

The model works on a square OT grid, and every image is mapped to [0,1]^2 per axis, so on a
non-square image the learned layout is stretched. If that causes the capacity gap between our
learned variants and their exemplar samplers on Anime Chibi and ShapeNet2D, the per-image gap
should grow with the image's anisotropy |log(w/h)|.

For each dataset and method this reports the Spearman correlation of each metric with the
anisotropy, the mean metric in anisotropy bins, and the same for the paired gap
(learned variant - its exemplar sampler, per image). Uses the stored per-image Table 1 metrics
(one scoring run per dataset, so the gaps are consistent within a dataset).

Writes outputs/aspect_ratio_capacity/summary.md and summary.json.

Run from the project root:
    python experiments/aspect_ratio_capacity.py
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = "experiments/outputs"
OUTPUT_DIR = "experiments/outputs/aspect_ratio_capacity"
DATASETS = ["Anime-Chibi_1600", "ShapeNet2D-4K_1600"]
SAMPLERS = ["BNOT", "GBN", "WVS"]
METRICS = {"M2_v1_capacity_delta_c": "cap dc", "M5_spatial_measure_rho_mean": "rho_s",
           "M1_v2_hot_energy": "HOT", "M2_v2_centroid_displacement": "centroid"}
BIN_EDGES = [0.0, 0.05, 0.2, 0.4, 10.0]  # on |log(w/h)|: square, near-square, mild, strong


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    return p.parse_args()


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def load(folder):
    return {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}


def bin_means(aniso, vals):
    out = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (aniso >= lo) & (aniso < hi)
        out.append((int(m.sum()), float(vals[m].mean()) if m.any() else float("nan")))
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    bins = [f"[{lo}, {hi})" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
    lines = ["# Aspect ratio vs metrics on the non-square test sets (E3 step 1)", "",
             "anisotropy a = |log(w/h)| of the source image. rho = Spearman correlation with a (p in brackets).",
             "gap = learned variant - its exemplar sampler, per image (paired).",
             f"Bins of a: {', '.join(bins)}; cells: mean (image count).", ""]
    for ds in args.datasets:
        base = Path(args.root) / f"quantitative_advance_metrics-{ds}"
        n = ds.rsplit("_", 1)[1]
        stems = sorted(p.stem for p in (base / "source").glob("*.png"))
        sizes = {s: png_size(base / "source" / f"{s}.png") for s in stems}
        per = {m: load(base / f"target_{m}_{n}_json") for m in SAMPLERS + [f"CN-{s}" for s in SAMPLERS]}
        common = [s for s in stems if all(s in v for v in per.values())]
        aniso = np.array([abs(np.log(sizes[s][0] / sizes[s][1])) for s in common])
        res[ds] = {"images": len(common), "anisotropy_quantiles": np.quantile(aniso, [0, .25, .5, .75, 1]).tolist(),
                   "bin_counts": [int(((aniso >= lo) & (aniso < hi)).sum()) for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])],
                   "metrics": {}}
        lines += [f"## {ds} ({len(common)} images; a quartiles "
                  + ", ".join(f"{q:.2f}" for q in res[ds]["anisotropy_quantiles"]) + ")", "",
                  "| metric | series | rho (p) | " + " | ".join(bins) + " |", "|" + "---|" * (3 + len(bins))]
        for key, short in METRICS.items():
            series = {m: np.array([per[m][s][key] for s in common]) for m in per}
            for s in SAMPLERS:
                series[f"gap CN-{s} - {s}"] = series[f"CN-{s}"] - series[s]
            for name, vals in series.items():
                rho, p = spearmanr(aniso, vals)
                bm = bin_means(aniso, vals)
                res[ds]["metrics"].setdefault(short, {})[name] = {"spearman": float(rho), "p": float(p), "bins": bm}
                lines.append(f"| {short} | {name} | {rho:+.2f} ({p:.1g}) | "
                             + " | ".join(f"{v:.4g} ({c})" for c, v in bm) + " |")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
