#!/usr/bin/env python3
"""Generalization diagnostics: summary of A (train vs held-out) and B (start from the exemplar).

Reads the scored run folders under experiments/outputs/diag_generalization:
    CN-GBN-{train,test}_1024/{GBN-exemplar, ddpm-500, ddim-25-eta1}
    CN-GBN-{train,test}-fromGT_1024/ddpm-K        (start from the exemplar noised to step K)

A. Per image, the ratio learned / teacher-exemplar of each metric, averaged over the training and
   over the held-out images (Mann-Whitney U between the two sets of ratios). Ratios near 1 on the
   training images and above 1 on the held-out ones = a generalization (data) gap; the same ratio
   on both = the model does not fit the regularity even where it was trained.
   Also the mean distance from each generated point to the nearest point of that image's exemplar,
   in cell widths: clearly smaller on training images = the exemplars are memorised.
B. The same ratios when sampling starts from the teacher's exemplar noised to step K: the smallest
   K at which the ratio leaves 1 is the noise level where the regularity is lost.

Writes <output>/summary.md.

    python experiments/diag_generalization_summary.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

OUT = Path("experiments/outputs/diag_generalization")
N = 1024
GRID = 32
METRICS = [("M1_v1_cvt_energy", "CVT"), ("M2_v1_capacity_delta_c", "cap dc"),
           ("M5_spatial_measure_rho_mean", "rho_s"), ("M1_v2_hot_energy", "HOT"),
           ("M2_v2_centroid_displacement", "centroid"), ("M3_emd_distance", "W2")]
FROM_GT = [10, 20, 50, 100, 200, 300, 500]


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def nn_dist(run, sampler, exemplar_dir, stems):
    """Mean distance generated point -> nearest exemplar point, in cell widths."""
    vals = []
    for s in stems:
        g, e = run / sampler / f"{s}.npy", exemplar_dir / f"{s}.npy"
        if g.exists() and e.exists():
            vals.append(cKDTree(np.load(e)).query(np.load(g))[0].mean() * GRID)
    return float(np.mean(vals)) if vals else float("nan")


def ratios(run, sampler, exemplar):
    r = load(run / f"{sampler}_json")
    keys = sorted(set(r) & set(exemplar))
    return keys, {m: np.array([r[k][m] / exemplar[k][m] for k in keys]) for m, _ in METRICS}


def main():
    ex = {s: load(OUT / f"CN-GBN-{s}_{N}" / "GBN-exemplar_json") for s in ("train", "test")}
    ex_dir = {s: OUT / f"CN-GBN-{s}_{N}" / "GBN-exemplar" for s in ("train", "test")}
    lines = ["# Generalization diagnostics (Ours-GBN, Icons-50, 1024 points)", "",
             "Ratios are learned / teacher exemplar per image, then averaged (1.00 = the teacher's value;",
             "rho_s is higher-is-better, the others lower-is-better). p: Mann-Whitney U, training vs held-out.", "",
             "## Teacher exemplar means", "", "| images | " + " | ".join(l for _, l in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for s in ("train", "test"):
        lines.append(f"| {s} ({len(ex[s])}) | " + " | ".join(
            f"{np.mean([v[m] for v in ex[s].values()]):.4g}" for m, _ in METRICS) + " |")

    lines += ["", "## A. Normal sampling (Smart Init + SDEdit): training vs held-out images", "",
              "| sampler | metric | ratio, training | ratio, held-out | p |", "|---|---|---|---|---|"]
    for sampler in ("ddpm-500", "ddim-25-eta1"):
        kt, rt = ratios(OUT / f"CN-GBN-train_{N}", sampler, ex["train"])
        kv, rv = ratios(OUT / f"CN-GBN-test_{N}", sampler, ex["test"])
        for m, lab in METRICS:
            p = mannwhitneyu(rt[m], rv[m]).pvalue
            lines.append(f"| {sampler} | {lab} | {rt[m].mean():.3f} | {rv[m].mean():.3f} | {p:.2g} |")
        dt = nn_dist(OUT / f"CN-GBN-train_{N}", sampler, ex_dir["train"], kt)
        dv = nn_dist(OUT / f"CN-GBN-test_{N}", sampler, ex_dir["test"], kv)
        lines.append(f"| {sampler} | distance to the exemplar's points (cells) | {dt:.3f} | {dv:.3f} | |")

    lines += ["", "## B. Start from the teacher's exemplar noised to step K, then DDPM from K", "",
              "| K | noise std (cells) | " + " | ".join(f"{l} train / held-out" for _, l in METRICS[:3])
              + " | moved from the exemplar (cells) train / held-out |", "|---|---|" + "---|" * 4]
    betas = np.linspace(1e-4, 1e-2, 1000)
    ac = np.cumprod(1 - betas)
    for k in FROM_GT:
        cells = []
        try:
            kt, rt = ratios(OUT / f"CN-GBN-train-fromGT_{N}", f"ddpm-{k}", ex["train"])
            kv, rv = ratios(OUT / f"CN-GBN-test-fromGT_{N}", f"ddpm-{k}", ex["test"])
        except Exception:
            continue
        if not kt or not kv:
            continue
        for m, _ in METRICS[:3]:
            cells.append(f"{rt[m].mean():.3f} / {rv[m].mean():.3f}")
        dt = nn_dist(OUT / f"CN-GBN-train-fromGT_{N}", f"ddpm-{k}", ex_dir["train"], kt)
        dv = nn_dist(OUT / f"CN-GBN-test-fromGT_{N}", f"ddpm-{k}", ex_dir["test"], kv)
        lines.append(f"| {k} | {np.sqrt(1 - ac[min(k, 999)]):.3f} | " + " | ".join(cells) + f" | {dt:.3f} / {dv:.3f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
