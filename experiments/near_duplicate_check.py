#!/usr/bin/env python3
"""Do near-duplicate training icons drive the Icons-50 test results? (reviewer concern W6)

The Icons-50 test set is the trainer's held-out 10% (seed-42 randperm, verified identical to
z_validation_data/Icons-50_1024/validation_manifest.json), but 997 of its 1000 icons have the same
emoji drawn by another vendor (or another version) in the training set, e.g. apple_0_duck in test
and google_2_duck in training. Too few test icons have an unseen concept to split on, so this
script measures, per test icon, how similar its closest same-concept training icon is, and checks
whether the learned sampler does better on test icons with closer training look-alikes.

Similarity: Pearson correlation of the 128 x 128 grayscale images (area resampling) between the
test icon and each training icon of the same concept; the maximum is the icon's look-alike score.
Quality: the learned variant relative to its teacher on the same icon (ratio of the per-image
metric, so the icon's own difficulty cancels).

Reports the Spearman correlation between look-alike score and each ratio, and the mean ratio in
the lowest and highest look-alike quartiles (Mann-Whitney U between them).

Writes <output>/summary.md and per_image.json.

    python experiments/near_duplicate_check.py
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

TRAIN_SOURCE = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/source"
TEST_SOURCE = "experiments/outputs/z_validation_data/Icons-50_1024/source"
TEACHER_JSON = "experiments/outputs/rescore_current_metrics-Icons-50_1024/GBN_json"
LEARNED = {
    "Ours-GBN DDIM-25": "experiments/outputs/few_step_noise-Icons-50_1024/CN-GBN_1024/ddim-25-eta1_json",
    "Ours-GBN DDPM-500 (paper points)": "experiments/outputs/rescore_current_metrics-Icons-50_1024/CN-GBN-paper_json",
}
METRICS = [("M1_v2_hot_energy", "HOT"), ("M2_v1_capacity_delta_c", "cap dc"),
           ("M2_v2_centroid_displacement", "centroid"), ("M3_emd_distance", "EMD"),
           ("M1_v1_cvt_energy", "CVT")]
OUTPUT_DIR = "experiments/outputs/near_duplicate_check"
SIDE = 128


def concept(name):
    """apple_0_duck.png -> duck (vendor and version removed)."""
    return "_".join(Path(name).stem.split("_")[2:])


def load_vec(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    v = cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_json_dir(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUTPUT_DIR)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    test = sorted(p for p in Path(TEST_SOURCE).iterdir() if p.suffix.lower() == ".png")
    test_names = {p.name for p in test}
    train = [Path(r) / f for r, _, fs in os.walk(TRAIN_SOURCE) for f in fs
             if f.lower().endswith(".png") and f not in test_names]
    by_concept = {}
    for p in train:
        by_concept.setdefault(concept(p.name), []).append(p)

    cache = {}
    vec = lambda p: cache.setdefault(str(p), load_vec(p))
    sim = {}
    for p in test:
        cands = by_concept.get(concept(p.name), [])
        sim[p.stem] = max((float(vec(p) @ vec(q)) for q in cands), default=float("nan"))

    teacher = load_json_dir(TEACHER_JSON)
    rows, lines = {}, ["# Near-duplicate check (Icons-50 test set vs same-concept training icons)", "",
                       f"{len(test)} test icons, {len(train)} training icons. Look-alike score = max Pearson "
                       f"correlation ({SIDE}x{SIDE} grayscale) with a same-concept training icon.", ""]
    s_all = np.array([sim[k] for k in sim if np.isfinite(sim[k])])
    lines += [f"Look-alike score: median {np.median(s_all):.3f}, quartiles "
              f"{np.percentile(s_all, 25):.3f} / {np.percentile(s_all, 75):.3f}; "
              f"{int(np.isnan([sim[k] for k in sim]).sum())} icons have no same-concept training icon.", ""]
    for name, d in LEARNED.items():
        learned = load_json_dir(d)
        keys = sorted(k for k in learned if k in teacher and np.isfinite(sim.get(k, np.nan)))
        s = np.array([sim[k] for k in keys])
        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        lo, hi = s <= q1, s >= q3
        lines += [f"## {name} / GBN ({len(keys)} icons)", "",
                  "| metric | Spearman(look-alike, ratio) | p | mean ratio, lowest quartile | highest quartile | Mann-Whitney p |",
                  "|---|---|---|---|---|---|"]
        for m, lab in METRICS:
            ratio = np.array([learned[k][m] / teacher[k][m] for k in keys])
            rho, p = spearmanr(s, ratio)
            pu = mannwhitneyu(ratio[lo], ratio[hi]).pvalue
            lines.append(f"| {lab} | {rho:+.3f} | {p:.2g} | {ratio[lo].mean():.3f} | {ratio[hi].mean():.3f} | {pu:.2g} |")
            rows.setdefault(name, {})[lab] = {"spearman": float(rho), "p": float(p),
                                              "ratio_low_q": float(ratio[lo].mean()),
                                              "ratio_high_q": float(ratio[hi].mean()), "mw_p": float(pu)}
        lines.append("")
    lines += ["Reading: a positive Spearman on a lower-is-better ratio would mean the learned sampler is",
              "relatively *worse* on icons with close training look-alikes; memorisation would show as a",
              "clearly negative correlation and a lower ratio in the highest quartile.", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "per_image.json").write_text(json.dumps({"lookalike": sim, "stats": rows}, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
