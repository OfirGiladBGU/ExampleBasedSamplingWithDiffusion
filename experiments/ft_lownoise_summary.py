#!/usr/bin/env python3
"""Low-noise fine-tune: paired comparison with the paper checkpoint on the 200 held-out images.

Reads, under experiments/outputs/diag_generalization (all scored with few_step_sampling_stage_2.py):
    CN-GBN-test_1024                    baseline (paper checkpoint, epoch 5000), normal sampling
    CN-GBN-test-fromGT_1024             baseline, started from the teacher's exemplar at step K
    CN-GBN-test-ft-<tag>_1024           fine-tuned variants (ft_lownoise_train.sbatch)
    CN-GBN-test-ft-<tag>-fromGT_1024
    timestep_loss.json, timestep_loss_ft-<tag>.json

Same images, Smart Init and noise seeds, so each comparison is paired per image. Because the
p-value comes from a paired rank test, the direction is reported from the paired columns (median
per-image change, share of images where the variant is worse), next to the ratio of each mean to
the teacher's exemplar.

Writes <output>/ft_lownoise_summary.md.

    python experiments/ft_lownoise_summary.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUT = Path("experiments/outputs/diag_generalization")
N = 1024
TAGS = [("g5", "gamma 5 (control: more training only)"), ("g50", "gamma 50"), ("g0", "no Min-SNR")]
METRICS = [("M1_v1_cvt_energy", "CVT", True), ("M2_v1_capacity_delta_c", "cap dc", True),
           ("M5_spatial_measure_rho_mean", "rho_s", False), ("M1_v2_hot_energy", "HOT", True),
           ("M2_v2_centroid_displacement", "centroid", True), ("M3_emd_distance", "W2", True)]
FROM_GT = [10, 20, 50, 100, 500]


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def paired_rows(base, var, ex):
    keys = sorted(set(base) & set(var) & set(ex))
    rows = []
    for m, lab, lower in METRICS:
        b = np.array([base[k][m] for k in keys]); v = np.array([var[k][m] for k in keys])
        e = np.array([ex[k][m] for k in keys])
        p = wilcoxon(v, b).pvalue if np.any(v != b) else 1.0
        worse = float(((v > b) if lower else (v < b)).mean())
        verdict = "no difference" if p >= 0.05 else ("worse" if worse > 0.5 else "better")
        rows.append((lab, b.mean() / e.mean(), v.mean() / e.mean(), np.median(v / b) - 1, worse, p, verdict))
    return len(keys), rows


def main():
    ex = load(OUT / f"CN-GBN-test_{N}" / "GBN-exemplar_json")
    lines = ["# Low-noise fine-tune vs the paper checkpoint (Ours-GBN, 200 held-out Icons-50 images)", "",
             "Paired per image (same Smart Init and noise). 'to teacher' = mean / the teacher exemplar's mean",
             "(1.00 = the teacher; rho_s is higher-is-better). Direction: read the paired columns.", ""]
    for tag, desc in TAGS:
        run = OUT / f"CN-GBN-test-ft-{tag}_{N}"
        if not run.is_dir():
            continue
        lines += [f"## ft-{tag}: {desc}", ""]
        for sampler in ("ddpm-500", "ddim-25-eta1"):
            if not (run / f"{sampler}_json").is_dir():
                continue
            n, rows = paired_rows(load(OUT / f"CN-GBN-test_{N}" / f"{sampler}_json"), load(run / f"{sampler}_json"), ex)
            lines += [f"### normal sampling, {sampler} ({n} images)", "",
                      "| metric | to teacher: paper ckpt | to teacher: fine-tuned | median per-image change | "
                      "fine-tuned worse on | p | paired verdict |", "|---|---|---|---|---|---|---|"]
            for lab, rb, rv, med, worse, p, verdict in rows:
                lines.append(f"| {lab} | {rb:.3f} | {rv:.3f} | {100 * med:+.1f}% | {100 * worse:.0f}% | {p:.2g} | {verdict} |")
            lines.append("")
        gt = OUT / f"CN-GBN-test-ft-{tag}-fromGT_{N}"
        if gt.is_dir():
            lines += ["### start from the teacher's exemplar noised to step K, then DDPM from K", "",
                      "| K | cap dc to teacher: paper -> fine-tuned | median change | p | CVT to teacher: paper -> fine-tuned | "
                      "median change | p | rho_s to teacher: paper -> fine-tuned | median change | p |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for k in FROM_GT:
                bdir, vdir = OUT / f"CN-GBN-test-fromGT_{N}" / f"ddpm-{k}_json", gt / f"ddpm-{k}_json"
                if not (bdir.is_dir() and vdir.is_dir()):
                    continue
                _, rows = paired_rows(load(bdir), load(vdir), ex)
                r = {lab: (rb, rv, med, p) for lab, rb, rv, med, _, p, _ in rows}
                cells = [f"{r[l][0]:.3f} -> {r[l][1]:.3f} | {100 * r[l][2]:+.1f}% | {r[l][3]:.2g}"
                         for l in ("cap dc", "CVT", "rho_s")]
                lines.append(f"| {k} | " + " | ".join(cells) + " |")
            lines.append("")

    base_loss = OUT / "timestep_loss.json"
    if base_loss.exists():
        tabs = {"paper": json.loads(base_loss.read_text())}
        for tag, _ in TAGS:
            f = OUT / f"timestep_loss_ft-{tag}.json"
            if f.exists():
                tabs[f"ft-{tag}"] = json.loads(f.read_text())
        names = list(tabs)
        lines += ["## Denoising error per noise level: x0 RMS error in cells, held-out (training) images", "",
                  "| t | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for i, row in enumerate(tabs["paper"]):
            lines.append(f"| {row['t']} | " + " | ".join(
                f"{tabs[n][i]['x0_rms_test']:.3f} ({tabs[n][i]['x0_rms_train']:.3f})" for n in names) + " |")
        lines.append("")
    (OUT / "ft_lownoise_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
