#!/usr/bin/env python3
"""E13 as a LaTeX table for the supplement: BNOT, WVS, GBN and our three variants sampled with the
25-step stochastic DDIM sampler (ddim-25-eta1), mean +- std over each test set, in the style of
the supplement's full table (tables_rebuttal/quantitative_comparison_full_table.tex).

All six columns are scored with the same (current) metric code: the samplers from
rescore_current_metrics-<dataset>/<S>_json, ours from e13_ddim25-<dataset>/CN-<S>_<N>/ddim-25-eta1_json.
Best and second-best displayed (rounded) means per row in bold and underlined; equal displayed
values share the mark, as in the paper's tables.

Writes OUTPUT (nothing else). Run from the project root:
    python experiments/e13_ddim25_tex_table.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

O = "experiments/outputs"
OUTPUT = f"{O}/e13_ddim25_tex/quantitative_comparison_ddim25_table.tex"
DATASETS = [("Icons-50_1024", "Icons-50"), ("CelebA-5K_1024", "CelebA"),
            ("ShapeNet2D-4K_1600", "ShapeNet2D"), ("Anime-Chibi_1600", "Anime Chibi")]
SAMPLERS = ["BNOT", "WVS", "GBN"]
METRICS = [("M1_v1_cvt_energy", r"CVT ($\times 10^{-5}$)\,$\downarrow$", 1e5, True),
           ("M1_v2_hot_energy", r"HOT ($\times 10^{-5}$)\,$\downarrow$", 1e5, True),
           ("M2_v1_capacity_delta_c", r"Capacity $\delta_c$\,$\downarrow$", 1, True),
           ("M2_v2_centroid_displacement", r"Centroid $D$\,$\downarrow$", 1, True),
           ("M3_emd_distance", r"EMD\,$\downarrow$", 1, True),
           ("M4_sinkhorn_ot_cost", r"Sinkhorn\,$\downarrow$", 1, True),
           ("M5_spatial_measure_rho_mean", r"Spatial $\rho_s$\,$\uparrow$", 1, False)]
DECIMALS = 3
STD_DDOF = 0          # numpy default; checked against the paper's full table (CelebA rows, same code)
TABLE_LABEL = "tab:quantitative_comparison_ddim25"
CAPTION = (r"Quantitative comparison of BNOT, WVS, GBN, and our three variants sampled with the "
           r"25-step stochastic DDIM sampler (Section~\ref{sec:fast_sampling}) instead of the 500-step "
           r"DDPM sampler of the main paper, with the same checkpoints, test sets and point budgets as "
           r"Table~\ref{tab:quantitative_comparison_full}. Mean $\pm$ standard deviation over the respective "
           r"test sets; best and second-best means per row are shown in bold and underlined, respectively. "
           r"All six columns are scored with the same metric implementation, so the BNOT, WVS and GBN "
           r"values can differ from Table~\ref{tab:quantitative_comparison_full} in the last digit.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--decimals", type=int, default=DECIMALS)
    ap.add_argument("--std-ddof", type=int, default=STD_DDOF)
    return ap.parse_args()


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in Path(d).glob("*.json")}


def main():
    a = parse_args()
    cols = SAMPLERS + [f"Ours-{s}" for s in SAMPLERS]
    out = [r"\begin{table*}[b]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\renewcommand{\arraystretch}{1.2}", r"\begin{tabular}{|l|c|c|c|c|c|c|}", r"\hline",
           r"\textbf{Metric} & BNOT & WVS & GBN & \textbf{Ours-BNOT} & \textbf{Ours-WVS} & \textbf{Ours-GBN} \\",
           r" & & & & \multicolumn{3}{c|}{\textbf{25-step DDIM sampler}} \\", r"\hline"]
    for ds, name in DATASETS:
        n = int(ds.rsplit("_", 1)[1])
        data = {s: load(Path(O) / f"rescore_current_metrics-{ds}/{s}_json") for s in SAMPLERS}
        data.update({f"Ours-{s}": load(Path(O) / f"e13_ddim25-{ds}/CN-{s}_{n}/ddim-25-eta1_json") for s in SAMPLERS})
        common = sorted(set.intersection(*(set(v) for v in data.values())))
        print(f"{ds}: {len(common)} images (per column: {[len(data[c]) for c in cols]})")
        out += [rf"\multicolumn{{7}}{{|l|}}{{\textbf{{{name} ({len(common)} images, {n} points)}}}} \\", r"\hline"]
        for key, lab, scale, lower in METRICS:
            v = {c: scale * np.array([data[c][k][key] for k in common]) for c in cols}
            mean = {c: round(float(v[c].mean()), a.decimals) for c in cols}
            std = {c: float(v[c].std(ddof=a.std_ddof)) for c in cols}
            if ds.startswith("CelebA") and key == "M1_v1_cvt_energy":
                print(f"  std check, CelebA BNOT CVT: ddof0 {v['BNOT'].std():.3f}, ddof1 {v['BNOT'].std(ddof=1):.3f} "
                      f"(paper full table: 5.927 +- 1.678)")
            ranked = sorted(set(mean.values()), reverse=not lower)
            cells = []
            for c in cols:
                s = f"{mean[c]:.{a.decimals}f} $\\pm$ {std[c]:.{a.decimals}f}"
                if mean[c] == ranked[0]:
                    s = rf"\textbf{{{s}}}"
                elif len(ranked) > 1 and mean[c] == ranked[1]:
                    s = rf"\underline{{{s}}}"
                cells.append(s)
            out.append(f"{lab} & " + " & ".join(cells) + r" \\")
        out.append(r"\hline")
    out += [r"\end{tabular}", rf"\caption{{{CAPTION}}}", rf"\label{{{TABLE_LABEL}}}", r"\end{table*}"]
    p = Path(a.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out) + "\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
