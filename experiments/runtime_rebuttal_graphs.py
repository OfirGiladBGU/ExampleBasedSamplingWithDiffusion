#!/usr/bin/env python3
"""Runtime graphs for the two rebuttal versions of the paper (user, 2026-11-20).

    nonfull/runtime_scaling_ddim25_by_points   the paper's Fig. 7 curves as submitted (BNOT, WVS,
                                               GBN, Ours DDPM-500; mean over 10 images, August
                                               measurement) + Ours DDIM-25 (runtime_v10, blocks16
                                               OT, mean over the 10 new images). Supplement.
    full/runtime_scaling_by_points             the corrected Fig. 7: runtime_v10, median over the
                                               10 new images, blocks16 OT (= icons_results_runtimes
                                               since 2026-11-20). Main paper.
    full/runtime_scaling_ddim25_by_points      the same + Ours DDIM-25. Supplement.

Every graph is drawn with merge_runtime_results_stage_3.plot_runtimes (the Fig. 7 style).
Writes OUT_DIR only. Run from the project root:
    python experiments/runtime_rebuttal_graphs.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_runtime_results_stage_3 as fig7  # noqa: E402

O = "experiments/outputs"
PAPER_JSON = f"{O}/a_archive/bnot_runtimes_eigen_solver_20261118/icons_results_runtimes_common/runtimes_avg.json"
V10_MEAN_JSON = f"{O}/runtime_v10/runtimes_avg_blocks16.json"
V10_MEDIAN_JSON = f"{O}/runtime_v10/runtimes_avg_blocks16_median.json"
OUT_DIR = f"{O}/runtime_rebuttal_graphs"
BUDGETS = [256, 576, 1024, 1600, 2304, 3136, 4096, 5184, 6400]
BASE_METHODS = ["BNOT", "WVS", "GBN", "CN"]
DDIM_KEY = "CN-DDIM25"
DDIM_COLOR = "tab:purple"
LABEL_OURS = "Ours"
LABEL_OURS_DDPM = "Ours (DDPM-500)"
LABEL_OURS_DDIM = "Ours (DDIM-25)"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUT_DIR)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    return ap.parse_args()


def draw(runtimes, methods, ours_label, path, budgets):
    fig7.METHODS = methods
    fig7.METHOD_LABELS = {**fig7.METHOD_LABELS, "CN": ours_label, DDIM_KEY: LABEL_OURS_DDIM}
    fig7.METHOD_COLORS = {**fig7.METHOD_COLORS, DDIM_KEY: DDIM_COLOR}
    fig7.plot_runtimes(runtimes, budgets, path, "points", fig7.DEFAULT_Y_SCALE, fig7.SHOW_STD)


def show(name, runtimes, methods, budgets):
    print(f"== {name}")
    for m in methods:
        print(f"   {m:10s} " + " ".join(f"{runtimes[m][str(b)]['avg']:8.2f}" if str(b) in runtimes[m] else "       -"
                                         for b in budgets))


def main():
    a = parse_args()
    out = Path(a.output)
    paper = json.loads(Path(PAPER_JSON).read_text())
    mean = json.loads(Path(V10_MEAN_JSON).read_text())
    median = json.loads(Path(V10_MEDIAN_JSON).read_text())
    print("budgets   " + " ".join(f"{b:8d}" for b in a.budgets))

    nonfull = {m: paper[m] for m in BASE_METHODS}
    nonfull[DDIM_KEY] = mean[DDIM_KEY]
    show("non-full: paper curves (mean) + DDIM-25 (runtime_v10 blocks16, mean)", nonfull, BASE_METHODS + [DDIM_KEY], a.budgets)
    draw(nonfull, BASE_METHODS + [DDIM_KEY], LABEL_OURS_DDPM,
         out / "nonfull" / f"{fig7.PLOT_BASENAME}_ddim25_by_points.png", a.budgets)

    show("full: runtime_v10 blocks16, median", median, BASE_METHODS + [DDIM_KEY], a.budgets)
    draw(median, BASE_METHODS, LABEL_OURS, out / "full" / f"{fig7.PLOT_BASENAME}_by_points.png", a.budgets)
    draw(median, BASE_METHODS + [DDIM_KEY], LABEL_OURS_DDPM,
         out / "full" / f"{fig7.PLOT_BASENAME}_ddim25_by_points.png", a.budgets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
