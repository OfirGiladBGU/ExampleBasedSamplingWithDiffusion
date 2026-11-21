#!/usr/bin/env python3
"""Runtime graphs re-measured on a new image set (user, 2026-11-19): summary, JSON and plots.

Every curve comes from the same 10 images (the first 10, sorted, of the Icons-50 validation set):
    BNOT, WVS   runtime_v10_cpu.sbatch   (one CPU-only EPYC 7702P node; BNOT at its true defaults)
    GBN, ours   runtime_v10_gpu.sbatch   (one dedicated RTX 6000 Ada, 16 reserved cores, node shared)
Per-image totals as in runtimes_avg.json: BNOT total_wall_seconds, WVS / GBN the bare total of the
driver's timestamp file, ours Smart Init + denoising + OT-grid to points (median over the
repetitions, as runtime_clean_stage_3.py). Then mean / std over the images per budget; a budget with
fewer than 10 images is left out of the JSON and reported.

Writes into OUT_DIR (the stored runtimes_avg.json and figures are not touched):
    summary.md                          per method and budget: mean, median, max, count
    runtimes_avg_<ot>.json              BNOT / WVS / GBN / CN (= ours, ddpm-500) / CN-DDIM25
    plots_<ot>/fig7/...                 Fig. 7 (BNOT, WVS, GBN, Ours DDPM-500)
    plots_<ot>/new_variant/...          Fig. 7 + Ours DDIM-25
    runtimes_avg_<ot>_median.json, plots_<ot>_median/...   the same with the median over the images
for <ot> = exact (pot-exact at every budget) and blocks16 (bisection-blocks16 from 3136 points).

Run from the project root:
    python experiments/runtime_v10_stage_3.py
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_runtime_results_stage_3 as fig7  # noqa: E402

O = "experiments/outputs"
TEACHERS = {"BNOT": f"{O}/runtime_v10_bnot_defaults500", "WVS": f"{O}/runtime_v10_wvs", "GBN": f"{O}/runtime_v10_gbn"}
OURS = {"exact": [f"{O}/runtime_v10_ours-Icons-50"],
        "blocks16": [f"{O}/runtime_v10_ours_blocks16-Icons-50", f"{O}/runtime_v10_ours-Icons-50"]}
OUT_DIR = f"{O}/runtime_v10"
LABEL = "CN-GBN"
TAG = "runtime"
BUDGETS = [256, 576, 1024, 1600, 2304, 3136, 4096, 5184, 6400]
EXPECTED_IMAGES = 10
STATS = ["mean", "median"]   # mean: the main graphs (as runtimes_avg.json); median: a backup set (suffix _median)
OURS_SERIES = {"CN": ("ddpm-500", "Ours"), "CN-DDIM25": ("ddim-25-eta1", "Ours (DDIM-25)")}
DDIM_COLOR = "tab:purple"
RE_BNOT = re.compile(r"total_wall_seconds:\s*([0-9.eE+-]+)")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUT_DIR)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    return ap.parse_args()


def teacher_times(folder, method, n):
    out = {}
    for f in sorted((Path(folder) / f"timestamps_{method}_{n}").rglob("*.txt")):
        text = f.read_text(errors="replace")
        m = RE_BNOT.search(text)
        if m:
            out[f.stem] = float(m.group(1))
            continue
        for line in text.splitlines():
            if line.strip():
                try:
                    out[f.stem] = float(line.split()[0])
                except ValueError:
                    pass
                break
    return out


def ours_times(folders, n, sampler):
    """{image: median over repetitions}; the first folder that has the budget is used."""
    for folder in folders:
        rows = {}
        for rep in sorted(Path(folder).glob("rep*")):
            f = rep / f"{LABEL}_{n}" / f"timings_{TAG}.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["sampler"] == sampler:
                        rows.setdefault(r["image"], []).append(r["time_smart_init"] + r["time_denoise"] + r["time_ot"])
        if rows:
            return {k: statistics.median(v) for k, v in rows.items()}
    return {}


def entry(t, stat="mean"):
    """runtimes_avg.json entry; with stat="median" the "avg" field holds the median (the plot reads "avg")."""
    v = list(t.values())
    centre = statistics.fmean(v) if stat == "mean" else statistics.median(v)
    return {"avg": centre, "std": statistics.stdev(v) if len(v) > 1 else 0.0, "count": len(v), "stat": stat}


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Runtime on the new image set (first 10 Icons-50 validation images)", "",
             "Seconds per image. BNOT: true defaults (500 / 500, epsilon 1.0), fixed build. BNOT / WVS: one "
             "CPU-only EPYC 7702P node, 1 thread. GBN / ours: one job on one dedicated RTX 6000 Ada with 16 "
             "reserved cores, node shared with other users' jobs, our runs never overlapping (ours: median "
             "of 3 repetitions per image). Budgets with fewer than 10 images are left out of the JSON.", ""]
    teachers = {m: {n: teacher_times(d, m, n) for n in a.budgets} for m, d in TEACHERS.items()}
    lines += ["| method | " + " | ".join(map(str, a.budgets)) + " |", "|" + "---|" * (len(a.budgets) + 1)]

    def cells(per):
        c = []
        for n in a.budgets:
            t = per.get(n, {})
            if not t:
                c.append("-")
                continue
            v = list(t.values())
            c.append(f"{statistics.fmean(v):.2f} / {statistics.median(v):.2f} / {max(v):.1f}"
                     + ("" if len(v) == EXPECTED_IMAGES else f" ({len(v)})"))
        return c

    for m, per in teachers.items():
        lines.append(f"| {m} | " + " | ".join(cells(per)) + " |")
    ours = {ot: {k: {n: ours_times(dirs, n, s) for n in a.budgets} for k, (s, _) in OURS_SERIES.items()}
            for ot, dirs in OURS.items()}
    for ot in OURS:
        for k, (s, lab) in OURS_SERIES.items():
            lines.append(f"| {lab}, {s}, {ot} OT | " + " | ".join(cells(ours[ot][k])) + " |")
    lines += ["", "Cells: mean / median / max; the image count in brackets when it is not 10."]

    for ot in OURS:
        for stat in STATS:
            suffix = "" if stat == "mean" else f"_{stat}"
            runtimes = {m: {str(n): entry(t, stat) for n, t in per.items() if len(t) == EXPECTED_IMAGES}
                        for m, per in teachers.items()}
            for k in OURS_SERIES:
                runtimes[k] = {str(n): entry(t, stat) for n, t in ours[ot][k].items() if len(t) == EXPECTED_IMAGES}
            (out / f"runtimes_avg_{ot}{suffix}.json").write_text(json.dumps(runtimes, indent=2))
            base_methods = list(fig7.METHODS)
            for name, extra in (("fig7", []), ("new_variant", ["CN-DDIM25"])):
                fig7.METHODS = base_methods + extra
                fig7.METHOD_LABELS = {**fig7.METHOD_LABELS, "CN-DDIM25": OURS_SERIES["CN-DDIM25"][1]}
                fig7.METHOD_COLORS = {**fig7.METHOD_COLORS, "CN-DDIM25": DDIM_COLOR}
                png = out / f"plots_{ot}{suffix}" / name / f"{fig7.PLOT_BASENAME}_by_points.png"
                fig7.plot_runtimes(runtimes, a.budgets, png, "points", fig7.DEFAULT_Y_SCALE, fig7.SHOW_STD)
            fig7.METHODS = base_methods
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
