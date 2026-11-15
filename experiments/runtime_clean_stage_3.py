#!/usr/bin/env python3
"""Summary and Fig. 7 plot of the clean runtime run (experiments/runtime_clean.sbatch).

Reads <run_dir>/rep<k>/CN-GBN_<N>/timings_runtime.jsonl (ddim-25-eta1 and ddpm-500, exact OT).
Per image, total = Smart Init + denoising + OT-grid to points; the median over the repetitions is
taken per image, then mean / std over the 10 pinned images, as in runtimes_avg.json.

Writes
    <run_dir>/runtimes_avg_clean.json     stored paper runtimes + CN-DDIM25 + CN-DDPM-exact
    <run_dir>/summary.md                  per budget: totals, Smart Init / denoise split, rep spread
    <run_dir>/plots/runtime_scaling_by_<x_axis>.{png,pdf}   (merge_runtime_results_stage_3 style)

The baselines (BNOT, WVS, GBN) and the paper's "CN" curve are the stored paper measurements
(icons_results_runtimes/runtimes_avg.json), not re-timed on this node.

Run from the project root:
    python experiments/runtime_clean_stage_3.py
    python experiments/runtime_clean_stage_3.py --no-with-ddpm-exact
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_runtime_results_stage_3 as fig7  # noqa: E402

RUN_DIR = "experiments/outputs/runtime_clean-Icons-50"
LABEL = "CN-GBN"
TAG = "runtime"
STORED_RUNTIMES = "experiments/outputs/icons_results_runtimes/runtimes_avg.json"
SERIES = {  # key: (sampler, legend label, colour)
    "CN-DDIM25": ("ddim-25-eta1", "Ours (DDIM-25)", "tab:purple"),
    "CN-DDPM-exact": ("ddpm-500", "Ours (DDPM-500, re-timed)", "tab:brown"),
}
WITH_DDPM_EXACT = True
X_AXIS = fig7.DEFAULT_X_AXIS_MODE
Y_SCALE = fig7.DEFAULT_Y_SCALE


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=RUN_DIR)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--stored_runtimes", default=STORED_RUNTIMES)
    ap.add_argument("--budgets", type=int, nargs="+", default=fig7.POINT_BUDGETS)
    ap.add_argument("--with-ddpm-exact", action=argparse.BooleanOptionalAction, default=WITH_DDPM_EXACT,
                    help="Also draw the re-timed ddpm-500 (exact OT)")
    ap.add_argument("--x_axis", choices=["grid", "points"], default=X_AXIS)
    ap.add_argument("--y_scale", choices=["log", "linear"], default=Y_SCALE)
    return ap.parse_args()


def load_rows(args, n, sampler):
    """{image: [row per repetition]} for one budget and sampler."""
    rows = {}
    for rep in sorted(Path(args.run_dir).glob("rep*")):
        f = rep / f"{args.label}_{n}" / f"timings_{args.tag}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["sampler"] == sampler:
                    rows.setdefault(r["image"], []).append(r)
    return rows


def summarize(args, sampler):
    res, detail = {}, {}
    for n in args.budgets:
        rows = load_rows(args, n, sampler)
        if not rows:
            continue
        total, si, den, spread = [], [], [], []
        for reps in rows.values():
            t = [r["time_smart_init"] + r["time_denoise"] + r["time_ot"] for r in reps]
            total.append(statistics.median(t))
            si.append(statistics.median(r["time_smart_init"] for r in reps))
            den.append(statistics.median(r["time_denoise"] for r in reps))
            spread.append((max(t) - min(t)) / statistics.median(t))
        n_reps = min(len(r) for r in rows.values())
        res[str(n)] = {"avg": statistics.fmean(total), "std": statistics.pstdev(total), "count": len(total)}
        detail[n] = {"total": statistics.fmean(total), "smart_init": statistics.fmean(si),
                     "denoise": statistics.fmean(den), "max_rep_spread": max(spread), "reps": n_reps}
    return res, detail


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    runtimes = json.loads(Path(args.stored_runtimes).read_text())
    keys = ["CN-DDIM25"] + (["CN-DDPM-exact"] if args.with_ddpm_exact else [])
    lines = ["# Clean runtime run (exclusive node, exact OT)", "",
             f"Source: `{run_dir}/rep*/`. Per image: median over repetitions; then mean over images.",
             "Baselines and the paper's CN curve in the plot are the stored paper measurements.", ""]
    for key in keys:
        sampler, label, color = SERIES[key]
        runtimes[key], detail = summarize(args, sampler)
        fig7.METHODS = fig7.METHODS + [key]
        fig7.METHOD_LABELS = {**fig7.METHOD_LABELS, key: label}
        fig7.METHOD_COLORS = {**fig7.METHOD_COLORS, key: color}
        lines += [f"## {label} (`{sampler}`)", "",
                  "| points | total s | Smart Init s | denoise s | reps | max spread over reps |",
                  "|---|---|---|---|---|---|"]
        for n, d in detail.items():
            lines.append(f"| {n} | {d['total']:.3f} | {d['smart_init']:.3f} | {d['denoise']:.3f} | "
                         f"{d['reps']} | {100 * d['max_rep_spread']:.1f}% |")
        lines.append("")
    (run_dir / "runtimes_avg_clean.json").write_text(json.dumps(runtimes, indent=2))
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    out = run_dir / fig7.PLOTS_DIRNAME / f"{fig7.PLOT_BASENAME}_by_{args.x_axis}.png"
    fig7.plot_runtimes(runtimes, args.budgets, out, args.x_axis, args.y_scale, fig7.SHOW_STD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
