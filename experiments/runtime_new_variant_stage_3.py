#!/usr/bin/env python3
"""Fig. 7 runtime plot with the new variant added (supplement version).

Reads the paper's stored runtimes (icons_results_runtimes/runtimes_avg.json: BNOT, WVS, GBN and our
DDPM-500 model "CN") and the new-variant timings written by experiments/runtime_new_variant.sbatch
(runtime_new_variant-Icons-50/CN-GBN_<N>/timings_runtime.jsonl, ddim-25-eta1 + exact OT, and
optionally ddpm-500 + exact OT). Per image, total = Smart Init + denoising + OT-grid to points;
averaged over the 10 pinned runtime images like runtimes_avg.json. The figure is drawn by
merge_runtime_results_stage_3.plot_runtimes, so it matches the paper figure's style.

Writes <run_dir>/runtimes_avg_new_variant.json and <run_dir>/plots/runtime_scaling_by_<x_axis>.{png,pdf}.

Run from the project root:
    python experiments/runtime_new_variant_stage_3.py
    python experiments/runtime_new_variant_stage_3.py --with-ddpm-exact
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_runtime_results_stage_3 as fig7  # noqa: E402

RUN_DIR = "experiments/outputs/runtime_new_variant-Icons-50"
LABEL = "CN-GBN"
TAG = "runtime"
STORED_RUNTIMES = "experiments/outputs/icons_results_runtimes/runtimes_avg.json"
NEW_KEY = "CN-DDIM25"
NEW_SAMPLER = "ddim-25-eta1"
NEW_LABEL = "Ours (DDIM-25)"
NEW_COLOR = "tab:purple"
DDPM_KEY = "CN-DDPM-exact"
DDPM_SAMPLER = "ddpm-500"
DDPM_LABEL = "Ours (DDPM-500, exact OT, re-timed)"
DDPM_COLOR = "tab:brown"
WITH_DDPM_EXACT = False
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
                    help="Also draw the re-timed ddpm-500 with the exact OT")
    ap.add_argument("--x_axis", choices=["grid", "points"], default=X_AXIS)
    ap.add_argument("--y_scale", choices=["log", "linear"], default=Y_SCALE)
    return ap.parse_args()


def timed_runtimes(args, sampler):
    """{budget: {avg, std, count}} of the total per-image time, as in runtimes_avg.json."""
    res = {}
    for n in args.budgets:
        f = Path(args.run_dir) / f"{args.label}_{n}" / f"timings_{args.tag}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        total = [r["time_smart_init"] + r["time_denoise"] + r["time_ot"] for r in rows if r["sampler"] == sampler]
        if total:
            res[str(n)] = {"avg": statistics.fmean(total), "std": statistics.pstdev(total), "count": len(total)}
    return res


def main():
    args = parse_args()
    runtimes = json.loads(Path(args.stored_runtimes).read_text())
    runtimes[NEW_KEY] = timed_runtimes(args, NEW_SAMPLER)
    fig7.METHODS = fig7.METHODS + [NEW_KEY]
    fig7.METHOD_LABELS = {**fig7.METHOD_LABELS, NEW_KEY: NEW_LABEL}
    fig7.METHOD_COLORS = {**fig7.METHOD_COLORS, NEW_KEY: NEW_COLOR}
    if args.with_ddpm_exact:
        runtimes[DDPM_KEY] = timed_runtimes(args, DDPM_SAMPLER)
        fig7.METHODS = fig7.METHODS + [DDPM_KEY]
        fig7.METHOD_LABELS[DDPM_KEY] = DDPM_LABEL
        fig7.METHOD_COLORS[DDPM_KEY] = DDPM_COLOR
    run_dir = Path(args.run_dir)
    (run_dir / "runtimes_avg_new_variant.json").write_text(json.dumps(runtimes, indent=2))
    for key in [k for k in (NEW_KEY, DDPM_KEY) if k in runtimes]:
        print(key, {n: round(v["avg"], 2) for n, v in runtimes[key].items()})
    out = run_dir / fig7.PLOTS_DIRNAME / f"{fig7.PLOT_BASENAME}_by_{args.x_axis}.png"
    fig7.plot_runtimes(runtimes, args.budgets, out, args.x_axis, args.y_scale, fig7.SHOW_STD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
