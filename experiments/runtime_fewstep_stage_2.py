#!/usr/bin/env python3
"""Runtime vs point budget for the few-step samplers (rebuttal E1): summary table and plot.

Reads the timings written by experiments/runtime_fewstep.sbatch
(outputs/runtime_fewstep-Icons-50/CN-GBN_<N>/timings_runtime.jsonl and throughput_runtime.jsonl):
one GPU job timed the paper sampler (ddpm-500) and the few-step samplers on the same 10 pinned
Icons-50 runtime images at every budget. Adds the paper's stored Fig. 7 runtimes of the
optimization-based samplers (outputs/icons_results_runtimes/runtimes_avg.json), and the paper's own
measurement of our model ("CN") so the re-timed ddpm-500 can be checked against it.

Per image, "total" = Smart Init (CPU) + denoising (GPU) + OT-grid to points. Image loading and
writing the result are not included.

Writes outputs/runtime_fewstep-Icons-50/summary.md, summary.json and plots/runtime_fewstep.{png,pdf}.

Run from the project root:
    python experiments/runtime_fewstep_stage_2.py
"""
import argparse
import json
import statistics
from pathlib import Path

RUN_DIR = "experiments/outputs/runtime_fewstep-Icons-50"
LABEL = "CN-GBN"
TAG = "runtime"
STORED_RUNTIMES = "experiments/outputs/icons_results_runtimes/runtimes_avg.json"
BUDGETS = [256, 576, 1024, 1600, 2304, 3136, 4096, 5184, 6400]
SAMPLERS = ["ddpm-500", "ddim-25-eta1", "ddim-10-eta1"]
SAMPLER_LABELS = {"ddpm-500": "Ours, DDPM-500 (paper sampler, re-timed)",
                  "ddim-25-eta1": "Ours, DDIM-25 (eta=1)", "ddim-10-eta1": "Ours, DDIM-10 (eta=1)"}
STORED_METHODS = ["BNOT", "WVS", "GBN", "CN"]
STORED_LABELS = {"BNOT": "BNOT", "WVS": "WVS", "GBN": "GBN", "CN": "Ours, DDPM-500 (paper Fig. 7)"}
FIG_WIDTH = 11.5
FIG_HEIGHT = 4.5
DPI = 300
Y_SCALE = "log"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=RUN_DIR)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--stored_runtimes", default=STORED_RUNTIMES)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    ap.add_argument("--y_scale", choices=["log", "linear"], default=Y_SCALE)
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(args):
    out = {"timings": {}, "throughput": {}, "gpu": set()}
    for n in args.budgets:
        d = Path(args.run_dir) / f"{args.label}_{n}"
        rows = read_jsonl(d / f"timings_{args.tag}.jsonl")
        for s in SAMPLERS:
            r = [x for x in rows if x["sampler"] == s]
            if not r:
                continue
            out["gpu"].update(x.get("gpu", "?") for x in r)
            total = [x["time_smart_init"] + x["time_denoise"] + x["time_ot"] for x in r]
            out["timings"].setdefault(s, {})[n] = {
                "count": len(r),
                "total_median": statistics.median(total),
                "total_mean": statistics.fmean(total),
                "denoise_median": statistics.median(x["time_denoise"] for x in r),
                "smart_init_median": statistics.median(x["time_smart_init"] for x in r),
            }
        for x in read_jsonl(d / f"throughput_{args.tag}.jsonl"):
            out["throughput"].setdefault(x["sampler"], {}).setdefault(n, {}).setdefault(x["batch"], []).append(x["stipples_per_s"])
    for s, per_n in out["throughput"].items():
        for n, per_b in per_n.items():
            for b in per_b:
                per_b[b] = statistics.median(per_b[b])
    out["gpu"] = sorted(out["gpu"])
    return out


def write_summary(args, res, stored):
    run_dir = Path(args.run_dir)
    lines = [f"# Runtime vs point budget, few-step samplers (Ours-GBN, 10 pinned Icons-50 images)",
             "", f"GPU of the new timings: {', '.join(res['gpu']) or '?'}. Optimization-based samplers and",
             "\"paper Fig. 7\" are the stored runtimes the paper plots (other job / machine).",
             "Seconds per stipple, mean over the images (new samplers: median total = Smart Init + denoise + OT).", ""]
    head = "| method | " + " | ".join(str(n) for n in args.budgets) + " |"
    lines += [head, "|" + "---|" * (len(args.budgets) + 1)]
    for m in STORED_METHODS:
        if m in stored:
            vals = [stored[m].get(str(n), {}).get("avg") for n in args.budgets]
            lines.append(f"| {STORED_LABELS[m]} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in vals) + " |")
    for s in SAMPLERS:
        t = res["timings"].get(s, {})
        lines.append(f"| {SAMPLER_LABELS[s]} | " + " | ".join(f"{t[n]['total_median']:.2f}" if n in t else "-" for n in args.budgets) + " |")
    lines += ["", "Denoising only (GPU), median seconds:", "", head, "|" + "---|" * (len(args.budgets) + 1)]
    for s in SAMPLERS:
        t = res["timings"].get(s, {})
        lines.append(f"| {s} | " + " | ".join(f"{t[n]['denoise_median']:.3f}" if n in t else "-" for n in args.budgets) + " |")
    si = res["timings"].get(SAMPLERS[0], {})
    lines.append("| Smart Init (CPU) | " + " | ".join(f"{si[n]['smart_init_median']:.3f}" if n in si else "-" for n in args.budgets) + " |")
    if res["throughput"]:
        batches = sorted({b for per_n in res["throughput"].values() for per_b in per_n.values() for b in per_b})
        lines += ["", "Throughput, stipples per second (median, denoising only):", ""]
        for b in batches:
            lines += [f"Batch {b}:", "", head, "|" + "---|" * (len(args.budgets) + 1)]
            for s in SAMPLERS:
                tp = res["throughput"].get(s, {})
                lines.append(f"| {s} | " + " | ".join(f"{tp[n][b]:.2f}" if n in tp and b in tp[n] else "-" for n in args.budgets) + " |")
            lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"stored": stored, **res}, indent=2, default=str))
    print("\n".join(lines))


def plot(args, res, stored):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    for m in STORED_METHODS:
        if m not in stored:
            continue
        xs = [n for n in args.budgets if str(n) in stored[m]]
        ax.plot(xs, [stored[m][str(n)]["avg"] for n in xs], marker="o", label=STORED_LABELS[m],
                linestyle="--" if m == "CN" else "-")
    for s in SAMPLERS:
        t = res["timings"].get(s, {})
        xs = [n for n in args.budgets if n in t]
        ax.plot(xs, [t[n]["total_median"] for n in xs], marker="s", linewidth=2.2, label=SAMPLER_LABELS[s])
    ax.set_yscale(args.y_scale)
    ax.set_xticks(args.budgets)
    ax.set_xticklabels([str(n) for n in args.budgets])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = Path(args.run_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"runtime_fewstep.{ext}", dpi=DPI)
    print(f"plot -> {out / 'runtime_fewstep.png'}")


def main():
    args = parse_args()
    stored = json.loads(Path(args.stored_runtimes).read_text())
    res = summarize(args)
    write_summary(args, res, stored)
    if args.plot:
        plot(args, res, stored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
