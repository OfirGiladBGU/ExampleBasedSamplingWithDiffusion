"""Few-step sampling stage 3: summarize metrics and timings against the DDPM-500 reference.

For every run folder (e.g. CN-GBN_1024) and sampler it reports, over the images scored in
all samplers of that run (paired):
    * mean / std of each M1-M5 metric, and the relative change vs the reference sampler
    * a paired Wilcoxon signed-rank p-value vs the reference sampler
    * median per-image denoising time (batch 1), mean Smart Init and OT times
    * median throughput (stipples / s) per batch size, when throughput.jsonl exists

Writes summary.json and summary.md next to the run folders.

Usage:
    python experiments/few_step_sampling_stage_3.py experiments/outputs/few_step_sampling-Icons-50_1024
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

OUTPUT_DIR = "experiments/outputs/few_step_sampling-Icons-50_1024"
REFERENCE = "ddpm-500"
METRICS = [
    "M1_v1_cvt_energy",
    "M1_v2_hot_energy",
    "M2_v1_capacity_delta_c",
    "M2_v2_centroid_displacement",
    "M2_v3_power_cell_cap_cv",
    "M3_emd_distance",
    "M4_sinkhorn_ot_cost",
    "M5_spatial_measure_rho_mean",
]
SHORT = {
    "M1_v1_cvt_energy": "CVT",
    "M1_v2_hot_energy": "HOT",
    "M2_v1_capacity_delta_c": "cap dc",
    "M2_v2_centroid_displacement": "centroid",
    "M2_v3_power_cell_cap_cv": "cell cv",
    "M3_emd_distance": "EMD",
    "M4_sinkhorn_ot_cost": "Sinkhorn",
    "M5_spatial_measure_rho_mean": "rho_s",
}


def parse_args():
    p = argparse.ArgumentParser(description="Summarize few-step sampling results")
    p.add_argument("output", nargs="?", default=OUTPUT_DIR)
    p.add_argument("--reference", default=REFERENCE)
    return p.parse_args()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sampler_order(name):
    kind, steps = name.split("-")[0], int(name.split("-")[1])
    return ({"ddpm": 0, "ddim": 1, "dpmpp2m": 2}.get(kind, 3), -steps, name)


def summarize_run(run_dir, reference):
    samplers = sorted((d.name[:-5] for d in run_dir.iterdir() if d.is_dir() and d.name.endswith("_json")),
                      key=sampler_order)
    per = {s: {p.stem: json.loads(p.read_text()) for p in (run_dir / f"{s}_json").glob("*.json")} for s in samplers}
    common = sorted(set.intersection(*(set(v) for v in per.values()))) if per else []

    timings = defaultdict(list)
    for path in sorted(run_dir.glob("timings*.jsonl")):
        for row in read_jsonl(path):
            timings[row["sampler"]].append(row)
    throughput = defaultdict(lambda: defaultdict(list))
    for path in sorted(run_dir.glob("throughput*.jsonl")):
        for row in read_jsonl(path):
            throughput[row["sampler"]][row["batch"]].append(row["stipples_per_s"])

    out = {"n_images": len(common), "samplers": {}}
    for s in samplers:
        entry = {"metrics": {}}
        for m in METRICS:
            vals = np.array([per[s][k][m] for k in common])
            e = {"mean": float(vals.mean()), "std": float(vals.std())} if len(vals) else {}
            if reference in per and s != reference and len(vals):
                ref = np.array([per[reference][k][m] for k in common])
                e["rel_change_vs_ref"] = float(vals.mean() / ref.mean() - 1.0)
                diff = vals - ref
                e["wilcoxon_p"] = float(wilcoxon(vals, ref).pvalue) if np.any(diff != 0) else 1.0
            entry["metrics"][m] = e
        if timings[s]:
            entry["time_denoise_median"] = st.median(r["time_denoise"] for r in timings[s])
            entry["time_smart_init_mean"] = st.mean(r["time_smart_init"] for r in timings[s])
            entry["time_ot_mean"] = st.mean(r["time_ot"] for r in timings[s])
        entry["throughput_median"] = {b: st.median(v) for b, v in sorted(throughput[s].items())}
        out["samplers"][s] = entry
    return out


def to_markdown(name, summary, reference):
    lines = [f"## {name}  ({summary['n_images']} paired images, reference {reference})", ""]
    samplers = summary["samplers"]
    batches = sorted({b for e in samplers.values() for b in e["throughput_median"]})
    head = ["sampler"] + [SHORT[m] for m in METRICS] + ["denoise s/img"] + [f"stipples/s @b{b}" for b in batches]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s, e in samplers.items():
        cells = [s]
        for m in METRICS:
            me = e["metrics"][m]
            if not me:
                cells.append("-")
                continue
            cell = f"{me['mean']:.4g}"
            if "rel_change_vs_ref" in me:
                star = "*" if me["wilcoxon_p"] < 0.05 else ""
                cell += f" ({me['rel_change_vs_ref'] * 100:+.1f}%{star})"
            cells.append(cell)
        cells.append(f"{e['time_denoise_median']:.3f}" if "time_denoise_median" in e else "-")
        cells += [f"{e['throughput_median'][b]:.1f}" if b in e["throughput_median"] else "-" for b in batches]
        lines.append("| " + " | ".join(cells) + " |")
    any_t = next((e for e in samplers.values() if "time_smart_init_mean" in e), None)
    if any_t:
        lines += ["", f"Sampler-independent CPU overhead per image: Smart Init {any_t['time_smart_init_mean']:.3f} s, "
                      f"OT conversion {any_t['time_ot_mean']:.4f} s."]
    lines += ["", "(+x%) = change of the mean vs the reference; * = paired Wilcoxon p < 0.05.", ""]
    return "\n".join(lines)


def main():
    args = parse_args()
    root = Path(args.output)
    summaries, md = {}, []
    for run_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if not any(d.name.endswith("_json") for d in run_dir.iterdir()):
            continue
        summaries[run_dir.name] = summarize_run(run_dir, args.reference)
        gpus = sorted({json.loads(p.read_text()).get("gpu", "?") for p in run_dir.glob("run_info*.json")})
        gpu = " / ".join(gpus) or "?"
        md.append(to_markdown(f"{run_dir.name} on {gpu}", summaries[run_dir.name], args.reference))
    (root / "summary.json").write_text(json.dumps(summaries, indent=2))
    (root / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
