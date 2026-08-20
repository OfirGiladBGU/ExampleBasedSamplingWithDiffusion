"""ablation_runtime_results_stage_2.py -- Part 2: average each method separately.

Scans OUTPUT_DIR for method subfolders that contain a timestamps/ folder (written by
stage 1), parses every per-image timing .txt, and writes the per-method averages to:

    OUTPUT_DIR/<method>/runtime_avg.json   = {timing_label: {mean, std, count}, ...}

Each timing .txt has lines like "Denoising Time: 1.2345 s"; the per-step breakdown
below "--- Denoising Step Timings ---" is ignored (it is not a "Label: <n> s" line).
"""

import argparse
import json
import re
from pathlib import Path

# Matches a whole line "Some Label: 1.2345 s" -> (label, seconds). Step lines like
# "step 0 (t=..): total=..s, p_sample=..s" have text after the colon and don't match.
TIME_LINE = re.compile(r"^\s*(.+?):\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$")

OUTPUT_DIR = "experiments/outputs/ablation_runtime_results"


def parse_timing_file(path):
    """label -> seconds for the summary timing lines of one .txt."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = TIME_LINE.match(line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5


def average_method(method_dir):
    ts_dir = Path(method_dir) / "timestamps"
    txts = sorted(ts_dir.glob("*.txt"))
    txts = [t for t in txts if not t.name.endswith("_profiler_summary.txt")]
    if not txts:
        return None
    per_label = {}
    for t in txts:
        for label, secs in parse_timing_file(t).items():
            per_label.setdefault(label, []).append(secs)
    result = {}
    for label, vals in per_label.items():
        mean, std = mean_std(vals)
        result[label] = {"mean": mean, "std": std, "count": len(vals)}
    return result


def parse_args():
    ap = argparse.ArgumentParser(description="Average per-image runtimes for each method subfolder.")
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--result-dirs", default="",
                    help="Comma-separated method subfolders; empty = auto-discover any with timestamps/.")
    ap.add_argument("--out-name", default="runtime_avg.json")
    return ap.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output)
    if args.result_dirs.strip():
        methods = [s.strip() for s in args.result_dirs.split(",") if s.strip()]
    else:
        methods = sorted(d.name for d in out_base.iterdir()
                         if d.is_dir() and (d / "timestamps").is_dir())
    if not methods:
        print(f"No method subfolders with a timestamps/ folder under {out_base}"); return 2

    for m in methods:
        res = average_method(out_base / m)
        if res is None:
            print(f"  {m}: no timing .txt files, skipped"); continue
        (out_base / m / args.out_name).write_text(json.dumps(res, indent=2))
        n = next(iter(res.values()))["count"] if res else 0
        tot = res.get("Total Inference Time", {}).get("mean")
        extra = f"  (mean Total Inference Time = {tot:.4f} s)" if tot is not None else ""
        print(f"  {m}: averaged {n} runs -> {out_base / m / args.out_name}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
