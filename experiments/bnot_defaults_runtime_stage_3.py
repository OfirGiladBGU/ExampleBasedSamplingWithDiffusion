#!/usr/bin/env python3
"""Fig. 7 and the new-variant runtime graph with BNOT at its true defaults (500 / 500, epsilon 1.0).

Reads the BNOT timings of experiments/bnot_retime_defaults.sbatch (<new>/timestamps_BNOT_<N>/**.txt,
"total_wall_seconds", as merge_runtime_results_stage_2.py reads them) and compares them per budget
with the current fixed-build curve (25 / 50 caps, <current>/timestamps_BNOT_<N>). Then, without
touching the stored files, writes into OUT_DIR:
    summary.md                      per budget: mean / median / max s of both settings, image counts
    runtimes_avg.json               icons_results_runtimes/runtimes_avg.json with BNOT replaced
    plots/runtime_scaling_by_points.{png,pdf}       Fig. 7 (merge_runtime_results_stage_3)
    new_variant/plots/...           the new-variant graph (runtime_clean_stage_3 on the clean run of
                                    our sampler, CLEAN_RUN, whose rep folders are symlinked here)

Run from the project root:
    python experiments/bnot_defaults_runtime_stage_3.py
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

O = "experiments/outputs"
NEW_DIR = f"{O}/icons_results_runtimes_bnot_defaults500"
CURRENT_DIR = f"{O}/icons_results_runtimes_bnot"
STORED_RUNTIMES = f"{O}/icons_results_runtimes/runtimes_avg.json"
CLEAN_RUN = f"{O}/runtime_clean_blocks16-Icons-50"
OUT_DIR = f"{O}/runtime_graphs_bnot_defaults500"
BUDGETS = [256, 576, 1024, 1600, 2304, 3136, 4096, 5184, 6400, 7744, 9216, 10816, 12544]
EXPECTED_IMAGES = 10
RE_BNOT = re.compile(r"total_wall_seconds:\s*([0-9.eE+-]+)")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default=NEW_DIR)
    ap.add_argument("--current", default=CURRENT_DIR)
    ap.add_argument("--stored_runtimes", default=STORED_RUNTIMES)
    ap.add_argument("--clean_run", default=CLEAN_RUN)
    ap.add_argument("--output", default=OUT_DIR)
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS)
    return ap.parse_args()


def times(folder, n):
    """{image stem: total_wall_seconds} of one budget."""
    out = {}
    for f in sorted((Path(folder) / f"timestamps_BNOT_{n}").rglob("*.txt")):
        m = RE_BNOT.search(f.read_text())
        if m:
            out[f.stem] = float(m.group(1))
    return out


def stats(v):
    return statistics.fmean(v), statistics.median(v), max(v)


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    runtimes = json.loads(Path(a.stored_runtimes).read_text())
    bnot = {}
    lines = ["# BNOT runtime at its true defaults (500 / 500, epsilon 1.0) vs the paper driver's caps (25 / 50)", "",
             f"Fixed build, 10 pinned Icons-50 runtime images, 1 thread, EPYC 7702P. New: `{a.new}`; "
             f"current curve: `{a.current}`. Seconds per image (total_wall_seconds).", "",
             "| points | 500/500 mean | median | max | images | 25/50 mean | median | max | images | mean ratio |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n in a.budgets:
        new, cur = times(a.new, n), times(a.current, n)
        if not new:
            continue
        if len(new) == EXPECTED_IMAGES:
            bnot[str(n)] = {"avg": statistics.fmean(new.values()), "std": statistics.pstdev(new.values()),
                            "count": len(new)}
        nm, nmed, nmax = stats(new.values())
        row = f"| {n} | {nm:.2f} | {nmed:.2f} | {nmax:.2f} | {len(new)} |"
        if cur:
            cm, cmed, cmax = stats(cur.values())
            row += f" {cm:.2f} | {cmed:.2f} | {cmax:.2f} | {len(cur)} | {nm / cm:.2f} |"
        else:
            row += " - | - | - | 0 | - |"
        lines.append(row)
        slow = sorted(new.items(), key=lambda kv: -kv[1])[:2]
        if cur and slow[0][1] > 3 * nmed:
            lines.append(f"|  | slowest: " + ", ".join(f"{s} {t:.0f} s (25/50: {cur.get(s, float('nan')):.0f} s)"
                                                       for s, t in slow) + " | | | | | | | | |")
    incomplete = [n for n in a.budgets if times(a.new, n) and str(n) not in bnot]
    lines += ["", "Budgets with fewer than 10 timings are left out of the JSON (and the graphs): "
              + (", ".join(map(str, incomplete)) if incomplete else "none") + "."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    runtimes["BNOT"] = bnot
    (out / "runtimes_avg.json").write_text(json.dumps(runtimes, indent=2))
    py = sys.executable
    subprocess.run([py, "experiments/merge_runtime_results_stage_3.py", "--runtimes_json", str(out / "runtimes_avg.json")],
                   check=True)
    nv = out / "new_variant"
    nv.mkdir(exist_ok=True)
    for rep in sorted(Path(a.clean_run).glob("rep*")):
        link = nv / rep.name
        if not link.exists():
            link.symlink_to(rep.resolve())
    subprocess.run([py, "experiments/runtime_clean_stage_3.py", "--run_dir", str(nv),
                    "--stored_runtimes", str(out / "runtimes_avg.json")], check=True, stdout=subprocess.DEVNULL)
    print(f"graphs -> {out / 'plots'} and {nv / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
