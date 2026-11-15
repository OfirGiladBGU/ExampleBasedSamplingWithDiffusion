#!/usr/bin/env python3
"""Truncated (paper) vs exact Smart Init assignment at large point budgets: paired metric summary.

Rebuttal E18. experiments/ot_exact_quality.sbatch samples the same images with the same Smart Init
points and start noise twice, once with the paper's ot.emd (stopped at numItermax from ~3136 points)
and once with the exact assignment (ot_solver pot-exact). few_step_sampling_stage_2.py scores both.
This script pairs the two per (budget, sampler) over the images scored in both, and reports the
means, the relative change exact vs paper and a paired Wilcoxon test, plus the Smart Init times.

Reads   <output>/CN-GBN-pot_<N>/<sampler>_json/*.json and CN-GBN-pot-exact_<N>/...
Writes  <output>/summary.md, summary.json

Run from the project root:
    python experiments/ot_exact_quality_stage_3.py
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.few_step_sampling_stage_3 import METRICS, SHORT, read_jsonl  # noqa: E402

OUTPUT_DIR = "experiments/outputs/ot_exact_quality-Icons-50"
LABEL = "CN-GBN"
PAPER = "pot"
EXACT = "pot-exact"
P_THRESHOLD = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", nargs="?", default=OUTPUT_DIR)
    p.add_argument("--label", default=LABEL)
    p.add_argument("--paper", default=PAPER, help="reference solver (folder <label>-<paper>_<N>)")
    p.add_argument("--exact", default=EXACT, help="compared solver (folder <label>-<exact>_<N>)")
    p.add_argument("--name", default="summary", help="output file stem (<output>/<name>.md / .json)")
    return p.parse_args()


def load(run_dir, sampler):
    return {p.stem: json.loads(p.read_text()) for p in (run_dir / f"{sampler}_json").glob("*.json")}


def main():
    args = parse_args()
    root = Path(args.output)
    PAPER, EXACT = args.paper, args.exact
    budgets = sorted({int(d.name.rsplit("_", 1)[1]) for d in root.glob(f"{args.label}-{PAPER}_*")})
    out, lines = {}, [f"# Smart Init assignment: {PAPER} vs {EXACT}", "",
                      "Same images, Smart Init points and start noise; only the point-to-grid assignment differs.",
                      f"Cells: {PAPER} mean -> {EXACT} mean (change, * = paired Wilcoxon p < {P_THRESHOLD}).", ""]
    head = ["points", "sampler", "images"] + [SHORT[m] for m in METRICS] + [f"Smart Init s ({PAPER} / {EXACT})"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in budgets:
        paper_dir, exact_dir = root / f"{args.label}-{PAPER}_{n}", root / f"{args.label}-{EXACT}_{n}"
        if not exact_dir.exists():
            continue
        t_si = {}
        for name, d in ((PAPER, paper_dir), (EXACT, exact_dir)):
            rows = [r for path in d.glob("timings*.jsonl") for r in read_jsonl(path)]
            t_si[name] = st.median(r["time_smart_init"] for r in rows) if rows else float("nan")
        samplers = sorted(d.name[:-5] for d in paper_dir.iterdir() if d.is_dir() and d.name.endswith("_json"))
        for s in samplers:
            a, b = load(paper_dir, s), load(exact_dir, s)
            common = sorted(set(a) & set(b))
            if not common:
                continue
            entry = {"images": len(common), "time_smart_init_median": t_si, "metrics": {}}
            cells = [str(n), s, str(len(common))]
            for m in METRICS:
                va = np.array([a[k][m] for k in common])
                vb = np.array([b[k][m] for k in common])
                p = float(wilcoxon(vb, va).pvalue) if np.any(vb != va) else 1.0
                rel = float(vb.mean() / va.mean() - 1.0)
                entry["metrics"][m] = {"paper": float(va.mean()), "exact": float(vb.mean()), "rel_change": rel,
                                       "wilcoxon_p": p}
                cells.append(f"{va.mean():.4g} -> {vb.mean():.4g} ({rel * 100:+.1f}%{'*' if p < P_THRESHOLD else ''})")
            cells.append(f"{t_si[PAPER]:.2f} / {t_si[EXACT]:.2f}")
            out.setdefault(str(n), {})[s] = entry
            lines.append("| " + " | ".join(cells) + " |")
    (root / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    (root / f"{args.name}.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
