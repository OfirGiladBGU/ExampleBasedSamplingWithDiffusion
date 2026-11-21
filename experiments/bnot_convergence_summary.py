#!/usr/bin/env python3
"""BNOT convergence check: does the iteration cap or the tolerance decide the result?

Reads experiments/outputs/bnot_convergence_check (bnot_convergence_check.sbatch): the 10 pinned
Icons-50 runtime images, fixed BNOT build, one run per configuration
    t_<tag>_<points>/**/<stem>.{npy,txt}   point sets and the solver's stats (iterations, capacity error)
    timestamps_<tag>_<points>/**/<stem>.txt  timings
    score_1024/<tag>_json/<stem>.json      paper metrics (few_step_sampling_stage_2.py), 1024 points
and writes summary.md: per configuration the outer iterations (and how many runs stopped at the
cap), time, capacity error, the distance of the points from the paper setting, and the metrics
with the paired change against the paper setting (epsilon 1.0, max_iters 25).

    python experiments/bnot_convergence_summary.py
"""
import json
import re
import statistics
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

OUT = Path("experiments/outputs/bnot_convergence_check")
BASE = "eps1_mi25"
CONFIGS = [  # tag, epsilon, max_iters, max_newton_iters, note
    ("eps1_mi25", 1.0, 25, 50, "the paper setting"),
    ("eps1_mi500", 1.0, 500, 500, "original BNOT defaults"),
    ("eps0p1_mi500", 0.1, 500, 500, "10x tighter tolerance"),
    ("eps0p01_mi500", 0.01, 500, 500, "100x tighter tolerance"),
    ("eps0p01_mi25", 0.01, 25, 50, "100x tighter, capped at 25"),
]
METRICS = [  # key, label, lower is better
    ("M1_v1_cvt_energy", "CVT", True), ("M1_v2_hot_energy", "HOT", True),
    ("M2_v1_capacity_delta_c", "cap dc", True), ("M2_v2_centroid_displacement", "centroid", True),
    ("M3_emd_distance", "W2", True), ("M5_spatial_measure_rho_mean", "rho_s", False),
]


def field(path, name):
    m = re.search(rf"^{name}:\s*([-0-9.eE+]+)", path.read_text(), re.M)
    return float(m.group(1)) if m else float("nan")


def load(tag, n):
    """stem -> dict(points, iterations, cap_err, seconds) for one configuration and budget."""
    res = {}
    for npy in sorted((OUT / "data" / f"t_{tag}_{n}").rglob("*.npy")):
        stats = npy.with_suffix(".txt")
        rel = npy.relative_to(OUT / "data" / f"t_{tag}_{n}").with_suffix(".txt")
        ts = OUT / "data" / f"timestamps_{tag}_{n}" / rel
        res[npy.stem] = {
            "points": np.load(npy),
            "iterations": field(stats, "iterations") if stats.exists() else float("nan"),
            "cap_err": field(stats, "rms_rel_capacity_error") if stats.exists() else float("nan"),
            "seconds": field(ts, "total_wall_seconds") if ts.exists() else float("nan"),
        }
    return res


def fmt_iters(vals):
    return " ".join(str(int(v)) if v == v else "?" for v in vals)


def main():
    lines = ["# BNOT convergence check (fixed build, 10 pinned Icons-50 runtime images, seed 7)", "",
             "Outer iterations = coarse (Lloyd) + fine (gradient ascent) position updates; the solver stops",
             "when the position-gradient norm falls below a threshold proportional to epsilon, or at",
             "max_iters. Capacity error = rms relative capacity error reported by the solver.", ""]
    for n in (1024, 6400):
        runs = {tag: load(tag, n) for tag, *_ in CONFIGS}
        if not any(runs.values()):
            continue
        base = runs.get(BASE, {})
        notes = []
        lines += [f"## {n} points", "",
                  "| configuration | epsilon | max_iters | images | outer iterations per image | stopped at the cap | "
                  "time s: mean / median / max | capacity error: median / max | "
                  "distance to the paper setting's nearest point: median / max (point spacings) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for tag, eps, mi, _, note in CONFIGS:
            r = runs[tag]
            if not r:
                continue
            stems = sorted(r)
            its = [r[s]["iterations"] for s in stems]
            secs = [r[s]["seconds"] for s in stems if r[s]["seconds"] == r[s]["seconds"]]
            caps = [r[s]["cap_err"] for s in stems if r[s]["cap_err"] == r[s]["cap_err"]]
            moved = []
            for s in stems:
                if s in base:  # nearest-neighbour distance, so a different site order does not matter
                    d = cKDTree(base[s]["points"]).query(r[s]["points"])[0] * np.sqrt(n)
                    moved.append((float(np.median(d)), float(d.max())))
            mv = (f"{statistics.median(m[0] for m in moved):.2g} / {max(m[1] for m in moved):.2g}"
                  if moved else "n/a")
            tm = (f"{statistics.fmean(secs):.1f} / {statistics.median(secs):.1f} / {max(secs):.1f}"
                  if secs else "n/a")
            cp = f"{statistics.median(caps):.1e} / {max(caps):.1e}" if caps else "n/a"
            missing = sorted(set(base) - set(r))
            if missing:
                notes.append(f"- {tag}, {n} points: no result within the time limit for "
                             + ", ".join(f"`{m}`" for m in missing))
            lines.append(f"| {tag} ({note}) | {eps} | {mi} | {len(stems)} | {fmt_iters(its)} | "
                         f"{sum(1 for v in its if v >= mi)} | {tm} | {cp} | {mv} |")
        lines += [""] + notes + ([""] if notes else [])

    score = OUT / "score_1024"
    js = {tag: {p.stem: json.loads(p.read_text()) for p in (score / f"{tag}_json").glob("*.json")}
          for tag, *_ in CONFIGS}
    if js.get(BASE):
        lines += ["## Paper metrics at 1024 points (same scorer as the tables)", "",
                  "Mean over the images; in brackets the median per-image change against the paper setting,",
                  "the number of images where the configuration is better, and the paired Wilcoxon p-value",
                  "(10 images: only large, consistent changes can reach p < 0.05).", "",
                  "| configuration | " + " | ".join(l for _, l, _ in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for tag, *_ in CONFIGS:
            if not js[tag]:
                continue
            stems = sorted(set(js[tag]) & set(js[BASE]))
            cells = []
            for key, _, lower in METRICS:
                x = np.array([js[BASE][s][key] for s in stems])
                y = np.array([js[tag][s][key] for s in stems])
                cell = f"{y.mean():.4g}"
                if tag != BASE and len(stems):
                    ch = 100 * np.median((y - x) / x)
                    better = int(((y < x) if lower else (y > x)).sum())
                    p = wilcoxon(y, x).pvalue if np.any(x != y) else 1.0
                    cell += f" ({ch:+.1f}%, {better}/{len(stems)}, p={p:.2g})"
                cells.append(cell)
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
