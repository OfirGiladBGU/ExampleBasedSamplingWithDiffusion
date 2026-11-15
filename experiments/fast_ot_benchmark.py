#!/usr/bin/env python3
"""Fast Smart Init assignment: transport cost and time of each candidate vs the exact solve.

For every image and budget, draws the Smart Init points exactly as inference does
(generate_smart_init_points_from_density, seed 42) and assigns them to the grid with every
method of control_v4/fast_smart_init.ASSIGNERS. Reports, per budget and method:
    * cost ratio vs exact (mean squared point-to-cell displacement; 1.000 = optimal), median / max
    * median assignment time, and the median rejection-sampling time (shared by all methods)

Images: the 10 pinned runtime images at every budget, plus the first 200 Icons-50 validation
images at the training budget (1024).

Writes <output>/summary.md and <output>/rows.jsonl.

    python experiments/fast_ot_benchmark.py
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control_v4.fast_smart_init import ASSIGNERS, transport_cost  # noqa: E402
from control_v4.smart_init import generate_smart_init_points_from_density  # noqa: E402

RUNTIME_SOURCE = "experiments/outputs/runtime_fewstep-Icons-50/source"
ICONS_SOURCE = "experiments/outputs/z_validation_data/Icons-50_1024/source"
ICONS_LIMIT = 200
GRIDS = [16, 24, 32, 40, 48, 56, 64, 72, 80]
OUTPUT_DIR = "experiments/outputs/fast_ot_benchmark"
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--grids", type=int, nargs="+", default=GRIDS)
    p.add_argument("--icons-limit", type=int, default=ICONS_LIMIT)
    p.add_argument("--methods", default=",".join(ASSIGNERS))
    return p.parse_args()


def images_in(folder, limit=-1):
    imgs = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return imgs[:limit] if limit > 0 else imgs


def run_one(image_path, n, methods):
    image_01 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    t0 = time.perf_counter()
    pts = generate_smart_init_points_from_density(image_01, n_points=n * n, seed=SEED)
    t_rej = time.perf_counter() - t0
    rows, exact = [], None
    for m in methods:
        t0 = time.perf_counter()
        perm = ASSIGNERS[m](pts)
        dt = time.perf_counter() - t0
        cost = transport_cost(pts, perm, n)
        exact = cost if m == "exact" else exact
        rows.append({"image": image_path.stem, "points": n * n, "method": m, "cost": cost,
                     "time": dt, "time_rejection": t_rej})
    for r in rows:
        r["ratio"] = r["cost"] / exact
        r["excess"] = r["cost"] - exact
    return rows


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    if methods[0] != "exact":
        methods = ["exact"] + [m for m in methods if m != "exact"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("runtime", p, g) for g in args.grids for p in images_in(RUNTIME_SOURCE)]
    jobs += [("icons200", p, 32) for p in images_in(ICONS_SOURCE, args.icons_limit)]
    rows = []
    with open(out / "rows.jsonl", "w") as f:
        for i, (subset, p, g) in enumerate(jobs, 1):
            for r in run_one(p, g, methods):
                r["subset"] = subset
                rows.append(r)
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{i}/{len(jobs)}] {subset} {g * g} {p.stem}", flush=True)

    lines = ["# Fast Smart Init assignment vs exact OT", "",
             "Cost = mean squared point-to-cell displacement (cell units); ratio vs exact on the same points.",
             "Cells: median ratio / max ratio / median excess (cell^2) / median time.",
             "(Near-uniform densities have a small optimal cost, so the ratio overstates their excess.)", ""]
    for subset in ("runtime", "icons200"):
        sub = [r for r in rows if r["subset"] == subset]
        budgets = sorted({r["points"] for r in sub})
        n_img = len({r["image"] for r in sub})
        lines += [f"## {subset} ({n_img} images)", "",
                  "| points | rejection s | " + " | ".join(methods) + " |",
                  "|---|---|" + "---|" * len(methods)]
        for b in budgets:
            rb = [r for r in sub if r["points"] == b]
            cells = [str(b), f"{st.median(r['time_rejection'] for r in rb if r['method'] == 'exact'):.4f}"]
            for m in methods:
                rm = [r for r in rb if r["method"] == m]
                ratios = [r["ratio"] for r in rm]
                cells.append(f"{st.median(ratios):.3f} / {max(ratios):.3f} / "
                             f"{st.median(r['excess'] for r in rm):.2f} / {st.median(r['time'] for r in rm):.4f} s")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
