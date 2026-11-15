#!/usr/bin/env python3
"""Benchmark the OT-grid assignment used by Smart Init (rebuttal E18).

data.Transforms.to_image_optimal_transport assigns N points to the N cells of a sqrt(N) x sqrt(N)
grid with POT's network simplex (ot.emd on uniform weights, default numItermax = 100000). With
uniform weights and as many points as cells the optimal plan is a permutation, so
scipy.optimize.linear_sum_assignment on the same squared-distance cost is the exact reference.
This script runs every solver of control_v4/ot_assignment.py on real Smart Init samples at every
point budget of the runtime figure and reports its time and its total cost above the exact optimum.
The paper path ("pot") is also flagged when ot.emd warns that it stopped before optimality.

Writes outputs/ot_assignment_benchmark/results.jsonl and summary.md.

Run from the project root (GPU job, the auction solver runs on CUDA):
    python experiments/ot_assignment_benchmark.py
"""
import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control_v4.ot_assignment import assign, grid_centers  # noqa: E402

SOURCE_DIR = "experiments/outputs/runtime_fewstep-Icons-50/source"
OUTPUT_DIR = "experiments/outputs/ot_assignment_benchmark"
GRIDS = [16, 24, 32, 40, 48, 56, 64, 72, 80]
IMAGES = 3
SMART_INIT_SEED = 42
REPEATS = 1
SOLVERS = ["pot", "pot-exact", "scipy", "auction"]
REFERENCE = "scipy"
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=SOURCE_DIR)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--grids", type=int, nargs="+", default=GRIDS)
    p.add_argument("--images", type=int, default=IMAGES, help="Number of images (sorted) per grid size")
    p.add_argument("--smart_init_seed", type=int, default=SMART_INIT_SEED)
    p.add_argument("--repeats", type=int, default=REPEATS, help="Timing repeats per solver (min is kept)")
    p.add_argument("--solvers", nargs="+", default=SOLVERS)
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


def assign_pot(grid, points):
    """The paper path of to_image_optimal_transport (default numItermax): cell i -> point perm[i]."""
    import ot
    N = len(points)
    G0 = ot.emd(np.ones(N) / N, np.ones(N) / N, ot.dist(grid, points))
    rows, cols = np.nonzero(G0)
    perm = np.empty(N, dtype=np.int64)
    perm[rows] = cols
    return perm


def run_solver(name, grid, points, device):
    if name == "pot":
        return assign_pot(grid, points)
    return assign(grid, points, name, device=device)


def timed(fn, *a, repeats=1):
    best, out = float("inf"), None
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a)
        best = min(best, time.perf_counter() - t0)
    return out, best


def main():
    args = parse_args()
    from control_v4.sample_control import load_condition
    from control_v4.smart_init import generate_smart_init_points_from_density

    if REFERENCE not in args.solvers:
        args.solvers = [REFERENCE] + args.solvers
    images = sorted(p for p in Path(args.source).iterdir() if p.suffix.lower() == ".png")[: args.images]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if "auction" in args.solvers:  # CUDA context and kernels, outside the timings
        run_solver("auction", grid_centers(8), np.random.rand(64, 2), args.device)

    rows = []
    with open(out / "results.jsonl", "w") as f:
        for g in args.grids:
            grid = grid_centers(g)
            for img in images:
                image_01 = load_condition(img, g, "cpu", sdf_features=False)[0]
                pts = generate_smart_init_points_from_density(image_01, n_points=g * g, seed=args.smart_init_seed)
                pts = np.asarray(pts, np.float64)
                cost = lambda p: float(((grid - pts[p]) ** 2).sum())  # noqa: E731
                r = {"grid": g, "n": g * g, "image": img.stem, "gpu": gpu, "solvers": {}}
                perms = {}
                for s in args.solvers:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        perms[s], t = timed(run_solver, s, grid, pts, args.device, repeats=args.repeats)
                    r["solvers"][s] = {"time": t, "cost": cost(perms[s]),
                                       "truncated": any("numItermax" in str(x.message) for x in w)}
                ref = r["solvers"][REFERENCE]["cost"]
                for s in args.solvers:
                    d = r["solvers"][s]
                    d["excess_pct"] = 100.0 * (d["cost"] - ref) / ref
                    d["cells_different"] = int((perms[s] != perms[REFERENCE]).sum())
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"n={g * g:5d} {img.stem[:28]:28s} " + "  ".join(
                    f"{s} {d['time']:7.3f}s {d['excess_pct']:+.2e}%{' TRUNC' if d['truncated'] else ''}"
                    for s, d in r["solvers"].items()), flush=True)

    lines = ["# OT-grid assignment solvers on Smart Init samples", "",
             f"{len(images)} Smart Init samples per budget (seed {args.smart_init_seed}); GPU: {gpu}.",
             f"Time: median seconds. Excess: total squared-distance cost above the exact optimum ({REFERENCE}), "
             "max over the samples, in %. Truncated: ot.emd stopped at numItermax.", "",
             "| points | " + " | ".join(f"{s} time | {s} excess %" for s in args.solvers) + " | pot truncated |",
             "|---|" + "---|---|" * len(args.solvers) + "---|"]
    for g in args.grids:
        r = [x for x in rows if x["grid"] == g]
        cells = []
        for s in args.solvers:
            cells.append(f"{statistics.median(x['solvers'][s]['time'] for x in r):.3f}")
            cells.append(f"{max(x['solvers'][s]['excess_pct'] for x in r):.2g}")
        trunc = sum(x["solvers"].get("pot", {}).get("truncated", False) for x in r)
        lines.append(f"| {g * g} | " + " | ".join(cells) + f" | {trunc}/{len(r)} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
