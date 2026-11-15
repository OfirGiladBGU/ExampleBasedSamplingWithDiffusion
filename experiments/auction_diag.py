#!/usr/bin/env python3
"""Diagnostics for the GPU auction assignment: rounds per phase and time for several variants."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control_v4.ot_assignment import assign_scipy, grid_centers  # noqa: E402

SOURCE_DIR = "experiments/outputs/runtime_fewstep-Icons-50/source"
GRIDS = [32, 64, 80]
DEVICE = "cuda"


def auction(grid, points, eps_final, eps_factor, keep, eps_start_rel, device=DEVICE):
    g = torch.as_tensor(grid, dtype=torch.float64, device=device)
    p = torch.as_tensor(points, dtype=torch.float64, device=device)
    N = g.shape[0]
    benefit = -((g * g).sum(1, keepdim=True) + (p * p).sum(1)[None, :] - 2.0 * g @ p.T)
    price = torch.zeros(N, dtype=torch.float64, device=device)
    arange = torch.arange(N, device=device)
    owner = torch.full((N,), -1, dtype=torch.int64, device=device)
    assigned = torch.full((N,), -1, dtype=torch.int64, device=device)
    eps = float(benefit.max() - benefit.min()) * eps_start_rel
    rounds = []
    while True:
        eps = max(eps / eps_factor, eps_final)
        if keep:
            # keep the pairs that already satisfy eps-complementary slackness
            net = benefit - price
            ok = assigned >= 0
            own = net[arange, assigned.clamp(min=0)]
            bad = ok & (own < net.max(1).values - eps)
            owner[assigned[bad]] = -1
            assigned[bad] = -1
        else:
            owner.fill_(-1)
            assigned.fill_(-1)
        r = 0
        while True:
            bidders = torch.nonzero(assigned < 0).squeeze(1)
            if bidders.numel() == 0:
                break
            r += 1
            top = (benefit[bidders] - price).topk(2, dim=1)
            target = top.indices[:, 0]
            bid = price[target] + (top.values[:, 0] - top.values[:, 1]) + eps
            best = torch.full((N,), -float("inf"), dtype=torch.float64, device=device)
            best.scatter_reduce_(0, target, bid, reduce="amax")
            cand = torch.where(bid == best[target], bidders, torch.full_like(bidders, -1))
            winner = torch.full((N,), -1, dtype=torch.int64, device=device)
            winner.scatter_reduce_(0, target, cand, reduce="amax")
            won = arange[winner >= 0]
            prev = owner[won]
            assigned[prev[prev >= 0]] = -1
            owner[won] = winner[won]
            assigned[winner[won]] = won
            price[won] = best[won]
        rounds.append(r)
        if eps <= eps_final:
            break
    return assigned.cpu().numpy(), rounds


VARIANTS = [
    # name, eps_final, eps_factor, keep, eps_start_rel
    ("baseline", 1e-9, 5.0, False, 1 / 5),
    ("keep", 1e-9, 5.0, True, 1 / 5),
    ("keep-f10", 1e-9, 10.0, True, 1 / 5),
    ("keep-f10-e1e-7", 1e-7, 10.0, True, 1 / 5),
    ("keep-f10-start1e-2", 1e-9, 10.0, True, 1e-2),
    ("keep-f4-start1e-2", 1e-9, 4.0, True, 1e-2),
]


def main():
    from control_v4.sample_control import load_condition
    from control_v4.smart_init import generate_smart_init_points_from_density
    img = sorted(p for p in Path(SOURCE_DIR).iterdir() if p.suffix.lower() == ".png")[0]
    auction(grid_centers(8), np.random.rand(64, 2), 1e-9, 5.0, True, 0.2)
    for g in GRIDS:
        grid = grid_centers(g)
        image_01 = load_condition(img, g, "cpu", sdf_features=False)[0]
        pts = np.asarray(generate_smart_init_points_from_density(image_01, n_points=g * g, seed=42), np.float64)
        cost = lambda q: float(((grid - pts[q]) ** 2).sum())  # noqa: E731
        ref = cost(assign_scipy(grid, pts))
        for name, ef, fac, keep, st in VARIANTS:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            perm, rounds = auction(grid, pts, ef, fac, keep, st)
            dt = time.perf_counter() - t0
            print(f"n={g * g:5d} {name:20s} {dt:7.2f}s excess {100 * (cost(perm) - ref) / ref:+.1e}% "
                  f"rounds {sum(rounds)} per phase {rounds}", flush=True)


if __name__ == "__main__":
    main()
