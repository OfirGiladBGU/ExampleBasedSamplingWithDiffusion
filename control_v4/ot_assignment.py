"""Point-to-grid OT assignment for Smart Init, with a choice of solver.

data.Transforms.to_image_optimal_transport assigns the N Smart Init points to the N cells of the
sqrt(N) x sqrt(N) OT grid with POT's network simplex (ot.emd, uniform weights). ot.emd stops at its
default numItermax = 100000 pivots, which is enough up to ~2304 points but not from ~3136 points up:
it then returns a feasible but non-optimal assignment (and a "numItermax reached" warning).

With uniform weights and as many points as cells the optimal plan is a permutation, so every solver
here returns cell i -> point perm[i] and the offsets are built exactly as in
to_image_optimal_transport:

    pot        the paper path, to_image_optimal_transport itself (truncated from ~3136 points)
    pot-exact  ot.emd with a numItermax high enough to reach optimality (CPU)
    scipy      scipy.optimize.linear_sum_assignment (Jonker-Volgenant, exact, CPU)
    auction    Bertsekas' auction algorithm with epsilon scaling on the GPU (torch). The final
               assignment is within N * eps_final of the optimal total cost.
    bisection       approximate: recursive median split of the points along a recursive split of
                    the grid, O(N log^2 N), milliseconds (control_v4/fast_smart_init.py)
    bisection-swap  bisection + local 2-opt swaps between nearby cells (cost never increases)
    bisection-blocks  bisection + exact assignment inside shifted 8 x 8 cell tiles (6 sweeps)
    bisection-blocks16  the same with 16 x 16 tiles and 4 sweeps (fast_smart_init.ASSIGNERS)
"""
import numpy as np
import torch

OT_SOLVERS = ("pot", "pot-exact", "scipy", "auction", "bisection", "bisection-swap", "bisection-blocks", "bisection-blocks16")
POT_EXACT_MAX_ITER = 1_000_000_000
AUCTION_EPS_FINAL = 1e-9     # absolute, on squared distances in the unit square
AUCTION_EPS_FACTOR = 5.0     # epsilon is divided by this between scaling phases
AUCTION_DEVICE = "cuda"


def grid_centers(n, d=2):
    """Cell centers in the exact order used by to_image_optimal_transport."""
    g = np.linspace(0, 1, n, endpoint=False)
    g = np.meshgrid(*[g] * d)
    g = np.vstack([a.ravel() for a in g]).T.reshape((n ** d, d))
    return g + 1 / (2 * n)


def assign_pot_exact(grid, points, num_iter_max=POT_EXACT_MAX_ITER):
    import ot
    N = len(points)
    G0, log = ot.emd(np.ones(N) / N, np.ones(N) / N, ot.dist(grid, points), numItermax=num_iter_max, log=True)
    if log.get("warning"):
        raise RuntimeError(f"ot.emd did not reach optimality: {log['warning']}")
    rows, cols = np.nonzero(G0)
    perm = np.empty(N, dtype=np.int64)
    perm[rows] = cols
    return perm


def assign_scipy(grid, points):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    rows, cols = linear_sum_assignment(cdist(grid, points, "sqeuclidean"))
    perm = np.empty(len(points), dtype=np.int64)
    perm[rows] = cols
    return perm


def assign_auction(grid, points, device=AUCTION_DEVICE, eps_final=AUCTION_EPS_FINAL, eps_factor=AUCTION_EPS_FACTOR):
    """Forward auction (cells bid for points) with epsilon scaling, Jacobi bidding on the GPU.

    Maximizes the benefit -||grid_i - point_j||^2. Every phase keeps the prices of the previous one
    and restarts the assignment; all unassigned cells bid at once and each point goes to its highest
    bidder. The last phase runs at eps <= eps_final, so the total cost is within N * eps_final of
    the optimum.
    """
    g = torch.as_tensor(grid, dtype=torch.float64, device=device)
    p = torch.as_tensor(points, dtype=torch.float64, device=device)
    N = g.shape[0]
    benefit = -((g * g).sum(1, keepdim=True) + (p * p).sum(1)[None, :] - 2.0 * g @ p.T)
    price = torch.zeros(N, dtype=torch.float64, device=device)
    arange = torch.arange(N, device=device)
    eps = float(benefit.max() - benefit.min()) / eps_factor
    while True:
        eps = max(eps / eps_factor, eps_final)
        owner = torch.full((N,), -1, dtype=torch.int64, device=device)    # point -> cell
        assigned = torch.full((N,), -1, dtype=torch.int64, device=device)  # cell -> point
        while True:
            bidders = torch.nonzero(assigned < 0).squeeze(1)
            if bidders.numel() == 0:
                break
            top = (benefit[bidders] - price).topk(2, dim=1)
            target = top.indices[:, 0]
            bid = price[target] + (top.values[:, 0] - top.values[:, 1]) + eps
            best = torch.full((N,), -float("inf"), dtype=torch.float64, device=device)
            best.scatter_reduce_(0, target, bid, reduce="amax")
            # one winner per point: among the bidders at the highest bid, the largest cell index
            cand = torch.where(bid == best[target], bidders, torch.full_like(bidders, -1))
            winner = torch.full((N,), -1, dtype=torch.int64, device=device)
            winner.scatter_reduce_(0, target, cand, reduce="amax")
            won = arange[winner >= 0]
            prev = owner[won]
            assigned[prev[prev >= 0]] = -1
            owner[won] = winner[won]
            assigned[winner[won]] = won
            price[won] = best[won]
        if eps <= eps_final:
            break
    return assigned.cpu().numpy()


def assign(grid, points, solver, device=AUCTION_DEVICE):
    """Cell i -> point perm[i] for the non-paper solvers."""
    if solver == "pot-exact":
        return assign_pot_exact(grid, points)
    if solver == "scipy":
        return assign_scipy(grid, points)
    if solver == "auction":
        return assign_auction(grid, points, device=device)
    if solver in ("bisection", "bisection-swap", "bisection-blocks", "bisection-blocks16"):
        from control_v4.fast_smart_init import ASSIGNERS as FAST
        return FAST[solver](points)
    raise ValueError(f"Unknown OT solver '{solver}', expected one of {OT_SOLVERS}")


def points_to_offsets(points, solver="pot", device=AUCTION_DEVICE):
    """Same output as to_image_optimal_transport(points): offsets (D, n, ..., n) in grid-cell units."""
    if solver == "pot":
        from data.Transforms import to_image_optimal_transport
        return to_image_optimal_transport(points)
    N, D = points.shape
    n = int(round(N ** (1 / D)))
    if n ** D != N:
        raise ValueError(f"Can't perform OT:  {N} != {n} ** {D}")
    grid = grid_centers(n, D)
    perm = assign(grid, np.asarray(points, dtype=np.float64), solver, device=device)
    offsets = (points[perm] - grid) * n
    return offsets.T.reshape(D, *[n] * D)
