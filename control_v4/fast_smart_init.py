"""Fast Smart Init: point-to-grid assignments that avoid the exact N x N OT solve.

The paper's Smart Init rejection-samples N points from the target density and assigns them to the
n x n OT grid with POT's ot.emd (data.Transforms.to_image_optimal_transport). The exact solve
dominates inference time from ~2300 points on (6400 points: ~6 s of a ~7 s total). The candidates
here produce the same (2, n, n) offset grid:

    exact      ot.emd with an iteration cap high enough to be optimal (reference)
    sortsort   sort by x into n columns of n points, then each column by y            O(N log N)
    bisection  recursive median split of the points, following a recursive split of the grid
               (columns or rows, whichever block side is longer)                      O(N log^2 N)
    *-swap     followed by local 2-opt passes: swap the points of two nearby cells whenever that
               lowers the squared transport cost
    *-blocks   followed by domain decomposition: exact assignment inside 8 x 8 cell tiles, with the
               tiling shifted by half a tile every other sweep (6 sweeps)
    kr         no rejection sampling and no assignment: the grid is pushed through the density's
               Knothe-Rosenblatt map (x marginal, then y within each equal-mass column strip), one
               jittered point per equal-mass cell, so cell (i, j) already owns its point
    kr-mid     as kr, at the middle of each cell's quantile range (deterministic)

Grid order matches to_image_optimal_transport: cell k = i * n + j has centre (x_j, y_i), and the
offsets are (point - centre) * n, shape (2, n, n), not clipped.
"""
import numpy as np


def grid_centers(n):
    """(N, 2) cell centres, cell k = i * n + j at (x_j, y_i)."""
    c = (np.arange(n) + 0.5) / n
    gx, gy = np.meshgrid(c, c)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def perm_to_offsets(points, perm, n):
    """points[perm[k]] is the point of cell k -> (2, n, n) offsets, as to_image_optimal_transport."""
    d = (points[perm] - grid_centers(n)) * n
    return d.T.reshape(2, n, n).astype(np.float32)


def transport_cost(points, perm, n):
    """Mean squared displacement point -> its cell centre, in cell units (1 = one cell width)."""
    d = (points[perm] - grid_centers(n)) * n
    return float((d * d).sum(axis=1).mean())


def _side(points):
    n = int(round(np.sqrt(points.shape[0])))
    if n * n != points.shape[0]:
        raise ValueError(f"{points.shape[0]} points is not a square number")
    return n


# ── Assignments of a given point set ─────────────────────────────────────────

def assign_exact(points, num_iter_max=10 ** 9):
    import ot

    n = _side(points)
    N = n * n
    M = ot.dist(grid_centers(n), points.astype(np.float64))
    a = np.full(N, 1.0 / N)
    G, log = ot.emd(a, a, M, numItermax=num_iter_max, log=True)
    if log.get("warning"):
        raise RuntimeError(f"ot.emd not optimal: {log['warning']}")
    rows, cols = np.nonzero(G)
    perm = np.empty(N, dtype=np.int64)
    perm[rows] = cols
    return perm


def assign_sortsort(points):
    n = _side(points)
    by_x = np.argsort(points[:, 0], kind="stable").reshape(n, n)       # column j = by_x[j]
    by_y = np.argsort(points[by_x, 1], axis=1, kind="stable")           # rank within the column
    cols = np.take_along_axis(by_x, by_y, axis=1)                       # cols[j, i] = point of (i, j)
    return cols.T.ravel()


def assign_bisection(points):
    """Recursive split; level by level, vectorized over the points."""
    n = _side(points)
    N = n * n
    node = np.zeros(N, dtype=np.int64)            # current block of every point
    boxes = np.array([[0, n, 0, n]])              # r0, nr, c0, nc per block
    while (boxes[:, 1] * boxes[:, 3]).max() > 1:
        r0, nr, c0, nc = boxes.T
        leaf = nr * nc == 1
        split_cols = (nc >= nr) & ~leaf
        split_rows = ~split_cols & ~leaf
        ncl, nrl = nc // 2, nr // 2
        n_left = np.where(split_cols, nr * ncl, np.where(split_rows, nrl * nc, 1))
        left = np.where(split_cols[:, None], np.stack([r0, nr, c0, ncl], 1),
                        np.where(split_rows[:, None], np.stack([r0, nrl, c0, nc], 1), boxes))
        right = np.where(split_cols[:, None], np.stack([r0, nr, c0 + ncl, nc - ncl], 1),
                         np.stack([r0 + nrl, nr - nrl, c0, nc], 1))
        # coordinate each point is ranked by inside its block: x for column splits, else y
        key = np.where(split_cols[node], points[:, 0], points[:, 1])
        order = np.lexsort((key, node))
        sorted_node = node[order]
        start = np.searchsorted(sorted_node, sorted_node, side="left")
        rank = np.empty(N, dtype=np.int64)
        rank[order] = np.arange(N) - start
        goes_left = rank < n_left[node]
        # children: left of block b -> 2b, right -> 2b + 1; leaves keep one child (their left)
        child = 2 * node + (~goes_left)
        boxes_new = np.empty((2 * len(boxes), 4), dtype=np.int64)
        boxes_new[0::2], boxes_new[1::2] = left, right
        used, node = np.unique(child, return_inverse=True)
        boxes = boxes_new[used]
    cell = boxes[node, 0] * n + boxes[node, 2]
    perm = np.empty(N, dtype=np.int64)
    perm[cell] = np.arange(N)
    return perm


SWAP_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0))


def refine_swaps(points, perm, max_passes=50, offsets=SWAP_OFFSETS):
    """Local 2-opt on the grid: swap the points of cells (i, j) and (i + di, j + dj) when
    (p_a - p_b) . (g_a - g_b) < 0, i.e. when the swap lowers the squared transport cost.
    Each (offset, parity) set of pairs is disjoint, so it is applied in one vectorized step."""
    n = _side(points)
    perm = perm.copy()
    g = grid_centers(n).reshape(n, n, 2)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pair_sets = []
    for di, dj in offsets:
        i2, j2 = ii + di, jj + dj
        inside = (i2 >= 0) & (i2 < n) & (j2 >= 0) & (j2 < n)
        colour = (ii // di) % 2 if di else (jj // dj) % 2
        for parity in (0, 1):
            m = inside & (colour == parity)
            a = (ii[m] * n + jj[m])
            b = (i2[m] * n + j2[m])
            dg = g[ii[m], jj[m]] - g[i2[m], j2[m]]
            pair_sets.append((a, b, dg))
    for _ in range(max_passes):
        swapped = 0
        for a, b, dg in pair_sets:
            pa, pb = perm[a], perm[b]
            better = ((points[pa] - points[pb]) * dg).sum(axis=1) < 0
            if better.any():
                perm[a[better]], perm[b[better]] = pb[better], pa[better]
                swapped += int(better.sum())
        if not swapped:
            break
    return perm


def refine_blocks(points, perm, block=8, sweeps=6):
    """Domain decomposition: split the grid into block x block cell tiles and solve the exact
    assignment inside each tile (its cells and the points they currently hold). Odd sweeps shift
    the tiling by block // 2, so points can move across tile borders. The cost never increases."""
    from scipy.optimize import linear_sum_assignment

    n = _side(points)
    perm = perm.copy()
    g = grid_centers(n)
    idx = np.arange(n * n).reshape(n, n)
    for s in range(sweeps):
        shift = (block // 2) * (s % 2)
        edges = sorted({0, n, *range((shift or block), n, block)})
        for r0, r1 in zip(edges[:-1], edges[1:]):
            for c0, c1 in zip(edges[:-1], edges[1:]):
                cells = idx[r0:r1, c0:c1].ravel()
                pts = perm[cells]
                d = g[cells][:, None, :] - points[pts][None, :, :]
                _, col = linear_sum_assignment((d * d).sum(axis=2))
                perm[cells] = pts[col]
    return perm


def assign_bisection_swap(points):
    return refine_swaps(points, assign_bisection(points))


def assign_bisection_blocks(points, block=8, sweeps=6):
    return refine_blocks(points, assign_bisection(points), block, sweeps)


ASSIGNERS = {
    "exact": assign_exact,
    "sortsort": assign_sortsort,
    "bisection": assign_bisection,
    "bisection-swap": assign_bisection_swap,
    "bisection-blocks": assign_bisection_blocks,
    "bisection-blocks16": lambda p: assign_bisection_blocks(p, 16, 4),
}


# ── Knothe-Rosenblatt grid warp (no rejection sampling, no assignment) ───────

def _inverse_cdf(mass, u):
    """Piecewise-linear inverse CDF of a 1D histogram (bins [k, k+1)); u in [0, 1]."""
    edges = np.arange(len(mass) + 1, dtype=np.float64)
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    return np.interp(u * cdf[-1], cdf, edges)


def kr_points(image_01, n, seed=42, jitter=True):
    """Points (N, 2) in [0, 1] in cell order k = i * n + j, from the ink density 1 - image."""
    prob = np.clip(1.0 - np.asarray(image_01, dtype=np.float64), 0.0, 1.0)
    h, w = prob.shape
    if prob.sum() <= 0.0:
        prob = np.ones_like(prob)
    prob = prob + 1e-9 * prob.mean()              # strictly increasing CDFs
    rng = np.random.RandomState(seed)
    ux = (np.arange(n) + (rng.uniform(size=(n, n)) if jitter else 0.5)) / n   # (i, j) -> column j
    uy = (np.arange(n)[:, None] + (rng.uniform(size=(n, n)) if jitter else 0.5)) / n

    # x: marginal over columns, one equal-mass strip per grid column j
    x = _inverse_cdf(prob.sum(axis=0), ux)                                      # (n, n)
    strip_edges = _inverse_cdf(prob.sum(axis=0), np.arange(n + 1) / n)          # (n + 1,)
    # y: within strip j, the mass of each pixel row between the strip's two x edges
    cx = np.concatenate([np.zeros((h, 1)), np.cumsum(prob, axis=1)], axis=1)    # (h, w + 1)

    def cx_at(xe):
        x0 = np.clip(np.floor(xe).astype(int), 0, w - 1)
        f = xe - x0
        return cx[:, x0] * (1 - f) + cx[:, x0 + 1] * f                          # (h, len(xe))

    row_mass = cx_at(strip_edges[1:]) - cx_at(strip_edges[:-1])                 # (h, n)
    y = np.empty((n, n))
    for j in range(n):
        y[:, j] = _inverse_cdf(np.maximum(row_mass[:, j], 0.0) + 1e-12, uy[:, j])
    pts = np.stack([(x / w).ravel(), (y / h).ravel()], axis=1)
    return pts.astype(np.float32)
