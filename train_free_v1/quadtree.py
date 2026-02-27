"""Capacity-constrained quadtree for density-adaptive stippling.

Recursively splits a grayscale image into cells where each leaf requires
at most ``max_points`` stipple points.  Density is encoded by cell *size*:
small cells in dark regions pack points tightly, large cells in bright
regions space them out.

Each leaf stores:
  - bounding box (x, y, width) in normalised [0, 1] coordinates
  - integer ``budget`` (1..max_points) for post-hoc culling after sampling
  - list of neighbour leaf indices for boundary-energy computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Cell:
    """A single quadtree leaf ready for diffusion sampling."""

    x: float               # left edge in [0, 1]
    y: float               # top  edge in [0, 1]
    width: float            # side length in [0, 1] (cells are square)
    budget: int             # target point count (1..max_points)
    neighbors: List[int] = field(default_factory=list)  # indices into leaf list

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.width / 2

    @property
    def spacing(self) -> float:
        """Expected inter-point distance assuming an 8x8 grid."""
        return self.width / 8


# ── internal node used during recursive splitting ────────────────────

class _QuadNode:
    __slots__ = ("x", "y", "w", "darkness", "children", "leaf_idx")

    def __init__(self, x: float, y: float, w: float, darkness: float):
        self.x = x
        self.y = y
        self.w = w
        self.darkness = darkness
        self.children: Optional[List[_QuadNode]] = None
        self.leaf_idx: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


# ── public API ───────────────────────────────────────────────────────

def build_quadtree(
    image: np.ndarray,
    total_budget: int,
    max_points: int = 64,
    min_cell_pixels: int = 4,
    skip_threshold: float = 0.5,
) -> List[Cell]:
    """Build a capacity-constrained quadtree from a grayscale image.

    Parameters
    ----------
    image : ndarray (H, W), uint8 or float [0, 255]
        Grayscale source image.  Dark pixels = high point density.
    total_budget : int
        Total number of stipple points across the entire image.
    max_points : int
        Maximum points per cell (must equal the diffusion grid size**2,
        i.e. 64 for an 8x8 model).
    min_cell_pixels : int
        Stop splitting when a cell covers fewer than this many pixels
        on the shortest side.
    skip_threshold : float
        Cells whose budget rounds to fewer than this many points are
        discarded entirely (white / near-white regions).

    Returns
    -------
    list[Cell]
        Leaf cells with budgets and populated neighbor lists.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        image = image.mean(axis=-1)

    H, W = image.shape
    assert H == W, f"Image must be square, got {H}x{W}"

    darkness = 255.0 - image
    darkness = np.clip(darkness, 0, 255)
    total_darkness = darkness.sum()
    if total_darkness < 1e-9:
        return []

    root = _build_node(darkness, 0.0, 0.0, 1.0, H, total_darkness,
                        total_budget, max_points, min_cell_pixels,
                        skip_threshold)
    if root is None:
        return []

    leaves: List[Cell] = []
    _collect_leaves(root, leaves)

    _find_neighbors(leaves)

    return leaves


# ── recursive splitting ──────────────────────────────────────────────

def _region_darkness(darkness_map: np.ndarray,
                     x: float, y: float, w: float, img_size: int) -> float:
    """Sum the darkness values inside a normalised bounding box."""
    px = int(round(x * img_size))
    py = int(round(y * img_size))
    pw = int(round(w * img_size))
    px = max(0, min(px, img_size))
    py = max(0, min(py, img_size))
    pw = max(1, pw)
    x2 = min(px + pw, img_size)
    y2 = min(py + pw, img_size)
    return float(darkness_map[py:y2, px:x2].sum())


def _build_node(
    darkness_map: np.ndarray,
    x: float, y: float, w: float,
    img_size: int,
    total_darkness: float,
    total_budget: int,
    max_points: int,
    min_cell_pixels: int,
    skip_threshold: float,
) -> Optional[_QuadNode]:
    dark = _region_darkness(darkness_map, x, y, w, img_size)
    budget = dark / total_darkness * total_budget

    if budget < skip_threshold:
        return None

    cell_pixels = int(round(w * img_size))
    can_split = cell_pixels >= 2 * min_cell_pixels

    if budget <= max_points or not can_split:
        return _QuadNode(x, y, w, dark)

    hw = w / 2
    children = []
    for dy in (0, hw):
        for dx in (0, hw):
            child = _build_node(darkness_map, x + dx, y + dy, hw,
                                img_size, total_darkness, total_budget,
                                max_points, min_cell_pixels, skip_threshold)
            if child is not None:
                children.append(child)

    if not children:
        return None

    if len(children) == 1:
        return children[0]

    node = _QuadNode(x, y, w, dark)
    node.children = children
    return node


def _collect_leaves(node: _QuadNode, out: List[Cell]) -> None:
    if node.is_leaf:
        budget = max(1, int(round(
            node.darkness / 1.0  # raw darkness stored; caller normalises
        )))
        # re-derive budget proportionally (darkness was stored raw)
        node.leaf_idx = len(out)
        out.append(Cell(x=node.x, y=node.y, width=node.w,
                        budget=budget))
    else:
        for ch in node.children:
            _collect_leaves(ch, out)


def _renormalize_budgets(leaves: List[Cell], total_budget: int,
                         total_darkness: float, max_points: int) -> None:
    """Assign integer budgets that sum to total_budget."""
    raw = np.array([c.budget for c in leaves], dtype=np.float64)
    total_raw = raw.sum()
    if total_raw < 1e-9:
        return
    proportional = raw / total_raw * total_budget
    budgets = np.clip(np.round(proportional).astype(int), 1, max_points)
    for cell, b in zip(leaves, budgets):
        cell.budget = int(b)


# ── neighbor detection ───────────────────────────────────────────────

def _cells_are_neighbors(a: Cell, b: Cell, tol: float = 1e-6) -> bool:
    """Two axis-aligned squares are neighbors if they share a boundary segment."""
    a_left, a_right = a.x, a.x + a.width
    a_top, a_bot = a.y, a.y + a.width
    b_left, b_right = b.x, b.x + b.width
    b_top, b_bot = b.y, b.y + b.width

    horiz_overlap = (min(a_right, b_right) - max(a_left, b_left)) > tol
    vert_overlap = (min(a_bot, b_bot) - max(a_top, b_top)) > tol

    touch_lr = abs(a_right - b_left) < tol or abs(b_right - a_left) < tol
    touch_tb = abs(a_bot - b_top) < tol or abs(b_bot - a_top) < tol

    if touch_lr and vert_overlap:
        return True
    if touch_tb and horiz_overlap:
        return True
    return False


def _find_neighbors(leaves: List[Cell]) -> None:
    """Populate neighbor lists for all leaves (brute-force, O(K^2))."""
    n = len(leaves)
    for i in range(n):
        for j in range(i + 1, n):
            if _cells_are_neighbors(leaves[i], leaves[j]):
                leaves[i].neighbors.append(j)
                leaves[j].neighbors.append(i)


# ── convenience ──────────────────────────────────────────────────────

def build_and_normalize(
    image: np.ndarray,
    total_budget: int,
    max_points: int = 64,
    min_cell_pixels: int = 4,
    skip_threshold: float = 0.5,
) -> List[Cell]:
    """Build quadtree and ensure budgets are correctly normalised.

    This is the recommended entry point.  It calls :func:`build_quadtree`
    and then rescales the per-cell budgets so they are proportional to
    local darkness and sum close to ``total_budget``.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 3:
        image = image.mean(axis=-1)

    H, W = image.shape
    darkness = np.clip(255.0 - image, 0, 255)
    total_darkness = darkness.sum()
    if total_darkness < 1e-9:
        return []

    leaves = build_quadtree(image, total_budget, max_points,
                            min_cell_pixels, skip_threshold)
    if not leaves:
        return leaves

    leaf_darkness = []
    for cell in leaves:
        d = _region_darkness(darkness, cell.x, cell.y, cell.width, H)
        leaf_darkness.append(d)
    leaf_darkness = np.array(leaf_darkness)

    total_leaf_dark = leaf_darkness.sum()
    if total_leaf_dark < 1e-9:
        return []

    proportional = leaf_darkness / total_leaf_dark * total_budget
    budgets = np.clip(np.round(proportional).astype(int), 1, max_points)
    for cell, b in zip(leaves, budgets):
        cell.budget = int(b)

    return leaves


# ── visualisation helper ─────────────────────────────────────────────

def visualize_quadtree(image: np.ndarray, leaves: List[Cell],
                       save_path: str) -> None:
    """Draw quadtree cell boundaries on the source image and save."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        return

    H, W = image.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Source image")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
    for cell in leaves:
        rect = patches.Rectangle(
            (cell.x * W, cell.y * H), cell.width * W, cell.width * H,
            linewidth=0.5, edgecolor="cyan", facecolor="none")
        axes[1].add_patch(rect)
        axes[1].text(
            (cell.cx) * W, (cell.cy) * H, str(cell.budget),
            ha="center", va="center", fontsize=4, color="yellow")
    axes[1].set_title(f"Quadtree: {len(leaves)} cells, "
                      f"{sum(c.budget for c in leaves)} total pts")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
