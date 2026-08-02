"""Visibility gate (M0) for the spacing-regularity axis -- can a human SEE the WVS/GBN
difference in dot-only renders?

The metric gate PASSED (WVS more regular than GBN at fixed density, d~=-2 on norm_nn_cv). But the
anisotropy branch also passed its metric gates and then DIED on M0: the property was not legible
in dot-only renders. Regularity (CVT-like vs blue-noise) is exactly the kind of difference that
can be metric-real yet invisible. This script renders both oracles as IDENTICAL-STYLE black dots
(re-rendered from extracted centroids, so dot size/AA cannot be the tell -- only arrangement
differs) at full size and at reading-thumbnail size, side by side, for eyeballing.

Selection: by default it picks icons spanning the separation range (largest, median, smallest
per-icon norm_nn_cv gap) so you judge best-case AND typical legibility, not a cherry pick.

Run on the cluster; then open the PNGs.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spacing_regularity as sr  # noqa: E402
from edge_alignment import extract_centroids  # noqa: E402
from gate0_run import (  # noqa: E402
    DEFAULT_GBN_ROOT, DEFAULT_WVS_ROOT, VALID_EXT, _stem_map, resolve_dirs,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw):
        return it


def render_dots(pts, size=512, radius=2):
    """White canvas, identical black filled dots at each point. Returns uint8 (size,size)."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 255)
    dr = ImageDraw.Draw(img)
    for x, y in pts:
        cx, cy = x * size, y * size
        dr.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=0)
    return np.asarray(img, dtype=np.uint8)


def select_stems(stems, wvs_tgt, gbn_tgt, n, mode, min_area, k):
    if mode == "first":
        return list(stems)[:n]
    # compute per-icon norm_nn_cv gap on a capped sample, then span it
    sample = list(stems)[: max(n * 30, 200)]
    gaps = []
    for stem in tqdm(sample, desc="ranking", unit="icon", dynamic_ncols=True):
        pw = extract_centroids(wvs_tgt[stem], min_area=min_area)
        pg = extract_centroids(gbn_tgt[stem], min_area=min_area)
        bw = sr.compute_spacing_bundle(pw, k=k)
        bg = sr.compute_spacing_bundle(pg, k=k)
        gap = bg["norm_nn_cv"] - bw["norm_nn_cv"]  # + => GBN less regular (expected)
        if gap == gap:
            gaps.append((gap, stem))
    gaps.sort()
    if not gaps:
        return list(stems)[:n]
    if mode == "worst":  # smallest / most adverse gaps (hardest to see)
        return [s for _, s in gaps[:n]]
    # mode == "span": largest, evenly down to smallest
    idxs = np.linspace(0, len(gaps) - 1, n).round().astype(int)[::-1]
    return [gaps[i][1] for i in idxs]


def main():
    p = argparse.ArgumentParser(description="Spacing-regularity visibility (M0) gate.")
    p.add_argument("--wvs-root", default=DEFAULT_WVS_ROOT)
    p.add_argument("--gbn-root", default=DEFAULT_GBN_ROOT)
    p.add_argument("--wvs-source", default=""); p.add_argument("--wvs-target", default="")
    p.add_argument("--gbn-source", default=""); p.add_argument("--gbn-target", default="")
    p.add_argument("--wvs-offsets", default=""); p.add_argument("--gbn-offsets", default="")
    p.add_argument("--n", type=int, default=6, help="how many icons to show")
    p.add_argument("--select", choices=("span", "worst", "first"), default="span")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--radius", type=int, default=2, help="dot radius in px (identical both sides)")
    p.add_argument("--thumb", type=int, default=110, help="reading-size thumbnail px")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--min-area", type=int, default=1)
    p.add_argument("--out", default="control_v4_mix/spacing_visibility")
    args = p.parse_args()

    _, wvs_tgt_d, _, gbn_tgt_d, _, _ = resolve_dirs(args)
    wvs_tgt = _stem_map(wvs_tgt_d, VALID_EXT); gbn_tgt = _stem_map(gbn_tgt_d, VALID_EXT)
    stems = sorted(set(wvs_tgt) & set(gbn_tgt))
    if not stems:
        print("ERROR: no matched stems."); return 2
    chosen = select_stems(stems, wvs_tgt, gbn_tgt, args.n, args.select, args.min_area, args.k)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib required: {exc}"); return 2
    os.makedirs(args.out, exist_ok=True)

    # Full-size montage: rows = icons, cols = [WVS full, GBN full, WVS thumb, GBN thumb].
    nrow = len(chosen)
    fig, axes = plt.subplots(nrow, 4, figsize=(14, 3.4 * nrow), dpi=110, squeeze=False)
    for i, stem in enumerate(chosen):
        pw = extract_centroids(wvs_tgt[stem], min_area=args.min_area)
        pg = extract_centroids(gbn_tgt[stem], min_area=args.min_area)
        bw = sr.compute_spacing_bundle(pw, k=args.k)["norm_nn_cv"]
        bg = sr.compute_spacing_bundle(pg, k=args.k)["norm_nn_cv"]
        full_w = render_dots(pw, args.size, args.radius)
        full_g = render_dots(pg, args.size, args.radius)
        thumb_r = max(1, int(round(args.radius * args.thumb / args.size)))
        th_w = render_dots(pw, args.thumb, thumb_r)
        th_g = render_dots(pg, args.thumb, thumb_r)
        panels = [(full_w, f"WVS  cv={bw:.3f}"), (full_g, f"GBN  cv={bg:.3f}"),
                  (th_w, f"WVS thumb {args.thumb}px"), (th_g, f"GBN thumb {args.thumb}px")]
        for j, (arr, title) in enumerate(panels):
            axc = axes[i][j]
            axc.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axc.set_title(f"{stem[:22]}\n{title}", fontsize=8)
            axc.axis("off")
    fig.suptitle("Spacing-regularity VISIBILITY gate -- identical dot style; only arrangement "
                 "differs.\nIf you cannot tell WVS from GBN here, the axis fails M0 (invisible), "
                 "as anisotropy did.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = os.path.join(args.out, f"visibility_{args.select}.png")
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    print(f"[write] {out_png}")
    print("Open it and judge: is the WVS-vs-GBN difference legible -- especially at thumbnail "
          "size (how a reader actually sees a stipple)? That is the M0 gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
