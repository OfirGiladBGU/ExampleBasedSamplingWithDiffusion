"""Count the points in every rendered target and list any that fall short of N.

Answers "did this already happen and I missed it?" for datasets written before the count assertion
existed -- e.g. the WVS Lloyd loop silently shrinking 1024 -> 1023 when voronoi.py's region filter
discards a cell whose circumcenter falls outside `bbox +- 0.1`.

Two columns, because they answer different questions:

  black_px    Number of black pixels. The targets are rendered as 1x1 dots, so this IS the point
              count -- the only way it can undercount is two points rounding into the same pixel,
              which cannot happen at these spacings (1024 points over 512px is ~16px apart).
              THIS is the column that tells you whether the generator emitted a short set.

  components  Connected components, i.e. what `extract_points_from_target` actually recovers.
              Two points landing in orthogonally adjacent pixels merge into one blob, so this can
              sit below black_px without anything being wrong with the generator. It matters
              because it is what the training path sees when reading PNGs rather than .npy.

A row where black_px == N but components < N is pixel adjacency (harmless, expected for clumpy
oracles). A row where black_px < N is a genuinely short point set.

    python control_v4_mix_metrics/audit_point_counts.py \
        --dir /groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_WVS/target
"""

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def _one(path_and_root):
    path, root = path_and_root
    from PIL import Image
    from scipy import ndimage
    try:
        img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        black = (255 - img) > 127
        n_black = int(black.sum())
        _lab, n_comp = ndimage.label(black)
        stem = os.path.splitext(os.path.relpath(path, root))[0].replace("\\", "/")
        return stem, n_black, int(n_comp), None
    except Exception as exc:
        return os.path.relpath(path, root), -1, -1, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="target directory to scan (recursive)")
    ap.add_argument("--expect", type=int, default=1024, help="expected point count N")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None,
                    help="write short stems here (default: <dir>/../POINT_COUNT_AUDIT.json)")
    ap.add_argument("--list", type=int, default=40, help="how many short stems to print")
    args = ap.parse_args()

    files = []
    for root, _, names in os.walk(args.dir):
        for n in names:
            if n.lower().endswith(".png"):
                files.append((os.path.join(root, n), args.dir))
    if not files:
        raise SystemExit(f"no PNGs under {args.dir}")
    print(f"scanning {len(files)} PNGs under {args.dir}  (expect N={args.expect})")

    rows, errors, t0 = [], [], time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(_one, f) for f in files]), 1):
                stem, nb, nc, err = fut.result()
                (errors if err else rows).append((stem, nb, nc, err))
                if i % 2000 == 0 or i == len(files):
                    print(f"  {i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    else:
        for f in files:
            stem, nb, nc, err = _one(f)
            (errors if err else rows).append((stem, nb, nc, err))

    black = np.array([r[1] for r in rows])
    comps = np.array([r[2] for r in rows])
    N = args.expect
    short = sorted([r for r in rows if r[1] < N], key=lambda r: r[1])
    adj_only = [r for r in rows if r[1] == N and r[2] < N]

    print("\n" + "=" * 78)
    print(f"{'':12s} {'min':>7} {'p01':>7} {'median':>7} {'max':>7} {'==N':>8} {'<N':>8}")
    for name, arr in (("black_px", black), ("components", comps)):
        print(f"{name:12s} {arr.min():>7d} {int(np.percentile(arr, 1)):>7d} "
              f"{int(np.median(arr)):>7d} {arr.max():>7d} "
              f"{100.0 * (arr == N).mean():7.2f}% {100.0 * (arr < N).mean():7.2f}%")

    print("\n" + "-" * 78)
    print(f"GENUINELY SHORT POINT SETS (black_px < {N}): {len(short)} / {len(rows)}")
    if short:
        for stem, nb, nc, _ in short[: args.list]:
            print(f"    {nb:5d} pts  (comp {nc:5d})   {stem}")
        if len(short) > args.list:
            print(f"    ... and {len(short) - args.list} more")
        print("\n  Distribution of the shortfall:")
        for d, c in sorted(Counter(N - r[1] for r in short).items()):
            print(f"    missing {d:3d}: {c} icon(s)")
    else:
        print("    none -- every target carries the full point count.")

    print(f"\nPIXEL-ADJACENCY ONLY (black_px == {N}, components < {N}): {len(adj_only)}")
    print("    Harmless: the points are all there, but touching dots merge under centroid\n"
          "    detection. Affects the PNG read path only; the .npy is unaffected.")
    if errors:
        print(f"\nERRORS: {len(errors)}")
        for stem, _, _, err in errors[:10]:
            print(f"    {stem}: {err}")

    out = args.out or os.path.join(os.path.dirname(os.path.normpath(args.dir)),
                                   "POINT_COUNT_AUDIT.json")
    with open(out, "w") as fh:
        json.dump({"dir": args.dir, "expect": N, "n_files": len(rows),
                   "n_short": len(short), "n_adjacency_only": len(adj_only),
                   "short": [{"stem": s, "black_px": nb, "components": nc}
                             for s, nb, nc, _ in short]}, fh, indent=2)
    print(f"\nwrote {out}")
    print("Re-generate just the short ones by passing their stems to the generator's --stems-file "
          "(or delete those targets and re-run; generation is resumable).")
    return 1 if short else 0


if __name__ == "__main__":
    raise SystemExit(main())
