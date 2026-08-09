"""Precompute the descriptor FIELD stack for every (image, oracle) pair -- Milestone M1 step 2.

For each target this writes a `(K, G, G)` float32 array holding the M0 conditioning descriptors
(`descriptor_fields.CONDITIONING_KEYS`) at the offsets-grid resolution, plus a `valid` mask channel:

    Icons-50_1024_Oracles/
        descriptors_<method>/Icons-50/<class>/<stem>.npy      (K+1, G, G) float32
        DESCRIPTOR_STATS.json                                  normalisation, written by --stage stats

Why precompute rather than compute in the Dataset
--------------------------------------------------
A field stack costs ~0.14 s of numpy/scipy per (image, oracle). At 7 oracles x 10k icons that is
~2.7 h of pure CPU per epoch if done on the fly -- it would dominate the data loader and starve the
GPU. Precomputed, it is ~10 min once at 16 workers and then a single mmap-able read per sample.

Why RAW values are stored and normalisation lives in a side-car
---------------------------------------------------------------
The plan asks that each descriptor be normalised to roughly [0,1] across the dataset so a requested
value is interpretable. That normalisation is a *decision* (which percentiles, computed over which
oracles), and baking it into the arrays would mean recomputing 70k files to change it. So the fields
are stored raw and `DESCRIPTOR_STATS.json` holds per-descriptor `lo`/`hi`; the Dataset applies
`(x - lo) / (hi - lo)` at load time. Changing the normalisation is then a stats re-run, not a
re-precompute.

Percentiles, not min/max: `edge_align` and `pcf_peak` both have long right tails (a single
near-degenerate window can sit orders of magnitude above the bulk), and min/max normalisation would
compress every real difference into a sliver of the range.

Points come from `<target>/<stem>.npy` when present and fall back to centroid detection on the PNG,
matching `train_control.extract_points_from_target`. The fallback is reported per oracle, because a
silent all-PNG run would look identical to success while quietly measuring quantised positions.

Usage (cluster-side):

    python control_v4_mix_metrics/precompute_descriptors.py --stage all --workers 16
    python control_v4_mix_metrics/precompute_descriptors.py --stage stats      # re-normalise only
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import descriptor_fields as DF
import point_io as PIO

DEFAULT_TRAIN_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training"
DEFAULT_DATASET = "Icons-50_1024_Oracles"
DEFAULT_METHODS = "gbn,wvs,bnot,fs,ordered,white,jitgrid"
DEFAULT_N = 1024
STATS_NAME = "DESCRIPTOR_STATS.json"

# Must match prepare_offsets.py. Descriptors measure the set AFTER duplicates are removed
# (descriptor_fields.drop_exact_duplicates), so they see exactly the kept points while the
# offsets carry the padding -- the two agree without sharing an RNG.
DROP_WHITE_POINTS = True
WHITE_THRESHOLD = 255

KEYS = list(DF.CONDITIONING_KEYS)          # ("nn_cv","cap_cv","edge_align","aniso","pcf_peak")


def load_points_for(target_dir, stem, n_points, min_points):
    """Exact coordinates when available, else centroid detection. Returns (pts, source)."""
    npy = os.path.join(target_dir, stem + ".npy")
    if os.path.exists(npy):
        pts = PIO.load_points(npy)
        if len(pts) >= min_points:
            return pts, "npy"
    png = os.path.join(target_dir, stem + ".png")
    if os.path.exists(png):
        pts = PIO.extract_centroids(png, n_points=n_points)
        if len(pts) >= min_points:
            return pts, "png"
    return None, "missing"


def _one(payload):
    (stem, src_path, out_root, methods, G, window, n_points, min_points, force,
     drop_white, white_thr) = payload
    try:
        gray = None
        grad = None
        used = {}
        for m in methods:
            dst = os.path.join(out_root, f"descriptors_{m}", stem + ".npy")
            if os.path.exists(dst) and not force:
                continue
            pts, src = load_points_for(os.path.join(out_root, f"target_{m}"), stem,
                                       n_points, min_points)
            used[m] = src
            if pts is None:
                continue                      # e.g. an icon its generator never produced
            if gray is None:                  # shared across oracles for this icon
                gray = PIO.load_gray01(src_path)
                grad = DF.gradient_magnitude(gray)
            if drop_white:
                pts, _ = PIO.drop_white_area_points(pts, gray, threshold=white_thr)
            fields, _diag = DF.descriptor_fields(pts, gray, G=G, window=window, grad=grad)
            stack = np.stack([fields[k] for k in KEYS] + [fields["valid"].astype(np.float64)])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, stack.astype(np.float32))
            os.replace(tmp, dst)
        return stem, used, None
    except Exception as exc:
        return stem, {}, f"{type(exc).__name__}: {exc}"


def stage_compute(args, stems, src_map, out_root):
    tasks = [(s, src_map[s], out_root, args.methods, args.grid, args.window,
              args.n_points, args.min_points, args.force,
              args.drop_white_points, args.white_threshold) for s in stems]
    t0, done, errors = time.time(), 0, []
    srcs = defaultdict(lambda: defaultdict(int))
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(_one, t) for t in tasks]):
                stem, used, err = fut.result()
                done += 1
                for m, u in used.items():
                    srcs[m][u] += 1
                if err:
                    errors.append((stem, err))
                if done % 500 == 0 or done == len(tasks):
                    el = time.time() - t0
                    print(f"  {done}/{len(tasks)} icons  {el:.0f}s "
                          f"({done / max(el, 1e-9):.1f} icon/s)", flush=True)
    else:
        for t in tasks:
            stem, used, err = _one(t)
            done += 1
            for m, u in used.items():
                srcs[m][u] += 1
            if err:
                errors.append((stem, err))

    print(f"\n  point source actually used per oracle:")
    for m in args.methods:
        d = dict(srcs.get(m, {}))
        note = ""
        if d.get("png"):
            note = "  <- fell back to PNG centroids (quantised); re-run that generator with npy"
        if d.get("missing"):
            note += f"  [{d['missing']} missing]"
        print(f"    {m:10s} {d}{note}")
    if errors:
        print(f"\n  ERRORS on {len(errors)} icons (first 10):")
        for s, e in errors[:10]:
            print(f"    {s}: {e}")
    return errors


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def stage_stats(args, stems, out_root):
    """Per-descriptor normalisation bounds, pooled over ALL oracles.

    Pooled on purpose: the point of the axis is that one model spans every oracle, so a descriptor
    value must mean the same thing regardless of which oracle produced it. Per-oracle normalisation
    would map each oracle's own range onto [0,1] and destroy exactly the between-oracle separation
    M0 established.
    """
    rng = np.random.RandomState(0)
    sample = list(stems)
    if args.stats_sample and len(sample) > args.stats_sample:
        sample = [sample[i] for i in rng.choice(len(sample), args.stats_sample, replace=False)]
    print(f"  pooling descriptor values over {len(sample)} icons x {len(args.methods)} oracles ...")

    pools = {k: [] for k in KEYS}
    n_read = 0
    for stem in sample:
        for m in args.methods:
            p = os.path.join(out_root, f"descriptors_{m}", stem + ".npy")
            if not os.path.exists(p):
                continue
            arr = np.load(p)
            valid = arr[-1] > 0.5
            if not valid.any():
                continue
            n_read += 1
            for i, k in enumerate(KEYS):
                v = arr[i][valid]
                v = v[np.isfinite(v)]
                if v.size:
                    pools[k].append(v.astype(np.float32))

    stats = {"keys": KEYS, "grid": args.grid, "window": args.window,
             "n_stacks_pooled": n_read, "lo_pct": args.lo_pct, "hi_pct": args.hi_pct,
             "descriptors": {}}
    print(f"\n  {'descriptor':12s} {'lo(p%.1f)' % args.lo_pct:>10s} {'hi(p%.1f)' % args.hi_pct:>10s} "
          f"{'median':>10s} {'min':>10s} {'max':>10s}")
    for k in KEYS:
        if not pools[k]:
            print(f"  {k:12s}  (no data)")
            continue
        v = np.concatenate(pools[k])
        lo = float(np.percentile(v, args.lo_pct))
        hi = float(np.percentile(v, args.hi_pct))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        stats["descriptors"][k] = {"lo": lo, "hi": hi,
                                   "median": float(np.median(v)),
                                   "min": float(v.min()), "max": float(v.max()),
                                   "n": int(v.size)}
        print(f"  {k:12s} {lo:10.4f} {hi:10.4f} {np.median(v):10.4f} "
              f"{v.min():10.4f} {v.max():10.4f}")

    path = os.path.join(out_root, STATS_NAME)
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\n  wrote {path}")
    print("  The Dataset applies (x - lo) / (hi - lo) and clips to [0, 1]; values outside the\n"
          "  percentile range are legitimate and simply saturate.")
    return 0


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out-root", default=None, help="override full dataset path")
    ap.add_argument("--source", default=None,
                    help="source images (default: <out-root>/source)")
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--stage", default="all", choices=("all", "compute", "stats"))
    ap.add_argument("--grid", type=int, default=32, help="field resolution G (= GRID_SIZE)")
    ap.add_argument("--window", type=int, default=5, help="window width in cells")
    ap.add_argument("--n-points", type=int, default=DEFAULT_N)
    ap.add_argument("--min-points", type=int, default=256)
    ap.add_argument("--stats-sample", type=int, default=1500,
                    help="icons sampled for the normalisation pass (0 = all)")
    ap.add_argument("--lo-pct", type=float, default=1.0)
    ap.add_argument("--hi-pct", type=float, default=99.0)
    ap.add_argument("--stems-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--drop-white-points", action=argparse.BooleanOptionalAction,
                    default=DROP_WHITE_POINTS,
                    help="drop points on background before measuring "
                         "(must match prepare_offsets.py)")
    ap.add_argument("--white-threshold", type=float, default=WHITE_THRESHOLD,
                    help="source pixel value 0-255; >= this counts as background")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(args.train_root, args.dataset)
    source = args.source or os.path.join(out_root, "source")
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not os.path.isdir(source):
        ap.error(f"source dir not found: {source} (run build_oracles_dataset.py --stage link)")

    want = PIO.read_stems_file(args.stems_file) if args.stems_file else None
    src_map = PIO.stem_map_for(source, want) if want else PIO.stem_map(source)
    stems = sorted(src_map)
    if args.limit:
        stems = stems[: args.limit]
        src_map = {s: src_map[s] for s in stems}
    if not stems:
        ap.error(f"no source images under {source}")

    print(f"dataset : {out_root}")
    print(f"icons   : {len(stems)}   oracles: {args.methods}")
    print(f"fields  : {len(KEYS)} descriptors {KEYS} + valid, at G={args.grid} window={args.window}")
    print(f"drop-white: {'ON (>= %g/255)' % args.white_threshold if args.drop_white_points else 'OFF'}"
          "   -- prepare_offsets.py MUST use the same setting")

    rc = 0
    if args.stage in ("all", "compute"):
        print("\n[compute] descriptor fields ...")
        if stage_compute(args, stems, src_map, out_root):
            rc = 1
    if args.stage in ("all", "stats"):
        print("\n[stats] normalisation bounds ...")
        rc = stage_stats(args, stems, out_root) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
