"""Build the per-oracle OT offsets the descriptor trainer needs -- in parallel, once, up front.

`train_control.ensure_offsets_dir` does this lazily at launch, but single-threaded: one exact
`ot.emd` network-simplex solve on a 1024x1024 cost matrix per (image, oracle). At 7 oracles x 10k
icons that is hours of wall clock before the first gradient step. This does the same work with a
process pool and then the trainer finds everything cached and starts immediately.

Two shortcuts, both safe:

  REUSE.  `icons-50_512_{GBN,WVS}/processed_offsets` already hold offsets for exactly these targets
          -- `build_oracles_dataset.py --stage link` HARDLINKED those PNGs, so they are the same
          inodes and the offsets are bit-identical to what we would recompute. They are hardlinked
          in rather than recomputed. NOTE: they were derived from PNG centroids, so once GBN/WVS are
          re-run with --export_npy those offsets are stale w.r.t. the exact coordinates -- rebuild
          with --no-reuse --force at that point.
  SKIP.   An existing `<stem>.npy` in the target offsets dir is left alone unless --force.

Point source matches the trainer: the exact `<stem>.npy` beside each target when present, else
centroid detection on the PNG. Short sets are repaired by DUPLICATION, identically to
`train_control._fit_points_to_n`, because OT requires exactly G*G points and a duplicate is the only
repair that downstream code can detect and undo.

    python control_v4_mix_metrics/prepare_offsets.py --workers 32
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import point_io as PIO  # noqa: E402

DEFAULT_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_Oracles"
DEFAULT_TRAIN_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training"
DEFAULT_ORACLES = "gbn,wvs,bnot,fs,ordered,white,jitgrid"
REUSE_FROM = {"gbn": "icons-50_512_GBN", "wvs": "icons-50_512_WVS", "bnot": "icons-50_512_BNOT"}
DEFAULT_N = 1024


def fit_to_n(points, n_points, seed=42):
    """Exactly n_points: subsample when long, DUPLICATE when short (never uniform random).

    Mirrors train_control._fit_points_to_n. A duplicate has nearest-neighbour distance 0, so
    `descriptor_fields.drop_exact_duplicates` can identify and remove it; an invented point cannot
    be undone.
    """
    rng = np.random.RandomState(seed)
    if len(points) > n_points:
        return points[rng.choice(len(points), n_points, replace=False)]
    if len(points) < n_points:
        if len(points) == 0:
            raise ValueError("empty point set")
        idx = rng.choice(len(points), n_points - len(points), replace=True)
        return np.vstack([points, points[idx]])
    return points


def load_points(target_dir, stem, n_points, min_points):
    npy = os.path.join(target_dir, stem + ".npy")
    if os.path.exists(npy):
        pts = PIO.load_points(npy)
        if len(pts) >= min_points:
            return fit_to_n(pts, n_points), "npy"
    png = os.path.join(target_dir, stem + ".png")
    if os.path.exists(png):
        pts = PIO.extract_centroids(png, n_points=None)
        if len(pts) >= min_points:
            return fit_to_n(pts, n_points), "png"
    return None, "missing"


def _one(payload):
    stem, target_dir, out_dir, n_points, min_points = payload
    from data.Transforms import to_image_optimal_transport
    dst = os.path.join(out_dir, stem + ".npy")
    try:
        pts, src = load_points(target_dir, stem, n_points, min_points)
        if pts is None:
            return stem, "missing", None
        offsets = to_image_optimal_transport(pts)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, offsets)
        os.replace(tmp, dst)
        return stem, src, None
    except Exception as exc:
        return stem, "error", f"{type(exc).__name__}: {exc}"


def link_existing(src_dir, out_dir, stems):
    """Hardlink precomputed offsets. Same inodes as the targets, so bit-identical."""
    n = 0
    for stem in stems:
        s = os.path.join(src_dir, stem + ".npy")
        d = os.path.join(out_dir, stem + ".npy")
        if os.path.exists(d) or not os.path.exists(s):
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        try:
            os.link(s, d)
        except OSError:
            import shutil
            shutil.copy2(s, d)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    ap.add_argument("--oracles", default=DEFAULT_ORACLES)
    ap.add_argument("--source", default=None, help="default: <root>/source")
    ap.add_argument("--n-points", type=int, default=DEFAULT_N)
    ap.add_argument("--min-points", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True,
                    help="hardlink icons-50_512_*/processed_offsets where available")
    ap.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    source = args.source or os.path.join(args.root, "source")
    oracles = [m.strip() for m in args.oracles.split(",") if m.strip()]
    stems = sorted(os.path.splitext(v)[0] for v in PIO.stem_map(source))
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"no source images under {source}")
    print(f"root    : {args.root}\nicons   : {len(stems)}  N={args.n_points}\n"
          f"oracles : {oracles}")

    grand = time.time()
    for m in oracles:
        target_dir = os.path.join(args.root, "target_" + m)
        out_dir = os.path.join(args.root, "offsets_" + m)
        if not os.path.isdir(target_dir):
            print(f"\n[{m}] SKIP -- no {target_dir}")
            continue
        os.makedirs(out_dir, exist_ok=True)

        linked = 0
        if args.reuse and not args.force and m in REUSE_FROM:
            src_dir = os.path.join(args.train_root, REUSE_FROM[m], "processed_offsets")
            if os.path.isdir(src_dir):
                linked = link_existing(src_dir, out_dir, stems)

        todo = [s for s in stems
                if args.force or not os.path.exists(os.path.join(out_dir, s + ".npy"))]
        print(f"\n[{m}] linked {linked}, to compute {len(todo)}", flush=True)
        if not todo:
            continue

        tasks = [(s, target_dir, out_dir, args.n_points, args.min_points) for s in todo]
        t0, done, srcs, errs = time.time(), 0, defaultdict(int), []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(_one, t) for t in tasks]):
                stem, src, err = fut.result()
                done += 1
                srcs[src] += 1
                if err:
                    errs.append((stem, err))
                if done % 500 == 0 or done == len(tasks):
                    el = time.time() - t0
                    print(f"  {done}/{len(tasks)}  {el:.0f}s ({done / max(el, 1e-9):.1f}/s)",
                          flush=True)
        print(f"  sources: {dict(srcs)}")
        if errs:
            print(f"  ERRORS {len(errs)} (first 5):")
            for s, e in errs[:5]:
                print(f"    {s}: {e}")

    print(f"\nall oracles done in {time.time() - grand:.0f}s")
    print("The trainer's ensure_offsets_dir will now find these cached and start immediately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
