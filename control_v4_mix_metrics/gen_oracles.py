"""Generate the cheap "extreme" oracles over the icon set at matched N (plan milestone M0/M1.1).

`icons-50_512_{GBN,WVS,BNOT}` already exist as rendered target PNGs. This adds Floyd-Steinberg,
ordered/Bayer dither, white noise and a jittered grid, computed from the SHARED source images so
every oracle sees identical rho and identical N -- "same rho, same budget, same everything except
the oracle".

Output is point sets (`.npy`, float32, (N, 2) in [0,1]) rather than rendered PNGs. Rendering and
then re-extracting centroids would round-trip the points through a dot rasteriser that merges
overlapping dots in dark regions, which corrupts exactly the local spacing statistics the axis
rests on. PNGs are still written when `--render` is passed, but only for the M0 visibility check
and for the round-trip control -- never as the measurement path.

Run on the cluster (the icon data is there and this box is SSHFS-mounted).

    python control_v4_mix_metrics/gen_oracles.py --out <dir> --workers 16

Resumable: existing `.npy` files are skipped unless `--force`.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import oracles as O
import point_io as PIO

DEFAULT_SOURCE = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN/source"
DEFAULT_N = 1024  # = control_v4 GRID_SIZE ** 2


def _one(args):
    stem, src_path, out_dir, names, n_points, seed, render, render_size = args
    try:
        gray = PIO.load_gray01(src_path)
        rho = O.rho_from_gray(gray)
        made = []
        for name in names:
            dst = os.path.join(out_dir, name, "points", stem + ".npy")
            if os.path.exists(dst):
                continue
            pts = O.ORACLES[name](rho, n_points, seed=seed)
            PIO.save_points(dst, pts)
            made.append(name)
            if render:
                img = O.render_dots_png(pts, size=render_size)
                rp = os.path.join(out_dir, name, "target", stem + ".png")
                os.makedirs(os.path.dirname(rp), exist_ok=True)
                img.save(rp)
        return stem, made, None
    except Exception as exc:  # keep one bad icon from killing a 10k-image run
        return stem, [], f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="shared source image dir (rho); all oracles read the same one")
    ap.add_argument("--out", required=True, help="output root; one subdir per oracle")
    ap.add_argument("--oracles", default="fs,ordered,white,jitgrid",
                    help="comma-separated subset of " + ",".join(O.ORACLES))
    ap.add_argument("--n-points", type=int, default=DEFAULT_N)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N icons")
    ap.add_argument("--stems-file", default=None,
                    help="restrict to these stems (one per line); skips the os.walk")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true",
                    help="also write dot PNGs (visibility check / round-trip control only)")
    ap.add_argument("--render-size", type=int, default=512)
    ap.add_argument("--force", action="store_true", help="regenerate even if the .npy exists")
    args = ap.parse_args()

    names = [n.strip() for n in args.oracles.split(",") if n.strip()]
    bad = [n for n in names if n not in O.ORACLES]
    if bad:
        ap.error(f"unknown oracle(s): {bad}; available: {list(O.ORACLES)}")

    if args.stems_file:
        want = PIO.read_stems_file(args.stems_file)
        src = PIO.stem_map_for(args.source, want)
        missing = [w for w in want if w not in src]
        if missing:
            print(f'  WARNING: {len(missing)} stems not found under source, e.g. {missing[:3]}')
    else:
        src = PIO.stem_map(args.source)
    if not src:
        ap.error(f"no source images under {args.source}")
    stems = sorted(src)
    if args.limit:
        stems = stems[: args.limit]
    print(f"source: {args.source}\noracles: {names}\nicons: {len(stems)}  N={args.n_points}")

    if args.force:
        for name in names:
            print(f"  --force: ignoring existing {name} outputs")

    tasks = []
    for stem in stems:
        if args.force:
            for name in names:
                p = os.path.join(args.out, name, "points", stem + ".npy")
                if os.path.exists(p):
                    os.remove(p)
        tasks.append((stem, src[stem], args.out, names, args.n_points, args.seed,
                      args.render, args.render_size))

    t0 = time.time()
    n_done = 0
    n_made = 0
    errors = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, t) for t in tasks]
            for fut in as_completed(futs):
                stem, made, err = fut.result()
                n_done += 1
                n_made += len(made)
                if err:
                    errors.append((stem, err))
                if n_done % 250 == 0 or n_done == len(tasks):
                    el = time.time() - t0
                    print(f"  {n_done}/{len(tasks)} icons  {n_made} sets  "
                          f"{el:.0f}s  ({n_done / max(el, 1e-9):.1f} icon/s)", flush=True)
    else:
        for t in tasks:
            stem, made, err = _one(t)
            n_done += 1
            n_made += len(made)
            if err:
                errors.append((stem, err))

    print(f"\ndone: {n_done} icons, {n_made} point sets written, {time.time() - t0:.0f}s")
    if errors:
        print(f"ERRORS on {len(errors)} icons (first 10):")
        for stem, err in errors[:10]:
            print(f"  {stem}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
