"""Build the generated oracles into Icons-50_1024_Oracles/target_<method>/ (M1 step 1).

GBN, WVS and BNOT come from their own repos' generators. This script produces the four cheap
"extreme" oracles that stretch descriptor space -- Floyd-Steinberg, ordered/Bayer dither, white
noise, jittered grid -- from the SAME source images, so every oracle sees identical rho and
identical N: "same rho, same budget, same everything except the oracle".

Layout produced (one folder per method, png and npy side by side, matching what the three stipplers
now write):

    Icons-50_1024_Oracles/
        source/                      <- optional, hardlinked from the shared source (--link-source)
        target_fs/Icons-50/<class>/<stem>.png
        target_fs/Icons-50/<class>/<stem>.npy
        target_ordered/...
        target_white/...
        target_jitgrid/...
        target_gbn/, target_wvs/, target_bnot/   <- optional, --link-existing (see below)
        N_REPORT.json                <- matched-N audit from --stage verify

`--export_png` / `--export_npy` mirror the stipplers' flags (both default on). The `.npy` holds the
generator's exact continuous coordinates, canonical `(N,2)` float64, x-then-y, `[0,1]`, y DOWN --
the same convention `train_control.extract_points_from_target` returns, so it is a drop-in
replacement for centroid detection.

Rendering is ONE PIXEL PER POINT, not an antialiased disc. Measured on
`Icons-50_1024_{GBN,WVS,BNOT}`, mean connected-component area is exactly 1.00 px, so that is the
dataset convention; and a 5 px disc was measured to lose 157 points/icon for white noise and 48 for
the jittered grid, because their minimum spacing (a ~4 px halftone raster, or arbitrarily tight
clumps) sits at or below the dot diameter and adjacent dots merge into one blob. Residual pixel
coincidences are nudged to the nearest free pixel rather than dropped, so the PNG holds exactly N.

Run on the cluster (the data is local there; this box is SSHFS-mounted).

    python control_v4_mix_metrics/build_oracles_dataset.py --workers 16
    python control_v4_mix_metrics/build_oracles_dataset.py --stage verify --workers 16

Resumable: a stem is skipped only when every output you actually asked for already exists.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import oracles as O
import point_io as PIO

DEFAULT_TRAIN_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training"
DEFAULT_OUT_NAME = "Icons-50_1024_Oracles"
DEFAULT_SOURCE_DS = "Icons-50_1024_GBN"          # shared source images (rho) live here
GEN_METHODS = ("fs", "ordered", "white", "jitgrid")
# Datasets whose targets can be hardlinked in once their own generators finish.
#
# These are the 1024-point runs that export BOTH .png and exact .npy. The older
# icons-50_512_* datasets are PNG-only, so linking those silently gives the dataset
# pixel-quantised targets and no exact coordinates -- which is what produced
# target_gbn/wvs/bnot with 10000 png and 0 npy.
#
# Source-image caveat: Icons-50_1024_WVS and Icons-50_1024_BNOT were solved against a
# source variant that differs from Icons-50_1024_GBN (== the Oracles source) by at most
# 7/255, mean 0.28/255, over ~25% of pixels -- a resampling difference, not different
# artwork. Their points therefore follow a rho ~0.1% away from the one the model is
# conditioned on. Far below the oracle separation M0 measured (|d| > 4), but it is a
# cross-oracle asymmetry, so it belongs in any writeup of the descriptor comparison.
LINK_DATASETS = {"gbn": "Icons-50_1024_GBN", "wvs": "Icons-50_1024_WVS", "bnot": "Icons-50_1024_BNOT"}
DEFAULT_N = 1024                                 # = control_v4 GRID_SIZE ** 2


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _gen_one(payload):
    stem, src_path, out_root, methods, n_points, seed, size, export_png, export_npy, force = payload
    try:
        gray = PIO.load_gray01(src_path)
        rho = O.rho_from_gray(gray)
        made = []
        for m in methods:
            png = os.path.join(out_root, f"target_{m}", stem + ".png")
            npy = os.path.join(out_root, f"target_{m}", stem + ".npy")
            ready = ((os.path.exists(png) or not export_png)
                     and (os.path.exists(npy) or not export_npy))
            if ready and not force:
                continue
            pts = O.ORACLES[m](rho, n_points, seed=seed)
            if len(pts) != n_points:
                return stem, [], f"{m}: produced {len(pts)} points, expected {n_points}"
            if export_npy:
                PIO.save_points(npy, pts)
            if export_png:
                img, _collisions = O.render_points_pixels(pts, size=size)
                os.makedirs(os.path.dirname(png), exist_ok=True)
                img.save(png)
            made.append(m)
        return stem, made, None
    except Exception as exc:          # one bad icon must not kill a 10k run
        return stem, [], f"{type(exc).__name__}: {exc}"


def stage_gen(args, stems, src_map, out_root):
    tasks = [(s, src_map[s], out_root, args.methods, args.n_points, args.seed,
              args.render_size, args.export_png, args.export_npy, args.force) for s in stems]
    t0, done, made, errors = time.time(), 0, 0, []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(_gen_one, t) for t in tasks]):
                stem, mk, err = fut.result()
                done += 1
                made += len(mk)
                if err:
                    errors.append((stem, err))
                if done % 500 == 0 or done == len(tasks):
                    el = time.time() - t0
                    print(f"  {done}/{len(tasks)} icons  {made} sets  {el:.0f}s "
                          f"({done / max(el, 1e-9):.1f} icon/s)", flush=True)
    else:
        for t in tasks:
            stem, mk, err = _gen_one(t)
            done += 1
            made += len(mk)
            if err:
                errors.append((stem, err))
    print(f"  generated {made} point sets over {done} icons in {time.time() - t0:.0f}s")
    if errors:
        print(f"  ERRORS on {len(errors)} icons (first 10):")
        for s, e in errors[:10]:
            print(f"    {s}: {e}")
    return errors


# ---------------------------------------------------------------------------
# link (optional): pull the three stippler datasets in beside the generated ones
# ---------------------------------------------------------------------------

def _place(src, dst, use_copy, force=False):
    if os.path.exists(dst):
        if not force:
            return "skip"
        os.remove(dst)                  # replace a stale link/copy from an earlier run
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not use_copy:
        try:
            os.link(src, dst)               # instant, no extra space; data is read-only
            return "link"
        except OSError:
            pass                            # cross-device or unsupported -> fall through
    shutil.copy2(src, dst)
    return "copy"


def _index_by_stem(src_dir, exts):
    """stem -> {ext: path}, from ONE os.walk.

    The obvious loop -- probe os.path.exists(stem + ext) for every stem and every extension --
    costs len(stems) * len(exts) stat calls: ~60k for the source directory alone, on a network
    filesystem, before a single file is placed. A walk lists the same information in one pass.
    """
    out = defaultdict(dict)
    exts = {e.lower() for e in exts}
    for root, _, files in os.walk(src_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in exts:
                continue
            rel = os.path.relpath(os.path.join(root, f), src_dir)
            out[os.path.splitext(rel)[0].replace("\\", "/")][ext] = os.path.join(root, f)
    return out


def stage_link(args, stems, train_root, out_root):
    """Place source/ and the stippler targets into the dataset.

    Hardlinks by default (instant, no extra space, and the data is read-only); --copy writes real
    independent files instead.

    Deliberately places whatever EXISTS, per extension. A dataset that currently has only `.png`
    (its generator has not been re-run with --export_npy yet) places its PNGs and reports the
    missing NPYs; it is not an error. Re-run later with --force to pick the NPYs up.
    """
    jobs = []
    if args.link_source:
        jobs.append(("source", os.path.join(train_root, DEFAULT_SOURCE_DS, "source"), None))
    for m, ds in args.link_existing:
        jobs.append((f"target_{m}", os.path.join(train_root, ds, "target"), ds))

    mode = "copying" if args.copy else "hardlinking"
    for name, src_dir, ds in jobs:
        if not os.path.isdir(src_dir):
            print(f"  SKIP {name}: not found ({src_dir})")
            continue
        is_target = name.startswith("target_")
        exts = (".png", ".npy") if is_target else tuple(PIO.VALID_EXT)

        print(f"  {name:16s} {mode} <- {src_dir}", flush=True)
        print(f"  {'':16s} indexing source tree ...", flush=True)
        index = _index_by_stem(src_dir, exts)
        print(f"  {'':16s} {len(index)} stems found; placing {len(stems)} ...", flush=True)

        counts = defaultdict(int)
        found = defaultdict(int)
        t0 = time.time()
        for i, stem in enumerate(stems, 1):
            for ext, sp in index.get(stem, {}).items():
                found[ext] += 1
                counts[_place(sp, os.path.join(out_root, name, stem + ext),
                              args.copy, force=args.force)] += 1
            if i % 1000 == 0 or i == len(stems):
                el = time.time() - t0
                rate = i / max(el, 1e-9)
                eta = (len(stems) - i) / max(rate, 1e-9)
                print(f"  {'':16s} {i}/{len(stems)}  {el:.0f}s  ({rate:.0f} stem/s"
                      f", eta {eta:.0f}s)  {dict(counts)}", flush=True)
        src_note = f"  <- {ds}" if ds else ""
        print(f"  {name:16s} {dict(counts)}{src_note}")
        if is_target:
            n = len(stems)
            print(f"                   png {found['.png']}/{n}   npy {found['.npy']}/{n}")
            if found[".png"] < n:
                print(f"                   {n - found['.png']} PNG(s) missing -- that generator "
                      f"has not finished, or those icons failed.")
            if found[".npy"] == 0:
                print("                   no NPYs yet (generator not yet re-run with "
                      "--export_npy). Re-run this stage with --force once it has.")
            elif found[".npy"] < n:
                print(f"                   {n - found['.npy']} NPY(s) missing -- generator still "
                      f"running.")
            if counts.get("skip"):
                print(f"                   {counts['skip']} already present and left alone; pass "
                      f"--force to replace them.")


# ---------------------------------------------------------------------------
# verify: matched N
# ---------------------------------------------------------------------------

def _verify_one(payload):
    stem, out_root, methods, min_area = payload
    out = {}
    for m in methods:
        rec = {}
        png = os.path.join(out_root, f"target_{m}", stem + ".png")
        npy = os.path.join(out_root, f"target_{m}", stem + ".npy")
        if os.path.exists(png):
            try:
                # No n_points cap: we want the TRUE detectable count, not a subsample of it.
                rec["png_n"] = int(len(PIO.extract_centroids(png, n_points=None,
                                                             min_area=min_area)))
                rec["blob_area"] = _mean_blob_area(png)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
        if os.path.exists(npy):
            try:
                rec["npy_n"] = int(len(PIO.load_points(npy)))
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
        if rec:
            out[m] = rec
    return stem, out


def _mean_blob_area(png):
    """Mean connected-component area in px. Diagnostic for whether the generated renders use a dot
    size comparable to the pre-existing stippler targets (which measure exactly 1.00)."""
    from PIL import Image
    from scipy import ndimage
    img = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
    binary = ((255 - img) > 127).astype(np.uint8)
    lab, n = ndimage.label(binary)
    return float(binary.sum() / n) if n else 0.0


def stage_verify(args, stems, out_root, methods):
    tasks = [(s, out_root, methods, args.min_area) for s in stems]
    agg = defaultdict(lambda: {"png": [], "npy": [], "area": [], "errors": 0})
    t0, done, results = time.time(), 0, []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(_verify_one, t) for t in tasks]):
                results.append(fut.result())
                done += 1
                if done % 1000 == 0 or done == len(tasks):
                    print(f"  verified {done}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    else:
        results = [_verify_one(t) for t in tasks]

    for _stem, rec in results:
        for m, r in rec.items():
            if "error" in r:
                agg[m]["errors"] += 1
            if "png_n" in r:
                agg[m]["png"].append(r["png_n"])
                agg[m]["area"].append(r["blob_area"])
            if "npy_n" in r:
                agg[m]["npy"].append(r["npy_n"])

    N = args.n_points
    report = {"n_points_target": N, "n_icons": len(stems), "methods": {}}
    print("\n" + "=" * 96)
    print(f"MATCHED-N AUDIT   target N = {N}   {len(stems)} icons")
    print("=" * 96)
    print(f"{'method':10s} | {'npy n==N':>9} {'npy min':>8} | {'png med':>8} {'png ==N':>8} "
          f"{'png <N':>8} {'dot px':>7} | errors")
    ok = True
    for m in methods:
        d = agg[m]
        if not d["png"] and not d["npy"]:
            print(f"{m:10s} |  (nothing found)")
            continue
        npy = np.array(d["npy"]) if d["npy"] else None
        png = np.array(d["png"]) if d["png"] else None
        e = {"n_icons": max(len(d["npy"]), len(d["png"])), "errors": d["errors"]}
        s_npy = f"{100.0 * (npy == N).mean():8.1f}% {npy.min():8d}" if npy is not None else \
                f"{'-':>9} {'-':>8}"
        if npy is not None:
            e.update(npy_all_N=bool((npy == N).all()), npy_min=int(npy.min()))
            if not (npy == N).all():
                ok = False
        if png is not None:
            e.update(png_median=int(np.median(png)), pct_png_exactly_N=float((png == N).mean() * 100),
                     pct_png_below_N=float((png < N).mean() * 100),
                     mean_blob_area_px=float(np.mean(d["area"])))
            s_png = (f"{int(np.median(png)):8d} {100.0 * (png == N).mean():7.1f}% "
                     f"{100.0 * (png < N).mean():7.1f}% {np.mean(d['area']):7.2f}")
        else:
            s_png = f"{'-':>8} {'-':>8} {'-':>8} {'-':>7}"
        print(f"{m:10s} | {s_npy} | {s_png} | {d['errors']}")
        report["methods"][m] = e

    print("""
READING THIS TABLE
  'npy n==N' is the real matched-N guarantee: enforced at generation, cannot drift. Below 100% means
  the generator is wrong, not the storage.
  'png' is what survives rasterisation, and it is EXPECTED to fall short for the clumpy oracles.
  The PNG always holds exactly N black pixels (coincidences are nudged to a free neighbour), but
  centroid detection counts CONNECTED COMPONENTS, and two points that land in orthogonally adjacent
  pixels form one blob. That is adjacency, not coincidence, and it is a real property of the point
  process: white noise clumps, so it merges often (~2%), while spacing-optimised oracles almost
  never do (GBN measured 1023/1024, WVS 1024/1024). Nothing to fix -- it is why the .npy exists.
  Read target_<m>/<stem>.npy wherever the count matters; the PNG is for visibility and for tools
  that expect an image.
  'dot px' is the mean blob area, and should sit at ~1.00 to match Icons-50_1024_{GBN,WVS,BNOT}.""")
    with open(os.path.join(out_root, "N_REPORT.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(out_root, 'N_REPORT.json')}")
    return 0 if ok else 3


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT,
                    help="parent of the icons-50_* datasets (default: %(default)s)")
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                    help="dataset dir under --train-root (default: %(default)s)")
    ap.add_argument("--out-root", default=None, help="override the full output path")
    ap.add_argument("--source", default=None,
                    help=f"shared source dir (default: <train-root>/{DEFAULT_SOURCE_DS}/source)")
    ap.add_argument("--stage", default="gen", choices=("gen", "link", "verify", "all"))
    ap.add_argument("--methods", default=",".join(GEN_METHODS))
    ap.add_argument("--n-points", type=int, default=DEFAULT_N)
    ap.add_argument("--render-size", type=int, default=512)
    ap.add_argument("--export_png", "--export-png", dest="export_png",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Write the rasterised target .png")
    ap.add_argument("--export_npy", "--export-npy", dest="export_npy",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Write exact continuous coordinates as target .npy")
    ap.add_argument("--link-source", action=argparse.BooleanOptionalAction, default=True,
                    help="link stage: hardlink source/ into the dataset")
    ap.add_argument("--link-existing", default="",
                    help="link stage: comma-separated subset of gbn,wvs,bnot to hardlink in. "
                         "Each may be given as NAME or NAME=DATASET_DIR to link from a different "
                         "folder, e.g. 'gbn=icons-50_1024_GBN,wvs,bnot'. Links whatever exists "
                         "(png-only is fine); re-run with --force to pick up NPYs later.")
    ap.add_argument("--stems-file", default=None, help="restrict to these stems (one per line)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=1, help="verify: drop blobs below this area")
    ap.add_argument("--copy", action="store_true", help="link stage: real copies, not hardlinks")
    ap.add_argument("--force", action="store_true", help="regenerate existing outputs")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(args.train_root, args.out_name)
    source = args.source or os.path.join(args.train_root, DEFAULT_SOURCE_DS, "source")
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in args.methods if m not in O.ORACLES]
    if bad:
        ap.error(f"unknown method(s) {bad}; available: {list(O.ORACLES)}")
    pairs = []
    for entry in (e.strip() for e in args.link_existing.split(",")):
        if not entry:
            continue
        name, _, ds = entry.partition("=")
        name = name.strip()
        if name not in LINK_DATASETS:
            ap.error(f"unknown dataset to link '{name}'; available: {list(LINK_DATASETS)}")
        pairs.append((name, ds.strip() or LINK_DATASETS[name]))
    args.link_existing = pairs
    if not args.export_png and not args.export_npy:
        ap.error("nothing to write: both --no-export_png and --no-export_npy given")
    if not os.path.isdir(source):
        ap.error(f"source dir not found: {source}")

    want = PIO.read_stems_file(args.stems_file) if args.stems_file else None
    src_map = PIO.stem_map_for(source, want) if want else PIO.stem_map(source)
    stems = sorted(src_map)
    if args.limit:
        stems = stems[: args.limit]
        src_map = {s: src_map[s] for s in stems}
    if not stems:
        ap.error(f"no source images found under {source}")

    os.makedirs(out_root, exist_ok=True)
    print(f"out-root  : {out_root}")
    print(f"source    : {source}")
    print(f"icons     : {len(stems)}   N = {args.n_points}   render {args.render_size}px "
          f"(1 px dots)")
    print(f"methods   : {args.methods}   png={args.export_png} npy={args.export_npy}")

    rc = 0
    if args.stage in ("gen", "all"):
        print("\n[gen] generating oracles ...")
        if stage_gen(args, stems, src_map, out_root):
            rc = 1
    if args.stage in ("link", "all"):
        print("\n[link] hardlinking source / finished stipplers ...")
        stage_link(args, stems, args.train_root, out_root)
    if args.stage in ("verify", "all"):
        print("\n[verify] auditing matched N ...")
        rc = stage_verify(args, stems, out_root,
                          args.methods + [m for m, _ in args.link_existing]) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
