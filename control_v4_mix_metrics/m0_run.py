"""Milestone 0 -- multi-oracle descriptor separation (BLOCKING, no training).

The plan's cheapest kill-shot: if the oracles do not occupy distinct regions of descriptor space,
conditioning on these descriptors cannot control anything and no amount of training fixes it.

Three checks, per the plan:

  SEPARATION     -- do the oracles occupy distinct, NON-OVERLAPPING regions, not merely different
                    means? Reported per descriptor as a pairwise matrix of paired Cohen's d, plus
                    a paired-agreement rate (does oracle A beat oracle B on the SAME icon, not
                    just on average) and an overlap check on the 10th/90th percentiles.
  NON-REDUNDANCY -- drop any descriptor correlating > ~0.9 with another, or with rho. Two
                    descriptors at 0.95 are one axis wearing two hats.
  VISIBILITY     -- render dots-only, unlabeled, full size. This is the check the anisotropy axis
                    failed: metric-real but invisible. It cannot be automated; the script writes
                    the panels and the verdict block tells you to go look at them.

Two controls run alongside, because both have a specific way of faking a pass:

  QUANTISATION   -- the dither oracles place points on a lattice (see `oracles.py`). A lattice
                    depresses NN-distance CV on its own. So GBN is requantised onto the same
                    lattice and re-measured; if that alone reproduces most of the gap to the
                    dither oracles, the separation is an artefact of the raster, not of the oracle.
  ROUND-TRIP     -- GBN/WVS/BNOT points come from centroid detection on rendered PNGs, the new
                    oracles come from exact coordinates. If that difference in measurement channel
                    moves a descriptor as much as the oracle identity does, the comparison is
                    invalid. Measured by rendering GBN's own points and re-extracting them.

Run on the cluster:

    python control_v4_mix_metrics/m0_run.py --gen-root <gen_oracles --out dir> \
        --out control_v4_mix_metrics/m0_outputs
"""

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import descriptor_fields as DF
import oracles as O
import point_io as PIO

DEFAULT_TRAIN_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training"
DISK_DIRS = {"gbn": "icons-50_512_GBN", "wvs": "icons-50_512_WVS", "bnot": "icons-50_512_BNOT"}
DEFAULT_N = 1024

REDUNDANCY_THRESHOLD = 0.90
D_THRESHOLD = 0.80
AGREEMENT_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Per-icon work
# ---------------------------------------------------------------------------

def _load_points(oracle, stem, ctx):
    if oracle in ctx["disk_targets"]:
        pts = PIO.extract_centroids(ctx["disk_targets"][oracle][stem], n_points=ctx["n_points"])
    else:
        pts = PIO.load_points(os.path.join(ctx["gen_root"], oracle, "points", stem + ".npy"))
    return pts


def _one_icon(payload):
    stem, ctx = payload
    try:
        gray = PIO.load_gray01(ctx["src"][stem])
        grad = DF.gradient_magnitude(gray)
        rho = O.rho_from_gray(gray)
        raster = O.halftone_raster(rho, ctx["n_points"])
        rows = []
        for oracle in ctx["oracles"]:
            pts = _load_points(oracle, stem, ctx)
            if len(pts) < ctx["min_points"]:
                rows.append({"stem": stem, "oracle": oracle, "n_points": int(len(pts)),
                             "skipped": True})
                continue
            fields, diag = DF.descriptor_fields(pts, gray, G=ctx["G"], window=ctx["window"],
                                                grad=grad)
            row = {"stem": stem, "oracle": oracle, "n_points": int(len(pts)), "skipped": False}
            row.update(DF.pool_fields(fields))
            row.update(diag)
            row["rho_mean"] = float(rho.mean())
            # COMMON-RASTER view: every oracle snapped to the SAME lattice the dither oracles are
            # already bound to. Continuous-vs-raster comparisons are only fair here, because in
            # the raw view the lattice itself moves nn_cv more than some oracle pairs differ.
            fq, _ = DF.descriptor_fields(PIO.quantise(pts, raster), gray, G=ctx["G"],
                                         window=ctx["window"], grad=grad)
            for k, v in DF.pool_fields(fq).items():
                row[k + "__q"] = v
            row["raster"] = int(raster)
            rows.append(row)

        # Controls, computed on this icon's GBN points only (cheap, one extra pair).
        ctrl = {"stem": stem}
        if ctx["controls"] and "gbn" in ctx["oracles"]:
            g = _load_points("gbn", stem, ctx)
            if len(g) >= ctx["min_points"]:
                ctrl["raster"] = int(raster)
                img = O.render_dots_png(g, size=ctx["render_size"])
                tmp = os.path.join(ctx["out"], "_rt_tmp", stem.replace("/", "__") + ".png")
                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                img.save(tmp)
                rt = PIO.extract_centroids(tmp, n_points=ctx["n_points"])
                os.remove(tmp)
                if len(rt) >= ctx["min_points"]:
                    frt, _ = DF.descriptor_fields(rt, gray, G=ctx["G"], window=ctx["window"],
                                                  grad=grad)
                    ctrl["gbn_roundtrip"] = DF.pool_fields(frt)
                    ctrl["roundtrip_n"] = int(len(rt))
        return rows, ctrl, None
    except Exception as exc:
        return [], {}, f"{stem}: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_cohens_d(a, b):
    """Paired Cohen's d on matched icons. Paired, not independent-samples, because every oracle
    sees the SAME rho on the same icon -- the icon-to-icon variance is shared nuisance and pooling
    it in would mask a consistent per-icon difference."""
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 1e-12 else float("inf" if d.mean() > 0 else "-inf")


def paired_agreement(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(max((d > 0).mean(), (d < 0).mean()))


def non_overlapping(a, b, q=0.10):
    a = np.asarray(a)[np.isfinite(a)]
    b = np.asarray(b)[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return False
    if a.mean() > b.mean():
        a, b = b, a
    return bool(np.quantile(a, 1 - q) < np.quantile(b, q))


def by_oracle(rows, key, oracles, stems):
    """Aligned per-descriptor vectors, one column per oracle, rows = icons present for ALL."""
    idx = {o: {} for o in oracles}
    for r in rows:
        if not r.get("skipped") and r["oracle"] in idx:
            idx[r["oracle"]][r["stem"]] = r.get(key, float("nan"))
    common = [s for s in stems if all(s in idx[o] for o in oracles)]
    return {o: np.array([idx[o][s] for s in common]) for o in oracles}, common


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def separation_report(rows, oracles, stems, keys):
    out = {}
    for key in keys:
        cols, common = by_oracle(rows, key, oracles, stems)
        pairs = {}
        for a, b in itertools.combinations(oracles, 2):
            pairs[f"{a}|{b}"] = {
                "d": paired_cohens_d(cols[a], cols[b]),
                "agreement": paired_agreement(cols[a], cols[b]),
                "non_overlapping": non_overlapping(cols[a], cols[b]),
            }
        out[key] = {
            "n_icons": len(common),
            "means": {o: float(np.nanmean(cols[o])) for o in oracles},
            "stds": {o: float(np.nanstd(cols[o])) for o in oracles},
            "pairs": pairs,
            "n_separated_pairs": sum(
                1 for v in pairs.values()
                if abs(v["d"]) >= D_THRESHOLD and v["agreement"] >= AGREEMENT_THRESHOLD),
            "n_pairs": len(pairs),
        }
    return out


def redundancy_report(rows, oracles, stems, keys):
    """Correlation among descriptors, and of each descriptor against rho.

    Pooled across oracles ON PURPOSE: two descriptors that merely re-label the same axis will
    track each other across the whole population, which is the thing worth catching. Correlation
    with rho_mean is the rho-decomposability smell test -- a descriptor that is mostly a readout
    of tone is not describing arrangement.
    """
    valid = [r for r in rows if not r.get("skipped")]
    mat = {k: np.array([r.get(k, np.nan) for r in valid], dtype=float) for k in keys}
    mat["rho_mean"] = np.array([r.get("rho_mean", np.nan) for r in valid], dtype=float)
    names = list(keys) + ["rho_mean"]
    corr = {}
    for a, b in itertools.combinations(names, 2):
        x, y = mat[a], mat[b]
        m = np.isfinite(x) & np.isfinite(y)
        corr[f"{a}|{b}"] = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 5 else float("nan")
    flagged = [k for k, v in corr.items() if np.isfinite(v) and abs(v) > REDUNDANCY_THRESHOLD]
    return {"corr": corr, "flagged": flagged}


def controls_report(ctrls, rows, stems, keys, oracles):
    """How much of any separation could be a measurement artefact rather than an oracle property.

    Deltas are in the SAME units as the between-oracle gaps, so they read directly against the
    separation table. Two artefacts, each with its own way of faking a pass:

    QUANTISATION -- snapping a continuous oracle onto the dither raster. If this shift exceeds the
    gap between a continuous and a raster-bound oracle, that pair's "separation" is the lattice.
    ROUND-TRIP   -- the disk oracles are measured through centroid detection on rendered PNGs
    while the generated ones are exact coordinates. If the channel moves a descriptor as much as
    the oracle does, the comparison is invalid regardless of the effect size.
    """
    out = {}
    ref = "gbn" if "gbn" in oracles else oracles[0]

    # Quantisation: read straight off the paired raw / __q columns already computed per icon.
    deltas = {}
    for key in keys:
        raw, q = [], []
        for r in rows:
            if r.get("oracle") == ref and not r.get("skipped") and (key + "__q") in r:
                raw.append(r.get(key, np.nan))
                q.append(r.get(key + "__q", np.nan))
        raw, q = np.array(raw, float), np.array(q, float)
        m = np.isfinite(raw) & np.isfinite(q)
        deltas[key] = {
            "mean_shift": float((q[m] - raw[m]).mean()) if m.sum() else float("nan"),
            "d": paired_cohens_d(q[m], raw[m]),
        }
    out[f"{ref}_quantised"] = {"n": int(sum(1 for r in rows if r.get("oracle") == ref)),
                               "deltas": deltas}

    have = {c["stem"]: c["gbn_roundtrip"] for c in ctrls if "gbn_roundtrip" in c}
    if have:
        deltas = {}
        for key in keys:
            b, q = [], []
            for r in rows:
                if r.get("oracle") == "gbn" and not r.get("skipped") and r["stem"] in have:
                    b.append(r.get(key, np.nan))
                    q.append(have[r["stem"]].get(key, np.nan))
            b, q = np.array(b, float), np.array(q, float)
            m = np.isfinite(b) & np.isfinite(q)
            deltas[key] = {
                "mean_shift": float((q[m] - b[m]).mean()) if m.sum() else float("nan"),
                "d": paired_cohens_d(q[m], b[m]),
            }
        out["gbn_roundtrip"] = {"n": len(have), "deltas": deltas}
    return out


def make_plots(rows, ctrls, oracles, stems, keys, out_dir, src, ctx, n_panels=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)

    # Distributions per descriptor.
    for key in keys:
        cols, _ = by_oracle(rows, key, oracles, stems)
        fig, ax = plt.subplots(figsize=(7, 4))
        for o in oracles:
            v = cols[o][np.isfinite(cols[o])]
            if v.size:
                ax.hist(v, bins=50, alpha=0.45, label=f"{o} ({v.mean():.3f})")
        ax.set_title(f"{key} by oracle")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, f"dist_{key}.png"), dpi=130)
        plt.close(fig)

    # VISIBILITY: dots-only, unlabeled, FULL SIZE. Thumbnails over-ink to solid black and are
    # useless for this judgement -- one row per icon, one column per oracle, no titles.
    vdir = os.path.join(out_dir, "visibility")
    os.makedirs(vdir, exist_ok=True)
    show = list(stems)[:n_panels]
    for stem in show:
        try:
            fig, axes = plt.subplots(1, len(oracles), figsize=(4.0 * len(oracles), 4.2))
            for ax, o in zip(np.atleast_1d(axes), oracles):
                pts = _load_points(o, stem, ctx)
                ax.scatter(pts[:, 0], 1.0 - pts[:, 1], s=3.0, c="black", edgecolors="none")
                ax.set_xlim(-0.01, 1.01)
                ax.set_ylim(-0.01, 1.01)
                ax.set_aspect("equal")
                ax.axis("off")
            fig.tight_layout(pad=0.2)
            fig.savefig(os.path.join(vdir, stem.replace("/", "__") + "_blind.png"), dpi=150)
            plt.close(fig)
            # Labelled key, written separately so the blind panel can be judged first.
            with open(os.path.join(vdir, "KEY.txt"), "a") as fh:
                fh.write(f"{stem}: " + " | ".join(oracles) + "\n")
        except Exception as exc:
            print(f"  visibility panel failed for {stem}: {exc}")


def print_and_verdict(sep, red, ctrl, oracles, keys, sep_q=None):
    print("\n" + "=" * 78)
    print("SEPARATION (paired Cohen's d / agreement / non-overlapping)")
    print("=" * 78)
    for key in keys:
        s = sep[key]
        print(f"\n{key}   [{s['n_separated_pairs']}/{s['n_pairs']} pairs separated]")
        print("  means: " + "  ".join(f"{o}={s['means'][o]:.4f}" for o in oracles))
        for pair, v in sorted(s["pairs"].items(), key=lambda kv: -abs(kv[1]["d"])):
            mark = "OK " if (abs(v["d"]) >= D_THRESHOLD
                             and v["agreement"] >= AGREEMENT_THRESHOLD) else "   "
            print(f"    {mark}{pair:22s} d={v['d']:+7.2f}  agree={v['agreement']:.2f}  "
                  f"clean={v['non_overlapping']}")

    print("\n" + "=" * 78)
    print("NON-REDUNDANCY (|r| > %.2f is a duplicate axis)" % REDUNDANCY_THRESHOLD)
    print("=" * 78)
    for pair, v in sorted(red["corr"].items(), key=lambda kv: -abs(kv[1] if np.isfinite(kv[1]) else 0)):
        flag = "  <-- REDUNDANT" if pair in red["flagged"] else ""
        print(f"  {pair:34s} r={v:+.3f}{flag}")

    print("\n" + "=" * 78)
    print("CONTROLS (artefact size, same units as the gaps above)")
    print("=" * 78)
    for name in ("gbn_quantised", "gbn_roundtrip"):
        if name not in ctrl:
            continue
        print(f"\n  {name}  (n={ctrl[name]['n']})")
        for key, v in ctrl[name]["deltas"].items():
            print(f"    {key:14s} shift={v['mean_shift']:+.4f}  d={v['d']:+.2f}")

    # ---- verdict -----------------------------------------------------------
    redundant = _redundant_keys(red, keys)
    usable = [k for k in keys if sep[k]["n_separated_pairs"] >= 1 and k not in redundant]
    print("\n" + "=" * 78)
    print("GATE M0 VERDICT")
    print("=" * 78)
    if redundant:
        print(f"  dropped as redundant: {sorted(redundant)}")
    total_pairs = sep[keys[0]]["n_pairs"]

    # A cross-family pair (continuous oracle vs raster-bound dither oracle) is judged on the
    # COMMON-RASTER view, where the lattice is shared and cancels. Same-family pairs are judged
    # raw, because quantising two continuous oracles throws away real information and would
    # under-report a genuine difference.
    covered, cross_lost = set(), []
    for pair in sep[keys[0]]["pairs"]:
        a, b = pair.split("|")
        cross = (a in O.RASTER_BOUND) != (b in O.RASTER_BOUND)
        table = sep_q if (cross and sep_q) else sep
        for k in usable:
            v = table[k]["pairs"][pair]
            if abs(v["d"]) >= D_THRESHOLD and v["agreement"] >= AGREEMENT_THRESHOLD:
                covered.add(pair)
                break
        else:
            if cross and sep_q:
                raw_ok = any(abs(sep[k]["pairs"][pair]["d"]) >= D_THRESHOLD
                             and sep[k]["pairs"][pair]["agreement"] >= AGREEMENT_THRESHOLD
                             for k in usable)
                if raw_ok:
                    cross_lost.append(pair)

    print(f"  usable (separating, non-redundant) descriptors: {usable or 'NONE'}")
    print(f"  oracle pairs separated by at least one descriptor: {len(covered)}/{total_pairs}")
    print(f"  (cross-family pairs judged on the common raster; same-family pairs judged raw)")
    if cross_lost:
        print(f"\n  ARTEFACT WARNING: {cross_lost} separate in the raw view but NOT once every")
        print("  oracle is on the same lattice. That separation was the dither raster, not the")
        print("  oracle. Do not count these pairs.")

    if not usable:
        print("\n  BLOCKER: no descriptor separates any oracle pair. Conditioning on these "
              "descriptors cannot control anything.\n  STOP and re-identify the descriptors "
              "before any training (plan: Milestone 0 blocker 1).")
        return 2
    if len(covered) < total_pairs:
        missing = sorted(set(sep[keys[0]]["pairs"]) - covered)
        print(f"\n  PARTIAL: these oracle pairs are NOT separated: {missing}")
        print("  Those oracles are interchangeable as far as the descriptors are concerned; "
              "either drop one of each pair or add a descriptor that distinguishes them.")
    print("\n  METRIC GATE PASSED for the descriptors listed above.")
    print("  NOT YET CLEARED: the visibility check is a HUMAN judgement and is not automated.")
    print("  Open plots/../visibility/*_blind.png (full size, dots only, unlabeled) and confirm")
    print("  the endpoints are obviously distinct -- especially GBN vs Floyd-Steinberg. The")
    print("  anisotropy axis passed every metric gate and died here. If the metrics separate but")
    print("  the renders do not, that is plan blocker 2: re-select oracles toward the extremes.")
    return 0


def _redundant_keys(red, keys):
    """Drop one member of any over-correlated pair, and anything tracking rho.

    The survivor is chosen by an explicit priority rather than by dict order: `nn_cv` is the
    measure that cleared the WVS<->GBN Gate 0, so if it correlates with a sibling it is the
    sibling that goes. Order-of-declaration tie-breaking would silently drop the primary and
    leave the axis resting on a descriptor with no prior validation.
    """
    priority = [DF.PRIMARY_KEY] + [k for k in keys if k != DF.PRIMARY_KEY]
    rank = {k: i for i, k in enumerate(priority)}
    drop = set()
    for pair in red["flagged"]:
        a, b = pair.split("|")
        if "rho_mean" in (a, b):
            drop.add(a if b == "rho_mean" else b)
            continue
        if a in drop or b in drop:
            continue
        drop.add(a if rank.get(a, 99) > rank.get(b, 99) else b)
    return drop


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-root", required=True, help="output root of gen_oracles.py")
    ap.add_argument("--out", default="control_v4_mix_metrics/m0_outputs")
    ap.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT,
                    help="parent of icons-50_512_{GBN,WVS,BNOT}")
    ap.add_argument("--source", default=None,
                    help="shared source dir (default: <train-root>/icons-50_512_GBN/source)")
    ap.add_argument("--oracles", default="gbn,wvs,bnot,fs,ordered,white,jitgrid")
    ap.add_argument("--n-points", type=int, default=DEFAULT_N)
    ap.add_argument("--min-points", type=int, default=256,
                    help="skip an icon/oracle below this count rather than padding it")
    ap.add_argument("--grid", type=int, default=32, help="descriptor field resolution G")
    ap.add_argument("--window", type=int, default=5, help="window width in cells (= spacings)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stems-file", default=None,
                    help="restrict to these stems (one per line); skips the os.walk")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-panels", type=int, default=8)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--render-size", type=int, default=512)
    args = ap.parse_args()

    oracles = [o.strip() for o in args.oracles.split(",") if o.strip()]
    os.makedirs(args.out, exist_ok=True)
    disk_roots = {k: os.path.join(args.train_root, v) for k, v in DISK_DIRS.items()}
    source = args.source or os.path.join(disk_roots["gbn"], "source")

    want = PIO.read_stems_file(args.stems_file) if args.stems_file else None
    src = PIO.stem_map_for(source, want) if want else PIO.stem_map(source)
    if not src:
        raise SystemExit(f"no source images under {source}")
    disk_targets = {}
    for o in oracles:
        if o in disk_roots:
            d = os.path.join(disk_roots[o], "target")
            if not os.path.isdir(d):
                raise SystemExit(f"missing target dir for oracle '{o}': {d}")
            disk_targets[o] = PIO.stem_map_for(d, want) if want else PIO.stem_map(d)

    # Stems present for EVERY oracle -- the paired statistics require it.
    stems = set(src)
    for o in oracles:
        if o in disk_targets:
            stems &= set(disk_targets[o])
        else:
            pd = os.path.join(args.gen_root, o, "points")
            if not os.path.isdir(pd):
                raise SystemExit(f"missing generated points for oracle '{o}': {pd}\n"
                                 f"run gen_oracles.py first")
            stems &= set(PIO.stem_map_for(pd, want, exts=(".npy",)) if want
                         else PIO.stem_map(pd, exts=(".npy",)))
    stems = sorted(stems)
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit("no stems common to all requested oracles")
    print(f"oracles: {oracles}\nmatched icons: {len(stems)}  G={args.grid} window={args.window}")

    ctx = {
        "src": src, "disk_targets": disk_targets, "gen_root": args.gen_root,
        "n_points": args.n_points, "min_points": args.min_points, "G": args.grid,
        "window": args.window, "oracles": oracles, "controls": not args.no_controls,
        "out": args.out, "render_size": args.render_size,
    }

    t0 = time.time()
    rows, ctrls, errors = [], [], []
    payloads = [(s, ctx) for s in stems]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one_icon, p) for p in payloads]
            for i, fut in enumerate(as_completed(futs), 1):
                r, c, err = fut.result()
                rows.extend(r)
                if len(c) > 1:
                    ctrls.append(c)
                if err:
                    errors.append(err)
                if i % 200 == 0 or i == len(futs):
                    print(f"  {i}/{len(futs)} icons  {time.time() - t0:.0f}s", flush=True)
    else:
        for i, p in enumerate(payloads, 1):
            r, c, err = _one_icon(p)
            rows.extend(r)
            if len(c) > 1:
                ctrls.append(c)
            if err:
                errors.append(err)

    if errors:
        print(f"\n{len(errors)} icon errors (first 5):")
        for e in errors[:5]:
            print("  " + e)

    keys = list(DF.FIELD_KEYS)
    sep = separation_report(rows, oracles, stems, keys)
    _sq = separation_report(rows, oracles, stems, [k + "__q" for k in keys])
    sep_q = {k: _sq[k + "__q"] for k in keys}   # re-key to the plain descriptor names
    red = redundancy_report(rows, oracles, stems, keys + list(DF.DIAG_KEYS))
    ctrl = controls_report(ctrls, rows, stems, keys, oracles)

    with open(os.path.join(args.out, "per_icon.json"), "w") as fh:
        json.dump(rows, fh)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"oracles": oracles, "n_icons": len(stems), "separation": sep,
                   "separation_common_raster": sep_q, "redundancy": red,
                   "controls": ctrl}, fh, indent=2)

    make_plots(rows, ctrls, oracles, stems, keys, args.out, src, ctx, n_panels=args.n_panels)
    rc = print_and_verdict(sep, red, ctrl, oracles, keys, sep_q=sep_q)
    print(f"\noutputs: {args.out}   ({time.time() - t0:.0f}s)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
