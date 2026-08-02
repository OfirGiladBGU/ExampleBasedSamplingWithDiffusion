"""Gate 0 (attempt 2) runner -- does WVS vs GBN separate on spacing/clumping REGULARITY,
after controlling for the density gradient?

Same discipline as attempt 1: direction-agnostic verdict, and a built-in confound cross-check.
Here the confound is DENSITY (not tone): the verdict rests on the density-NORMALIZED primary
descriptor (norm_nn_cv), and the runner explicitly compares it against the global (confounded)
nn_cv so a density-only difference cannot masquerade as a regularity axis.

Run on the cluster. Defaults mirror control_v4/train_control.py.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spacing_regularity as sr  # noqa: E402
from edge_alignment import extract_centroids, load_gray01  # noqa: E402
from gate0_run import (  # noqa: E402
    DEFAULT_GBN_ROOT, DEFAULT_WVS_ROOT, VALID_EXT, _num, _stem_map, resolve_dirs,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw):
        return it

REPORT_KEYS = (
    "norm_nn_cv", "clumped_pct_local", "voronoi_area_cv_norm", "local_gap_ratio_mean",
    "nn_cv_global", "clumped_pct_global",
)
PRIMARY_KEY = sr.PRIMARY_KEY  # norm_nn_cv


def _regularity_scores(values, key):
    """Orient so a LARGER score always means MORE regular."""
    v = np.asarray(values, dtype=np.float64)
    return v if key in sr.REGULAR_HIGHER_KEYS else -v


def cohens_d(wvs, gbn, key):
    a = _regularity_scores(wvs, key); b = _regularity_scores(gbn, key)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled < sr.EPS:
        return float("nan")
    return float((np.mean(b) - np.mean(a)) / pooled)


def paired_agreement(wvs, gbn, key):
    a = _regularity_scores(wvs, key); b = _regularity_scores(gbn, key)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan"), 0
    return float(np.mean(b[m] > a[m])), int(m.sum())


def clean_separation(wvs, gbn, key, hi_q=0.90, lo_q=0.10):
    a = _regularity_scores(wvs, key); b = _regularity_scores(gbn, key)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return False
    gbn_above = np.quantile(b, lo_q) > np.quantile(a, hi_q)
    wvs_above = np.quantile(a, lo_q) > np.quantile(b, hi_q)
    return bool(gbn_above or wvs_above)


def summarize(per_key):
    desc = {}
    for k in REPORT_KEYS:
        wvs = np.array(per_key[k]["wvs"], dtype=np.float64)
        gbn = np.array(per_key[k]["gbn"], dtype=np.float64)
        agree, n_pair = paired_agreement(wvs, gbn, k)
        d_val = cohens_d(wvs, gbn, k)
        consistency = max(agree, 1.0 - agree) if agree == agree else float("nan")
        more_regular = "n/a" if d_val != d_val else ("GBN" if d_val > 0 else "WVS")
        desc[k] = {
            "wvs_mean": float(np.nanmean(wvs)) if wvs.size else None,
            "gbn_mean": float(np.nanmean(gbn)) if gbn.size else None,
            "cohens_d": d_val, "paired_consistency": consistency,
            "more_regular": more_regular, "n_pairs": n_pair,
            "clean_separation": clean_separation(wvs, gbn, k),
        }
    return desc


def print_table(desc, n_icons):
    print("=" * 92)
    print(f"SPACING REGULARITY   |   icons: {n_icons}")
    print("=" * 92)
    hdr = (f"{'descriptor':<22}{'WVS mean':>12}{'GBN mean':>12}{'cohens_d':>11}"
           f"{'consist':>9}{'regular':>9}{'clean':>7}")
    print(hdr); print("-" * len(hdr))
    for k in REPORT_KEYS:
        d = desc[k]
        tag = "" if k in sr.DENSITY_CONTROLLED_KEYS else "  (confounded)"
        print(f"{k:<22}{_num(d['wvs_mean']):>12.4f}{_num(d['gbn_mean']):>12.4f}"
              f"{_num(d['cohens_d']):>11.3f}{_num(d['paired_consistency']):>9.2f}"
              f"{str(d['more_regular']):>9}{('Y' if d['clean_separation'] else '.'):>7}{tag}")
    print("  consist = direction-agnostic paired consistency (1.00 = every icon agrees)")
    print("  regular = which oracle is MORE regular; (confounded) rows mix in the density gradient")


def build_args():
    p = argparse.ArgumentParser(description="Gate 0 attempt 2: spacing-regularity separation.")
    p.add_argument("--wvs-root", default=DEFAULT_WVS_ROOT)
    p.add_argument("--gbn-root", default=DEFAULT_GBN_ROOT)
    p.add_argument("--wvs-source", default=""); p.add_argument("--wvs-target", default="")
    p.add_argument("--gbn-source", default=""); p.add_argument("--gbn-target", default="")
    p.add_argument("--wvs-offsets", default=""); p.add_argument("--gbn-offsets", default="")
    p.add_argument("--k", type=int, default=8, help="neighbours for local density estimate")
    p.add_argument("--min-area", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-overlays", type=int, default=4)
    p.add_argument("--out", default="control_v4_mix/spacing_gate0_outputs")
    return p


def make_plots(per_key, stems, src_map, wvs_tgt, gbn_tgt, args, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plots] matplotlib unavailable ({exc}); skipping."); return
    pdir = os.path.join(out_dir, "plots"); os.makedirs(pdir, exist_ok=True)

    for key in (PRIMARY_KEY, "nn_cv_global"):
        wvs = np.array(per_key[key]["wvs"], float); gbn = np.array(per_key[key]["gbn"], float)
        m = np.isfinite(wvs) & np.isfinite(gbn)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
        ax[0].scatter(wvs[m], gbn[m], s=12, alpha=0.5, color="#3b6ea5")
        if m.any():
            lo = float(np.min([wvs[m].min(), gbn[m].min()]))
            hi = float(np.max([wvs[m].max(), gbn[m].max()]))
            ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax[0].set_xlabel(f"WVS {key}"); ax[0].set_ylabel(f"GBN {key}")
        ax[0].set_title(f"paired per-icon: {key}"); ax[0].legend()
        ax[1].hist(wvs[m], bins=30, alpha=0.6, label="WVS", color="#c05555")
        ax[1].hist(gbn[m], bins=30, alpha=0.6, label="GBN", color="#3b6ea5")
        ax[1].set_xlabel(key); ax[1].set_ylabel("icons"); ax[1].legend()
        ax[1].set_title(f"distribution: {key}")
        fig.tight_layout(); fig.savefig(os.path.join(pdir, f"{key}.png"), bbox_inches="tight")
        plt.close(fig)

    n_show = min(args.n_overlays, len(stems))
    if n_show > 0:
        fig, axes = plt.subplots(n_show, 2, figsize=(8, 4 * n_show), dpi=120, squeeze=False)
        for i, stem in enumerate(list(stems)[:n_show]):
            for j, (title, tgt) in enumerate((("WVS", wvs_tgt), ("GBN", gbn_tgt))):
                pts = extract_centroids(tgt[stem], min_area=args.min_area)
                axc = axes[i][j]
                if len(pts) >= args.k + 2:
                    d1, rho, s, _ = sr.local_density_and_spacing(pts, k=args.k)
                    u = d1 / np.maximum(s, sr.EPS)
                    sc = axc.scatter(pts[:, 0], 1 - pts[:, 1], c=u, s=4, cmap="viridis",
                                     vmin=0, vmax=1.5)
                    fig.colorbar(sc, ax=axc, fraction=0.046)
                axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.set_aspect("equal")
                axc.set_title(f"{stem[:24]} [{title}]  (color = norm NN dist u)", fontsize=8)
                axc.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(pdir, "overlays.png"), bbox_inches="tight")
        plt.close(fig)
    print(f"[plots] wrote plots to {pdir}")


def main():
    args = build_args().parse_args()
    wvs_src, wvs_tgt_d, gbn_src, gbn_tgt_d, _, _ = resolve_dirs(args)
    print("Scanning data directories...", flush=True)
    print(f"  wvs target: {wvs_tgt_d}\n  gbn target: {gbn_tgt_d}", flush=True)
    src_map = dict(_stem_map(gbn_src, VALID_EXT)); src_map.update(_stem_map(wvs_src, VALID_EXT))
    wvs_tgt = _stem_map(wvs_tgt_d, VALID_EXT); gbn_tgt = _stem_map(gbn_tgt_d, VALID_EXT)
    stems = sorted(set(wvs_tgt) & set(gbn_tgt))
    if not stems:
        print("ERROR: no matched stems across WVS/GBN targets."); return 2
    if args.limit:
        stems = stems[: args.limit]
    print(f"Matched icons: {len(stems)}  (k={args.k})", flush=True)

    per_key = {k: {"wvs": [], "gbn": []} for k in REPORT_KEYS}
    rows = []
    for stem in tqdm(stems, desc="spacing", unit="icon", dynamic_ncols=True):
        pw = extract_centroids(wvs_tgt[stem], min_area=args.min_area)
        pg = extract_centroids(gbn_tgt[stem], min_area=args.min_area)
        bw = sr.compute_spacing_bundle(pw, k=args.k)
        bg = sr.compute_spacing_bundle(pg, k=args.k)
        for oracle, b in (("WVS", bw), ("GBN", bg)):
            row = {"stem": stem, "oracle": oracle}; row.update(b); rows.append(row)
        for k in REPORT_KEYS:
            per_key[k]["wvs"].append(bw.get(k, np.nan))
            per_key[k]["gbn"].append(bg.get(k, np.nan))

    desc = summarize(per_key)
    print_table(desc, len(stems))
    make_plots(per_key, stems, src_map, wvs_tgt, gbn_tgt, args, args.out)

    os.makedirs(args.out, exist_ok=True)
    fields = ["stem", "oracle", "n_points"] + list(REPORT_KEYS)
    with open(os.path.join(args.out, "per_icon.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"n_icons": len(stems), "descriptors": desc}, fh, indent=2)
    print(f"[write] {os.path.join(args.out, 'per_icon.csv')}")
    print(f"[write] {os.path.join(args.out, 'summary.json')}")

    # ---- Verdict on the density-CONTROLLED primary + confound cross-check ----
    p = desc[PRIMARY_KEY]
    cons = _num(p["paired_consistency"]); d = _num(p["cohens_d"])
    passed = (cons >= 0.90) and (abs(d) >= 0.80)
    g = desc["nn_cv_global"]
    gd = _num(g["cohens_d"]); gcons = _num(g["paired_consistency"])

    print("\n" + "#" * 92)
    print(f"GATE 0 (attempt 2) VERDICT -- density-CONTROLLED primary '{PRIMARY_KEY}':")
    print(f"  consistency = {cons:.2f}   cohen's d = {d:.2f}   "
          f"more regular = {p['more_regular']}   clean = {p['clean_separation']}")
    if passed:
        print("  --> PASS: WVS and GBN separate on spacing regularity AFTER controlling for")
        print("      density. This is a non-decomposable point-process property -> viable axis.")
        print("      Next: confirm it is VISIBLE in dot-only renders (the anisotropy-M0 trap),")
        print("      then proceed to Phase 2. Re-name the axis to the measured direction.")
    else:
        print("  --> BLOCKER: no clean, consistent density-controlled regularity separation.")
        print("      The spacing-regularity axis does not carry a contribution either.")
    print(f"\n  [density-confound cross-check] global nn_cv: d={gd:.2f} consistency={gcons:.2f}")
    if abs(gd) >= 0.80 and abs(d) < 0.50:
        print("      WARNING: the GLOBAL (confounded) measure separates but the density-")
        print("      controlled one does NOT. The apparent difference is a DENSITY effect")
        print("      (rho-decomposable), exactly like the tone confound in attempt 1. Do not")
        print("      build the axis on it.")
    elif abs(d) >= 0.80 and abs(gd) < abs(d):
        print("      (Density-controlled separation is as strong or stronger than the global")
        print("       one -> the signal is genuinely about regularity, not density.)")
    print("#" * 92)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
