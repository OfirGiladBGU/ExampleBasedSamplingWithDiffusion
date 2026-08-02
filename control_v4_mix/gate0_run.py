"""Gate 0 runner -- compute the edge-alignment descriptor on the real WVS and GBN icon outputs
and decide the BLOCKING go/no-go verdict.

WVS and GBN share the SAME source images (same rho); only the target stipple PNGs differ, so
each icon yields a matched pair (P_WVS, P_GBN) under identical rho. For every matched stem we
extract each oracle point set, compute the full descriptor bundle, and aggregate separation
statistics. PASS => WVS and GBN separate cleanly and consistently (green-light Phase 2).
BLOCKER => stop and re-identify the visual property before training anything.

Data lives on the cluster; run this there. Defaults mirror control_v4/train_control.py globals.

Example
-------
python control_v4_mix/gate0_run.py --point-source both \
    --out control_v4_mix/gate0_outputs
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_alignment as ea  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional -- degrade to a plain iterator
    def tqdm(iterable, **kwargs):
        return iterable

# Defaults mirror train_control.py: WVS = icons-50_512_WVS, GBN = icons-50_512_GBN (shared source).
DEFAULT_WVS_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_WVS"
DEFAULT_GBN_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN"
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Descriptors reported in the summary, in display order.
REPORT_KEYS = (
    "s_raw", "observed_frac", "expected_frac", "grad_at_points",
    "dist_mean", "dist_median",
    "frac_within_1px", "frac_within_2px", "frac_within_4px", "frac_within_8px",
)
PRIMARY_KEY = "s_raw"


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------

def _stem_map(directory, exts):
    """stem -> full path for files with the given extensions (recursive)."""
    out = {}
    if not directory or not os.path.isdir(directory):
        return out
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                stem = os.path.splitext(os.path.relpath(os.path.join(root, f), directory))[0]
                out[stem] = os.path.join(root, f)
    return out


def resolve_dirs(args):
    wvs_source = args.wvs_source or os.path.join(args.wvs_root, "source")
    wvs_target = args.wvs_target or os.path.join(args.wvs_root, "target")
    gbn_source = args.gbn_source or os.path.join(args.gbn_root, "source")
    gbn_target = args.gbn_target or os.path.join(args.gbn_root, "target")
    wvs_off = args.wvs_offsets or os.path.join(args.wvs_root, "processed_offsets")
    gbn_off = args.gbn_offsets or os.path.join(args.gbn_root, "processed_offsets")
    return wvs_source, wvs_target, gbn_source, gbn_target, wvs_off, gbn_off


# -----------------------------------------------------------------------------
# Separation statistics
# -----------------------------------------------------------------------------

def _aligned_scores(values, key):
    """Orient so that a LARGER score always means 'more edge-aligned' (GBN-like)."""
    v = np.asarray(values, dtype=np.float64)
    return -v if key in ea.GBN_LOWER_KEYS else v


def cohens_d(wvs, gbn, key):
    """Signed effect size in the aligned direction (positive => GBN more aligned)."""
    a = _aligned_scores(wvs, key)
    b = _aligned_scores(gbn, key)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled < ea.EPS:
        return float("nan")
    return float((np.mean(b) - np.mean(a)) / pooled)


def paired_agreement(wvs, gbn, key):
    """Fraction of icons where GBN is more edge-aligned than WVS (paired, same stem)."""
    a = _aligned_scores(wvs, key)
    b = _aligned_scores(gbn, key)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan"), 0
    return float(np.mean(b[m] > a[m])), int(m.sum())


def clean_separation(wvs, gbn, key, hi_q=0.90, lo_q=0.10):
    """Near-non-overlapping check: 90th pct of WVS-aligned below 10th pct of GBN-aligned."""
    a = _aligned_scores(wvs, key); b = _aligned_scores(gbn, key)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return False
    # Direction-agnostic: either oracle may be the more edge-aligned one.
    gbn_above = np.quantile(b, lo_q) > np.quantile(a, hi_q)
    wvs_above = np.quantile(a, lo_q) > np.quantile(b, hi_q)
    return bool(gbn_above or wvs_above)


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------

def run_variant(stems, src_map, wvs_tgt, gbn_tgt, args, off_wvs=None, off_gbn=None,
                point_source="raw"):
    """Compute descriptor bundles for both oracles across all stems for one point-source mode."""
    rows = []
    per_key = {k: {"wvs": [], "gbn": []} for k in REPORT_KEYS}
    for stem in tqdm(stems, desc=f"gate0[{point_source}]", unit="icon", dynamic_ncols=True):
        gray = ea.load_gray01(src_map[stem])
        g = args.grid_size
        # The source image is SHARED by both oracles, so build the edge band / distance
        # transform ONCE per icon and reuse it for both point sets.
        fields = ea.precompute_fields(gray, method=args.edge, edge_quantile=args.edge_quantile)

        if point_source == "gridded":
            if off_wvs is None or stem not in off_wvs or off_gbn is None or stem not in off_gbn:
                continue
            pw = ea.points_from_offsets(np.load(off_wvs[stem]), g)
            pg = ea.points_from_offsets(np.load(off_gbn[stem]), g)
        else:
            pw = ea.extract_centroids(wvs_tgt[stem], min_area=args.min_area, max_points=args.max_points)
            pg = ea.extract_centroids(gbn_tgt[stem], min_area=args.min_area, max_points=args.max_points)

        bw = ea.compute_descriptor_bundle(gray, pw, method=args.edge,
                                          edge_quantile=args.edge_quantile, fields=fields)
        bg = ea.compute_descriptor_bundle(gray, pg, method=args.edge,
                                          edge_quantile=args.edge_quantile, fields=fields)

        for oracle, b in (("WVS", bw), ("GBN", bg)):
            row = {"stem": stem, "oracle": oracle, "point_source": point_source}
            row.update({k: b.get(k) for k in b})
            rows.append(row)
        for k in REPORT_KEYS:
            per_key[k]["wvs"].append(bw.get(k, np.nan))
            per_key[k]["gbn"].append(bg.get(k, np.nan))

    summary = {"point_source": point_source, "n_icons": len(rows) // 2, "descriptors": {}}
    for k in REPORT_KEYS:
        wvs = np.array(per_key[k]["wvs"], dtype=np.float64)
        gbn = np.array(per_key[k]["gbn"], dtype=np.float64)
        agree, n_pair = paired_agreement(wvs, gbn, k)
        d_val = cohens_d(wvs, gbn, k)
        # Consistency is direction-agnostic: 1.0 means every icon agrees on a direction,
        # 0.5 means the pairing is a coin flip. An agreement of 0.00 is therefore PERFECT
        # consistency (every icon says WVS is the more edge-aligned one), not "no signal".
        consistency = max(agree, 1.0 - agree) if agree == agree else float("nan")
        if d_val != d_val:
            more_aligned = "n/a"
        else:
            more_aligned = "GBN" if d_val > 0 else "WVS"
        summary["descriptors"][k] = {
            "wvs_mean": float(np.nanmean(wvs)) if wvs.size else None,
            "wvs_std": float(np.nanstd(wvs)) if wvs.size else None,
            "gbn_mean": float(np.nanmean(gbn)) if gbn.size else None,
            "gbn_std": float(np.nanstd(gbn)) if gbn.size else None,
            "cohens_d_aligned": d_val,
            "paired_agreement": agree,
            "paired_consistency": consistency,
            "more_aligned": more_aligned,
            "n_pairs": n_pair,
            "clean_separation": clean_separation(wvs, gbn, k),
        }

    # Normalization for the primary descriptor (WVS median -> 0, GBN median -> 1).
    sw = np.array(per_key[PRIMARY_KEY]["wvs"], dtype=np.float64)
    sg = np.array(per_key[PRIMARY_KEY]["gbn"], dtype=np.float64)
    s_wvs_ref = float(np.nanmedian(sw)) if sw.size else None
    s_gbn_ref = float(np.nanmedian(sg)) if sg.size else None
    summary["s_normalization"] = {
        "primary_key": PRIMARY_KEY, "wvs_ref_zero": s_wvs_ref, "gbn_ref_one": s_gbn_ref,
        "note": "s_norm = (s_raw - wvs_ref_zero) / (gbn_ref_one - wvs_ref_zero)",
    }
    return rows, summary, per_key


def _num(x):
    """None -> nan; otherwise float(x). Avoids the ``x or default`` trap where a legitimate
    0.0 (e.g. a paired agreement of 0.0) is falsy and would be silently replaced."""
    return float("nan") if x is None else float(x)


def verdict(summary):
    """Gate 0 asks 'do the two oracles separate cleanly and consistently?' -- NOT 'does GBN
    separate in the direction we guessed'. The gate is therefore direction-agnostic: it passes
    on a consistent, large-effect separation either way and reports WHICH oracle is more
    edge-aligned. A sign opposite to the hypothesis is a finding about the axis, not a gate
    failure -- but it does mean the axis must be re-named/re-interpreted before Phase 2.
    """
    d = summary["descriptors"].get(PRIMARY_KEY, {})
    agree = _num(d.get("paired_agreement"))
    coh = _num(d.get("cohens_d_aligned"))
    consistency = _num(d.get("paired_consistency"))
    passed = (consistency >= 0.90) and (abs(coh) >= 0.80)
    return passed, agree, coh, bool(d.get("clean_separation")), consistency, \
        d.get("more_aligned", "n/a")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def make_plots(per_key, rows, stems, src_map, wvs_tgt, gbn_tgt, args, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plots] matplotlib unavailable ({exc}); skipping plots.")
        return
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) paired scatter + histograms for the primary + one distance descriptor.
    for key in (PRIMARY_KEY, "dist_mean"):
        wvs = np.array(per_key[key]["wvs"], dtype=np.float64)
        gbn = np.array(per_key[key]["gbn"], dtype=np.float64)
        m = np.isfinite(wvs) & np.isfinite(gbn)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
        ax[0].scatter(wvs[m], gbn[m], s=14, alpha=0.7, color="#3b6ea5")
        lo = float(np.nanmin([wvs[m].min(), gbn[m].min()])) if m.any() else 0.0
        hi = float(np.nanmax([wvs[m].max(), gbn[m].max()])) if m.any() else 1.0
        ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax[0].set_xlabel(f"WVS {key}"); ax[0].set_ylabel(f"GBN {key}")
        ax[0].set_title(f"paired per-icon: {key}"); ax[0].legend()
        ax[1].hist(wvs[m], bins=24, alpha=0.6, label="WVS", color="#c05555")
        ax[1].hist(gbn[m], bins=24, alpha=0.6, label="GBN", color="#3b6ea5")
        ax[1].set_xlabel(key); ax[1].set_ylabel("icons"); ax[1].set_title(f"distribution: {key}")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{tag}_{key}.png"), bbox_inches="tight")
        plt.close(fig)

    # 2) overlay panels for a few icons (source + edge band + points), WVS vs GBN.
    n_show = min(args.n_overlays, len(stems))
    if n_show > 0 and args.point_source != "gridded":
        show = list(stems)[:n_show]
        fig, axes = plt.subplots(n_show, 2, figsize=(8, 4 * n_show), dpi=120,
                                 squeeze=False)
        for i, stem in enumerate(show):
            gray = ea.load_gray01(src_map[stem])
            band, _, _, _ = ea.edge_band_and_distance(gray, args.edge, args.edge_quantile)
            pw = ea.extract_centroids(wvs_tgt[stem], min_area=args.min_area)
            pg = ea.extract_centroids(gbn_tgt[stem], min_area=args.min_area)
            h, w = gray.shape
            for j, (title, pts) in enumerate((("WVS", pw), ("GBN", pg))):
                axc = axes[i][j]
                axc.imshow(gray, cmap="gray", vmin=0, vmax=1)
                axc.imshow(np.ma.masked_where(~band, band), cmap="autumn", alpha=0.35)
                if len(pts):
                    axc.scatter(pts[:, 0] * w, pts[:, 1] * h, s=3, color="#1f77b4")
                axc.set_title(f"{stem} [{title}]", fontsize=8)
                axc.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{tag}_overlays.png"), bbox_inches="tight")
        plt.close(fig)
    print(f"[plots] wrote plots for '{tag}' to {plots_dir}")


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def print_summary(summary):
    print("=" * 84)
    print(f"POINT SOURCE: {summary['point_source']}   |   icons: {summary['n_icons']}")
    print("=" * 84)
    hdr = (f"{'descriptor':<16}{'WVS mean':>12}{'GBN mean':>12}"
           f"{'cohens_d':>11}{'consist':>9}{'aligned':>9}{'clean':>7}")
    print(hdr); print("-" * len(hdr))
    for k in REPORT_KEYS:
        d = summary["descriptors"][k]
        wm = d["wvs_mean"]; gm = d["gbn_mean"]
        print(f"{k:<16}{(wm if wm is not None else float('nan')):>12.4f}"
              f"{(gm if gm is not None else float('nan')):>12.4f}"
              f"{_num(d['cohens_d_aligned']):>11.3f}"
              f"{_num(d.get('paired_consistency')):>9.2f}"
              f"{str(d.get('more_aligned', 'n/a')):>9}"
              f"{('Y' if d['clean_separation'] else '.'):>7}")
    print("  consist = direction-agnostic paired consistency (1.00 = every icon agrees; "
          "0.50 = coin flip)")
    print("  aligned = which oracle is MORE edge-aligned on that descriptor")


def write_outputs(all_rows, summaries, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # CSV
    fieldnames = ["stem", "oracle", "point_source", "n_points", "s_raw", "observed_frac",
                  "expected_frac", "grad_at_points", "dist_mean", "dist_median",
                  "frac_within_1px", "frac_within_2px", "frac_within_4px", "frac_within_8px"]
    csv_path = os.path.join(out_dir, "per_icon.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\n[write] {csv_path}\n[write] {json_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Gate 0: WVS<->GBN edge-alignment separation.")
    p.add_argument("--wvs-root", default=DEFAULT_WVS_ROOT)
    p.add_argument("--gbn-root", default=DEFAULT_GBN_ROOT)
    p.add_argument("--wvs-source", default=""); p.add_argument("--wvs-target", default="")
    p.add_argument("--gbn-source", default=""); p.add_argument("--gbn-target", default="")
    p.add_argument("--wvs-offsets", default=""); p.add_argument("--gbn-offsets", default="")
    p.add_argument("--point-source", choices=("raw", "gridded", "both"), default="raw")
    p.add_argument("--edge", choices=("sobel", "canny"), default="sobel")
    p.add_argument("--edge-quantile", type=float, default=0.80)
    p.add_argument("--grid-size", type=int, default=32)
    p.add_argument("--min-area", type=int, default=1)
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap #icons (quick smoke test)")
    p.add_argument("--n-overlays", type=int, default=5)
    p.add_argument("--out", default="control_v4_mix/gate0_outputs")
    return p


def main():
    args = build_arg_parser().parse_args()
    wvs_src, wvs_tgt_d, gbn_src, gbn_tgt_d, wvs_off_d, gbn_off_d = resolve_dirs(args)

    print("Scanning data directories (this walks the dataset; on a network FS it can take "
          "a moment)...", flush=True)
    print(f"  wvs source: {wvs_src}\n  wvs target: {wvs_tgt_d}", flush=True)
    print(f"  gbn source: {gbn_src}\n  gbn target: {gbn_tgt_d}", flush=True)
    src_wvs = _stem_map(wvs_src, VALID_EXT)
    src_gbn = _stem_map(gbn_src, VALID_EXT)
    wvs_tgt = _stem_map(wvs_tgt_d, VALID_EXT)
    gbn_tgt = _stem_map(gbn_tgt_d, VALID_EXT)
    # Shared source: prefer WVS source, fall back to GBN source for any missing stem.
    src_map = dict(src_gbn); src_map.update(src_wvs)

    stems = sorted(set(wvs_tgt) & set(gbn_tgt) & set(src_map))
    if not stems:
        print("ERROR: no matched stems across WVS target, GBN target, and source.")
        print(f"  wvs_target={wvs_tgt_d}\n  gbn_target={gbn_tgt_d}\n  source(wvs)={wvs_src}")
        return 2
    if args.limit:
        stems = stems[: args.limit]
    print(f"Matched icons: {len(stems)}  (edge={args.edge}, q={args.edge_quantile}, "
          f"point-source={args.point_source})", flush=True)

    off_wvs = _stem_map(wvs_off_d, (".npy",)) if args.point_source in ("gridded", "both") else None
    off_gbn = _stem_map(gbn_off_d, (".npy",)) if args.point_source in ("gridded", "both") else None

    modes = ["raw", "gridded"] if args.point_source == "both" else [args.point_source]
    all_rows, summaries = [], {}
    primary_summary = None
    for mode in modes:
        if mode == "gridded" and (not off_wvs or not off_gbn):
            print(f"[gridded] offsets not found (wvs={wvs_off_d}, gbn={gbn_off_d}); skipping.")
            continue
        rows, summary, per_key = run_variant(
            stems, src_map, wvs_tgt, gbn_tgt, args,
            off_wvs=off_wvs, off_gbn=off_gbn, point_source=mode)
        all_rows.extend(rows)
        summaries[mode] = summary
        print_summary(summary)
        make_plots(per_key, rows, stems, src_map, wvs_tgt, gbn_tgt, args, args.out, tag=mode)
        if mode == "raw" or primary_summary is None:
            primary_summary = summary

    write_outputs(all_rows, summaries, args.out)

    # Verdict on the primary (raw) point source.
    passed, agree, coh, clean, consistency, direction = verdict(primary_summary)
    print("\n" + "#" * 84)
    print(f"GATE 0 VERDICT ({primary_summary['point_source']} point source, "
          f"descriptor={PRIMARY_KEY}):")
    print(f"  paired consistency = {consistency:.2f} (agreement GBN>WVS = {agree:.2f})   "
          f"cohen's d = {coh:.2f}   clean/non-overlapping = {clean}")
    print(f"  more edge-aligned oracle: {direction}")
    if passed:
        print("  --> PASS: WVS and GBN separate cleanly and consistently.")
        if direction == "WVS":
            print("      NOTE: the sign is INVERTED vs the plan's hypothesis (which expected")
            print("      GBN to hug contours). The gate passes -- the two oracles are")
            print("      distinguishable -- but the axis must be re-interpreted before")
            print("      Phase 2, and you should confirm visually (overlay plots) that the")
            print("      WVS/GBN directories are not swapped.")
        print("      Before Phase 2, check the SCALE profile: if frac_within_1px/2px separate")
        print("      but frac_within_4px/8px and dist_mean do NOT, the difference may be a")
        print("      sub-pixel dot-rendering artifact rather than a real arrangement axis.")
    else:
        print("  --> BLOCKER: separation is weak/inconsistent. STOP. Re-identify the visual "
              "property before training. Check whether ANY descriptor above separated, and "
              "inspect the overlay plots.")
    if "gridded" in summaries:
        gp, ga, gc, _, gcons, gdir = verdict(summaries["gridded"])
        print(f"  [gridded {args.grid_size}x{args.grid_size}] consistency={gcons:.2f} "
              f"d={gc:.2f} pass={gp} aligned={gdir}")
        print("      NOTE: OT offsets store EXACT continuous positions "
              "((p - grid_center) * G), so")
        print("      decoding them is lossless and 'gridded' is expected to match 'raw' almost")
        print("      exactly. A match here confirms the offset round-trip; it is NOT evidence")
        print("      about a representation ceiling.")
    print("#" * 84)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
