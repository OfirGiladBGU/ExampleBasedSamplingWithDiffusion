"""DECISIVE Gate 0 test: is the WVS/GBN difference ARRANGEMENT, or just TONE RESPONSE?

Why this exists
---------------
The first Gate 0 pass found WVS and GBN separating on s_raw (WVS ~1.0, GBN ~0.57) with the
effect confined ENTIRELY to the r=0 edge band (gone by r=2px), while mean distance-to-edge was
a coin flip. The r=0 band is the anti-aliased boundary ring -- i.e. the intermediate-INTENSITY
pixels. That signature suggests the two oracles differ in how they map intensity -> point
density, NOT in how they arrange points relative to structure.

This matters because a tone-response difference is rho-DECOMPOSABLE: a reviewer reproduces it
by editing rho and re-running a single oracle (Path A). Per the plan that destroys the
contribution. An arrangement difference is not reproducible that way.

The flaw it fixes
-----------------
edge_alignment.rho_normalized_alignment normalizes against GLOBAL rho mass, which does NOT
control for a differing intensity->density response. So a pure tone-response difference shows up
there as fake "alignment". This script removes that confound.

Two parts
---------
1. TONE RESPONSE -- empirical intensity -> points-per-pixel curve for each oracle. If the curves
   differ, the oracles genuinely disagree about how much ink a given grey level deserves, and
   that difference is rho-editable (decomposable).

2. INTENSITY-STRATIFIED ALIGNMENT (the decisive number) -- s_strat. Within each intensity bin we
   ask what fraction of THAT BIN's pixels are edge-band pixels, then compute the band occupancy
   each oracle would show if it had no edge preference beyond its own tone distribution:

       expected = sum_b  pts_total[b] * (pix_band[b] / pix_total[b])  /  sum_b pts_total[b]
       observed = sum_b  pts_band[b]                                  /  sum_b pts_total[b]
       s_strat  = observed / expected

   s_strat is arrangement WITH TONE CONTROLLED FOR. If WVS and GBN have the same s_strat, the
   entire s_raw gap was tone response -> decomposable -> the style axis is dead as framed.
   If they still differ, a real arrangement axis survives the control.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_alignment as ea  # noqa: E402
from gate0_run import (  # noqa: E402
    DEFAULT_GBN_ROOT, DEFAULT_WVS_ROOT, VALID_EXT, _stem_map, cohens_d, resolve_dirs,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw):
        return it

N_BINS = 16


def stratified(gray, pts, band, edges):
    """Return (s_strat, pts_total_per_bin, pts_band_per_bin) for one point set."""
    h, w = gray.shape
    nb = len(edges) - 1
    if len(pts) == 0:
        return float("nan"), np.zeros(nb), np.zeros(nb)
    yi, xi = ea._pixel_index(np.asarray(pts), h, w)
    gi = gray[yi, xi]
    b = np.clip(np.digitize(gi, edges) - 1, 0, nb - 1)
    pts_total = np.bincount(b, minlength=nb).astype(np.float64)
    in_band = band[yi, xi]
    pts_band = np.bincount(b[in_band], minlength=nb).astype(np.float64)

    pix_b = np.clip(np.digitize(gray.ravel(), edges) - 1, 0, nb - 1)
    pix_total = np.bincount(pix_b, minlength=nb).astype(np.float64)
    pix_band = np.bincount(pix_b[band.ravel()], minlength=nb).astype(np.float64)
    band_frac = np.divide(pix_band, np.maximum(pix_total, 1.0))

    tot = pts_total.sum()
    if tot <= 0:
        return float("nan"), pts_total, pts_band
    observed = pts_band.sum() / tot
    expected = float((pts_total * band_frac).sum() / tot)
    s = observed / expected if expected > ea.EPS else float("nan")
    return s, pts_total, pts_band


def main():
    p = argparse.ArgumentParser(description="Tone-response vs arrangement (decisive Gate 0).")
    p.add_argument("--wvs-root", default=DEFAULT_WVS_ROOT)
    p.add_argument("--gbn-root", default=DEFAULT_GBN_ROOT)
    p.add_argument("--wvs-source", default=""); p.add_argument("--wvs-target", default="")
    p.add_argument("--gbn-source", default=""); p.add_argument("--gbn-target", default="")
    p.add_argument("--wvs-offsets", default=""); p.add_argument("--gbn-offsets", default="")
    p.add_argument("--edge", choices=("sobel", "canny"), default="sobel")
    p.add_argument("--edge-quantile", type=float, default=0.80)
    p.add_argument("--min-area", type=int, default=1)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--out", default="control_v4_mix/gate0_outputs")
    args = p.parse_args()

    wvs_src, wvs_tgt_d, gbn_src, gbn_tgt_d, _, _ = resolve_dirs(args)
    src_map = dict(_stem_map(gbn_src, VALID_EXT)); src_map.update(_stem_map(wvs_src, VALID_EXT))
    wvs_tgt = _stem_map(wvs_tgt_d, VALID_EXT); gbn_tgt = _stem_map(gbn_tgt_d, VALID_EXT)
    stems = sorted(set(wvs_tgt) & set(gbn_tgt) & set(src_map))[: args.limit]
    if not stems:
        print("ERROR: no matched stems."); return 2
    print(f"Using {len(stems)} icons.", flush=True)

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    sw_all, sg_all = [], []
    tone_w = np.zeros(N_BINS); tone_g = np.zeros(N_BINS); pix_all = np.zeros(N_BINS)

    for stem in tqdm(stems, desc="tone/arrangement", unit="icon", dynamic_ncols=True):
        gray = ea.load_gray01(src_map[stem])
        fields = ea.precompute_fields(gray, method=args.edge, edge_quantile=args.edge_quantile)
        band = fields["band"]
        pw = ea.extract_centroids(wvs_tgt[stem], min_area=args.min_area)
        pg = ea.extract_centroids(gbn_tgt[stem], min_area=args.min_area)
        sw, ptw, _ = stratified(gray, pw, band, edges)
        sg, ptg, _ = stratified(gray, pg, band, edges)
        sw_all.append(sw); sg_all.append(sg)
        tone_w += ptw; tone_g += ptg
        pb = np.clip(np.digitize(gray.ravel(), edges) - 1, 0, N_BINS - 1)
        pix_all += np.bincount(pb, minlength=N_BINS)

    # ---- 1. TONE RESPONSE ---------------------------------------------------
    print("\n" + "=" * 88)
    print("1. TONE RESPONSE -- points per 1000 pixels at each source intensity")
    print("=" * 88)
    hdr = f"{'intensity bin':<18}{'pixels':>12}{'pts/1k WVS':>13}{'pts/1k GBN':>13}{'GBN/WVS':>10}"
    print(hdr); print("-" * len(hdr))
    for i in range(N_BINS):
        if pix_all[i] <= 0:
            continue
        rw = 1000.0 * tone_w[i] / pix_all[i]
        rg = 1000.0 * tone_g[i] / pix_all[i]
        ratio = rg / rw if rw > ea.EPS else float("nan")
        print(f"[{edges[i]:.2f},{edges[i+1]:.2f})      {pix_all[i]:>12.0f}"
              f"{rw:>13.3f}{rg:>13.3f}{ratio:>10.2f}")
    print("\n  A ratio that drifts away from 1.00 across intensity = the oracles disagree on the")
    print("  intensity->density mapping. THAT DIFFERENCE IS rho-EDITABLE (decomposable).")

    # ---- 2. STRATIFIED ALIGNMENT (decisive) ---------------------------------
    sw_arr = np.array(sw_all, dtype=np.float64); sg_arr = np.array(sg_all, dtype=np.float64)
    m = np.isfinite(sw_arr) & np.isfinite(sg_arr)
    mw, mg = float(np.nanmean(sw_arr)), float(np.nanmean(sg_arr))
    d = cohens_d(sw_arr, sg_arr, "s_raw")  # 'higher = more aligned' orientation
    agree = float(np.mean(sg_arr[m] > sw_arr[m])) if m.any() else float("nan")
    consistency = max(agree, 1.0 - agree) if agree == agree else float("nan")

    print("\n" + "=" * 88)
    print("2. INTENSITY-STRATIFIED ALIGNMENT (decisive) -- arrangement with tone CONTROLLED FOR")
    print("=" * 88)
    print(f"  s_strat  WVS = {mw:.4f}   GBN = {mg:.4f}   gap = {abs(mg - mw):.4f}")
    print(f"  cohen's d = {d:.3f}   paired consistency = {consistency:.2f}")
    print("  (s_strat ~= 1.0 means: no edge preference beyond what this oracle's own tone")
    print("   distribution already implies.)")

    print()
    if abs(d) < 0.5 or consistency < 0.75:
        print("  VERDICT: once tone is controlled for, the oracles do NOT separate on")
        print("  arrangement. The s_raw gap seen earlier was a TONE-RESPONSE difference, which")
        print("  is rho-decomposable (reproducible by editing rho + running one oracle).")
        print("  => The WVS<->GBN style axis does NOT carry a contribution as framed.")
        print("     Per the plan's stop condition: STOP and re-identify the visual property.")
        print("     Candidates the plan itself names: detail preservation, or clumping/spacing")
        print("     regularity (a spacing property, not an alignment one).")
    else:
        print("  VERDICT: the oracles STILL separate after controlling for tone.")
        print("  => A real arrangement difference survives. The axis is viable, but note the")
        print("     direction and re-name it to match the measured sign.")
    print("\n  Cross-check: mean distance-to-edge was a coin flip (consistency ~0.55) in the")
    print("  full run. A real arrangement axis would normally move that too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
