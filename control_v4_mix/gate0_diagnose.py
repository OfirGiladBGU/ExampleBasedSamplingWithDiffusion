"""Gate 0 diagnostic -- is the WVS/GBN separation a real arrangement difference, or a
sub-pixel dot-rendering artifact?

Motivation: on the icon set the separation showed up ONLY at the 1-2px scale (frac_within_1px,
frac_within_2px, s_raw) and vanished by 4-8px, with near-identical mean distance-to-edge. That
profile is equally consistent with (a) a genuine "edge band" arrangement difference, or (b) the
two oracles rendering dots at slightly different size/offset, which shifts extracted centroids
about a pixel relative to the source's anti-aliased boundary.

This script separates those two explanations with two independent checks:

  1. RENDERING CHECK -- dot blob statistics per oracle (count, mean/median blob area in px).
     If GBN and WVS dots differ substantially in size, centroid positions carry a systematic
     sub-pixel bias and the 1px-scale signal is suspect.

  2. SCALE CHECK -- s_raw recomputed with the edge band dilated by r = 0,1,2,4,8 px.
     * A REAL arrangement axis keeps separating as r grows (points genuinely sit in the
       interior vs near the contour).
     * An ARTIFACT washes out quickly: s_raw for both oracles converges toward 1.0 by r=4-8.

Run it on a modest --limit; it is only meant to be read by eye.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_alignment as ea  # noqa: E402
from gate0_run import (  # noqa: E402
    DEFAULT_GBN_ROOT, DEFAULT_WVS_ROOT, VALID_EXT, _stem_map, resolve_dirs,
)

RADII = (0, 1, 2, 4, 8)


def blob_stats(target_png):
    """Count and pixel-area distribution of rendered dots."""
    img = np.asarray(Image.open(target_png).convert("L"), dtype=np.uint8)
    binary = ((255 - img) > 127).astype(np.uint8)
    lab, n = ndimage.label(binary)
    if n == 0:
        return 0, float("nan"), float("nan")
    areas = ndimage.sum(np.ones_like(binary), lab, index=np.arange(1, n + 1))
    return int(n), float(np.mean(areas)), float(np.median(areas))


def s_raw_at_radii(gray, pts, fields, radii=RADII):
    """s_raw with the edge band dilated by r pixels, for each r."""
    rho = ea.density_field(gray)
    rho_total = float(rho.sum())
    h, w = gray.shape
    yi, xi = ea._pixel_index(np.asarray(pts), h, w)
    band0 = fields["band"]
    out = []
    for r in radii:
        band = band0 if r == 0 else ndimage.binary_dilation(band0, iterations=int(r))
        expected = float(rho[band].sum() / (rho_total + ea.EPS)) if rho_total > ea.EPS else 0.0
        observed = float(band[yi, xi].mean()) if len(pts) else float("nan")
        out.append(observed / (expected + ea.EPS) if expected > 0 else float("nan"))
    return out


def main():
    p = argparse.ArgumentParser(description="Gate 0 artifact-vs-real diagnostic.")
    p.add_argument("--wvs-root", default=DEFAULT_WVS_ROOT)
    p.add_argument("--gbn-root", default=DEFAULT_GBN_ROOT)
    p.add_argument("--wvs-source", default=""); p.add_argument("--wvs-target", default="")
    p.add_argument("--gbn-source", default=""); p.add_argument("--gbn-target", default="")
    p.add_argument("--wvs-offsets", default=""); p.add_argument("--gbn-offsets", default="")
    p.add_argument("--edge", choices=("sobel", "canny"), default="sobel")
    p.add_argument("--edge-quantile", type=float, default=0.80)
    p.add_argument("--min-area", type=int, default=1)
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    wvs_src, wvs_tgt_d, gbn_src, gbn_tgt_d, _, _ = resolve_dirs(args)
    src_map = dict(_stem_map(gbn_src, VALID_EXT)); src_map.update(_stem_map(wvs_src, VALID_EXT))
    wvs_tgt = _stem_map(wvs_tgt_d, VALID_EXT)
    gbn_tgt = _stem_map(gbn_tgt_d, VALID_EXT)
    stems = sorted(set(wvs_tgt) & set(gbn_tgt) & set(src_map))[: args.limit]
    if not stems:
        print("ERROR: no matched stems.")
        return 2

    # ---- 1. RENDERING CHECK -------------------------------------------------
    print("=" * 88)
    print("1. RENDERING CHECK -- dot blob statistics (are the dots drawn the same way?)")
    print("=" * 88)
    hdr = f"{'stem':<28}{'N_wvs':>7}{'N_gbn':>7}{'area_wvs':>10}{'area_gbn':>10}{'ratio':>8}"
    print(hdr); print("-" * len(hdr))
    aw_all, ag_all = [], []
    for stem in stems:
        nw, aw, _ = blob_stats(wvs_tgt[stem])
        ng, ag, _ = blob_stats(gbn_tgt[stem])
        aw_all.append(aw); ag_all.append(ag)
        ratio = ag / aw if (aw and aw == aw and aw > 0) else float("nan")
        print(f"{stem[:27]:<28}{nw:>7}{ng:>7}{aw:>10.2f}{ag:>10.2f}{ratio:>8.2f}")
    mw, mg = np.nanmean(aw_all), np.nanmean(ag_all)
    print(f"\n  mean dot area: WVS={mw:.2f}px  GBN={mg:.2f}px  ratio={mg / mw:.2f}")
    if abs(mg / mw - 1.0) > 0.25:
        print("  !! Dot sizes differ by >25%. Centroid positions carry a systematic sub-pixel")
        print("     bias between oracles -- treat any 1px-scale separation as SUSPECT.")
    else:
        print("  Dot sizes are comparable; a sub-pixel rendering bias is unlikely to explain")
        print("  a large 1px-scale separation.")

    # ---- 2. SCALE CHECK -----------------------------------------------------
    print("\n" + "=" * 88)
    print("2. SCALE CHECK -- s_raw vs edge-band dilation radius (real axis persists; "
          "artifact washes out)")
    print("=" * 88)
    hdr2 = f"{'radius(px)':<12}{'s_raw WVS':>12}{'s_raw GBN':>12}{'gap':>10}{'more_aligned':>15}"
    print(hdr2); print("-" * len(hdr2))
    acc_w = {r: [] for r in RADII}
    acc_g = {r: [] for r in RADII}
    for stem in stems:
        gray = ea.load_gray01(src_map[stem])
        fields = ea.precompute_fields(gray, method=args.edge, edge_quantile=args.edge_quantile)
        pw = ea.extract_centroids(wvs_tgt[stem], min_area=args.min_area)
        pg = ea.extract_centroids(gbn_tgt[stem], min_area=args.min_area)
        sw = s_raw_at_radii(gray, pw, fields)
        sg = s_raw_at_radii(gray, pg, fields)
        for r, a, b in zip(RADII, sw, sg):
            acc_w[r].append(a); acc_g[r].append(b)
    for r in RADII:
        a = float(np.nanmean(acc_w[r])); b = float(np.nanmean(acc_g[r]))
        who = "WVS" if a > b else ("GBN" if b > a else "-")
        print(f"{r:<12}{a:>12.4f}{b:>12.4f}{abs(a - b):>10.4f}{who:>15}")
    g0 = abs(float(np.nanmean(acc_w[0])) - float(np.nanmean(acc_g[0])))
    g8 = abs(float(np.nanmean(acc_w[8])) - float(np.nanmean(acc_g[8])))
    print()
    if g0 > 0 and g8 / (g0 + ea.EPS) < 0.25:
        print("  VERDICT: the gap collapses to <25% of its r=0 value by r=8px, i.e. the whole")
        print("  effect lives in the exact top-gradient pixels and vanishes once the band is")
        print("  dilated. The signal is NOT a coarse-scale arrangement difference.")
        print()
        print("  Interpreting it depends on the rendering check above:")
        print("   * If dot sizes DIFFER -> sub-pixel centroid bias between oracles.")
        print("   * If dot sizes MATCH (they do here: 1px dots, matched N) -> the more likely")
        print("     explanation is TONE RESPONSE. The r=0 band is the anti-aliased boundary")
        print("     ring, i.e. the intermediate-INTENSITY pixels; the oracles differing there")
        print("     means they map intensity -> point density differently. That difference is")
        print("     rho-DECOMPOSABLE (reproducible by editing rho + running one oracle), which")
        print("     per the plan destroys the contribution.")
        print()
        print("  DECISIVE NEXT STEP -- controls for tone and settles it:")
        print("      python control_v4_mix/gate0_tone_vs_arrangement.py --limit 500")
        print("  Do NOT build the style axis on the s_raw signal until that test is run.")
    else:
        print("  VERDICT: the gap PERSISTS at coarse scales.")
        print("  => Consistent with a REAL arrangement difference. The axis is measurable")
        print("     at scales larger than dot-rendering noise.")
    print("\n  (Also open the overlay plots from gate0_run.py and confirm by eye which oracle")
    print("   hugs contours -- and that the WVS/GBN directories are not swapped.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
