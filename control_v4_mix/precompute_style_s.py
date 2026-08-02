"""Precompute the per-(icon, oracle) style value s for the WVS<->GBN spacing-regularity axis.

s = measured norm_nn_cv (density-controlled spacing-regularity descriptor from Gate 0), decoded
from each oracle's OT offset grid and normalized so WVS median -> 0 and GBN median -> 1. This is
the continuous conditioning signal the Phase-2 model is trained on: WVS and GBN each form a
cluster (with real per-icon spread), and values between them specify a meaningful intermediate
regularity.

Offsets are exported by the normal training pipeline (control_v4/train_control.ensure_offsets_dir);
this script ensures they exist for both oracle roots, then computes s. Output: style_s.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spacing_regularity as sr  # noqa: E402
from edge_alignment import points_from_offsets  # noqa: E402

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw):
        return it

DEFAULT_WVS_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_WVS"
DEFAULT_GBN_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN"


def offsets_dir_for(root, explicit):
    return explicit if explicit else os.path.join(root, "processed_offsets")


def stem_map(offsets_dir):
    out = {}
    if not os.path.isdir(offsets_dir):
        return out
    for r, _, files in os.walk(offsets_dir):
        for f in files:
            if f.endswith(".npy"):
                stem = os.path.splitext(os.path.relpath(os.path.join(r, f), offsets_dir))[0]
                out[stem] = os.path.join(r, f)
    return out


def ensure_offsets(root, offsets_dir, grid_size):
    """Export offsets from targets if missing, reusing the training pipeline's exporter."""
    if stem_map(offsets_dir):
        return
    print(f"[{root}] offsets missing -> exporting via train_control.ensure_offsets_dir ...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from control_v4.train_control import ensure_offsets_dir  # heavy import; only when needed
    ensure_offsets_dir(os.path.join(root, "source"), os.path.join(root, "target"),
                       offsets_dir, grid_size)


def norm_nn_cv_for_dir(offsets_dir, grid_size, k, limit=None):
    """stem -> norm_nn_cv for every offset file in a dir."""
    smap = stem_map(offsets_dir)
    stems = sorted(smap)
    if limit:
        stems = stems[:limit]
    out = {}
    for stem in tqdm(stems, desc=os.path.basename(os.path.dirname(offsets_dir)), unit="icon",
                     dynamic_ncols=True):
        pts = points_from_offsets(np.load(smap[stem]), grid_size)
        out[stem] = sr.compute_spacing_bundle(pts, k=k)["norm_nn_cv"]
    return out


def main():
    p = argparse.ArgumentParser(description="Precompute style s (normalized norm_nn_cv).")
    p.add_argument("--wvs-root", default=DEFAULT_WVS_ROOT)
    p.add_argument("--gbn-root", default=DEFAULT_GBN_ROOT)
    p.add_argument("--wvs-offsets", default=""); p.add_argument("--gbn-offsets", default="")
    p.add_argument("--grid-size", type=int, default=32)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="control_v4_mix/style_s.json")
    args = p.parse_args()

    wvs_off = offsets_dir_for(args.wvs_root, args.wvs_offsets)
    gbn_off = offsets_dir_for(args.gbn_root, args.gbn_offsets)
    ensure_offsets(args.wvs_root, wvs_off, args.grid_size)
    ensure_offsets(args.gbn_root, gbn_off, args.grid_size)

    raw_wvs = norm_nn_cv_for_dir(wvs_off, args.grid_size, args.k, args.limit)
    raw_gbn = norm_nn_cv_for_dir(gbn_off, args.grid_size, args.k, args.limit)

    vw = np.array([v for v in raw_wvs.values() if np.isfinite(v)], dtype=np.float64)
    vg = np.array([v for v in raw_gbn.values() if np.isfinite(v)], dtype=np.float64)
    wvs_ref = float(np.median(vw)); gbn_ref = float(np.median(vg))
    denom = (gbn_ref - wvs_ref) if abs(gbn_ref - wvs_ref) > 1e-9 else 1.0

    def norm(v):
        return float((v - wvs_ref) / denom) if np.isfinite(v) else None

    stems = sorted(set(raw_wvs) & set(raw_gbn))
    entries = {stem: {"WVS": norm(raw_wvs[stem]), "GBN": norm(raw_gbn[stem]),
                      "WVS_raw": float(raw_wvs[stem]), "GBN_raw": float(raw_gbn[stem])}
               for stem in stems}
    refs = {"wvs_ref_zero": wvs_ref, "gbn_ref_one": gbn_ref, "grid_size": args.grid_size,
            "k": args.k, "n_icons": len(stems),
            "note": "s_norm = (norm_nn_cv - wvs_ref_zero) / (gbn_ref_one - wvs_ref_zero)"}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"_refs": refs, "entries": entries}, fh, indent=2)

    s_wvs = np.array([entries[s]["WVS"] for s in stems], float)
    s_gbn = np.array([entries[s]["GBN"] for s in stems], float)
    print("\n" + "=" * 70)
    print(f"style_s written: {args.out}   ({len(stems)} icons)")
    print(f"  norm_nn_cv medians: WVS={wvs_ref:.4f} -> s=0   GBN={gbn_ref:.4f} -> s=1")
    print(f"  s(WVS): mean={np.nanmean(s_wvs):+.3f} sd={np.nanstd(s_wvs):.3f}  "
          f"s(GBN): mean={np.nanmean(s_gbn):+.3f} sd={np.nanstd(s_gbn):.3f}")
    overlap = float(np.mean(s_wvs > np.nanmedian(s_gbn)))
    print(f"  cluster check: WVS above GBN-median fraction = {overlap:.3f} "
          "(should be ~0 -> clean two-cluster separation)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
        ax.hist(s_wvs, bins=40, alpha=0.6, label="WVS", color="#c05555")
        ax.hist(s_gbn, bins=40, alpha=0.6, label="GBN", color="#3b6ea5")
        ax.axvline(0, color="#c05555", ls="--", lw=1); ax.axvline(1, color="#3b6ea5", ls="--", lw=1)
        ax.set_xlabel("style s (normalized norm_nn_cv)"); ax.set_ylabel("icons"); ax.legend()
        ax.set_title("Style s distribution (WVS~0, GBN~1)")
        png = os.path.splitext(args.out)[0] + "_hist.png"
        fig.tight_layout(); fig.savefig(png, bbox_inches="tight"); plt.close(fig)
        print(f"  histogram: {png}")
    except Exception as exc:
        print(f"  (histogram skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
