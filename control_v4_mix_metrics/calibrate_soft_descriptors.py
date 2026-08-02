"""Do the differentiable descriptors agree with the exact ones? Run BEFORE enabling the loss.

If a soft descriptor drifts from the exact one, training optimises a quantity that is not the
quantity being reported -- the same class of error as an estimator that looks informative and
carries nothing. This measures the agreement instead of assuming it.

Three checks per descriptor:

  AGREEMENT   Pearson r and max |soft - exact| over real oracle point sets, per cell. `nn_cv`,
              `aniso` and `edge_align` should be EXACT (r = 1, error at float tolerance) because
              topk / gather / grid_sample reproduce the numpy computation rather than approximating
              it. `pcf_peak` is soft-binned and should merely track closely.
  ORACLE RANK The soft version must ORDER the oracles the same way the exact one does. Absolute
              agreement matters less than preserving the between-oracle separation M0 established --
              that ordering is what the conditioning means.
  GRADIENT    d(descriptor)/d(coords) must be finite and non-zero. A descriptor with no gradient
              contributes nothing to the loss no matter how well it correlates.

    python control_v4_mix_metrics/calibrate_soft_descriptors.py --limit 40
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import descriptor_fields as DF          # noqa: E402
import point_io as PIO                  # noqa: E402
import soft_descriptors as SD           # noqa: E402

DEFAULT_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_Oracles"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--oracles", default="gbn,wvs,fs,ordered,white,jitgrid")
    ap.add_argument("--limit", type=int, default=40, help="icons per oracle")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pcf-sigma", type=float, default=None,
                    help="soft-binning width for pcf_peak, in bin widths (default 0.75). Wider "
                         "smoothing flattens SHARP peaks hardest, which is why it mis-ranks "
                         "ordered dither; sweep it down to find where soft meets exact.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    oracles = [m.strip() for m in args.oracles.split(",") if m.strip()]
    source = os.path.join(args.root, "source")
    stems = sorted(os.path.splitext(v)[0] for v in PIO.stem_map(source))[: args.limit]
    if not stems:
        raise SystemExit(f"no source images under {source}")
    src_map = PIO.stem_map(source)

    stats_path = os.path.join(args.root, "DESCRIPTOR_STATS.json")
    stats_raw = json.load(open(stats_path))["descriptors"]

    keys = list(SD.SOFT_KEYS)
    soft_kw = {} if args.pcf_sigma is None else {"sigma_bins": args.pcf_sigma}
    print(f"  pcf sigma_bins = {args.pcf_sigma if args.pcf_sigma is not None else 0.75}")
    print(f"device={device}  icons={len(stems)}  oracles={oracles}\nkeys={keys}\n")

    pairs = {k: {"soft": [], "exact": []} for k in keys}
    per_oracle = {m: {k: {"soft": [], "exact": []} for k in keys} for m in oracles}

    for m in oracles:
        tdir = os.path.join(args.root, f"target_{m}")
        for stem in stems:
            npy = os.path.join(tdir, stem + ".npy")
            png = os.path.join(tdir, stem + ".png")
            if os.path.exists(npy):
                pts = PIO.load_points(npy)
            elif os.path.exists(png):
                pts = PIO.extract_centroids(png)
            else:
                continue
            if len(pts) < 256:
                continue
            gray = PIO.load_gray01(src_map[stem])
            grad_np = DF.gradient_magnitude(gray)

            ex_fields, _ = DF.descriptor_fields(pts, gray, G=args.grid, window=args.window,
                                                grad=grad_np)
            ex_valid = ex_fields["valid"]

            rho_np = np.clip(1.0 - gray, 0.0, 1.0)
            coords = torch.tensor(pts[None], dtype=torch.float32, device=device)
            grad_t = torch.tensor(grad_np[None, None], dtype=torch.float32, device=device)
            rho_t = torch.tensor(rho_np[None, None], dtype=torch.float32, device=device)
            with torch.no_grad():
                sf, sv = SD.soft_descriptor_fields(coords, grad_t, rho_map=rho_t, G=args.grid,
                                                   window=args.window, keys=keys, stats=None,
                                                   **soft_kw)
            sf = sf[0].cpu().numpy()
            mask = ex_valid & (sv[0, 0].cpu().numpy() > 0.5)
            for i, k in enumerate(keys):
                a = sf[i][mask]
                b = ex_fields[k][mask]
                ok = np.isfinite(a) & np.isfinite(b)
                pairs[k]["soft"].append(a[ok]); pairs[k]["exact"].append(b[ok])
                per_oracle[m][k]["soft"].append(np.nanmean(a[ok]) if ok.any() else np.nan)
                per_oracle[m][k]["exact"].append(np.nanmean(b[ok]) if ok.any() else np.nan)

    print("AGREEMENT (per cell, pooled over all oracles)")
    print(f"  {'key':12s} {'pearson r':>10s} {'max |diff|':>11s} {'mean |diff|':>12s} {'n cells':>9s}"
          f"   verdict")
    for k in keys:
        a = np.concatenate(pairs[k]["soft"]); b = np.concatenate(pairs[k]["exact"])
        r = float(np.corrcoef(a, b)[0, 1])
        mx, mn = float(np.abs(a - b).max()), float(np.abs(a - b).mean())
        verdict = ("EXACT" if mx < 1e-4 else
                   "tracks" if r > 0.98 else
                   "WEAK -- do not optimise" if r < 0.9 else "usable")
        print(f"  {k:12s} {r:10.5f} {mx:11.3e} {mn:12.3e} {len(a):9d}   {verdict}")

    print("\nORACLE RANK (per-oracle means; the ordering must survive)")
    for k in keys:
        se = [(m, np.nanmean(per_oracle[m][k]["soft"]), np.nanmean(per_oracle[m][k]["exact"]))
              for m in oracles]
        rs = [m for m, _, _ in sorted(se, key=lambda t: t[1])]
        re_ = [m for m, _, _ in sorted(se, key=lambda t: t[2])]
        print(f"  {k:12s} {'MATCH' if rs == re_ else 'DIFFERS'}")
        print(f"    soft  {' < '.join(rs)}")
        if rs != re_:
            print(f"    exact {' < '.join(re_)}")

    print("\nGRADIENT (must be finite and non-zero)")
    coords = torch.rand(1, 1024, 2, device=device, requires_grad=True)
    grad_t = torch.rand(1, 1, 512, 512, device=device)
    rho_t = torch.rand(1, 1, 512, 512, device=device)
    for k in keys:
        coords.grad = None
        f, _ = SD.soft_descriptor_fields(coords, grad_t, rho_map=rho_t, G=args.grid,
                                         window=args.window, keys=[k], stats=None,
                                         **(soft_kw if k == "pcf_peak" else {}))
        f.nansum().backward()
        g = coords.grad
        finite = bool(torch.isfinite(g).all())
        mx = float(g.abs().max())
        print(f"  {k:12s} finite={finite}  max|d/dcoords|={mx:.3e}"
              + ("" if (finite and mx > 0) else "   <-- UNUSABLE"))

    print("\nnormalisation bounds that the loss will apply:")
    for k in keys:
        print(f"  {k:12s} lo={stats_raw[k]['lo']:.4f}  hi={stats_raw[k]['hi']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
