"""Find the soft-binning width at which the differentiable pcf_peak matches the exact one.

`pcf_peak` is the only descriptor in `soft_descriptors.py` that is a genuine approximation: hard
histogram binning is replaced by a Gaussian, and the width of that Gaussian trades agreement against
gradient smoothness.

The failure mode is specific and worth stating, because it is not a uniform bias. Smoothing flattens
a peak in proportion to its SHARPNESS, so the sharpest oracle (ordered dither, an exact lattice)
loses the most height. At sigma = 0.75 that was enough to REORDER the oracles -- soft ranked ordered
second where exact ranks it first -- which is worse than a scale error, because the ordering is what
the conditioning value means.

The exact descriptors are computed ONCE and cached; only the soft side is recomputed per sigma. That
makes the sweep roughly as cheap as a single calibration run rather than N of them.

Pick the LARGEST sigma that passes both bars (r >= --r-target and rank MATCH): wider smoothing gives
a better-conditioned gradient, so there is no reason to go narrower than necessary.

    python control_v4_mix_metrics/sweep_pcf_sigma.py --limit 20
    python control_v4_mix_metrics/sweep_pcf_sigma.py --sigmas 0.6,0.4,0.3,0.2,0.1 --limit 40
"""

import argparse
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
    ap.add_argument("--sigmas", default="0.75,0.5,0.35,0.25,0.15,0.08")
    ap.add_argument("--limit", type=int, default=20, help="icons per oracle")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--r-target", type=float, default=0.98)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    oracles = [m.strip() for m in args.oracles.split(",") if m.strip()]
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    source = os.path.join(args.root, "source")
    src_map = PIO.stem_map(source)
    stems = sorted(os.path.splitext(v)[0] for v in src_map)[: args.limit]
    if not stems:
        raise SystemExit(f"no source images under {source}")

    print(f"device={device}  icons/oracle={len(stems)}  oracles={oracles}")
    print(f"sigmas={sigmas}  r-target={args.r_target}\n")

    # ---- cache the expensive half once -------------------------------------
    print("computing exact pcf_peak (cached for every sigma) ...", flush=True)
    cache = []
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
            ex, _ = DF.descriptor_fields(pts, gray, G=args.grid, window=args.window, grad=grad_np)
            cache.append({
                "oracle": m,
                "coords": torch.tensor(pts[None], dtype=torch.float32, device=device),
                "grad": torch.tensor(grad_np[None, None], dtype=torch.float32, device=device),
                "rho": torch.tensor(np.clip(1.0 - gray, 0, 1)[None, None],
                                    dtype=torch.float32, device=device),
                "exact": ex["pcf_peak"],
                "valid": ex["valid"],
            })
    print(f"  cached {len(cache)} point sets\n")

    print(f"{'sigma':>7s} {'pearson r':>10s} {'mean|diff|':>11s} {'max|diff|':>10s} "
          f"{'rank':>8s} {'max|d/dcoords|':>15s}   verdict")
    results = []
    for sig in sigmas:
        A, B = [], []
        per = {m: {"s": [], "e": []} for m in oracles}
        for c in cache:
            with torch.no_grad():
                sf, sv = SD.soft_descriptor_fields(
                    c["coords"], c["grad"], rho_map=c["rho"], G=args.grid, window=args.window,
                    keys=["pcf_peak"], stats=None, sigma_bins=sig)
            msk = c["valid"] & (sv[0, 0].cpu().numpy() > 0.5)
            a = sf[0, 0].cpu().numpy()[msk]
            b = c["exact"][msk]
            ok = np.isfinite(a) & np.isfinite(b)
            if not ok.any():
                continue
            A.append(a[ok]); B.append(b[ok])
            per[c["oracle"]]["s"].append(float(np.nanmean(a[ok])))
            per[c["oracle"]]["e"].append(float(np.nanmean(b[ok])))
        A = np.concatenate(A); B = np.concatenate(B)
        r = float(np.corrcoef(A, B)[0, 1])
        md, mx = float(np.abs(A - B).mean()), float(np.abs(A - B).max())
        rs = sorted(oracles, key=lambda m: np.nanmean(per[m]["s"]))
        re_ = sorted(oracles, key=lambda m: np.nanmean(per[m]["e"]))
        rank_ok = rs == re_

        cg = torch.rand(1, 1024, 2, device=device, requires_grad=True)
        f, _ = SD.soft_descriptor_fields(
            cg, torch.rand(1, 1, 512, 512, device=device),
            rho_map=torch.rand(1, 1, 512, 512, device=device),
            G=args.grid, window=args.window, keys=["pcf_peak"], stats=None, sigma_bins=sig)
        f.nansum().backward()
        gmax = float(cg.grad.abs().max())

        passed = (r >= args.r_target) and rank_ok and np.isfinite(gmax) and gmax > 0
        results.append((sig, r, rank_ok, gmax, passed))
        print(f"{sig:7.2f} {r:10.5f} {md:11.3e} {mx:10.3e} "
              f"{'MATCH' if rank_ok else 'DIFFERS':>8s} {gmax:15.3e}   "
              f"{'PASS' if passed else ''}")
        if not rank_ok:
            print(f"        soft  {' < '.join(rs)}")
            print(f"        exact {' < '.join(re_)}")

    ok = [t for t in results if t[4]]
    print()
    if ok:
        best = max(ok, key=lambda t: t[0])           # widest passing sigma
        print(f"RECOMMENDED  sigma_bins = {best[0]}  (r = {best[1]:.5f}, rank MATCH)")
        print("  Widest passing value: smoothing helps the gradient, so do not go narrower than\n"
              "  agreement requires. Pass it as --pcf-sigma / sigma_bins when enabling the loss.")
    else:
        print("NO SIGMA PASSED.")
        print("  pcf_peak cannot be optimised faithfully at these settings -- the soft statistic\n"
              "  does not reproduce the exact one it would be judged against. Run the consistency\n"
              "  loss on nn_cv + edge_align + aniso, which calibrated EXACT (r >= 0.9986) with\n"
              "  matching oracle rank. The descriptor is still available as a CONDITIONING input;\n"
              "  only the loss term would omit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
