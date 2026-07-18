"""run_m0.py -- Milestone 0 (Legibility), the BLOCKING first gate.

Question: rendered as DOTS ONLY, can a human distinguish kappa = 1/1.5/2 and
read the orientation field?

Method: ONE clean isotropic P0, warped by density-preserving area-preserving
maps (aniso_pilot.py) into tiles that share region, density, and exact point
budget -- differing only in oriented spacing. Rendered dots-only.

Objective backbone (support, not the gate): two independent estimators built
for area-preserving anisotropy (the signal is near-field only):
  strength  = aniso_pilot.global_near_field_anisotropy (coherent, closeness-wtd)
  nnratio   = aniso_pilot.nn_vector_anisotropy         (k=1 NN-vector covariance)
Both axes should point at theta+90 (the compressed axis).

Outputs (in --out): tiles/S##.png, blind_sheet.png, verify_sheet.png,
answer_key.csv, objective_report.txt, strength_vs_kappa.png, metrics.json.
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aniso_pilot as ap


def wrap_pi(a):
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def build_conditions():
    conds = []
    for kappa in [1.0, 1.25, 1.5, 2.0]:
        conds.append(dict(family="kappa", theta_deg=30.0, kappa=kappa))
    for th in [0.0, 45.0, 90.0, 135.0]:
        conds.append(dict(family="orient", theta_deg=th, kappa=2.0))
    conds.append(dict(family="swirl", theta_deg=None, kappa=None))
    return conds


def apply_warp(P0, cond):
    if cond["family"] == "swirl":
        return ap.warp_divfree_swirl(P0, strength=2.2, sigma=0.28)
    return ap.warp_constant(P0, np.deg2rad(cond["theta_deg"]), cond["kappa"])


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--n", type=int, default=1000)
    pa.add_argument("--density", type=float, default=1600.0)
    pa.add_argument("--hf", type=float, default=1.0, help="near-field bandwidth (in NN dist)")
    pa.add_argument("--rf", type=float, default=3.0, help="near-field cutoff (in NN dist)")
    pa.add_argument("--dotsize", type=float, default=5.0)
    pa.add_argument("--out", type=str, default="m0_out")
    args = pa.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "tiles"), exist_ok=True)

    radius = ap.radius_for_count(args.density)
    P0 = ap.poisson_disk_bridson(radius, domain=(-0.6, -0.6, 1.6, 1.6), rng=rng)
    print("P0: %d points, radius=%.4f" % (len(P0), radius))

    conds = build_conditions()
    crop = (0.0, 0.0, 1.0, 1.0)
    cropped = [ap.crop_points(apply_warp(P0, c), crop) for c in conds]
    target = min(args.n, min(len(c) for c in cropped))
    print("cropped counts:", [len(c) for c in cropped], "-> exact N =", target)

    order = rng.permutation(len(conds))
    codes = {int(i): "S%02d" % (rank + 1) for rank, i in enumerate(order)}

    tiles = []
    tile_pts = {}
    for i, (cond, pool) in enumerate(zip(conds, cropped)):
        pts = ap.subsample(pool, target, rng)
        tile_pts[i] = pts
        code = codes[i]
        ap.render_dots(pts, os.path.join(args.out, "tiles", code + ".png"),
                       crop=crop, size=args.dotsize)

        g = ap.global_near_field_anisotropy(pts, h_factor=args.hf, r_factor=args.rf)
        nnc = ap.nn_vector_anisotropy(pts, kk=1)
        rec = dict(code=code, index=i, family=cond["family"],
                   theta_deg=cond["theta_deg"], kappa=cond["kappa"],
                   n=int(len(pts)), strength=g["strength"],
                   nnratio=nnc["ratio"], orient_err_deg=None)
        if cond["family"] in ("kappa", "orient") and cond["kappa"] > 1.0:
            expected = np.deg2rad(cond["theta_deg"]) + np.pi / 2.0
            rec["orient_err_deg"] = float(np.rad2deg(abs(wrap_pi(g["axis"] - expected))))
        tiles.append(rec)
        print("  %s  %-6s theta=%s kappa=%s  strength=%.3f nnratio=%.2f orient_e=%s" % (
            code, cond["family"], str(cond["theta_deg"]), str(cond["kappa"]),
            g["strength"], nnc["ratio"],
            ("%.1f" % rec["orient_err_deg"]) if rec["orient_err_deg"] else "-"))

    tiles_by_code = sorted(tiles, key=lambda r: r["code"])
    _sheet(tiles_by_code, tile_pts, conds, codes, args, crop,
           os.path.join(args.out, "blind_sheet.png"), reveal=False)
    _sheet(tiles_by_code, tile_pts, conds, codes, args, crop,
           os.path.join(args.out, "verify_sheet.png"), reveal=True)

    with open(os.path.join(args.out, "answer_key.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "family", "theta_deg", "kappa", "n"])
        for r in tiles_by_code:
            w.writerow([r["code"], r["family"], r["theta_deg"], r["kappa"], r["n"]])

    kfam = sorted([r for r in tiles if r["family"] == "kappa"], key=lambda r: r["kappa"])
    kc = [r["kappa"] for r in kfam]
    ks = [r["strength"] for r in kfam]
    fig, axp = plt.subplots(figsize=(5, 4))
    axp.plot(kc, ks, "o-", color="black")
    axp.set_xlabel("commanded kappa")
    axp.set_ylabel("near-field anisotropy strength")
    axp.set_title("M0 geometry: anisotropy strength vs kappa")
    axp.grid(alpha=0.3)
    fig.savefig(os.path.join(args.out, "strength_vs_kappa.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    floor = ks[0]
    monotonic = all(ks[i + 1] > ks[i] for i in range(len(ks) - 1))
    separated = all(ks[j] > floor + 0.03 for j in range(1, len(ks)))
    oe = [r["orient_err_deg"] for r in tiles
          if r["family"] == "orient" and r["orient_err_deg"] is not None]
    orient_ok = all(e < 15.0 for e in oe) if oe else False

    lines = []
    lines.append("Milestone 0 -- objective geometry check (support, not the gate)")
    lines.append("=" * 62)
    lines.append("P0 points: %d   exact N per tile: %d   near-field hf=%.1f rf=%.1f"
                 % (len(P0), target, args.hf, args.rf))
    lines.append("(stretch along theta -> measured axis = theta+90; nnratio grows with kappa)")
    lines.append("")
    lines.append("%-5s %-7s %7s %7s %10s %9s %9s" % (
        "code", "family", "theta", "kappa", "strength", "nnratio", "orient_e"))
    for r in tiles_by_code:
        lines.append("%-5s %-7s %7s %7s %10.3f %9.2f %9s" % (
            r["code"], r["family"], str(r["theta_deg"]), str(r["kappa"]),
            r["strength"], r["nnratio"],
            ("%.1f" % r["orient_err_deg"]) if r["orient_err_deg"] is not None else "-"))
    lines.append("")
    lines.append("strength floor at commanded kappa=1   : %.3f" % floor)
    lines.append("strength monotonic in commanded kappa : %s" % monotonic)
    lines.append("kappa>1 separated from floor (+0.03)  : %s" % separated)
    lines.append("orientation error < 15 deg (kappa=2)  : %s" % orient_ok)
    lines.append("")
    passed = monotonic and separated and orient_ok
    lines.append("Geometry check %s." % ("PASSES" if passed else "has ISSUES"))
    lines.append("")
    lines.append("THE M0 GATE IS PERCEPTUAL: open blind_sheet.png and, without the")
    lines.append("key, (1) rank tiles by anisotropy, (2) read orientation of the")
    lines.append("anisotropic ones. Then score against answer_key.csv. If oriented")
    lines.append("spacing is NOT visible in pure dots -> STOP the anisotropy track;")
    lines.append("fall back to regularity/spectrum control.")
    report = "\n".join(lines)
    with open(os.path.join(args.out, "objective_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(seed=args.seed, n=target, hf=args.hf, rf=args.rf,
                       p0=len(P0), tiles=tiles, floor=floor, monotonic=monotonic,
                       separated=separated, orient_ok=orient_ok, passed=passed),
                  f, indent=2)


def _sheet(tiles_by_code, tile_pts, conds, codes, args, crop, path, reveal):
    idx_by_code = {codes[i]: i for i in range(len(conds))}
    n = len(tiles_by_code)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    x0, y0, x1, y1 = crop
    for a in axes:
        a.axis("off")
    for ax_i, rec in enumerate(tiles_by_code):
        ax = axes[ax_i]
        i = idx_by_code[rec["code"]]
        pts = tile_pts[i]
        ax.scatter(pts[:, 0], pts[:, 1], s=args.dotsize, c="black", edgecolors="none")
        ax.set_xlim(x0 - 0.01, x1 + 0.01)
        ax.set_ylim(y0 - 0.01, y1 + 0.01)
        ax.set_aspect("equal")
        ax.axis("off")
        if reveal:
            c = conds[i]
            if c["family"] == "swirl":
                ax.set_title("%s  swirl (field)" % rec["code"], fontsize=10)
            else:
                ax.set_title("%s  k=%.2f  th=%.0f" % (rec["code"], c["kappa"], c["theta_deg"]),
                             fontsize=10)
                if c["kappa"] > 1.0:
                    th = np.deg2rad(c["theta_deg"])
                    dx, dy = 0.12 * np.cos(th), 0.12 * np.sin(th)
                    ax.plot([0.5 - dx, 0.5 + dx], [0.5 - dy, 0.5 + dy],
                            color="red", lw=2, alpha=0.7)
        else:
            ax.set_title(rec["code"], fontsize=11)
    ttl = "M0 verification (answers shown)" if reveal else \
          "M0 blind legibility test -- dots only. Rank by anisotropy; read orientation."
    fig.suptitle(ttl, fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
