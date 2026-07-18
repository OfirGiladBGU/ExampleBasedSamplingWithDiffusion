"""run_kappa_ladder.py -- fine kappa ladder to locate the M0 perceptual floor.

At fixed orientation, render dots-only tiles for kappa = kmin..kmax in fine
steps, density/N-matched (same construction as run_m0), so you can read off the
smallest kappa at which oriented spacing becomes visible. Also plots the
objective near-field anisotropy strength across the same fine grid.

Outputs (in --out):
  tiles/k###.png       individual dots-only tiles (named by kappa*100)
  ladder_ordered.png   labelled strip, increasing kappa  (find the threshold)
  ladder_blind.png     shuffled coded strip              (unbiased check)
  ladder_key.csv       blind code -> kappa
  strength_vs_kappa.png + ladder_strength.csv   objective strength + nnratio
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aniso_pilot as ap


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--theta_deg", type=float, default=30.0)
    pa.add_argument("--kmin", type=float, default=1.0)
    pa.add_argument("--kmax", type=float, default=2.0)
    pa.add_argument("--step", type=float, default=0.1)
    pa.add_argument("--n", type=int, default=1000)
    pa.add_argument("--density", type=float, default=1600.0)
    pa.add_argument("--hf", type=float, default=1.0)
    pa.add_argument("--rf", type=float, default=3.0)
    pa.add_argument("--dotsize", type=float, default=5.0)
    pa.add_argument("--ncols", type=int, default=6)
    pa.add_argument("--out", type=str, default="ladder_out")
    args = pa.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "tiles"), exist_ok=True)

    radius = ap.radius_for_count(args.density)
    P0 = ap.poisson_disk_bridson(radius, domain=(-0.6, -0.6, 1.6, 1.6), rng=rng)
    theta = np.deg2rad(args.theta_deg)
    crop = (0.0, 0.0, 1.0, 1.0)

    kappas = np.round(np.arange(args.kmin, args.kmax + 1e-9, args.step), 3)
    cropped = [ap.crop_points(ap.warp_constant(P0, theta, float(k)), crop) for k in kappas]
    target = min(args.n, min(len(c) for c in cropped))
    print("P0=%d  theta=%.0f  kappas=%s  exact N=%d"
          % (len(P0), args.theta_deg, list(kappas), target))

    tile_pts = []
    strengths = []
    nnratios = []
    for k, pool in zip(kappas, cropped):
        pts = ap.subsample(pool, target, rng)
        tile_pts.append(pts)
        ap.render_dots(pts, os.path.join(args.out, "tiles", "k%03d.png" % round(k * 100)),
                       crop=crop, size=args.dotsize)
        g = ap.global_near_field_anisotropy(pts, h_factor=args.hf, r_factor=args.rf)
        nnc = ap.nn_vector_anisotropy(pts, kk=1)
        strengths.append(g["strength"])
        nnratios.append(nnc["ratio"])
        print("  kappa=%.2f  strength=%.3f  nnratio=%.2f" % (k, g["strength"], nnc["ratio"]))

    _grid(tile_pts, ["k=%.2f" % k for k in kappas], args, crop,
          os.path.join(args.out, "ladder_ordered.png"),
          "kappa ladder (theta=%.0f) -- mark where anisotropy first appears" % args.theta_deg)

    order = rng.permutation(len(kappas))
    blind_pts = [tile_pts[i] for i in order]
    codes = ["B%02d" % (j + 1) for j in range(len(order))]
    _grid(blind_pts, codes, args, crop,
          os.path.join(args.out, "ladder_blind.png"),
          "kappa ladder BLIND -- rank by anisotropy, then check the key")
    with open(os.path.join(args.out, "ladder_key.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "kappa"])
        for c, i in zip(codes, order):
            w.writerow([c, float(kappas[i])])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kappas, strengths, "o-", color="black", label="near-field strength")
    ax.set_xlabel("kappa")
    ax.set_ylabel("near-field anisotropy strength")
    ax2 = ax.twinx()
    ax2.plot(kappas, nnratios, "s--", color="tab:red", label="nn-vector ratio")
    ax2.set_ylabel("nn-vector ratio", color="tab:red")
    ax.set_title("Objective anisotropy vs kappa (theta=%.0f)" % args.theta_deg)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.out, "strength_vs_kappa.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(args.out, "ladder_strength.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kappa", "strength", "nnratio"])
        for k, s, r in zip(kappas, strengths, nnratios):
            w.writerow([float(k), s, r])
    print("done -> %s" % args.out)


def _grid(pts_list, labels, args, crop, path, suptitle):
    n = len(pts_list)
    cols = min(args.ncols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    x0, y0, x1, y1 = crop
    for a in axes:
        a.axis("off")
    for i, (pts, lab) in enumerate(zip(pts_list, labels)):
        ax = axes[i]
        ax.scatter(pts[:, 0], pts[:, 1], s=args.dotsize, c="black", edgecolors="none")
        ax.set_xlim(x0 - 0.01, x1 + 0.01)
        ax.set_ylim(y0 - 0.01, y1 + 0.01)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(lab, fontsize=10)
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
