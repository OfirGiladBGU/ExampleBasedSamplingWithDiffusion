"""visual_primer.py -- what does anisotropy actually LOOK like?

Anisotropy is a LOCAL, sub-cell property: at 1024 dots in a small panel it is
essentially invisible. You have to ZOOM. Each tile below therefore shows the
full point set plus a magnified crop of the middle.

Convention (the one the sampler/model use): J = metric_M(theta, kappa), so the
exclusion ellipse extends FURTHER PERPENDICULAR to theta. Points therefore pack
CLOSER ALONG theta and spread FURTHER ACROSS it:

    theta = 0deg   -> dots form HORIZONTAL chains, gaps stack vertically
    theta = 90deg  -> dots form VERTICAL chains, gaps stack horizontally

Density is identical in every tile -- same 1024 points, same area. Only the
arrangement changes. That is the whole point: a density/tone map cannot express
this, which is why GBN/WVS cannot produce it.

Outputs:
  primer_kappa.png  kappa ladder at fixed theta -> how STRENGTH maps to looks
  primer_theta.png  fixed kappa, rotating theta -> how ORIENTATION reads

  python visual_primer.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import aniso_m2 as m2
import aniso_pilot as ap


N = 1024
ZOOM = (0.36, 0.36, 0.28)      # x0, y0, size of the magnified crop
DOT_FULL = 5.0
DOT_ZOOM = 26.0
OUT_DIR = _HERE


def make(kappa, theta_deg, seed=0):
    th_fn, ka_fn, kmax = m2.const_field(theta_deg, kappa)
    rng = np.random.default_rng(seed)
    P, s, _ = m2.sample_exact_n(th_fn, ka_fn, 0.62 / np.sqrt(N), N, rng, kmax)
    st = ap.global_near_field_anisotropy(P)["strength"]
    return P, float(st)


def draw_tile(ax_full, ax_zoom, P, title, sub):
    ax_full.scatter(P[:, 0], P[:, 1], s=DOT_FULL, c="black", edgecolors="none")
    x0, y0, w = ZOOM
    ax_full.add_patch(Rectangle((x0, y0), w, w, fill=False, ec="tab:red", lw=1.6))
    ax_full.set_xlim(0, 1); ax_full.set_ylim(0, 1)
    ax_full.set_aspect("equal"); ax_full.axis("off")
    ax_full.set_title(title, fontsize=11)

    m = ((P[:, 0] >= x0) & (P[:, 0] <= x0 + w) &
         (P[:, 1] >= y0) & (P[:, 1] <= y0 + w))
    ax_zoom.scatter(P[m, 0], P[m, 1], s=DOT_ZOOM, c="black", edgecolors="none")
    ax_zoom.set_xlim(x0, x0 + w); ax_zoom.set_ylim(y0, y0 + w)
    ax_zoom.set_aspect("equal")
    for sp in ax_zoom.spines.values():
        sp.set_edgecolor("tab:red"); sp.set_linewidth(1.6)
    ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
    ax_zoom.set_xlabel(sub, fontsize=10)


def figure_kappa():
    kappas = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    fig, axes = plt.subplots(2, len(kappas), figsize=(3.1 * len(kappas), 6.8))
    for i, k in enumerate(kappas):
        P, st = make(k, 0.0, seed=1)
        draw_tile(axes[0, i], axes[1, i], P,
                  "kappa = %.1f" % k, "strength %.3f" % st)
    fig.suptitle("ANISOTROPY LADDER  (theta = 0deg, identical density in every tile)\n"
                 "top: full field   bottom: 3.5x zoom of the red box  -- "
                 "look for HORIZONTAL chains of dots emerging as kappa grows",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(OUT_DIR, "primer_kappa.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def figure_theta():
    thetas = [0.0, 45.0, 90.0, 135.0]
    fig, axes = plt.subplots(2, len(thetas), figsize=(3.4 * len(thetas), 7.0))
    for i, t in enumerate(thetas):
        P, st = make(2.0, t, seed=2)
        draw_tile(axes[0, i], axes[1, i], P,
                  "theta = %.0fdeg" % t, "strength %.3f" % st)
    fig.suptitle("ORIENTATION  (kappa = 2.0 everywhere, identical density)\n"
                 "the chains rotate WITH theta: 0deg horizontal, 90deg vertical",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(OUT_DIR, "primer_theta.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    figure_kappa()
    figure_theta()
    print()
    print("HOW TO READ THE NUMBERS WE HAVE BEEN QUOTING")
    print("  strength ~0.19  teacher targets (kappa=2)      -> obvious when zoomed")
    print("  strength ~0.10  model w/ teacher-init          -> visible when zoomed")
    print("  strength ~0.03  model in-domain, upgrade path  -> faint, needs the zoom")
    print("  strength ~0.01  isotropic floor                -> nothing to see")
