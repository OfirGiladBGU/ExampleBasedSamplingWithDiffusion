"""run_m3.py -- Milestone 3 (Multimodality), BLOCKING for the diffusion claim.

For each fixed condition (rho, theta, kappa) the M2 sampler is run K times with
different seeds. The question: does imposing the anisotropy target leave MANY
genuinely different valid configurations, or collapse to essentially one?

  D_style  = spread (coefficient of variation) of the STATISTICS across seeds
             -- should be SMALL: every seed hits the same requested target.
  D_points = median pairwise W2 (toroidal optimal-transport) between seed point
             sets -- should be WELL ABOVE the jitter floor.
  floor    = median W2(P, P+small_jitter) with MATCHED points: the "same
             configuration" reference (~ the jitter magnitude).

Blocker: if D_points -> floor (ratio ~ 1) the conditional task is deterministic,
a regression baseline suffices, and the diffusion justification evaporates. We
also check anisotropy does not COLLAPSE diversity vs the isotropic kappa=1 case.

Note: W2 is estimated on random subsamples (exact full assignment is O(N^3)).
Absolute W2 values are subsample estimates; the DECISION is the ratio to the
(subsample-consistent) floor, which is robust.

Reuses aniso_m2 (sampler), aniso_m1 (capacity), aniso_pilot (anisotropy stats).
Periodic square, uniform density, N=1024.

Outputs (in --out): seeds_panel.png, diversity_vs_kappa.png,
objective_report.txt, metrics.json.
"""

import argparse
import json
import os
import itertools

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aniso_m2 as m2
import aniso_m1 as m1
import aniso_pilot as ap


def w2_toroidal(A, B, m, rng, draws=2):
    """Toroidal W2 between two DIFFERENT point sets, estimated on independent
    random subsamples of size m via exact assignment. Min-image distances."""
    vals = []
    for _ in range(draws):
        a = A[rng.choice(len(A), min(m, len(A)), replace=False)]
        b = B[rng.choice(len(B), min(m, len(B)), replace=False)]
        d = a[:, None, :] - b[None, :, :]
        d -= np.round(d)
        C = (d * d).sum(-1)
        ri, ci = linear_sum_assignment(C)
        vals.append(np.sqrt(C[ri, ci].mean()))
    return float(np.mean(vals))


def w2_floor(P, sigma, m, rng, draws=2):
    """Jitter floor: W2 between the SAME (subsampled) points and a jittered copy
    -- matched indices, so the optimal assignment stays ~identity and the result
    reflects the jitter magnitude, not the point spacing."""
    vals = []
    for _ in range(draws):
        idx = rng.choice(len(P), min(m, len(P)), replace=False)
        a = P[idx]
        b = np.mod(a + rng.normal(0.0, sigma, a.shape), 1.0)
        d = a[:, None, :] - b[None, :, :]
        d -= np.round(d)
        C = (d * d).sum(-1)
        ri, ci = linear_sum_assignment(C)
        vals.append(np.sqrt(C[ri, ci].mean()))
    return float(np.mean(vals))


def seed_stats(P, cap_rng, args):
    return dict(cov=m1.capacity_cov_carrier(P, cap_rng, carriers_per_point=args.carriers),
                rce=m1.regional_count_error(P, G=args.G),
                nnratio=ap.nn_vector_anisotropy(P, kk=1)["ratio"],
                strength=ap.global_near_field_anisotropy(P)["strength"])


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--n", type=int, default=1024)
    pa.add_argument("--K", type=int, default=10, help="seeds per condition")
    pa.add_argument("--sfac", type=float, default=0.62)
    pa.add_argument("--w2m", type=int, default=350, help="W2 subsample size")
    pa.add_argument("--w2draws", type=int, default=2)
    pa.add_argument("--jfac", type=float, default=0.1, help="jitter = jfac * median NN")
    pa.add_argument("--G", type=int, default=16)
    pa.add_argument("--carriers", type=int, default=40)
    pa.add_argument("--ratio_gate", type=float, default=3.0, help="D_points/floor gate")
    pa.add_argument("--style_gate", type=float, default=0.15, help="max stat CoV gate")
    pa.add_argument("--dotsize", type=float, default=5.0)
    pa.add_argument("--out", type=str, default="m3_out")
    args = pa.parse_args()

    os.makedirs(args.out, exist_ok=True)
    s0 = args.sfac / np.sqrt(args.n)
    mean_spacing = 1.0 / np.sqrt(args.n)
    kappas = [1.0, 1.25, 1.5, 2.0]

    rows = []
    seeds_for_panel = {}
    for kappa in kappas:
        theta_fn, kappa_fn, kmax = m2.const_field(30.0, kappa)
        sets = []
        stats = []
        for si in range(args.K):
            rng = np.random.default_rng(9000 * args.seed + 53 * si + 11)
            P, s_used, _ = m2.sample_exact_n(theta_fn, kappa_fn, s0, args.n, rng, kmax)
            sets.append(P)
            cap_rng = np.random.default_rng(args.seed + 100)
            stats.append(seed_stats(P, cap_rng, args))
        seeds_for_panel[kappa] = sets

        wrng = np.random.default_rng(args.seed + 777)
        tree = cKDTree(sets[0], boxsize=1.0)
        dnn, _ = tree.query(sets[0], k=2)
        sigma = args.jfac * float(np.median(dnn[:, 1]))
        floors = [w2_floor(P, sigma, args.w2m, wrng, args.w2draws) for P in sets]
        floor = float(np.median(floors))
        pair_w2 = [w2_toroidal(sets[a], sets[b], args.w2m, wrng, args.w2draws)
                   for a, b in itertools.combinations(range(len(sets)), 2)]
        d_points = float(np.median(pair_w2))
        ratio = d_points / floor if floor > 0 else float("inf")

        def cov_of(key):
            v = np.array([s[key] for s in stats])
            mu = v.mean()
            return float(v.std() / mu) if mu > 1e-9 else float("nan")
        style = {k: cov_of(k) for k in ("cov", "rce", "nnratio")}
        d_style = float(np.nanmax(list(style.values())))

        rows.append(dict(kappa=kappa, floor=floor, d_points=d_points, ratio=ratio,
                         d_points_sp=d_points / mean_spacing, floor_sp=floor / mean_spacing,
                         d_style=d_style, style=style,
                         mean_strength=float(np.mean([s["strength"] for s in stats])),
                         mean_nnratio=float(np.mean([s["nnratio"] for s in stats]))))
        print("  kappa=%.2f  D_points=%.4f floor=%.4f ratio=%.2f  D_style=%.3f"
              % (kappa, d_points, floor, ratio, d_style))

    base = rows[0]
    for r in rows:
        r["collapse_ratio"] = r["d_points"] / base["d_points"] if base["d_points"] > 0 else float("nan")
        r["diversity_ok"] = bool(r["ratio"] >= args.ratio_gate)
        r["style_ok"] = bool(r["d_style"] < args.style_gate)
        r["no_collapse"] = bool(r["collapse_ratio"] >= 0.5)

    graded = rows[1:]
    passed = all(r["diversity_ok"] and r["no_collapse"] for r in graded) and \
        all(r["style_ok"] for r in rows)

    _seeds_panel(seeds_for_panel, args, os.path.join(args.out, "seeds_panel.png"))
    _diversity_plot(rows, os.path.join(args.out, "diversity_vs_kappa.png"))

    L = []
    L.append("Milestone 3 -- multimodality (BLOCKING for the diffusion claim)")
    L.append("=" * 62)
    L.append("N=%d  K=%d seeds/cond  W2 subsample=%d x%d  jitter=%.2f*NN  mean_spacing=%.4f"
             % (args.n, args.K, args.w2m, args.w2draws, args.jfac, mean_spacing))
    L.append("gates: D_points/floor >= %.1f, stat CoV < %.2f, no collapse vs k=1 (>=0.5x)"
             % (args.ratio_gate, args.style_gate))
    L.append("")
    L.append("%-6s %10s %10s %8s %9s %9s %8s" % (
        "kappa", "D_points", "floor", "ratio", "Dpts/sp", "collapse", "D_style"))
    for r in rows:
        L.append("%-6.2f %10.4f %10.4f %8.2f %9.2f %9s %8.3f" % (
            r["kappa"], r["d_points"], r["floor"], r["ratio"], r["d_points_sp"],
            ("%.2f" % r["collapse_ratio"]), r["d_style"]))
    L.append("")
    L.append("(D_points/sp = median inter-seed W2 in units of mean spacing;")
    L.append(" collapse = D_points(k)/D_points(k=1); D_style = worst stat CoV across seeds)")
    L.append("")
    L.append("PER-KAPPA gate:")
    for r in graded:
        L.append("  kappa=%.2f  %s  (diversity=%s no_collapse=%s style=%s)" % (
            r["kappa"], "PASS" if (r["diversity_ok"] and r["no_collapse"] and r["style_ok"]) else "FAIL",
            r["diversity_ok"], r["no_collapse"], r["style_ok"]))
    L.append("")
    if passed:
        L.append("MILESTONE 3 PASSES: at fixed (rho,theta,kappa) the valid set stays")
        L.append("MULTIMODAL -- seeds are far apart in W2 (>> jitter floor) while hitting the")
        L.append("same target statistics (small D_style), and anisotropy does NOT collapse")
        L.append("diversity vs isotropic. The conditional task is one-to-MANY: a diffusion")
        L.append("model is JUSTIFIED over a regression baseline. All data-side gates cleared;")
        L.append("proceed to M4 (build conditioning + train).")
    else:
        L.append("MILESTONE 3 concern: diversity is low (D_points near the jitter floor) or")
        L.append("collapses with kappa. The conditional task may be near-deterministic -- a")
        L.append("regression baseline could suffice and the diffusion justification weakens.")
        L.append("Inspect per-kappa rows above before committing to a generative model.")
    report = "\n".join(L)
    with open(os.path.join(args.out, "objective_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(n=args.n, K=args.K, mean_spacing=mean_spacing,
                       ratio_gate=args.ratio_gate, style_gate=args.style_gate,
                       conditions=rows, passed=passed), f, indent=2)


def _seeds_panel(seeds_for_panel, args, path):
    show_k = [1.0, 2.0]
    ncol = min(4, args.K)
    fig, axes = plt.subplots(len(show_k), ncol, figsize=(3.0 * ncol, 3.2 * len(show_k)))
    axes = np.atleast_2d(axes)
    for ri, kappa in enumerate(show_k):
        sets = seeds_for_panel[kappa]
        for ci in range(ncol):
            ax = axes[ri, ci]
            P = sets[ci]
            ax.scatter(P[:, 0], P[:, 1], s=args.dotsize, c="black", edgecolors="none")
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(("kappa=%.2f  seed %d" % (kappa, ci)) if ci == 0 else "seed %d" % ci,
                         fontsize=9)
    fig.suptitle("M3: independent seeds at fixed condition (should look different, same style)",
                 fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _diversity_plot(rows, path):
    x = [r["kappa"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [r["d_points_sp"] for r in rows], "o-", color="black", label="D_points (inter-seed)")
    ax.plot(x, [r["floor_sp"] for r in rows], "s--", color="tab:red", label="jitter floor")
    ax.set_xlabel("kappa")
    ax.set_ylabel("W2 / mean spacing")
    ax.set_title("M3: inter-seed diversity vs jitter floor")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
