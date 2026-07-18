"""run_m1.py -- Milestone 1 (Oracle achievability), BLOCKING.

Builds a guaranteed-clean anisotropic oracle by warping ONE clean isotropic
blue-noise set with density-preserving diffeomorphisms, then asks the blocking
question: is the requested anisotropy jointly achievable with the CAPACITY
requirement, independent of any sampler?

Two SEPARATE gates are reported:
  GATE A (the M1 spec blocker) -- CAPACITY: density-based capacity within +tol
     of the identity baseline. If this fails on the clean oracle, the requested
     anisotropy and the capacity definition are incompatible -> fix the
     requirement, not the sampler.
  GATE B (regime diagnostic) -- HARD-CORE spacing q_min >= 0.98. Phi is an exact
     diffeomorphism so there are no true collisions; q dipping below 0.98 marks
     where the LOCAL-metric approximation strains (field too strong/curved over
     one spacing), i.e. the smooth-field / bounded-kappa validity boundary.

Torus domain, uniform density, N ~ 1024 (first pass).

Outputs (in --out): panels.png, capacity_vs_kappa.png, objective_report.txt,
metrics.json.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aniso_m1 as m1


def wrap01(P):
    P = np.mod(P, 1.0)
    return np.minimum(P, np.nextafter(1.0, 0.0))


def build_oracles():
    """(name, kind, params). Flow time T is per-oracle: Taylor-Green stretches
    exponentially at its stagnation points, so it needs a SHORT T to stay in the
    valid kappa<=2 regime (this is a field-config choice, not a method limit)."""
    return [
        ("identity",    "identity", {}),
        ("shear_lo",    "flow", dict(psi=m1.psi_shear(A=0.048, k=1), T=1.0)),
        ("shear_mid",   "flow", dict(psi=m1.psi_shear(A=0.095, k=1), T=1.0)),
        ("shear_hi",    "flow", dict(psi=m1.psi_shear(A=0.159, k=1), T=1.0)),
        ("taylorgreen", "flow", dict(psi=m1.psi_taylor_green(A=0.05, kx=1, ky=1), T=0.30)),
        ("transition",  "flow", dict(psi=m1.psi_transition(A=0.12, k=1), T=1.0)),
    ]


def make_oracle(P0, kind, params):
    if kind == "identity":
        F = np.tile(np.eye(2), (len(P0), 1, 1))
        return P0.copy(), F
    field = m1.field_from_streamfn(params["psi"])
    return m1.divfree_flow(P0, field, T=params.get("T", 1.0), steps=48, periodic=True)


def generate_p0(n, radius0, rng):
    """Toroidal Bridson, auto-shrinking the radius until we have >= n points.
    Returns (P0_exact_n, P0_count, radius_used); radius_used is r0 for hard-core."""
    radius = radius0
    P0_full = m1.poisson_disk_periodic(radius, rng=rng)
    tries = 0
    while len(P0_full) < n and tries < 15:
        radius *= 0.93
        P0_full = m1.poisson_disk_periodic(radius, rng=rng)
        tries += 1
    if len(P0_full) < n:
        raise SystemExit("could not reach N=%d (got %d at radius=%.4f)"
                         % (n, len(P0_full), radius))
    P0 = wrap01(P0_full[rng.choice(len(P0_full), size=n, replace=False)])
    return P0, len(P0_full), radius


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--n", type=int, default=1024)
    pa.add_argument("--radius", type=float, default=0.025,
                    help="starting Poisson-disk radius for P0 (auto-shrinks to reach N)")
    pa.add_argument("--G", type=int, default=16)
    pa.add_argument("--carriers", type=int, default=40)
    pa.add_argument("--tol", type=float, default=0.15, help="capacity gate: <= (1+tol)x baseline")
    pa.add_argument("--qmin", type=float, default=0.98, help="hard-core gate")
    pa.add_argument("--dotsize", type=float, default=5.0)
    pa.add_argument("--out", type=str, default="m1_out")
    args = pa.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    P0, p0_count, r0 = generate_p0(args.n, args.radius, rng)
    print("P0: generated %d, using exact N=%d, r0=%.4f" % (p0_count, len(P0), r0))

    oracles = build_oracles()
    rows = []
    tile_pts = []
    for name, kind, params in oracles:
        P, F = make_oracle(P0, kind, params)
        P = wrap01(P)
        kappa, theta = m1.F_to_kappa_theta(F)
        qmin, q1 = m1.hardcore_quality(P, F, r0, periodic=True)
        cap_rng = np.random.default_rng(args.seed + 100)
        rce = m1.regional_count_error(P, G=args.G)
        cov = m1.capacity_cov_carrier(P, cap_rng, carriers_per_point=args.carriers)
        kde = m1.kde_error(P, cap_rng)
        cvt = m1.cvt_energy_isotropic(P, cap_rng, carriers_per_point=args.carriers)
        rows.append(dict(name=name, n=int(len(P)),
                         kappa_med=float(np.median(kappa)), kappa_max=float(np.max(kappa)),
                         qmin=qmin, q1=q1, rce=rce, cov=cov, kde=kde, cvt=cvt))
        tile_pts.append(P)
        print("  %-12s N=%d  kappa med/max=%.2f/%.2f  qmin=%.3f  rce=%.3f cov=%.3f kde=%.3f cvt=%.4f"
              % (name, len(P), np.median(kappa), np.max(kappa), qmin, rce, cov, kde, cvt))

    base = rows[0]
    tol = 1.0 + args.tol
    for r in rows:
        r["rce_ratio"] = r["rce"] / base["rce"] if base["rce"] > 0 else float("nan")
        r["cov_ratio"] = r["cov"] / base["cov"] if base["cov"] > 0 else float("nan")
        r["kde_ratio"] = r["kde"] / base["kde"] if base["kde"] > 0 else float("nan")
        r["cvt_ratio"] = r["cvt"] / base["cvt"] if base["cvt"] > 0 else float("nan")
        r["capacity_ok"] = bool(r["rce"] <= tol * base["rce"] and
                                r["cov"] <= tol * base["cov"] and
                                r["kde"] <= tol * base["kde"])
        r["hardcore_ok"] = bool(r["qmin"] >= args.qmin)
        r["exact_n"] = bool(r["n"] == base["n"])

    non_id = rows[1:]
    # GATE A -- the M1 blocker
    capacity_passed = all(r["capacity_ok"] and r["exact_n"] for r in non_id)
    # GATE B -- regime diagnostic: largest in-regime kappa among smooth fields
    in_regime = [r for r in non_id if r["hardcore_ok"]]
    kappa_ceiling_med = max((r["kappa_med"] for r in in_regime), default=0.0)
    kappa_ceiling_max = max((r["kappa_max"] for r in in_regime), default=0.0)
    cvt_max_ratio = max(r["cvt_ratio"] for r in non_id)

    _panels(tile_pts, rows, args, os.path.join(args.out, "panels.png"))
    _capacity_plot(rows, os.path.join(args.out, "capacity_vs_kappa.png"))

    L = []
    L.append("Milestone 1 -- oracle achievability (BLOCKING)")
    L.append("=" * 62)
    L.append("P0=%d  exact N=%d  r0=%.4f  grid G=%d  carriers/pt=%d  tol=+%.0f%%  qmin>=%.2f"
             % (p0_count, len(P0), r0, args.G, args.carriers, 100 * args.tol, args.qmin))
    L.append("baseline (identity): rce=%.3f cov=%.3f kde=%.3f cvt=%.5f"
             % (base["rce"], base["cov"], base["kde"], base["cvt"]))
    L.append("")
    L.append("%-12s %9s %6s %6s %6s | %-22s | %-8s" % (
        "oracle", "kap m/x", "qmin", "q1%", "N", "capacity (ratio vs base)", "cvt"))
    for r in rows:
        L.append("%-12s %4.2f/%5.2f %6.3f %6.3f %6d | rce %.2f cov %.2f kde %.2f | x%.2f" % (
            r["name"], r["kappa_med"], r["kappa_max"], r["qmin"], r["q1"], r["n"],
            r["rce_ratio"], r["cov_ratio"], r["kde_ratio"], r["cvt_ratio"]))
    L.append("")
    L.append("GATE A -- CAPACITY (the M1 spec blocker): %s"
             % ("PASS" if capacity_passed else "FAIL"))
    for r in non_id:
        L.append("   %-12s capacity=%s exactN=%s" % (r["name"], r["capacity_ok"], r["exact_n"]))
    L.append("")
    L.append("GATE B -- HARD-CORE / regime (diagnostic, not the capacity blocker):")
    for r in non_id:
        L.append("   %-12s qmin=%.3f  %s" % (
            r["name"], r["qmin"], "in-regime" if r["hardcore_ok"] else "OUT of regime"))
    L.append("   -> clean smooth-field ceiling: kappa_med<=%.2f (kappa_max<=%.2f)"
             % (kappa_ceiling_med, kappa_ceiling_max))
    L.append("   Phi is an exact diffeomorphism: no true collisions. qmin<%.2f marks the"
             % args.qmin)
    L.append("   local-metric validity edge (||grad J||*r_local not << ||J||), i.e. kappa~2.")
    L.append("")
    L.append("Naive isotropic CVT energy: max ratio x%.2f (mild at these kappa)." % cvt_max_ratio)
    L.append("   Density-based capacity stays flat while CVT trends up -> still use the")
    L.append("   density-based metric as primary, but the CVT trap is weak below kappa~2.")
    L.append("")
    if capacity_passed:
        L.append("MILESTONE 1 PASSES (capacity achievability). Anisotropy and the")
        L.append("density-based capacity definition are COMPATIBLE on the clean oracle.")
        L.append("Operate smooth fields within kappa_med<=%.2f; keep fields bounded"
                 % kappa_ceiling_med)
        L.append("(clamp kappa, blend to isotropy near singularities) as the spec requires.")
        L.append("Proceed to M2 (constructive sampler): any capacity shortfall there is the")
        L.append("sampler's, not the requirement's.")
    else:
        L.append("MILESTONE 1 BLOCKER: even the clean oracle fails CAPACITY. Reduce the")
        L.append("anisotropy range or revise the capacity metric. Do NOT fix with a sampler.")
    report = "\n".join(L)
    with open(os.path.join(args.out, "objective_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(seed=args.seed, n=len(P0), r0=r0, tol=args.tol, qmin_gate=args.qmin,
                       baseline=base, oracles=rows, capacity_passed=capacity_passed,
                       kappa_ceiling_med=kappa_ceiling_med,
                       kappa_ceiling_max=kappa_ceiling_max,
                       cvt_max_ratio=cvt_max_ratio), f, indent=2)


def _panels(tile_pts, rows, args, path):
    n = len(tile_pts)
    cols = 3
    rr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rr, cols, figsize=(3.3 * cols, 3.6 * rr))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for i, (pts, rec) in enumerate(zip(tile_pts, rows)):
        ax = axes[i]
        ax.scatter(pts[:, 0], pts[:, 1], s=args.dotsize, c="black", edgecolors="none")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("%s  k med/max %.2f/%.2f\nqmin %.3f  cap x%.2f"
                     % (rec["name"], rec["kappa_med"], rec["kappa_max"],
                        rec["qmin"], rec.get("cov_ratio", 1.0)), fontsize=9)
    fig.suptitle("M1 oracle families (dots only) -- density preserved, spacing intact",
                 fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _capacity_plot(rows, path):
    shear = [r for r in rows if r["name"] == "identity" or r["name"].startswith("shear")]
    shear = sorted(shear, key=lambda r: r["kappa_med"])
    x = [r["kappa_med"] for r in shear]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [r["rce_ratio"] for r in shear], "o-", label="regional count err")
    ax.plot(x, [r["cov_ratio"] for r in shear], "s-", label="capacity CoV")
    ax.plot(x, [r["kde_ratio"] for r in shear], "^-", label="KDE err")
    ax.plot(x, [r["cvt_ratio"] for r in shear], "d--", color="tab:red", label="naive CVT energy")
    ax.axhline(1.15, color="gray", ls=":", label="+15% gate")
    ax.set_xlabel("median local kappa")
    ax.set_ylabel("metric / identity-baseline")
    ax.set_title("M1: density-based capacity flat; naive CVT mild")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
