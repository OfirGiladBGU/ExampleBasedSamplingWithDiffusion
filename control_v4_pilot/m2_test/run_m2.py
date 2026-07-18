"""run_m2.py -- Milestone 2 (Constructive sampler), BLOCKING.

Runs the Li-style anisotropic dart thrower (aniso_m2) and asks the M2 blocker:
can the sampler SIMULTANEOUSLY satisfy, on its own --
  * hard-core spacing with ZERO violations (acceptance invariant, not a penalty)
  * exact N
  * capacity close to the isotropic (kappa=1) teacher (bounded penalty)
  * commanded orientation on smooth fields
If not, the teacher family is not buildable -> STOP.

Convention: with the spec metric J = metric_M(theta,kappa), the forbidden
exclusion ellipse extends PERPENDICULAR to theta, so spacing elongates perp and
the nearest-neighbour concentration axis equals theta.

Orientation readout: orientation_bias() de-rotates each neighbourhood by its OWN
commanded theta(x) and then sums coherently over the whole tile. That pools all
neighbour pairs to beat down per-point noise while measuring the SYSTEMATIC
offset between realized and commanded orientation -- and it reduces exactly to
the coherent constant-field metric when theta is constant. (A per-point axis is
far too noisy for a varying field: its median floors near 30deg regardless.)

Reference: the kappa=1 run of the SAME sampler is the capacity baseline. Periodic
square, uniform density, N=1024. Multiple seeds per condition.

Outputs (in --out): panels.png, strength_vs_kappa.png, objective_report.txt,
metrics.json.
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import aniso_m2 as m2
import aniso_m1 as m1
import aniso_pilot as ap


def orientation_bias(points, theta_fn, h_factor=1.0, r_factor=3.0, periodic=True):
    """De-rotated coherent orientation readout for a (possibly varying) field.
    D = sum_{i,j} w_ij exp(2i(phi_ij - theta(x_i))) / sum w_ij.
      bias_deg   = 0.5*angle(D)  -- systematic realized-vs-commanded offset (~0 good)
      alignment  = |D|           -- coherence of realized orientation with commanded
    For a constant field this equals the coherent global axis-error metric."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    tree = cKDTree(pts, boxsize=1.0 if periodic else None)
    dnn, _ = tree.query(pts, k=2)
    nn = float(np.median(dnn[:, 1]))
    h, R = h_factor * nn, r_factor * nn
    nbrs = tree.query_ball_point(pts, R)
    th = np.asarray(theta_fn(pts), dtype=float)
    D = 0.0 + 0.0j
    W = 0.0
    for i in range(n):
        js = [j for j in nbrs[i] if j != i]
        if not js:
            continue
        dv = pts[js] - pts[i]
        if periodic:
            dv -= np.round(dv)
        dist = np.hypot(dv[:, 0], dv[:, 1])
        w = np.exp(-(dist / h) ** 2)
        phi = np.arctan2(dv[:, 1], dv[:, 0])
        D += np.sum(w * np.exp(2j * (phi - th[i])))
        W += float(w.sum())
    if W <= 0:
        return float("nan"), 0.0
    D /= W
    return float(np.rad2deg(0.5 * np.angle(D))), float(abs(D))


def wrap_pi(a):
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def build_conditions():
    return [dict(name="k1.00", kind="const", theta_deg=30.0, kappa=1.0),
            dict(name="k1.25", kind="const", theta_deg=30.0, kappa=1.25),
            dict(name="k1.50", kind="const", theta_deg=30.0, kappa=1.5),
            dict(name="k2.00", kind="const", theta_deg=30.0, kappa=2.0),
            dict(name="smooth", kind="smooth", theta_deg=0.0, kappa=1.5,
                 amp_deg=30.0, freq=1)]


def make_field(cond):
    if cond["kind"] == "const":
        return m2.const_field(cond["theta_deg"], cond["kappa"])
    th0 = np.deg2rad(cond["theta_deg"])
    amp = np.deg2rad(cond["amp_deg"])
    freq = cond["freq"]
    kappa = float(cond["kappa"])
    theta_fn = lambda X: th0 + amp * np.sin(2 * np.pi * freq * X[:, 1])
    kappa_fn = lambda X: np.full(len(X), kappa)
    return theta_fn, kappa_fn, kappa


def eval_once(P, cond, theta_fn, kappa_fn, s, cap_rng, args):
    qmin, q1, viol = m2.hardcore_field(P, theta_fn, kappa_fn, s)
    rce = m1.regional_count_error(P, G=args.G)
    cov = m1.capacity_cov_carrier(P, cap_rng, carriers_per_point=args.carriers)
    kde = m1.kde_error(P, cap_rng)
    orient_err = np.nan
    align = np.nan
    nnratio = ap.nn_vector_anisotropy(P, kk=1)["ratio"]
    if cond["kappa"] > 1.0:
        bias, align = orientation_bias(P, theta_fn)
        orient_err = abs(bias)
    return dict(qmin=qmin, q1=q1, viol=int(viol), rce=rce, cov=cov, kde=kde,
                orient_err=orient_err, align=align, nnratio=nnratio)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--n", type=int, default=1024)
    pa.add_argument("--seeds", type=int, default=6)
    pa.add_argument("--sfac", type=float, default=0.62)
    pa.add_argument("--G", type=int, default=16)
    pa.add_argument("--carriers", type=int, default=40)
    pa.add_argument("--tol", type=float, default=0.15)
    pa.add_argument("--orient_gate", type=float, default=15.0)
    pa.add_argument("--dotsize", type=float, default=5.0)
    pa.add_argument("--out", type=str, default="m2_out")
    args = pa.parse_args()

    os.makedirs(args.out, exist_ok=True)
    s0 = args.sfac / np.sqrt(args.n)
    conds = build_conditions()

    rows = []
    panel_pts = []
    for cond in conds:
        theta_fn, kappa_fn, kmax = make_field(cond)
        per = []
        first_P = None
        for si in range(args.seeds):
            rng = np.random.default_rng(1000 * args.seed + 31 * si + 7)
            P, s_used, att = m2.sample_exact_n(theta_fn, kappa_fn, s0, args.n, rng, kmax)
            cap_rng = np.random.default_rng(args.seed + 100)
            ev = eval_once(P, cond, theta_fn, kappa_fn, s_used, cap_rng, args)
            ev["reached"] = bool(len(P) == args.n)
            per.append(ev)
            if first_P is None:
                first_P = P
        panel_pts.append((cond, first_P))

        def agg(key):
            vals = [p[key] for p in per
                    if not (isinstance(p[key], float) and np.isnan(p[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        rows.append(dict(
            name=cond["name"], kind=cond["kind"], kappa=cond["kappa"],
            reached_all=all(p["reached"] for p in per),
            viol_max=max(p["viol"] for p in per),
            qmin_min=min(p["qmin"] for p in per),
            rce=agg("rce"), cov=agg("cov"), kde=agg("kde"),
            orient_err=agg("orient_err"), align=agg("align"), nnratio=agg("nnratio")))
        print("  %-7s kappa=%.2f reachedAll=%s viol=%d qmin=%.3f cov=%.3f orient_e=%.1f align=%.2f"
              % (cond["name"], cond["kappa"], rows[-1]["reached_all"], rows[-1]["viol_max"],
                 rows[-1]["qmin_min"], rows[-1]["cov"], rows[-1]["orient_err"], rows[-1]["align"]))

    base = next(r for r in rows if r["kappa"] == 1.0)
    tol = 1.0 + args.tol
    for r in rows:
        r["rce_ratio"] = r["rce"] / base["rce"] if base["rce"] > 0 else float("nan")
        r["cov_ratio"] = r["cov"] / base["cov"] if base["cov"] > 0 else float("nan")
        r["kde_ratio"] = r["kde"] / base["kde"] if base["kde"] > 0 else float("nan")
        r["capacity_ok"] = bool(r["rce"] <= tol * base["rce"] and
                                r["cov"] <= tol * base["cov"] and
                                r["kde"] <= tol * base["kde"])
        r["spacing_ok"] = bool(r["viol_max"] == 0 and r["qmin_min"] >= 0.98)
        r["exact_ok"] = bool(r["reached_all"])
        r["orient_ok"] = bool(r["kappa"] == 1.0 or
                              (r["orient_err"] == r["orient_err"] and
                               r["orient_err"] < args.orient_gate))
        r["pass"] = r["capacity_ok"] and r["spacing_ok"] and r["exact_ok"] and r["orient_ok"]

    graded = [r for r in rows if r["kappa"] > 1.0]
    passed = all(r["pass"] for r in graded)

    _panels(panel_pts, rows, args, os.path.join(args.out, "panels.png"))
    _strength_plot(rows, os.path.join(args.out, "strength_vs_kappa.png"))

    L = []
    L.append("Milestone 2 -- constructive anisotropic sampler (BLOCKING)")
    L.append("=" * 62)
    L.append("N=%d  seeds/cond=%d  s0=%.4f  carriers/pt=%d  tol=+%.0f%%  orient_gate=%.0f deg"
             % (args.n, args.seeds, s0, args.carriers, 100 * args.tol, args.orient_gate))
    L.append("baseline (kappa=1 sampler): rce=%.3f cov=%.3f kde=%.3f"
             % (base["rce"], base["cov"], base["kde"]))
    L.append("orient_e = de-rotated coherent bias (systematic realized-vs-commanded, deg)")
    L.append("")
    L.append("%-7s %5s %6s %6s %6s | %-22s | %8s %6s" % (
        "cond", "kappa", "N?", "viol", "qmin", "capacity (ratio vs k=1)", "orient_e", "align"))
    for r in rows:
        L.append("%-7s %5.2f %6s %6d %6.3f | rce %.2f cov %.2f kde %.2f | %7.1f %6.2f" % (
            r["name"], r["kappa"], "OK" if r["reached_all"] else "NO", r["viol_max"], r["qmin_min"],
            r["rce_ratio"], r["cov_ratio"], r["kde_ratio"], r["orient_err"], r["align"]))
    L.append("")
    L.append("PER-CONDITION gate (0-viol spacing & qmin>=0.98, exact N, capacity")
    L.append("within +%.0f%% of k=1, orientation bias < %.0f deg):" % (100 * args.tol, args.orient_gate))
    for r in graded:
        L.append("  %-7s  %s  (spacing=%s exactN=%s capacity=%s orient=%s)" % (
            r["name"], "PASS" if r["pass"] else "FAIL",
            r["spacing_ok"], r["exact_ok"], r["capacity_ok"], r["orient_ok"]))
    L.append("")
    if passed:
        L.append("MILESTONE 2 PASSES: the constructive sampler hits zero-violation hard-core")
        L.append("spacing, exact N, capacity within tolerance of the isotropic teacher, and")
        L.append("commanded orientation -- on constant AND smooth fields. The teacher family")
        L.append("is BUILDABLE. Proceed to M3 (multimodality).")
    else:
        L.append("MILESTONE 2 BLOCKER: the sampler cannot jointly satisfy spacing + exact N +")
        L.append("bounded capacity + commanded orientation. Inspect which sub-gate fails per")
        L.append("condition above. If capacity is the shortfall with spacing intact, try the")
        L.append("optional CCVT balanced-assignment refinement before declaring failure.")
    report = "\n".join(L)
    with open(os.path.join(args.out, "objective_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(n=args.n, seeds=args.seeds, s0=s0, tol=args.tol,
                       baseline=base, conditions=rows, passed=passed), f, indent=2)


def _panels(panel_pts, rows, args, path):
    n = len(panel_pts)
    cols = 3
    rr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rr, cols, figsize=(3.3 * cols, 3.6 * rr))
    axes = np.atleast_1d(axes).ravel()
    rowmap = {r["name"]: r for r in rows}
    for a in axes:
        a.axis("off")
    for i, (cond, P) in enumerate(panel_pts):
        ax = axes[i]
        r = rowmap[cond["name"]]
        ax.scatter(P[:, 0], P[:, 1], s=args.dotsize, c="black", edgecolors="none")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("%s k=%.2f  viol %d qmin %.2f\ncap x%.2f  orient_e %.1f"
                     % (cond["name"], cond["kappa"], r["viol_max"], r["qmin_min"],
                        r["cov_ratio"], r["orient_err"]), fontsize=9)
    fig.suptitle("M2 constructive sampler (dots only) -- hard-core spacing enforced",
                 fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _strength_plot(rows, path):
    c = sorted([r for r in rows if r["kind"] == "const"], key=lambda r: r["kappa"])
    x = [r["kappa"] for r in c]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [r["align"] for r in c], "o-", color="black", label="orientation alignment |D|")
    ax2 = ax.twinx()
    ax2.plot(x, [r["nnratio"] for r in c], "s--", color="tab:red", label="nn-vector ratio")
    ax2.set_ylabel("nn-vector ratio", color="tab:red")
    ax.set_xlabel("commanded kappa")
    ax.set_ylabel("orientation alignment |D|")
    ax.set_title("M2 sampler: realized anisotropy vs commanded kappa")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
