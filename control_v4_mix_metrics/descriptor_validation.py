"""Descriptor-only conditioning: the data-side go/no-go. No training, no model changes.

Implements `descriptor_only_data_validation.md` sections 5.2, 5.4 and 5.6, and re-emits 5.1/5.3/5.5
from the same source so every number in the report comes from one load.

WHY THIS READS THE PRECOMPUTED ARRAYS
m0_run.py measures descriptors from points, on the fly. That answered "are these oracles separable
in principle". This spec asks a different question -- "is the conditioning the model will actually
receive sufficient" -- so it reads `descriptors_<m>/<stem>.npy` and applies the SAME
DESCRIPTOR_STATS normalisation `OracleStippleDataset` applies. A discrepancy between the two is
itself a finding: it would mean the trainer sees something the analysis never validated.

TWO GRANULARITIES, BOTH REPORTED
  per-cell   Each valid grid cell is one sample. This is what conditioning IS -- the model is handed
             a (K, G, G) field and must produce the right local arrangement cell by cell. Collisions
             here are the ones that cause averaging.
  per-icon   Mean over valid cells. Comparable to M0 and to the plan's box plots, and far less
             sensitive to window noise.
They can disagree, and the disagreement is informative: separable per-icon but colliding per-cell
means the oracles differ in their global level while overlapping locally -- which under
spatially-varying control is a collision, not a separation.

    python control_v4_mix_metrics/descriptor_validation.py --limit 400
    python control_v4_mix_metrics/descriptor_validation.py --limit 0 --workers 16   # all icons
"""

import argparse
import itertools
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import point_io as PIO  # noqa: E402

DEFAULT_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_Oracles"
DEFAULT_ORACLES = "gbn,wvs,bnot,fs,ordered,white,jitgrid"
# Section 3: white noise and jittered grid stay in the MEASUREMENT set as low-regularity anchors,
# but they are scaffolding -- never a claimed style. The go/no-go is judged on the marketed set.
DEFAULT_SCAFFOLD = "white,jitgrid"
STATS_NAME = "DESCRIPTOR_STATS.json"

D_THRESHOLD = 0.8          # |Cohen's d| below this = not separated by that descriptor
CORR_THRESHOLD = 0.9       # section 5.5
COVERAGE_BINS = 20         # section 5.6 occupancy grid per descriptor pair
# A bin counts as occupied at >= this FRACTION of all samples (absolute floor COVERAGE_MIN_COUNT).
# Relative on purpose: an absolute count is meaningless across sample sizes. At 8.4M cells and 400
# bins the mean bin holds ~21k samples, so any fixed small count is cleared by the tails alone and
# every pair reports 1.000 "occupied" -- which measures "no bin is exactly empty", not coverage.
# 1e-4 is 1/25 of the uniform-density share (1/400 = 2.5e-3).
COVERAGE_MIN_FRAC = 1e-4
COVERAGE_MIN_COUNT = 5
# Fraction of samples allowed to sit in the densest 10% of bins before the joint counts as
# CLUSTERED. Occupancy alone cannot see this: every bin can be occupied while almost all the mass
# sits on a few pins, which is precisely the failure section 2 warns about.
CLUSTERED_MASS = 0.50


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load_stats(root, keys):
    """lo/hi per descriptor, exactly as the Dataset applies them."""
    p = os.path.join(root, STATS_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run precompute_descriptors.py --stage stats first")
    st = json.load(open(p))
    d = st.get("descriptors", {})
    missing = [k for k in keys if k not in d]
    if missing:
        raise SystemExit(f"{STATS_NAME} has no bounds for {missing}")
    return ({k: float(d[k]["lo"]) for k in keys},
            {k: float(d[k]["hi"]) for k in keys},
            st.get("keys", keys))


def rho_cells(src_path, G):
    """Mean ink per grid cell -- the tone the descriptor must NOT merely restate (5.5)."""
    gray = PIO.load_gray01(src_path)
    rho = np.clip(1.0 - gray, 0.0, 1.0)
    h, w = rho.shape
    ys = np.linspace(0, h, G + 1).astype(int)
    xs = np.linspace(0, w, G + 1).astype(int)
    return np.array([[rho[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].mean()
                      for i in range(G)] for j in range(G)])


def _one_icon(payload):
    """Return per-cell rows for every oracle of one icon, already normalised."""
    stem, src_path, root, oracles, keys, lo, hi, max_cells, seed = payload
    try:
        out = {}
        rho = None
        for m in oracles:
            p = os.path.join(root, f"descriptors_{m}", stem + ".npy")
            if not os.path.exists(p):
                continue
            arr = np.load(p).astype(np.float64)          # (K+1, G, G): K descriptors + valid
            valid = arr[-1] > 0.5
            if not valid.any():
                continue
            G = arr.shape[-1]
            if rho is None:
                rho = rho_cells(src_path, G)
            feats = np.stack([(arr[i] - lo[k]) / (hi[k] - lo[k]) for i, k in enumerate(keys)])
            feats = np.clip(feats, 0.0, 1.0)             # the Dataset clips; saturation is legitimate
            finite = np.isfinite(feats).all(0) & valid
            if not finite.any():
                continue
            cells = np.stack([feats[i][finite] for i in range(len(keys))], 1)   # (n, K)
            r = rho[finite]
            if max_cells and len(cells) > max_cells:
                # zlib.crc32, NOT hash(): Python randomises str hashing per PROCESS, and this runs
                # in a ProcessPoolExecutor, so hash() would give each worker a different subsample
                # and the whole report would be irreproducible between runs.
                key = zlib.crc32((stem + "|" + m).encode()) & 0x7FFFFFFF
                rng = np.random.RandomState((key + seed) % (2 ** 31))
                sel = rng.choice(len(cells), max_cells, replace=False)
                cells, r = cells[sel], r[sel]
            out[m] = (cells.astype(np.float32), r.astype(np.float32))
        return stem, out, None
    except Exception as exc:
        return stem, {}, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    """Unpaired Cohen's d. Per-cell samples are not paired across oracles (different cells
    survive the valid mask), so the paired form M0 used does not apply here."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / s) if s > 1e-12 else 0.0


def overlap_fraction(a, b, bins=64):
    """Histogram intersection in [0,1]. 1.0 = indistinguishable, 0.0 = disjoint."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    ha, _ = np.histogram(a, bins=bins, range=(0, 1), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(0, 1), density=False)
    return float(np.minimum(ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)).sum())


def separation_table(by_oracle, oracles, keys, out, label):
    """5.3 -- and the flag that matters: a pair separated by NO descriptor."""
    print(f"\n{'=' * 100}")
    print(f"5.3  PAIRWISE SEPARATION ({label})   |d| >= {D_THRESHOLD} counts as separated")
    print("=" * 100)
    header = f"  {'pair':22s}" + "".join(f"{k:>13s}" for k in keys) + f"{'best':>9s}"
    print(header)
    unseparated = []
    table = {}
    for a, b in itertools.combinations(oracles, 2):
        ds = []
        for i, k in enumerate(keys):
            ds.append(cohens_d(by_oracle[a][:, i], by_oracle[b][:, i]))
        best = max((abs(d) for d in ds if np.isfinite(d)), default=float("nan"))
        table[f"{a}|{b}"] = {k: ds[i] for i, k in enumerate(keys)}
        flag = "" if best >= D_THRESHOLD else "   <-- NOT SEPARATED"
        print(f"  {a + '|' + b:22s}" + "".join(f"{d:13.3f}" for d in ds) + f"{best:9.2f}{flag}")
        if not (best >= D_THRESHOLD):
            unseparated.append((a, b, best))
    out["separation_" + label] = table
    out["unseparated_" + label] = [[a, b, float(x)] for a, b, x in unseparated]
    return unseparated


def redundancy_table(X, rho, keys, out, label):
    """5.5 -- descriptor vs descriptor, and descriptor vs rho."""
    print(f"\n{'=' * 100}")
    print(f"5.5  REDUNDANCY ({label})   flag at |r| > {CORR_THRESHOLD}")
    print("=" * 100)
    names = list(keys) + ["rho"]
    M = np.column_stack([X, rho])
    print(f"  {'':14s}" + "".join(f"{n:>12s}" for n in names))
    flagged = []
    corr = np.full((len(names), len(names)), np.nan)
    for i in range(len(names)):
        row = ""
        for j in range(len(names)):
            m = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
            c = np.corrcoef(M[m, i], M[m, j])[0, 1] if m.sum() > 5 else np.nan
            corr[i, j] = c
            row += f"{c:12.3f}"
            if i < j and np.isfinite(c) and abs(c) > CORR_THRESHOLD:
                flagged.append((names[i], names[j], float(c)))
        print(f"  {names[i]:14s}{row}")
    out["redundancy_" + label] = {"names": names, "corr": corr.tolist(),
                                  "flagged": [[a, b, c] for a, b, c in flagged]}
    if flagged:
        print("\n  FLAGGED (one axis wearing two hats):")
        for a, b, c in flagged:
            note = "  <-- against rho: this is a tone readout, not arrangement" if b == "rho" else ""
            print(f"    {a} ~ {b}: r = {c:+.3f}{note}")
    else:
        print("\n  none above threshold.")
    return flagged


def classifier_report(X, y, oracles, keys, out, label, seed=0):
    """5.4 -- descriptor sufficiency. Under descriptor-only conditioning HIGH accuracy is GOOD.

    Logistic regression on purpose: the question is whether the descriptors LINEARLY carry oracle
    identity, i.e. whether a simple conditioning head could exploit them. A deep classifier could
    succeed on features the ControlNet's FiLM path cannot practically use, which would overstate
    sufficiency.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\n  [5.4 SKIPPED] scikit-learn not installed:  pip install scikit-learn")
        return None

    print(f"\n{'=' * 100}")
    print(f"5.4  DESCRIPTOR SUFFICIENCY CLASSIFIER ({label})   high accuracy = GOOD")
    print("=" * 100)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    # multi_class= is deprecated in sklearn 1.5 and removed in 1.7; lbfgs is multinomial by default.
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = float((pred == yte).mean())
    chance = 1.0 / len(oracles)
    print(f"  samples {len(X)}   features {len(keys)}   classes {len(oracles)}")
    print(f"  accuracy {acc:.4f}   (chance {chance:.4f})")

    cm = confusion_matrix(yte, pred, labels=list(range(len(oracles))))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    print(f"\n  confusion (row = true, normalised):")
    print(f"  {'':10s}" + "".join(f"{o:>10s}" for o in oracles))
    for i, o in enumerate(oracles):
        print(f"  {o:10s}" + "".join(f"{v:10.3f}" for v in cmn[i]))

    confused = []
    for i, j in itertools.combinations(range(len(oracles)), 2):
        m = (cmn[i, j] + cmn[j, i]) / 2.0
        if m > 0.10:
            confused.append((oracles[i], oracles[j], float(m)))
    confused.sort(key=lambda t: -t[2])
    if confused:
        print("\n  CONFUSED PAIRS (mean off-diagonal > 0.10) -- these are the collisions that")
        print("  need a new descriptor; under descriptor-only conditioning they average:")
        for a, b, v in confused:
            print(f"    {a} <-> {b}: {v:.3f}")
    else:
        print("\n  no pair confused above 0.10 -- descriptors determine oracle identity.")
    out["classifier_" + label] = {"accuracy": acc, "chance": chance,
                                  "labels": list(oracles), "confusion": cm.tolist(),
                                  "confused_pairs": [[a, b, v] for a, b, v in confused]}
    return acc, confused


def coverage_report(X, keys, out, label):
    """5.6 -- the VALID CONTROL REGION, and explicitly the gaps.

    Occupancy of a 2-D bin grid per descriptor pair. Marginals can each be full while the joint
    sits on a few pins, which is exactly the clustering the spec warns about, so this is measured
    jointly and never from the 1-D histograms.
    """
    n_tot = len(X)
    thr = max(COVERAGE_MIN_COUNT, int(COVERAGE_MIN_FRAC * n_tot))
    print(f"\n{'=' * 100}")
    print(f"5.6  JOINT COVERAGE / VALID CONTROL REGION ({label})")
    print(f"     {COVERAGE_BINS}x{COVERAGE_BINS} bins on [0,1]^2; occupied at >= {thr} samples "
          f"({COVERAGE_MIN_FRAC:g} of {n_tot}); uniform share would be "
          f"{n_tot // (COVERAGE_BINS ** 2)}/bin")
    print("=" * 100)
    print(f"  {'descriptor pair':30s}{'occupied':>10s}{'empty run':>11s}{'mass in top 10% bins':>22s}")
    cov = {}
    worst, clustered = [], []
    for i, j in itertools.combinations(range(len(keys)), 2):
        m = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
        H, _, _ = np.histogram2d(X[m, i], X[m, j], bins=COVERAGE_BINS, range=[[0, 1], [0, 1]])
        occ = (H >= thr)
        frac = float(occ.mean())
        # Concentration, independent of the occupancy threshold: if the densest tenth of the bins
        # holds most of the samples, the joint is a few pins no matter how many bins are non-empty.
        flat = np.sort(H.ravel())[::-1]
        k = max(1, len(flat) // 10)
        mass = float(flat[:k].sum() / max(flat.sum(), 1))
        run = 0
        for line in list(occ) + list(occ.T):
            c = 0
            for v in line:
                c = 0 if v else c + 1
                run = max(run, c)
        cov[f"{keys[i]}|{keys[j]}"] = {"occupied": frac, "largest_empty_run": int(run),
                                       "mass_top10pct_bins": mass}
        flag = "  <-- CLUSTERED" if mass > CLUSTERED_MASS else ""
        print(f"  {keys[i] + ' x ' + keys[j]:30s}{frac:10.3f}{run:11d}{mass:22.3f}{flag}")
        worst.append((frac, keys[i], keys[j]))
        if mass > CLUSTERED_MASS:
            clustered.append(f"{keys[i]}|{keys[j]}")
    worst.sort()
    out["coverage_" + label] = cov
    out["clustered_" + label] = clustered
    print(f"\n  sparsest pairs (fill these with procedural sweeps, section 6):")
    for frac, a, b in worst[:3]:
        print(f"    {a} x {b}: {frac:.3f} occupied")
    if clustered:
        print(f"  CLUSTERED (>{CLUSTERED_MASS:g} of mass in the densest 10% of bins): "
              + ", ".join(clustered))
    return cov, clustered


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def make_plots(per_cell, per_icon, oracles, keys, out_dir):
    """5.1 box plots and 5.2 the pairwise scatter matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [plots SKIPPED] matplotlib not installed")
        return
    os.makedirs(out_dir, exist_ok=True)

    # 5.1 -- one panel per descriptor, one box per oracle
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (i, k) in zip(axes, enumerate(keys)):
        data = [per_icon[o][:, i] for o in oracles]
        try:                     # matplotlib >= 3.9 renamed labels -> tick_labels
            ax.boxplot(data, tick_labels=list(oracles), showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=list(oracles), showfliers=False)
        ax.set_title(k)
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("5.1  per-descriptor distributions (per-icon means, normalised)")
    fig.tight_layout()
    p = os.path.join(out_dir, "5_1_box_plots.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"  wrote {p}")

    # 5.2 -- joint coverage. Marginals lie; this is the plot that shows clustering.
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(oracles), 10)))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                for oi, o in enumerate(oracles):
                    ax.hist(per_cell[o][:, i], bins=40, range=(0, 1), histtype="step",
                            color=colors[oi], label=o if i == 0 else None)
                ax.set_yticks([])
            else:
                for oi, o in enumerate(oracles):
                    d = per_cell[o]
                    sel = slice(None, None, max(1, len(d) // 1500))
                    ax.scatter(d[sel, j], d[sel, i], s=1.5, alpha=0.25,
                               color=colors[oi], linewidths=0)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            if i == n - 1:
                ax.set_xlabel(keys[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(keys[i], fontsize=8)
            ax.tick_params(labelsize=6)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[oi], label=o)
               for oi, o in enumerate(oracles)]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle("5.2  pairwise joint coverage (per-cell, normalised) -- look for GAPS, not spread")
    fig.tight_layout()
    p = os.path.join(out_dir, "5_2_scatter_matrix.png")
    fig.savefig(p, dpi=100)
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--source", default=None, help="default: <root>/source")
    ap.add_argument("--oracles", default=DEFAULT_ORACLES)
    ap.add_argument("--scaffold", default=DEFAULT_SCAFFOLD,
                    help="measured but not part of the marketed set (section 3)")
    ap.add_argument("--limit", type=int, default=400, help="icons; 0 = all")
    ap.add_argument("--max-cells", type=int, default=120,
                    help="cells sampled per (icon, oracle); keeps the scatter matrix tractable")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: <root>/../descriptor_validation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    oracles = [m.strip() for m in args.oracles.split(",") if m.strip()]
    scaffold = {m.strip() for m in args.scaffold.split(",") if m.strip()}
    marketed = [m for m in oracles if m not in scaffold]
    source = args.source or os.path.join(args.root, "source")
    out_dir = args.out or os.path.join(os.path.dirname(args.root.rstrip("/")),
                                       "descriptor_validation")

    import descriptor_fields as DF
    keys = list(DF.CONDITIONING_KEYS)
    lo, hi, stats_keys = load_stats(args.root, keys)
    if list(stats_keys) != keys:
        print(f"  WARNING: {STATS_NAME} keys {list(stats_keys)} != CONDITIONING_KEYS {keys}")

    src_map = PIO.stem_map(source)
    stems = sorted(src_map)
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"no source images under {source}")

    print(f"root      : {args.root}")
    print(f"icons     : {len(stems)}   oracles: {oracles}")
    print(f"marketed  : {marketed}   scaffold (measured, not claimed): {sorted(scaffold)}")
    print(f"descriptors: {keys}")
    print(f"out       : {out_dir}")

    tasks = [(s, src_map[s], args.root, oracles, keys, lo, hi, args.max_cells, args.seed)
             for s in stems]
    cells = defaultdict(list)
    rhos = defaultdict(list)
    icons = defaultdict(list)
    errors = []
    t0 = time.time()
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one_icon, t) for t in tasks]
            for n, fut in enumerate(as_completed(futs), 1):
                stem, res, err = fut.result()
                if err:
                    errors.append((stem, err))
                for m, (c, r) in res.items():
                    cells[m].append(c)
                    rhos[m].append(r)
                    icons[m].append(c.mean(0))
                if n % 200 == 0 or n == len(tasks):
                    print(f"  loaded {n}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    else:
        for n, t in enumerate(tasks, 1):
            stem, res, err = _one_icon(t)
            if err:
                errors.append((stem, err))
            for m, (c, r) in res.items():
                cells[m].append(c)
                rhos[m].append(r)
                icons[m].append(c.mean(0))

    present = [m for m in oracles if cells.get(m)]
    missing = [m for m in oracles if m not in present]
    if missing:
        print(f"\n  NOTE: no descriptor arrays for {missing} -- excluded from the analysis.")
    oracles = present
    marketed = [m for m in marketed if m in present]
    per_cell = {m: np.concatenate(cells[m]) for m in oracles}
    per_rho = {m: np.concatenate(rhos[m]) for m in oracles}
    per_icon = {m: np.stack(icons[m]) for m in oracles}
    print(f"\n  loaded {sum(len(v) for v in per_cell.values())} cells / "
          f"{sum(len(v) for v in per_icon.values())} (icon, oracle) pairs in {time.time() - t0:.0f}s")
    if errors:
        print(f"  ERRORS on {len(errors)} icons (first 5):")
        for s, e in errors[:5]:
            print(f"    {s}: {e}")

    out = {"root": args.root, "oracles": oracles, "marketed": marketed, "keys": keys,
           "n_icons": len(stems), "max_cells": args.max_cells}

    unsep_cell = separation_table(per_cell, oracles, keys, out, "per_cell")
    unsep_icon = separation_table(per_icon, oracles, keys, out, "per_icon")

    Xc = np.concatenate([per_cell[m] for m in oracles])
    rc = np.concatenate([per_rho[m] for m in oracles])
    yc = np.concatenate([np.full(len(per_cell[m]), i) for i, m in enumerate(oracles)])
    redundancy_table(Xc, rc, keys, out, "per_cell")
    clf_cell = classifier_report(Xc, yc, oracles, keys, out, "per_cell", seed=args.seed)

    Xi = np.concatenate([per_icon[m] for m in oracles])
    yi = np.concatenate([np.full(len(per_icon[m]), i) for i, m in enumerate(oracles)])
    classifier_report(Xi, yi, oracles, keys, out, "per_icon", seed=args.seed)

    _cov, clustered = coverage_report(Xc, keys, out, "per_cell")

    os.makedirs(out_dir, exist_ok=True)
    make_plots(per_cell, per_icon, oracles, keys, out_dir)

    # -------------------------------------------------------------- section 7
    print(f"\n{'=' * 100}")
    print("7  GO / NO-GO")
    print("=" * 100)
    checks = []
    mk = [p for p in unsep_cell if p[0] in marketed and p[1] in marketed]
    checks.append(("1. every MARKETED pair separated by >=1 descriptor (per-cell)", not mk,
                   "" if not mk else "collisions: " + ", ".join(f"{a}|{b}" for a, b, _ in mk)))
    allp = [p for p in unsep_cell]
    checks.append(("   (incl. scaffold oracles)", not allp,
                   "" if not allp else "collisions: " + ", ".join(f"{a}|{b}" for a, b, _ in allp)))
    red = out["redundancy_per_cell"]["flagged"]
    checks.append(("3. no descriptor redundant with another or with rho", not red,
                   "" if not red else ", ".join(f"{a}~{b} r={c:+.2f}" for a, b, c in red)))
    cov = out["coverage_per_cell"]
    thin = [k for k, v in cov.items() if v["occupied"] < 0.15]
    checks.append(("2a. descriptor space covered (occupancy)", not thin,
                   "" if not thin else f"{len(thin)} pair(s) below 0.15 occupancy: "
                                       + ", ".join(thin[:4])))
    checks.append(("2b. descriptor space not CLUSTERED (mass concentration)", not clustered,
                   "" if not clustered else ", ".join(clustered[:4])))
    if clf_cell:
        acc, confused = clf_cell
        cm = [c for c in confused if c[0] in marketed and c[1] in marketed]
        checks.append((f"4. classifier sufficiency (acc {acc:.3f})", not cm,
                       "" if not cm else "confused: " + ", ".join(f"{a}<->{b}" for a, b, _ in cm)))
    else:
        # A skipped check is NOT a pass. Section 7 requirement 1 cites 5.3 AND 5.4, and the
        # classifier is the one that measures joint sufficiency -- 5.3 only ever looks at one
        # descriptor at a time, so it cannot see a pair that is separable on no single axis but
        # separable on a combination, nor rank how ambiguous a "separated" pair still is.
        checks.append(("4. classifier sufficiency (5.4)", False,
                       "NOT RUN -- scikit-learn missing. This blocks a GO; install it and re-run."))
    for name, ok, note in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
        if note:
            print(f"          {note}")
    weakest = min(((max(abs(v) for v in d.values() if np.isfinite(v)), p)
                   for p, d in out["separation_per_cell"].items()
                   if p.split("|")[0] in marketed and p.split("|")[1] in marketed),
                  default=(float("nan"), "-"))
    print(f"\n  weakest MARKETED pair (per-cell): {weakest[1]} at |d| = {weakest[0]:.2f}")
    print(f"  |d| = 0.8 still leaves ~69% distribution overlap, 1.2 ~ 55%, 2.0 ~ 32%. Passing the")
    print(f"  threshold is not the same as being unambiguous -- 5.4 is what settles that.")
    print("\n  Not checked here (require a human): 4. anchors visually distinct dots-only (5.7 --")
    print("  use m0_run.py's panels), and the section 6 sweep plan for each gap.")

    p = os.path.join(out_dir, "descriptor_validation.json")
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\n  wrote {p}")
    return 0 if all(c[1] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
