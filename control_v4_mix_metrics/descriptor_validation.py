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


def grad_cells(gray, G):
    """Mean gradient magnitude per grid cell.

    Used to stratify the analysis by how much CONTOUR a cell contains. Descriptors that describe
    alignment to structure (edge_align) can only mean anything where structure exists, so pooling
    flat and edge cells together dilutes exactly the signal that distinguishes a contour-following
    sampler from an isotropic one.
    """
    import descriptor_fields as _DF
    g = _DF.gradient_magnitude(gray)
    h, w = g.shape
    ys = np.linspace(0, h, G + 1).astype(int)
    xs = np.linspace(0, w, G + 1).astype(int)
    return np.array([[g[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].mean()
                      for i in range(G)] for j in range(G)])


def _load_points(target_dir, stem, min_points):
    """Exact coordinates when present, else PNG centroids -- matching precompute_descriptors."""
    npy = os.path.join(target_dir, stem + ".npy")
    if os.path.exists(npy):
        pts = PIO.load_points(npy)
        if len(pts) >= min_points:
            return pts
    png = os.path.join(target_dir, stem + ".png")
    if os.path.exists(png):
        pts = PIO.extract_centroids(png, n_points=None)
        if len(pts) >= min_points:
            return pts
    return None


def _one_icon(payload):
    """Return per-cell rows for every oracle of one icon, already normalised.

    Candidate descriptors are computed from POINTS here rather than read from disk, because they
    are deliberately not precomputed -- nothing is promoted into CONDITIONING_KEYS (and so nothing
    triggers a 70k-file recompute or a FiLM width change) until it has earned a place.
    """
    # NOT `p`: the loop below already binds `p` to the descriptor path, which would shadow this.
    pl = payload
    stem, src_path, root = pl["stem"], pl["src"], pl["root"]
    oracles, keys = pl["oracles"], pl["keys"]
    lo, hi, max_cells, seed = pl["lo"], pl["hi"], pl["max_cells"], pl["seed"]
    cand = pl["candidates"]
    try:
        if cand:
            import candidate_descriptors as CD
        out = {}
        rho = None
        grd = None
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
                gray = PIO.load_gray01(src_path)
                rho = np.clip(1.0 - gray, 0.0, 1.0)
                h, w = rho.shape
                ys = np.linspace(0, h, G + 1).astype(int)
                xs = np.linspace(0, w, G + 1).astype(int)
                rho = np.array([[rho[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].mean()
                                 for i in range(G)] for j in range(G)])
                grd = grad_cells(gray, G)
            feats = np.stack([(arr[i] - lo[k]) / (hi[k] - lo[k]) for i, k in enumerate(keys)])
            feats = np.clip(feats, 0.0, 1.0)             # the Dataset clips; saturation is legitimate
            finite = np.isfinite(feats).all(0) & valid
            if not finite.any():
                continue
            cols = [feats[i][finite] for i in range(len(keys))]
            if cand:
                # Same point set the stored descriptors were measured on: exact .npy when present,
                # and the SAME background filter. Measuring a candidate on a different point set
                # than the incumbents would make every comparison between them meaningless.
                pts = _load_points(os.path.join(root, f"target_{m}"), stem, pl["min_points"])
                cf = None
                if pts is not None:
                    if pl["drop_white"]:
                        gray_full = PIO.load_gray01(src_path)
                        pts, _ = PIO.drop_white_area_points(pts, gray_full,
                                                            threshold=pl["white_thr"])
                    cf = CD.candidate_fields(pts, G=G, window=pl["window"], k=pl["cand_k"])
                for c in cand:
                    if cf is None:
                        cols.append(np.full(int(finite.sum()), np.nan))
                    else:
                        v = np.where(cf["valid"], cf[c], np.nan)
                        cols.append(v[finite])
            cells = np.stack(cols, 1)                                          # (n, K + C) RAW cand
            r = np.stack([rho[finite], grd[finite]], 1)                        # (n, 2)
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
    # n_jobs has no effect on lbfgs since 1.8 and warns, so it is not passed.
    clf = LogisticRegression(max_iter=2000)
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


def stratified_confusion(X, y, grad, oracles, out, label, sub=400000, seed=0):
    """5.4c -- does oracle identity depend on how much CONTOUR a cell contains?

    The motivating asymmetry: `edge_align` separates gbn from all six other oracles and separates
    no other pair, yet gbn is the WORST-identified class per cell. Both can hold if gbn's identity
    is spatially localised -- contour alignment can only exist where a contour does, so in flat
    interior cells gbn, bnot and wvs are all just isotropic blue noise and are genuinely the same
    thing. Pooling flat and edge cells then dilutes gbn's signature with cells that carry none.

    If accuracy rises sharply with gradient, the flat-cell confusion is NOT a missing descriptor --
    it is real local equivalence, section 3's "same descriptors AND same arrangement". The response
    is to scope the claim (control is over contour behaviour where contours exist), not to add a
    descriptor or drop an oracle.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    print(f"\n{'=' * 100}")
    print(f"5.4c  CONFUSION vs LOCAL GRADIENT ({label})")
    print("=" * 100)
    rng = np.random.RandomState(seed)
    if len(X) > sub:
        sel = rng.choice(len(X), sub, replace=False)
        X, y, grad = X[sel], y[sel], grad[sel]
    edges = np.percentile(grad, [0, 33.3, 66.7, 100])
    names = ["flat (low 1/3)", "mid", "contour (top 1/3)"]
    print(f"  {'stratum':22s}{'gradient range':>22s}{'n':>10s}{'accuracy':>11s}"
          f"{'gbn diag':>10s}{'gbn->bnot':>11s}")
    res = {}
    gi = list(oracles).index("gbn") if "gbn" in oracles else None
    bi = list(oracles).index("bnot") if "bnot" in oracles else None
    for s in range(3):
        m = (grad >= edges[s]) & (grad <= edges[s + 1] if s == 2 else grad < edges[s + 1])
        if m.sum() < 2000 or len(np.unique(y[m])) < 2:
            print(f"  {names[s]:22s}{'(too few samples)':>22s}")
            continue
        Xtr, Xte, ytr, yte = train_test_split(X[m], y[m], test_size=0.3,
                                              random_state=seed, stratify=y[m])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        pred = clf.predict(sc.transform(Xte))
        acc = float((pred == yte).mean())
        cm = confusion_matrix(yte, pred, labels=list(range(len(oracles))))
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        gd = float(cmn[gi, gi]) if gi is not None else float("nan")
        gb = float(cmn[gi, bi]) if (gi is not None and bi is not None) else float("nan")
        print(f"  {names[s]:22s}{f'[{edges[s]:.4f}, {edges[s + 1]:.4f}]':>22s}"
              f"{int(m.sum()):10d}{acc:11.4f}{gd:10.3f}{gb:11.3f}")
        res[names[s]] = {"accuracy": acc, "gbn_diag": gd, "gbn_to_bnot": gb,
                         "n": int(m.sum())}
    out["stratified_" + label] = res
    if len(res) == 3:
        lo_a = res[names[0]]["accuracy"]
        hi_a = res[names[2]]["accuracy"]
        print(f"\n  accuracy flat -> contour: {lo_a:.3f} -> {hi_a:.3f}  ({hi_a - lo_a:+.3f})")
        if hi_a - lo_a > 0.05:
            print("  Identity is GRADIENT-DEPENDENT. Flat-cell confusion is local equivalence, not")
            print("  a missing descriptor: where there is no contour there is nothing to align to,")
            print("  and the oracles genuinely agree. Scope the claim rather than add a descriptor.")
        else:
            print("  Identity does NOT depend on gradient, so the confusion is not explained by")
            print("  flat cells -- it is a genuine descriptor gap. Adding one is justified.")
    return res


def ablation_report(X, y, oracles, keys, out, label, sub=400000, seed=0):
    """Leave-one-out: what does each descriptor actually contribute?

    Two costs are reported because they answer different questions. Classifier accuracy says how
    much a descriptor helps DISCRIMINATE; "pairs losing separation" says whether any oracle pair
    depends on it as its ONLY separator. A descriptor can be near-useless for accuracy and still be
    the sole thing keeping one pair apart -- dropping that one would be a mistake -- so the decision
    needs both columns, not just the accuracy delta.

    Subsampled: this refits K+1 times, and the ranking is what matters, not the third decimal.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    print(f"\n{'=' * 100}")
    print(f"5.4b  DESCRIPTOR ABLATION -- leave-one-out ({label})")
    print("=" * 100)
    rng = np.random.RandomState(seed)
    if len(X) > sub:
        sel = rng.choice(len(X), sub, replace=False)
        X, y = X[sel], y[sel]

    def acc_without(drop):
        cols = [i for i in range(len(keys)) if i != drop]
        Xtr, Xte, ytr, yte = train_test_split(X[:, cols], y, test_size=0.3,
                                              random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        return float((clf.predict(sc.transform(Xte)) == yte).mean())

    base = acc_without(-1)
    sep = out.get("separation_" + label, {})
    print(f"  {'removed':14s}{'accuracy':>10s}{'delta':>9s}{'pairs losing separation':>26s}")
    print(f"  {'(none)':14s}{base:10.4f}{'':>9s}{'':>26s}")
    rows = {}
    for d, k in enumerate(keys):
        a = acc_without(d)
        lost = []
        for pair, dd in sep.items():
            rest = [abs(v) for kk, v in dd.items() if kk != k and np.isfinite(v)]
            if max(rest, default=0.0) < D_THRESHOLD <= max(
                    [abs(v) for v in dd.values() if np.isfinite(v)], default=0.0):
                lost.append(pair)
        rows[k] = {"accuracy": a, "delta": a - base, "pairs_lost": lost}
        note = ", ".join(lost[:3]) if lost else "none"
        print(f"  {k:14s}{a:10.4f}{a - base:+9.4f}{note:>26s}")
    out["ablation_" + label] = {"base_accuracy": base, "per_descriptor": rows, "n": int(len(X))}
    dead = [k for k, v in rows.items() if abs(v["delta"]) < 0.005 and not v["pairs_lost"]]
    print()
    if dead:
        print(f"  CANDIDATES FOR REMOVAL (accuracy delta < 0.005 AND sole separator for no pair):")
        for k in dead:
            print(f"    {k}")
        print("  Removing one changes CONDITIONING_KEYS, the (K+1,G,G) arrays and the FiLM input")
        print("  width -- a descriptor recompute and a model-shape change, not just a config edit.")
    else:
        print("  every descriptor either moves accuracy or is the sole separator for some pair.")
    return rows


def gap_anatomy(X, keys, out, label, pairs, thr):
    """5.6b -- is a gap UNSAMPLED or UNREACHABLE? They need opposite responses.

    The occupancy threshold marks a bin empty at < thr samples, but a bin holding 1..thr-1 samples
    is a very different object from one holding exactly 0:

      thin (1..thr-1)  reachable, just undersampled -> a procedural sweep fills it.
      hard-empty (0)   not one cell in the whole dataset landed there. Either no available sampler
                       reaches it, or the combination is geometrically impossible (descriptors are
                       not independent -- e.g. a highly periodic set cannot also be highly
                       irregular). Generating data for an infeasible region is not possible, and
                       the honest response is to exclude it from the claimed control region.
    """
    print(f"\n{'=' * 100}")
    print(f"5.6b  GAP ANATOMY ({label}) -- unsampled vs unreachable")
    print("=" * 100)
    res = {}
    for a, b in pairs:
        i, j = keys.index(a), keys.index(b)
        m = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
        H, xe, ye = np.histogram2d(X[m, i], X[m, j], bins=COVERAGE_BINS, range=[[0, 1], [0, 1]])
        hard = (H == 0)
        thin = (H > 0) & (H < thr)
        print(f"\n  {a} x {b}")
        print(f"    bins {H.size}   occupied {int((H >= thr).sum())}   "
              f"thin(1..{thr - 1}) {int(thin.sum())}   HARD EMPTY(0) {int(hard.sum())}")
        if hard.any():
            hi, hj = np.where(hard)
            print(f"    hard-empty spans {a} in [{xe[hi.min()]:.2f}, {xe[hi.max() + 1]:.2f}]  "
                  f"x  {b} in [{ye[hj.min()]:.2f}, {ye[hj.max() + 1]:.2f}]")
            # Which corner? Names the shape of the missing sampler, or the impossibility.
            ci = "high" if hi.mean() > COVERAGE_BINS / 2 else "low"
            cj = "high" if hj.mean() > COVERAGE_BINS / 2 else "low"
            print(f"    concentrated at {ci} {a} + {cj} {b}")
        res[f"{a}|{b}"] = {"occupied": int((H >= thr).sum()), "thin": int(thin.sum()),
                           "hard_empty": int(hard.sum())}
    out["gap_anatomy_" + label] = res
    print("\n  THIN bins are sweep targets. HARD-EMPTY bins are the boundary of the reachable set:")
    print("  verify one is actually achievable before planning to generate data for it.")
    return res


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
    ap.add_argument("--candidates", default="",
                    help="comma list or 'all' from candidate_descriptors.CANDIDATE_KEYS. Computed "
                         "from points on the fly and EVALUATED ONLY -- nothing is written to disk "
                         "and CONDITIONING_KEYS is untouched.")
    ap.add_argument("--candidate-k", type=int, default=6,
                    help="neighbours per point for the candidates (6 = the hexatic convention)")
    ap.add_argument("--window", type=int, default=5, help="candidate window in cells")
    ap.add_argument("--min-points", type=int, default=256,
                    help="skip a target with fewer points than this when loading for candidates")
    ap.add_argument("--drop-white-points", action=argparse.BooleanOptionalAction, default=True,
                    help="must match precompute_descriptors.py, or candidates are measured on a "
                         "different point set than the stored descriptors")
    ap.add_argument("--white-threshold", type=float, default=255)
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

    candidates = []
    if args.candidates:
        import candidate_descriptors as CD
        candidates = (list(CD.CANDIDATE_KEYS) if args.candidates.strip().lower() == "all"
                      else [c.strip() for c in args.candidates.split(",") if c.strip()])
        bad = [c for c in candidates if c not in CD.CANDIDATE_KEYS]
        if bad:
            raise SystemExit(f"unknown candidate(s) {bad}; available: {list(CD.CANDIDATE_KEYS)}")
        print(f"candidates: {candidates}  (evaluated only -- NOT added to CONDITIONING_KEYS)")

    tasks = [{"stem": s, "src": src_map[s], "root": args.root, "oracles": oracles, "keys": keys,
              "lo": lo, "hi": hi, "max_cells": args.max_cells, "seed": args.seed,
              "candidates": candidates, "min_points": args.min_points,
              "drop_white": args.drop_white_points, "white_thr": args.white_threshold,
              "window": args.window, "cand_k": args.candidate_k}
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

    if candidates:
        # Candidates have no DESCRIPTOR_STATS entry, so their bounds come from this sample, using
        # the SAME percentiles the incumbents were normalised with -- otherwise a candidate would
        # be judged on a different scale than the descriptors it is being compared against.
        allc = np.concatenate([per_cell[m] for m in oracles])
        for c_i, c in enumerate(candidates):
            col = len(keys) + c_i
            v = allc[:, col]
            v = v[np.isfinite(v)]
            if v.size == 0:
                print(f"  WARNING: candidate '{c}' produced no finite values")
                continue
            c_lo, c_hi = np.percentile(v, 1.0), np.percentile(v, 99.0)
            if c_hi - c_lo < 1e-9:
                c_hi = c_lo + 1e-9
            print(f"  candidate '{c}': lo(p1)={c_lo:.4f} hi(p99)={c_hi:.4f} "
                  f"median={np.median(v):.4f}  ({(~np.isfinite(allc[:, col])).sum()} cells invalid)")
            for m in oracles:
                per_cell[m][:, col] = np.clip((per_cell[m][:, col] - c_lo) / (c_hi - c_lo), 0, 1)
        keys = list(keys) + list(candidates)
        # NaN cells (candidate invalid where the descriptor stack was valid) would break the
        # classifier; the field convention elsewhere is 0 for "nothing measurable here".
        for m in oracles:
            per_cell[m] = np.nan_to_num(per_cell[m], nan=0.0)

    per_icon = {m: np.stack([c.mean(0) for c in
                             np.split(per_cell[m], np.cumsum([len(x) for x in cells[m]])[:-1])])
                for m in oracles}
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
    aux = np.concatenate([per_rho[m] for m in oracles])       # (n, 2): rho, gradient
    rc, gc = aux[:, 0], aux[:, 1]
    yc = np.concatenate([np.full(len(per_cell[m]), i) for i, m in enumerate(oracles)])
    redundancy_table(Xc, rc, keys, out, "per_cell")
    clf_cell = classifier_report(Xc, yc, oracles, keys, out, "per_cell", seed=args.seed)
    stratified_confusion(Xc, yc, gc, oracles, out, "per_cell", seed=args.seed)

    Xi = np.concatenate([per_icon[m] for m in oracles])
    yi = np.concatenate([np.full(len(per_icon[m]), i) for i, m in enumerate(oracles)])
    classifier_report(Xi, yi, oracles, keys, out, "per_icon", seed=args.seed)

    ablation_report(Xc, yc, oracles, keys, out, "per_cell", seed=args.seed)
    _cov, clustered = coverage_report(Xc, keys, out, "per_cell")
    # Anatomy of the three sparsest joints -- the ones a sweep plan would target.
    sparsest = sorted(_cov.items(), key=lambda kv: kv[1]["occupied"])[:3]
    gap_anatomy(Xc, keys, out, "per_cell",
                [tuple(p.split("|")) for p, _ in sparsest],
                max(COVERAGE_MIN_COUNT, int(COVERAGE_MIN_FRAC * len(Xc))))

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
