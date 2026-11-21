#!/usr/bin/env python3
"""Does the BNOT solver fix (SuiteSparse QR instead of Eigen SparseQR) change the point sets?

Question (user, 2026-11-18): after the fix, do the BNOT exemplars have to be regenerated and the
Ours-BNOT models retrained? Test: a random sample of the stored training exemplars of each dataset is
regenerated with the OLD binary (Eigen, build.eigen_20261118) and with the FIXED binary, same inputs
and settings (seed 7, max_iters 25, max_newton_iters 50, epsilon 1.0, invert), and compared
  old vs fixed   the effect of the fix itself
  old vs stored  that the test reproduces the stored exemplar (i.e. uses the real settings)
by the distance from each point to the nearest point of the other set, in point spacings
(1 / sqrt(N) of the unit square), and by the paper metrics of both sets.

Steps (driven by bnot_fix_validation.sbatch, one array task per dataset):
    stage    pick the sample (seeded), copy its inputs to <out>/<dataset>/data
    choose   pick the input size whose fixed-build result reproduces the stored exemplar
    compare  distances + metrics -> <out>/<dataset>/result.json
    summary  all datasets -> <out>/summary.md
"""
import argparse
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

TRAIN = Path("/groups/asharf_group/ofirgila/ControlNet/training")
OUT = Path("experiments/outputs/bnot_fix_validation")
DATASETS = {  # name: (folder, points, sample size)
    "Icons-50": ("Icons-50_1024_BNOT", 1024, 60),
    "CelebA-5K": ("CelebA-5K_1024_BNOT", 1024, 40),
    "ShapeNet2D-4K": ("ShapeNet2D-4K_1600_BNOT", 1600, 40),
    "Anime-Chibi": ("Anime-Chibi_1600_BNOT", 1600, 40),
}
CANDIDATES = ["none", "512 512", "448 448"]  # --image_size candidates, tried on the first images
SEED = 0
METRICS = [("M1_v1_cvt_energy", "CVT"), ("M1_v2_hot_energy", "HOT"), ("M2_v1_capacity_delta_c", "cap dc"),
           ("M2_v2_centroid_displacement", "centroid"), ("M3_emd_distance", "W2"),
           ("M5_spatial_measure_rho_mean", "rho_s")]


def nn_spacings(a, b, n):
    from scipy.spatial import cKDTree
    return cKDTree(b).query(a)[0] * np.sqrt(n)


def stage(name):
    folder, n, k = DATASETS[name]
    root = TRAIN / folder
    use_original = (root / "original").is_dir()
    inp = root / ("original" if use_original else "source")
    rels = sorted(p.relative_to(inp) for p in inp.rglob("*.png")
                  if (root / "target" / p.relative_to(inp)).with_suffix(".npy").exists())
    rng = np.random.RandomState(SEED)
    picks = sorted(rng.choice(len(rels), size=min(k, len(rels)), replace=False))
    stems = set()
    data = OUT / name / "data"
    dst = data / ("original" if use_original else "source")
    chosen = []
    for i in picks:
        rel = rels[i]
        if rel.stem in stems:  # the scorer needs unique stems
            continue
        stems.add(rel.stem)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inp / rel, dst / rel)
        chosen.append(rel.as_posix())
    (OUT / name / "sample.json").write_text(json.dumps(
        {"dataset": folder, "points": n, "input": "original" if use_original else "source", "images": chosen}, indent=1))
    print(f"{name}: {len(chosen)} of {len(rels)} images staged from {inp} ({n} points)")


def choose(name):
    folder, n, _ = DATASETS[name]
    info = json.loads((OUT / name / "sample.json").read_text())
    best = None
    for c, cand in enumerate(CANDIDATES):
        dists = []
        for rel in info["images"][:2]:
            new = (OUT / name / "data" / f"cand{c}" / rel).with_suffix(".npy")
            stored = (TRAIN / folder / "target" / rel).with_suffix(".npy")
            if new.exists():
                dists.append(float(nn_spacings(np.load(new), np.load(stored), n).max()))
        worst = max(dists) if len(dists) == 2 else float("inf")
        print(f"{name}: image_size {cand}: max distance to the stored exemplar {worst:.3g} spacings")
        if best is None or worst < best[1]:
            best = (cand, worst)
    (OUT / name / "chosen.txt").write_text(best[0] + "\n")
    info["image_size"] = best[0]
    info["discovery_max_distance"] = best[1]
    (OUT / name / "sample.json").write_text(json.dumps(info, indent=1))
    print(f"{name}: chosen image_size = {best[0]}")


def stat(path, key):
    import re
    if not path.exists():
        return float("nan")
    m = re.search(rf"^{key}:\s*([-0-9.eE+]+)", path.read_text(), re.M)
    return float(m.group(1)) if m else float("nan")


def compare(name):
    folder, n, _ = DATASETS[name]
    info = json.loads((OUT / name / "sample.json").read_text())
    data = OUT / name / "data"
    rows = []
    for rel in info["images"]:
        old = (data / "target_old" / rel).with_suffix(".npy")
        new = (data / "target_new" / rel).with_suffix(".npy")
        stored = (TRAIN / folder / "target" / rel).with_suffix(".npy")
        row = {"image": rel, "old": old.exists(), "new": new.exists()}
        if old.exists() and new.exists():
            a, b, s = np.load(old), np.load(new), np.load(stored)
            row["same_count"] = bool(len(a) == len(b))
            d = nn_spacings(b, a, n)
            row.update(old_new_max=float(d.max()), old_new_median=float(np.median(d)),
                       old_stored_max=float(nn_spacings(a, s, n).max()),
                       new_stored_max=float(nn_spacings(b, s, n).max()),
                       iters_old=stat(old.with_suffix(".txt"), "iterations"),
                       iters_new=stat(new.with_suffix(".txt"), "iterations"),
                       cap_old=stat(old.with_suffix(".txt"), "rms_rel_capacity_error"),
                       cap_new=stat(new.with_suffix(".txt"), "rms_rel_capacity_error"))
        rows.append(row)
    score = OUT / name / "score"
    js = {t: {p.stem: json.loads(p.read_text()) for p in (score / f"{t}_json").glob("*.json")} for t in ("old", "new")}
    metrics = {}
    for key, label in METRICS:
        stems = sorted(s for s in set(js["old"]) & set(js["new"]) if key in js["old"][s] and key in js["new"][s])
        if stems:
            x = np.array([js["old"][s][key] for s in stems])
            y = np.array([js["new"][s][key] for s in stems])
            rel = np.abs(y - x) / np.abs(x)
            metrics[label] = {"images": len(stems), "mean_old": float(x.mean()), "mean_new": float(y.mean()),
                              "max_rel_diff": float(rel.max()), "mean_change": float((y.mean() - x.mean()) / x.mean())}
    (OUT / name / "result.json").write_text(json.dumps({"info": info, "rows": rows, "metrics": metrics}, indent=1))
    done = [r for r in rows if "old_new_max" in r]
    print(f"{name}: {len(done)} of {len(rows)} images compared; old vs fixed max {max(r['old_new_max'] for r in done):.3g} spacings")


def summary():
    lines = ["# BNOT solver fix: do the point sets change?", "",
             "A random sample of the stored training exemplars of each dataset, regenerated with the old",
             "binary (Eigen SparseQR) and with the fixed binary (SuiteSparse QR), same inputs and settings",
             "(seed 7, max_iters 25, max_newton_iters 50, epsilon 1.0). Distances are from each point to the",
             "nearest point of the other set, in point spacings (1 / sqrt(N)); 0 = identical.", "",
             "| dataset | points | images compared | old vs fixed: median over images of the max distance | "
             "old vs fixed: largest distance | images above 0.001 / 0.01 spacings | "
             "old vs stored exemplar: median / largest | input size used |",
             "|---|---|---|---|---|---|---|---|"]
    detail, mlines = [], []
    for name, (folder, n, _) in DATASETS.items():
        f = OUT / name / "result.json"
        if not f.exists():
            lines.append(f"| {name} | {n} | not finished | | | | | |")
            continue
        r = json.loads(f.read_text())
        rows = [x for x in r["rows"] if "old_new_max" in x]
        miss = [x["image"] for x in r["rows"] if "old_new_max" not in x]
        on = [x["old_new_max"] for x in rows]
        os_ = [x["old_stored_max"] for x in rows]
        lines.append(f"| {name} | {n} | {len(rows)} of {len(r['rows'])} | {statistics.median(on):.2g} | {max(on):.2g} | "
                     f"{sum(v > 1e-3 for v in on)} / {sum(v > 1e-2 for v in on)} | "
                     f"{statistics.median(os_):.2g} / {max(os_):.2g} | {r['info'].get('image_size')} |")
        for x in sorted(rows, key=lambda v: -v["old_new_max"])[:3]:
            detail.append(f"- {name} `{x['image']}`: old vs fixed max {x['old_new_max']:.2g} (median {x['old_new_median']:.2g}); "
                          f"outer iterations {x['iters_old']:.0f} / {x['iters_new']:.0f}; "
                          f"capacity error {x['cap_old']:.1e} / {x['cap_new']:.1e}; old vs stored {x['old_stored_max']:.2g}")
        if miss:
            detail.append(f"- {name}: no result within the time limit for " + ", ".join(f"`{m}`" for m in miss))
        for label, m in r["metrics"].items():
            mlines.append(f"| {name} | {label} | {m['images']} | {m['mean_old']:.6g} | {m['mean_new']:.6g} | "
                          f"{100 * m['mean_change']:+.4f}% | {100 * m['max_rel_diff']:.4f}% |")
    lines += ["", "## The three images with the largest difference per dataset", ""] + detail
    lines += ["", "## Paper metrics of the two point sets (same scorer as the tables)", "",
              "| dataset | metric | images | mean, old binary | mean, fixed binary | change of the mean | "
              "largest per-image difference |", "|---|---|---|---|---|---|---|"] + mlines
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["stage", "choose", "compare", "summary"])
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    a = ap.parse_args()
    {"stage": stage, "choose": choose, "compare": compare}.get(a.step, lambda _: summary())(a.dataset)
