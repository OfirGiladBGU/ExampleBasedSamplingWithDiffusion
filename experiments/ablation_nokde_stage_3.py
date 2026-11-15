#!/usr/bin/env python3
"""E8 (KDE-loss ablation), stage 3: aggregate and compare with the full model.

  * metrics_avg.json per results dir (ablation_advance_metrics_stage_3.py), i.e. the noKDE curves
    over epochs in the Table 2 format;
  * a paired comparison at epoch 5000, same 50 images and same noise: sdedit_noKDE vs sdedit_seeded
    (the paper's full checkpoint), mean per metric, change in %, Wilcoxon signed-rank p-value;
  * Table 2's stored (unseeded) sdedit row at epoch 5000 for reference.

Writes OUTPUT_DIR/summary.md.

    python experiments/ablation_nokde_stage_3.py
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_nokde_stage_1 import OUTPUT_DIR  # noqa: E402

TABLE2_SDEDIT = "experiments/outputs/ablation_advance_metrics_e500_b50_1024/sdedit/metrics_avg.json"
NOKDE, REFERENCE = "sdedit_noKDE", "sdedit_seeded"
EPOCH = "5000"
METRICS = [  # key, label, lower is better
    ("M1_v1_cvt_energy", "CVT", True),
    ("M1_v2_hot_energy", "HOT", True),
    ("M2_v1_capacity_delta_c", "δc", True),
    ("M2_v2_centroid_displacement", "centroid D", True),
    ("M2_v3_power_cell_cap_cv", "power-cell CV", True),
    ("M3_emd_distance", "EMD", True),
    ("M4_sinkhorn_ot_cost", "Sinkhorn", True),
    ("M5_spatial_measure_rho_mean", "ρs", False),
]


def load_epoch(out, results_dir, epoch):
    d = Path(out) / results_dir / f"epoch_{epoch}_json"
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUTPUT_DIR)
    args = ap.parse_args()

    for rd in (NOKDE, REFERENCE):
        subprocess.run([sys.executable, "experiments/ablation_advance_metrics_stage_3.py",
                        "--output", args.output, "--results-dir", rd], check=True)

    a, b = load_epoch(args.output, NOKDE, EPOCH), load_epoch(args.output, REFERENCE, EPOCH)
    stems = sorted(set(a) & set(b))
    table2 = json.loads(Path(TABLE2_SDEDIT).read_text())
    curves = json.loads((Path(args.output) / NOKDE / "metrics_avg.json").read_text())

    lines = ["# E8: KDE-loss ablation (Full + SDEdit settings, Table 2 protocol)", "",
             f"Epoch {EPOCH}, {len(stems)} paired images, same noise per image. "
             "p: Wilcoxon signed-rank, noKDE vs full.", "",
             "| metric | full (seeded) | noKDE | change | p | Table 2 sdedit (unseeded) |",
             "|---|---|---|---|---|---|"]
    for key, label, lower in METRICS:
        x = [b[s][key] for s in stems]
        y = [a[s][key] for s in stems]
        mx, my = statistics.fmean(x), statistics.fmean(y)
        p = wilcoxon(y, x).pvalue if any(u != v for u, v in zip(x, y)) else 1.0
        worse = (my > mx) if lower else (my < mx)
        tag = "worse" if worse else "better"
        lines.append(f"| {label} | {mx:.5g} | {my:.5g} | {100 * (my - mx) / mx:+.1f}% ({tag}) | {p:.3g} | "
                     f"{table2[key][EPOCH]:.5g} |")
    lines += ["", "## noKDE over epochs (mean of 50 images)", "",
              "| epoch | " + " | ".join(label for _, label, _ in METRICS) + " |",
              "|---|" + "---|" * len(METRICS)]
    for ep in sorted(curves[METRICS[0][0]], key=int):
        lines.append(f"| {ep} | " + " | ".join(f"{curves[k][ep]:.5g}" for k, _, _ in METRICS) + " |")
    (Path(args.output) / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
