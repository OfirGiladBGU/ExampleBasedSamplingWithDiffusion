"""Does the result depend on the ink-gain calibration?

Stage 2: needs stage 1 -- it reads the calibrated ink gain from main/summary.json (pass
--base-gain to skip that) -- plus a checkpoint and a GPU. Stage 3 draws the paper's second
graph from the gain_sweep.csv this writes, so it must run before stage 3.

The gain sets how quickly overlapping dots saturate -- how severe the dot gain being
compensated for is. A single value invites the objection that the effect was manufactured by
picking an extreme one. This re-runs stage 1 at several multiples of the calibrated gain and
tabulates the outcome at each.

The expected, honest pattern: the advantage shrinks towards zero as the gain does, because
with no dot gain there is nothing to correct. What must NOT happen is the ordering changing,
or the advantage appearing only in a narrow band.

    python experiments/tone_results_stage_2_gain_sweep.py --control_ckpt_path <ckpt>
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = "experiments/outputs/tone_results"
OUT_DIR = f"{ROOT_DIR}/gain_sweep"
MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]


# --------------------------------------------------------------------------
# Defaults for every command-line argument. Edit here, not in parse_args.
# --------------------------------------------------------------------------
BASE_GAIN         = None
REFERENCE         = f"{ROOT_DIR}/main/summary.json"
MULTIPLIERS_STR   = ",".join(str(m) for m in MULTIPLIERS)
STEPS             = 150
LIMIT             = 12
CONFIGS           = "none,curve,field"
CONTROL_CKPT_PATH = "control_v4/train_outputs_Icons-50_1024_GBN_full/dynamic_ep5000.ckpt"


def parse_args():
    p = argparse.ArgumentParser(description="Ink-gain robustness sweep")
    p.add_argument("--base-gain", type=float, default=BASE_GAIN,
                   help="Calibrated gain; read from the main run's summary.json if omitted")
    p.add_argument("--reference", default=REFERENCE)
    p.add_argument("--multipliers", default=MULTIPLIERS_STR)
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--limit", type=int, default=LIMIT)
    p.add_argument("--configs", default=CONFIGS)
    p.add_argument("--control_ckpt_path",
                   default=CONTROL_CKPT_PATH)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _write(out, rows):
    keys = sorted({k for r in rows for k in r})
    with open(out / "gain_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    a = parse_args()
    base = a.base_gain
    if base is None:
        ref = Path(a.reference)
        if not ref.exists():
            print(f"ERROR: {ref} not found; run stage 1 first or pass --base-gain")
            return 2
        base = float(json.loads(ref.read_text())["config"]["ink_gain"])
    mults = [float(m) for m in a.multipliers.split(",") if m.strip()]
    print(f"calibrated gain {base:.4f}; sweeping x{mults}")

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in mults:
        gain = base * m
        run_dir = out / f"gain_{m:g}x"
        print(f"\n=== gain x{m:g} = {gain:.4f} -> {run_dir} ===", flush=True)
        cmd = [sys.executable, "experiments/tone_results_stage_1_optimize.py",
               "--control_ckpt_path", a.control_ckpt_path,
               "--output", str(run_dir), "--steps", str(a.steps),
               "--limit", str(a.limit), "--configs", a.configs,
               "--ink-gain", f"{gain:.8f}"]
        print("  " + " ".join(cmd[1:]))
        if a.dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  run failed (exit {r.returncode}); continuing")
            continue
        csv_path = run_dir / "tone_summary.csv"
        if not csv_path.exists():
            continue
        per = {}
        for rec in csv.DictReader(open(csv_path, encoding="utf-8")):
            per.setdefault(rec["config"], []).append(float(rec["psnr"]))
        row = {"multiplier": m, "gain": gain}
        for c, v in per.items():
            row[f"{c}_psnr"] = float(np.mean(v))
        if "none_psnr" in row and "field_psnr" in row:
            row["gain_over_none"] = row["field_psnr"] - row["none_psnr"]
        if "curve_psnr" in row and "field_psnr" in row:
            row["gain_over_curve"] = row["field_psnr"] - row["curve_psnr"]
        rows.append(row)
        # Write after every multiplier: a long sweep that is interrupted should still leave
        # the results it already earned, rather than nothing.
        _write(out, rows)

    if not rows:
        print("\nnothing to summarize")
        return 0


    print("\n" + "=" * 78)
    print(f"{'gain':>10} {'x':>6} {'none':>8} {'curve':>8} {'field':>8} "
          f"{'vs none':>9} {'vs curve':>9}")
    for r in rows:
        print(f"{r['gain']:>10.3f} {r['multiplier']:>6g} "
              f"{r.get('none_psnr', float('nan')):>8.2f} "
              f"{r.get('curve_psnr', float('nan')):>8.2f} "
              f"{r.get('field_psnr', float('nan')):>8.2f} "
              f"{r.get('gain_over_none', float('nan')):>+9.2f} "
              f"{r.get('gain_over_curve', float('nan')):>+9.2f}")
    print(f"\n-> {out / 'gain_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
