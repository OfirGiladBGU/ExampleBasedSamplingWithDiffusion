"""Stage 6: point-distribution quality of the tone-corrected full-sampler stipples (rebuttal, tone section).

Stage 5 showed that the requested densities optimized through the DDIM-25 chain improve the
rendered tone of the full sampler. This stage checks what that costs in blue-noise quality: for
every (requested density, sampler) of a stage-5 run, the mean of the few_step_sampling_stage_2
metrics (computed per run folder, <samples>/<config>-seed<S>_<N>/<sampler>_json) over images and
seeds, and the clumped fraction measured exactly as in stage 4 (threshold: half the median
nearest-neighbour distance of our uncorrected one-step stipple of the same image), so the numbers
sit next to the direct point-optimization rows of stage 4.

Usage:
    (sbatch --array=0-1 --export=ALL,ROOT=<samples>,RUN=<run>,SRC=experiments/outputs/tone_results/source
        experiments/few_step_sampling_metrics.sbatch, per run folder)
    python experiments/tone_results_stage_6_distribution.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = "experiments/outputs/tone_results"
RUN_DIR = f"{ROOT_DIR}/full_sampler_ddim25"
CLUMP_REF = f"{ROOT_DIR}/direct_points/ours-none"
CONFIGS = "none,curve,field,curve-initrho0,field-initrho0"
SAMPLERS = "ddpm-500,ddim-25-eta1"
POINTS = 1024
SUMMARY_METRICS = {"M1_v2_hot_energy": "HOT", "M2_v1_capacity_delta_c": "cap dc",
                   "M2_v2_centroid_displacement": "D", "M5_spatial_measure_rho_mean": "rho_s"}


def parse_args():
    p = argparse.ArgumentParser(description="Point-distribution metrics of the stage-5 stipples")
    p.add_argument("--run", default=RUN_DIR, help="Stage-5 output folder (with samples/ and summary.json)")
    p.add_argument("--clump-ref", default=CLUMP_REF, help="Reference stipples of the clumping threshold")
    p.add_argument("--configs", default=CONFIGS)
    p.add_argument("--samplers", default=SAMPLERS)
    p.add_argument("--points", type=int, default=POINTS)
    return p.parse_args()


def nn_dist(pts):
    d = torch.cdist(torch.as_tensor(pts)[None], torch.as_tensor(pts)[None])[0]
    d.fill_diagonal_(float("inf"))
    return d.min(1).values


def main():
    a = parse_args()
    run = Path(a.run)
    tone = json.loads((run / "summary.json").read_text())
    samples = run / "samples"
    lines = ["# Tone correction through the full sampler: point-distribution quality", "",
             f"Stipples of {run.name}; metrics from few_step_sampling_stage_2 against the original image, mean over "
             "images and seeds. PSNR from the stage-5 summary.", "",
             "| sampler | requested density | PSNR | " + " | ".join(SUMMARY_METRICS.values()) + " | clumped % | scored |",
             "|" + "---|" * (5 + len(SUMMARY_METRICS))]
    res = {}
    for s in a.samplers.split(","):
        for c in a.configs.split(","):
            mets, clumped = [], []
            for d in sorted(samples.glob(f"{c}-seed*_{a.points}")):
                if d.name.split("-seed")[0] != c:
                    continue
                for f in sorted((d / s).glob("*.npy")):
                    j = d / f"{s}_json" / f"{f.stem}.json"
                    if j.exists():
                        mets.append(json.loads(j.read_text()))
                    ref = Path(a.clump_ref) / f.name
                    thr = 0.5 * float(nn_dist(np.load(ref)).median())
                    clumped.append(100.0 * float((nn_dist(np.load(f)) < thr).double().mean()))
            if not clumped:
                continue
            r = {short: float(np.mean([m[key] for m in mets])) if mets else float("nan")
                 for key, short in SUMMARY_METRICS.items()}
            r.update({"clumped_pct": float(np.mean(clumped)), "scored": len(mets), "stipples": len(clumped),
                      "psnr": tone.get(s, {}).get(c, {}).get("psnr", float("nan"))})
            res.setdefault(s, {})[c] = r
            lines.append(f"| {s} | {c} | {r['psnr']:.2f} | " + " | ".join(f"{r[k]:.4g}" for k in SUMMARY_METRICS.values())
                         + f" | {r['clumped_pct']:.1f} | {r['scored']}/{r['stipples']} |")
    lines += ["", "clumped %: points whose nearest neighbour is closer than half the median nearest-neighbour "
                  "distance of our uncorrected one-step stipple of the same image (as in stage 4)."]
    (run / "distribution.md").write_text("\n".join(lines) + "\n")
    (run / "distribution.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
