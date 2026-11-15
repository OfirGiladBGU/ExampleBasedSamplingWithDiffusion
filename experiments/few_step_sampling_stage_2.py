"""Few-step sampling stage 2: compute M1-M5 metrics for the stage-1 point sets.

Same scoring as quantitative_advance_metrics_stage_1.py (exact quadrature, points vs the
grayscale source density), applied to every <sampler>/<stem>.npy under the run folder and
written to <sampler>_json/<stem>.json. The work list (sampler x image) can be split into
--num-shards pieces for a Slurm job array, and each piece runs --workers processes.

Usage:
    python experiments/few_step_sampling_stage_2.py \\
        experiments/outputs/few_step_sampling-Icons-50_1024/CN-GBN_1024 --workers 8
    # job array: --shard $SLURM_ARRAY_TASK_ID --num-shards 16
"""

import os

# One BLAS thread per worker; parallelism comes from the process pool
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.quantitative_advance_metrics_stage_1 import (
    find_source_image,
    image_to_grayscale_01,
    load_points_from_npy,
    serialize_metrics,
)
from utils.stippling_metrics_advance import compute_all_advanced_metrics

SOURCE_DIR = "experiments/outputs/z_validation_data/Icons-50_1024/source"


def parse_args():
    p = argparse.ArgumentParser(description="Compute M1-M5 metrics for few-step sampling outputs")
    p.add_argument("run_dir", help="Stage-1 folder, e.g. .../few_step_sampling-Icons-50_1024/CN-GBN_1024")
    p.add_argument("--source", default=SOURCE_DIR)
    p.add_argument("--samplers", default="all", help="'all' or comma-separated sampler folder names")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def score_one(job):
    npy_path, source_dir, json_path = job
    if json_path.exists():
        return json_path, "skip", 0.0
    t0 = time.perf_counter()
    try:
        points = load_points_from_npy(npy_path)
        image_01 = image_to_grayscale_01(find_source_image(npy_path, source_dir))
        payload = serialize_metrics(compute_all_advanced_metrics(points, image_01, mc_approx=False))
        tmp = json_path.with_suffix(f".{os.getpid()}.tmp")  # per-process: concurrent jobs must not share it
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(json_path)
        return json_path, "ok", time.perf_counter() - t0
    except Exception as e:
        return json_path, f"ERROR - {e}", time.perf_counter() - t0


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if args.samplers == "all":
        sampler_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and not d.name.endswith("_json"))
    else:
        sampler_dirs = [run_dir / s.strip() for s in args.samplers.split(",")]

    jobs = []
    for d in sampler_dirs:
        out = run_dir / f"{d.name}_json"
        out.mkdir(exist_ok=True)
        for npy in sorted(d.glob("*.npy")):
            jobs.append((npy, args.source, out / f"{npy.stem}.json"))
    jobs = jobs[args.shard::args.num_shards]
    print(f"{len(jobs)} point sets in shard {args.shard}/{args.num_shards} "
          f"({len(sampler_dirs)} samplers, {args.workers} workers)")

    n_err = 0
    with Pool(args.workers) as pool:
        for i, (json_path, status, dt) in enumerate(pool.imap_unordered(score_one, jobs), 1):
            if status.startswith("ERROR"):
                n_err += 1
            if status != "skip":
                print(f"[{i}/{len(jobs)}] {json_path.parent.name}/{json_path.name}: {status} ({dt:.1f}s)", flush=True)
    print(f"Done, {n_err} errors")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
