#!/bin/bash
# Smoke test for the fast Smart Init solvers and the -noise sampler option (CPU, ~2 min).
#   sbatch experiments/fast_ot_smoke.sbatch
#SBATCH --partition cpu
#SBATCH --time 0-00:30:00
#SBATCH --job-name fast-ot-smoke
#SBATCH --output logs/fast-ot-smoke-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

python - <<'EOF'
import numpy as np
from experiments.few_step_sampling_stage_1 import parse_sampler, start_step_for, starts_from_noise, eval_timesteps_for
for s in ("ddim-25-eta1", "ddim-25-eta1-noise", "ddpm-1000-noise", "ddpm-500"):
    st = start_step_for(s, 500, 1000)
    print(s, parse_sampler(s), "noise" if starts_from_noise(s) else "", "start", st,
          eval_timesteps_for(st, parse_sampler(s)[1])[:3] if not s.startswith("ddpm") else "")
assert start_step_for("ddim-25-eta1", 500, 1000) == 500 and start_step_for("ddim-25-eta1-noise", 500, 1000) == 1000

from data.Transforms import to_image_optimal_transport
from control_v4.ot_assignment import points_to_offsets
p = np.random.RandomState(0).rand(1024, 2).astype(np.float32)
ref = to_image_optimal_transport(p)
assert np.allclose(points_to_offsets(p, "pot-exact"), ref, atol=1e-5)
for s in ("bisection", "bisection-swap", "bisection-blocks"):
    o = points_to_offsets(p, s)
    assert o.shape == ref.shape and np.isfinite(o).all()
    print(s, "cost ratio vs exact", round(float((o ** 2).sum(0).mean() / (ref ** 2).sum(0).mean()), 3))
print("smoke checks ok")
EOF
python -u experiments/fast_ot_benchmark.py --grids 32 64 --icons-limit 3 --output experiments/outputs/fast_ot_benchmark_smoke
echo "$(date) done"
