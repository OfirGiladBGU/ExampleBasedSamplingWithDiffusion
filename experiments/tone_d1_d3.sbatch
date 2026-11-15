#!/bin/bash
# Rebuttal D1 + D3 (tone section): the gradient-free fixed-point baseline with GBN / our sampler in the
# loop (tone_results_stage_7_fixed_point.py) and the per-step cost of the gradient paths
# (tone_results_stage_8_cost.py). RTX 6000 Ada: GBN is built for sm_89, and D3 times on it.
#   sbatch experiments/tone_d1_d3.sbatch
#   smoke: sbatch --export=ALL,D1_ARGS="--limit 2 --iters 1 --samplers gbn,ddim-25-eta1 --output experiments/outputs/tone_results/fixed_point_smoke",RUN_D3=0 experiments/tone_d1_d3.sbatch
#SBATCH --partition main
#SBATCH --time 0-05:00:00
#SBATCH --job-name tone-d1d3
#SBATCH --output logs/tone-d1d3-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --constraint=rtx_6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv,noheader

if [ "${RUN_D3:-1}" = "1" ]; then
    python -u experiments/tone_results_stage_8_cost.py
fi
python -u experiments/tone_results_stage_7_fixed_point.py ${D1_ARGS:-}
echo "$(date) done"
