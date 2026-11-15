#!/bin/bash
# OT-grid assignment solvers (paper POT, exact POT, scipy, GPU auction) on Smart Init samples.
#   sbatch experiments/ot_assignment_benchmark.sbatch
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name ot-bench
#SBATCH --output logs/ot-bench-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv
python -u experiments/ot_assignment_benchmark.py "$@"
echo "$(date) done"
