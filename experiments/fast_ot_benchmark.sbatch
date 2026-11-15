#!/bin/bash
# Fast Smart Init assignment benchmark (CPU): cost vs exact OT and time, real icons.
#   sbatch experiments/fast_ot_benchmark.sbatch
#SBATCH --partition cpu
#SBATCH --time 0-04:00:00
#SBATCH --job-name fast-ot-bench
#SBATCH --output logs/fast-ot-bench-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
lscpu | grep "Model name"

python -u experiments/fast_ot_benchmark.py "$@"
echo "$(date) done"
