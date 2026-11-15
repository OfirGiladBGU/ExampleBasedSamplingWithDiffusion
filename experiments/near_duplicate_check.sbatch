#!/bin/bash
# Reviewer W6: do near-duplicate (same emoji, other vendor) training icons drive the test results?
#   sbatch experiments/near_duplicate_check.sbatch
#SBATCH --partition cpu
#SBATCH --time 0-02:00:00
#SBATCH --job-name near-dup
#SBATCH --output logs/near-dup-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
python -u experiments/near_duplicate_check.py
echo "$(date) done"
