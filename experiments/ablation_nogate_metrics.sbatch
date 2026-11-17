#!/bin/bash
# Reviewer W5 (gated-injection ablation): score the exported noGate point sets with the Table 2
# scorer, one array task per epoch folder (task i = epoch (i + 1) * 500).
#   sbatch --dependency=afterok:<export job> experiments/ablation_nogate_metrics.sbatch
#SBATCH --partition cpu
#SBATCH --time 0-06:00:00
#SBATCH --job-name nogate-metrics
#SBATCH --output logs/nogate-metrics-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9

set -e
echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

EPOCH=$(( (SLURM_ARRAY_TASK_ID + 1) * 500 ))
python -u experiments/ablation_nokde_stage_2.py --results-dir sdedit_noGate --epochs "epoch_${EPOCH}_npy"
echo "$(date) done"
