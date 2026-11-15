#!/bin/bash
# Rebuttal E8: score the exported point sets, one array task per epoch folder.
# Tasks 0-9: sdedit_noKDE epochs 500 ... 5000; task 10: sdedit_seeded (epoch 5000).
#   sbatch --dependency=afterok:<export job> experiments/ablation_nokde_metrics.sbatch
# Then: python experiments/ablation_nokde_stage_3.py
#SBATCH --partition cpu
#SBATCH --time 0-06:00:00
#SBATCH --job-name nokde-metrics
#SBATCH --output logs/nokde-metrics-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-10

echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

if [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
    python -u experiments/ablation_nokde_stage_2.py --results-dir sdedit_seeded
else
    EPOCH=$(( (SLURM_ARRAY_TASK_ID + 1) * 500 ))
    python -u experiments/ablation_nokde_stage_2.py --results-dir sdedit_noKDE --epochs "epoch_${EPOCH}_npy"
fi
echo "$(date) done"
