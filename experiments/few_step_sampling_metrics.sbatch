#!/bin/bash
# Few-step sampling stage 2 (rebuttal E1): metrics, sharded over a CPU job array.
#   sbatch --export=ALL,RUN=CN-GBN_1024 experiments/few_step_sampling_metrics.sbatch
#   (ROOT = parent of the run folder, SRC = source images; both default to the Icons-50 E1 run)
#SBATCH --partition cpu
#SBATCH --time 0-12:00:00
#SBATCH --job-name fewstep-metrics
#SBATCH --output logs/fewstep-metrics-%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-15

echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST | run $RUN"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

python -u experiments/few_step_sampling_stage_2.py \
    "${ROOT:-experiments/outputs/few_step_sampling-Icons-50_1024}/${RUN:-CN-GBN_1024}" \
    --source "${SRC:-experiments/outputs/z_validation_data/Icons-50_1024/source}" \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT" \
    --workers "$SLURM_CPUS_PER_TASK"
