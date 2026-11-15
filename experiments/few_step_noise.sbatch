#!/bin/bash
# Pure-noise start (no Smart Init, no SDEdit) with stochastic DDIM, Icons-50 test set (1000 images,
# 1024 points), same checkpoint and per-image noise seeds as E1. The run folder links the E1
# ddpm-500 (paper sampler) and ddim-25-eta1 (SDEdit) results, so few_step_sampling_stage_3.py
# compares all of them paired.
#   sbatch experiments/few_step_noise.sbatch
# Then: sbatch --dependency=afterok:<this job> \
#   --export=ALL,ROOT=experiments/outputs/few_step_noise-Icons-50_1024,RUN=CN-GBN_1024 \
#   experiments/few_step_sampling_metrics.sbatch
# Then: python experiments/few_step_sampling_stage_3.py experiments/outputs/few_step_noise-Icons-50_1024
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name fewstep-noise
#SBATCH --output logs/fewstep-noise-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2

echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

GROUPS_=(
    ddim-25-eta1-noise,ddim-50-eta1-noise
    ddim-100-eta1-noise
    ddpm-1000-noise
)

python -u experiments/few_step_sampling_stage_1.py \
    --control_ckpt_path control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt \
    --label CN-GBN --output experiments/outputs/few_step_noise-Icons-50_1024 \
    --samplers "${GROUPS_[$SLURM_ARRAY_TASK_ID]}" --tag "noise$SLURM_ARRAY_TASK_ID" --no-throughput
echo "$(date) done"
