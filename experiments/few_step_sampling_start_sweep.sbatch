#!/bin/bash
# Few-step sampling stage 1, start-step sweep: the paper DDPM loop started at t = 500/300/200/100/50,
# from the noised Smart Init (SDEdit) and from the clean Smart Init (-clean, no forward noise).
# ddpm-500 (noised) comes from few_step_sampling_gpu.sbatch. Groups are balanced to ~600 NFE/image.
#   sbatch experiments/few_step_sampling_start_sweep.sbatch
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name fewstep-start
#SBATCH --output logs/fewstep-start-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2

echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

GROUPS_=(
    ddpm-500-clean,ddpm-50-clean,ddpm-50
    ddpm-300-clean,ddpm-300
    ddpm-200-clean,ddpm-200,ddpm-100-clean,ddpm-100
)

python -u experiments/few_step_sampling_stage_1.py \
    --control_ckpt_path control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt \
    --label CN-GBN \
    --samplers "${GROUPS_[$SLURM_ARRAY_TASK_ID]}" \
    --tag "start$SLURM_ARRAY_TASK_ID" \
    --no-throughput \
    "$@"
