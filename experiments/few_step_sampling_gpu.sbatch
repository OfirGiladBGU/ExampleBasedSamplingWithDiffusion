#!/bin/bash
# Few-step sampling stage 1 (rebuttal E1): one array task per checkpoint.
#   sbatch experiments/few_step_sampling_gpu.sbatch
#SBATCH --partition main
#SBATCH --time 0-10:00:00
#SBATCH --job-name fewstep-sample
#SBATCH --output logs/fewstep-sample-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-1

echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

CKPTS=(
    control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
    control_v4/train_outputs_Icons-50_1024_BNOT/checkpoints/dynamic_ep5000.ckpt
)
LABELS=(CN-GBN CN-BNOT)

python -u experiments/few_step_sampling_stage_1.py \
    --control_ckpt_path "${CKPTS[$SLURM_ARRAY_TASK_ID]}" \
    --label "${LABELS[$SLURM_ARRAY_TASK_ID]}" \
    "$@"
