#!/bin/bash
# Low-noise fine-tune (after the generalization diagnostics, REBUTTAL_DRAFT.md A19): the capacity /
# CVT gap arises in the last low-noise steps, which Min-SNR (gamma = 5) weights by 0.002-0.02.
# Fine-tune the paper's full Ours-GBN checkpoint (epoch 5000) for 300 more epochs with:
#   task 0  g5    gamma = 5   control: the paper's weighting, only more training
#   task 1  g50   gamma = 50  low-noise weight x10 (t = 10: 0.008 -> 0.08; t = 50: 0.09 -> 0.88)
#   task 2  g0    no Min-SNR  plain eps loss, every step weight 1
# Everything else is the full run's configuration. The epoch-5000 checkpoint is copied into the new
# run folder and resumed (--resume-latest), so the optimizer state carries over.
#   sbatch experiments/ft_lownoise_train.sbatch
# Then: sbatch --dependency=aftercorr:<this job> experiments/ft_lownoise_eval.sbatch
#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name ft-lownoise
#SBATCH --output logs/ft-lownoise-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-2

set -e
TAGS=(g5 g50 g0)
GAMMAS=(5 50 0)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
EPOCHS=${EPOCHS:-5300}

echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST | $TAG gamma $GAMMA -> epoch $EPOCHS"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

SRC_CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
OUT=control_v4/train_outputs_Icons-50_1024_GBN_ft-$TAG
mkdir -p "$OUT/checkpoints"
[ -n "$(ls "$OUT/checkpoints" 2>/dev/null)" ] || cp "$SRC_CKPT" "$OUT/checkpoints/"

DATA=/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN
python -u control_v4/train_control.py \
    --source "$DATA/source" --target "$DATA/target" --offsets "$DATA/processed_offsets" \
    --out "$OUT" --epochs "$EPOCHS" --resume-latest \
    --base-ckpt-path "" --no-freeze-denoiser --enable-gecco --enable-adaptive-gate-injection \
    --min-snr-gamma "$GAMMA" --save_every 50 --keep-every 100 --wandb-tags "ft_lownoise_$TAG"
ls -la "$OUT/checkpoints"
echo "$(date) done"
