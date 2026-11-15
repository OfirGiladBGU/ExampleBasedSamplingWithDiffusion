#!/bin/bash
# Rebuttal E8: KDE-loss ablation. Ours-GBN on Icons-50 (1024 points), identical to the paper's
# "full" run (control_v4/train_outputs_Icons-50_1024_GBN_full/run_params.json) except
# --density-loss-weight 0. Resumable (resume_latest): resubmit the same script if it times out.
#   sbatch experiments/train_ablation_nokde.sbatch
#SBATCH --partition main
#SBATCH --time 7-00:00:00
#SBATCH --job-name train-noKDE
#SBATCH --output logs/train-noKDE-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

DATA=/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN
python -u control_v4/train_control.py \
    --source "$DATA/source" --target "$DATA/target" --offsets "$DATA/processed_offsets" \
    --out control_v4/train_outputs_Icons-50_1024_GBN_noKDE \
    --base-ckpt-path "" --no-freeze-denoiser --enable-gecco --enable-adaptive-gate-injection \
    --density-loss-weight 0 --wandb-tags ablation_noKDE
echo "$(date) done"
