#!/bin/bash
# Reviewer W5: gated injection ablation. Ours-GBN on Icons-50 (1024 points), identical to the
# paper's "full" run (control_v4/train_outputs_Icons-50_1024_GBN_full/run_params.json: jointly
# trained, GECCO features, KDE loss 0.8) except --no-enable-adaptive-gate-injection (plain zero
# convolutions). Resumable (resume_latest): resubmit the same script if it times out.
#   sbatch experiments/train_ablation_nogate.sbatch
#SBATCH --partition main
#SBATCH --time 7-00:00:00
#SBATCH --job-name train-noGate
#SBATCH --output logs/train-noGate-%j.out
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
    --out control_v4/train_outputs_Icons-50_1024_GBN_noGate \
    --base-ckpt-path "" --no-freeze-denoiser --enable-gecco --no-enable-adaptive-gate-injection \
    --wandb-tags ablation_noGate
echo "$(date) done"
