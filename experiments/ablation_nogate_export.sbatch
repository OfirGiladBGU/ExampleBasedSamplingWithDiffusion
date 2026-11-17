#!/bin/bash
# Reviewer W5 (gated-injection ablation): export the noGate checkpoints with the Table 2
# "Full + SDEdit" settings on the same 50 validation images and with the same per-image noise as
# the seeded full-model reference of E8 (sdedit_seeded, already exported and scored).
#   -> experiments/outputs/ablation_nokde_e500_b50_1024/sdedit_noGate/epoch_*_npy
#   sbatch experiments/ablation_nogate_export.sbatch                       # every 500 epochs
#   CKPTS=ep1000.,ep1500. sbatch experiments/ablation_nogate_export.sbatch   # a subset
# Pass CKPTS through the environment as above: `sbatch --export=ALL,CKPTS=a,b` splits on the
# commas and only the first name arrives.
# CKPTS: 'all' or comma-separated checkpoint-name substrings; each must match a file.
# Then: ablation_nogate_metrics.sbatch, then
#   python experiments/ablation_nokde_stage_3.py --variant sdedit_noGate --label noGate \
#       --title "W5: gated-injection ablation" --summary summary_noGate.md
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name nogate-export
#SBATCH --output logs/nogate-export-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -e
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

DIR=control_v4/train_outputs_Icons-50_1024_GBN_noGate/checkpoints
CKPTS=${CKPTS:-all}
if [ "$CKPTS" != all ]; then
    for c in ${CKPTS//,/ }; do
        ls "$DIR"/*"$c"* > /dev/null 2>&1 || { echo "no checkpoint matches '$c' in $DIR"; exit 1; }
    done
fi
python -u experiments/ablation_nokde_stage_1.py --model nogate --checkpoints "$CKPTS"
for d in experiments/outputs/ablation_nokde_e500_b50_1024/sdedit_noGate/epoch_*_npy; do
    echo "$d: $(ls "$d" | wc -l) files"
done
echo "$(date) done"
