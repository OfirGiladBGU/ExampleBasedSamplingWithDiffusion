#!/bin/bash
# E15 / reviewer W7: spectra, anisotropy and pair correlations of the new sampler (ddim-25-eta1).
# Samples the spectral-analysis sources (uniform grey levels x 25 realizations + quad4 / checker
# patterns, 1024 points) with Ours-GBN and Ours-WVS, runs spectral_analysis_stage_1.py on them, and
# summarizes anisotropy per frequency band next to the stored paper methods.
#   sbatch experiments/spectral_ddim25.sbatch
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name spectral-ddim25
#SBATCH --output logs/spectral-ddim25-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

SRC=experiments/outputs/spectral_analysis/source
OUT=experiments/outputs/spectral_analysis_ddim25
declare -A CKPT=(
    [CN-GBN]=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
    [CN-WVS]=control_v4/train_outputs_Icons-50_1024_WVS/checkpoints/dynamic_ep5000.ckpt
)
for L in CN-GBN CN-WVS; do
    python -u experiments/few_step_sampling_stage_1.py \
        --control_ckpt_path "${CKPT[$L]}" --label "$L" --source "$SRC" --output "$OUT" \
        --grid_size 32 --samplers ddim-25-eta1 --png-count 0 --no-throughput
    python -u experiments/spectral_analysis_stage_1.py "$OUT/${L}_1024/ddim-25-eta1" \
        --output "experiments/outputs/spectral_analysis/target_${L}-DDIM25_1024_spectral"
done
python -u experiments/spectral_anisotropy_summary.py
echo "$(date) done"
