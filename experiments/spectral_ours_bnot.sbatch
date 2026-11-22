#!/bin/bash
# Spectral analysis with Ours-BNOT (rebuttal, user request 2026-11-22).
#   1. Ours-BNOT on the 150 spectral-analysis sources (uniform grey levels x 25 realizations +
#      quad4 / checker patterns, 1024 points). Same pipeline and defaults as the stored Ours-WVS /
#      Ours-GBN sets of this analysis (control_data_gen.py: Smart Init seed 42, start at t=500,
#      DDPM, 500 steps), Icons-50 BNOT checkpoint. Resumable: finished images are skipped.
#      -> experiments/outputs/spectral_analysis/target_CN-BNOT_1024/
#   2. spectral_analysis_stage_1.py on it -> target_CN-BNOT_1024_spectral/
#   3. spectral_analysis_stage_2.py with the columns BNOT WVS GBN || Ours-BNOT Ours-WVS Ours-GBN
#      (the order of Table 1) -> experiments/outputs/spectral_analysis/plots_full_bnot/
#      (plots_full/, the figures of the submitted paper, is not touched)
#   4. spectral_anisotropy_summary.py (now with an Ours-BNOT row).
#   sbatch experiments/spectral_ours_bnot.sbatch
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name spectral-ours-bnot
#SBATCH --output logs/spectral-ours-bnot-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -eo pipefail
ROOT=/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd $ROOT
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

S=experiments/outputs/spectral_analysis
python -u experiments/control_data_gen.py --data_path $ROOT/$S \
    --control_ckpt_path control_v4/train_outputs_Icons-50_1024_BNOT/checkpoints/dynamic_ep5000.ckpt \
    --grid_size 32 --target_folder target_CN-BNOT_1024 --no-track_time
echo "point sets: $(ls $S/target_CN-BNOT_1024/*.npy | wc -l) of $(ls $S/source/*.png | wc -l)"

python -u experiments/spectral_analysis_stage_1.py $S/target_CN-BNOT_1024

python -u experiments/spectral_analysis_stage_2.py --output $S/plots_full_bnot \
    --methods 'target_BNOT_1024:BNOT,target_WVS_1024:WVS,target_GBN_1024:GBN,||,target_CN-BNOT_1024:Ours-BNOT,target_CN-WVS_1024:Ours-WVS,target_CN-GBN_1024:Ours-GBN' \
    --invariance-methods target_CN-BNOT_1024,target_CN-WVS_1024,target_CN-GBN_1024

python -u experiments/spectral_anisotropy_summary.py
ls -la $S/plots_full_bnot | head -40
echo "$(date) done"
