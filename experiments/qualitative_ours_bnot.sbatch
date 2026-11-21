#!/bin/bash
# Qualitative figures with an Ours-BNOT column (rebuttal, user request 2026-11-22).
#   1. Ours-BNOT on the quadratic density gradient (top row of the main figure): the one cell that
#      did not exist. Same pipeline and defaults as the stored Ours-WVS / Ours-GBN cells
#      (control_data_gen.py: Smart Init seed 42, SDEdit start at t=500, DDPM, 500 steps), one
#      realization, Icons-50 BNOT checkpoint. Skipped if the point set is already there.
#   2. The three qualitative figures, same rows as the submitted ones, columns
#      Target || BNOT WVS GBN || Ours-BNOT Ours-WVS Ours-GBN (the order of Table 1):
#        qualitative_comparison_full_bnot             main figure (gradient + 4 test images)
#        qualitative_comparison_appendix_left_bnot    supplement, Icons-50 + CelebA
#        qualitative_comparison_appendix_right_bnot   supplement, ShapeNet2D + Anime Chibi
#      The test-set cells are the stored point sets z_validation_data/<dataset>/target_CN-BNOT_<N>
#      (the ones scored in Table 1); nothing is re-sampled for them.
#   3. Strip percentages of the gradient row (fig3_strip_realizations.py, now with Ours-BNOT).
# -> experiments/outputs/qualitative_showcase_full_bnot/
#   sbatch experiments/qualitative_ours_bnot.sbatch
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name qual-ours-bnot
#SBATCH --output logs/qual-ours-bnot-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -eo pipefail
ROOT=/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd $ROOT
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

Q=experiments/outputs/images_results_metrics/quadratic
if [ -f $Q/target_CN-BNOT_1024/quadratic_density_gradient.npy ]; then
    echo "gradient: Ours-BNOT point set exists, not re-sampled"
else
    python -u experiments/control_data_gen.py --data_path $ROOT/$Q \
        --control_ckpt_path control_v4/train_outputs_Icons-50_1024_BNOT/checkpoints/dynamic_ep5000.ckpt \
        --grid_size 32 --target_folder target_CN-BNOT_1024 --no-track_time
fi
ls -la $Q/target_CN-BNOT_1024

OUT=experiments/outputs/qualitative_showcase_full_bnot
COLS='Target,||,BNOT,WVS,GBN,||,Ours-BNOT,Ours-WVS,Ours-GBN'
python -u experiments/qualitative_comparison_showcase_full.py --output $OUT \
    --out-name qualitative_comparison_full_bnot --columns "$COLS"
python -u experiments/qualitative_comparison_showcase.py --output $OUT \
    --out-name qualitative_comparison_appendix_left_bnot --columns "$COLS" \
    --samples '{"icons": [102, 104, 106, 110], "faces": [102, 106, 204, 210]}'
python -u experiments/qualitative_comparison_showcase.py --output $OUT \
    --out-name qualitative_comparison_appendix_right_bnot --columns "$COLS" \
    --samples '{"shapenet2d": [27, 103, 162, 308], "chibi": [60, 167, 248, 255]}'

python -u experiments/fig3_strip_realizations.py
ls -la $OUT
echo "$(date) done"
