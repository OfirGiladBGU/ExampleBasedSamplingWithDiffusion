#!/bin/bash
# Fig. 4 images (point-count generalization) with a fast Smart Init assignment, for a visual
# check against the exact-OT figure (grid_size_exact_ot.sbatch). Same two staged images, grids
# 24-80, the new variant (ddim-25-eta1); budgets other than 1024 are checked visually only.
#   sbatch --export=ALL,SOLVER=bisection experiments/fast_ot_visual.sbatch
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name fast-ot-visual
#SBATCH --output logs/fast-ot-visual-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

SOLVER=${SOLVER:?set SOLVER}
SAMPLERS=${SAMPLERS:-ddim-25-eta1}
GRIDS=${GRIDS:-"24 32 48 64 80"}
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST | $SOLVER"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
SRC=experiments/outputs/grid_size_results/source
OUT=experiments/outputs/grid_size_results/fast_ot_$SOLVER
for G in $GRIDS; do
    echo "== grid $G ($((G * G)) points) $(date)"
    python -u experiments/few_step_sampling_stage_1.py \
        --control_ckpt_path "$CKPT" --label CN-GBN --source "$SRC" --output "$OUT" \
        --grid_size "$G" --samplers "$SAMPLERS" --ot_solver "$SOLVER" --png-count 2 --no-throughput
done
for S in ${SAMPLERS//,/ }; do
    python experiments/grid_size_results_stage_2.py --fewstep-dir "$OUT" --sampler "$S" \
        -o "experiments/outputs/grid_size_results/pointcount_generalization_${SOLVER}_${S}.pdf"
done
echo "$(date) done"
