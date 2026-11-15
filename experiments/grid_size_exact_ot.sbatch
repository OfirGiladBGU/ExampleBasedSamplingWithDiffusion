#!/bin/bash
# Point-count generalization figure with the exact OT assignment (rebuttal, Fig. pointcount fix).
# The paper figure (grid_size_results_stage_1.py -> sample_control.py) used POT's ot.emd with its
# default iteration cap, which truncates from ~3136 points (the 4096 column). This re-samples the
# same two staged images at the same grid sizes with the paper sampler (ddpm-500) and the few-step
# variant (ddim-25-eta1), Smart Init assigned with pot-exact.
#   sbatch experiments/grid_size_exact_ot.sbatch
# Then:
#   python experiments/grid_size_results_stage_2.py --fewstep-dir experiments/outputs/grid_size_results/exact_ot \
#       --sampler ddpm-500 -o experiments/outputs/grid_size_results/pointcount_generalization_exactOT.pdf
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name gridsize-exact
#SBATCH --output logs/gridsize-exact-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
SRC=experiments/outputs/grid_size_results/source
OUT=experiments/outputs/grid_size_results/exact_ot
SAMPLERS=${SAMPLERS:-ddpm-500,ddim-25-eta1}
GRIDS=${GRIDS:-"24 32 48 64 80"}

for G in $GRIDS; do
    echo "== grid $G ($((G * G)) points) $(date)"
    python -u experiments/few_step_sampling_stage_1.py \
        --control_ckpt_path "$CKPT" --label CN-GBN --source "$SRC" --output "$OUT" \
        --grid_size "$G" --samplers "$SAMPLERS" --ot_solver pot-exact --png-count 2 --no-throughput
done
echo "$(date) done"
