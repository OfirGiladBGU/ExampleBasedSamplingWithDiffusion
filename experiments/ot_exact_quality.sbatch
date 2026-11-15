#!/bin/bash
# Smart Init with the paper's truncated ot.emd vs the exact assignment, at large point budgets
# (rebuttal E18). Same checkpoint, Smart Init seed and start noise; only the assignment differs.
#   sbatch experiments/ot_exact_quality.sbatch <grid> <pot|pot-exact> [<source> <output> <limit> <png-count>]
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name ot-exact-quality
#SBATCH --output logs/ot-exact-quality-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

G=${1:?grid size}
SOLVER=${2:?pot or pot-exact}
SRC=${3:-experiments/outputs/z_validation_data/Icons-50_1024/source}
OUT=${4:-experiments/outputs/ot_exact_quality-Icons-50}
LIMIT=${5:-200}
PNGS=${6:-16}
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST | grid $G | $SOLVER | $SRC"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
python -u experiments/few_step_sampling_stage_1.py \
    --control_ckpt_path "$CKPT" --label "CN-GBN-$SOLVER" --source "$SRC" --output "$OUT" \
    --grid_size "$G" --samplers ddpm-500,ddim-25-eta1 --ot_solver "$SOLVER" \
    --limit "$LIMIT" --png-count "$PNGS" --no-throughput
echo "$(date) done"
