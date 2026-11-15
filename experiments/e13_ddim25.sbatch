#!/bin/bash
# E13: the new variant (ddim-25-eta1, SDEdit start, exact OT) for all 12 checkpoints of Table 1
# (4 test sets x Ours-WVS / Ours-GBN / Ours-BNOT), one array task per checkpoint.
# -> experiments/outputs/e13_ddim25-<dataset>/CN-<sampler>_<N>/ddim-25-eta1/
#   sbatch experiments/e13_ddim25.sbatch
# Then per task the scorer, e.g.
#   sbatch --dependency=afterok:<job>_<task> --export=ALL,ROOT=experiments/outputs/e13_ddim25-<dataset>,\
#       RUN=CN-<sampler>_<N>,SRC=experiments/outputs/z_validation_data/<dataset>/source \
#       experiments/few_step_sampling_metrics.sbatch
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name e13-ddim25
#SBATCH --output logs/e13-ddim25-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-11

DATASETS=(Icons-50_1024 CelebA-5K_1024 ShapeNet2D-4K_1600 Anime-Chibi_1600)
SAMPLERS=(WVS GBN BNOT)
D=${DATASETS[$((SLURM_ARRAY_TASK_ID / 3))]}
M=${SAMPLERS[$((SLURM_ARRAY_TASK_ID % 3))]}
N=${D##*_}
G=$(python3 -c "print(int(round($N ** 0.5)))")
CKPT=control_v4/train_outputs_${D}_${M}/checkpoints/dynamic_ep5000.ckpt
[ "$D" = Icons-50_1024 ] && [ "$M" = GBN ] && CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt

echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST | $D $M grid $G | $CKPT"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

python -u experiments/few_step_sampling_stage_1.py \
    --control_ckpt_path "$CKPT" --label "CN-$M" \
    --source "experiments/outputs/z_validation_data/$D/source" --output "experiments/outputs/e13_ddim25-$D" \
    --grid_size "$G" --samplers ddim-25-eta1 --ot_solver pot-exact --png-count 16 --no-throughput
echo "$(date) done"
