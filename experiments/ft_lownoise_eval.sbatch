#!/bin/bash
# Evaluate a low-noise fine-tune (ft_lownoise_train.sbatch) on the 200 held-out diagnostic images
# (outputs/diag_generalization/test_*), with the same Smart Init and noise seeds as the baseline
# runs CN-GBN-test_1024 and CN-GBN-test-fromGT_1024, so every comparison is paired per image:
#   normal sampling (ddpm-500, ddim-25-eta1), start from the teacher's exemplar noised to step K,
#   and the denoising error per noise level.
#   sbatch --dependency=aftercorr:<train job> experiments/ft_lownoise_eval.sbatch
#   sbatch --array=2 --export=ALL,EPOCH=5100 experiments/ft_lownoise_eval.sbatch   # one variant, other epoch
# Then score CN-GBN-test-ft-<tag>_1024 and CN-GBN-test-ft-<tag>-fromGT_1024 with
# few_step_sampling_metrics.sbatch and run ft_lownoise_summary.py.
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name ft-lownoise-eval
#SBATCH --output logs/ft-lownoise-eval-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2

set -e
TAGS=(g5 g50 g0)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
EPOCH=${EPOCH:-5300}
RUNTAG=$TAG                                   # epoch 5300 keeps the plain name ft-<tag>
[ "$EPOCH" = 5300 ] || RUNTAG="$TAG-ep$EPOCH"   # other epochs: ft-<tag>-ep<epoch> (trend runs)
echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST | ft-$TAG epoch $EPOCH"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

OUT=experiments/outputs/diag_generalization
CKPT=control_v4/train_outputs_Icons-50_1024_GBN_ft-$TAG/checkpoints/dynamic_ep$EPOCH.ckpt
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }

python -u experiments/few_step_sampling_stage_1.py --control_ckpt_path "$CKPT" \
    --label "CN-GBN-test-ft-$RUNTAG" --source "$OUT/test_source" --output "$OUT" --grid_size 32 \
    --samplers ddpm-500,ddim-25-eta1 --ot_solver pot-exact --png-count 4 --no-throughput
python -u experiments/few_step_sampling_stage_1.py --control_ckpt_path "$CKPT" \
    --label "CN-GBN-test-ft-$RUNTAG-fromGT" --source "$OUT/test_source" --output "$OUT" --grid_size 32 \
    --samplers ddpm-10,ddpm-20,ddpm-50,ddpm-100,ddpm-500 --init_dir "$OUT/test_init" \
    --ot_solver pot-exact --png-count 4 --no-throughput
python -u experiments/diag_generalization_timestep_loss.py --control_ckpt_path "$CKPT" \
    --name "timestep_loss_ft-$RUNTAG"
echo "$(date) done"
