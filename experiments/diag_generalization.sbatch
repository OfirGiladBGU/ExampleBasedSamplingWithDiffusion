#!/bin/bash
# Generalization diagnostics (why the learned sampler loses local regularity), Ours-GBN, Icons-50.
# Run diag_generalization_stage_0.py first. Array tasks:
#   0  A, training images:  normal sampling (Smart Init + SDEdit), ddpm-500 and ddim-25-eta1
#   1  A, held-out images:  the same
#   2  B, training images:  start from the teacher's exemplar noised to step K, DDPM from K
#   3  B, held-out images:  the same
#   4  C: denoising error per noise level, training vs held-out (diag_generalization_timestep_loss.py)
#   sbatch experiments/diag_generalization.sbatch
# Then score the four run folders with few_step_sampling_metrics.sbatch and run
# diag_generalization_summary.py.
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name diag-gen
#SBATCH --output logs/diag-gen-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-4

set -e
echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

OUT=experiments/outputs/diag_generalization
CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
FROM_GT=ddpm-10,ddpm-20,ddpm-50,ddpm-100,ddpm-200,ddpm-300,ddpm-500
LIMIT=${LIMIT:--1}

sample() {  # split, label suffix, samplers, extra args
    python -u experiments/few_step_sampling_stage_1.py --control_ckpt_path "$CKPT" \
        --label "CN-GBN-$1$2" --source "$OUT/$1_source" --output "$OUT" --grid_size 32 \
        --samplers "$3" --ot_solver pot-exact --png-count 4 --no-throughput --limit "$LIMIT" "${@:4}"
}
case $SLURM_ARRAY_TASK_ID in
    0) sample train "" ddpm-500,ddim-25-eta1 ;;
    1) sample test  "" ddpm-500,ddim-25-eta1 ;;
    2) sample train -fromGT "$FROM_GT" --init_dir "$OUT/train_init" ;;
    3) sample test  -fromGT "$FROM_GT" --init_dir "$OUT/test_init" ;;
    4) python -u experiments/diag_generalization_timestep_loss.py --limit "$LIMIT" ;;
esac
echo "$(date) done"
