#!/bin/bash
# Seed robustness for the rebuttal (E14, reviewer W5 "single runs / within noise") and the Fig. 3
# strip percentages averaged over realizations (reviewer minor point).
#   task 0, 1  E14: Ours-GBN on the 1000 Icons-50 test images with a new Smart Init seed and new
#              noise seeds (seed 0 is the E1 run few_step_sampling-Icons-50_1024/CN-GBN_1024),
#              ddpm-500 and ddim-25-eta1 -> outputs/e14_seeds-Icons-50_1024/CN-GBN-seed<k>_1024
#   task 2     Fig. 3: the quadratic density gradient, 25 realizations (Smart Init and noise seeds)
#              of Ours-GBN and Ours-WVS -> outputs/fig3_strip_realizations/, strip masses by
#              fig3_strip_realizations.py; and the parameter counts (model_stats.py)
#   sbatch experiments/rebuttal_seeds.sbatch
# Then score tasks 0 / 1 with few_step_sampling_metrics.sbatch and run e14_summary.py.
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name rebuttal-seeds
#SBATCH --output logs/rebuttal-seeds-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2

set -eo pipefail
echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

GBN=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
WVS=control_v4/train_outputs_Icons-50_1024_WVS/checkpoints/dynamic_ep5000.ckpt
SAMPLERS=ddpm-500,ddim-25-eta1

if [ "$SLURM_ARRAY_TASK_ID" -lt 2 ]; then
    K=$((SLURM_ARRAY_TASK_ID + 1))
    python -u experiments/few_step_sampling_stage_1.py --control_ckpt_path "$GBN" \
        --label "CN-GBN-seed$K" --output experiments/outputs/e14_seeds-Icons-50_1024 --grid_size 32 \
        --samplers "$SAMPLERS" --ot_solver pot-exact --png-count 0 --no-throughput \
        --smart_init_seed $((42 + K)) --noise_seed $((K * 100000))
else
    SRC=experiments/outputs/images_results_metrics/quadratic/source
    OUT=experiments/outputs/fig3_strip_realizations
    for S in $(seq 0 24); do
        for M in GBN WVS; do
            CK=$GBN; [ "$M" = WVS ] && CK=$WVS
            python -u experiments/few_step_sampling_stage_1.py --control_ckpt_path "$CK" \
                --label "CN-$M-s$S" --source "$SRC" --output "$OUT" --grid_size 32 \
                --samplers "$SAMPLERS" --ot_solver pot-exact --png-count 0 --no-throughput \
                --smart_init_seed $((1000 + S)) --noise_seed $((S * 1000)) | tail -1
        done
    done
    python -u experiments/fig3_strip_realizations.py
    python -u experiments/model_stats.py
fi
echo "$(date) done"
