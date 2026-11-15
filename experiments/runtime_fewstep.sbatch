#!/bin/bash
# Runtime vs point budget for the few-step samplers (rebuttal E1, Fig. 7 update).
# One job on one GPU times the paper sampler (ddpm-500) and the few-step samplers on the same
# 10 pinned Icons-50 runtime images (merge_runtime_results_stage_0.IMAGES) at every Fig. 7 budget
# from 256 to 6400 points, so the timings are directly comparable. Ours-GBN, as in the paper.
#   sbatch experiments/runtime_fewstep.sbatch
# Then: python experiments/runtime_fewstep_stage_2.py
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name runtime-fewstep
#SBATCH --output logs/runtime-fewstep-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
OUT=experiments/outputs/runtime_fewstep-Icons-50
SRC=$OUT/source                       # symlinks to the 10 pinned runtime images
SAMPLERS=ddpm-500,ddim-25-eta1,ddim-10-eta1
GRIDS="16 24 32 40 48 56 64 72 80"    # 256 ... 6400 points

for G in $GRIDS; do
    echo "== grid $G ($((G * G)) points) $(date)"
    python -u experiments/few_step_sampling_stage_1.py \
        --control_ckpt_path "$CKPT" --label CN-GBN --source "$SRC" --output "$OUT" \
        --grid_size "$G" --samplers "$SAMPLERS" --png-count 0 \
        --batch-sizes 1,8,32 --throughput-images 2 --tag runtime
done
echo "$(date) done"
