#!/bin/bash
# Runtime vs point budget of the new variant (ddim-25-eta1 + exact OT) for the supplement version of
# Fig. 7. Same GPU type, checkpoint and 10 pinned Icons-50 runtime images as runtime_fewstep.sbatch;
# only the Smart Init assignment changes (pot-exact). ddpm-500 is re-timed too, so the paper sampler
# with the exact OT is available on the same machine.
#   sbatch experiments/runtime_new_variant.sbatch
# Then: python experiments/runtime_new_variant_stage_3.py
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name runtime-newvar
#SBATCH --output logs/runtime-newvar-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
SRC=experiments/outputs/runtime_fewstep-Icons-50/source   # symlinks to the 10 pinned runtime images
OUT=${OUT:-experiments/outputs/runtime_new_variant-Icons-50}
SAMPLERS=${SAMPLERS:-ddim-25-eta1,ddpm-500}
GRIDS=${GRIDS:-"16 24 32 40 48 56 64 72 80"}    # 256 ... 6400 points

for G in $GRIDS; do
    echo "== grid $G ($((G * G)) points) $(date)"
    python -u experiments/few_step_sampling_stage_1.py \
        --control_ckpt_path "$CKPT" --label CN-GBN --source "$SRC" --output "$OUT" \
        --grid_size "$G" --samplers "$SAMPLERS" --ot_solver pot-exact --png-count 0 \
        --no-throughput --tag runtime
done
echo "$(date) done"
