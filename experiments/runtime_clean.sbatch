#!/bin/bash
# Clean runtime measurement of our sampler for the runtime graph (Fig. 7, new variant).
# Why: runtime_new_variant.sbatch (job 23736983) shared ise-6000-09 with the grid-size job, and its
# Smart Init times at 3136-6400 points are inflated (5184: 9.3 s vs 3.9 s on an exclusive node).
# Protocol:
#   * one exclusive single-GPU cpu256 node (RTX 6000 Ada), nothing else running on it;
#   * CPU thread pools pinned to THREADS (default 16) and recorded;
#   * ddim-25-eta1 and ddpm-500, Smart Init with the exact OT assignment (pot-exact);
#   * the 10 pinned runtime images, 256 ... 6400 points, warm-up per budget (stage 1);
#   * REPS full repetitions (default 3), budgets in order inside each repetition, each repetition in
#     its own folder rep<k>/ (stage 1 skips existing outputs, so reps cannot overwrite each other).
# Per image the summary takes the median over the repetitions (runtime_clean_stage_3.py).
#   sbatch experiments/runtime_clean.sbatch
# Then: python experiments/runtime_clean_stage_3.py
# OT_SOLVER (default pot-exact) picks the Smart Init assignment, e.g. bisection-blocks16 above 2304 points.
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name runtime-clean
#SBATCH --output logs/runtime-clean-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --constraint=cpu256&rtx_6000
#SBATCH --exclusive
#SBATCH --mem=64G

THREADS=${THREADS:-16}
REPS=${REPS:-3}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST | threads $THREADS | reps $REPS"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|MHz"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,clocks.max.sm --format=csv
echo "other processes on the GPU:"; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
echo "load: $(cat /proc/loadavg)"
echo "git: $(git rev-parse --short HEAD)"

CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
SRC=experiments/outputs/runtime_fewstep-Icons-50/source   # symlinks to the 10 pinned runtime images
OUT=${OUT:-experiments/outputs/runtime_clean-Icons-50}
SAMPLERS=${SAMPLERS:-ddim-25-eta1,ddpm-500}
GRIDS=${GRIDS:-"16 24 32 40 48 56 64 72 80"}    # 256 ... 6400 points

for R in $(seq 1 "$REPS"); do
    for G in $GRIDS; do
        echo "== rep $R grid $G ($((G * G)) points) $(date) load $(cut -d' ' -f1-3 /proc/loadavg)"
        python -u experiments/few_step_sampling_stage_1.py \
            --control_ckpt_path "$CKPT" --label CN-GBN --source "$SRC" --output "$OUT/rep$R" \
            --grid_size "$G" --samplers "$SAMPLERS" --ot_solver "${OT_SOLVER:-pot-exact}" --png-count 0 \
            --no-throughput --tag runtime
    done
done
echo "$(date) done"
