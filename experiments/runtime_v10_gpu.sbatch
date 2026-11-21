#!/bin/bash
# Runtime graphs on a new image set (user, 2026-11-19), GPU methods: GBN, then our sampler, in ONE job
# on ONE dedicated RTX 6000 Ada with THREADS reserved CPU cores. The node itself is shared with other
# users' jobs (user, 2026-11-20: no exclusive node, the wait was days); none of our own runs overlap:
# everything below runs one after the other. Load and GPU processes are logged per step.
# The first 10 images (sorted) of the Icons-50 validation set, 256 ... 6400 points.
#   GBN   the paper's ICONS-TIMES-V2 parameters (budget_teachers_driver.py --method gbn), env sd
#   ours  Ours-GBN epoch 5000, ddim-25-eta1 and ddpm-500, REPS repetitions, thread pools pinned to
#         THREADS; Smart Init assignment pot-exact at every budget, and bisection-blocks16 at
#         3136 ... 6400 (the variant runtime_clean_blocks16 uses there)
# Output: experiments/outputs/runtime_v10_gbn/timestamps_GBN_<N>/, runtime_v10_ours-Icons-50/rep<k>/,
# runtime_v10_ours_blocks16-Icons-50/rep<k>/. Nothing else is touched.
#   sbatch experiments/runtime_v10_gpu.sbatch
# Then: python experiments/runtime_v10_stage_3.py (after runtime_v10_cpu.sbatch too)
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name runtime-v10-gpu
#SBATCH --output logs/runtime-v10-gpu-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --constraint=rtx_6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

THREADS=${THREADS:-16}
REPS=${REPS:-3}
GRIDS=${GRIDS:-"16 24 32 40 48 56 64 72 80"}    # 256 ... 6400 points
BLOCKS_GRIDS=${BLOCKS_GRIDS:-"56 64 72 80"}     # 3136 ... 6400 points
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS

R=/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
O=experiments/outputs
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST | threads $THREADS | reps $REPS"
cd $R
module load anaconda
lscpu | grep -E "Model name|^CPU\(s\)|Socket"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
echo "other processes on the GPU:"; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
echo "load: $(cat /proc/loadavg) | git: $(git rev-parse --short HEAD)"

BUDGETS=""; for G in $GRIDS; do BUDGETS="$BUDGETS $((G * G))"; done
echo "===== GBN $(date +%T)"
(
    source activate sd
    python -u experiments/archive/E4_budget_generalization_scripts/budget_teachers_driver.py \
        --method gbn --data_path $O/runtime_v10_gbn --budgets $BUDGETS
)

echo "===== ours $(date +%T)"
source activate qmcdiffusion
CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
SRC=$O/runtime_v10_source
run_ours() {  # out solver grids
    for K in $(seq 1 "$REPS"); do
        for G in $3; do
            echo "== $2 rep $K grid $G ($((G * G)) points) $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg) | gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
            python -u experiments/few_step_sampling_stage_1.py \
                --control_ckpt_path "$CKPT" --label CN-GBN --source "$SRC" --output "$1/rep$K" \
                --grid_size "$G" --samplers ddim-25-eta1,ddpm-500 --ot_solver "$2" --png-count 0 \
                --no-throughput --tag runtime
        done
    done
}
run_ours $O/runtime_v10_ours-Icons-50 pot-exact "$GRIDS"
run_ours $O/runtime_v10_ours_blocks16-Icons-50 bisection-blocks16 "$BLOCKS_GRIDS"
echo "load at end: $(cat /proc/loadavg)"
echo "$(date) done"
