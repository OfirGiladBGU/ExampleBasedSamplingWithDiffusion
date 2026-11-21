#!/bin/bash
# Runtime graphs on a new image set (user, 2026-11-19), CPU methods: BNOT and WVS, one after the
# other on ONE CPU-only AMD EPYC 7702P node (cpu128), 1 thread, 256 ... 6400 points, the first 10
# images (sorted) of the Icons-50 validation set, staged by budget_teachers_stage_0.py --prefix runtime_v10.
#   BNOT  true defaults (max_iters 500, max_newton_iters 500, epsilon 1.0), fixed build, one image per
#         call (bnot_retime_defaults.sbatch run as a script)
#   WVS   the paper's ICONS-TIMES-V2 parameters (budget_teachers_driver.py --method wvs)
# Output: experiments/outputs/runtime_v10_{bnot,wvs}*/timestamps_<M>_<N>/. Nothing else is touched.
#   sbatch experiments/runtime_v10_cpu.sbatch
# Then: python experiments/runtime_v10_stage_3.py (after runtime_v10_gpu.sbatch too)
#SBATCH --partition cpu
#SBATCH --time 1-00:00:00
#SBATCH --job-name runtime-v10-cpu
#SBATCH --output logs/runtime-v10-cpu-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --constraint=cpu128

R=/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
O=$R/experiments/outputs
BUDGETS=${BUDGETS:-256 576 1024 1600 2304 3136 4096 5184 6400}
CPU=$(lscpu | grep 'Model name' | sed 's/.*: *//')
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST | $CPU | load $(cut -d' ' -f1-3 /proc/loadavg)"
case "$CPU" in *7702P*) ;; *) echo "not an EPYC 7702P node: stopping"; exit 3 ;; esac
cd $R
module load anaconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

echo "===== BNOT (true defaults) $(date +%T)"
SRC=$O/runtime_v10_bnot_shard0 OUT=$O/runtime_v10_bnot_defaults500 BUDGETS="$BUDGETS" \
    bash experiments/bnot_retime_defaults.sbatch

echo "===== WVS $(date +%T) | load $(cut -d' ' -f1-3 /proc/loadavg)"
(
    source activate sd
    python -u experiments/archive/E4_budget_generalization_scripts/budget_teachers_driver.py \
        --method wvs --data_path $O/runtime_v10_wvs --budgets $BUDGETS
)
echo "load at end $(cut -d' ' -f1-3 /proc/loadavg)"
echo "$(date) done"
