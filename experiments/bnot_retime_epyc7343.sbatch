#!/bin/bash
# Paper Fig. 7, BNOT curve: re-time the fixed BNOT build (BNOT_new commit 7ba8a7c, SuiteSparse QR)
# on the machine type the paper states for its runtimes (RTX 6000 node, AMD EPYC 7343), with the
# paper's protocol: the 10 pinned Icons-50 runtime images, 512 x 512, seed 7, max_iters 25,
# max_newton_iters 50, epsilon 1.0, invert, one image at a time, 1 thread.
# The first fixed-build timing (icons_results_runtimes_bnot) ran on an AMD EPYC 7702P node.
# Output: experiments/outputs/icons_results_runtimes_bnot_epyc7343/ (nothing else is touched).
# A GPU is requested only because non-GPU jobs are routed away from the RTX 6000 nodes; BNOT does
# not use it. The job stops (exit 3) if the node's CPU is not an EPYC 7343: resubmit then.
#   sbatch experiments/bnot_retime_epyc7343.sbatch
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name bnot-retime-7343
#SBATCH --output logs/bnot-retime-7343-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

R=/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
OLD=$R/experiments/outputs/icons_results_runtimes_bnot
NEW=$R/experiments/outputs/icons_results_runtimes_bnot_epyc7343
BUDGETS=${BUDGETS:-256 576 1024 1600 2304 3136 4096 5184 6400}
CPU=$(lscpu | grep 'Model name' | sed 's/.*: *//')
echo "$(date) | node $SLURM_JOB_NODELIST | $CPU | load $(cut -d' ' -f1-3 /proc/loadavg) | budgets $BUDGETS"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
case "$CPU" in *7343*) ;; *) echo "not an EPYC 7343 node: stopping"; exit 3 ;; esac
cd $R
module load anaconda
source activate bnot
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $NEW
[ -d $NEW/original ] || cp -r $OLD/original $NEW/original
[ -d $NEW/source ] || cp -r $OLD/source $NEW/source
echo "staged: $(find $NEW/original -type f | wc -l) originals, $(find $NEW/source -type f | wc -l) sources"
echo "binary: $(ls -la /groups/asharf_group/ofirgila/BNOT_new/ibnot_cli/build/ibnot_new_cli)"
python -u experiments/archive/E4_budget_generalization_scripts/budget_teachers_driver.py \
    --method bnot --data_path $NEW --budgets $BUDGETS
echo "== native_wall_seconds per budget: mean, median, max, images (EPYC 7343) | mean on the EPYC 7702P run"
for N in $BUDGETS; do
    for D in $NEW $OLD; do
        find $D/timestamps_BNOT_$N -name '*.txt' -exec grep -h native_wall_seconds {} + | awk '{print $2}' | sort -n |
            awk '{a[NR]=$1; s+=$1} END {if (NR) printf "%.2f %.2f %.2f %d", s/NR, (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2), a[NR], NR}'
        echo -n " | "
    done | sed "s/^/$N: /"
    echo
done
echo "load at end $(cut -d' ' -f1-3 /proc/loadavg)"
echo "$(date) done"
