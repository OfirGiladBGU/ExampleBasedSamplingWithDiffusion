#!/bin/bash
# BNOT convergence check (user question 2026-11-18): the paper runs BNOT with max_iters 25 and
# epsilon 1.0. Is the 25-iteration cap what stops it, or the tolerance, and what would a more
# converged BNOT give? Fixed build (BNOT_new 7ba8a7c), the 10 pinned Icons-50 runtime images,
# 512 x 512, seed 7, invert, one image at a time, 1 thread. Configurations (bnot_convergence_summary.py):
#   eps1_mi25       epsilon 1.0,  max_iters 25,  max_newton_iters 50   the paper setting
#   eps1_mi500      epsilon 1.0,  500 / 500                            the original BNOT defaults
#   eps0p1_mi500    epsilon 0.1,  500 / 500
#   eps0p01_mi500   epsilon 0.01, 500 / 500
#   eps0p01_mi25    epsilon 0.01, 25 / 50                              tight tolerance, the paper's cap
# 1024 points: all five, scored with the paper metrics. 6400 points: timing for three of them.
# Output: experiments/outputs/bnot_convergence_check/ (summary.md). Nothing else is touched.
#   sbatch experiments/bnot_convergence_check.sbatch
#SBATCH --partition cpu
#SBATCH --time 0-10:00:00
#SBATCH --job-name bnot-converge
#SBATCH --output logs/bnot-converge-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --constraint=cpu128|cpu256

R=/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
SRC=$R/experiments/outputs/icons_results_runtimes_bnot
OUT=$R/experiments/outputs/bnot_convergence_check
DATA=$OUT/data
BNOT=/groups/asharf_group/ofirgila/BNOT_new/bnot_data_gen.py
echo "$(date) | node $SLURM_JOB_NODELIST | $(lscpu | grep 'Model name' | sed 's/.*: *//')"
cd $R
module load anaconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $DATA
[ -d $DATA/original ] || cp -r $SRC/original $DATA/original
[ -d $DATA/source ] || cp -r $SRC/source $DATA/source
echo "staged: $(find $DATA/original -type f | wc -l) originals | binary: $(ls -la /groups/asharf_group/ofirgila/BNOT_new/ibnot_cli/build/ibnot_new_cli)"

run() {  # tag epsilon max_iters max_newton_iters points
    # One image per call with a time limit: at a tight tolerance BNOT can stall for hours on an icon
    # with large empty regions (first attempt, job 24308577: waving hand, epsilon 0.1, > 8 min).
    local T=t_$1_$5 LIMIT=600
    [ "$5" -gt 2000 ] && LIMIT=1800
    if [ -d $DATA/timestamps_$1_$5 ]; then echo "== $T: already done"; return; fi
    echo "== $T: epsilon $2, max_iters $3, max_newton_iters $4, $5 points, limit $LIMIT s per image | $(date +%T)"
    rm -rf $DATA/timestamps $DATA/$T
    for i in 0 1 2 3 4 5 6 7 8 9; do
        timeout -k 10 $LIMIT python -u $BNOT --data_path $DATA --target_folder $T --n -1 --start $i --end $((i + 1))             --image_size 512 512 --num_sites $5 --seed 7 --max_iters $3 --max_newton_iters $4 --epsilon $2             --invert --no-apply_preprocess --no-keep_pgm --keep_stats --track_time --overwrite > $OUT/last_call.log 2>&1
        rc=$?
        if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "   image $i: STOPPED after $LIMIT s ($(grep -o 'Processing .*' $OUT/last_call.log | tail -1))"
        elif [ $rc -ne 0 ]; then echo "   image $i: exit $rc"; tail -2 $OUT/last_call.log; fi
    done
    echo "   $(find $DATA/$T -name '*.npy' | wc -l) of 10 point sets | $(date +%T)"
    mkdir -p $DATA/timestamps; mv $DATA/timestamps $DATA/timestamps_$1_$5
}

(  # each environment in its own subshell
    source activate bnot
    run eps1_mi25 1.0 25 50 1024
    run eps1_mi500 1.0 500 500 1024
    run eps0p1_mi500 0.1 500 500 1024
    run eps0p01_mi500 0.01 500 500 1024
    run eps0p01_mi25 0.01 25 50 1024
)

echo "== scoring the 1024-point sets | $(date +%T)"
mkdir -p $OUT/score_src
find $DATA/source -name '*.png' -exec cp -n {} $OUT/score_src/ \;
for T in eps1_mi25 eps1_mi500 eps0p1_mi500 eps0p01_mi500 eps0p01_mi25; do
    mkdir -p $OUT/score_1024/$T
    find $DATA/t_${T}_1024 -name '*.npy' -exec cp {} $OUT/score_1024/$T/ \;
done
(
    source activate qmcdiffusion
    python -u experiments/few_step_sampling_stage_2.py $OUT/score_1024 --source $OUT/score_src --workers 8 | tail -5
    python experiments/bnot_convergence_summary.py > /dev/null && echo "summary.md written (1024 points)"
)
(
    source activate bnot
    run eps1_mi25 1.0 25 50 6400
    run eps0p1_mi500 0.1 500 500 6400
    run eps0p01_mi500 0.01 500 500 6400
)
(
    source activate qmcdiffusion
    python experiments/bnot_convergence_summary.py
)
echo "$(date) done"
