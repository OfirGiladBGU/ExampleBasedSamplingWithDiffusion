#!/bin/bash
# Fig. 7, BNOT curve with BNOT's true defaults (user, 2026-11-19): the fixed BNOT_new build
# (SuiteSparse QR), max_iters 500, max_newton_iters 500, epsilon 1.0 (the original ibnot defaults),
# instead of the paper driver's 25 / 50 caps. Same protocol as the current fixed-build curve
# (icons_results_runtimes_bnot): the 10 pinned Icons-50 runtime images, 512 x 512, seed 7, invert,
# 1 thread, CPU-only EPYC 7702P node (cpu128). One image per call, so a stall costs one image; a
# stopped image is reported, never silently dropped.
# Output: experiments/outputs/icons_results_runtimes_bnot_defaults500/ (nothing else is touched).
#   sbatch experiments/bnot_retime_defaults.sbatch
#   sbatch --export=ALL,BUDGETS="7744 9216 10816 12544" experiments/bnot_retime_defaults.sbatch
# Then: python experiments/bnot_defaults_runtime_stage_3.py
#SBATCH --partition cpu
#SBATCH --time 0-12:00:00
#SBATCH --job-name bnot-time500
#SBATCH --output logs/bnot-time500-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=cpu128

R=/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
SRC=${SRC:-$R/experiments/outputs/icons_results_runtimes_bnot}
OUT=${OUT:-$R/experiments/outputs/icons_results_runtimes_bnot_defaults500}
BNOT=/groups/asharf_group/ofirgila/BNOT_new/bnot_data_gen.py
BUDGETS=${BUDGETS:-256 576 1024 1600 2304 3136 4096 5184 6400}
MAX_ITERS=${MAX_ITERS:-500}
MAX_NEWTON=${MAX_NEWTON:-500}
EPS=${EPS:-1.0}
LIMIT=${LIMIT:-3600}           # seconds per image; only a guard against a stall
CPU=$(lscpu | grep 'Model name' | sed 's/.*: *//')
echo "$(date) | node $SLURM_JOB_NODELIST | $CPU | load $(cut -d' ' -f1-3 /proc/loadavg) | budgets $BUDGETS"
echo "max_iters $MAX_ITERS | max_newton_iters $MAX_NEWTON | epsilon $EPS | limit $LIMIT s per image"
case "$CPU" in *7702P*) ;; *) echo "not an EPYC 7702P node (the current BNOT curve's CPU): stopping"; exit 3 ;; esac
cd $R
module load anaconda
source activate bnot
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $OUT
[ -d $OUT/original ] || cp -r $SRC/original $OUT/original
[ -d $OUT/source ] || cp -r $SRC/source $OUT/source
echo "staged: $(find $OUT/original -type f | wc -l) originals | binary: $(ls -la /groups/asharf_group/ofirgila/BNOT_new/ibnot_cli/build/ibnot_new_cli)"

for N in $BUDGETS; do
    if [ -d $OUT/timestamps_BNOT_$N ]; then echo "== $N: already done"; continue; fi
    echo "== $N points | $(date +%T) | load $(cut -d' ' -f1 /proc/loadavg)"
    rm -rf $OUT/timestamps $OUT/target_BNOT_$N
    for i in 0 1 2 3 4 5 6 7 8 9; do
        timeout -k 10 $LIMIT python -u $BNOT --data_path $OUT --target_folder target_BNOT_$N --n -1 --start $i --end $((i + 1)) \
            --image_size 512 512 --num_sites $N --seed 7 --max_iters $MAX_ITERS --max_newton_iters $MAX_NEWTON \
            --epsilon $EPS --invert --no-apply_preprocess --no-keep_pgm --keep_stats --track_time --overwrite \
            > $OUT/last_call.log 2>&1
        rc=$?
        if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "   image $i: STOPPED after $LIMIT s ($(grep -o 'Processing .*' $OUT/last_call.log | tail -1))"
        elif [ $rc -ne 0 ]; then echo "   image $i: exit $rc"; tail -2 $OUT/last_call.log; fi
    done
    mkdir -p $OUT/timestamps
    mv $OUT/timestamps $OUT/timestamps_BNOT_$N
    echo "   $(find $OUT/timestamps_BNOT_$N -name '*.txt' | wc -l) of 10 timings | $(date +%T)"
done
echo "load at end $(cut -d' ' -f1-3 /proc/loadavg)"
echo "$(date) done"
