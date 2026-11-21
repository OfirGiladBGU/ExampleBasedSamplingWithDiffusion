#!/bin/bash
# Does the BNOT solver fix change the exemplars? (user question 2026-11-18: must the BNOT exemplars
# be regenerated and Ours-BNOT retrained?) One array task per dataset: a random sample of the stored
# training exemplars is regenerated with the old binary (Eigen SparseQR) and with the fixed binary
# (SuiteSparse QR), same inputs and settings, and compared point by point and with the paper
# metrics. See experiments/bnot_fix_validation.py.
# Output: experiments/outputs/bnot_fix_validation/<dataset>/ and summary.md. Nothing else is touched.
#   sbatch experiments/bnot_fix_validation.sbatch
#SBATCH --partition cpu
#SBATCH --time 0-08:00:00
#SBATCH --job-name bnot-fixval
#SBATCH --output logs/bnot-fixval-%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-3
#SBATCH --constraint=cpu128|cpu256

NAMES=(Icons-50 CelebA-5K ShapeNet2D-4K Anime-Chibi)
D=${NAMES[$SLURM_ARRAY_TASK_ID]}
R=/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
OUT=$R/experiments/outputs/bnot_fix_validation
DATA=$OUT/$D/data
TRAIN=/groups/asharf_group/ofirgila/ControlNet/training
BNOT=/groups/asharf_group/ofirgila/BNOT_new/bnot_data_gen.py
NEWBIN=/groups/asharf_group/ofirgila/BNOT_new/ibnot_cli/build/ibnot_new_cli
OLDBIN=/groups/asharf_group/ofirgila/BNOT_new/ibnot_cli/build.eigen_20261118/ibnot_new_cli
echo "$(date) | task $SLURM_ARRAY_TASK_ID = $D | node $SLURM_JOB_NODELIST | $(lscpu | grep 'Model name' | sed 's/.*: *//')"
cd $R
module load anaconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ls -la $NEWBIN $OLDBIN || exit 1
rm -rf $OUT/$D
mkdir -p $OUT/$D

( source activate qmcdiffusion; python experiments/bnot_fix_validation.py stage --dataset $D ) || exit 1
N=$(python3 -c "import json; print(json.load(open('$OUT/$D/sample.json'))['points'])")
K=$(python3 -c "import json; print(len(json.load(open('$OUT/$D/sample.json'))['images']))")
FOLDER=$(python3 -c "import json; print(json.load(open('$OUT/$D/sample.json'))['dataset'])")
echo "points $N | images $K | dataset folder $FOLDER"

gen() {  # target_folder executable first_index last_index [--image_size W H]
    local T=$1 BIN=$2 A=$3 B=$4; shift 4
    for i in $(seq $A $B); do
        timeout -k 10 900 python -u $BNOT --data_path $DATA --target_folder $T --executable $BIN --n -1 \
            --start $i --end $((i + 1)) "$@" --num_sites $N --seed 7 --max_iters 25 --max_newton_iters 50 \
            --epsilon 1.0 --invert --no-apply_preprocess --no-keep_pgm --keep_stats --no-track_time --overwrite \
            > $OUT/$D/last_call.log 2>&1
        rc=$?
        if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "   $T image $i: STOPPED after 900 s"
        elif [ $rc -ne 0 ]; then echo "   $T image $i: exit $rc"; tail -3 $OUT/$D/last_call.log; fi
    done
    echo "   $T: $(find $DATA/$T -name '*.npy' 2>/dev/null | wc -l) point sets | $(date +%T)"
}

echo "== which input size reproduces the stored exemplars (fixed binary, first 2 images)"
(
    source activate bnot
    gen cand0 $NEWBIN 0 1
    gen cand1 $NEWBIN 0 1 --image_size 512 512
    gen cand2 $NEWBIN 0 1 --image_size 448 448
)
( source activate qmcdiffusion; python experiments/bnot_fix_validation.py choose --dataset $D ) || exit 1
SIZE=$(cat $OUT/$D/chosen.txt)
[ "$SIZE" = none ] && SIZEARGS="" || SIZEARGS="--image_size $SIZE"

echo "== old binary, then fixed binary, $K images, size args: '$SIZEARGS'"
(
    source activate bnot
    gen target_old $OLDBIN 0 $((K - 1)) $SIZEARGS
    gen target_new $NEWBIN 0 $((K - 1)) $SIZEARGS
)

echo "== scoring both sets | $(date +%T)"
mkdir -p $OUT/$D/score/old $OUT/$D/score/new $OUT/$D/score_src
for T in old new; do
    find $DATA/target_$T -name '*.npy' | while read f; do
        cp "$f" $OUT/$D/score/$T/
        rel=${f#$DATA/target_$T/}
        cp -n "$TRAIN/$FOLDER/source/${rel%.npy}.png" $OUT/$D/score_src/ 2>/dev/null
    done
done
echo "   $(ls $OUT/$D/score/old | wc -l) old, $(ls $OUT/$D/score/new | wc -l) new, $(ls $OUT/$D/score_src | wc -l) source images"
(
    source activate qmcdiffusion
    python -u experiments/few_step_sampling_stage_2.py $OUT/$D/score --source $OUT/$D/score_src --workers 8 | tail -3
    python experiments/bnot_fix_validation.py compare --dataset $D
    python experiments/bnot_fix_validation.py summary > /dev/null
)
echo "$(date) done"
