#!/bin/bash
# Evaluate a tone stage-1 run through the full sampler (stage 5): write the requested densities
# as condition images, sample them with ddpm-500 and ddim-25-eta1 for 3 noise seeds, with the
# Smart Init drawn from the requested density and (initrho0) from the original target, then summarize.
#   sbatch --partition main --gpus=rtx_6000:1 ... experiments/tone_full_sampler_eval.sh <stage-1 dir> <output dir>
STAGE1=${1:?stage-1 dir}; OUT=${2:?output dir}
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
CK=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
python -u experiments/tone_results_stage_5_full_sampler.py --prepare --stage1 "$STAGE1" --output "$OUT"
for S in 0 1 2; do
    for C in none curve field; do
        python -u experiments/few_step_sampling_stage_1.py --control_ckpt_path $CK --label $C-seed$S \
            --source $OUT/conditions/$C --output $OUT/samples --grid_size 32 --samplers ddpm-500,ddim-25-eta1 \
            --noise_seed $S --smart_init_seed $((42 + S)) --png-count 4 --no-throughput
    done
    for C in curve field; do
        python -u experiments/few_step_sampling_stage_1.py --control_ckpt_path $CK --label $C-initrho0-seed$S \
            --source $OUT/conditions/$C --smart_init_source $OUT/conditions/none --output $OUT/samples \
            --grid_size 32 --samplers ddpm-500,ddim-25-eta1 \
            --noise_seed $S --smart_init_seed $((42 + S)) --png-count 4 --no-throughput
    done
done
python -u experiments/tone_results_stage_5_full_sampler.py --summarize --stage1 "$STAGE1" --output "$OUT" \
    --configs none,curve,field,curve-initrho0,field-initrho0
echo "$(date) done"
