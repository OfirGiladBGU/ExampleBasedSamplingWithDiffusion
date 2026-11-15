#!/bin/bash
# Generic rebuttal sampling job: one checkpoint on one image folder with few_step_sampling_stage_1.py.
# Default samplers: the paper's ddpm-500 and the few-step ddim-25-eta1, from the same Smart Init and noise.
#   sbatch experiments/rebuttal_sample.sbatch <ckpt> <label> <source> <output> <grid> [extra stage-1 args]
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name rebuttal-sample
#SBATCH --output logs/rebuttal-sample-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

CKPT=${1:?checkpoint}; LABEL=${2:?label}; SRC=${3:?source}; OUT=${4:?output}; G=${5:?grid}
shift 5
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST | $LABEL | $SRC | grid $G | $*"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

python -u experiments/few_step_sampling_stage_1.py \
    --control_ckpt_path "$CKPT" --label "$LABEL" --source "$SRC" --output "$OUT" --grid_size "$G" \
    --samplers ddpm-500,ddim-25-eta1 --png-count 16 --no-throughput "$@"
echo "$(date) done"
