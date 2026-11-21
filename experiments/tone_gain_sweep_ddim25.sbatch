#!/bin/bash
# Ink-gain sweep on the real sampler (rebuttal; full-chain version of the supplement's gain sweep).
# Same size as the submitted sweep (12 targets, 150 optimizer steps, gains x0.25..x4), but the
# gradient goes through the DDIM-25 chain (stage 1 --mode ddim, as E16) and every condition is
# scored on ddpm-500 stipples with fresh seeds 0/1/2, rendered at its own gain.
#   sbatch experiments/tone_gain_sweep_ddim25.sbatch
#SBATCH --partition main
#SBATCH --time 0-10:00:00
#SBATCH --job-name tone-gain-ddim25
#SBATCH --output logs/tone-gain-ddim25-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -e
SWEEP=experiments/outputs/tone_results/gain_sweep_ddim25
CKPT=control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt
STEPS=150
LIMIT=12
SEEDS="0 1 2"
CONFIGS="none curve field"
SAMPLER=ddpm-500
GRID=32

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

# 1. optimize the densities through the DDIM-25 chain at every gain
python -u experiments/tone_results_stage_2_gain_sweep.py --mode ddim --output "$SWEEP" \
    --steps $STEPS --limit $LIMIT --control_ckpt_path "$CKPT"

# 2. condition images
python -u experiments/tone_results_stage_10_gain_sweep_fullchain.py --prepare --sweep "$SWEEP"

# 3. real sampler, fresh seeds
for D in "$SWEEP"/gain_*x; do
  for C in $CONFIGS; do
    for S in $SEEDS; do
      echo "$(date) | sample $D $C seed $S"
      python -u experiments/few_step_sampling_stage_1.py --control_ckpt_path "$CKPT" \
          --label "$C-seed$S" --source "$D/conditions/$C" --output "$D/samples" --grid_size $GRID \
          --samplers $SAMPLER --noise_seed $S --png-count 0 --no-throughput
    done
  done
done

# 4. score at each gain, table and figure
python -u experiments/tone_results_stage_10_gain_sweep_fullchain.py --summarize --sweep "$SWEEP"
echo "$(date) done"
