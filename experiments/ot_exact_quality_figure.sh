#!/bin/bash
# Capacity-generalization figure images (grid_size_results/source) at 2304-6400 points, one solver.
#   sbatch --partition main --time 0-03:00:00 --gpus=rtx_6000:1 --cpus-per-task=4 --mem=24G \
#       --job-name ot-exact-fig --output logs/ot-exact-fig-%j.out experiments/ot_exact_quality_figure.sh <pot|pot-exact>
for G in 48 56 64 72 80; do
    bash experiments/ot_exact_quality.sbatch "$G" "$1" experiments/outputs/grid_size_results/source \
        experiments/outputs/ot_exact_quality-capacity_figure -1 2
done
