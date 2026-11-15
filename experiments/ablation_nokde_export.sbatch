#!/bin/bash
# Rebuttal E8: export the noKDE checkpoints (every 500 epochs) and the seeded full-model reference
# (epoch 5000) with the Table 2 "Full + SDEdit" settings, 50 validation images.
#   sbatch experiments/ablation_nokde_export.sbatch
# Then the scorer array: sbatch --dependency=afterok:<this job> experiments/ablation_nokde_metrics.sbatch
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name nokde-export
#SBATCH --output logs/nokde-export-%j.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion
nvidia-smi --query-gpu=name --format=csv

python -u experiments/ablation_nokde_stage_1.py --model reference
python -u experiments/ablation_nokde_stage_1.py --model nokde
echo "$(date) done"
