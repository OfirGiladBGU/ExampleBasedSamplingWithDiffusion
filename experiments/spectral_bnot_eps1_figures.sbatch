#!/bin/bash
# Spectral figures with BNOT at its default tolerance (epsilon 1.0) -- rebuttal, user request
# 2026-11-22. The BNOT column of the submitted spectral figures (target_BNOT_1024) was generated
# at epsilon 0.01, while Table 1's BNOT and the Ours-BNOT training exemplars use epsilon 1.0, so
# Ours-BNOT was being compared with a BNOT it was never trained on. This job only re-plots: the
# epsilon 1.0 point sets (target_BNOT_1024_eps1p0, 150 sources, 25 distinct realizations per case,
# generated 2026-09-04) and their stage-1 analysis (recomputed 2026-09-22) already exist.
#   columns: BNOT(eps 1.0) WVS GBN || Ours-BNOT Ours-WVS Ours-GBN
#   -> experiments/outputs/spectral_analysis/plots_full_bnot_eps1/
#   (plots_full/ = submitted figures, plots_full_bnot/ = epsilon 0.01 BNOT + Ours-BNOT: untouched)
# CPU only: no GPU is requested.
#   sbatch experiments/spectral_bnot_eps1_figures.sbatch
#SBATCH --partition cpu
#SBATCH --time 0-00:30:00
#SBATCH --job-name spectral-bnot-eps1
#SBATCH --output logs/spectral-bnot-eps1-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -eo pipefail
echo "$(date) | job $SLURM_JOB_ID | node $SLURM_JOB_NODELIST"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

S=experiments/outputs/spectral_analysis
python -u experiments/spectral_analysis_stage_2.py --output $S/plots_full_bnot_eps1 \
    --methods 'target_BNOT_1024_eps1p0:BNOT,target_WVS_1024:WVS,target_GBN_1024:GBN,||,target_CN-BNOT_1024:Ours-BNOT,target_CN-WVS_1024:Ours-WVS,target_CN-GBN_1024:Ours-GBN' \
    --invariance-methods target_CN-BNOT_1024,target_CN-WVS_1024,target_CN-GBN_1024
python -u experiments/spectral_anisotropy_summary.py
ls -la $S/plots_full_bnot_eps1 | head -40
echo "$(date) done"
