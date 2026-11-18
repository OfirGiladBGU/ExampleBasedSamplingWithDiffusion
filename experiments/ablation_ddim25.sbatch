#!/bin/bash
# Ablation table (Table 2) with the few-step sampler, for the supplement: every architecture variant
# at epoch 5000, sampled from the Smart Init + SDEdit start (t = 500) with ddim-25-eta1 (and with the
# paper's ddpm-500 for reference), on the Table 2 images (the first 50 of the validation manifest),
# same Smart Init and noise for every variant, so the variants are paired per image.
# Note: the paper's Table 2 samples its architecture rows from pure noise with 1000 DDPM steps and
# only its last row with SDEdit; here every row uses the SDEdit start.
#   task: 0 vanilla  1 unfrozen  2 gecco  3 agi  4 full  5 noGate  6 noKDE
#   sbatch experiments/ablation_ddim25.sbatch
# Then score each CN-<variant>_1024 with few_step_sampling_metrics.sbatch and run
# ablation_ddim25_summary.py.
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name ablation-ddim25
#SBATCH --output logs/ablation-ddim25-%A_%a.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-6

set -e
#          name      pretrained frozen base     gecco        gate
VARIANTS=("vanilla  config/GBN/model.ckpt  --no-gecco  --no-gate-injection"
          "unfrozen -                      --no-gecco  --no-gate-injection"
          "gecco    config/GBN/model.ckpt  --gecco     --no-gate-injection"
          "agi      config/GBN/model.ckpt  --no-gecco  --gate-injection"
          "full     -                      --gecco     --gate-injection"
          "noGate   -                      --gecco     --no-gate-injection"
          "noKDE    -                      --gecco     --gate-injection")
read -r NAME BASE GECCO GATE <<< "${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
[ "$BASE" = - ] && BASE=""
echo "$(date) | job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID | node $SLURM_JOB_NODELIST | $NAME base='$BASE' $GECCO $GATE"
cd /groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion
module load anaconda
source activate qmcdiffusion

OUT=experiments/outputs/ablation_ddim25_e5000_b50_1024
RES=experiments/outputs/ablation_advance_metrics_e500_b50_1024/resources
SRC=$OUT/source
# The Table 2 images: the first 50 names of the validation manifest (idempotent, safe in parallel).
python - "$RES" "$SRC" <<'EOF'
import json, os, sys
from pathlib import Path
res, src = Path(sys.argv[1]), Path(sys.argv[2])
src.mkdir(parents=True, exist_ok=True)
for name in json.loads((res / "validation_manifest.json").read_text())[:50]:
    link = src / name
    try:
        link.symlink_to((res / "source" / name).resolve())
    except FileExistsError:
        pass
print(len(list(src.iterdir())), "images staged")
EOF

python -u experiments/few_step_sampling_stage_1.py \
    --control_ckpt_path "control_v4/train_outputs_Icons-50_1024_GBN_$NAME/checkpoints/dynamic_ep5000.ckpt" \
    --base_ckpt_path "$BASE" $GECCO $GATE \
    --label "CN-$NAME" --source "$SRC" --output "$OUT" --grid_size 32 \
    --samplers ddim-25-eta1,ddpm-500 --ot_solver pot-exact --png-count 8 --no-throughput
echo "$(date) done"
