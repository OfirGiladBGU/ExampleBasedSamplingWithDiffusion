#!/bin/bash
# Stage 1 of the video pipeline on the cluster: sampler trajectories + tone snapshots.
# Submit from the project root:   sbatch video/server/run_stage_1.sbatch
# Then copy video/assets/trajectories/ back to the local checkout.
#SBATCH --job-name=video_stage1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=video/server/stage1_%j.log
# Add the partition/account lines your group uses, e.g.
# #SBATCH --partition=main

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stippling

python video/stage_1_trajectories.py "$@"
python video/stage_1_tone.py
