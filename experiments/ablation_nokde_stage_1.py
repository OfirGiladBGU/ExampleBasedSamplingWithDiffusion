#!/usr/bin/env python3
"""E8 (KDE-loss ablation), stage 1: export point sets exactly like Table 2's "Full + SDEdit" row.

Runs ablation_advance_metrics_stage_1.py with the settings of its "Full + SDEdit" block (set here
on the module, so the shared script is not edited) and the same 50 validation images:
    --model nokde      control_v4/train_outputs_Icons-50_1024_GBN_noKDE/checkpoints, every 500 epochs
                       -> OUTPUT_DIR/sdedit_noKDE/epoch_*_npy
    --model reference  the paper's full checkpoint at epoch 5000, re-sampled with the same noise
                       -> OUTPUT_DIR/sdedit_seeded/epoch_5000_npy
    --model nogate     control_v4/train_outputs_Icons-50_1024_GBN_noGate/checkpoints (reviewer W5:
                       the full run trained without gated injection), every 500 epochs
                       -> OUTPUT_DIR/sdedit_noGate/epoch_*_npy

Table 2's runs did not seed the start / step noise, so its numbers carry sampling noise of their own
(sdedit δc 0.0273 at epoch 4500 vs 0.0255 at 5000). Here every image is sampled right after
torch.manual_seed(crc32(stem) + NOISE_SEED), identically for both models, so stage 3 can compare
noKDE and the full model per image (paired).

OUTPUT_DIR is its own folder (so the Table 2 plots of stage 4 do not pick these runs up); its
resources/ is a symlink to the Table 2 one (same manifest and source images).

    python experiments/ablation_nokde_stage_1.py --model nokde
    python experiments/ablation_nokde_stage_1.py --model reference
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_advance_metrics_stage_1 as s1  # noqa: E402

OUTPUT_DIR = "experiments/outputs/ablation_nokde_e500_b50_1024"
MODELS = {  # name: (weights dir, results dir, checkpoint filter, gated injection)
    "nokde": ("control_v4/train_outputs_Icons-50_1024_GBN_noKDE/checkpoints", "sdedit_noKDE", "all", True),
    "reference": ("control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints", "sdedit_seeded",
                  "dynamic_ep5000", True),
    "nogate": ("control_v4/train_outputs_Icons-50_1024_GBN_noGate/checkpoints", "sdedit_noGate", "all", False),
}
NOISE_SEED = 0


def configure(weights_dir, results_dir, gate_injection=True):
    """The "Full + SDEdit" block of ablation_advance_metrics_stage_1.py."""
    s1.WEIGHTS_DIR = weights_dir
    s1.RESULTS_DIR = results_dir
    s1.BASE_CKPT_PATH = ""
    s1.FREEZE_DENOISER = False
    s1.ENABLE_GECCO = True
    s1.ENABLE_ADAPTIVE_GATE_INJECTION = gate_injection  # False only for the noGate model
    s1.INFER_TRUNCATION_RATIO = 0.5


def seeded(process_single_image):
    def run(*args, image_path, **kwargs):
        seed = (zlib.crc32(Path(image_path).stem.encode()) + NOISE_SEED) % 2 ** 31
        torch.manual_seed(seed)
        np.random.seed(seed)
        return process_single_image(*args, image_path=image_path, **kwargs)
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--output", default=OUTPUT_DIR)
    args, rest = ap.parse_known_args()

    weights_dir, results_dir, ckpt_filter, gate_injection = MODELS[args.model]
    configure(weights_dir, results_dir, gate_injection)
    s1.process_single_image = seeded(s1.process_single_image)
    sys.argv = [sys.argv[0], "--output", args.output, "--checkpoints", ckpt_filter, *rest]
    return s1.main()


if __name__ == "__main__":
    raise SystemExit(main())
