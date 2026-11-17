#!/usr/bin/env python3
"""Generalization diagnostics C: denoising error per noise level, training vs held-out images.

For each staged image (diag_generalization_stage_0.py) the teacher's exemplar offsets x0 are noised
to step t, x_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps, and the model predicts eps (the training
objective, same conditioning as inference). Reported per t, for the training and the held-out
images:
    eps MSE                       the unweighted training loss at that step
    x0 RMS error (cells)          sqrt((1 - a_t) / a_t * MSE): how far, in grid-cell widths, the
                                  model's estimate of the clean point positions is off
    Min-SNR weight                min(SNR, gamma) / SNR, the weight this step has in training

A held-out / training ratio well above 1 at some t means the model has memorised that noise level;
a large x0 error at small t on both means the fine positions are not learned at all.

Writes <output>/timestep_loss.md and timestep_loss.json.

    python experiments/diag_generalization_timestep_loss.py
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.few_step_sampling_stage_1 import (  # noqa: E402
    BASE_CONFIG_PATH, CONTROL_CKPT, ENABLE_ADAPTIVE_GATE_INJECTION, ENABLE_GECCO, EVAL_TIMESTEPS,
    eps_model, load_pipeline, prepare_image,
)
from control_v4.smart_init import add_noise_at_t  # noqa: E402

OUTPUT_DIR = "experiments/outputs/diag_generalization"
T_LIST = [2, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 950]
REPEATS = 4
MIN_SNR_GAMMA = 5.0
GRID_SIZE = 32


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--control_ckpt_path", default=CONTROL_CKPT)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--name", default="timestep_loss", help="output file stem (for other checkpoints)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.output)

    diffusion, control_net = load_pipeline(BASE_CONFIG_PATH, "", a.control_ckpt_path, GRID_SIZE, ENABLE_GECCO,
                                           ENABLE_ADAPTIVE_GATE_INJECTION, False, False, False, a.device)
    diffusion.set_num_timesteps(EVAL_TIMESTEPS)
    diffusion.eval()
    ac = diffusion.alphas_cumprod

    res = {}
    for split in ("train", "test"):
        images = sorted((out / f"{split}_source").glob("*.png"))
        if a.limit > 0:
            images = images[: a.limit]
        args = SimpleNamespace(grid_size=GRID_SIZE, device=a.device, smart_init_source="",
                               init_dir=str(out / f"{split}_init"), noise_seed=0)
        mse = np.zeros((len(images), len(T_LIST)))
        for i, img in enumerate(images):
            x0, _, _ = prepare_image(img, diffusion, control_net, args, i, batch=REPEATS)
            gen = torch.Generator().manual_seed(1000 + i)
            with torch.no_grad():
                for j, t in enumerate(T_LIST):
                    eps = torch.randn(x0.shape, generator=gen).to(a.device)
                    x_t = add_noise_at_t(x0, ac[t], noise=eps)
                    pred = eps_model(diffusion, x_t, t)
                    mse[i, j] = float(((pred - eps) ** 2).mean())
            if (i + 1) % 50 == 0:
                print(f"{split} {i + 1}/{len(images)}", flush=True)
        res[split] = mse

    lines = ["# Denoising error per noise level: training vs held-out images (Ours-GBN, Icons-50)", "",
             f"Checkpoint: `{a.control_ckpt_path}`", "",
             f"{res['train'].shape[0]} training and {res['test'].shape[0]} held-out images, {REPEATS} noise draws each.",
             "x0 RMS error is in grid-cell widths (1 cell = 1/32 of the image side).", "",
             "| t | noise std (cells) | Min-SNR weight | eps MSE train | eps MSE held-out | held-out / train | "
             "x0 RMS err train (cells) | x0 RMS err held-out (cells) |", "|---|---|---|---|---|---|---|---|"]
    table = []
    for j, t in enumerate(T_LIST):
        at = float(ac[t])
        snr = at / (1 - at)
        w = min(snr, MIN_SNR_GAMMA) / snr
        mt, mv = res["train"][:, j].mean(), res["test"][:, j].mean()
        xt, xv = np.sqrt((1 - at) / at * mt), np.sqrt((1 - at) / at * mv)
        table.append({"t": t, "noise_std": float(np.sqrt(1 - at)), "min_snr_weight": w, "mse_train": mt,
                      "mse_test": mv, "x0_rms_train": float(xt), "x0_rms_test": float(xv)})
        lines.append(f"| {t} | {np.sqrt(1 - at):.3f} | {w:.3f} | {mt:.4f} | {mv:.4f} | {mv / mt:.2f} | "
                     f"{xt:.3f} | {xv:.3f} |")
    (out / f"{a.name}.md").write_text("\n".join(lines) + "\n")
    (out / f"{a.name}.json").write_text(json.dumps(table, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
