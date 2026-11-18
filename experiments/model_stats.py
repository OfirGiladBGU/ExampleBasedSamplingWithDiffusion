#!/usr/bin/env python3
"""Parameter counts of the paper model (reviewer: "parameter count is not stated").

Loads the Ours-GBN Icons-50 checkpoint exactly as inference does and counts the parameters of the
base denoiser and of the control branch (GECCO features + gated injection).

Writes experiments/outputs/model_stats.md.

    python experiments/model_stats.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from control_v4.DynamicControlNet import DynamicControlledDenoiser  # noqa: E402
from experiments.few_step_sampling_stage_1 import (  # noqa: E402
    BASE_CONFIG_PATH, CONTROL_CKPT, ENABLE_ADAPTIVE_GATE_INJECTION, ENABLE_GECCO, load_pipeline,
)


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    diffusion, control_net = load_pipeline(BASE_CONFIG_PATH, "", CONTROL_CKPT, 32, ENABLE_GECCO,
                                           ENABLE_ADAPTIVE_GATE_INJECTION, False, False, False, "cuda")
    model = diffusion.model
    base = count(model.locked if isinstance(model, DynamicControlledDenoiser) else model)
    ctrl = count(control_net)
    lines = ["# Model size (Ours-GBN, Icons-50, 1024 points)", "",
             f"Checkpoint: `{CONTROL_CKPT}` ({Path(CONTROL_CKPT).stat().st_size / 1e6:.0f} MB on disk, "
             "includes the optimizer state)", "",
             "| part | parameters |", "|---|---|",
             f"| base denoiser (U-Net, jointly trained) | {base:,} ({base / 1e6:.1f} M) |",
             f"| control branch (GECCO features + gated injection) | {ctrl:,} ({ctrl / 1e6:.1f} M) |",
             f"| total | {base + ctrl:,} ({(base + ctrl) / 1e6:.1f} M) |", ""]
    Path("experiments/outputs/model_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
