"""Stage 8: cost of one tone-correction optimizer step per gradient path (rebuttal D3).

Times the stage-1 optimizer step (forward through the sampler, render, backward to the density
field, Adam) for each gradient path on the same GPU and image, and records the peak GPU memory:
    onestep   the paper's one-step x0 estimate at t = 120
    ddim      the full DDIM-25 chain (E16), gradients through every step
The whole-run time per image follows as steps x seconds per step; the stored stage-1 runs give the
measured totals (summary.json "seconds"), which are added to the table for reference.

Usage:
    python experiments/tone_results_stage_8_cost.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tone_results_stage_1_optimize as stage1  # noqa: E402
from tone_results_utils import DensityField, build_stippler, render_darkness  # noqa: E402

ROOT_DIR = "experiments/outputs/tone_results"
OUT_DIR = f"{ROOT_DIR}/cost"
MANIFEST = f"{ROOT_DIR}/manifest.json"
MODES = "onestep,ddim"
STORED_RUNS = {"onestep": f"{ROOT_DIR}/main", "ddim": f"{ROOT_DIR}/ddim25"}
IMAGES = 2
WARMUP = 3
TIMED_STEPS = 20
STEPS = stage1.STEPS
INK_GAIN = 1.0378684401512146   # the calibrated gain of the stage-1 runs
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Per-step cost of the tone-correction gradient paths")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--modes", default=MODES)
    p.add_argument("--images", type=int, default=IMAGES)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--timed-steps", type=int, default=TIMED_STEPS)
    p.add_argument("--steps", type=int, default=STEPS, help="Optimizer steps of a full run (for the total)")
    p.add_argument("--ink-gain", type=float, default=INK_GAIN)
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


def time_mode(mode, a, images):
    sys.argv = [sys.argv[0], "--mode", mode, "--device", a.device]
    sa = stage1.parse_args()
    stippler = build_stippler(sa, a.device)
    per_step, peak, fwd = [], [], []
    for e in images:
        image_01 = stippler.prepare(e["path"])
        target = torch.as_tensor(1.0 - np.asarray(image_01, np.float32), device=a.device)[None, None]
        target = torch.nn.functional.interpolate(target, size=(sa.render_res,) * 2, mode="area")
        field = DensityField(image_01, param="field", field_res=sa.field_res, device=a.device).to(a.device)
        opt = torch.optim.Adam(field.parameters(), lr=sa.lr)
        torch.cuda.reset_peak_memory_stats()
        for k in range(a.warmup + a.timed_steps):
            if k == a.warmup:
                torch.cuda.synchronize()
                t0 = time.time()
            opt.zero_grad(set_to_none=True)
            dark = render_darkness(stippler(field()), sa.render_res, sa.dot_sigma_px, a.ink_gain)
            l2, tv = field.regularizer()
            loss = torch.nn.functional.mse_loss(dark, target) + sa.w_l2 * l2 + sa.w_tv * tv
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        per_step.append((time.time() - t0) / a.timed_steps)
        peak.append(torch.cuda.max_memory_allocated() / 2 ** 30)
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(a.timed_steps):
                stippler(field())
            torch.cuda.synchronize()
            fwd.append((time.time() - t0) / a.timed_steps)
    del stippler
    torch.cuda.empty_cache()
    return {"s_per_step": float(np.mean(per_step)), "s_forward_only": float(np.mean(fwd)),
            "peak_gib": float(np.max(peak)), "images": len(images)}


def main():
    a = parse_args()
    images = json.loads(Path(a.manifest).read_text())["images"][:a.images]
    gpu = torch.cuda.get_device_name(0)
    res = {"gpu": gpu, "modes": {}}
    for mode in a.modes.split(","):
        r = time_mode(mode, a, images)
        stored = Path(STORED_RUNS.get(mode, "")) / "summary.json"
        if stored.is_file():
            s = json.loads(stored.read_text())["results"]
            r["stored_s_per_image_field"] = float(np.mean([v["field"]["seconds"] for v in s.values()]))
        res["modes"][mode] = r
        print(mode, r, flush=True)
    lines = ["# Cost of the tone-correction gradient paths (D3)", "",
             f"GPU {gpu}; one optimizer step = sampler forward + render + backward + Adam, mean over "
             f"{a.timed_steps} steps on {len(images)} image(s) after {a.warmup} warm-up steps.", "",
             f"| gradient path | s / step | sampler forward only (s) | peak GPU memory (GiB) | {a.steps} steps (s, predicted) "
             "| stored run, s / image / config |", "|---|---|---|---|---|---|"]
    names = {"onestep": "one step (paper, t = 120)", "ddim": "full DDIM-25 chain (E16)"}
    for mode, r in res["modes"].items():
        lines.append(f"| {names.get(mode, mode)} | {r['s_per_step']:.3f} | {r['s_forward_only']:.3f} | "
                     f"{r['peak_gib']:.2f} | {r['s_per_step'] * a.steps:.0f} | "
                     f"{r.get('stored_s_per_image_field', float('nan')):.0f} |")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
