"""Verify that the gradient through the sampler is CORRECT, not merely non-zero.

Stage extra: needs stage 0 (manifest.json), a checkpoint and a GPU. Independent of every
numbered stage -- nothing reads its output, the numbers it prints go straight into the paper
text, so it can run at any point after stage 0.

A non-zero gradient proves only that something is connected. This compares the analytic
gradient, obtained by backpropagating through the ControlNet and the denoiser, against a
central finite difference of the same objective:

    dL/dtheta_i  ~=  ( L(theta + h e_i) - L(theta - h e_i) ) / 2h

Agreement means autograd is differentiating the function we think it is. The comparison is
only meaningful because the objective is deterministic -- the diffusion noise is drawn once
and held (see tone_results_utils.DifferentiableStippler.prepare) -- so the two evaluations differ
only in theta.

    python experiments/tone_results_stage_extra_gradcheck.py --control_ckpt_path <ckpt>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tone_results_utils import DensityField, build_stippler, render_darkness  # noqa: E402

ROOT_DIR = "experiments/outputs/tone_results"
OUT_DIR = f"{ROOT_DIR}/gradcheck"


# --------------------------------------------------------------------------
# Defaults for every command-line argument. Edit here, not in parse_args.
# --------------------------------------------------------------------------
MANIFEST                       = f"{ROOT_DIR}/manifest.json"
IMAGE                          = None
COORDS                         = 24
FD_STEP                        = 2e-2
FIELD_RES                      = 16
RENDER_RES                     = 256
DOT_SIGMA_PX                   = 2.0
INK_GAIN                       = 3.0
SEED                           = 0
DEVICE                         = "cuda"
MODE                           = "onestep"
UNROLL_STEPS                   = 4
T_PROBE                        = 120
BASE_CONFIG_PATH               = "config/GBN/config.json"
BASE_CKPT_PATH                 = ""
CONTROL_CKPT_PATH              = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt"
GRID_SIZE                      = 32
EVAL_TIMESTEPS                 = 1000
INFER_TRUNCATION_RATIO         = 0.5
ENABLE_GECCO                   = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
SMART_INIT_FEATURES            = False
SDF_FEATURES                   = False
BATCH_COORDS_FEATURES          = False
SDF_TRUNCATE_PX                = 8.0
SMART_INIT_SEED                = 42


def parse_args():
    p = argparse.ArgumentParser(description="Finite-difference check of the sampler gradient")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--image", default=IMAGE, help="Stem to use; default is the first entry")
    p.add_argument("--coords", type=int, default=COORDS, help="Parameters to probe")
    p.add_argument("--h", type=float, default=FD_STEP, help="Finite-difference step")
    p.add_argument("--field-res", type=int, default=FIELD_RES,
                   help="Coarser than the main run: each parameter then has a larger effect, "
                        "so the difference stands clear of floating-point noise")
    p.add_argument("--render-res", type=int, default=RENDER_RES)
    p.add_argument("--dot-sigma-px", type=float, default=DOT_SIGMA_PX)
    p.add_argument("--ink-gain", type=float, default=INK_GAIN)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--mode", default=MODE, choices=["onestep", "unrolled"])
    p.add_argument("--unroll-steps", type=int, default=UNROLL_STEPS)
    p.add_argument("--t-probe", type=int, default=T_PROBE)
    p.add_argument("--base_config_path", default=BASE_CONFIG_PATH)
    p.add_argument("--base_ckpt_path", default=BASE_CKPT_PATH)
    p.add_argument("--control_ckpt_path",
                   default=CONTROL_CKPT_PATH)
    p.add_argument("--grid_size", type=int, default=GRID_SIZE)
    p.add_argument("--eval_timesteps", type=int, default=EVAL_TIMESTEPS)
    p.add_argument("--infer_truncation_ratio", type=float, default=INFER_TRUNCATION_RATIO)
    p.add_argument("--enable_gecco", default=ENABLE_GECCO, action=argparse.BooleanOptionalAction)
    p.add_argument("--enable_adaptive_gate_injection", default=ENABLE_ADAPTIVE_GATE_INJECTION,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--smart_init_features", default=SMART_INIT_FEATURES, action=argparse.BooleanOptionalAction)
    p.add_argument("--sdf_features", default=SDF_FEATURES, action=argparse.BooleanOptionalAction)
    p.add_argument("--batch_coords_features", default=BATCH_COORDS_FEATURES, action=argparse.BooleanOptionalAction)
    p.add_argument("--sdf_truncate_px", type=float, default=SDF_TRUNCATE_PX)
    p.add_argument("--smart_init_seed", type=int, default=SMART_INIT_SEED)
    return p.parse_args()


def main():
    a = parse_args()
    # Create the output folder up front, not at the write. The gradient check is minutes of
    # GPU work and its only write is the very last statement, so a missing directory used to
    # discard the whole run; failing here costs nothing.
    Path(a.output).mkdir(parents=True, exist_ok=True)
    device = a.device if (torch.cuda.is_available() or a.device == "cpu") else "cpu"
    man = json.loads(Path(a.manifest).read_text())
    entry = next((e for e in man["images"] if e["stem"] == a.image), man["images"][0])
    print(f"image: {entry['stem']}   mode: {a.mode}   device: {device}")

    stippler = build_stippler(a, device)
    image_01 = stippler.prepare(entry["path"])

    target = torch.as_tensor(1.0 - np.asarray(image_01, np.float32), device=device)[None, None]
    target = torch.nn.functional.interpolate(target, size=(a.render_res,) * 2, mode="area")
    field = DensityField(image_01, param="field", field_res=a.field_res, device=device).to(device)

    def objective():
        coords = stippler(field())
        dark = render_darkness(coords, a.render_res, a.dot_sigma_px, a.ink_gain)
        return torch.nn.functional.mse_loss(dark, target)

    # determinism first: the same theta must give the same value, or a finite difference
    # measures noise rather than slope
    with torch.no_grad():
        v1, v2 = float(objective()), float(objective())
    print(f"determinism: {v1:.10e} vs {v2:.10e}  (diff {abs(v1 - v2):.3e})")
    if abs(v1 - v2) > 1e-12 * max(1.0, abs(v1)):
        print("  WARNING: the objective is not deterministic; the check below is meaningless.")

    field.zero_grad(set_to_none=True)
    loss = objective()
    loss.backward()
    analytic = field.delta.grad.detach().clone().reshape(-1)
    print(f"analytic gradient: |g| = {float(analytic.norm()):.6e}, "
          f"{int((analytic != 0).sum())}/{analytic.numel()} entries non-zero")

    # probe the parameters with the largest analytic gradient: where the slope is ~0 the
    # finite difference is dominated by rounding and the ratio is uninformative
    idx = torch.argsort(analytic.abs(), descending=True)[:a.coords]
    flat = field.delta.detach().reshape(-1)

    rows = []
    for i in idx.tolist():
        orig = float(flat[i])
        with torch.no_grad():
            flat[i] = orig + a.h
            lp = float(objective())
            flat[i] = orig - a.h
            lm = float(objective())
            flat[i] = orig
        fd = (lp - lm) / (2 * a.h)
        an = float(analytic[i])
        rows.append((i, an, fd))

    an = np.array([r[1] for r in rows])
    fd = np.array([r[2] for r in rows])
    cos = float(an @ fd / (np.linalg.norm(an) * np.linalg.norm(fd) + 1e-30))
    rel = float(np.linalg.norm(an - fd) / (np.linalg.norm(an) + np.linalg.norm(fd) + 1e-30))
    slope = float(np.polyfit(fd, an, 1)[0]) if len(fd) > 1 else float("nan")
    agree = float(np.mean(np.sign(an) == np.sign(fd)))

    print(f"\n{'idx':>6} {'analytic':>14} {'finite diff':>14} {'ratio':>9}")
    for i, x, y in rows[:12]:
        print(f"{i:>6} {x:>14.6e} {y:>14.6e} {x / y if y else float('nan'):>9.3f}")

    print(f"\nprobed {len(rows)} parameters, h = {a.h}")
    print(f"  cosine similarity  {cos:+.4f}   (1.0 = identical direction)")
    print(f"  sign agreement     {agree:.3f}")
    print(f"  relative error     {rel:.4f}")
    print(f"  best-fit slope     {slope:+.4f}   (1.0 = same magnitude)")
    ok = cos > 0.9 and agree > 0.9
    print(f"\n{'PASS' if ok else 'FAIL'}: the analytic gradient "
          f"{'matches' if ok else 'does NOT match'} the finite difference.")

    out = Path(a.output) / "gradcheck.json"
    out.write_text(json.dumps({
        "image": entry["stem"], "mode": a.mode, "h": a.h, "n_probed": len(rows),
        "cosine": cos, "sign_agreement": agree, "relative_error": rel, "slope": slope,
        "deterministic_diff": abs(v1 - v2), "pass": bool(ok),
        "pairs": [{"index": int(i), "analytic": x, "finite_difference": y} for i, x, y in rows],
    }, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
