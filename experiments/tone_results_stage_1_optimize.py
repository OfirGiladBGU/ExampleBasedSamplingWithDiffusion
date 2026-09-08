"""Stage 1: solve for the pre-compensated density, per image, by gradient descent through
the sampler.

For each target image I this runs three configurations and writes them side by side:

    none    the conditioning is I itself -- what you get today. The rendered stipple is too
            dark wherever dots overlap.
    curve   a global monotone transfer curve, fitted by the same optimizer against the same
            objective. This is the classical fix, and it is the honest baseline.
    field   a spatially varying correction, our method. Only reachable because gradients pass
            through the sampler.

Nothing about "none" needs optimizing; it is evaluated once to give the starting point.

Usage:
    python experiments/tone_results_stage_1_optimize.py --control_ckpt_path <ckpt> --steps 300
    python experiments/tone_results_stage_1_optimize.py --self-test      # CPU, no checkpoint
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

from tone_results_utils import (DensityField, psnr, render_darkness,  # noqa: E402
                         splat_points, ssim)

# One tree for the whole experiment; each run writes into its own subfolder.
ROOT_DIR = "experiments/outputs/tone_results"
OUT_DIR = f"{ROOT_DIR}/main"

# Rendering. RENDER_RES is the resolution the objective is evaluated at; DOT_SIGMA_PX is the
# dot radius in pixels of that grid, and INK_GAIN sets how quickly overlapping dots saturate.
# Together these define the "printer" being compensated for -- they are the experiment's
# physical model, not tuning knobs, so they are reported in the manifest.
RENDER_RES = 256
# Dot radius, from geometry rather than taste: 1024 points on a 256 px canvas sit ~8 px apart,
# so sigma = 2 px gives dots roughly half the mean spacing -- they overlap in dense regions,
# which is the effect being compensated for, without merging everywhere.
DOT_SIGMA_PX = 2.0
# -1 calibrates the gain on the uncorrected renders (see calibrate_gain); a positive value
# fixes it explicitly.
INK_GAIN = -1.0

STEPS = 300
LR = 0.05
W_L2 = 1e-3          # keep the correction small
W_TV = 1e-2          # and smooth: a tone fix, not a redrawing
FIELD_RES = 64
KNOTS = 17


# --------------------------------------------------------------------------
# Defaults for every command-line argument. Edit here, not in parse_args.
# --------------------------------------------------------------------------
MANIFEST                       = f"{ROOT_DIR}/manifest.json"
# "random" is the gradient-free control and the comparison the paper leads with, so it is
# ON by default -- a bare run must reproduce the reported figures and table.
CONFIGS                        = "none,random,curve,field"
LIMIT                          = -1
CALIB_IMAGES                   = 6
MODE                           = "onestep"
UNROLL_STEPS                   = 8
T_PROBE                        = 120
SEED                           = 0
DEVICE                         = "cuda"
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
    p = argparse.ArgumentParser(description="Optimize the conditioning density through the sampler")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--configs", default=CONFIGS)
    p.add_argument("--limit", type=int, default=LIMIT, help="Process only the first N images")

    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--w-l2", type=float, default=W_L2)
    p.add_argument("--w-tv", type=float, default=W_TV)
    p.add_argument("--field-res", type=int, default=FIELD_RES)
    p.add_argument("--knots", type=int, default=KNOTS)

    p.add_argument("--render-res", type=int, default=RENDER_RES)
    p.add_argument("--dot-sigma-px", type=float, default=DOT_SIGMA_PX)
    p.add_argument("--ink-gain", type=float, default=INK_GAIN,
                   help="-1 calibrates so mean rendered tone matches mean requested tone")
    p.add_argument("--calib-images", type=int, default=CALIB_IMAGES,
                   help="Images used to calibrate the ink gain")

    p.add_argument("--mode", default=MODE, choices=["onestep", "unrolled"])
    p.add_argument("--unroll-steps", type=int, default=UNROLL_STEPS)
    p.add_argument("--t-probe", type=int, default=T_PROBE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=DEVICE)

    # model, named as control_v4 names them
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

    p.add_argument("--self-test", action="store_true",
                   help="Run the whole loop on CPU with a stub sampler and one synthetic "
                        "image, to check the plumbing without a GPU or a checkpoint.")
    return p.parse_args()


# ── stub sampler, for the self-test only ───────────────────────────────────

class StubStippler(torch.nn.Module):
    """A differentiable stand-in that is NOT the model.

    Points sit on a jittered lattice and carry an opacity read from rho at their location, so
    the rendered darkness responds to rho differentiably. Enough to exercise the renderer, the
    optimizer, the metrics and the figures end to end; it says nothing about the real sampler.
    """

    def __init__(self, grid_size=32, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        ax = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        gy, gx = torch.meshgrid(ax, ax, indexing="ij")
        base = torch.stack([gx, gy], -1).reshape(1, -1, 2)
        jit = (torch.rand(base.shape, generator=g) - 0.5) * (0.7 / grid_size)
        self.register_buffer("coords", (base + jit).clamp(0, 1).to(device))
        self.last_weights = None

    def prepare(self, image_path):
        return None

    def forward(self, rho, generator=None):
        g = self.coords * 2.0 - 1.0                       # (1,N,2) -> grid_sample coords
        w = torch.nn.functional.grid_sample(
            rho, g[:, None, :, :], mode="bilinear", align_corners=False,
            padding_mode="border").reshape(1, -1)
        self.last_weights = w
        return self.coords


# ── one configuration on one image ─────────────────────────────────────────

def run_config(stippler, image_01, cfg_name, a, device, log=None):
    target_dark = torch.as_tensor(1.0 - np.asarray(image_01, np.float32),
                                  device=device)[None, None]
    target_dark = torch.nn.functional.interpolate(
        target_dark, size=(a.render_res,) * 2, mode="area")

    param = "curve" if cfg_name == "curve" else "field"   # none/random/field share it
    fieldmod = DensityField(image_01, param=param, field_res=a.field_res,
                            knots=a.knots, device=device).to(device)

    gen = torch.Generator(device=device).manual_seed(a.seed)

    def render(rho):
        coords = stippler(rho, generator=gen)
        w = getattr(stippler, "last_weights", None)
        return render_darkness(coords, a.render_res, a.dot_sigma_px, a.ink_gain, weights=w), coords

    # The gradient-free control gets twice the step count in sampler evaluations, because a
    # gradient step costs one forward plus one backward.
    if cfg_name == "random":
        history = run_random_search(fieldmod, render, target_dark, a, 2 * a.steps)
        with torch.no_grad():
            rho = fieldmod()
            dark, coords = render(rho)
        return _pack(rho, dark, coords, target_dark, history, 0.0)

    # "none" is the uncorrected starting point: evaluate, do not optimize.
    if cfg_name == "none":
        with torch.no_grad():
            rho = fieldmod()
            dark, coords = render(rho)
        return _pack(rho, dark, coords, target_dark, history=[], seconds=0.0)

    opt = torch.optim.Adam(fieldmod.parameters(), lr=a.lr)
    history = []
    t0 = time.time()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        rho = fieldmod()
        dark, _ = render(rho)
        data = torch.nn.functional.mse_loss(dark, target_dark)
        l2, tv = fieldmod.regularizer()
        loss = data + a.w_l2 * l2 + a.w_tv * tv
        loss.backward()

        gnorm = float(torch.nn.utils.clip_grad_norm_(fieldmod.parameters(), 1e9))
        if not np.isfinite(gnorm):
            raise FloatingPointError(
                f"[{cfg_name}] non-finite gradient at step {step}. The gradient path through "
                f"the sampler is broken; do not trust any result from this run.")
        if step == 0 and gnorm == 0.0:
            raise FloatingPointError(
                f"[{cfg_name}] the gradient w.r.t. the conditioning is exactly zero. Nothing "
                f"is connecting rho to the rendered image -- check that set_condition is "
                f"receiving the tensor being optimized.")
        opt.step()
        history.append({"step": step, "loss": float(loss.detach()),
                        "data": float(data.detach()), "grad_norm": gnorm})
        if log and (step % max(1, a.steps // 10) == 0 or step == a.steps - 1):
            print(f"    [{cfg_name}] step {step:4d}  loss {float(loss.detach()):.6f}  "
                  f"data {float(data.detach()):.6f}  |g| {gnorm:.3e}", flush=True)

    with torch.no_grad():
        rho = fieldmod()
        dark, coords = render(rho)
    return _pack(rho, dark, coords, target_dark, history, time.time() - t0)


def run_random_search(fieldmod, render, target_dark, a, budget):
    """(1+1) random search over the same parameters, with no gradients.

    The control for the whole experiment: if this matches the gradient run at an equal number
    of sampler evaluations, then differentiability bought nothing. `budget` counts sampler
    evaluations, so a gradient step (one forward + one backward) is charged as two.

    Step size adapts by the 1/5th success rule, which is the standard way to keep such a
    search competitive rather than deliberately weak.
    """
    with torch.no_grad():
        best = float(torch.nn.functional.mse_loss(render(fieldmod())[0], target_dark))
    sigma, hits, history = 0.1, 0, []
    theta = fieldmod.delta.detach().clone()
    for k in range(budget):
        cand = theta + sigma * torch.randn_like(theta)
        with torch.no_grad():
            fieldmod.delta.copy_(cand)
            loss = float(torch.nn.functional.mse_loss(render(fieldmod())[0], target_dark))
        if loss < best:
            best, theta, hits = loss, cand, hits + 1
        if (k + 1) % 20 == 0:                       # 1/5th success rule
            sigma *= 1.5 if hits > 4 else 0.75
            sigma, hits = float(np.clip(sigma, 1e-3, 1.0)), 0
        history.append({"step": k, "loss": best, "data": best, "grad_norm": 0.0})
    with torch.no_grad():
        fieldmod.delta.copy_(theta)
    return history


def calibrate_gain(stippler, images, arrays, a, device):
    """Solve for the gain g with mean(1 - exp(-g * ink)) == mean(target darkness).

    Uses the UNCORRECTED conditioning, so the calibration is a property of the renderer and
    the sampler, not of any correction being compared. Monotone in g, so bisection is exact.
    """
    inks, tgts = [], []
    for entry in images[:max(1, a.calib_images)]:
        image_01 = arrays.get(entry["stem"])
        if image_01 is None:
            image_01 = stippler.prepare(entry["path"])
        rho = torch.as_tensor(1.0 - np.asarray(image_01, np.float32), device=device)[None, None]
        with torch.no_grad():
            coords = stippler(rho)
            w = getattr(stippler, "last_weights", None)
            inks.append(splat_points(coords, a.render_res, a.dot_sigma_px, weights=w))
        tgts.append(torch.nn.functional.interpolate(rho, size=(a.render_res,) * 2, mode="area"))
    ink = torch.cat(inks).clamp_min(0.0)
    want = float(torch.cat(tgts).mean())

    lo, hi = 1e-4, 1e4
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float((1.0 - torch.exp(-mid * ink)).mean()) < want:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    got = float((1.0 - torch.exp(-g * ink)).mean())
    print(f"calibrated ink gain = {g:.4f}  (mean rendered tone {got:.4f} vs target {want:.4f}, "
          f"over {len(inks)} image(s))")
    return g


def _pack(rho, dark, coords, target_dark, history, seconds):
    d = dark.detach().cpu().numpy()[0, 0]
    t = target_dark.detach().cpu().numpy()[0, 0]
    return {
        "rho": rho.detach().cpu().numpy()[0, 0],
        "darkness": d,
        "coords": coords.detach().cpu().numpy()[0],
        "psnr": psnr(d, t), "ssim": ssim(d, t),
        "mae": float(np.abs(d - t).mean()),
        "history": history, "seconds": seconds,
    }


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    configs = [c.strip() for c in a.configs.split(",") if c.strip()]

    if a.self_test:
        device = "cpu"
        a.steps = min(a.steps, 30)
        a.render_res = min(a.render_res, 96)
        print("SELF-TEST: stub sampler on CPU. This validates plumbing only -- the numbers "
              "below say nothing about the trained model.")
        yy, xx = np.mgrid[0:128, 0:128] / 127.0
        image_01 = (0.25 + 0.7 * xx * (1 - 0.5 * yy)).astype(np.float32)
        images = [{"stem": "selftest_ramp", "path": None}]
        stippler = StubStippler(a.grid_size, seed=a.seed, device=device)
        arrays = {"selftest_ramp": image_01}
    else:
        device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
        if device != a.device:
            print(f"WARNING: CUDA unavailable, falling back to {device}")
        man = json.loads(Path(a.manifest).read_text())
        images = man["images"] if a.limit < 0 else man["images"][:a.limit]
        from tone_results_utils import build_stippler
        stippler = build_stippler(a, device)
        arrays = {}

    if a.ink_gain is not None and a.ink_gain < 0:
        a.ink_gain = calibrate_gain(stippler, images, arrays, a, device)

    rows, results = [], {}
    for i, entry in enumerate(images, 1):
        stem = entry["stem"]
        print(f"[{i}/{len(images)}] {stem}", flush=True)
        image_01 = arrays.get(stem)
        if image_01 is None:
            image_01 = stippler.prepare(entry["path"])

        per = {}
        for c in configs:
            per[c] = run_config(stippler, image_01, c, a, device, log=True)
            r = per[c]
            print(f"    [{c:5s}] PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.4f}  "
                  f"MAE {r['mae']:.4f}  {r['seconds']:.1f}s", flush=True)
            rows.append({"stem": stem, "config": c, "psnr": r["psnr"], "ssim": r["ssim"],
                         "mae": r["mae"], "seconds": r["seconds"]})

        np.savez_compressed(
            out / f"{stem}.npz", image_01=image_01,
            **{f"{c}_{k}": per[c][k] for c in configs for k in ("rho", "darkness", "coords")},
            **{f"{c}_loss": np.array([h["loss"] for h in per[c]["history"]], np.float64)
               for c in configs})
        results[stem] = {c: {k: per[c][k] for k in ("psnr", "ssim", "mae", "seconds")}
                         for c in configs}

    (out / "summary.json").write_text(json.dumps(
        {"config": {k: v for k, v in vars(a).items() if not callable(v)},
         "results": results}, indent=2, default=str), encoding="utf-8")

    import csv
    with open(out / "tone_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "config", "psnr", "ssim", "mae", "seconds"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(images)} image(s) x {len(configs)} config(s) -> {out}")
    for c in configs:
        sel = [r for r in rows if r["config"] == c]
        if sel:
            print(f"  {c:6s} mean PSNR {np.mean([r['psnr'] for r in sel]):6.2f}  "
                  f"mean SSIM {np.mean([r['ssim'] for r in sel]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
