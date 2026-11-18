"""Few-step sampling stage 1: sample a test set with DDPM / DDIM / DPM-Solver++ from the same weights.

Rebuttal experiment E1. One trained Control V4 checkpoint is sampled with several reverse
samplers, all starting from the *same* noised Smart Init (SDEdit start at t_start = 500 of
T = 1000), so that every difference in the outputs comes from the sampler alone:

    ddpm-500     the paper sampler (diffusion.p_sample, every step from t_start-1 down to 0)
    ddpm-K       the same DDPM loop started at step K instead (noised to alpha_cumprod[K])
    ddim-K       deterministic DDIM (eta = 0) with K model evaluations
    ddim-K-eta1  stochastic DDIM (eta = 1)
    dpmpp2m-K    DPM-Solver++(2M), data-prediction form, with K model evaluations
    <any>-clean  start from the Smart Init offsets as they are, without forward noise
    <any>-noise  start from pure Gaussian noise at t = T - 1 (no Smart Init, no SDEdit); the
                 K timesteps then span T - 1 ... 0. When every sampler of a run is -noise, the
                 Smart Init is not computed at all and its time is recorded as 0.

The K evaluation timesteps are round(linspace(t_start - 1, 0, K)), so ddim-500 visits exactly
the timesteps of ddpm-500 and every sampler ends with an evaluation at t = 0 followed by the
x0 prediction. The model predicts eps; clipping is off, as in DiffusionModel.p_mean_variance.

Per image the start noise is drawn from a CPU generator seeded with (noise_seed + image index)
and the Smart Init with smart_init_seed, so all samplers share the same Smart Init and noise
tensor (scaled to their own start step) and reruns are reproducible. The DDPM step noise also uses the global RNG reseeded per image.

Timing: GPU time of the reverse loop (cuda synchronize before/after), plus the sampler-independent
Smart Init (rejection sampling + point-to-grid assignment with --ot_solver) and OT-conversion
times, once per image. Batch size 1. With --throughput, the first
--throughput-images images are re-run at each --batch-sizes value (denoising only) to report
stipples per second.

Outputs (one folder per sampler, same layout as the Table 1 targets):
    OUTPUT_DIR/<label>_<points>/<sampler>/<stem>.npy      (N, 2) points, [x/w, y/h], y-down
    OUTPUT_DIR/<label>_<points>/<sampler>/<stem>.png      first --png-count images only
    OUTPUT_DIR/<label>_<points>/timings[_tag].jsonl       one line per (image, sampler)
    OUTPUT_DIR/<label>_<points>/throughput[_tag].jsonl    with --throughput
    OUTPUT_DIR/<label>_<points>/run_info[_tag].json       args, GPU, torch version
(--tag keeps the log files of concurrent jobs on the same run folder apart.)

Usage:
    python experiments/few_step_sampling_stage_1.py \\
        --control_ckpt_path control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt \\
        --label CN-GBN --limit 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_v4.DynamicControlNet import DynamicControlledDenoiser
from control_v4.sample_control import load_condition, load_pipeline, save_sample_image
from control_v4.ot_assignment import OT_SOLVERS
from control_v4.smart_init import (
    add_noise_at_t,
    generate_smart_init_points_from_density,
    smart_init_points_to_offsets,
)
from data.Transforms import to_pointset_optimal_transport

# Defaults (paper configuration, see experiments/control_data_gen.py)
BASE_CONFIG_PATH = "config/GBN/config.json"
CONTROL_CKPT = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt"
LABEL = "CN-GBN"
SOURCE_DIR = "experiments/outputs/z_validation_data/Icons-50_1024/source"
OUTPUT_DIR = "experiments/outputs/few_step_sampling-Icons-50_1024"
GRID_SIZE = 32
EVAL_TIMESTEPS = 1000
T_START = 500
SMART_INIT_SEED = 42
NOISE_SEED = 0
SAMPLERS = "ddpm-500,ddim-100,ddim-50,ddim-25,ddim-10,dpmpp2m-20,dpmpp2m-10"
PNG_COUNT = 16
BATCH_SIZES = "1,8,32,64"
THROUGHPUT_IMAGES = 8
DEVICE = "cuda"
SMART_INIT_SOURCE = ""  # empty: Smart Init from the condition image itself
OT_SOLVER = "pot"  # paper path; ot.emd is truncated from ~3136 points, see control_v4/ot_assignment.py

ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True


def parse_args():
    p = argparse.ArgumentParser(description="Sample a test set with several reverse samplers from one checkpoint")
    p.add_argument("--control_ckpt_path", default=CONTROL_CKPT)
    p.add_argument("--base_config_path", default=BASE_CONFIG_PATH)
    p.add_argument("--label", default=LABEL, help="Method tag used in the output folder, e.g. CN-GBN")
    p.add_argument("--source", default=SOURCE_DIR, help="Folder with the grayscale test images")
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--grid_size", type=int, default=GRID_SIZE)
    p.add_argument("--eval_timesteps", type=int, default=EVAL_TIMESTEPS)
    p.add_argument("--t_start", type=int, default=T_START, help="SDEdit start step (paper: 0.5 * T = 500)")
    p.add_argument("--smart_init_seed", type=int, default=SMART_INIT_SEED)
    p.add_argument("--noise_seed", type=int, default=NOISE_SEED, help="Per-image seed is noise_seed + image index")
    p.add_argument("--samplers", default=SAMPLERS,
                   help="Comma-separated: ddpm-K, ddim-K, ddim-K-eta1, dpmpp2m-K, optional -clean / -noise suffix")
    p.add_argument("--tag", default="", help="Suffix for the timings / throughput / run_info files")
    p.add_argument("--limit", type=int, default=-1, help="Use only the first N images (sorted); -1 = all")
    p.add_argument("--png-count", type=int, default=PNG_COUNT, help="Export PNGs for the first N images")
    p.add_argument("--throughput", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-sizes", default=BATCH_SIZES)
    p.add_argument("--throughput-images", type=int, default=THROUGHPUT_IMAGES)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--smart_init_source", default=SMART_INIT_SOURCE,
                   help="Folder with same-named images to draw the Smart Init from instead of the condition")
    p.add_argument("--ot_solver", choices=OT_SOLVERS, default=OT_SOLVER,
                   help="Smart Init point-to-grid assignment solver (pot = the paper path)")
    p.add_argument("--init_dir", default="",
                   help="Folder with <stem>.npy offset grids (2, n, n) used instead of the Smart Init, "
                        "e.g. the teacher's exemplar; each sampler noises it to its own start step")
    # Architecture of the checkpoint. The defaults are the paper's full model; the Table 2 ablation
    # variants need the frozen pretrained base and / or GECCO or gated injection switched off.
    p.add_argument("--base_ckpt_path", default="",
                   help="Pretrained base denoiser for frozen-base checkpoints (e.g. config/GBN/model.ckpt); "
                        "empty = the jointly trained base stored in the control checkpoint")
    p.add_argument("--gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    p.add_argument("--gate-injection", action=argparse.BooleanOptionalAction,
                   default=ENABLE_ADAPTIVE_GATE_INJECTION)
    return p.parse_args()


# ── Samplers ──────────────────────────────────────────────────────────────────

def parse_sampler(name):
    """'ddim-25-eta1' -> ('ddim', 25, 1.0, False); 'ddpm-300-clean' -> ('ddpm', 300, 0.0, True)."""
    parts = name.split("-")
    kind, steps = parts[0], int(parts[1])
    eta, clean = 0.0, False
    for extra in parts[2:]:
        if extra.startswith("eta"):
            eta = float(extra[3:])
        elif extra == "clean":
            clean = True
        elif extra == "noise":
            pass  # see starts_from_noise
        else:
            raise ValueError(f"Unknown sampler option '{extra}' in {name}")
    if kind not in ("ddpm", "ddim", "dpmpp2m"):
        raise ValueError(f"Unknown sampler: {name}")
    return kind, steps, eta, clean


def eval_timesteps_for(t_start, steps):
    """K evaluation timesteps from t_start - 1 down to 0 (ddim-500 == the DDPM timesteps)."""
    ts = np.rint(np.linspace(t_start - 1, 0, steps)).astype(np.int64)
    return list(dict.fromkeys(ts.tolist()))  # drop duplicates, keep order


def eps_model(diffusion, x, t):
    t_tensor = torch.full((x.shape[0],), int(t), dtype=torch.int64, device=x.device)
    return diffusion.model(x, t_tensor, None)


def sample_ddpm(diffusion, x, t_start):
    """The paper loop of process_single_image (resample_jumps = 0)."""
    for i in reversed(range(t_start)):
        t_tensor = torch.full((x.shape[0],), i, dtype=torch.int64, device=x.device)
        x = diffusion.p_sample(x, cond=None, t=t_tensor, clip_denoised=diffusion.sample_clip, with_sampling=True)
    return x


def sample_ddim(diffusion, x, timesteps, eta=0.0):
    ac = diffusion.alphas_cumprod
    for i, t in enumerate(timesteps):
        a_t = ac[t]
        a_prev = ac[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.ones_like(a_t)
        eps = eps_model(diffusion, x, t)
        x0 = (x - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
        sigma = eta * ((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev)).sqrt()
        x = a_prev.sqrt() * x0 + (1.0 - a_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
        if eta > 0 and i + 1 < len(timesteps):
            x = x + sigma * torch.randn_like(x)
    return x


def sample_dpmpp_2m(diffusion, x, timesteps):
    """DPM-Solver++(2M), data prediction, first order on the first step and on the final
    step to the clean sample (Lu et al. 2022; diffusers' lower_order_final)."""
    ac = diffusion.alphas_cumprod
    alpha = lambda t: ac[t].sqrt()
    sigma = lambda t: (1.0 - ac[t]).sqrt()
    lam = lambda t: torch.log(alpha(t)) - torch.log(sigma(t))

    x0_prev, h_prev = None, None
    for i, s in enumerate(timesteps):
        eps = eps_model(diffusion, x, s)
        x0 = (x - sigma(s) * eps) / alpha(s)
        if i + 1 == len(timesteps):
            return x0
        t = timesteps[i + 1]
        h = lam(t) - lam(s)
        if x0_prev is None:
            d = x0
        else:
            r = h_prev / h
            d = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * x0_prev
        x = (sigma(t) / sigma(s)) * x - alpha(t) * torch.expm1(-h) * d
        x0_prev, h_prev = x0, h
    return x


def starts_from_noise(name):
    return "noise" in name.split("-")[2:]


def start_step_for(name, t_start, num_timesteps=EVAL_TIMESTEPS):
    """ddpm-K starts at step K; the few-step samplers start at the shared t_start, or at T with -noise."""
    kind, steps, _, _ = parse_sampler(name)
    if kind == "ddpm":
        return steps
    return num_timesteps if starts_from_noise(name) else t_start


def run_sampler(diffusion, x_init, noise, name, t_start):
    kind, steps, eta, clean = parse_sampler(name)
    start = start_step_for(name, t_start, len(diffusion.alphas_cumprod))
    if starts_from_noise(name):
        x = noise.clone()
    elif clean:
        x = x_init.clone()
    else:
        x = add_noise_at_t(x_init, diffusion.alphas_cumprod[start], noise=noise)
    with torch.no_grad():
        if kind == "ddpm":
            return sample_ddpm(diffusion, x, start)
        timesteps = eval_timesteps_for(start, steps)
        if kind == "ddim":
            return sample_ddim(diffusion, x, timesteps, eta=eta)
        return sample_dpmpp_2m(diffusion, x, timesteps)


def cuda_sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


# ── Per-image setup ───────────────────────────────────────────────────────────

def prepare_image(image_path, diffusion, control_net, args, image_index, batch=1):
    """Set the condition on the denoiser and return the shared Smart Init and start noise."""
    image_01, high_res, target_density, high_res_sdf, target_sdf = load_condition(
        image_path, args.grid_size, args.device, sdf_features=False,
    )

    si_image = image_01
    if args.smart_init_source:
        si_image = load_condition(Path(args.smart_init_source) / image_path.name, args.grid_size,
                                  args.device, sdf_features=False)[0]
    if getattr(args, "init_dir", ""):  # start from given offsets (e.g. the teacher's exemplar)
        x_init = torch.from_numpy(np.load(Path(args.init_dir) / f"{image_path.stem}.npy").astype(np.float32))
        x_init = x_init.unsqueeze(0).to(args.device)
        time_si = 0.0
    elif getattr(args, "skip_smart_init", False):  # every sampler starts from pure noise
        x_init = torch.zeros((1, 2, args.grid_size, args.grid_size), device=args.device)
        time_si = 0.0
    else:
        t0 = time.perf_counter()
        smart_points = generate_smart_init_points_from_density(
            si_image, n_points=args.grid_size * args.grid_size, seed=args.smart_init_seed,
        )
        x_init = torch.from_numpy(smart_init_points_to_offsets(smart_points, args.ot_solver, args.device)).unsqueeze(0).to(args.device)
        time_si = time.perf_counter() - t0

    base_denoiser = diffusion.model.locked if isinstance(diffusion.model, DynamicControlledDenoiser) else diffusion.model
    controlled = DynamicControlledDenoiser(base_denoiser, control_net)
    controlled.set_condition(high_res, high_res_sdf, target_density, target_sdf, None)
    diffusion.model = controlled

    gen = torch.Generator().manual_seed(args.noise_seed + image_index)
    noise = torch.randn((batch, *x_init.shape[1:]), generator=gen).to(args.device)
    return x_init.expand(batch, -1, -1, -1), noise, time_si


def offsets_to_points(sample):
    pts = to_pointset_optimal_transport(sample)
    return pts.reshape(pts.shape[0], np.prod(pts.shape[1:])).T


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    samplers = [s.strip() for s in args.samplers.split(",") if s.strip()]
    for s in samplers:
        parse_sampler(s)
    args.skip_smart_init = all(starts_from_noise(s) for s in samplers)

    source = Path(args.source)
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit > 0:
        images = images[: args.limit]

    run_dir = Path(args.output) / f"{args.label}_{args.grid_size ** 2}"
    for s in samplers:
        (run_dir / s).mkdir(parents=True, exist_ok=True)

    print(f"Control weights: {Path(args.control_ckpt_path).resolve()}")
    diffusion, control_net = load_pipeline(
        args.base_config_path, args.base_ckpt_path, args.control_ckpt_path, args.grid_size,
        args.gecco, args.gate_injection, False, False, False, args.device,
    )
    diffusion.set_num_timesteps(args.eval_timesteps)
    diffusion.eval()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    run_info = {**vars(args), "samplers": samplers, "n_images": len(images), "gpu": gpu_name,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "start_steps": {s: start_step_for(s, args.t_start, args.eval_timesteps) for s in samplers},
                "timesteps": {s: eval_timesteps_for(start_step_for(s, args.t_start, args.eval_timesteps),
                                                    parse_sampler(s)[1])
                              for s in samplers if not s.startswith("ddpm")}}
    suffix = f"_{args.tag}" if args.tag else ""
    (run_dir / f"run_info{suffix}.json").write_text(json.dumps(run_info, indent=2))
    print(f"GPU: {gpu_name} | {len(images)} images | samplers: {samplers}")

    # Warm-up (cuDNN autotune, allocator) so the first timed image is not penalised
    x_init, noise, _ = prepare_image(images[0], diffusion, control_net, args, 0)
    for s in samplers:
        run_sampler(diffusion, x_init, noise, s if not s.startswith("ddpm") else "ddim-50", args.t_start)
    cuda_sync(args.device)

    timings_path = run_dir / f"timings{suffix}.jsonl"
    with open(timings_path, "a") as tf:
        for idx, image_path in enumerate(images):
            stem = image_path.stem
            todo = [s for s in samplers if args.overwrite or not (run_dir / s / f"{stem}.npy").exists()]
            if not todo:
                continue

            x_init, noise, time_si = prepare_image(image_path, diffusion, control_net, args, idx)
            for s in todo:
                torch.manual_seed(args.noise_seed + idx)  # DDPM / eta>0 step noise
                cuda_sync(args.device)
                t0 = time.perf_counter()
                out = run_sampler(diffusion, x_init, noise, s, args.t_start)
                cuda_sync(args.device)
                time_denoise = time.perf_counter() - t0

                t0 = time.perf_counter()
                pts = offsets_to_points(out[0].detach().cpu().numpy())
                time_ot = time.perf_counter() - t0

                np.save(run_dir / s / f"{stem}.npy", pts)
                if idx < args.png_count:
                    save_sample_image(image_path, pts, run_dir / s / f"{stem}.png")
                tf.write(json.dumps({"image": stem, "sampler": s, "time_smart_init": time_si,
                                     "time_denoise": time_denoise, "time_ot": time_ot, "gpu": gpu_name,
                                     "ot_solver": args.ot_solver}) + "\n")
            tf.flush()
            print(f"[{idx + 1}/{len(images)}] {stem}: " + ", ".join(todo))

    if args.throughput:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
        with open(run_dir / f"throughput{suffix}.jsonl", "a") as tf:
            for idx, image_path in enumerate(images[: args.throughput_images]):
                for b in batch_sizes:
                    x_init, noise, _ = prepare_image(image_path, diffusion, control_net, args, idx, batch=b)
                    for s in samplers:
                        torch.manual_seed(args.noise_seed + idx)
                        cuda_sync(args.device)
                        t0 = time.perf_counter()
                        run_sampler(diffusion, x_init, noise, s, args.t_start)
                        cuda_sync(args.device)
                        dt = time.perf_counter() - t0
                        tf.write(json.dumps({"image": image_path.stem, "sampler": s, "batch": b,
                                             "time_denoise": dt, "stipples_per_s": b / dt}) + "\n")
                tf.flush()
                print(f"[throughput {idx + 1}/{args.throughput_images}] {image_path.stem}")

    print(f"Done: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
