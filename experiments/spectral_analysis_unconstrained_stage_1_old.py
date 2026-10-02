"""spectral_analysis_unconstrained_stage_1_old.py

Unconstrained spectral analysis, stage 1: build the point sets.

The base GBN diffusion model (config/GBN, no ControlNet) takes no conditioning image -- it cannot
be given a capacity at all, so it always samples the unconstrained blue-noise process on the unit
square. The one setting in which it is comparable to the conditioned samplers is the uniform
FULL-BLACK target (rho = 1.0), where a constrained sampler is asked for exactly that unconstrained
process. This stage runs every method there, on the 25 independent realizations created by
spectral_analysis_unconstrained_stage_0_old.py (fully black, no white margin), read from OUT_DIR.

Each method has its own flag, so they can be split over jobs (CPU-only methods in a CPU job, the
rest in a GPU job):

    RUN_WVS        WVS  (Rougier-2017/src/wvs_data_gen.py, CPU)          -> target_WVS_<N>/
    RUN_BNOT       BNOT (BNOT_new/bnot_data_gen.py, CPU)                 -> target_BNOT_<N>/
    RUN_GBN        GBN  (GaussianBlueNoise/scripts/gbn_data_gen.py, GPU) -> target_GBN_<N>/
    RUN_BASELINE   base GBN diffusion model, no image (GPU)              -> target_Base-GBN_<N>/
    RUN_CONTROL    our ControlNet (experiments/control_data_gen.py, GPU) -> target_CN-WVS_<N>/, target_CN-GBN_<N>/
    COPY_EXISTING  copy COPY_METHODS from SOURCE_BASE_DIR instead -- only valid when stage 0 built
                   the same images that analysis used, which is checked pixel by pixel (the original
                   spectral analysis has a 1 px white margin, so by default it does NOT apply)

The samplers use the settings of the original spectral analysis: its config blocks in each
generator script (n points, native image size), and for BNOT its final configuration from
experiments/outputs/spectral_analysis/FINDINGS.md -- seed -1 (a fresh seed per realization; a
fixed seed makes all 25 identical), epsilon 0.01, max_iters 25, and an external per-image timeout
with a retry, since a rare seed can stall BNOT's Newton solve for hours.

Generators skip images whose outputs already exist, so an interrupted run resumes.

Run from the project root:
    python experiments/spectral_analysis_unconstrained_stage_1_old.py --no-run-gbn --no-run-baseline --no-run-control   # CPU job
    python experiments/spectral_analysis_unconstrained_stage_1_old.py --no-run-wvs --no-run-bnot                       # GPU job
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── Input / output ──────────────────────────────────────────────────────────
OUT_DIR = "experiments/outputs/spectral_analysis_unconstrained"    # stage-0 images + manifest live here
SOURCE_BASE_DIR = "experiments/outputs/spectral_analysis"          # only for COPY_EXISTING

GRID_SIZE = 32             # 32 x 32 = 1024 points

# ── What to run ─────────────────────────────────────────────────────────────
RUN_WVS = True             # CPU
RUN_BNOT = True            # CPU
RUN_GBN = True             # GPU
RUN_BASELINE = True        # GPU
RUN_CONTROL = True         # GPU
COPY_EXISTING = False      # copy COPY_METHODS from SOURCE_BASE_DIR (refused unless the images match)
COPY_METHODS = ["WVS", "BNOT", "GBN"]

# ── External samplers (outside this repository; each runs from its own folder) ──
WVS_DIR = "/groups/asharf_group/ofirgila/Rougier-2017"
WVS_PYTHON = "/home/ofirgila/.conda/envs/sd/bin/python"
WVS_SCRIPT = "src/wvs_data_gen.py"

GBN_DIR = "/groups/asharf_group/ofirgila/GaussianBlueNoise"
GBN_PYTHON = "/home/ofirgila/.conda/envs/sd/bin/python"
GBN_SCRIPT = "scripts/gbn_data_gen.py"

BNOT_DIR = "/groups/asharf_group/ofirgila/BNOT_new"
BNOT_PYTHON = "/home/ofirgila/.conda/envs/bnot/bin/python"
BNOT_SCRIPT = "bnot_data_gen.py"
BNOT_SEED = -1             # fresh random seed per realization
BNOT_EPSILON = 0.01
BNOT_MAX_ITERS = 25
BNOT_TIMEOUT_S = 600       # per image; FINDINGS.md: ample for all 150 images, one hit it once
BNOT_ATTEMPTS = 3          # a timed-out image is retried (with a new seed, as BNOT_SEED is -1)

# ── Base model (unconstrained) ──────────────────────────────────────────────
BASELINE_CONFIG_PATH = "config/GBN/config.json"
BASELINE_CKPT_PATH = "config/GBN/model.ckpt"
BASELINE_METHOD = "Base-GBN"                # -> target_Base-GBN_<N>
BASELINE_TIMESTEPS = 1000
BASELINE_INFER_TRUNCATION_RATIO = 1.0       # full denoising from pure noise
BASELINE_SEED = 42                          # one seed; the 25 realizations are the 25 batch samples
BASELINE_RESAMPLE_JUMPS = 0
DEVICE = "cuda"

# ── Our ControlNet (same checkpoints as the original spectral analysis) ─────
CONTROL_DATA_GEN = "experiments/control_data_gen.py"
CONTROL_RUNS = [
    # (method -> target_<method>_<N>, control checkpoint)
    ("CN-WVS", "control_v4/train_outputs_Icons-50_1024_WVS/checkpoints/dynamic_ep5000.ckpt"),
    ("CN-GBN", "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Unconstrained spectral analysis, stage 1: build the point sets")
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--source-base", default=SOURCE_BASE_DIR)
    p.add_argument("--grid-size", type=int, default=GRID_SIZE)
    p.add_argument("--run-wvs", action=argparse.BooleanOptionalAction, default=RUN_WVS)
    p.add_argument("--run-bnot", action=argparse.BooleanOptionalAction, default=RUN_BNOT)
    p.add_argument("--run-gbn", action=argparse.BooleanOptionalAction, default=RUN_GBN)
    p.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=RUN_BASELINE)
    p.add_argument("--run-control", action=argparse.BooleanOptionalAction, default=RUN_CONTROL)
    p.add_argument("--copy-existing", action=argparse.BooleanOptionalAction, default=COPY_EXISTING)
    p.add_argument("--copy-methods", default=",".join(COPY_METHODS))
    p.add_argument("--bnot-timeout", type=int, default=BNOT_TIMEOUT_S)
    p.add_argument("--baseline-config", default=BASELINE_CONFIG_PATH)
    p.add_argument("--baseline-ckpt", default=BASELINE_CKPT_PATH)
    p.add_argument("--baseline-timesteps", type=int, default=BASELINE_TIMESTEPS)
    p.add_argument("--baseline-infer-truncation-ratio", type=float, default=BASELINE_INFER_TRUNCATION_RATIO)
    p.add_argument("--baseline-seed", type=int, default=BASELINE_SEED)
    p.add_argument("--baseline-resample-jumps", type=int, default=BASELINE_RESAMPLE_JUMPS)
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


# ── copy (only when the images are the same) ────────────────────────────────

def same_sources(out, src_base, stems):
    import cv2
    for s in stems:
        a = cv2.imread(str(out / "source" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(src_base / "source" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or a.shape != b.shape or not np.array_equal(a, b):
            return False
    return True


def copy_existing(src_base, out, methods, n_points, stems):
    if not same_sources(out, src_base, stems):
        print(f"  [copy refused] the stage-0 images differ from {src_base / 'source'} (e.g. its 1 px "
              f"white margin), so its results do not apply here; run the samplers instead")
        return
    for m in methods:
        src, dst = src_base / f"target_{m}_{n_points}", out / f"target_{m}_{n_points}"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for s in stems:
            for ext in (".npy", ".png"):
                f = src / f"{s}{ext}"
                if f.exists():
                    shutil.copy2(f, dst / f.name)
                    n += ext == ".npy"
        flag = "" if n == len(stems) else f"   [WARN] expected {len(stems)}"
        print(f"  copied {m:8s} {n:3d} point sets  {src} -> {dst}{flag}")


# ── external samplers ───────────────────────────────────────────────────────

def run_external(name, cwd, cmd):
    print(f"  {name}: (cd {cwd}) {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def run_wvs(out, n_points):
    run_external("WVS", WVS_DIR, [WVS_PYTHON, WVS_SCRIPT, "--data_path", str(out.resolve()),
                                  "--target_folder", f"target_WVS_{n_points}", "--n", "-1",
                                  "--n_point", str(n_points), "--no-track_time"])


def run_gbn(out, n_points):
    run_external("GBN", GBN_DIR, [GBN_PYTHON, GBN_SCRIPT, "--data_path", str(out.resolve()),
                                  "--target_folder", f"target_GBN_{n_points}", "--n", "-1",
                                  "--n_points", str(n_points), "--no-track_time"])


def run_bnot(out, n_points, stems, timeout_s):
    """One image per call, each under a timeout and retried, as FINDINGS.md prescribes.

    The image index is its position in the generator's sorted list, which is the stems' sorted
    order. The process group is killed on timeout, so the solver binary does not outlive it.
    """
    dst = out / f"target_BNOT_{n_points}"
    for i, s in enumerate(sorted(stems)):
        for attempt in range(1, BNOT_ATTEMPTS + 1):
            if (dst / f"{s}.npy").exists():
                break
            cmd = [BNOT_PYTHON, BNOT_SCRIPT, "--data_path", str(out.resolve()),
                   "--target_folder", dst.name, "--n", "-1", "--start", str(i), "--end", str(i + 1),
                   "--num_sites", str(n_points), "--seed", str(BNOT_SEED),
                   "--epsilon", str(BNOT_EPSILON), "--max_iters", str(BNOT_MAX_ITERS), "--no-track_time"]
            proc = subprocess.Popen(cmd, cwd=BNOT_DIR, start_new_session=True)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                print(f"  BNOT {s}: timed out after {timeout_s}s (attempt {attempt}/{BNOT_ATTEMPTS})")
                continue
            if proc.returncode != 0:
                print(f"  BNOT {s}: exit code {proc.returncode} (attempt {attempt}/{BNOT_ATTEMPTS})")
        status = "ok" if (dst / f"{s}.npy").exists() else "FAILED"
        print(f"  BNOT [{i + 1}/{len(stems)}] {s}: {status}", flush=True)


# ── base model and ControlNet ───────────────────────────────────────────────

def run_baseline(args, out, n_points, stems, margin_frac):
    import torch
    from stress_test_stage_1 import _resolve_ckpt_path, offsets_batch_to_pointsets, run_sdedit_branch
    from utils.Config import ParseSampleConfig

    device = torch.device(args.device)
    torch.manual_seed(args.baseline_seed)
    np.random.seed(args.baseline_seed)

    ckpt = _resolve_ckpt_path(args.baseline_ckpt)
    diffusion = ParseSampleConfig(args.baseline_config)
    diffusion.load_state_dict(torch.load(ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)
    denoiser = diffusion.model
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad = False

    g = args.grid_size
    x_noisy = torch.randn(len(stems), 2, g, g, device=device)
    print(f"  base model {args.baseline_config} + {ckpt}: {len(stems)} samples of {g}x{g} from pure "
          f"noise, {int(args.baseline_timesteps * args.baseline_infer_truncation_ratio)} steps")
    raw = run_sdedit_branch(diffusion, denoiser, x_noisy, device,
                            timesteps=args.baseline_timesteps,
                            truncation_ratio=args.baseline_infer_truncation_ratio,
                            resample_jumps=args.baseline_resample_jumps, desc=BASELINE_METHOD)
    pts_batch = offsets_batch_to_pointsets(raw)

    dst = out / f"target_{BASELINE_METHOD}_{n_points}"
    dst.mkdir(parents=True, exist_ok=True)
    for s, pts in zip(stems, pts_batch):
        pts = np.clip(np.asarray(pts, dtype=np.float64), 0.0, np.nextafter(1.0, 0.0))
        # Into the square the images' content occupies. With stage 0's default (no margin) this is
        # the identity; with a margin it matches the analysis' crop, which then inverts it exactly.
        np.save(dst / f"{s}.npy", margin_frac + pts * (1.0 - 2.0 * margin_frac))
    np.save(dst / "raw_offsets.npy", raw)
    print(f"  wrote {len(stems)} point sets ({pts_batch.shape[1]} points each) -> {dst}")


def run_control(args, out, n_points):
    for method, ckpt in CONTROL_RUNS:
        run_external(method, os.getcwd(),
                     [sys.executable, CONTROL_DATA_GEN, "--data_path", str(out), "--control_ckpt_path", ckpt,
                      "--grid_size", str(args.grid_size), "--target_folder", f"target_{method}_{n_points}",
                      "--no-track_time", "--device", args.device])


def main():
    args = parse_args()
    out, src_base = Path(args.output), Path(args.source_base)
    n_points = args.grid_size ** 2
    mpath = out / "manifest.json"
    if not mpath.exists():
        print(f"ERROR: {mpath} not found; run spectral_analysis_unconstrained_stage_0_old.py first")
        return 2
    manifest = json.loads(mpath.read_text())
    stems = [im["stem"] for im in manifest["images"]]
    missing = [st for st in stems if not (out / "source" / f"{st}.png").exists()]
    if missing:
        print(f"ERROR: {len(missing)} stage-0 image(s) missing from {out / 'source'}, e.g. {missing[:2]}")
        return 2
    res, margin_px = int(manifest.get("resolution", 0) or 0), int(manifest.get("margin_px", 0) or 0)
    margin_frac = margin_px / res if res > 0 and margin_px > 0 else 0.0

    print(f"Output : {out}")
    print(f"Images : {stems[0]} .. {stems[-1]} ({len(stems)} realizations, "
          f"rho = {manifest['images'][0]['rho']:.2f}, {margin_px}px white margin)")
    print(f"Run    : WVS={args.run_wvs} BNOT={args.run_bnot} GBN={args.run_gbn} "
          f"base={args.run_baseline} control={args.run_control} copy={args.copy_existing}")

    if args.copy_existing:
        copy_existing(src_base, out, [m.strip() for m in args.copy_methods.split(",") if m.strip()],
                      n_points, stems)
    if args.run_wvs:
        run_wvs(out, n_points)
    if args.run_bnot:
        run_bnot(out, n_points, stems, args.bnot_timeout)
    if args.run_gbn:
        run_gbn(out, n_points)
    if args.run_baseline:
        run_baseline(args, out, n_points, stems, margin_frac)
    if args.run_control:
        run_control(args, out, n_points)

    print("\nPoint sets now in the output folder:")
    for d in sorted(out.glob(f"target_*_{n_points}")):
        print(f"  {d.name:24s} {len(list(d.glob('uniform_*.npy'))):3d}")
    print("Next: spectral_analysis_unconstrained_stage_2_old.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
