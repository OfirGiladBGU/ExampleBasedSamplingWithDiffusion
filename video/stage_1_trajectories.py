"""Stage 1 (server, GPU): sample every trajectory listed in video/shots.json and save the
point positions along the reverse process.

For each entry of shots.json["trajectories"] this runs our sampler (control_v4, SDEdit
truncation as in the paper) on the entry's image, with denoising snapshots enabled, and
packs the result into one file:

    video/assets/trajectories/<name>.npz
        image       (H,W) float32 in [0,1], the condition image
        prior       (N,2) the clean rejection-sampling prior, before noise is added
        points      (K,N,2) point positions after denoising steps 0, STEP_INTERVAL, ... and the final result
        steps       (K,) elapsed denoising steps of each row of `points`
        t_start     reverse-process start step (500 with truncation 0.5)
        grid_size   N = grid_size**2

Positions are in the unit square with y pointing down (image convention), exactly as
sample_control.py exports them.

    python video/stage_1_trajectories.py                       # all trajectories
    python video/stage_1_trajectories.py --only monkey_g24,monkey_g64
    python video/stage_1_trajectories.py --truncation 1.0      # full schedule from pure noise

Run from the project root, in the training environment (conda env "stippling"), on a GPU
node; see video/server/run_stage_1.sbatch.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "video"))

from common import TRAJ_DIR, load_shots  # noqa: E402

BASE_CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT_PATH = ""          # full-weights checkpoints restore the denoiser themselves
OUTPUT_DIR = str(TRAJ_DIR)
ONLY = ""                    # comma-separated trajectory names; "" = all
TRUNCATION = 0.5             # the paper's SDEdit start (t = 500 of 1000)
EVAL_TIMESTEPS = 1000
STEP_INTERVAL = 10           # save every 10th denoising step -> 51 frames for t_start = 500
SMART_INIT_SEED = 42
ENABLE_GECCO = True
ENABLE_ADAPTIVE_GATE_INJECTION = True
DEVICE = "cuda"
OVERWRITE = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--only", default=ONLY, help="Comma-separated trajectory names")
    p.add_argument("--truncation", type=float, default=TRUNCATION)
    p.add_argument("--eval-timesteps", type=int, default=EVAL_TIMESTEPS)
    p.add_argument("--step-interval", type=int, default=STEP_INTERVAL)
    p.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    p.add_argument("--base-config-path", default=BASE_CONFIG_PATH)
    p.add_argument("--base-ckpt-path", default=BASE_CKPT_PATH)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE)
    return p.parse_args()


def offsets_to_points(off_2gg):
    """(2,G,G) offset grid -> (N,2) points, the same conversion as sample_control's export."""
    from data.Transforms import to_pointset_optimal_transport
    pts = to_pointset_optimal_transport(np.asarray(off_2gg, dtype=np.float64))
    return pts.reshape(pts.shape[0], -1).T


def run_one(name, spec, ckpt, a, pipelines):
    from control_v4.sample_control import load_condition, load_pipeline, process_single_image
    from control_v4.smart_init import generate_smart_init_points_from_density

    grid = int(spec["grid_size"])
    image = Path(spec["image"])
    if not image.exists():
        raise FileNotFoundError(f"[{name}] image not found: {image}")
    key = (ckpt, grid)
    if key not in pipelines:
        pipelines[key] = load_pipeline(
            base_config_path=a.base_config_path, base_ckpt_path=a.base_ckpt_path,
            control_ckpt_path=ckpt, grid_size=grid, enable_gecco=ENABLE_GECCO,
            enable_adaptive_gate_injection=ENABLE_ADAPTIVE_GATE_INJECTION,
            smart_init_features=False, sdf_features=False, batch_coords_features=False,
            device=a.device)
    diffusion, control_net = pipelines[key]

    work = Path(tempfile.mkdtemp(prefix=f"video_{name}_"))
    try:
        process_single_image(
            image_path=image, diffusion=diffusion, control_net=control_net,
            grid_size=grid, eval_timesteps=a.eval_timesteps, truncation_ratio=a.truncation,
            t_start_step=-1, resample_jumps=0, smart_init_features=False, sdf_features=False,
            smart_init_seed=a.smart_init_seed, device=a.device, n_samples=1,
            show_denoising_interval=a.step_interval, output_dir=work,
            export_conditions=False, export_png=False, export_npy=True, track_time=False,
            show_denoising=True, npy_dir=work / "final", denoising_dir=work / "steps")

        stem = image.stem
        step_files = sorted((work / "steps" / "npy").glob(f"{stem}_step_*.npy"))
        steps, frames = [], []
        for f in step_files:
            steps.append(int(f.stem.rsplit("_", 1)[-1]))
            frames.append(offsets_to_points(np.load(f)[0]))
        final = np.load(work / "final" / f"{stem}.npy").astype(np.float64)
        t_start = int(np.clip(int(a.eval_timesteps * a.truncation), 1, a.eval_timesteps))
        steps.append(t_start)
        frames.append(final)

        image_01 = load_condition(image, grid, a.device, sdf_features=False, sdf_truncate_px=8.0)[0]
        prior = generate_smart_init_points_from_density(image_01, n_points=grid * grid,
                                                        seed=a.smart_init_seed)
        return dict(image=np.asarray(image_01, np.float32), prior=np.asarray(prior, np.float32),
                    points=np.stack(frames).astype(np.float32), steps=np.asarray(steps, np.int32),
                    t_start=np.int32(t_start), grid_size=np.int32(grid))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    a = parse_args()
    data = load_shots()
    jobs = data["trajectories"]
    names = list(jobs) if not a.only else [n.strip() for n in a.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in jobs]
    if unknown:
        raise SystemExit(f"unknown trajectory name(s): {unknown}")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    pipelines = {}
    for i, name in enumerate(names, 1):
        dst = out / f"{name}.npz"
        if dst.exists() and not a.overwrite:
            print(f"[{i}/{len(names)}] {name}: exists, skipping (--overwrite to redo)")
            continue
        spec = jobs[name]
        ckpt = data["checkpoints"][spec["checkpoint"]]
        print(f"[{i}/{len(names)}] {name}: {spec['image']}  grid {spec['grid_size']}  ckpt {ckpt}", flush=True)
        res = run_one(name, spec, ckpt, a, pipelines)
        np.savez_compressed(dst, **res)
        print(f"    {res['points'].shape[0]} frames x {res['points'].shape[1]} points -> {dst}", flush=True)
    print("stage 1 trajectories done")


if __name__ == "__main__":
    main()
