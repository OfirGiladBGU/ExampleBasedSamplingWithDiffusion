"""Stage 1 (local GPU): sample the unconditional baseline point-diffusion model, for the
"learned methods" shot.

This is the same baseline branch as experiments/run_stress_test.py: the model of
Doignies et al. trained on one fixed density (the stress-test density), sampled from pure
noise over the full schedule, with no conditioning input at all. Writes

    video/assets/results/baseline_stress1.npz
        samples     (S,N,2) final point sets, unit square, y down
        trajectory  (K,S,N,2) the same S samples after every STEP_INTERVAL-th step, from pure noise
        steps       (K,) elapsed denoising steps of each trajectory row (last = final)

which the shot shows next to the target density instead of the crop from the paper figure.

    python video/stage_1_baseline.py
    python video/stage_1_baseline.py --config <config.json> --ckpt <model.ckpt> --samples 4

Run from the project root in an environment with torch (e.g. conda env "qmcdiffusion").
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "video"))

from common import ASSETS_DIR, RESULTS_DIR  # noqa: E402

CONFIG_PATH = str(ASSETS_DIR / "weights" / "GBN_stress1" / "config.json")
CKPT_PATH = str(ASSETS_DIR / "weights" / "GBN_stress1" / "model.ckpt")
OUTPUT = str(RESULTS_DIR / "baseline_stress1.npz")
SAMPLES = 4
GRID_SIZE = 32               # 1024 points, as in the paper's stress-test figure
TIMESTEPS = 1000
SEED = 0
STEP_INTERVAL = 10           # snapshot every 10th of the 999 steps -> 101 frames
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--ckpt", default=CKPT_PATH)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--grid-size", type=int, default=GRID_SIZE)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--step-interval", type=int, default=STEP_INTERVAL)
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


def main():
    a = parse_args()
    import torch
    from tqdm import tqdm
    from data.Transforms import to_pointset_optimal_transport
    from utils.Config import ParseSampleConfig

    for f in (a.config, a.ckpt):
        if not Path(f).exists():
            raise SystemExit(f"missing baseline file: {f}\n(copy config_trained/GBN_stress1/ from the server "
                             f"to {Path(a.config).parent})")
    torch.manual_seed(a.seed)
    diffusion = ParseSampleConfig(a.config)
    diffusion.load_state_dict(torch.load(a.ckpt, map_location="cpu")["diffu"])
    diffusion.to(a.device)
    diffusion.model.eval()
    diffusion.set_num_timesteps(a.timesteps)
    diffusion.eval()

    # pure noise, full schedule: exactly the baseline branch of run_stress_test.py
    img = torch.randn((a.samples, 2, a.grid_size, a.grid_size), device=a.device)
    t_start = diffusion.num_timesteps - 1

    def to_points(batch):
        out = []
        for off in batch.detach().cpu().numpy():
            pts = to_pointset_optimal_transport(off.astype(np.float64))
            out.append(pts.reshape(pts.shape[0], -1).T)
        return np.stack(out)

    frames, steps = [to_points(img)], [0]              # step 0 = the pure-noise start
    with torch.no_grad():
        for k, i in enumerate(tqdm(reversed(range(t_start)), total=t_start, desc="baseline"), 1):
            t = torch.full((a.samples,), i, dtype=torch.int64, device=a.device)
            img = diffusion.p_sample(img, cond=None, t=t, clip_denoised=diffusion.sample_clip, with_sampling=True)
            if k % a.step_interval == 0 or k == t_start:
                frames.append(to_points(img))
                steps.append(k)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, samples=frames[-1].astype(np.float32),
                        trajectory=np.stack(frames).astype(np.float32),
                        steps=np.asarray(steps, np.int32), t_start=np.int32(t_start))
    print(f"{a.samples} baseline samples x {a.grid_size ** 2} points, {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
