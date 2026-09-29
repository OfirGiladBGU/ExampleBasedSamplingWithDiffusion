"""Stage 1 (server, GPU): record the tone-correction optimization of one image, step by step.

This re-runs the paper's "field" configuration (tone_results_stage_1_optimize.py, same
settings and the calibrated ink gain of the paper run) on one image, and keeps a snapshot
of the optimized density, the rendered stipple and the points every SNAPSHOT_EVERY steps.
The experiment code is used unchanged: the sampler and the renderer are only wrapped to
observe their inputs and outputs.

    video/assets/trajectories/tone_<stem>.npz
        image      (H,W) the target image in [0,1]
        target     (R,R) target darkness (1 - image, at render resolution)
        rho        (K,H,W) requested density at each snapshot (step 0 = the target itself)
        dark       (K,R,R) rendered darkness at each snapshot
        coords     (K,N,2) stipple points at each snapshot
        steps      (K,) optimizer step of each snapshot
        psnr       (K,) PSNR of the rendered stipple against the target

    python video/stage_1_tone.py
    python video/stage_1_tone.py --stem lg_2_woman_with_bunny_ears --snapshot-every 5

Run from the project root in the training environment, on a GPU node.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "video"))

from common import TRAJ_DIR  # noqa: E402

STEM = "lg_2_woman_with_bunny_ears"
MANIFEST = "experiments/outputs/tone_results/manifest.json"
PAPER_SUMMARY = "experiments/outputs/tone_results/main/summary.json"   # source of the ink gain
OUTPUT_DIR = str(TRAJ_DIR)
SNAPSHOT_EVERY = 5          # 300 optimizer steps -> 61 snapshots
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", default=STEM, help="Image stem from the tone manifest")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--paper-summary", default=PAPER_SUMMARY,
                   help="summary.json of the paper run; its settings and ink gain are reused")
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


def paper_args(summary_path, device):
    """The optimizer's own argparse namespace, filled from the paper run's saved config."""
    import tone_results_stage_1_optimize as T
    saved = json.loads(Path(summary_path).read_text())["config"]
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        a = T.parse_args()             # defaults of the experiment script ...
    finally:
        sys.argv = argv
    for k, v in saved.items():         # ... overridden by what the paper run actually used
        if hasattr(a, k) and v is not None and not isinstance(v, (dict, list)):
            setattr(a, k, type(getattr(a, k))(v) if getattr(a, k) is not None else v)
    a.device = device
    a.configs = "field"
    return a


def main():
    a = parse_args()
    import torch
    import tone_results_stage_1_optimize as T
    from tone_results_utils import build_stippler, psnr

    cfg = paper_args(a.paper_summary, a.device)
    entry = next((e for e in json.loads(Path(a.manifest).read_text())["images"] if e["stem"] == a.stem), None)
    if entry is None:
        raise SystemExit(f"{a.stem} is not in {a.manifest}")
    print(f"{a.stem}: steps {cfg.steps}, lr {cfg.lr}, ink gain {cfg.ink_gain:.4f}, mode {cfg.mode}")

    stippler = build_stippler(cfg, a.device)
    image_01 = stippler.prepare(entry["path"])

    # Observe the optimizer through its two callables: the sampler sees rho, the renderer
    # returns the darkness. One render call per optimizer step, plus a final one.
    record = {"rho": [], "dark": [], "coords": []}
    calls = {"n": 0}
    real_render = T.render_darkness

    class Watched(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, rho, generator=None):
            self._rho = rho
            return self.inner(rho, generator=generator)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    watched = Watched(stippler)

    def render_and_record(coords, *args, **kwargs):
        dark = real_render(coords, *args, **kwargs)
        k = calls["n"]
        calls["n"] += 1
        if k % a.snapshot_every == 0 or k == cfg.steps:
            record["rho"].append(watched._rho.detach().float().cpu().numpy()[0, 0])
            record["dark"].append(dark.detach().float().cpu().numpy()[0, 0])
            record["coords"].append(coords.detach().float().cpu().numpy()[0])
            record.setdefault("steps", []).append(min(k, cfg.steps))
        return dark

    T.render_darkness = render_and_record
    try:
        result = T.run_config(watched, image_01, "field", cfg, a.device, log=True)
    finally:
        T.render_darkness = real_render

    target = torch.as_tensor(1.0 - np.asarray(image_01, np.float32))[None, None]
    target = torch.nn.functional.interpolate(target, size=(cfg.render_res,) * 2, mode="area")[0, 0].numpy()
    dark = np.stack(record["dark"])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"tone_{a.stem}.npz"
    np.savez_compressed(
        dst, image=np.asarray(image_01, np.float32), target=target.astype(np.float32),
        rho=np.stack(record["rho"]).astype(np.float32), dark=dark.astype(np.float32),
        coords=np.stack(record["coords"]).astype(np.float32),
        steps=np.asarray(record["steps"], np.int32),
        psnr=np.asarray([psnr(d, target) for d in dark], np.float32))
    print(f"{len(dark)} snapshots, PSNR {psnr(dark[0], target):.2f} -> {psnr(dark[-1], target):.2f} dB"
          f" (paper run's final: {result['psnr']:.2f}) -> {dst}")


if __name__ == "__main__":
    main()
