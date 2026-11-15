"""Stage 4: the direct point-optimization baseline for tone correction (rebuttal E7).

Instead of optimizing the conditioning density through the sampler (stage 1, "field"), move the
points themselves: minimize the same rendered-tone objective of stage 1,

    || render(points) - target darkness ||^2  +  mu * E_rep(points),

with Adam, from a given starting stipple. E_rep is a Gaussian blue-noise repulsion energy (the
mean over points of sum_j exp(-|x_i - x_j|^2 / (2 s^2)), s = SIGMA_FACTOR / sqrt(N)), which keeps
the points from clumping; mu = 0 is the unregularized fit. Starting stipples, all at 1024 points on
the same targets as stage 1:
    ours  our sampler's uncorrected output (stage 1 "none" coordinates)
    GBN   the GBN stipple of the image (the Table 1 point set)
    BNOT  the BNOT stipple of the image (the Table 1 point set)

The renderer (resolution, dot size, ink gain) is read from the stage-1 run, so the tone numbers are
directly comparable with stage 1's none / curve / field.

Every resulting point set, and stage 1's ours-none / ours-curve / ours-field, is written as
<output>/<config>/<stem>.npy ([x/w, y/h], y-down) so that few_step_sampling_stage_2.py can score
its point distribution; the tone metrics go to <output>/tone.json.

Usage:
    python experiments/tone_results_stage_4_direct_points.py
    python experiments/few_step_sampling_stage_2.py experiments/outputs/tone_results/direct_points \\
        --source experiments/outputs/tone_results/source --workers 8
    python experiments/tone_results_stage_4_direct_points.py --summarize
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

from tone_results_utils import psnr, render_darkness, ssim  # noqa: E402

ROOT_DIR = "experiments/outputs/tone_results"
STAGE1_DIR = f"{ROOT_DIR}/main"
OUT_DIR = f"{ROOT_DIR}/direct_points"
MANIFEST = f"{ROOT_DIR}/manifest.json"
TEACHER_DIR = "experiments/outputs/z_validation_data/Icons-50_1024/target_{}_1024"
STARTS = "ours,GBN,BNOT"
STAGE1_CONFIGS = "none,curve,field"
MUS = "0,0.001,0.01,0.1"
STEPS = 300
LR = 1e-3
SIGMA_FACTOR = 0.5
DEVICE = "cuda"
SUMMARY_METRICS = {"M1_v2_hot_energy": "HOT", "M2_v1_capacity_delta_c": "cap dc",
                   "M2_v2_centroid_displacement": "D", "M5_spatial_measure_rho_mean": "rho_s"}


def parse_args():
    p = argparse.ArgumentParser(description="Direct point-optimization baseline for tone correction")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--stage1", default=STAGE1_DIR, help="Stage-1 run: renderer settings and our coordinates")
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--teacher-dir", default=TEACHER_DIR, help="Format string with the method name")
    p.add_argument("--starts", default=STARTS)
    p.add_argument("--stage1-configs", default=STAGE1_CONFIGS)
    p.add_argument("--mus", default=MUS)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--sigma-factor", type=float, default=SIGMA_FACTOR)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--summarize", action="store_true", help="Only merge tone.json with the stage-2 metrics")
    return p.parse_args()


def repulsion(coords, sigma):
    d2 = torch.cdist(coords, coords).pow(2)
    k = torch.exp(-d2 / (2.0 * sigma * sigma))
    n = coords.shape[1]
    return (k.sum((1, 2)) - n) / n                      # drop the i == j terms


def optimize(coords0, target_dark, a, cfg, mu):
    x = torch.nn.Parameter(coords0.clone())
    opt = torch.optim.Adam([x], lr=a.lr)
    sigma = a.sigma_factor / np.sqrt(coords0.shape[1])
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True)
        dark = render_darkness(x, cfg["render_res"], cfg["dot_sigma_px"], cfg["ink_gain"])
        loss = torch.nn.functional.mse_loss(dark, target_dark)
        if mu > 0:
            loss = loss + mu * repulsion(x, sigma).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
    return x.detach()


def tone_scores(coords, target_dark, cfg):
    with torch.no_grad():
        d = render_darkness(coords, cfg["render_res"], cfg["dot_sigma_px"], cfg["ink_gain"])
    d, t = d.cpu().numpy()[0, 0], target_dark.cpu().numpy()[0, 0]
    return {"psnr": psnr(d, t), "ssim": ssim(d, t), "mae": float(np.abs(d - t).mean())}


def run(a):
    cfg = json.loads((Path(a.stage1) / "summary.json").read_text())["config"]
    images = json.loads(Path(a.manifest).read_text())["images"]
    out = Path(a.output)
    mus = [float(m) for m in a.mus.split(",")]
    tone = {}
    for i, entry in enumerate(images, 1):
        stem = entry["stem"]
        z = np.load(Path(a.stage1) / f"{stem}.npz")
        target = torch.as_tensor(1.0 - z["image_01"].astype(np.float32), device=a.device)[None, None]
        target = torch.nn.functional.interpolate(target, size=(cfg["render_res"],) * 2, mode="area")
        sets = {f"ours-{c}": z[f"{c}_coords"] for c in a.stage1_configs.split(",")}
        starts = {"ours": z["none_coords"]}
        for m in [s for s in a.starts.split(",") if s != "ours"]:
            starts[m] = np.load(Path(a.teacher_dir.format(m)) / f"{stem}.npy")
            sets[f"{m}-start"] = starts[m]
        t0 = time.time()
        for s in a.starts.split(","):
            c0 = torch.as_tensor(starts[s], dtype=torch.float32, device=a.device)[None]
            for mu in mus:
                sets[f"{s}-direct-mu{mu:g}"] = optimize(c0, target, a, cfg, mu).cpu().numpy()[0]
        tone[stem] = {}
        for name, pts in sets.items():
            (out / name).mkdir(parents=True, exist_ok=True)
            np.save(out / name / f"{stem}.npy", np.asarray(pts, np.float64))
            tone[stem][name] = tone_scores(torch.as_tensor(pts, dtype=torch.float32, device=a.device)[None],
                                           target, cfg)
        print(f"[{i}/{len(images)}] {stem}  {time.time() - t0:.1f}s  " + "  ".join(
            f"{k} {v['psnr']:.2f}" for k, v in tone[stem].items()), flush=True)
    (out / "tone.json").write_text(json.dumps({"config": vars(a), "renderer": {
        k: cfg[k] for k in ("render_res", "dot_sigma_px", "ink_gain")}, "results": tone}, indent=2))


def summarize(a):
    out = Path(a.output)
    tone = json.loads((out / "tone.json").read_text())["results"]
    names = list(next(iter(tone.values())))
    lines = ["# Tone correction: density optimization through the sampler vs direct point optimization", "",
             f"{len(tone)} targets, 1024 points. Tone: rendered darkness vs target (stage-1 renderer). "
             "Distribution metrics: few_step_sampling_stage_2 against the original image.", "",
             "| point set | PSNR | SSIM | MAE | " + " | ".join(SUMMARY_METRICS.values()) + " | clumped % |",
             "|" + "---|" * (5 + len(SUMMARY_METRICS))]
    res = {}
    for name in names:
        stems = sorted(tone)
        mets = [json.loads((out / f"{name}_json" / f"{s}.json").read_text())
                for s in stems if (out / f"{name}_json" / f"{s}.json").exists()]
        clumped = []
        for s in stems:
            pts = np.load(out / name / f"{s}.npy")
            ref = np.load(out / "ours-none" / f"{s}.npy")
            nn = torch.cdist(torch.as_tensor(pts)[None], torch.as_tensor(pts)[None])[0]
            nn.fill_diagonal_(float("inf"))
            nn_ref = torch.cdist(torch.as_tensor(ref)[None], torch.as_tensor(ref)[None])[0]
            nn_ref.fill_diagonal_(float("inf"))
            thr = 0.5 * float(nn_ref.min(1).values.median())
            clumped.append(100.0 * float((nn.min(1).values < thr).double().mean()))
        r = {k: float(np.mean([tone[s][name][k] for s in stems])) for k in ("psnr", "ssim", "mae")}
        r.update({short: float(np.mean([m[key] for m in mets])) if mets else float("nan")
                  for key, short in SUMMARY_METRICS.items()})
        r["clumped_pct"], r["scored"] = float(np.mean(clumped)), len(mets)
        res[name] = r
        lines.append(f"| {name} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['mae']:.4f} | "
                     + " | ".join(f"{r[s]:.4g}" for s in SUMMARY_METRICS.values())
                     + f" | {r['clumped_pct']:.1f} |")
    lines += ["", "clumped %: points whose nearest neighbour is closer than half the median nearest-neighbour "
                  "distance of our uncorrected stipple of the same image."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))


def main():
    a = parse_args()
    if a.summarize:
        summarize(a)
    else:
        run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
