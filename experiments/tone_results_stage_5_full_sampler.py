"""Stage 5: does the tone correction carry over to the full sampler? (rebuttal, tone section)

Stage 1 optimizes the requested density through the one-step x0 estimate at t = 120 and also
reports the stipples of that one-step path. A user of the method would instead request the
optimized density rho* from the full sampler. This stage checks that:

    prepare    writes each requested density as an 8-bit condition image (intensity = 1 - rho):
               <output>/conditions/<config>/<stem>.png for config in none (rho0, the target itself),
               curve and field (stage 1's optimized densities)
    (sample)   rebuttal_sample.sbatch on every condition folder: full SDEdit sampling with the
               paper's ddpm-500 and the few-step ddim-25-eta1, Smart Init from the requested
               density, several noise seeds
    summarize  renders every stipple with the stage-1 renderer and scores it against the ORIGINAL
               target darkness: PSNR / SSIM / MAE per (config, sampler), mean over images and seeds,
               and the per-image gain of field and curve over none (paired over images, seeds averaged)

Usage:
    python experiments/tone_results_stage_5_full_sampler.py --prepare
    (sbatch experiments/rebuttal_sample.sbatch ... per condition folder and seed)
    python experiments/tone_results_stage_5_full_sampler.py --summarize
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tone_results_utils import psnr, render_darkness, ssim  # noqa: E402

ROOT_DIR = "experiments/outputs/tone_results"
STAGE1_DIR = f"{ROOT_DIR}/main"
OUT_DIR = f"{ROOT_DIR}/full_sampler"
MANIFEST = f"{ROOT_DIR}/manifest.json"
CONFIGS = "none,curve,field"
SAMPLERS = "ddpm-500,ddim-25-eta1"
LABEL_FORMAT = "{config}-seed{seed}"
POINTS = 1024
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Tone correction through the full sampler")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--stage1", default=STAGE1_DIR)
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--configs", default=CONFIGS)
    p.add_argument("--samplers", default=SAMPLERS)
    p.add_argument("--points", type=int, default=POINTS)
    p.add_argument("--device", default=DEVICE)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    return p.parse_args()


def prepare(a):
    images = json.loads(Path(a.manifest).read_text())["images"]
    for c in a.configs.split(","):
        d = Path(a.output) / "conditions" / c
        d.mkdir(parents=True, exist_ok=True)
        for e in images:
            rho = np.load(Path(a.stage1) / f"{e['stem']}.npz")[f"{c}_rho"]
            img = np.clip(np.rint((1.0 - rho) * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(d / f"{e['stem']}.png")
        print(f"{c}: {len(images)} condition images -> {d}")


def summarize(a):
    cfg = json.loads((Path(a.stage1) / "summary.json").read_text())["config"]
    images = json.loads(Path(a.manifest).read_text())["images"]
    samples = Path(a.output) / "samples"
    configs = a.configs.split(",")
    seeds = sorted({int(d.name.rsplit("-seed", 1)[1].split("_")[0]) for d in samples.glob("*-seed*_*")})
    scores = {}                                       # (config, sampler) -> stem -> list over seeds
    for e in images:
        z = np.load(Path(a.stage1) / f"{e['stem']}.npz")
        t = torch.as_tensor(1.0 - z["image_01"].astype(np.float32), device=a.device)[None, None]
        t = torch.nn.functional.interpolate(t, size=(cfg["render_res"],) * 2, mode="area").cpu().numpy()[0, 0]
        for c in configs:
            for s in a.samplers.split(","):
                for seed in seeds:
                    f = samples / f"{LABEL_FORMAT.format(config=c, seed=seed)}_{a.points}" / s / f"{e['stem']}.npy"
                    if not f.exists():
                        continue
                    pts = torch.as_tensor(np.load(f), dtype=torch.float32, device=a.device)[None]
                    with torch.no_grad():
                        d = render_darkness(pts, cfg["render_res"], cfg["dot_sigma_px"], cfg["ink_gain"])
                    d = d.cpu().numpy()[0, 0]
                    scores.setdefault((c, s), {}).setdefault(e["stem"], []).append(
                        (psnr(d, t), ssim(d, t), float(np.abs(d - t).mean())))
    one_step = json.loads((Path(a.stage1) / "summary.json").read_text())["results"]
    lines = ["# Tone correction through the full sampler", "",
             f"{len(images)} targets, seeds {seeds}, Smart Init from the requested density; rendered with the "
             f"stage-1 renderer (ink gain {cfg['ink_gain']:.4f}) and scored against the original target.", "",
             "| sampler | requested density | PSNR | SSIM | MAE | PSNR gain vs none (images improved) |",
             "|---|---|---|---|---|---|"]
    res = {}
    for s in [f"stage-1 path ({cfg.get('mode', 'onestep')})"] + a.samplers.split(","):
        for c in configs:
            if s.startswith("stage-1"):
                if c not in next(iter(one_step.values())):
                    continue
                per = {k: [(v[c]["psnr"], v[c]["ssim"], v[c]["mae"])] for k, v in one_step.items()}
            else:
                per = scores.get((c, s), {})
            if not per:
                continue
            m = np.array([np.mean(v, axis=0) for v in per.values()])
            row = {"psnr": float(m[:, 0].mean()), "ssim": float(m[:, 1].mean()), "mae": float(m[:, 2].mean()),
                   "images": len(per)}
            gain = ""
            if c != "none":
                base = {k: v["none"]["psnr"] for k, v in one_step.items()} if s.startswith("stage-1") else \
                    {k: np.mean([x[0] for x in v]) for k, v in scores.get(("none", s), {}).items()}
                stems = sorted(set(base) & set(per))
                g = np.array([np.mean([x[0] for x in per[k]]) - base[k] for k in stems])
                p = float(wilcoxon(g).pvalue) if np.any(g != 0) else 1.0
                row.update({"gain_db": float(g.mean()), "improved": int((g > 0).sum()), "p": p})
                gain = f"{g.mean():+.2f} dB ({int((g > 0).sum())}/{len(g)}, p={p:.1g})"
            res.setdefault(s, {})[c] = row
            lines.append(f"| {s} | {c} | {row['psnr']:.2f} | {row['ssim']:.4f} | {row['mae']:.4f} | {gain} |")
    out = Path(a.output)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))


def main():
    a = parse_args()
    if a.prepare:
        prepare(a)
    else:
        summarize(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
