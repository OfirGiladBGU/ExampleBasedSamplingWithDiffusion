#!/usr/bin/env python3
"""Rendered-tone fidelity of every Table 1 method (rebuttal E17).

A perceptual number next to the point-set metrics: each stipple is rendered with the tone
experiment's printer model (Gaussian dots, Beer-Lambert ink saturation; resolution, dot size and
ink gain read from the stage-1 tone run) and compared with the target darkness (1 - image, area
downsampled to the render resolution) by PSNR, SSIM and MAE. The renderer is fixed, so the
numbers compare methods on equal terms; they are not tuned per method or per dataset.

Reads the point sets through the re-score folders, outputs/rescore_current_metrics-<Dataset>/<METHOD>
(symlinks to the Table 1 point sets), plus any extra folder given with --extra NAME=PATH.

Writes outputs/rendered_tone_fidelity/summary.md and summary.json.

Run from the project root (GPU recommended):
    python experiments/rendered_tone_fidelity.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tone_results_utils import psnr, render_darkness, ssim  # noqa: E402

ROOT = "experiments/outputs"
OUTPUT_DIR = "experiments/outputs/rendered_tone_fidelity"
TONE_RUN = "experiments/outputs/tone_results/main/summary.json"
DATASETS = ["Icons-50_1024", "CelebA-5K_1024", "ShapeNet2D-4K_1600", "Anime-Chibi_1600"]
METHODS = ["BNOT", "GBN", "WVS", "CN-BNOT-paper", "CN-GBN-paper", "CN-WVS-paper"]
EXTRA = ["Icons-50_1024:CN-GBN-ddim25=experiments/outputs/few_step_sampling-Icons-50_1024/CN-GBN_1024/ddim-25-eta1",
         "Icons-50_1024:CN-BNOT-ddim25=experiments/outputs/few_step_sampling-Icons-50_1024/CN-BNOT_1024/ddim-25-eta1"]
REFERENCE = "GBN"
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--tone-run", default=TONE_RUN, help="Stage-1 tone summary.json with the renderer settings")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--extra", nargs="*", default=EXTRA, help="DATASET:NAME=FOLDER of extra point sets")
    p.add_argument("--reference", default=REFERENCE, help="Paired Wilcoxon reference method")
    p.add_argument("--device", default=DEVICE)
    return p.parse_args()


def target_darkness(path, res, device):
    img = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    t = torch.as_tensor(1.0 - img, device=device)[None, None]
    return torch.nn.functional.interpolate(t, size=(res, res), mode="area")


def main():
    args = parse_args()
    cfg = json.loads(Path(args.tone_run).read_text())["config"]
    res_px, sigma, gain = cfg["render_res"], cfg["dot_sigma_px"], cfg["ink_gain"]
    extra = {}
    for e in args.extra:
        ds, rest = e.split(":", 1)
        name, folder = rest.split("=", 1)
        extra.setdefault(ds, {})[name] = Path(folder)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    lines = ["# Rendered-tone fidelity (E17)", "",
             f"Renderer from the tone experiment: {res_px} px, dot sigma {sigma} px, ink gain {gain:.4f}. "
             f"Mean (std) over the test images; * = paired Wilcoxon p < 0.05 vs {args.reference}.", ""]
    for ds in args.datasets:
        base = Path(args.root) / f"rescore_current_metrics-{ds}"
        source = Path(args.root) / "z_validation_data" / ds / "source"
        folders = {m: base / m for m in args.methods}
        folders.update(extra.get(ds, {}))
        stems = sorted(set.intersection(*({p.stem for p in f.glob("*.npy")} for f in folders.values())))
        scores = {m: {"psnr": [], "ssim": [], "mae": []} for m in folders}
        for stem in stems:
            t = target_darkness(source / f"{stem}.png", res_px, args.device)
            tn = t.cpu().numpy()[0, 0]
            for m, f in folders.items():
                pts = torch.as_tensor(np.load(f / f"{stem}.npy"), dtype=torch.float32, device=args.device)[None]
                with torch.no_grad():
                    d = render_darkness(pts, res_px, sigma, gain).cpu().numpy()[0, 0]
                scores[m]["psnr"].append(psnr(d, tn))
                scores[m]["ssim"].append(ssim(d, tn))
                scores[m]["mae"].append(float(np.abs(d - tn).mean()))
        summary[ds] = {"images": len(stems), "methods": {}}
        lines += [f"## {ds} ({len(stems)} images)", "", "| method | PSNR | SSIM | MAE |", "|---|---|---|---|"]
        for m in folders:
            row, cells = {}, [m]
            for k in ("psnr", "ssim", "mae"):
                v = np.array(scores[m][k])
                ref = np.array(scores[args.reference][k])
                p = float(wilcoxon(v, ref).pvalue) if m != args.reference and np.any(v != ref) else 1.0
                row[k] = {"mean": float(v.mean()), "std": float(v.std()), "p_vs_ref": p}
                cells.append(f"{v.mean():.4g} ({v.std():.2g}){'*' if p < 0.05 else ''}")
            summary[ds]["methods"][m] = row
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        print("\n".join(lines[-len(folders) - 5:]), flush=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
