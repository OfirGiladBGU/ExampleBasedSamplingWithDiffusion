"""Stage 1 (local, CPU): pack the model's real final results for the shots, from the
experiment outputs already on disk (experiments/outputs/, unzipped from the server).

Until stage_1_trajectories.py has run on the GPU there is no step-by-step reverse process
to animate. This stage gives every trajectory shot a truthful stand-in instead: the
rejection-sampling prior the model starts from (recomputed here with the same seed) and
the model's actual final stipple, which the scene crossfades between and labels as such.
When a trajectory file exists, the scenes use it instead, with no change to shots.json.

    video/assets/results/<name>.npz        image (H,W), prior (N,2), final (N,2), grid_size
    video/assets/results/tone_<stem>.npz   image, target, rho/dark/points per strategy, loss
                                           histories, psnr, and the renderer settings of the run

    python video/stage_1_results.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, load_shots  # noqa: E402

OUTPUT_DIR = str(RESULTS_DIR)
SMART_INIT_SEED = 42          # the seed sample_control.py uses for the prior
RENDER_RES = 256              # tone experiment render resolution
TONE_CONFIGS = ["none", "curve", "random", "field"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--smart-init-seed", type=int, default=SMART_INIT_SEED)
    return p.parse_args()


def rejection_prior(image_01, n_points, seed):
    """Same algorithm and RNG stream as control_v4.smart_init.generate_smart_init_points_from_density
    (copied because that module imports torch), so this is the prior the model starts from."""
    prob = np.clip(1.0 - np.asarray(image_01, np.float32), 0.0, 1.0)
    h, w = prob.shape
    rng = np.random.RandomState(seed)
    if prob.sum() <= 0.0:
        return np.stack([rng.uniform(0, 1, n_points), rng.uniform(0, 1, n_points)], 1).astype(np.float32)
    points, batch = [], max(4096, n_points * 8)
    while len(points) < n_points:
        xs, ys, ps = rng.uniform(0.0, w, batch), rng.uniform(0.0, h, batch), rng.uniform(0.0, 1.0, batch)
        xi = np.clip(xs.astype(np.int32), 0, w - 1)
        yi = np.clip(ys.astype(np.int32), 0, h - 1)
        keep = ps < prob[yi, xi]
        for x, y in zip(xs[keep] / max(w, 1), ys[keep] / max(h, 1)):
            points.append([x, y])
            if len(points) >= n_points:
                break
    return np.asarray(points, np.float32)


def load_gray(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f"cannot read {path}")
    return g.astype(np.float32) / 255.0


def psnr(a, b):
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def main():
    a = parse_args()
    data = load_shots()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    for name, spec in data["trajectories"].items():
        if "result" not in spec:
            print(f"{name}: no 'result' path in shots.json, skipped")
            continue
        image = load_gray(spec["image"])
        final = np.load(spec["result"]).astype(np.float32)
        n = int(spec["grid_size"]) ** 2
        if final.shape != (n, 2):
            raise SystemExit(f"{name}: {spec['result']} has shape {final.shape}, expected ({n}, 2)")
        prior = rejection_prior(image, n, a.smart_init_seed)
        np.savez_compressed(out / f"{name}.npz", image=image, prior=prior, final=final,
                            grid_size=np.int32(spec["grid_size"]))
        print(f"{name}: {n} points -> {out / (name + '.npz')}")

    for shot in data["shots"]:
        v = shot["visual"]
        if v["type"] != "tone" or "result" not in v:
            continue
        d = np.load(v["result"])
        image = d["image_01"].astype(np.float32)
        target = cv2.resize(1.0 - image, (RENDER_RES, RENDER_RES), interpolation=cv2.INTER_AREA)
        cfg = json.loads((Path(v["result"]).parent / "summary.json").read_text())["config"]
        pack = {"image": image, "target": target, "ink_gain": np.float32(cfg["ink_gain"]),
                "dot_sigma_px": np.float32(cfg["dot_sigma_px"]), "render_res": np.int32(cfg["render_res"])}
        for c in TONE_CONFIGS:
            pack[f"{c}_coords"] = d[f"{c}_coords"].astype(np.float32)
            pack[f"{c}_rho"] = d[f"{c}_rho"].astype(np.float32)
            pack[f"{c}_dark"] = d[f"{c}_darkness"].astype(np.float32)
            pack[f"{c}_loss"] = d[f"{c}_loss"].astype(np.float32)
            pack[f"{c}_psnr"] = np.float32(psnr(d[f"{c}_darkness"], target))
        dst = out / f"{v['data']}.npz"
        np.savez_compressed(dst, **pack)
        print(f"{v['data']}: PSNR {pack['none_psnr']:.2f} -> {pack['field_psnr']:.2f} dB -> {dst}")


if __name__ == "__main__":
    main()
