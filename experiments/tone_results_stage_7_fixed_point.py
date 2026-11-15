"""Stage 7: gradient-free tone correction with the sampler in the loop (rebuttal D1).

The question a reviewer can ask of the tone section: does it need a differentiable sampler at all?
Any sampler, including a per-density optimizer, can be corrected without gradients by the
classical multiplicative fixed point

    rho_{k+1} = clip( rho_k * ( blur(target) / blur(render(sampler(rho_k))) ) ** ALPHA , 0, 1 )

(the ratio is computed on the stage-1 render grid, clamped to [1/RATIO_CLAMP, RATIO_CLAMP] and
upsampled to the condition resolution). This stage runs it with
    gbn            the GBN optimizer itself (budget_teachers_driver.gbn_cmd, its own env)
    ddpm-500 ...   our real sampler (few_step_sampling_stage_1.py, Smart Init from the
                   requested density, fixed noise seed OPT_SEED for every iteration)
for ITERS updates on the 24 stage-1 targets, keeps each image's best iterate (by the in-loop
PSNR) and re-evaluates it on fresh randomness: a new GBN run, or our sampler on EVAL_SEEDS (the
seeds of stage 5, so the gains are paired with the stage-5 rows of the same images). GBN is
also run on the densities that stage 1 optimized through our DDIM-25 chain (TRANSFER_CONFIGS),
to see whether our corrections carry over to the optimizer.

All tone numbers use the stage-1 renderer and the original target, as in stages 1, 4 and 5.

Usage (one GPU job: GBN is built for sm_89, see budget_teachers_gbn.sbatch):
    python experiments/tone_results_stage_7_fixed_point.py
    python experiments/tone_results_stage_7_fixed_point.py --summarize
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# gbn_cmd moved to the E4 archive on 2026-11-01; the GBN command builder itself is unchanged
sys.path.insert(0, str(Path(__file__).resolve().parent / "archive" / "E4_budget_generalization_scripts"))
from budget_teachers_driver import gbn_cmd  # noqa: E402
from tone_results_utils import psnr, render_darkness, ssim  # noqa: E402

ROOT_DIR = "experiments/outputs/tone_results"
STAGE1_DIR = f"{ROOT_DIR}/main"
MANIFEST = f"{ROOT_DIR}/manifest.json"
OUT_DIR = f"{ROOT_DIR}/fixed_point"
STAGE5_DIR = f"{ROOT_DIR}/full_sampler_ddim25"
TRANSFER_CONFIGS = "curve,field"
SAMPLERS = "gbn,ddpm-500,ddim-25-eta1"
ITERS = 6
ALPHA = 0.5             # damped: with 1.0 GBN's mean PSNR peaked at iteration 2 and then oscillated
BLUR_PX = 4.0           # render-grid px; 1024 points on 256 px sit ~8 px apart
RATIO_CLAMP = 2.0
RATIO_EPS = 1e-3
OPT_SEED = 100          # our sampler's noise seed during the iteration (not one of EVAL_SEEDS)
EVAL_SEEDS = "0,1,2"
SMART_INIT_SEED_BASE = 42   # stage 5: --smart_init_seed 42 + seed
GRID_SIZE = 32
CONTROL_CKPT = "control_v4/train_outputs_Icons-50_1024_GBN_full/checkpoints/dynamic_ep5000.ckpt"
FEWSTEP_SCRIPT = "experiments/few_step_sampling_stage_1.py"
GBN_ENV = "sd"          # GBN runs in its own conda env (budget_teachers_gbn.sbatch)
LIMIT = -1
GBN_SUBDIR = "Icons-50/tone"
GBN_TIMEOUT_S = 300      # per image; GBN once hung for hours on one corrected density
GBN_RETRIES = 1
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Gradient-free tone correction with the sampler in the loop")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--stage1", default=STAGE1_DIR, help="Stage-1 run: targets and renderer settings")
    p.add_argument("--stage5", default=STAGE5_DIR, help="Stage-5 run: conditions to transfer, paired rows")
    p.add_argument("--output", default=OUT_DIR)
    p.add_argument("--samplers", default=SAMPLERS)
    p.add_argument("--transfer-configs", default=TRANSFER_CONFIGS)
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--blur-px", type=float, default=BLUR_PX)
    p.add_argument("--ratio-clamp", type=float, default=RATIO_CLAMP)
    p.add_argument("--opt-seed", type=int, default=OPT_SEED)
    p.add_argument("--eval-seeds", default=EVAL_SEEDS)
    p.add_argument("--grid_size", type=int, default=GRID_SIZE)
    p.add_argument("--control_ckpt_path", default=CONTROL_CKPT)
    p.add_argument("--gbn-env", default=GBN_ENV)
    p.add_argument("--gbn-timeout", type=float, default=GBN_TIMEOUT_S, help="Seconds per image before GBN is killed")
    p.add_argument("--gbn-retries", type=int, default=GBN_RETRIES)
    p.add_argument("--limit", type=int, default=LIMIT, help="Use only the first N targets (smoke test)")
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--summarize", action="store_true", help="Only rebuild summary.md from results.json")
    return p.parse_args()


# ── condition images and samplers ───────────────────────────────────────────

def write_conditions(rhos, folder):
    """rho (density) -> 8-bit intensity PNG, as stage 5 writes its conditions."""
    folder.mkdir(parents=True, exist_ok=True)
    for stem, rho in rhos.items():
        img = np.clip(np.rint((1.0 - rho) * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(folder / f"{stem}.png")


def run_gbn(a, rhos, work):
    """GBN, one process per image so that a hang costs one image, not the run.

    Returns ({stem: points or None}, seconds per image). None: GBN timed out or failed on every
    try; the caller keeps that image's previous density and leaves it out of the scores.
    """
    pts, t0 = {}, time.time()
    for stem, rho in rhos.items():
        d = work / stem
        for sub in ("original", "source"):
            write_conditions({stem: rho}, d / sub / GBN_SUBDIR)
        # activate GBN's env in a subshell; `conda run` would take GBN's own --n flag as its --name
        cmd = ["bash", "-c", 'source activate "$0" >/dev/null && exec python "$@"', a.gbn_env] + gbn_cmd(d, a.grid_size ** 2)[1:]
        out = d / f"target_GBN_{a.grid_size ** 2}" / GBN_SUBDIR / f"{stem}.npy"
        pts[stem] = None
        for attempt in range(1 + a.gbn_retries):
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, start_new_session=True)
            try:
                rc = proc.wait(timeout=a.gbn_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)       # the whole group: GBN's solver is a child process
                proc.wait()
                rc = None
            if rc == 0 and out.exists():
                pts[stem] = np.load(out)
                break
            print(f"    GBN {'timed out' if rc is None else f'exit {rc}'} on {stem} (try {attempt + 1})", flush=True)
    return pts, (time.time() - t0) / len(rhos)


def run_ours(a, rhos, work, sampler, seed):
    """Our sampler over all images (one process: the model is loaded once)."""
    write_conditions(rhos, work / "source")
    cmd = [sys.executable, FEWSTEP_SCRIPT, "--control_ckpt_path", a.control_ckpt_path, "--label", "fp",
           "--source", str(work / "source"), "--output", str(work), "--grid_size", str(a.grid_size),
           "--samplers", sampler, "--png-count", "0", "--no-throughput", "--noise_seed", str(seed),
           "--smart_init_seed", str(SMART_INIT_SEED_BASE + seed)]
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    sec = (time.time() - t0) / len(rhos)
    out = work / f"fp_{a.grid_size * a.grid_size}" / sampler
    return {s: np.load(out / f"{s}.npy") for s in rhos}, sec


def run_sampler(a, rhos, work, sampler, seed):
    return run_gbn(a, rhos, work) if sampler == "gbn" else run_ours(a, rhos, work, sampler, seed)


# ── rendering and the update ────────────────────────────────────────────────

def render(pts, cfg, device):
    c = torch.as_tensor(pts, dtype=torch.float32, device=device)[None]
    with torch.no_grad():
        return render_darkness(c, cfg["render_res"], cfg["dot_sigma_px"], cfg["ink_gain"]).cpu().numpy()[0, 0]


def scores(d, t):
    return {"psnr": psnr(d, t), "ssim": ssim(d, t), "mae": float(np.abs(d - t).mean())}


def update(rho, dark, target, a):
    """One multiplicative step; the ratio lives on the render grid, rho on the condition grid."""
    r = (gaussian_filter(target, a.blur_px) + RATIO_EPS) / (gaussian_filter(dark, a.blur_px) + RATIO_EPS)
    r = np.clip(r, 1.0 / a.ratio_clamp, a.ratio_clamp) ** a.alpha
    r = torch.nn.functional.interpolate(torch.as_tensor(r, dtype=torch.float32)[None, None],
                                        size=rho.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    return np.clip(rho * r, 0.0, 1.0)


# ── the experiment ──────────────────────────────────────────────────────────

def run(a):
    cfg = json.loads((Path(a.stage1) / "summary.json").read_text())["config"]
    stems = [e["stem"] for e in json.loads(Path(a.manifest).read_text())["images"]]
    stems = stems if a.limit < 0 else stems[:a.limit]
    out = Path(a.output)
    rho0, target = {}, {}
    for s in stems:
        img = np.load(Path(a.stage1) / f"{s}.npz")["image_01"].astype(np.float32)
        rho0[s] = 1.0 - img
        t = torch.as_tensor(rho0[s])[None, None]
        target[s] = torch.nn.functional.interpolate(t, size=(cfg["render_res"],) * 2, mode="area").numpy()[0, 0]
    res = {"config": {k: v for k, v in vars(a).items()}, "renderer": {k: cfg[k] for k in
           ("render_res", "dot_sigma_px", "ink_gain")}, "samplers": {}}
    for sampler in a.samplers.split(","):
        r = res["samplers"][sampler] = {"history": {s: [] for s in stems}, "seconds": []}
        rho, best = dict(rho0), {s: (-np.inf, None) for s in stems}
        for k in range(a.iters + 1):
            pts, sec = run_sampler(a, rho, out / sampler / f"iter{k}", sampler, a.opt_seed)
            r["seconds"].append(sec)
            for s in stems:
                if pts[s] is None:                          # GBN failed: keep rho, skip this iterate
                    r["history"][s].append(None)
                    continue
                d = render(pts[s], cfg, a.device)
                sc = scores(d, target[s])
                r["history"][s].append(sc)
                if sc["psnr"] > best[s][0]:
                    best[s] = (sc["psnr"], rho[s].copy(), k)
                if k < a.iters:
                    rho[s] = update(rho[s], d, target[s], a)
            ok = [r["history"][s][-1]["psnr"] for s in stems if r["history"][s][-1] is not None]
            print(f"[{sampler}] iter {k}: mean PSNR {np.mean(ok):.2f} over {len(ok)} images"
                  f"  ({sec:.1f} s/image)", flush=True)
        r["best_iter"] = {s: best[s][2] for s in stems if best[s][1] is not None}
        best_rho = {s: best[s][1] for s in stems if best[s][1] is not None}
        write_conditions(best_rho, out / sampler / "best_conditions")
        # fresh evaluation of the selected densities
        r["fresh"] = {}
        for seed in ([None] if sampler == "gbn" else [int(x) for x in a.eval_seeds.split(",")]):
            name = "fresh" if seed is None else f"fresh-seed{seed}"
            pts, _ = run_sampler(a, best_rho, out / sampler / name, sampler, seed if seed is not None else 0)
            r["fresh"][name] = {s: scores(render(p, cfg, a.device), target[s]) for s, p in pts.items() if p is not None}
        if sampler == "gbn":
            r["transfer"] = {}
            for c in a.transfer_configs.split(","):
                cond = Path(a.stage5) / "conditions" / c
                rhos = {s: 1.0 - np.asarray(Image.open(cond / f"{s}.png"), np.float32) / 255.0 for s in stems}
                pts, _ = run_gbn(a, rhos, out / sampler / f"transfer-{c}")
                r["transfer"][c] = {s: scores(render(p, cfg, a.device), target[s]) for s, p in pts.items() if p is not None}
        (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    return res


# ── summary ────────────────────────────────────────────────────────────────

def stage5_psnr(a, cfg, stems, config, sampler, device):
    """Per-image PSNR (mean over the stage-5 seeds) of a stage-5 row, scored like everything here."""
    out = {}
    for s in stems:
        img = np.load(Path(a.stage1) / f"{s}.npz")["image_01"].astype(np.float32)
        t = torch.nn.functional.interpolate(torch.as_tensor(1.0 - img)[None, None],
                                            size=(cfg["render_res"],) * 2, mode="area").numpy()[0, 0]
        v = [psnr(render(np.load(f), cfg, device), t) for seed in a.eval_seeds.split(",")
             for f in [Path(a.stage5) / "samples" / f"{config}-seed{seed}_{a.grid_size ** 2}" / sampler / f"{s}.npy"]
             if f.exists()]
        if v:
            out[s] = float(np.mean(v))
    return out


def summarize(a):
    out = Path(a.output)
    res = json.loads((out / "results.json").read_text())
    cfg = json.loads((Path(a.stage1) / "summary.json").read_text())["config"]
    lines = ["# Gradient-free tone correction with the sampler in the loop (D1)", "",
             f"Multiplicative fixed point, {a.iters} updates, alpha {a.alpha:g}, blur {a.blur_px:g} px, ratio clamp "
             f"{a.ratio_clamp:g}; best iterate per image, then re-evaluated on fresh randomness. PSNR against the "
             "original target with the stage-1 renderer; gains are paired over images against the sampler's "
             "own uncorrected output.", "",
             "| sampler | correction | sampler calls / image | PSNR | SSIM | MAE | PSNR gain vs none (images improved) |",
             "|---|---|---|---|---|---|---|"]

    def row(sampler, name, calls, per, base):
        m = {k: float(np.mean([v[k] for v in per.values()])) for k in ("psnr", "ssim", "mae")}
        gain = ""
        if base is not None:
            st = sorted(set(base) & set(per))
            g = np.array([per[s]["psnr"] - base[s] for s in st])
            p = float(wilcoxon(g).pvalue) if np.any(g != 0) else 1.0
            gain = f"{g.mean():+.2f} dB ({int((g > 0).sum())}/{len(g)}, p={p:.1g})"
        lines.append(f"| {sampler} | {name} | {calls} | {m['psnr']:.2f} | {m['ssim']:.4f} | {m['mae']:.4f} | {gain} |")

    for sampler, r in res["samplers"].items():
        stems = list(r["history"])
        it0 = {s: r["history"][s][0] for s in stems if r["history"][s][0] is not None}
        loop = {s: max((x for x in r["history"][s] if x is not None), key=lambda x: x["psnr"])
                for s in stems if any(x is not None for x in r["history"][s])}
        calls = len(next(iter(r["history"].values())))
        if sampler == "gbn":
            base = {s: it0[s]["psnr"] for s in stems}
            row(sampler, "none (iteration 0)", 1, it0, None)
            row(sampler, "fixed point, in-loop best", calls, loop, base)
            row(sampler, "fixed point, fresh run", calls + 1, r["fresh"]["fresh"], base)
            for c, per in r.get("transfer", {}).items():
                row(sampler, f"our {c} rho* (DDIM-25 gradient)", 1, per, base)
        else:
            fresh = {s: {k: float(np.mean([r["fresh"][n][s][k] for n in r["fresh"]])) for k in ("psnr", "ssim", "mae")}
                     for s in stems if all(s in r["fresh"][n] for n in r["fresh"])}
            base = stage5_psnr(a, cfg, stems, "none", sampler, a.device)
            s5 = json.loads((Path(a.stage5) / "summary.json").read_text()).get(sampler, {})
            if "none" in s5:
                lines.append(f"| {sampler} | none (stage 5, seeds {a.eval_seeds}) | 1 | {s5['none']['psnr']:.2f} | "
                             f"{s5['none']['ssim']:.4f} | {s5['none']['mae']:.4f} |  |")
            row(sampler, f"fixed point, in-loop best (seed {a.opt_seed})", calls, loop, None)
            row(sampler, f"fixed point, fresh seeds {a.eval_seeds}", calls + 1, fresh, base)
            for c in ("curve", "field"):
                if c in s5:
                    lines.append(f"| {sampler} | {c}, DDIM-25 gradient (stage 5) | 1 | {s5[c]['psnr']:.2f} | "
                                 f"{s5[c]['ssim']:.4f} | {s5[c]['mae']:.4f} | {s5[c]['gain_db']:+.2f} dB "
                                 f"({s5[c]['improved']}/24, p={s5[c]['p']:.1g}) |")
        best = np.bincount(list(r["best_iter"].values()), minlength=calls)
        failed = sum(x is None for h in r["history"].values() for x in h)
        lines += [f"| {sampler} | (best iterate histogram {best.tolist()}; "
                  f"{np.mean(r['seconds']):.1f} s per image per call; {failed} failed sampler calls) | | | | | |"]
    lines += ["", "Calls per image counts sampler evaluations, including the fresh re-evaluation. The gradient "
                  "rows ran 300 optimizer steps through the DDIM-25 chain (see the D3 cost table)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    a = parse_args()
    if not a.summarize:
        run(a)
    summarize(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
