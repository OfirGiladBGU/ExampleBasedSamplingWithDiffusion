"""run_m4_overfit.py -- M4 control-branch overfit test (control_v4 diffusion).

Freeze the trained stippling ControlNet (base denoiser + ep10000 density branch)
and train ONLY a new parallel anisotropy branch on FLAT-density teacher samples
whose only difference is the (theta,kappa) control field. Then sample at fixed
density with different control fields and check the output anisotropy tracks the
command, and that it survives grid-transfer.

DIAGNOSTIC LADDER (reported side by side, so a failure is attributable):
  teacher   -- anisotropy of the GT offset grids through the SAME measurement
               pipeline. This is the number the model should reproduce (~0.2 at
               kappa=2). If this is low, the teacher/round-trip is broken.
  init=teacher -- SDEdit sampling started from a NOISED TEACHER sample. Tests
               whether branch+model can EXPRESS the anisotropy at all.
  init=smart   -- SDEdit from the isotropic smart-init (the real generative
               case). If teacher-init works but smart-init doesn't, the limit is
               SDEdit anchoring, not the branch.

Run from the repo root, e.g.:
  python control_v4_pilot/run_m4_overfit.py --steps 8000 --out control_v4_pilot/m4_out
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.Config import ParseSampleConfig
from control_v4.DynamicControlNet import DynamicControlNet
from control_v4.smart_init import build_smart_init_from_image, add_noise_at_t

import aniso_control as ac
import m4_teacher
import aniso_pilot as ap


# ── defaults (edit here; overridable on the command line) ────────────
BASE_CONFIG = os.path.join(_ROOT, "config/GBN/config.json")
BASE_CKPT = os.path.join(_ROOT, "config/GBN/model.ckpt")
# Density branch trained with TRUNCATION_RATIO=1.0 -> valid at ALL timesteps.
EP_CKPT = os.path.join(
    _ROOT,
    "control_v4/train_outputs_icons50_512_full/checkpoints/dynamic_controlnet_v4_ep10000.pt",
)
# Alternative: the truncated (0.30) run. Only safe if train_trunc <= 0.30.
EP_CKPT_TRUNC030 = os.path.join(
    _ROOT,
    "control_v4/train_outputs_icons50_512_no_random/checkpoints/dynamic_controlnet_v4_ep10000.pt",
)
GRID_SIZE = 32
STEPS = 8000                 # gradient steps (was 400 "epochs" == 400 steps)
LR = 2e-4
LOG_EVERY = 250
N_PER_COND = 1
IMG_VALUE = 0.0            # BLACK canvas: dark = ink = points (icons convention).
                           # 0.5 mid-gray is ambiguous/OOD and underfills.
TRAIN_TRUNCATION_RATIO = 1.0    # t-range the aniso branch is TRAINED on
SAMPLE_TRUNCATION_RATIO = 1.0   # SDEdit start for INFERENCE only
EVAL_TIMESTEPS = 1000
N_SAMPLES = 1
INFER_GRIDS = "32,64"
DEVICE = "cuda"
DOTSIZE = 5.0
OUTPUT_DIR = os.path.join(_HERE, "m4_out")


def wrap_pi(a):
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def stack_teacher(samples, device):
    def st(key):
        return torch.from_numpy(np.stack([s[key] for s in samples])).float().to(device)
    return dict(offsets=st("offsets"), control=st("control"),
                density=st("target_density"), high_res=st("high_res"))


def measure(coords_np):
    g = ap.global_near_field_anisotropy(coords_np)
    nn = ap.nn_vector_anisotropy(coords_np, kk=1)
    return float(g["strength"]), float(g["axis"]), float(nn["ratio"])


def orient_error(axis_rad, theta_deg):
    return float(np.rad2deg(abs(wrap_pi(axis_rad - np.deg2rad(theta_deg)))))


def sample_dual(diffusion, dual, control_field, density, high_res, G,
                n_samples, eval_timesteps, trunc, device, seed,
                x_init_offsets=None):
    """SDEdit-truncated sampling. x_init_offsets: (2,G,G) tensor to start from;
    if None, use the isotropic smart-init built from the (flat) image."""
    dual.set_condition(high_res, density, control_field)
    orig_model = diffusion.model
    diffusion.model = dual
    diffusion.set_num_timesteps(eval_timesteps)
    diffusion.eval()
    try:
        if x_init_offsets is None:
            img2d = high_res[0, 0].detach().cpu().numpy()
            _, smart_off, _ = build_smart_init_from_image(
                img2d, grid_size=G, n_points=G * G, seed=seed)
            x_init = torch.from_numpy(smart_off).unsqueeze(0).float().to(device)
        else:
            x_init = x_init_offsets.unsqueeze(0).float().to(device)
        if x_init.shape[0] != n_samples:
            x_init = x_init.expand(n_samples, -1, -1, -1).contiguous()
        t_start = int(np.clip(int(diffusion.num_timesteps * trunc),
                              1, diffusion.num_timesteps - 1))
        alpha_t = diffusion.alphas_cumprod[t_start]
        img = add_noise_at_t(x_init, alpha_t)
        with torch.no_grad():
            for i in reversed(range(t_start)):
                t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
                img = diffusion.p_sample(img, cond=None, t=t_tensor,
                                         clip_denoised=diffusion.sample_clip)
    finally:
        diffusion.model = orig_model
        diffusion.reset_timesteps()
        diffusion.train()
    return img


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--base_config", default=BASE_CONFIG)
    pa.add_argument("--base_ckpt", default=BASE_CKPT)
    pa.add_argument("--ep_ckpt", default=EP_CKPT)
    pa.add_argument("--grid", type=int, default=GRID_SIZE)
    pa.add_argument("--steps", type=int, default=STEPS)
    pa.add_argument("--lr", type=float, default=LR)
    pa.add_argument("--log_every", type=int, default=LOG_EVERY)
    pa.add_argument("--n_per_cond", type=int, default=N_PER_COND)
    pa.add_argument("--img_value", type=float, default=IMG_VALUE)
    pa.add_argument("--train_trunc", type=float, default=TRAIN_TRUNCATION_RATIO,
                    help="train the aniso branch on t in [0, train_trunc*T)")
    pa.add_argument("--sample_trunc", type=float, default=SAMPLE_TRUNCATION_RATIO,
                    help="SDEdit start for sampling (inference only); 1.0 = from noise")
    pa.add_argument("--eval_timesteps", type=int, default=EVAL_TIMESTEPS)
    pa.add_argument("--n_samples", type=int, default=N_SAMPLES)
    pa.add_argument("--infer_grids", default=INFER_GRIDS)
    pa.add_argument("--device", default=DEVICE)
    pa.add_argument("--dotsize", type=float, default=DOTSIZE)
    pa.add_argument("--out", default=OUTPUT_DIR)
    pa.add_argument("--eval_only", action="store_true",
                    help="skip training; load aniso_overfit.pt from --out")
    pa.add_argument("--load_aniso", default="",
                    help="explicit path to an aniso checkpoint (default: <out>/aniso_overfit.pt)")
    args = pa.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    print("device:", device)

    # ── frozen base diffusion denoiser ───────────────────────────────
    diffusion = ParseSampleConfig(args.base_config, device=device)
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"], strict=False)
    diffusion.to(device)
    diffusion.eval()
    denoiser = ac.freeze(diffusion.model)
    T_train = diffusion.num_timesteps
    trunc_cutoff = max(1, int(T_train * args.train_trunc))
    print("base T=%d  train t in [0,%d)  sample_trunc=%.2f"
          % (T_train, trunc_cutoff, args.sample_trunc))

    # ── frozen density branch (ep10000: gecco on, adaptive gate) ─────
    density_control = DynamicControlNet(
        denoiser, grid_size=args.grid,
        enable_gecco=True, enable_adaptive_gate_injection=True,
        smart_init_features=False, sdf_features=False, batch_coords_features=False,
    ).to(device)
    density_control.safe_load_state_dict(torch.load(args.ep_ckpt, map_location="cpu"), strict=False)
    ac.freeze(density_control)
    print("loaded + froze density branch from", os.path.basename(args.ep_ckpt))

    # ── trainable anisotropy branch ──────────────────────────────────
    aniso = ac.AnisoControlNet(denoiser, grid_size=args.grid, include_density=True).to(device)
    aniso.train()
    print("aniso trainable params: %.2fM"
          % (sum(p.numel() for p in aniso.parameters() if p.requires_grad) / 1e6))

    # ── teacher ──────────────────────────────────────────────────────
    samples, conds = m4_teacher.generate(G=args.grid, n_per_cond=args.n_per_cond,
                                         img_value=args.img_value)
    tb = stack_teacher(samples, device)
    print("teacher: %d samples over %d conditions" % (len(samples), len(conds)))

    # teacher reference anisotropy (same measurement pipeline as the model)
    ref = {}
    for cond in conds:
        idx = next(i for i, s in enumerate(samples) if s["cond_name"] == cond["name"])
        c = ac.offsets_to_coords(tb["offsets"][idx:idx + 1])[0].detach().cpu().numpy()
        st, ax, nnr = measure(c)
        ref[cond["name"]] = dict(strength=st,
                                 orient_err=(orient_error(ax, cond["theta_deg"])
                                             if cond["kappa"] > 1.0 else None),
                                 nnratio=nnr)
        print("  teacher %-10s strength=%.3f orient_e=%s"
              % (cond["name"], st,
                 ("%.1f" % ref[cond["name"]]["orient_err"])
                 if ref[cond["name"]]["orient_err"] is not None else "-"))

    # ── train only the aniso branch ──────────────────────────────────
    opt = torch.optim.AdamW(aniso.parameters(), lr=args.lr)
    S = tb["offsets"].shape[0]
    loss_hist = []
    gnorm_hist = []
    n_steps = 0 if args.eval_only else args.steps
    if args.eval_only:
        ck = args.load_aniso or os.path.join(args.out, "aniso_overfit.pt")
        aniso.load_state_dict(torch.load(ck, map_location="cpu")["aniso_control"])
        aniso.to(device)
        loss_hist = [float("nan")]
        gnorm_hist = [float("nan")]
        print("eval_only: loaded aniso branch from", ck)
    for step in range(n_steps):
        t = torch.randint(0, trunc_cutoff, (S,), device=device)
        noise = torch.randn_like(tb["offsets"])
        offsets_t = diffusion.q_sample(tb["offsets"], t, noise)
        ctrl_d = density_control(offsets_t, t, tb["high_res"], tb["density"])
        ctrl_a = aniso(offsets_t, t, tb["control"], target_density_map=tb["density"])
        controls = ac.sum_controls(ctrl_d, ctrl_a)
        noise_pred = denoiser(offsets_t, t, controls=controls)
        loss = F.mse_loss(noise_pred, noise)
        opt.zero_grad()
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(aniso.parameters(), 1.0))
        opt.step()
        loss_hist.append(float(loss.item()))
        gnorm_hist.append(gnorm)
        if (step + 1) % args.log_every == 0:
            recent = float(np.mean(loss_hist[-args.log_every:]))
            grecent = float(np.mean(gnorm_hist[-args.log_every:]))
            print("  step %6d/%d  loss(avg %d) %.6f  grad_norm %.3e"
                  % (step + 1, args.steps, args.log_every, recent, grecent))

    if not args.eval_only:
        torch.save({"aniso_control": aniso.state_dict(), "steps": args.steps,
                    "conditions": conds}, os.path.join(args.out, "aniso_overfit.pt"))
    aniso.eval()

    # ── branch-contribution probe: is the aniso branch doing anything? ──
    with torch.no_grad():
        t_probe = torch.full((tb["offsets"].shape[0],), trunc_cutoff // 2,
                             dtype=torch.int64, device=device)
        n_probe = torch.randn_like(tb["offsets"])
        x_probe = diffusion.q_sample(tb["offsets"], t_probe, n_probe)
        cd = density_control(x_probe, t_probe, tb["high_res"], tb["density"])
        ca = aniso(x_probe, t_probe, tb["control"], target_density_map=tb["density"])
        czero = aniso(x_probe, t_probe, torch.zeros_like(tb["control"]),
                      target_density_map=tb["density"])

        def mag(ctrl):
            enc, mid = ctrl
            vals = [mid.abs().mean().item()]
            vals += [b.abs().mean().item() for lvl in enc for b in lvl]
            return float(np.mean(vals))

        def flat(ctrl):
            enc, mid = ctrl
            parts = [mid.reshape(-1)] + [b.reshape(-1) for lvl in enc for b in lvl]
            return torch.cat(parts)

        contrib = dict(density=mag(cd), aniso=mag(ca), aniso_zeroctrl=mag(czero))
        contrib["ratio_aniso_over_density"] = contrib["aniso"] / max(contrib["density"], 1e-12)
        contrib["control_sensitivity"] = abs(contrib["aniso"] - contrib["aniso_zeroctrl"]) \
            / max(contrib["aniso"], 1e-12)
        fa, fz = flat(ca), flat(czero)
        contrib["control_sensitivity_l2"] = float(
            ((fa - fz).norm() / torch.clamp(fa.norm(), min=1e-12)).item())
    print("branch contribution: density=%.3e aniso=%.3e ratio=%.3f sens(mag)=%.3f sens(L2)=%.3f"
          % (contrib["density"], contrib["aniso"], contrib["ratio_aniso_over_density"],
             contrib["control_sensitivity"], contrib["control_sensitivity_l2"]))

    # ── validate ─────────────────────────────────────────────────────
    dual = ac.DualControlledDenoiser(denoiser, density_control, aniso).to(device)
    infer_grids = [int(g) for g in args.infer_grids.split(",")]
    rows = []
    panels = []          # (label, cond, coords, strength, orient_err)
    for G in infer_grids:
        density = torch.full((1, 1, G, G), args.img_value, device=device)
        high_res = torch.full((1, 1, 512, 512), args.img_value, device=device)
        inits = ["smart"] + (["teacher", "ctrl0", "teacher_ctrl0"]
                             if G == args.grid else [])
        for init_mode in inits:
            for cond in conds:
                cmap = torch.from_numpy(
                    m4_teacher.control_map(cond["theta_deg"], cond["kappa"], G)
                ).unsqueeze(0).float().to(device)
                if init_mode in ("ctrl0", "teacher_ctrl0"):
                    cmap = torch.zeros_like(cmap)
                x_init = None
                if init_mode in ("teacher", "teacher_ctrl0"):
                    idx = next(i for i, s in enumerate(samples) if s["cond_name"] == cond["name"])
                    x_init = tb["offsets"][idx]
                raw = sample_dual(diffusion, dual, cmap, density, high_res, G,
                                  args.n_samples, args.eval_timesteps, args.sample_trunc,
                                  device, seed=42, x_init_offsets=x_init)
                coords = ac.offsets_to_coords(raw)[0].detach().cpu().numpy()
                st, ax, nnr = measure(coords)
                oe = orient_error(ax, cond["theta_deg"]) if cond["kappa"] > 1.0 else None
                rows.append(dict(grid=G, init=init_mode, cond=cond["name"],
                                 theta_deg=cond["theta_deg"], kappa=cond["kappa"],
                                 strength=st, nnratio=nnr, orient_err=oe))
                panels.append(("G%d/%s" % (G, init_mode), cond, coords, st, oe))
                print("  G=%d init=%-7s %-10s strength=%.3f nnratio=%.2f orient_e=%s"
                      % (G, init_mode, cond["name"], st, nnr,
                         ("%.1f" % oe) if oe is not None else "-"))

    _panels(panels, conds, args, os.path.join(args.out, "panels.png"))

    # ── report ───────────────────────────────────────────────────────
    L = []
    L.append("M4 overfit test -- anisotropy control branch on frozen control_v4")
    L.append("=" * 66)
    L.append("steps=%d lr=%g n_per_cond=%d img_value=%.2f eval_T=%d"
             % (args.steps, args.lr, args.n_per_cond, args.img_value, args.eval_timesteps))
    L.append("train_trunc=%.2f (aniso branch t-range)  sample_trunc=%.2f (inference only)"
             % (args.train_trunc, args.sample_trunc))
    L.append("density branch: %s" % os.path.basename(os.path.dirname(os.path.dirname(args.ep_ckpt))))
    L.append("final loss (avg last %d): %.6f  (start %.6f)"
             % (args.log_every, float(np.mean(loss_hist[-args.log_every:])),
                float(np.mean(loss_hist[:args.log_every]))))
    L.append("final grad_norm (avg last %d): %.3e" % (args.log_every, float(np.mean(gnorm_hist[-args.log_every:]))))
    L.append("branch contribution: density=%.3e aniso=%.3e ratio=%.3f sens(mag)=%.3f sens(L2)=%.3f"
             % (contrib["density"], contrib["aniso"], contrib["ratio_aniso_over_density"],
                contrib["control_sensitivity"], contrib["control_sensitivity_l2"]))
    L.append("  (ratio ~0 => aniso branch injects nothing; control_sensitivity ~0 =>")
    L.append("   branch output ignores the control field)")
    L.append("FLAT density everywhere: only the control field differs across conditions.")
    L.append("(convention: NN concentration axis = theta)")
    L.append("")
    L.append("TEACHER reference (what the model should reproduce):")
    L.append("%-12s %9s %8s" % ("cond", "strength", "orient_e"))
    for cond in conds:
        r = ref[cond["name"]]
        L.append("%-12s %9.3f %8s" % (
            cond["name"], r["strength"],
            ("%.1f" % r["orient_err"]) if r["orient_err"] is not None else "-"))
    L.append("")
    L.append("MODEL samples:")
    L.append("%-5s %-8s %-12s %6s %6s %9s %8s" % (
        "grid", "init", "cond", "kappa", "theta", "strength", "orient_e"))
    for r in rows:
        L.append("%-5d %-8s %-12s %6.2f %6.0f %9.3f %8s" % (
            r["grid"], r["init"], r["cond"], r["kappa"], r["theta_deg"], r["strength"],
            ("%.1f" % r["orient_err"]) if r["orient_err"] is not None else "-"))
    L.append("")
    for G in infer_grids:
        for init_mode in sorted(set(r["init"] for r in rows if r["grid"] == G)):
            gr = [r for r in rows if r["grid"] == G and r["init"] == init_mode]
            base = next(r for r in gr if r["kappa"] == 1.0)
            an = [r for r in gr if r["kappa"] > 1.0]
            steer = all(r["strength"] > base["strength"] + 0.03 for r in an)
            orient = all((r["orient_err"] is not None and r["orient_err"] < 20.0) for r in an)
            L.append("G=%d init=%-7s: kappa=1 str=%.3f ; steering: %s ; orientation<20deg: %s"
                     % (G, init_mode, base["strength"], steer, orient))
    L.append("")
    L.append("DECISIVE ABLATION (teacher vs teacher_ctrl0, same init, control on/off):")
    for cond in conds:
        a = next((r for r in rows if r["init"] == "teacher" and r["cond"] == cond["name"]), None)
        b = next((r for r in rows if r["init"] == "teacher_ctrl0" and r["cond"] == cond["name"]), None)
        if a and b:
            L.append("  %-12s control ON str=%.3f (oe %s) | control OFF str=%.3f (oe %s)"
                     % (cond["name"], a["strength"],
                        ("%.1f" % a["orient_err"]) if a["orient_err"] is not None else "-",
                        b["strength"],
                        ("%.1f" % b["orient_err"]) if b["orient_err"] is not None else "-"))
    L.append("  ON >> OFF  -> the CONTROL FIELD is steering (real result).")
    L.append("  ON ~= OFF  -> anisotropy came from the teacher INIT, not the control.")
    L.append("")
    L.append("HOW TO READ THIS:")
    L.append(" * model strength ~ teacher strength  -> branch learned + expresses it.")
    L.append(" * teacher-init works but smart-init does not -> SDEdit anchoring is the")
    L.append("   limit (lower --trunc, or init from an anisotropic seed), NOT the branch.")
    L.append(" * both near the kappa=1 floor while loss is still falling -> undertrained;")
    L.append("   raise --steps.")
    report = "\n".join(L)
    with open(os.path.join(args.out, "objective_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(args=vars(args), teacher_ref=ref, rows=rows, contrib=contrib,
                       loss_final=float(np.mean(loss_hist[-args.log_every:])),
                       loss_hist=loss_hist[::max(1, len(loss_hist) // 200)]), f, indent=2)


def _panels(panels, conds, args, path):
    labels = []
    for lab, *_ in panels:
        if lab not in labels:
            labels.append(lab)
    ncols = len(conds)
    nrows = len(labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows))
    axes = np.atleast_2d(axes)
    for lab, cond, coords, st, oe in panels:
        ri = labels.index(lab)
        ci = [c["name"] for c in conds].index(cond["name"])
        ax = axes[ri, ci]
        ax.scatter(coords[:, 0], coords[:, 1], s=args.dotsize, c="black", edgecolors="none")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("%s %s\nstr %.2f%s" % (lab, cond["name"], st,
                                            ("  oe %.0f" % oe) if oe is not None else ""),
                     fontsize=8)
    fig.suptitle("M4 overfit: flat density, control field varied (dots only)", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
