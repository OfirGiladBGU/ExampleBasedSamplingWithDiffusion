"""run_m4_generalize.py -- does the anisotropy rule transfer to UNSEEN images?

Train the anisotropy branch ONLY on a few basic primitives (flat/disk/square/
stripe) with CONTINUOUS (theta,kappa), then evaluate on held-out REAL ICONS the
branch has never seen. Base denoiser and density branch stay frozen throughout.

Evaluation uses the UPGRADE framing (this is also the product story):
  1. baseline : frozen model's own stipple of the icon, control OFF  -> isotropic
  2. upgrade  : re-noise that baseline at --upgrade_trunc, denoise with control ON
  3. control  : identical, but control OFF (isolates the branch's effect)
Anisotropy is measured overall AND on INTERIOR points only (away from ink
boundaries), because thin strokes physically constrain local spacing.

  python control_v4_pilot/run_m4_generalize.py --steps 40000
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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.Config import ParseSampleConfig
from control_v4.DynamicControlNet import DynamicControlNet
from control_v4.smart_init import build_smart_init_from_image, add_noise_at_t

import aniso_control as ac
import aniso_pilot as ap
import aniso_density as ad
import aniso_m2 as m2
import m4_teacher


# ── defaults ─────────────────────────────────────────────────────────
BASE_CONFIG = os.path.join(_ROOT, "config/GBN/config.json")
BASE_CKPT = os.path.join(_ROOT, "config/GBN/model.ckpt")
EP_CKPT = os.path.join(
    _ROOT,
    "control_v4/train_outputs_icons50_512_full/checkpoints/dynamic_controlnet_v4_ep10000.pt",
)
DATASET = os.path.join(_HERE, "primitives_ds.npz")
ICONS_DIR = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512/source"
GRID_SIZE = 32
HIMG = 512
STEPS = 40000
BATCH = 8
LR = 2e-4
LOG_EVERY = 500
TRAIN_TRUNCATION_RATIO = 1.0     # branch trained over full t (full density ckpt)
BASE_TRUNCATION_RATIO = 0.30     # producing the baseline isotropic stipple
UPGRADE_TRUNCATION_RATIO = 0.30  # re-noise level for the upgrade pass
EVAL_TIMESTEPS = 1000
N_TEST_ICONS = 4
INTERIOR_PX = 3.0                # absolute floor (px) for "interior"
INTERIOR_PCTL = 50               # interior = deeper than this pct of the icon ink depth
DEVICE = "cuda"
DOTSIZE = 4.0
OUTPUT_DIR = os.path.join(_HERE, "m4_gen_out")

EVAL_CONDS = [dict(name="k2.0_t00", theta_deg=0.0, kappa=2.0),
              dict(name="k2.0_t90", theta_deg=90.0, kappa=2.0),
              dict(name="k1.6_t45", theta_deg=45.0, kappa=1.6)]


def wrap_pi(a):
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def measure(coords):
    if len(coords) < 40:
        return float("nan"), float("nan")
    g = ap.global_near_field_anisotropy(coords)
    return float(g["strength"]), float(g["axis"])


def orient_err(axis, theta_deg):
    if axis != axis:
        return float("nan")
    return float(np.rad2deg(abs(wrap_pi(axis - np.deg2rad(theta_deg)))))


def load_icons(icons_dir, n, himg, grid, rng):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = []
    for root, _, fs in os.walk(icons_dir):
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    if not files:
        raise SystemExit("no icons found in %s" % icons_dir)
    pick = [files[i] for i in rng.choice(len(files), size=min(n, len(files)), replace=False)]
    out = []
    for p in pick:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (himg, himg), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        dens = img.reshape(grid, himg // grid, grid, himg // grid).mean(axis=(1, 3))
        dist = distance_transform_edt(img < 0.5)          # px inside ink
        out.append(dict(name=os.path.basename(p), img=img,
                        dens=dens.astype(np.float32), dist=dist))
    return out


def build_aniso_init(img, theta_deg, kappa_init, G, device, seed=0):
    """Weak classical anisotropic seed for the upgrade: sample the icon's own ink
    density at LOW kappa and the commanded theta, map to an offset grid. The model
    then STRENGTHENS this toward the target kappa -- the teacher-init regime that is
    the only thing shown to break the ~0.03 from-isotropic magnitude ceiling.
    Returns None if the sampler could not place exactly N points (caller falls back)."""
    rho = ad.ink_density(np.asarray(img, dtype=np.float32))
    N = G * G
    th_fn, ka_fn, kmax = m2.const_field(float(theta_deg), float(kappa_init))
    rng = np.random.default_rng(1234 + seed)
    P, r0, att = ad.sample_exact_n_density(rho, th_fn, ka_fn, N, rng,
                                           r0_init=0.85, kappa_max=kmax, progress=False)
    if len(P) != N:
        return None
    off = m4_teacher.points_to_offset_grid(P, G)          # (2,G,G)
    return torch.from_numpy(off).unsqueeze(0).float().to(device)


def icons_from_arrays(imgs, dens, names):
    """Build eval icon dicts from stored held-out arrays (mirror of load_icons)."""
    out = []
    for k in range(len(imgs)):
        img = np.asarray(imgs[k], dtype=np.float32)
        out.append(dict(name=str(names[k]), img=img,
                        dens=np.asarray(dens[k], dtype=np.float32),
                        dist=distance_transform_edt(img < 0.5)))
    return out


def sample_from(diffusion, dual, control, density, high_res, G, trunc,
                eval_timesteps, device, x_init=None, seed=0):
    """SDEdit-style: start from x_init (or smart-init), noise at trunc, denoise."""
    dual.set_condition(high_res, density, control)
    orig = diffusion.model
    diffusion.model = dual
    diffusion.set_num_timesteps(eval_timesteps)
    diffusion.eval()
    try:
        if x_init is None:
            img2d = high_res[0, 0].detach().cpu().numpy()
            _, smart_off, _ = build_smart_init_from_image(
                img2d, grid_size=G, n_points=G * G, seed=seed)
            xi = torch.from_numpy(smart_off).unsqueeze(0).float().to(device)
        else:
            xi = x_init.clone().to(device)
        t_start = int(np.clip(int(diffusion.num_timesteps * trunc), 1,
                              diffusion.num_timesteps - 1))
        img = add_noise_at_t(xi, diffusion.alphas_cumprod[t_start])
        with torch.no_grad():
            for i in reversed(range(t_start)):
                tt = torch.full((xi.shape[0],), i, dtype=torch.int64, device=device)
                img = diffusion.p_sample(img, cond=None, t=tt,
                                         clip_denoised=diffusion.sample_clip)
    finally:
        diffusion.model = orig
        diffusion.reset_timesteps()
        diffusion.train()
    return img


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--base_config", default=BASE_CONFIG)
    pa.add_argument("--base_ckpt", default=BASE_CKPT)
    pa.add_argument("--ep_ckpt", default=EP_CKPT)
    pa.add_argument("--dataset", default=DATASET)
    pa.add_argument("--icons_dir", default=ICONS_DIR)
    pa.add_argument("--grid", type=int, default=GRID_SIZE)
    pa.add_argument("--himg", type=int, default=HIMG)
    pa.add_argument("--steps", type=int, default=STEPS)
    pa.add_argument("--batch", type=int, default=BATCH)
    pa.add_argument("--lr", type=float, default=LR)
    pa.add_argument("--log_every", type=int, default=LOG_EVERY)
    pa.add_argument("--train_trunc", type=float, default=TRAIN_TRUNCATION_RATIO)
    pa.add_argument("--base_trunc", type=float, default=BASE_TRUNCATION_RATIO)
    pa.add_argument("--upgrade_trunc", type=float, default=UPGRADE_TRUNCATION_RATIO)
    pa.add_argument("--eval_timesteps", type=int, default=EVAL_TIMESTEPS)
    pa.add_argument("--n_test_icons", type=int, default=N_TEST_ICONS)
    pa.add_argument("--interior_px", type=float, default=INTERIOR_PX)
    pa.add_argument("--interior_pctl", type=float, default=INTERIOR_PCTL)
    pa.add_argument("--device", default=DEVICE)
    pa.add_argument("--dotsize", type=float, default=DOTSIZE)
    pa.add_argument("--eval_only", action="store_true")
    pa.add_argument("--control_gain", type=float, default=1.0,
                    help="inference-time gain on the ANISO branch residual (1.0 = as "
                         "trained). Magnitude is the blocker: orientation error is a "
                         "monotone function of achieved strength, so amplifying the "
                         "residual is the cheapest lever. Density branch is untouched.")
    pa.add_argument("--eval_seed", type=int, default=0,
                    help="reseeds the re-noise realizations; vary it to measure the "
                         "seed variance of the transfer metrics (10 held-out icons at "
                         "strength ~0.02 are noisy -- a single seed can mislead).")
    pa.add_argument("--init", choices=["iso", "aniso"], default="iso",
                    help="upgrade seed. 'iso' = frozen model's isotropic stipple (from-nothing "
                         "story, empirically capped ~0.03). 'aniso' = weak low-kappa classical "
                         "sample the model then strengthens (teacher-init regime).")
    pa.add_argument("--init_kappa", type=float, default=1.2,
                    help="kappa of the weak anisotropic seed when --init aniso; the branch must "
                         "amplify init_kappa -> the separately-commanded target kappa.")
    pa.add_argument("--eval_train_n", type=int, default=0,
                    help="DIAGNOSTIC: evaluate on the first N TRAINING icons (branch saw these) "
                         "instead of the held-out set. In-domain keep%% >> held-out -> data/"
                         "generalization gap; in-domain also weak -> the branch is weak everywhere.")
    pa.add_argument("--include_primitives", action="store_true",
                    help="ALSO evaluate on the training primitives (IN-DOMAIN control): "
                         "separates a generalization gap from a nucleation gap")
    pa.add_argument("--density", dest="include_density", action="store_true",
                    help="give the ANISO branch the density map. DEFAULT IS OFF: the density "
                         "ablation showed hiding density forces a content-independent local rule "
                         "and is what let orientation transfer to unseen icons. (The frozen "
                         "density branch always gets density regardless -- density stays a hard "
                         "condition; this flag only concerns the orientation branch.)")
    pa.add_argument("--load_aniso", default="")
    pa.add_argument("--out", default=OUTPUT_DIR)
    args = pa.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    G = args.grid

    # ── frozen stack ─────────────────────────────────────────────────
    diffusion = ParseSampleConfig(args.base_config, device=device)
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"],
                              strict=False)
    diffusion.to(device)
    diffusion.eval()
    denoiser = ac.freeze(diffusion.model)
    T_train = diffusion.num_timesteps
    trunc_cutoff = max(1, int(T_train * args.train_trunc))

    density_control = DynamicControlNet(
        denoiser, grid_size=G, enable_gecco=True, enable_adaptive_gate_injection=True,
        smart_init_features=False, sdf_features=False, batch_coords_features=False,
    ).to(device)
    density_control.safe_load_state_dict(torch.load(args.ep_ckpt, map_location="cpu"),
                                         strict=False)
    ac.freeze(density_control)

    include_density = args.include_density
    aniso = ac.AnisoControlNet(denoiser, grid_size=G,
                               include_density=include_density).to(device)
    print("AnisoControlNet include_density=%s%s"
          % (include_density, "" if include_density
             else "  [density hidden from aniso branch -- transfer-safe default]"))

    # ── primitives training set ──────────────────────────────────────
    ds = np.load(args.dataset, allow_pickle=True)
    off = torch.from_numpy(ds["offsets"]).float().to(device)
    ctl = torch.from_numpy(ds["control"]).float().to(device)
    pidx = torch.from_numpy(ds["prim_idx"]).long().to(device)
    prim_hr = torch.from_numpy(ds["prim_imgs"]).float().unsqueeze(1).to(device)
    prim_dn = torch.from_numpy(ds["prim_dens"]).float().unsqueeze(1).to(device)
    S = off.shape[0]
    print("train set: %d samples over %d primitives (%s)"
          % (S, prim_hr.shape[0], ", ".join(list(ds["prim_names"]))))

    # ── train (branch only) ──────────────────────────────────────────
    opt = torch.optim.AdamW(aniso.parameters(), lr=args.lr)
    loss_hist = []
    if args.eval_only:
        ck = args.load_aniso or os.path.join(args.out, "aniso_generalize.pt")
        aniso.load_state_dict(torch.load(ck, map_location="cpu")["aniso_control"])
        aniso.to(device)
        print("eval_only: loaded", ck)
    else:
        aniso.train()
        for step in range(args.steps):
            sel = torch.randint(0, S, (args.batch,), device=device)
            x0 = off[sel]
            c = ctl[sel]
            p = pidx[sel]
            hr = prim_hr[p]
            dn = prim_dn[p]
            t = torch.randint(0, trunc_cutoff, (args.batch,), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            cd = density_control(xt, t, hr, dn)
            ca = aniso(xt, t, c, target_density_map=dn)
            pred = denoiser(xt, t, controls=ac.sum_controls(cd, ca))
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aniso.parameters(), 1.0)
            opt.step()
            loss_hist.append(float(loss.item()))
            if (step + 1) % args.log_every == 0:
                print("  step %6d/%d  loss %.6f"
                      % (step + 1, args.steps, float(np.mean(loss_hist[-args.log_every:]))))
        torch.save({"aniso_control": aniso.state_dict(), "steps": args.steps},
                   os.path.join(args.out, "aniso_generalize.pt"))
    aniso.eval()

    # ── evaluate on UNSEEN icons ─────────────────────────────────────
    dual = ac.DualControlledDenoiser(denoiser, density_control, aniso).to(device)
    dual.aniso_gain = args.control_gain
    if args.control_gain != 1.0:
        print("aniso residual gain = %.2f" % args.control_gain)
    if args.eval_train_n > 0:
        n = min(args.eval_train_n, len(ds["prim_imgs"]))
        icons = icons_from_arrays(ds["prim_imgs"][:n], ds["prim_dens"][:n],
                                  ["TRAIN_" + str(x) for x in list(ds["prim_names"])[:n]])
        print("IN-DOMAIN eval: first %d TRAINING icons (branch saw these)" % n)
    elif "test_imgs" in getattr(ds, "files", []):
        icons = icons_from_arrays(ds["test_imgs"], ds["test_dens"],
                                  list(ds["test_names"]))
        print("held-out eval: %d test icons from dataset (disjoint from training)"
              % len(icons))
    else:
        icons = load_icons(args.icons_dir, args.n_test_icons, args.himg, G, rng)
    if args.include_primitives:
        # In-domain control: same upgrade protocol, on shapes the branch trained on.
        prim_np = ds["prim_imgs"]
        prim_nm = list(ds["prim_names"])
        extra = []
        for k, nm in enumerate(prim_nm):
            img = prim_np[k].astype(np.float32)
            dens = img.reshape(G, args.himg // G, G, args.himg // G).mean(axis=(1, 3))
            extra.append(dict(name="PRIM_" + str(nm), img=img,
                              dens=dens.astype(np.float32),
                              dist=distance_transform_edt(img < 0.5)))
        icons = extra + icons
        print("in-domain control: prepended %d primitives to the eval set" % len(extra))
    zero_ctl = torch.zeros((1, 3, G, G), device=device)
    torch.manual_seed(1000 + args.eval_seed)   # vary re-noise realizations across eval seeds
    np.random.seed(1000 + args.eval_seed)
    rows = []
    panels = []
    for ic in icons:
        hr = torch.from_numpy(ic["img"]).float()[None, None].to(device)
        dn = torch.from_numpy(ic["dens"]).float()[None, None].to(device)
        # Interior threshold adapts to THIS icon: a fixed pixel depth is
        # meaningless across icons with different stroke widths.
        dvals = ic["dist"][ic["dist"] > 0]
        thr = float(max(args.interior_px,
                        np.percentile(dvals, args.interior_pctl) if dvals.size else 0.0))
        print("  %-22s ink-depth p50=%.1fpx max=%.1fpx -> interior thr=%.1fpx"
              % (ic["name"][:22], float(np.median(dvals)) if dvals.size else 0.0,
                 float(dvals.max()) if dvals.size else 0.0, thr))
        base = sample_from(diffusion, dual, zero_ctl, dn, hr, G, args.base_trunc,
                           args.eval_timesteps, device, x_init=None, seed=42)
        base_xy = ac.offsets_to_coords(base)[0].detach().cpu().numpy()
        bs, _ = measure(base_xy)
        panels.append((ic["name"], "baseline", base_xy, bs, None))

        for cond in EVAL_CONDS:
            cmap = torch.from_numpy(
                m4_teacher.control_map(cond["theta_deg"], cond["kappa"], G)
            ).unsqueeze(0).float().to(device)
            if args.init == "aniso":
                ai = build_aniso_init(ic["img"], cond["theta_deg"], args.init_kappa,
                                      G, device, seed=args.eval_seed)
                x_init_up = ai if ai is not None else base
            else:
                x_init_up = base
            up_on = sample_from(diffusion, dual, cmap, dn, hr, G, args.upgrade_trunc,
                                args.eval_timesteps, device, x_init=x_init_up, seed=7)
            up_off = sample_from(diffusion, dual, zero_ctl, dn, hr, G, args.upgrade_trunc,
                                 args.eval_timesteps, device, x_init=x_init_up, seed=7)
            for tag, raw in (("ON", up_on), ("OFF", up_off)):
                xy = ac.offsets_to_coords(raw)[0].detach().cpu().numpy()
                st, ax = measure(xy)
                px = np.clip((xy * args.himg).astype(int), 0, args.himg - 1)
                inside = ic["dist"][px[:, 1], px[:, 0]] > thr
                sti, axi = measure(xy[inside])
                rows.append(dict(icon=ic["name"], cond=cond["name"],
                                 theta_deg=cond["theta_deg"], kappa=cond["kappa"],
                                 control=tag, baseline_strength=bs,
                                 strength=st, orient_err=orient_err(ax, cond["theta_deg"]),
                                 strength_interior=sti,
                                 orient_err_interior=orient_err(axi, cond["theta_deg"]),
                                 n_interior=int(inside.sum())))
                if tag == "ON":
                    panels.append((ic["name"], cond["name"], xy, st, rows[-1]["orient_err"]))
                print("  %-22s %-9s %-3s str=%.3f (int %.3f) oe=%.1f (int %.1f) n_int=%d"
                      % (ic["name"], cond["name"], tag, st, sti,
                         rows[-1]["orient_err"], rows[-1]["orient_err_interior"],
                         int(inside.sum())))

    _panels(panels, args, os.path.join(args.out, "panels.png"))

    # ── report ───────────────────────────────────────────────────────
    L = []
    L.append("M4 generalization -- primitives -> UNSEEN icons")
    L.append("=" * 66)
    L.append("train: %d samples / %d primitives, continuous theta,kappa" % (S, prim_hr.shape[0]))
    if loss_hist:
        w = max(1, min(args.log_every, len(loss_hist) // 4))
        L.append("steps=%d batch=%d  loss %.6f -> %.6f"
                 % (args.steps, args.batch, float(np.mean(loss_hist[:w])),
                    float(np.mean(loss_hist[-w:]))))
    L.append("base_trunc=%.2f (baseline stipple)  upgrade_trunc=%.2f  train_trunc=%.2f"
             % (args.base_trunc, args.upgrade_trunc, args.train_trunc))
    L.append("aniso branch %s density  (default OFF for transfer; frozen density branch always has it)"
             % ("SEES" if args.include_density else "BLIND to"))
    L.append("eval_seed=%d  control_gain=%.2f" % (args.eval_seed, args.control_gain))
    L.append("init=%s%s" % (args.init,
             ("  init_kappa=%.2f" % args.init_kappa) if args.init == "aniso" else ""))
    L.append("NOTE: interior columns are shape-contaminated on irregular icons")
    L.append("      (missing neighbours near ink edges fake orientation); gate uses ALL points.")
    L.append("")
    L.append("%-22s %-9s %-3s %8s %8s %8s %8s" % (
        "icon", "cond", "ctl", "str", "str_int", "oe", "oe_int"))
    for r in rows:
        L.append("%-22s %-9s %-3s %8.3f %8.3f %8.1f %8.1f" % (
            r["icon"][:22], r["cond"], r["control"], r["strength"],
            r["strength_interior"], r["orient_err"], r["orient_err_interior"]))
    L.append("")
    prim_rows = [r for r in rows if r["icon"].startswith("PRIM_")]
    if prim_rows:
        pon = [r for r in prim_rows if r["control"] == "ON"]
        pof = [r for r in prim_rows if r["control"] == "OFF"]
        L.append("IN-DOMAIN (primitives, same upgrade protocol): ON str=%.3f | OFF str=%.3f | ON oe=%.1f"
                 % (float(np.nanmean([r["strength"] for r in pon])),
                    float(np.nanmean([r["strength"] for r in pof])),
                    float(np.nanmean([r["orient_err"] for r in pon]))))
        L.append("  in-domain ON>>OFF but icons flat -> GENERALIZATION gap")
        L.append("  in-domain ON~=OFF too          -> NUCLEATION gap (protocol, not transfer)")
        L.append("")
    rows_ic = [r for r in rows if not r["icon"].startswith("PRIM_")]
    on = [r for r in rows_ic if r["control"] == "ON"]
    offr = [r for r in rows_ic if r["control"] == "OFF"]
    if on and offr:
        mon = float(np.nanmean([r["strength"] for r in on]))
        mof = float(np.nanmean([r["strength"] for r in offr]))
        moe = float(np.nanmean([r["orient_err"] for r in on]))
        L.append("MEAN over unseen icons (all points): control ON str=%.3f | OFF str=%.3f | ON orient_err=%.1f deg"
                 % (mon, mof, moe))
        L.append("GATE: ON > OFF + 0.03 -> %s ; orient_err < 20deg -> %s"
                 % (mon > mof + 0.03, moe < 20.0))
        L.append("")
        L.append("per condition (unseen icons, all points):")
        for c in sorted(set(r["cond"] for r in rows_ic)):
            con = [r for r in rows_ic if r["cond"] == c and r["control"] == "ON"]
            cof = [r for r in rows_ic if r["cond"] == c and r["control"] == "OFF"]
            L.append("  %-9s ON str=%.3f OFF str=%.3f | ON oe=%5.1f OFF oe=%5.1f" % (
                c, float(np.nanmean([r["strength"] for r in con])),
                float(np.nanmean([r["strength"] for r in cof])),
                float(np.nanmean([r["orient_err"] for r in con])),
                float(np.nanmean([r["orient_err"] for r in cof]))))
        L.append("")
        if mon > mof + 0.03 and moe < 20.0:
            L.append("RESULT: the anisotropy rule TRANSFERS to unseen images. Trained only on")
            L.append("a few primitives, the branch steers oriented structure on real icons it")
            L.append("never saw -- at fixed density, which density conditioning cannot do.")
        else:
            L.append("RESULT: no transfer yet. If ON~=OFF the branch cannot flip an isotropic")
            L.append("configuration (the nucleation gap) -- next levers: more control coverage,")
            L.append("higher upgrade_trunc, or an anisotropy-aware init.")
    report = "\n".join(L)
    with open(os.path.join(args.out, "objective_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(args=vars(args), rows=rows), f, indent=2)


def _panels(panels, args, path):
    icons = []
    cols = []
    for nm, tag, *_ in panels:
        if nm not in icons:
            icons.append(nm)
        if tag not in cols:
            cols.append(tag)
    fig, axes = plt.subplots(len(icons), len(cols),
                             figsize=(3.0 * len(cols), 3.2 * len(icons)))
    axes = np.atleast_2d(axes)
    for nm, tag, xy, st, oe in panels:
        ax = axes[icons.index(nm), cols.index(tag)]
        ax.scatter(xy[:, 0], 1.0 - xy[:, 1], s=args.dotsize, c="black", edgecolors="none")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("%s\n%s str %.2f%s" % (nm[:16], tag, st,
                                            ("  oe %.0f" % oe) if oe is not None else ""),
                     fontsize=7)
    fig.suptitle("Trained on primitives only -> applied to unseen icons", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
