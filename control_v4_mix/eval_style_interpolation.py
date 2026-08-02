"""Interpolation eval for the WVS<->GBN style-conditioned checkpoint (Phase 2 validation).

Loads ONE style checkpoint and, on the held-out val icons, samples each icon at a SWEEP of style
values s (default 0, 0.25, 0.5, 0.75, 1.0) by overriding the conditioning scalar. For every (icon,
s) it decodes the predicted offsets to points and measures the Gate-0 descriptor norm_nn_cv (density-
controlled spacing regularity) plus clumping and Chamfer distance to BOTH teachers.

The whole point is s=0.5: does the model produce a genuine MIX between WVS (uniform fill, low
norm_nn_cv) and GBN (contour-banding, high norm_nn_cv), or does it snap to an endpoint / go muddy?

Checks:
  * endpoint fidelity  -- s=0 tracks the WVS teacher, s=1 tracks the GBN teacher.
  * monotonicity       -- measured norm_nn_cv rises smoothly with s (no snapping).
  * s=0.5 mix          -- reported as "% of the way" from WVS to GBN (ideal ~50%).
Outputs a metrics table + JSON/CSV, an interpolation curve, and a dot-render montage
(WVS teacher | s-sweep | GBN teacher) -- the visual test of whether the mix is legible.

Run from project root.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from control_v4_mix.spacing_regularity import compute_spacing_bundle, EPS
from control_v4_mix.StyleStippleDataset import StyleStippleDataset
from control_v4_mix.DynamicControlNetStyle import DynamicControlNetStyle
from control_v4_mix.train_control_style import (
    dynamic_collate, ensure_offsets_dir, sample_eval_batch,
)

WVS_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_WVS"
GBN_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/icons-50_512_GBN"
CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"
STYLE_S_JSON = "control_v4_mix/style_s.json"
CKPT = ("control_v4_mix/train_outputs_style_wvs_gbn/checkpoints/"
        "dynamic_controlnet_v4_ep1000.pt")
GRID_SIZE = 32
VAL_SPLIT = 0.1
SPLIT_SEED = 42


# ------------------------------------------------------------------ helpers
def decode_points(offsets_2gg):
    return to_pointset_optimal_transport(offsets_2gg).reshape(2, -1).T


def chamfer(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float(da.mean() + db.mean())


def render_dots(pts, size=256, radius=1):
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 255)
    dr = ImageDraw.Draw(img)
    for x, y in pts:
        cx, cy = x * size, y * size
        dr.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=0)
    return np.asarray(img, dtype=np.uint8)


def load_style_models(ckpt, config, base_ckpt, grid_size, device):
    diffusion = ParseSampleConfig(config)
    diffusion.load_state_dict(torch.load(base_ckpt, map_location="cpu")["diffu"])
    diffusion.to(device); diffusion.eval()
    denoiser = diffusion.model
    control_net = DynamicControlNetStyle(
        denoiser, grid_size=grid_size, enable_gecco=True,
        smart_init_features=False, sdf_features=False, batch_coords_features=False,
        enable_adaptive_gate_injection=True,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu")
    control_net.safe_load_state_dict(state, strict=False)
    n_style = sum(p.numel() for p in control_net.style_mlp.parameters())
    loaded_style = any(k for k in _control_keys(state) if "style_mlp" in k)
    if isinstance(state, dict) and state.get("denoiser") is not None:
        denoiser.load_state_dict(state["denoiser"], strict=False)
        print("Loaded trained (unfrozen) base denoiser from checkpoint.")
    control_net.eval()
    print(f"style_mlp params: {n_style:,}; style weights present in ckpt: {loaded_style}")
    if not loaded_style:
        print("  !! WARNING: no style_mlp weights found in the checkpoint -- s will have NO effect. "
              "Is this actually a style-trained checkpoint?")
    return diffusion, denoiser, control_net


def _control_keys(state):
    if not isinstance(state, dict):
        return []
    for key in ("control_net", "model_state_dict", "state_dict"):
        v = state.get(key)
        if isinstance(v, dict):
            return list(v.keys())
    return list(state.keys())


def build_icons(source_dir, wvs_off, gbn_off, style_json, grid_size, n, icon_names=None):
    # WVS-only view gives one sample per icon with the shared condition + WVS teacher offsets.
    ds = StyleStippleDataset(source_dir, {"WVS": wvs_off}, style_json, grid_size=grid_size)
    gbn_map = {}
    for r, _, files in os.walk(gbn_off):
        for f in files:
            if f.endswith(".npy"):
                stem = os.path.splitext(os.path.relpath(os.path.join(r, f), gbn_off))[0]
                gbn_map[stem] = os.path.join(r, f)

    if icon_names:
        # Explicit list: match requested basenames (with or without .png) across ALL icons,
        # preserving the requested order. NOTE some may be TRAIN-set icons (the model saw them);
        # fine for a qualitative check, but not a held-out claim.
        base_to_i = {}
        for i in range(len(ds)):
            base_to_i.setdefault(os.path.basename(ds.samples[i][0]), i)
        idxs, missing = [], []
        for name in icon_names:
            key = os.path.splitext(os.path.basename(name))[0]
            if key in base_to_i:
                idxs.append(base_to_i[key])
            else:
                missing.append(name)
        if missing:
            print(f"  !! icons not found (skipped): {missing}")
    else:
        n_files = len(ds.filenames)
        val_len = min(max(int(n_files * VAL_SPLIT), 0), max(n_files - 1, 0))
        order = torch.randperm(n_files, generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
        train_len = n_files - val_len
        val_files = set(ds.filenames[i] for i in order[train_len:])
        idxs = [i for i in range(len(ds)) if ds.samples[i][1] in val_files]

    items = []
    for i in idxs:
        stem = ds.samples[i][0]
        if stem not in gbn_map:
            continue
        s = ds[i]
        items.append({
            "stem": stem,
            "high_res": s["high_res"], "target_density": s["target_density"],
            "wvs_off": s["offsets"].numpy(),
            "gbn_off": np.load(gbn_map[stem]),
        })
        if not icon_names and len(items) >= n:
            break
    print(f"icons scored: {len(items)}"
          + ("" if not icon_names else f" (explicit list; {len(icon_names)} requested)"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-ckpt", default=CKPT)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--base-ckpt", default=CKPT_PATH)
    ap.add_argument("--style-s-json", default=STYLE_S_JSON)
    ap.add_argument("--wvs-root", default=WVS_ROOT)
    ap.add_argument("--gbn-root", default=GBN_ROOT)
    ap.add_argument("--s-values", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--icons", default="",
                    help="comma-separated icon filenames/basenames to eval instead of the val split")
    ap.add_argument("--n-montage", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--truncation", type=float, default=1.0)
    ap.add_argument("--resample-jumps", type=int, default=0)
    ap.add_argument("--eval-timesteps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="control_v4_mix/eval_style_interpolation_out")
    args = ap.parse_args()

    s_values = [float(x) for x in args.s_values.split(",")]
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    G = GRID_SIZE

    wvs_off = ensure_offsets_dir(os.path.join(args.wvs_root, "source"),
                                 os.path.join(args.wvs_root, "target"),
                                 os.path.join(args.wvs_root, "processed_offsets"), G)
    gbn_off = ensure_offsets_dir(os.path.join(args.gbn_root, "source"),
                                 os.path.join(args.gbn_root, "target"),
                                 os.path.join(args.gbn_root, "processed_offsets"), G)
    source_dir = os.path.join(args.wvs_root, "source")

    refs = json.load(open(args.style_s_json))["_refs"]
    wvs_ref, gbn_ref = refs["wvs_ref_zero"], refs["gbn_ref_one"]
    denom = (gbn_ref - wvs_ref) if abs(gbn_ref - wvs_ref) > 1e-9 else 1.0
    to_s = lambda cv: (cv - wvs_ref) / denom

    diffusion, denoiser, control_net = load_style_models(
        args.control_ckpt, args.config, args.base_ckpt, G, device)

    icon_names = [x.strip() for x in args.icons.split(",") if x.strip()] or None
    items = build_icons(source_dir, wvs_off, gbn_off, args.style_s_json, G, args.n_samples,
                        icon_names=icon_names)
    if not items:
        print("ERROR: no val icons."); return 2

    # teacher references per icon
    for it in items:
        it["wvs_cv"] = compute_spacing_bundle(decode_points(it["wvs_off"]))["norm_nn_cv"]
        it["gbn_cv"] = compute_spacing_bundle(decode_points(it["gbn_off"]))["norm_nn_cv"]

    # preds[s][icon_idx] = decoded points
    preds = {s: [None] * len(items) for s in s_values}
    per_rows = []
    for s in s_values:
        for c0 in range(0, len(items), args.chunk):
            chunk = items[c0:c0 + args.chunk]
            batch = dynamic_collate([
                {"high_res": it["high_res"], "target_density": it["target_density"],
                 "offsets": torch.from_numpy(it["wvs_off"]).float(),
                 "style_s": torch.tensor(0.0)} for it in chunk
            ])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            B = batch["high_res"].shape[0]
            batch["style_s"] = torch.full((B,), float(s), device=device)
            torch.manual_seed(args.seed + c0)  # same init noise across s for this chunk
            pred = sample_eval_batch(
                diffusion, denoiser, control_net, batch, device, n_samples=B,
                eval_timesteps=args.eval_timesteps, resample_jumps=args.resample_jumps,
                show_tqdm=False, tqdm_desc=f"s={s}", truncation_ratio=args.truncation,
            ).detach().cpu().numpy()
            for j in range(B):
                pts = decode_points(pred[j])
                preds[s][c0 + j] = pts
                b = compute_spacing_bundle(pts)
                per_rows.append({
                    "stem": chunk[j]["stem"], "s": s,
                    "norm_nn_cv": b["norm_nn_cv"], "measured_s": to_s(b["norm_nn_cv"]),
                    "clumped_pct_local": b["clumped_pct_local"],
                    "chamfer_wvs": chamfer(pts, decode_points(chunk[j]["wvs_off"])),
                    "chamfer_gbn": chamfer(pts, decode_points(chunk[j]["gbn_off"])),
                })

    # ---- aggregate ----
    def col(s, key):
        return np.array([r[key] for r in per_rows if r["s"] == s], float)
    wvs_cv_mean = float(np.mean([it["wvs_cv"] for it in items]))
    gbn_cv_mean = float(np.mean([it["gbn_cv"] for it in items]))

    print("\n" + "=" * 92)
    print(f"STYLE INTERPOLATION  ckpt={os.path.basename(args.control_ckpt)}  "
          f"icons={len(items)}  trunc={args.truncation}")
    print("=" * 92)
    print(f"teacher norm_nn_cv: WVS(s=0 target)={wvs_cv_mean:.4f}  GBN(s=1 target)={gbn_cv_mean:.4f}")
    hdr = (f"{'s':>6}{'norm_nn_cv':>12}{'measured_s':>12}{'clump%':>9}"
           f"{'chamf_WVS':>11}{'chamf_GBN':>11}")
    print(hdr); print("-" * len(hdr))
    agg = {}
    for s in s_values:
        cv = float(np.mean(col(s, "norm_nn_cv")))
        ms = float(np.mean(col(s, "measured_s")))
        cl = float(np.mean(col(s, "clumped_pct_local")))
        cw = float(np.mean(col(s, "chamfer_wvs")))
        cg = float(np.mean(col(s, "chamfer_gbn")))
        agg[s] = dict(norm_nn_cv=cv, measured_s=ms, clump=cl, chamfer_wvs=cw, chamfer_gbn=cg)
        print(f"{s:>6.2f}{cv:>12.4f}{ms:>12.3f}{cl:>9.2f}{cw:>11.4f}{cg:>11.4f}")

    # ---- verdicts ----
    cvs = [agg[s]["norm_nn_cv"] for s in s_values]
    mono = all(cvs[i] <= cvs[i + 1] + 1e-4 for i in range(len(cvs) - 1))
    span = cvs[-1] - cvs[0]
    print("\nVERDICTS")
    print(f"  monotonic (norm_nn_cv rises with s): {'YES' if mono else 'NO'}  "
          f"[{', '.join(f'{c:.3f}' for c in cvs)}]")
    # endpoint fidelity: model endpoints vs teacher endpoints (in norm_nn_cv space)
    e0 = agg[s_values[0]]["norm_nn_cv"]; e1 = agg[s_values[-1]]["norm_nn_cv"]
    print(f"  endpoint s={s_values[0]}: model {e0:.3f} vs WVS teacher {wvs_cv_mean:.3f}")
    print(f"  endpoint s={s_values[-1]}: model {e1:.3f} vs GBN teacher {gbn_cv_mean:.3f}")
    if 0.5 in agg and abs(span) > 1e-6:
        frac = (agg[0.5]["norm_nn_cv"] - cvs[0]) / span
        print(f"  s=0.5 MIX: norm_nn_cv={agg[0.5]['norm_nn_cv']:.3f} = {100*frac:.0f}% of the way "
              f"WVS->GBN (ideal ~50%). chamfer_WVS={agg[0.5]['chamfer_wvs']:.4f} "
              f"chamfer_GBN={agg[0.5]['chamfer_gbn']:.4f}")
        if frac < 0.15 or frac > 0.85:
            print("    -> SNAPPING: s=0.5 sits at an endpoint, not a mix.")
        else:
            print("    -> intermediate: s=0.5 is a genuine blend on the metric.")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # interpolation curve
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
        ax.plot(s_values, cvs, "o-", color="#3b6ea5", label="model norm_nn_cv")
        ax.axhline(wvs_cv_mean, color="#c05555", ls="--", lw=1, label="WVS teacher")
        ax.axhline(gbn_cv_mean, color="#2a7d2a", ls="--", lw=1, label="GBN teacher")
        ax.set_xlabel("requested style s"); ax.set_ylabel("measured norm_nn_cv")
        ax.set_title("Style interpolation: regularity vs s"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "interp_curve.pdf"),
                                        bbox_inches="tight"); plt.close(fig)
        # dot montage: WVS teacher | s-sweep | GBN teacher
        nshow = len(items) if icon_names else min(args.n_montage, len(items))
        cols = ["WVS\nteacher"] + [f"s={s:g}" for s in s_values] + ["GBN\nteacher"]
        fig, axes = plt.subplots(nshow, len(cols), figsize=(2.1 * len(cols), 2.2 * nshow),
                                 dpi=120, squeeze=False)
        for i in range(nshow):
            panels = [decode_points(items[i]["wvs_off"])] + \
                     [preds[s][i] for s in s_values] + [decode_points(items[i]["gbn_off"])]
            for j, pts in enumerate(panels):
                axc = axes[i][j]
                # Vector scatter (not a rasterized dot image) so the PDF stays crisp at any zoom.
                axc.scatter(pts[:, 0], pts[:, 1], s=1.5, c="black", linewidths=0)
                axc.set_xlim(0, 1); axc.set_ylim(0, 1)
                axc.invert_yaxis()  # image orientation (y increases downward)
                axc.set_aspect("equal")
                axc.set_xticks([]); axc.set_yticks([])
                if i == 0:
                    axc.set_title(cols[j], fontsize=9)
            axes[i][0].set_ylabel(items[i]["stem"][:16], fontsize=7)
        fig.suptitle("WVS teacher | model s-sweep | GBN teacher  (can you SEE the mix at s=0.5?)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(os.path.join(args.out, "interp_montage.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"\n[plots] {args.out}/interp_curve.pdf , interp_montage.pdf")
    except Exception as exc:
        print(f"[plots] skipped: {exc}")

    # ---- write metrics ----
    with open(os.path.join(args.out, "interp_metrics.json"), "w") as f:
        json.dump({"ckpt": args.control_ckpt, "wvs_cv_mean": wvs_cv_mean,
                   "gbn_cv_mean": gbn_cv_mean, "agg": {str(k): v for k, v in agg.items()},
                   "monotonic": mono}, f, indent=2)
    with open(os.path.join(args.out, "interp_per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        for r in per_rows:
            w.writerow(r)
    print(f"[write] {args.out}/interp_metrics.json , interp_per_sample.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
