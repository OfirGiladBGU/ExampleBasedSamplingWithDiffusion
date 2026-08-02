"""Simplex-interpolation eval for the multi-oracle style checkpoint (WVS / GBN / DITHER / ...).

Loads a multistyle checkpoint and, on held-out val icons, samples each icon at a set of style
VECTORS -- the one-hot vertices plus mixtures like [0.5,0.5,0] and the centroid [1/3,1/3,1/3].
For each (icon, vector) it decodes offsets to points and measures norm_nn_cv + clumping + Chamfer
to every oracle teacher. The point: does a mixture vector produce a genuine BLEND of the oracles?

Run from project root. --oracles MUST match the training order (fixes the one-hot index / K).
"""

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig
from control_v4_mix.spacing_regularity import compute_spacing_bundle
from control_v4_mix.MultiStyleStippleDataset import MultiStyleStippleDataset
from control_v4_mix.data_split import source_train_val_split, split_from_manifest
from control_v4_mix.DynamicControlNetMultiStyle import DynamicControlNetMultiStyle, style_scalar_to_vec
from control_v4_mix.train_control_multistyle import (
    dynamic_collate, ensure_offsets_dir, sample_eval_batch,
)
from control_v4_mix.oracles_config import ORACLES_DEFAULT, resolve_oracles

CONFIG_PATH = "config/GBN/config.json"
CKPT_PATH = "config/GBN/model.ckpt"
CKPT = "control_v4_mix/train_outputs_multistyle/checkpoints/dynamic_controlnet_v4_ep1000.pt"
GRID_SIZE = 32
VAL_SPLIT = 0.1
SPLIT_SEED = 42


def decode_points(off):
    return to_pointset_optimal_transport(off).reshape(2, -1).T


def chamfer(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return float(tb.query(a, k=1)[0].mean() + ta.query(b, k=1)[0].mean())


def render_dots(pts, size=256, radius=1):
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 255)
    dr = ImageDraw.Draw(img)
    for x, y in pts:
        dr.ellipse([x * size - radius, y * size - radius, x * size + radius, y * size + radius], fill=0)
    return np.asarray(img, dtype=np.uint8)


def parse_oracles(spec):
    pairs = resolve_oracles(spec)
    return [n for n, _ in pairs], {n: r for n, r in pairs}


def default_style_vecs(K):
    vecs = [tuple(1.0 if i == k else 0.0 for i in range(K)) for k in range(K)]        # vertices
    for a, b in itertools.combinations(range(K), 2):                                   # pairwise mids
        v = [0.0] * K; v[a] = v[b] = 0.5; vecs.append(tuple(v))
    if K >= 3:
        vecs.append(tuple(round(1.0 / K, 4) for _ in range(K)))                        # centroid
    return vecs


def load_models(ckpt, config, base_ckpt, K, grid_size, device):
    diffusion = ParseSampleConfig(config)
    diffusion.load_state_dict(torch.load(base_ckpt, map_location="cpu")["diffu"])
    diffusion.to(device); diffusion.eval()
    denoiser = diffusion.model
    control_net = DynamicControlNetMultiStyle(
        denoiser, grid_size=grid_size, style_dim=K, enable_gecco=True,
        smart_init_features=False, sdf_features=False, batch_coords_features=False,
        enable_adaptive_gate_injection=True,
    ).to(device)
    state = torch.load(ckpt, map_location="cpu")
    control_net.safe_load_state_dict(state, strict=False)
    for key in ("control_net", "model_state_dict", "state_dict"):
        v = state.get(key) if isinstance(state, dict) else None
        if isinstance(v, dict):
            present = any("style_mlp" in k for k in v)
            in_dim = v.get("style_mlp.0.weight")
            if in_dim is not None and in_dim.shape[1] != K:
                print(f"  !! ckpt style_dim={in_dim.shape[1]} but --oracles gives K={K}. Mismatch!")
            print(f"style weights present: {present}")
            break
    if isinstance(state, dict) and state.get("denoiser") is not None:
        denoiser.load_state_dict(state["denoiser"], strict=False)
        print("Loaded trained (unfrozen) base denoiser from checkpoint.")
    control_net.eval()
    return diffusion, denoiser, control_net


def build_icons(source_dir, oracle_names, oracle_offsets, grid, n, icon_names=None, val_manifest=None):
    ds = MultiStyleStippleDataset(source_dir, oracle_names, oracle_offsets, grid_size=grid)
    stem_to_idx = {}
    for i, (stem, rel, name, k) in enumerate(ds.samples):
        stem_to_idx.setdefault(stem, i)
    # keep only icons that have ALL oracles (fair teacher comparison)
    stems = [s for s in stem_to_idx if all(s in ds.off_maps[nm] for nm in oracle_names)]

    if icon_names:
        base_to_stem = {}
        for s in stems:
            base_to_stem.setdefault(os.path.basename(s), s)
        chosen, missing = [], []
        for name in icon_names:
            key = os.path.splitext(os.path.basename(name))[0]
            (chosen.append(base_to_stem[key]) if key in base_to_stem else missing.append(name))
        if missing:
            print(f"  !! icons not found (skipped): {missing}")
    else:
        # Same held-out val set as training: prefer the manifest, else the source-folder split.
        if val_manifest and os.path.isfile(val_manifest):
            _, val_files_list = split_from_manifest(source_dir, val_manifest)
        else:
            _, val_files_list = source_train_val_split(source_dir, VAL_SPLIT, seed=SPLIT_SEED)
        val_files = set(val_files_list)
        chosen = [s for s in sorted(stems) if ds.samples[stem_to_idx[s]][1] in val_files][:n]

    items = []
    for stem in chosen:
        s = ds[stem_to_idx[stem]]
        items.append({
            "stem": stem, "high_res": s["high_res"], "target_density": s["target_density"],
            "teachers": {nm: np.load(ds.off_maps[nm][stem]) for nm in oracle_names},
        })
    print(f"icons scored: {len(items)}")
    return items, ds.K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-ckpt", default=CKPT)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--base-ckpt", default=CKPT_PATH)
    ap.add_argument("--oracles", default=ORACLES_DEFAULT)
    ap.add_argument("--style-vecs", default="", help="'a,b,c;a,b,c;...' (default: vertices+mixes)")
    ap.add_argument("--s-values", default="",
                    help="BACKWARD-COMPAT scalar sweep 0=WVS..1=GBN (K=2 only; maps s -> [1-s, s])")
    ap.add_argument("--icons", default="")
    ap.add_argument("--val-manifest", default="control_v4_mix/validation_manifest.json",
                    help="JSON list of val basenames; defines the exact val set (matches training)")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--n-montage", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--truncation", type=float, default=1.0)
    ap.add_argument("--resample-jumps", type=int, default=0)
    ap.add_argument("--eval-timesteps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="control_v4_mix/eval_multistyle_out")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    G = GRID_SIZE
    names, roots = parse_oracles(args.oracles)
    K = len(names)

    oracle_offsets = {}
    for nm in names:
        oracle_offsets[nm] = ensure_offsets_dir(os.path.join(roots[nm], "source"),
                                                os.path.join(roots[nm], "target"),
                                                os.path.join(roots[nm], "processed_offsets"), G)
    source_dir = os.path.join(roots[names[0]], "source")

    if args.s_values.strip():
        if K != 2:
            print("ERROR: --s-values (scalar 0=WVS,1=GBN) requires exactly K=2 oracles; "
                  "use --style-vecs for K>2.")
            return 2
        vecs = [tuple(style_scalar_to_vec(float(x)).tolist())
                for x in args.s_values.split(",") if x.strip()]
        print(f"scalar backward-compat mode: s -> [1-s, s]  (0={names[0]}, 1={names[1]})")
    elif args.style_vecs.strip():
        vecs = [tuple(float(x) for x in v.split(",")) for v in args.style_vecs.split(";") if v.strip()]
    else:
        vecs = default_style_vecs(K)
    for v in vecs:
        assert len(v) == K, f"style vec {v} has len {len(v)} != K={K}"

    diffusion, denoiser, control_net = load_models(
        args.control_ckpt, args.config, args.base_ckpt, K, G, device)
    icon_names = [x.strip() for x in args.icons.split(",") if x.strip()] or None
    items, _ = build_icons(source_dir, names, oracle_offsets, G, args.n_samples, icon_names,
                           val_manifest=args.val_manifest)
    if not items:
        print("ERROR: no icons."); return 2

    for it in items:
        it["teacher_cv"] = {nm: compute_spacing_bundle(decode_points(off))["norm_nn_cv"]
                            for nm, off in it["teachers"].items()}

    preds = {v: [None] * len(items) for v in vecs}
    per_rows = []
    for v in vecs:
        for c0 in range(0, len(items), args.chunk):
            chunk = items[c0:c0 + args.chunk]
            batch = dynamic_collate([
                {"high_res": it["high_res"], "target_density": it["target_density"],
                 "offsets": torch.from_numpy(it["teachers"][names[0]]).float(),
                 "style_vec": torch.zeros(K)} for it in chunk
            ])
            batch = {k: (x.to(device) if torch.is_tensor(x) else x) for k, x in batch.items()}
            B = batch["high_res"].shape[0]
            batch["style_vec"] = torch.tensor(v, device=device, dtype=torch.float32).unsqueeze(0).expand(B, K).contiguous()
            torch.manual_seed(args.seed + c0)
            pred = sample_eval_batch(diffusion, denoiser, control_net, batch, device, n_samples=B,
                                     eval_timesteps=args.eval_timesteps, resample_jumps=args.resample_jumps,
                                     show_tqdm=False, tqdm_desc=str(v), truncation_ratio=args.truncation
                                     ).detach().cpu().numpy()
            for j in range(B):
                pts = decode_points(pred[j]); preds[v][c0 + j] = pts
                b = compute_spacing_bundle(pts)
                row = {"stem": chunk[j]["stem"], "vec": "|".join(f"{x:g}" for x in v),
                       "norm_nn_cv": b["norm_nn_cv"], "clumped_pct_local": b["clumped_pct_local"]}
                for nm in names:
                    row[f"chamfer_{nm}"] = chamfer(pts, decode_points(chunk[j]["teachers"][nm]))
                per_rows.append(row)

    # ---- table ----
    print("\n" + "=" * 96)
    print(f"MULTI-STYLE INTERPOLATION  ckpt={os.path.basename(args.control_ckpt)}  "
          f"K={K} {names}  icons={len(items)}")
    print("=" * 96)
    tcv = {nm: float(np.mean([it['teacher_cv'][nm] for it in items])) for nm in names}
    print("teacher norm_nn_cv: " + "  ".join(f"{nm}={tcv[nm]:.4f}" for nm in names))
    cols = "".join(f"{'chamf_' + nm:>13}" for nm in names)
    hdr = f"{'style vec':>22}{'norm_nn_cv':>12}{'clump%':>9}" + cols
    print(hdr); print("-" * len(hdr))
    def sub(v, key):
        vs = "|".join(f"{x:g}" for x in v)
        return np.array([r[key] for r in per_rows if r["vec"] == vs], float)
    agg = {}
    for v in vecs:
        cv = float(np.mean(sub(v, "norm_nn_cv"))); cl = float(np.mean(sub(v, "clumped_pct_local")))
        ch = {nm: float(np.mean(sub(v, f"chamfer_{nm}"))) for nm in names}
        agg["|".join(f"{x:g}" for x in v)] = dict(norm_nn_cv=cv, clump=cl, chamfer=ch)
        vs = "[" + ",".join(f"{x:g}" for x in v) + "]"
        print(f"{vs:>22}{cv:>12.4f}{cl:>9.2f}" + "".join(f"{ch[nm]:>13.4f}" for nm in names))
    print("\n  For a mixture vec, chamfer should be LOW to the oracles it blends and HIGHER to the")
    print("  excluded one; norm_nn_cv should sit between the blended oracles' teacher values.")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nshow = len(items) if icon_names else min(args.n_montage, len(items))
        col_labels = [f"{nm}\nteacher" for nm in names] + \
                     ["[" + ",".join(f"{x:g}" for x in v) + "]" for v in vecs]
        ncol = len(col_labels)
        fig, axes = plt.subplots(nshow, ncol, figsize=(1.7 * ncol, 2.0 * nshow), dpi=120, squeeze=False)
        for i in range(nshow):
            panels = [decode_points(items[i]["teachers"][nm]) for nm in names] + \
                     [preds[v][i] for v in vecs]
            for jc, pts in enumerate(panels):
                axc = axes[i][jc]
                axc.imshow(render_dots(pts), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                if i == 0:
                    axc.set_title(col_labels[jc], fontsize=8)
                axc.axis("off")
            axes[i][0].set_ylabel(items[i]["stem"][:14], fontsize=6)
        fig.suptitle("oracle teachers | model at style vectors  (do mixtures blend the oracles?)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(os.path.join(args.out, "multistyle_montage.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {args.out}/multistyle_montage.pdf")
    except Exception as exc:
        print(f"[plot] skipped: {exc}")

    with open(os.path.join(args.out, "multistyle_metrics.json"), "w") as f:
        json.dump({"ckpt": args.control_ckpt, "oracles": names, "teacher_cv": tcv, "agg": agg}, f, indent=2)
    with open(os.path.join(args.out, "multistyle_per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys())); w.writeheader()
        for r in per_rows:
            w.writerow(r)
    print(f"[write] {args.out}/multistyle_metrics.json , multistyle_per_sample.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
