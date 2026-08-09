"""Pair-interpolation eval panel for a K>=3 multistyle checkpoint.

Renders, for a few icons, the model along TWO simplex EDGES at t = 0, 0.5, 1:
    WVS <-> DITHER   and   GBN <-> DITHER
i.e. the vectors  [WVS], [0.5*A+0.5*B], [DITHER]  for each pair. Endpoints are one-hot vertices;
the 0.5 column is the unsupervised mid-edge blend. Vectors are built from oracle NAMES so the
column order is correct regardless of --oracles ordering.

Run from project root, e.g.:
  python control_v4_mix/eval_pair_panel.py \
      --control-ckpt control_v4_mix/train_outputs_multistyle_wvs_gbn_dither/checkpoints/dynamic_controlnet_v4_ep700.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_v4_mix.eval_multistyle_interpolation import (
    CONFIG_PATH, CKPT_PATH, GRID_SIZE, load_models, build_icons, decode_points, render_dots,
)
from control_v4_mix.oracles_config import resolve_oracles
from control_v4_mix.train_control_multistyle import dynamic_collate, ensure_offsets_dir, sample_eval_batch

DEFAULT_CKPT = ("control_v4_mix/train_outputs_multistyle_wvs_gbn_dither/checkpoints/"
                "dynamic_controlnet_v4_ep1000.pt")
DEFAULT_ICONS = ("microsoft_4_airplane.png,emoji-one_4_monkey.png,"
                 "samsung_2_volleyball.png,emoji-one_0_factory.png")
PAIRS = [("WVS", "DITHER"), ("GBN", "DITHER")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--base-ckpt", default=CKPT_PATH)
    ap.add_argument("--oracles", default="WVS;GBN;DITHER")
    ap.add_argument("--icons", default=DEFAULT_ICONS)
    ap.add_argument("--truncation", type=float, default=1.0)
    ap.add_argument("--eval-timesteps", type=int, default=1000)
    ap.add_argument("--resample-jumps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--radius", type=int, default=1)  # (unused; kept for compatibility)
    ap.add_argument("--dot-size", type=float, default=2.0,
                    help="vector scatter marker size in points^2 (crisp at any zoom)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="control_v4_mix/eval_pair_panel_out")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    G = GRID_SIZE
    pairs = resolve_oracles(args.oracles)
    names = [n for n, _ in pairs]
    K = len(names)
    idx = {n: i for i, n in enumerate(names)}
    for a, b in PAIRS:
        if a not in idx or b not in idx:
            print(f"ERROR: pair ({a},{b}) needs oracles present in --oracles={names}")
            return 2

    oracle_offsets = {}
    for nm, root in pairs:
        oracle_offsets[nm] = ensure_offsets_dir(os.path.join(root, "source"),
                                                os.path.join(root, "target"),
                                                os.path.join(root, "processed_offsets"), G)
    source_dir = os.path.join(pairs[0][1], "source")

    diffusion, denoiser, control_net = load_models(args.control_ckpt, args.config, args.base_ckpt,
                                                   K, G, device)
    icon_names = [x.strip() for x in args.icons.split(",") if x.strip()]
    items, _ = build_icons(source_dir, names, oracle_offsets, G, len(icon_names), icon_names)
    if not items:
        print("ERROR: none of the requested icons were found in ALL oracles.")
        return 2

    def onehot(nm):
        v = [0.0] * K; v[idx[nm]] = 1.0; return tuple(v)

    def mid(a, b):
        v = [0.0] * K; v[idx[a]] = 0.5; v[idx[b]] = 0.5; return tuple(v)

    # columns per pair: [A vertex, A+B mid, B vertex]; collect unique vectors to sample once.
    col_specs = []  # (pair_label, col_label, vec)
    for a, b in PAIRS:
        col_specs.append((f"{a}<->{b}", f"{a}\n{onehot(a)}", onehot(a)))
        col_specs.append((f"{a}<->{b}", f"mix\n{mid(a,b)}", mid(a, b)))
        col_specs.append((f"{a}<->{b}", f"{b}\n{onehot(b)}", onehot(b)))
    uniq_vecs = list(dict.fromkeys(v for _, _, v in col_specs))

    # sample each unique vector for all icons (same init noise across vectors per icon).
    preds = {v: [None] * len(items) for v in uniq_vecs}
    for v in uniq_vecs:
        batch = dynamic_collate([
            {"high_res": it["high_res"], "target_density": it["target_density"],
             "offsets": torch.from_numpy(it["teachers"][names[0]]).float(),
             "style_vec": torch.zeros(K)} for it in items
        ])
        batch = {k: (x.to(device) if torch.is_tensor(x) else x) for k, x in batch.items()}
        B = batch["high_res"].shape[0]
        batch["style_vec"] = torch.tensor(v, device=device, dtype=torch.float32).unsqueeze(0).expand(B, K).contiguous()
        torch.manual_seed(args.seed)
        pred = sample_eval_batch(diffusion, denoiser, control_net, batch, device, n_samples=B,
                                 eval_timesteps=args.eval_timesteps, resample_jumps=args.resample_jumps,
                                 show_tqdm=False, tqdm_desc=str(v), truncation_ratio=args.truncation
                                 ).detach().cpu().numpy()
        for j in range(B):
            preds[v][j] = decode_points(pred[j])

    # ---- render panel: rows = icons, cols = col_specs ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = len(col_specs)
    nrow = len(items)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.35 * nrow), dpi=140, squeeze=False)
    for i, it in enumerate(items):
        for jc, (_, col_label, v) in enumerate(col_specs):
            axc = axes[i][jc]
            _pts = preds[v][i]
            # Vector scatter (not a rasterized bitmap) so the PDF stays crisp on zoom.
            # y is flipped to keep image orientation (top-left origin).
            axc.scatter(_pts[:, 0], 1.0 - _pts[:, 1], s=args.dot_size, c="black",
                        marker="o", linewidths=0, edgecolors="none")
            axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.set_aspect("equal")
            axc.set_facecolor("white")
            if i == 0:
                axc.set_title(col_label, fontsize=8)
            if jc == 0:
                axc.set_ylabel(it["stem"].split("/")[-1][:18], fontsize=7)
            axc.set_xticks([]); axc.set_yticks([])
    # divider between the two pair-blocks (equal columns -> at x=0.5)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.02, 0.93], color="0.5", lw=1.2, ls="--"))
    fig.suptitle(f"K={K} {os.path.basename(args.control_ckpt)}  |  "
                 f"LEFT: {PAIRS[0][0]}<->{PAIRS[0][1]}   RIGHT: {PAIRS[1][0]}<->{PAIRS[1][1]}   "
                 "(t = 0, 0.5, 1)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf = os.path.join(args.out, "pair_panel.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(os.path.join(args.out, "pair_panel.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_pdf}")
    print(f"icons: {[it['stem'].split('/')[-1] for it in items]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
