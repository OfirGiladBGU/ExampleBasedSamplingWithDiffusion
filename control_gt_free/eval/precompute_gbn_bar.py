"""Compute the GBN 'bar to beat' ONCE: hard composite metrics + empirical PCF target.

The GT offsets in ``--offsets`` are the GBN/analytic reference outputs. This script:
  1. converts each reference point set and scores it with the HARD scipy validators
     (M1 CVT energy, M2 delta_c, M5 spatial measure) -> the numbers the model must beat;
  2. builds the EMPIRICAL density-warped soft-PCF target by averaging the soft PCF over the
     reference set (README default: pcf_target = EMPIRICAL from GBN). This is a "reference,
     not comparison" aggregate signature -- never a per-sample target -> no student/master cap.

Output: a .pt bundle {pcf_target, edges, gbn_hard_summary, config} written to --out.
Run once; training and validation load it.
"""

import argparse
import json
import os

import numpy as np
import torch

from control_gt_free.DynamicStippleDataset import DynamicStippleDataset
from control_gt_free.losses.soft_membership import offsets_to_coords, density_to_rho
from control_gt_free.losses.loss_pcf import pcf_from_points, default_edges

from utils.stippling_metrics_advance import (
    compute_m1_v1_cvt_energy, compute_m2_capacity_constraint, compute_m5_spatial_measure,
)
from data.Transforms import to_pointset_optimal_transport


def points_from_offsets_np(offsets_np):
    """(2,G,G) OT offsets -> (N,2) points in [0,1] via the exact validator-side transform."""
    pts_grid = to_pointset_optimal_transport(offsets_np)  # (2, G, G) in [0,1]
    g = pts_grid.shape[-1]
    return pts_grid.reshape(2, g * g).T  # (N, 2), col0 = x, col1 = y


def load_overfit_references(root, grid_size):
    """PCF references from an overfit output dir: each gt_offsets.npy + its sibling source.png.

    Returns dicts shaped like the dataset samples the averaging loop expects
    (offsets (2,G,G), high_res (1,H,W) in [0,1], target_density (1,G,G)).
    """
    import glob
    import cv2
    import torch.nn.functional as F

    samples = []
    for off_path in sorted(glob.glob(os.path.join(root, "**", "gt_offsets.npy"), recursive=True)):
        src_path = os.path.join(os.path.dirname(off_path), "source.png")
        if not os.path.exists(src_path):
            continue
        offsets_np = np.load(off_path).astype(np.float32)
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        high_res = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        target_density = F.interpolate(high_res, size=(grid_size, grid_size), mode="area")
        samples.append({
            "offsets": torch.from_numpy(offsets_np),
            "high_res": high_res.squeeze(0),
            "target_density": target_density.squeeze(0),
        })
    return samples


def build_bar_from_dataset(source, offsets, out, grid_size=32, max_refs=64,
                           pcf_bins=48, pcf_rmax=2.5, pcf_warp_grid=0,
                           device="cpu", hard_metrics=False):
    """Build + save the empirical PCF-target bar from a (source, offsets) dataset.

    Reusable API so train_control.py can auto-build the bar flag-free. hard_metrics=False
    skips the slow scipy Voronoi summary and computes only the PCF target (fast startup).
    """
    dev = torch.device(device)
    ds = DynamicStippleDataset(source, offsets, grid_size=grid_size,
                               smart_init_features=False, sdf_features=False, preload_ram=False)
    n = min(len(ds), max_refs)
    if n == 0:
        raise RuntimeError(f"[gbn-bar] empty dataset (source={source}, offsets={offsets})")
    edges = default_edges(pcf_bins, pcf_rmax, device=dev)
    pcf_accum = torch.zeros(pcf_bins, device=dev)
    hard = {"cvt_energy": [], "voronoi_mass_cv": [], "spatial_measure_rho_cv": [],
            "spatial_measure_rho_mean": []}
    for i in range(n):
        s_i = ds[i]
        offsets_np = s_i["offsets"].numpy()
        offs_t = torch.from_numpy(offsets_np).unsqueeze(0).to(dev)
        pts = offsets_to_coords(offs_t)
        d = s_i["target_density"].unsqueeze(0).to(dev)
        wg = pcf_warp_grid or None
        if wg is not None:
            d = torch.nn.functional.interpolate(d, size=(wg, wg), mode="area")
        pcf_accum += pcf_from_points(pts, density_to_rho(d), edges).squeeze(0)
        if hard_metrics:
            image01 = s_i["high_res"].squeeze(0).numpy()
            pts_np = points_from_offsets_np(offsets_np)
            hard["cvt_energy"].append(compute_m1_v1_cvt_energy(pts_np, image01).get("cvt_energy", 0.0))
            hard["voronoi_mass_cv"].append(compute_m2_capacity_constraint(pts_np, image01).get("voronoi_mass_cv", 0.0))
            m5 = compute_m5_spatial_measure(pts_np, image01)
            hard["spatial_measure_rho_cv"].append(m5.get("spatial_measure_rho_cv", 0.0))
            hard["spatial_measure_rho_mean"].append(m5.get("spatial_measure_rho_mean", 0.0))
    pcf_target = (pcf_accum / max(1, n)).cpu()
    summary = ({k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in hard.items()}
               if hard_metrics else None)
    bundle = {"pcf_target": pcf_target, "edges": edges.cpu(), "gbn_hard_summary": summary,
              "config": {"grid_size": grid_size, "pcf_bins": pcf_bins, "pcf_rmax": pcf_rmax,
                         "pcf_warp_grid": pcf_warp_grid, "n_refs": n}}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(bundle, out)
    print(f"[gbn-bar] wrote {out} (PCF target over {n} refs, hard_metrics={hard_metrics})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None, help="Dir of grayscale source images")
    ap.add_argument("--offsets", default=None, help="Dir of GBN reference .npy offsets")
    ap.add_argument("--from-overfit-dir", default=None,
                    help="Build the PCF target directly from an overfit output dir: each "
                         "gt_offsets.npy + sibling source.png is used as a blue-noise reference "
                         "(no dataset needed). Quick path for the transfer probe.")
    ap.add_argument("--out", required=True, help="Output .pt bundle path")
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--max-refs", type=int, default=200, help="Max reference samples to average")
    ap.add_argument("--pcf-bins", type=int, default=48)
    ap.add_argument("--pcf-rmax", type=float, default=2.5)
    ap.add_argument("--pcf-warp-grid", type=int, default=0,
                    help="rho-map resolution for the PCF warp (0 = full image res)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.from_overfit_dir:
        samples = load_overfit_references(args.from_overfit_dir, args.grid_size)
        if not samples:
            raise SystemExit(f"[gbn-bar] no gt_offsets.npy (+ source.png) under {args.from_overfit_dir}")
        get_sample = lambda i: samples[i]
        n = len(samples)
        print(f"[gbn-bar] building PCF target from {n} overfit reference(s) in {args.from_overfit_dir}")
    else:
        if not (args.source and args.offsets):
            raise SystemExit("[gbn-bar] need --source and --offsets, or --from-overfit-dir")
        ds = DynamicStippleDataset(
            args.source, args.offsets, grid_size=args.grid_size,
            smart_init_features=False, sdf_features=False, preload_ram=False,
        )
        n = min(len(ds), args.max_refs)
        get_sample = lambda i: ds[i]
        print(f"[gbn-bar] averaging over {n} / {len(ds)} reference samples")

    edges = default_edges(args.pcf_bins, args.pcf_rmax, device=device)
    pcf_accum = torch.zeros(args.pcf_bins, device=device)
    hard = {"cvt_energy": [], "voronoi_mass_cv": [], "spatial_measure_rho_cv": [],
            "spatial_measure_rho_mean": []}

    for i in range(n):
        s = get_sample(i)
        offsets_np = s["offsets"].numpy()
        image01 = s["high_res"].squeeze(0).numpy()  # (H, W) in [0,1], white=1

        # empirical soft-PCF target (torch, same code path as training)
        offs_t = torch.from_numpy(offsets_np).unsqueeze(0).to(device)
        pts = offsets_to_coords(offs_t)  # (1, N, 2)
        d = s["target_density"].unsqueeze(0).to(device)
        warp_grid = args.pcf_warp_grid or None
        if warp_grid is not None:
            d = torch.nn.functional.interpolate(d, size=(warp_grid, warp_grid), mode="area")
        rho_map = density_to_rho(d)
        pcf = pcf_from_points(pts, rho_map, edges)  # (1, n_bins)
        pcf_accum += pcf.squeeze(0)

        # hard validators on the same reference (the bar)
        pts_np = points_from_offsets_np(offsets_np)
        hard["cvt_energy"].append(compute_m1_v1_cvt_energy(pts_np, image01).get("cvt_energy", 0.0))
        hard["voronoi_mass_cv"].append(
            compute_m2_capacity_constraint(pts_np, image01).get("voronoi_mass_cv", 0.0))
        m5 = compute_m5_spatial_measure(pts_np, image01)
        hard["spatial_measure_rho_cv"].append(m5.get("spatial_measure_rho_cv", 0.0))
        hard["spatial_measure_rho_mean"].append(m5.get("spatial_measure_rho_mean", 0.0))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n}]")

    pcf_target = (pcf_accum / max(1, n)).cpu()
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in hard.items()}

    bundle = {
        "pcf_target": pcf_target,
        "edges": edges.cpu(),
        "gbn_hard_summary": summary,
        "config": {
            "grid_size": args.grid_size, "pcf_bins": args.pcf_bins,
            "pcf_rmax": args.pcf_rmax, "pcf_warp_grid": args.pcf_warp_grid,
            "n_refs": n,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(bundle, args.out)
    print(f"[gbn-bar] wrote {args.out}")
    print("[gbn-bar] HARD bar to beat (GBN reference):")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
