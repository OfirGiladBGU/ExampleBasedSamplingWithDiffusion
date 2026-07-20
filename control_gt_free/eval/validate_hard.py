"""Hard validation: run the scipy/FFT validators + dump a visual grid. The ONLY truth.

Guardrails: soft losses are training-only; every reported number here comes from the
untouched validators in ``utils/stippling_metrics_advance.py``; a ~20-image visual grid is
saved every call and must be inspected for clumping (a metric win with visible clumping is
NOT a win).

Also holds the shared runtime used by both validation and training-eval:
  * ``build_conditioner`` -- reconstruct the single-stage velocity net + conditioner from a cfg;
  * ``sample_offsets`` -- grad-free Flow-Matching ODE sampling (noise->data or smart-init->data).

``run_hard_validation`` reports M1/M2/M5 for the model, logs the soft-vs-hard capacity gap
(tau health) and current tau, and compares against the precomputed GBN bar.
"""

import argparse
import json
import os

import numpy as np
import torch

from control_gt_free.flow_matching import FlowMatching
from control_gt_free.losses.soft_membership import (
    offsets_to_coords, density_grid, grid_centers_flat,
)
from control_gt_free.losses.loss_capacity import soft_capacity_cv

from utils.stippling_metrics_advance import (
    compute_m1_cvt_energy, compute_m2_capacity_constraint, compute_m5_spatial_measure,
)


def build_conditioner(cfg, device):
    """Rebuild the single-stage (concat/spade) velocity net + conditioner from a config."""
    conditioning = cfg.get("conditioning", "concat")
    use_grid = bool(cfg.get("concat_smart_init_grid", False))
    n_cond = 1 + (1 if use_grid else 0)
    base_cfg = cfg["base_config_path"]
    if conditioning == "spade":
        from control_gt_free.single_stage import build_spade_velocity_network, SPADEConditioner
        net = build_spade_velocity_network(base_cfg, cond_channels=n_cond, device=device)
        cond = SPADEConditioner(net, use_smart_init_grid=use_grid).to(device)
    else:
        from control_gt_free.single_stage import build_conditional_velocity_network, SingleStageConditioner
        net = build_conditional_velocity_network(base_cfg, extra_in_channels=n_cond, device=device)
        cond = SingleStageConditioner(net, use_smart_init_grid=use_grid).to(device)
    return cond


@torch.no_grad()
def sample_offsets(fm, conditioner, batch, device, n_samples, grid_size,
                   ode_steps=50, ode_method="euler", t_start=1.0, show_tqdm=False):
    """Grad-free ODE sampling -> offset grids (n_samples, 2, G, G)."""
    target_density = batch["target_density"].to(device)
    smart_grid = batch.get("smart_init_grid")
    conditioner.set_condition(
        target_density,
        smart_grid.to(device) if (conditioner.use_smart_init_grid and smart_grid is not None) else None,
    )
    conditioner.eval()
    shape = [n_samples, 2, grid_size, grid_size]

    x_start = None
    if fm.coupling == "smartinit":
        src = batch.get("smart_init_offsets")
        if src is None:
            raise ValueError("coupling='smartinit' needs smart_init_offsets in the batch")
        x_start = fm.start_state(shape, device=device, smart_init=src.to(device))

    def velocity_fn(x, t_net):
        return conditioner(x, t_net)

    return fm.ode_sample(velocity_fn, shape, device=device, n_steps=ode_steps,
                         method=ode_method, t_start=t_start, x_start=x_start, show_tqdm=show_tqdm)


def offsets_to_points_np(offsets):
    """(B,2,G,G) tensor -> list of (N,2) numpy point sets in [0,1]."""
    coords = offsets_to_coords(offsets, clamp=True).detach().cpu().numpy()
    return [coords[b] for b in range(coords.shape[0])]


def hard_metrics_for_points(points_np, image01):
    m1 = compute_m1_cvt_energy(points_np, image01)
    m2 = compute_m2_capacity_constraint(points_np, image01)
    m5 = compute_m5_spatial_measure(points_np, image01)
    return {
        "cvt_energy": m1.get("cvt_energy", 0.0),
        "voronoi_mass_cv": m2.get("voronoi_mass_cv", 0.0),
        "spatial_measure_rho_cv": m5.get("spatial_measure_rho_cv", 0.0),
        "spatial_measure_rho_mean": m5.get("spatial_measure_rho_mean", 0.0),
    }


def save_visual_grid(points_list, out_path, ncols=5, title=None):
    """Scatter ~20 output point sets for the mandatory clumping inspection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    n = len(points_list)
    if n == 0:
        return False
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            p = points_list[i]
            ax.scatter(p[:, 0], 1.0 - p[:, 1], s=1.2, c="black")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def summarize(rows):
    keys = list(rows[0].keys()) if rows else []
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows]))} for k in keys}


def run_hard_validation(fm, conditioner, val_batches, device, grid_size, out_dir,
                        tag="model", gbn_bar=None, tau=None, loss_grid=32,
                        ode_steps=50, ode_method="euler", t_start=1.0,
                        max_visual=20, show_tqdm=False):
    """Sample over val_batches, score with the HARD validators, save a visual grid, and log
    the soft-vs-hard capacity gap. Returns per-metric mean/std + tau-health + GBN comparison.
    """
    conditioner.eval()
    rows, visual_points = [], []
    soft_cv_accum, hard_cv_accum, n_gap = 0.0, 0.0, 0

    for batch in val_batches:
        density = batch["target_density"].to(device)
        n = density.shape[0]
        offs = sample_offsets(fm, conditioner, batch, device, n_samples=n, grid_size=grid_size,
                              ode_steps=ode_steps, ode_method=ode_method, t_start=t_start,
                              show_tqdm=show_tqdm)
        pts_np_list = offsets_to_points_np(offs)
        images = batch["high_res"].cpu().numpy()
        for b in range(n):
            rows.append(hard_metrics_for_points(pts_np_list[b], images[b, 0]))
            if len(visual_points) < max_visual:
                visual_points.append(pts_np_list[b])
        if tau is not None:
            pts = offsets_to_coords(offs)
            rho, gxy = density_grid(density, loss_grid, device=device)
            soft_cv = soft_capacity_cv(pts, gxy, rho, tau)
            hard_cv = float(np.mean([r["voronoi_mass_cv"] for r in rows[-n:]]))
            soft_cv_accum += soft_cv; hard_cv_accum += hard_cv; n_gap += 1

    summary = summarize(rows)
    result = {"tag": tag, "n_samples": len(rows), "hard": summary}
    if n_gap > 0:
        soft_cv, hard_cv = soft_cv_accum / n_gap, hard_cv_accum / n_gap
        result["tau"] = float(tau)
        result["cap_gap"] = {"soft_cv": soft_cv, "hard_cv": hard_cv, "abs_gap": abs(soft_cv - hard_cv)}
    if gbn_bar is not None:
        cmp = {}
        for k, v in summary.items():
            bar = gbn_bar.get(k, {}).get("mean")
            if bar is not None:
                cmp[k] = {"model": v["mean"], "gbn": bar, "delta": v["mean"] - bar,
                          "model_better": v["mean"] < bar}
        result["vs_gbn"] = cmp
    grid_path = os.path.join(out_dir, f"visual_grid_{tag}.png")
    save_visual_grid(visual_points, grid_path, title=f"{tag} (hard-validated outputs)")
    result["visual_grid"] = grid_path
    return result


def load_val_batches(source, offsets, grid_size, batch_size, max_batches):
    from torch.utils.data import DataLoader
    from control_gt_free.DynamicStippleDataset import DynamicStippleDataset
    from control_gt_free.train_control import dynamic_collate
    ds = DynamicStippleDataset(source, offsets, grid_size=grid_size,
                               smart_init_features=False, sdf_features=False, preload_ram=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=dynamic_collate)
    batches = []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(b)
    return batches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Checkpoint from train_control.py (concat/spade)")
    ap.add_argument("--conditioning", choices=["concat", "spade"], default="concat",
                    help="Fallback if the checkpoint has no embedded gtfree_config")
    ap.add_argument("--base_config_path", default="config/GBN/config.json")
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--t-scale", type=float, default=1000.0)
    ap.add_argument("--fm-coupling", choices=["gaussian", "smartinit"], default="gaussian")
    ap.add_argument("--concat-smart-init-grid", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--source", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--out", required=True, help="Output dir for the visual grid + report")
    ap.add_argument("--gbn-bar", default=None, help="Bundle from precompute_gbn_bar.py")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=3)
    ap.add_argument("--ode-steps", type=int, default=50)
    ap.add_argument("--ode-method", default="euler")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    state = torch.load(args.ckpt, map_location="cpu")
    # Accept checkpoints from train_control.py (key "control_net" + "gtfree_config") or the
    # older {"conditioner","config"} format; fall back to CLI flags if no config is embedded.
    cfg = state.get("config") or state.get("gtfree_config")
    if cfg is None:
        cfg = {"conditioning": args.conditioning, "concat_smart_init_grid": args.concat_smart_init_grid,
               "base_config_path": args.base_config_path, "grid_size": args.grid_size,
               "t_scale": args.t_scale, "fm_coupling": args.fm_coupling}
    grid_size = int(cfg.get("grid_size", args.grid_size))

    conditioner = build_conditioner(cfg, device)
    sd = state.get("conditioner", state.get("control_net"))
    if sd is None:
        raise KeyError("checkpoint has neither 'conditioner' nor 'control_net' state dict")
    conditioner.load_state_dict(sd, strict=False)
    fm = FlowMatching(device=device, t_scale=float(cfg.get("t_scale", 1000.0)),
                      coupling=cfg.get("fm_coupling", "gaussian"))

    gbn_bar = None
    if args.gbn_bar:
        gbn_bar = torch.load(args.gbn_bar, map_location="cpu").get("gbn_hard_summary")

    batches = load_val_batches(args.source, args.offsets, grid_size, args.batch_size, args.max_batches)
    result = run_hard_validation(
        fm, conditioner, batches, device, grid_size, args.out, tag="model",
        gbn_bar=gbn_bar, ode_steps=args.ode_steps, ode_method=args.ode_method, show_tqdm=True,
    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "hard_report.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
