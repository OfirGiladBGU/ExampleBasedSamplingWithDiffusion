"""Evaluate Dynamic ControlNet V3 on one unique sample per image type.

Uses the same visualization/export logic style as overfit:
- source/target snapshots
- GT offset artifacts
- comparison panel via visualize_overfit_metrics

Dataset expected layout:
    DATA_ROOT/
      source/*.png
      target/*.png

Filename convention:
    gen_gray_<TYPE>_<RANDOM>_<INDEX>.png

Example:
    python control_v3/eval_types_test_batch.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    plt = None
    HAS_MPL = False

from data.Transforms import to_image_optimal_transport, to_pointset_optimal_transport
from control_v3.DynamicControlNet import DynamicControlNet, DynamicControlledDenoiser
from utils.Config import ParseSampleConfig
from utils.stippling_metrics import visualize_overfit_metrics

# ── defaults ────────────────────────────────────────────────────────
DATA_ROOT = "/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_1024_test_batch"
SOURCE_DIR = os.path.join(DATA_ROOT, "source")
TARGET_DIR = os.path.join(DATA_ROOT, "target")

CONFIG_PATH = "config/GBN/config.json"
BASE_CKPT = "config/GBN/model.ckpt"
CONTROL_CKPT = "control_v3/train_outputs/dynamic_controlnet_v3_ep99.pt"

OUTPUT_DIR = "control_v3/eval_types_outputs_ep99"
GRID_SIZE = 32
N_POINTS = GRID_SIZE ** 2
N_SAMPLES = 1
TIMESTEPS = 1000
RESAMPLE_JUMPS = 2
ENABLE_GECCO = True
DEVICE = "cuda"
EXPORT_GT_OFFSET = True


TYPE_SPECS = [
    # OLD versions
    ("Linear Gradient", "enable_linear", ["Linear_Gradient"]),
    ("Sinusoidal Gradient", "enable_sinusoidal", ["Sinusoidal_Gradient"]),
    ("Cosine Gradient", "enable_cosine", ["Cosine_Gradient"]),
    ("Radial Sinusoidal Gradient", "enable_radial_sinusoidal", ["Radial_Sinusoidal_Gradient"]),
    ("Radial Cosine Gradient", "enable_radial_cosine", ["Radial_Cosine_Gradient"]),
    # NEW versions
    ("Linear Gradient New", "enable_linear_new", ["Linear_Gradient_New"]),
    ("Sinusoidal Gradient New", "enable_sinusoidal_new", ["Sinusoidal_Gradient_New"]),
    ("Cosine Gradient New", "enable_cosine_new", ["Cosine_Gradient_New"]),
    ("Radial Sinusoidal Gradient New", "enable_radial_sinusoidal_new", ["Radial_Sinusoidal_Gradient_New"]),
    ("Radial Cosine Gradient New", "enable_radial_cosine_new", ["Radial_Cosine_Gradient_New"]),
    # Additional
    ("Wave", "enable_wave", ["Wave"]),
    ("Radial Wave", "enable_radial_wave", ["Radial_Wave"]),
    ("Noise", "enable_noise", ["Noise"]),
    ("Combined Shape", "enable_combined_shape", ["Combined_Shape"]),
]


def slugify(name):
    return name.lower().replace(" ", "_")


def extract_points_from_image(img_path, n_points):
    """Detect dot centroids in a stippled image -> (N, 2) in [0, 1]."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.uint8)

    inv = 255 - img_np
    binary = (inv > 127).astype(np.uint8)

    from scipy import ndimage
    labelled, n_labels = ndimage.label(binary)
    centroids = ndimage.center_of_mass(binary, labelled, range(1, n_labels + 1))

    h, w = img_np.shape
    pts = np.array([[cx / w, cy / h] for cy, cx in centroids], dtype=np.float64)

    rng = np.random.RandomState(42)
    if len(pts) > n_points:
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    elif len(pts) < n_points:
        deficit = n_points - len(pts)
        pts = np.vstack([pts, rng.rand(deficit, 2)])
        print(
            f"  WARNING: padded {deficit} random points "
            f"(only {len(pts) - deficit} detected)"
        )
    return pts


def load_condition(img_path, grid_size, device):
    """Load source image and return (high_res, target_density) tensors."""
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0

    high_res = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    target_density = F.interpolate(
        high_res, size=(grid_size, grid_size), mode="area"
    )
    return high_res, target_density


def export_gt_offset_artifacts(out_dir, gt_offsets):
    """Save GT offset diagnostics inside out_dir/gt_offset/."""
    gt_dir = os.path.join(out_dir, "gt_offset")
    os.makedirs(gt_dir, exist_ok=True)

    mag = np.sqrt(gt_offsets[0] ** 2 + gt_offsets[1] ** 2)
    if mag.max() > 0:
        mag_u8 = np.round((mag / mag.max()) * 255.0).astype(np.uint8)
    else:
        mag_u8 = np.zeros_like(mag, dtype=np.uint8)
    Image.fromarray(mag_u8).save(os.path.join(gt_dir, "gt_offsets_magnitude_32x32.png"))

    pts_grid = to_pointset_optimal_transport(gt_offsets)
    pts = pts_grid.reshape(2, -1).T

    n = gt_offsets.shape[-1]
    clipped = np.clip(pts, 0.0, 1.0 - 1e-12)
    ij = np.floor(clipped * n).astype(np.int64)
    counts = np.zeros((n, n), dtype=np.int32)
    for x_idx, y_idx in ij:
        counts[y_idx, x_idx] += 1

    Image.fromarray(((counts > 0).astype(np.uint8) * 255), mode="L").save(
        os.path.join(gt_dir, "gt_points_binary_32x32.png")
    )

    if HAS_MPL:
        dx, dy = gt_offsets[0], gt_offsets[1]
        yy, xx = np.mgrid[0:n, 0:n]
        fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
        q = ax.quiver(
            xx,
            yy,
            dx,
            dy,
            np.sqrt(dx * dx + dy * dy),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            cmap="viridis",
            width=0.004,
        )
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("grid x")
        ax.set_ylabel("grid y")
        ax.set_title("Offset Vector Field")
        fig.colorbar(q, ax=ax, label="|offset|")
        plt.tight_layout()
        plt.savefig(os.path.join(gt_dir, "offset_quiver.png"))
        plt.close()


def sample_from_model(diffusion, control_net, denoiser, high_res, target_density,
                      device, n_samples=1, timesteps=1000, resample_jumps=2):
    """Run reverse diffusion and return point sets + raw offsets."""
    controlled = DynamicControlledDenoiser(denoiser, control_net)
    controlled.set_condition(high_res, target_density)

    orig_model = diffusion.model
    diffusion.model = controlled
    diffusion.set_num_timesteps(timesteps)
    diffusion.eval()

    shape = [n_samples, 2, GRID_SIZE, GRID_SIZE]
    with torch.no_grad():
        if resample_jumps == 0:
            raw = diffusion.p_sample_loop(shape, img=None, cond=None,
                                          with_tqdm=True, with_sampling=True)
        else:
            img = diffusion.noise_fn(shape).to(device)
            for i in tqdm(reversed(range(diffusion.num_timesteps - 1)),
                          total=diffusion.num_timesteps - 1,
                          desc="sampling",
                          leave=False):
                t_tensor = torch.full((n_samples,), i, dtype=torch.int64, device=device)
                for u in range(resample_jumps + 1):
                    img = diffusion.p_sample(img, cond=None, t=t_tensor,
                                             clip_denoised=diffusion.sample_clip,
                                             with_sampling=True)
                    if u == resample_jumps or i == 0:
                        break
                    beta_i = diffusion.betas[i]
                    noise = torch.randn_like(img)
                    img = (1.0 - beta_i).sqrt() * img + beta_i.sqrt() * noise
            raw = img

    raw_np = raw.cpu().numpy()

    diffusion.model = orig_model
    diffusion.reset_timesteps()
    diffusion.train()

    pointsets = []
    for s in raw_np:
        ps = to_pointset_optimal_transport(s)
        ps = ps.reshape(ps.shape[0], np.prod(ps.shape[1:])).T
        pointsets.append(ps)
    return np.array(pointsets), raw_np


def parse_type_token(fname):
    """Extract type token from gen_gray_<TYPE>_<RAND>_<IDX>.png."""
    stem = os.path.splitext(fname)[0]
    if not stem.startswith("gen_gray_"):
        return None
    body = stem[len("gen_gray_"):]
    parts = body.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[:-2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--source-dir", default=SOURCE_DIR)
    parser.add_argument("--target-dir", default=TARGET_DIR)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--base-ckpt", default=BASE_CKPT)
    parser.add_argument("--control-ckpt", default=CONTROL_CKPT)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--resample-jumps", type=int, default=RESAMPLE_JUMPS)
    parser.add_argument("--enable-gecco", action=argparse.BooleanOptionalAction, default=ENABLE_GECCO)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--export-gt-offset", action=argparse.BooleanOptionalAction, default=EXPORT_GT_OFFSET)

    parser.add_argument("--enable-linear", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-sinusoidal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-cosine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-radial-sinusoidal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-radial-cosine", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--enable-linear-new", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-sinusoidal-new", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-cosine-new", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-radial-sinusoidal-new", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-radial-cosine-new", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--enable-wave", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-radial-wave", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-combined-shape", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        raise FileNotFoundError(f"Missing source dir: {args.source_dir}")
    if not os.path.isdir(args.target_dir):
        raise FileNotFoundError(f"Missing target dir: {args.target_dir}")
    if not os.path.isfile(args.control_ckpt):
        raise FileNotFoundError(f"Missing control checkpoint: {args.control_ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    source_files = sorted([f for f in os.listdir(args.source_dir) if f.lower().endswith(".png")])
    type_to_fname = {}
    for fname in source_files:
        t = parse_type_token(fname)
        if t is None:
            continue
        if t not in type_to_fname:
            type_to_fname[t] = fname

    enabled_types = []
    for display_name, arg_name, type_tokens in TYPE_SPECS:
        if not getattr(args, arg_name):
            continue
        enabled_types.append((display_name, type_tokens))

    selected = []
    missing = []
    for display_name, tokens in enabled_types:
        found = None
        for token in tokens:
            if token in type_to_fname:
                found = type_to_fname[token]
                break
        if found is None:
            missing.append(display_name)
        else:
            selected.append((display_name, found))

    print(f"Enabled types: {len(enabled_types)}")
    print(f"Found samples: {len(selected)}")
    if missing:
        print("Missing types in this dataset:")
        for name in missing:
            print(f"  - {name}")

    if not selected:
        raise RuntimeError("No enabled type was found in dataset")

    device = torch.device(args.device)

    diffusion = ParseSampleConfig(args.config)
    diffusion.load_state_dict(torch.load(args.base_ckpt, map_location="cpu")["diffu"])
    diffusion.to(device)

    denoiser = diffusion.model
    for p in denoiser.parameters():
        p.requires_grad = False
    denoiser.eval()

    control_net = DynamicControlNet(
        denoiser,
        grid_size=args.grid_size,
        enable_gecco=args.enable_gecco,
    ).to(device)
    ctrl_state = torch.load(args.control_ckpt, map_location="cpu")
    control_net.load_state_dict(ctrl_state["control_net"])
    control_net.eval()

    print(f"Loaded checkpoint: {args.control_ckpt}")
    print(f"GECCO enabled   : {args.enable_gecco}")

    summary_rows = []
    for display_name, fname in tqdm(selected, desc="types", total=len(selected)):
        source_path = os.path.join(args.source_dir, fname)
        target_path = os.path.join(args.target_dir, fname)
        if not os.path.exists(target_path):
            print(f"Skipping {display_name}: missing target {target_path}")
            continue

        type_slug = slugify(display_name)
        out_dir = os.path.join(args.out_dir, type_slug)
        os.makedirs(out_dir, exist_ok=True)

        source_np = np.array(Image.open(source_path).convert("L"))
        target_np = np.array(Image.open(target_path).convert("L"))

        Image.fromarray(source_np).save(os.path.join(out_dir, "source.png"))
        Image.fromarray(target_np).save(os.path.join(out_dir, "target.png"))

        gt_points = extract_points_from_image(target_path, N_POINTS)
        gt_offsets = to_image_optimal_transport(gt_points)
        np.save(os.path.join(out_dir, "gt_offsets.npy"), gt_offsets)

        if args.export_gt_offset:
            export_gt_offset_artifacts(out_dir, gt_offsets)

        high_res, target_density = load_condition(source_path, args.grid_size, device)
        pred_points, raw_offsets = sample_from_model(
            diffusion=diffusion,
            control_net=control_net,
            denoiser=denoiser,
            high_res=high_res,
            target_density=target_density,
            device=device,
            n_samples=args.n_samples,
            timesteps=args.timesteps,
            resample_jumps=args.resample_jumps,
        )

        np.save(os.path.join(out_dir, "pred_points.npy"), pred_points)
        np.save(os.path.join(out_dir, "pred_offsets_raw.npy"), raw_offsets)

        vis_path = os.path.join(out_dir, "comparison.png")
        visualize_overfit_metrics(
            source_img=source_np,
            target_img=target_np,
            gt_points=gt_points,
            pred_pointsets=list(pred_points),
            save_path=vis_path,
            step=None,
            gt_offsets=gt_offsets,
        )

        summary_rows.append((display_name, fname, out_dir))

    summary_txt = os.path.join(args.out_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        for display_name, fname, out_dir in summary_rows:
            f.write(f"{display_name}\t{fname}\t{out_dir}\n")

    print("Done.")
    print(f"Saved outputs to: {args.out_dir}")
    print(f"Summary file: {summary_txt}")


if __name__ == "__main__":
    main()
