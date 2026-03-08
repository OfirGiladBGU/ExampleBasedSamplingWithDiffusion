"""Visualize OT offset tensors as vector fields.

Usage:
    python control_v1/visualize_offsets.py \
        --input control_v1/overfit_outputs/target1_test/gt_offsets.npy
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to offsets .npy (shape (2,H,W))")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <input_dir>/gt_offset)")
    parser.add_argument("--stride", type=int, default=1, help="Subsample arrows every N cells")
    parser.add_argument("--scale", type=float, default=1.0, help="Arrow scale factor")
    args = parser.parse_args()

    in_path = Path(args.input)
    off = np.load(in_path)
    if off.ndim != 3 or off.shape[0] != 2:
        raise ValueError(f"Expected shape (2,H,W), got {off.shape}")

    h, w = int(off.shape[1]), int(off.shape[2])
    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent / "gt_offset"
    out_dir.mkdir(parents=True, exist_ok=True)

    dx = off[0]
    dy = off[1]
    mag = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)  # [-pi, pi]

    # Save scalar arrays for optional downstream analysis.
    np.save(out_dir / "offset_dx.npy", dx)
    np.save(out_dir / "offset_dy.npy", dy)
    np.save(out_dir / "offset_magnitude.npy", mag)
    np.save(out_dir / "offset_angle_rad.npy", ang)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for vector visualization") from exc

    # 1) Quiver: one arrow per grid cell, colored by magnitude.
    yy, xx = np.mgrid[0:h, 0:w]
    s = max(1, int(args.stride))
    xx_s = xx[::s, ::s]
    yy_s = yy[::s, ::s]
    dx_s = dx[::s, ::s]
    dy_s = dy[::s, ::s]
    mag_s = mag[::s, ::s]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    q = ax.quiver(
        xx_s,
        yy_s,
        dx_s * args.scale,
        dy_s * args.scale,
        mag_s,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        cmap="viridis",
        width=0.004,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Offset Vector Field (quiver)")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    cbar = fig.colorbar(q, ax=ax)
    cbar.set_label("|offset|")
    fig.tight_layout()
    fig.savefig(out_dir / "offset_quiver.png")
    plt.close(fig)

    # 2) Direction+magnitude map: hue = angle, value = normalized magnitude.
    hue = (ang + np.pi) / (2.0 * np.pi)
    sat = np.ones_like(hue)
    val = mag / (mag.max() + 1e-12)
    hsv = np.stack([hue, sat, val], axis=-1)

    import matplotlib.colors as mcolors

    rgb = mcolors.hsv_to_rgb(hsv)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(rgb)
    ax.set_title("Offset Direction/Magnitude (HSV)")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "offset_hsv.png")
    plt.close(fig)

    summary = [
        f"input={in_path}",
        f"shape={off.shape}",
        f"min_mag={float(mag.min()):.6f}",
        f"max_mag={float(mag.max()):.6f}",
        f"mean_mag={float(mag.mean()):.6f}",
        f"stride={s}",
        f"scale={float(args.scale):.3f}",
    ]
    (out_dir / "offset_vector_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")

    print("Saved visualization files to:", out_dir)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
