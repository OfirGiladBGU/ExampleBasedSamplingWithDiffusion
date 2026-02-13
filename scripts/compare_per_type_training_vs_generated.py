#!/usr/bin/env python
"""Compare per-type training samples with generated samples."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from data.Transforms import to_pointset_optimal_transport
from utils.Config import ParseSampleConfig

TYPES = [
    "Radial_Cosine_Gradient",
    "Noise",
    "Wave",
    "Radial_Wave",
    "Cosine_Gradient",
    "Sinusoidal_Gradient",
    "Combined_Shape",
    "Linear_Gradient",
    "Radial_Sinusoidal_Gradient",
]


def _sanitize_name(name: str) -> str:
    return name.replace(" ", "_")


def _convert_to_pointset(sample: np.ndarray) -> np.ndarray:
    points = to_pointset_optimal_transport(sample)
    points = points.reshape(points.shape[0], -1).T
    return points


def load_training_samples(dataset_file: str):
    with h5py.File(dataset_file, "r") as f:
        group_name = list(f.keys())[0]
        scale_name = list(f[group_name].keys())[0]
        scale_group = f[group_name][scale_name]
        data = np.array(scale_group["data"])  # (N, P, 2)
        prop = np.array(scale_group["prop"])  # (N, 9)
        data_t = np.array(scale_group["data_t"])  # (N, 2, H, W)
    return data, prop, data_t.shape[1:]


def generate_samples(diffu, cond_vec, sample_shape, num_samples, device, timesteps):
    diffu.set_num_timesteps(timesteps)
    cond = torch.from_numpy(cond_vec.astype(np.float32)).to(device)
    cond = cond.repeat(num_samples, 1)

    with torch.no_grad():
        generated = diffu.p_sample_loop(
            [num_samples, *sample_shape],
            img=None,
            cond=cond,
            with_tqdm=True,
            with_sampling=True,
        )

    generated = generated.cpu().numpy()
    return [_convert_to_pointset(sample) for sample in generated]


def plot_comparison(output_path, train_samples, train_indices, gen_samples, title):
    num_samples = len(train_samples)
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    fig.suptitle(
        f"Training Samples (Top) vs Generated Samples (Bottom)\n{title}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for col in range(num_samples):
        ax_train = axes[0, col]
        train_sample = train_samples[col]
        ax_train.scatter(
            train_sample[:, 0],
            train_sample[:, 1],
            s=3,
            alpha=0.7,
            c="darkgreen",
            edgecolors="none",
        )
        ax_train.set_xlim(-0.05, 1.05)
        ax_train.set_ylim(-0.05, 1.05)
        ax_train.set_aspect("equal")
        ax_train.invert_yaxis()
        ax_train.set_title(
            f"Training #{train_indices[col] + 1}\n({len(train_sample)} pts)",
            fontsize=10,
            fontweight="bold",
        )
        ax_train.grid(True, alpha=0.2)
        if col == 0:
            ax_train.set_ylabel("TRAINING\nDATA", fontsize=12, fontweight="bold", rotation=0, labelpad=50)

        ax_gen = axes[1, col]
        gen_sample = gen_samples[col]
        ax_gen.scatter(
            gen_sample[:, 0],
            gen_sample[:, 1],
            s=3,
            alpha=0.7,
            c="darkblue",
            edgecolors="none",
        )
        ax_gen.set_xlim(-0.05, 1.05)
        ax_gen.set_ylim(-0.05, 1.05)
        ax_gen.set_aspect("equal")
        ax_gen.invert_yaxis()
        ax_gen.set_title(
            f"Generated #{col + 1}\n({len(gen_sample)} pts)",
            fontsize=10,
            fontweight="bold",
        )
        ax_gen.grid(True, alpha=0.2)
        if col == 0:
            ax_gen.set_ylabel("GENERATED\nBY MODEL", fontsize=12, fontweight="bold", rotation=0, labelpad=50)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare per-type training vs generated samples")
    parser.add_argument("--dataset", default="db/gradient_dataset_balanced.hdf5", help="Training dataset HDF5 file")
    parser.add_argument("--config", default="outputs/models/gradient_models_balanced/config.json", help="Model config file")
    parser.add_argument("--model", default="outputs/models/gradient_models_balanced/model.ckpt", help="Checkpoint file")
    parser.add_argument("--output", default="outputs/results_balanced/compare_per_type", help="Output directory")
    parser.add_argument("--samples", type=int, default=5, help="Samples per type")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, prop, sample_shape = load_training_samples(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffu = ParseSampleConfig(args.config)
    checkpoint = torch.load(args.model, map_location=device)
    diffu.load_state_dict(checkpoint["diffu"])
    diffu.to(device)
    diffu.eval()

    for type_idx, gtype in enumerate(TYPES):
        indices = np.where(prop[:, type_idx] > 0.5)[0]
        if len(indices) == 0:
            print(f"No training samples found for {gtype}, skipping.")
            continue

        replace = len(indices) < args.samples
        chosen = np.random.choice(indices, args.samples, replace=replace)
        train_samples = [data[idx] for idx in chosen]

        cond_vec = np.zeros(9, dtype=np.float32)
        cond_vec[type_idx] = 1.0

        gen_samples = generate_samples(
            diffu,
            cond_vec,
            sample_shape,
            args.samples,
            device,
            args.timesteps,
        )

        output_path = output_dir / f"compare_{_sanitize_name(gtype)}.png"
        plot_comparison(output_path, train_samples, chosen, gen_samples, gtype)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
