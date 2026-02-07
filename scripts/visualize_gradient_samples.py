"""
Visualize sample point sets from the gradient dataset.
Shows both raw point sets and their optimal transport transformed versions.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def visualize_gradient_samples(dataset_file="data/datasets/test_gradient_dataset.hdf5", num_samples=5):
    """
    Load and visualize samples from gradient dataset.
    
    Args:
        dataset_file: Path to HDF5 dataset
        num_samples: Number of samples to display
    """
    
    if not Path(dataset_file).exists():
        print(f"Dataset file not found: {dataset_file}")
        print("Create it first with: python build_gradient_dataset.py")
        return
    
    with h5py.File(dataset_file, 'r') as f:
        # Get first group and scale
        group_name = list(f.keys())[0]
        group = f[group_name]
        scale_name = list(group.keys())[0]
        scale_group = group[scale_name]
        
        # Load data
        raw_points = np.array(scale_group['data'])  # (N, num_points, 2)
        transformed = np.array(scale_group['data_t'])  # (N, 2, h, w)
        
        num_points = raw_points.shape[1]
        h, w = transformed.shape[2:]
        
        print(f"Group: {group_name}, Scale: {scale_name}")
        print(f"Raw points shape: {raw_points.shape}")
        print(f"Transformed shape: {transformed.shape}")
        print(f"Points per sample: {num_points}")
        print(f"Transformed grid: {h}×{w}")
        print()
        
        # Select samples
        num_samples = min(num_samples, len(raw_points))
        indices = np.linspace(0, len(raw_points) - 1, num_samples, dtype=int)
        
        # Create figure with two rows
        fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Gradient Point Set Samples\n({num_points} points, normalized to [0,1])', 
                     fontsize=14, fontweight='bold')
        
        for col, idx in enumerate(indices):
            points = raw_points[idx]  # (num_points, 2)
            trans = transformed[idx]  # (2, h, w)
            
            # Plot raw points
            ax_raw = axes[0, col]
            ax_raw.scatter(points[:, 0], points[:, 1], s=2, alpha=0.6, c='black')
            ax_raw.set_xlim(-0.05, 1.05)
            ax_raw.set_ylim(-0.05, 1.05)
            ax_raw.set_aspect('equal')
            ax_raw.invert_yaxis()  # Match image coordinates
            ax_raw.set_title(f'Sample {idx+1}\n(Raw Points)', fontsize=10)
            ax_raw.grid(True, alpha=0.2)
            
            # Plot transformed (as image)
            ax_trans = axes[1, col]
            # Combine channels for visualization
            trans_viz = np.sqrt(np.sum(trans**2, axis=0))  # Magnitude
            im = ax_trans.imshow(trans_viz, cmap='hot', origin='upper')
            ax_trans.set_title(f'Sample {idx+1}\n(OT Transformed)', fontsize=10)
            ax_trans.set_xlabel(f'Grid: {h}×{w}')
            
            # Add grid lines for transformed
            for i in np.linspace(0, h, 5):
                ax_trans.axhline(i, color='cyan', alpha=0.2, linewidth=0.5)
            for i in np.linspace(0, w, 5):
                ax_trans.axvline(i, color='cyan', alpha=0.2, linewidth=0.5)
        
        plt.tight_layout()
        
        # Save and show
        plt.savefig('outputs/gradient_samples_visualization.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved to: outputs/gradient_samples_visualization.png")
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize gradient samples")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/test_gradient_dataset.hdf5",
        help="HDF5 dataset file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to visualize"
    )
    
    args = parser.parse_args()
    visualize_gradient_samples(args.dataset, args.samples)
