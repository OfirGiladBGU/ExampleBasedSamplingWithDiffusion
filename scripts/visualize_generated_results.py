"""
Visualize the generated samples from the trained diffusion model.
Shows 5 point pattern samples with statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_generated_samples(samples_file="outputs/results_gradients/samples/gradient_samples.npy", output_file="outputs/generated_results_visualization.png"):
    """
    Load and visualize generated point pattern samples.
    
    Args:
        samples_file: Path to numpy file with generated samples
        output_file: Where to save the visualization
    """
    
    if not Path(samples_file).exists():
        print(f"❌ Samples file not found: {samples_file}")
        print("Generate samples first using sample.py")
        return
    
    # Load generated samples
    samples = np.load(samples_file, allow_pickle=True)
    
    print("=" * 60)
    print("📊 GENERATED SAMPLES ANALYSIS")
    print("=" * 60)
    print(f"Number of samples: {len(samples)}")
    print(f"Points per sample: {samples[0].shape[0]}")
    print(f"Coordinate dimensions: {samples[0].shape[1]}")
    print()
    
    # Analyze each sample
    for i, sample in enumerate(samples):
        x_min, y_min = sample.min(axis=0)
        x_max, y_max = sample.max(axis=0)
        x_mean, y_mean = sample.mean(axis=0)
        print(f"Sample {i+1}:")
        print(f"  X range: [{x_min:.4f}, {x_max:.4f}]")
        print(f"  Y range: [{y_min:.4f}, {y_max:.4f}]")
        print(f"  Centroid: ({x_mean:.4f}, {y_mean:.4f})")
    
    print("=" * 60)
    print()
    
    # Create visualization with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Generated Point Pattern Samples\n(From Trained Diffusion Model)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (ax, sample) in enumerate(zip(axes, samples)):
        # Plot points
        ax.scatter(sample[:, 0], sample[:, 1], 
                  s=3, alpha=0.7, c='darkblue', edgecolors='none')
        
        # Styling
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match image coordinates
        ax.set_title(f'Sample {idx+1}\n({len(sample)} points)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel('X', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Y', fontsize=10)
        
        # Add frame
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('gray')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved to: {output_file}")
    
    # Also save individual samples
    individual_dir = Path(samples_file).parent / "individual_samples"
    individual_dir.mkdir(exist_ok=True)
    
    for idx, sample in enumerate(samples):
        fig_ind, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(sample[:, 0], sample[:, 1], 
                  s=5, alpha=0.7, c='darkblue', edgecolors='none')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Generated Sample {idx+1} ({len(sample)} points)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # Add statistics text box
        stats_text = f"Points: {len(sample)}\n"
        stats_text += f"X: [{sample[:, 0].min():.3f}, {sample[:, 0].max():.3f}]\n"
        stats_text += f"Y: [{sample[:, 1].min():.3f}, {sample[:, 1].max():.3f}]"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        output_path = individual_dir / f"sample_{idx+1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_ind)
    
    print(f"✅ Individual samples saved to: {individual_dir}/")
    print()
    
    # Display the main figure
    plt.show()
    
    return samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize generated samples")
    parser.add_argument(
        "--samples",
        type=str,
        default="outputs/results_gradients/samples/gradient_samples.npy",
        help="Path to generated samples .npy file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated_results_visualization.png",
        help="Output visualization file"
    )
    
    args = parser.parse_args()
    visualize_generated_samples(args.samples, args.output)
