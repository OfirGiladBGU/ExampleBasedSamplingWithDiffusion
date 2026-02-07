"""
Compare training samples with generated samples side-by-side.
This helps visualize what the model learned.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path


def compare_samples(
    dataset_file="data/datasets/gradient_dataset.hdf5",
    generated_file="outputs/results_gradients/samples/gradient_samples.npy",
    num_samples=5,
    output_file="outputs/comparison_training_vs_generated.png"
):
    """
    Show training samples vs generated samples for comparison.
    
    Args:
        dataset_file: HDF5 training dataset
        generated_file: Generated samples numpy file
        num_samples: Number of samples to compare
    """
    
    # Load generated samples
    if not Path(generated_file).exists():
        print(f"❌ Generated samples not found: {generated_file}")
        return
    
    generated = np.load(generated_file, allow_pickle=True)
    
    # Load original training data
    training_samples = None
    if Path(dataset_file).exists():
        with h5py.File(dataset_file, 'r') as f:
            group_name = list(f.keys())[0]
            scale_name = list(f[group_name].keys())[0]
            training_samples = np.array(f[group_name][scale_name]['data'])
            print(f"✅ Loaded {len(training_samples)} training samples from {dataset_file}")
    else:
        print(f"⚠️  Training dataset not found: {dataset_file}")
        print("Will only show generated samples.")
    
    # Create comparison figure
    if training_samples is not None:
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        fig.suptitle('Training Samples (Top) vs Generated Samples (Bottom)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Select random training samples
        train_indices = np.random.choice(len(training_samples), num_samples, replace=False)
        
        for col in range(num_samples):
            # Plot training sample (top row)
            ax_train = axes[0, col]
            train_sample = training_samples[train_indices[col]]
            ax_train.scatter(train_sample[:, 0], train_sample[:, 1],
                           s=3, alpha=0.7, c='darkgreen', edgecolors='none')
            ax_train.set_xlim(-0.05, 1.05)
            ax_train.set_ylim(-0.05, 1.05)
            ax_train.set_aspect('equal')
            ax_train.invert_yaxis()
            ax_train.set_title(f'Training #{train_indices[col]+1}\n({len(train_sample)} pts)', 
                             fontsize=11, fontweight='bold')
            ax_train.grid(True, alpha=0.2)
            if col == 0:
                ax_train.set_ylabel('TRAINING\nDATA', fontsize=12, fontweight='bold', rotation=0, labelpad=50)
            
            # Plot generated sample (bottom row)
            ax_gen = axes[1, col]
            if col < len(generated):
                gen_sample = generated[col]
                ax_gen.scatter(gen_sample[:, 0], gen_sample[:, 1],
                             s=3, alpha=0.7, c='darkblue', edgecolors='none')
                ax_gen.set_xlim(-0.05, 1.05)
                ax_gen.set_ylim(-0.05, 1.05)
                ax_gen.set_aspect('equal')
                ax_gen.invert_yaxis()
                ax_gen.set_title(f'Generated #{col+1}\n({len(gen_sample)} pts)', 
                               fontsize=11, fontweight='bold')
                ax_gen.grid(True, alpha=0.2)
                if col == 0:
                    ax_gen.set_ylabel('GENERATED\nBY MODEL', fontsize=12, fontweight='bold', rotation=0, labelpad=50)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"✅ Comparison saved to: {output_file}")
        
    else:
        # Only show generated samples
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        fig.suptitle('Generated Point Pattern Samples', 
                     fontsize=16, fontweight='bold')
        
        for col in range(min(num_samples, len(generated))):
            ax = axes[col]
            gen_sample = generated[col]
            ax.scatter(gen_sample[:, 0], gen_sample[:, 1],
                      s=3, alpha=0.7, c='darkblue', edgecolors='none')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f'Sample {col+1} ({len(gen_sample)} points)', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"✅ Visualization saved to: {output_file}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("📊 STATISTICS COMPARISON")
    print("=" * 60)
    
    if training_samples is not None:
        train_densities = [len(s) / ((s.max(axis=0) - s.min(axis=0)).prod()) 
                          for s in training_samples[:100]]
        print(f"Training samples: {len(training_samples)} total")
        print(f"  Points per sample: {training_samples[0].shape[0]}")
        print(f"  Avg density: {np.mean(train_densities):.1f} points/unit²")
    
    gen_densities = [len(s) / ((s.max(axis=0) - s.min(axis=0)).prod()) 
                     for s in generated]
    print(f"\nGenerated samples: {len(generated)} total")
    print(f"  Points per sample: {generated[0].shape[0]}")
    print(f"  Avg density: {np.mean(gen_densities):.1f} points/unit²")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare training vs generated samples")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/gradient_dataset.hdf5",
        help="Training dataset HDF5 file"
    )
    parser.add_argument(
        "--generated",
        type=str,
        default="outputs/results_gradients/samples/gradient_samples.npy",
        help="Generated samples numpy file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to compare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/comparison_training_vs_generated.png",
        help="Output comparison image path"
    )
    
    args = parser.parse_args()
    compare_samples(args.dataset, args.generated, args.samples, args.output)
