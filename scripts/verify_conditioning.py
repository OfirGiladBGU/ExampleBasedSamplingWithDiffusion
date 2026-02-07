"""
Verify that the conditioned dataset has proper type-based conditioning.
Checks the structure and shows sample conditioning vectors.
"""

import h5py
import numpy as np
import sys
from pathlib import Path

# Gradient type names (must match build_gradient_dataset.py)
GRADIENT_TYPES = [
    'Radial_Cosine_Gradient',
    'Noise',
    'Wave',
    'Radial_Wave',
    'Cosine_Gradient',
    'Sinusoidal_Gradient',
    'Combined_Shape',
    'Linear_Gradient',
    'Radial_Sinusoidal_Gradient'
]


def verify_conditioning_dataset(hdf5_file):
    """
    Verify that conditioning is properly stored in HDF5 file.
    """
    
    if not Path(hdf5_file).exists():
        print(f"❌ File not found: {hdf5_file}")
        return False
    
    print(f"📊 Analyzing: {hdf5_file}")
    print("=" * 80)
    
    with h5py.File(hdf5_file, 'r') as f:
        if 'gradients' not in f:
            print("❌ No 'gradients' group found")
            return False
        
        group = f['gradients']
        scales = list(group.keys())
        print(f"Found {len(scales)} scale groups: {scales}")
        print()
        
        for scale_name in sorted(scales):
            scale_group = group[scale_name]
            
            if 'prop' not in scale_group:
                print(f"❌ {scale_name}: No 'prop' dataset")
                continue
            
            prop_data = np.array(scale_group['prop'])
            
            print(f"📈 {scale_name}:")
            print(f"   Shape: {prop_data.shape}")
            print(f"   Samples: {prop_data.shape[0]}")
            print(f"   Conditioning dims: {prop_data.shape[1]}")
            
            if prop_data.shape[1] == 9:
                print(f"   ✅ Correct: 9 dimensions (one per gradient type)")
            elif prop_data.shape[1] == 1:
                print(f"   ⚠️  Warning: Only 1 dimension (conditioning may not be enabled)")
            else:
                print(f"   ❌ Error: {prop_data.shape[1]} dimensions (expected 9 or 1)")
            
            # Analyze conditioning distribution
            non_zero_rows = np.any(prop_data != 0, axis=1).sum()
            print(f"   Non-zero samples: {non_zero_rows}/{prop_data.shape[0]}")
            
            # Show the type distribution
            if prop_data.shape[1] == 9:
                # Count which type each sample belongs to
                type_counts = np.zeros(9, dtype=int)
                for i in range(prop_data.shape[0]):
                    # Find which dimension is 1 (one-hot encoding)
                    idx = np.where(prop_data[i] == 1.0)[0]
                    if len(idx) > 0:
                        type_counts[idx[0]] += 1
                
                print(f"   Type distribution:")
                for type_idx, type_name in enumerate(GRADIENT_TYPES):
                    count = type_counts[type_idx]
                    if count > 0:
                        pct = 100.0 * count / prop_data.shape[0]
                        print(f"      [{type_idx}] {type_name}: {count} ({pct:.1f}%)")
            
            # Show sample conditioning vectors
            print(f"   Sample conditioning vectors:")
            for i in range(min(3, prop_data.shape[0])):
                vec = prop_data[i]
                if np.any(vec != 0):
                    type_idx = np.argmax(vec)
                    print(f"      Sample {i}: {vec} → {GRADIENT_TYPES[type_idx]}")
                else:
                    print(f"      Sample {i}: {vec} → (unconditioned)")
            
            if prop_data.shape[0] > 3:
                print(f"      ... and {prop_data.shape[0] - 3} more samples")
            
            print()
    
    print("=" * 80)
    print("✅ Dataset verification complete!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify conditioning in HDF5 dataset")
    parser.add_argument(
        "--file",
        type=str,
        default="data/datasets/gradient_dataset_conditioned_test.hdf5",
        help="HDF5 file to verify"
    )
    
    args = parser.parse_args()
    verify_conditioning_dataset(args.file)
