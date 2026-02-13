"""
Build HDF5 database from gradient point pattern images.

This script:
1. Loads target images (512x512 point patterns)
2. Extracts black point coordinates using threshold
3. Normalizes to [0, 1] range
4. Groups by type (from filename) and number of points
5. Creates HDF5 database with conditioning on gradient type
"""

import numpy as np
import os
import re
from PIL import Image
from tqdm import tqdm
from data.DatasetBuilder import QMCDatabaseBuilder
from data.Transforms import to_image_optimal_transport

# Gradient types in order of appearance
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
TYPE_TO_INDEX = {gtype: i for i, gtype in enumerate(GRADIENT_TYPES)}
NUM_TYPES = len(GRADIENT_TYPES)


def extract_points_from_image(image_path, threshold=128):
    """
    Extract point coordinates from binary image.
    
    Args:
        image_path: Path to PNG image (512x512)
        threshold: Pixel value threshold (0-255). Pixels below this are "black" points
    
    Returns:
        np.array of shape (N, 2) with point coordinates in [0, 1]
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Handle both 1-bit and 8-bit images
    if img_array.dtype == np.bool_:
        # 1-bit image: False=black (points), True=white
        points_mask = ~img_array
    else:
        # 8-bit image: threshold
        points_mask = img_array < threshold
    
    # Get coordinates of black pixels
    y_coords, x_coords = np.where(points_mask)
    
    if len(x_coords) == 0:
        return np.array([]).reshape(0, 2)
    
    # Normalize to [0, 1]
    num_points = len(x_coords)
    h, w = img_array.shape
    
    points = np.stack([
        x_coords / float(w),
        y_coords / float(h)
    ], axis=1)
    
    return points.astype(np.float32)


def extract_gradient_type(filename):
    """
    Extract gradient type from filename.
    
    Filename format: gen_gray_<type>_<number>_<number>.png
    
    Args:
        filename: Filename with gradient type
    
    Returns:
        Gradient type string, or None if not recognized
    """
    # Extract the part between 'gen_gray_' and the trailing numbers
    match = re.match(r'gen_gray_(.+?)_\d+_\d+\.png', filename)
    if match:
        return match.group(1)
    return None


def gradient_type_to_conditioning(gtype):
    """
    Convert gradient type to one-hot conditioning vector.
    
    Args:
        gtype: Gradient type string (e.g., 'Linear_Gradient')
    
    Returns:
        One-hot vector of shape (NUM_TYPES,)
    """
    if gtype not in TYPE_TO_INDEX:
        print(f"Warning: Unknown gradient type '{gtype}', using zeros")
        return np.zeros(NUM_TYPES, dtype=np.float32)
    
    vec = np.zeros(NUM_TYPES, dtype=np.float32)
    vec[TYPE_TO_INDEX[gtype]] = 1.0
    return vec


def properties_func(group_name, points):
    """
    Return conditioning values (empty for unconditional model).
    """
    return []


def properties_func_classified(group_name, points):
    """
    Return conditioning based on class (e.g., size category).
    """
    num_points = points.shape[0]
    if num_points < 500:
        return [1, 0, 0]  # Small
    elif num_points < 1500:
        return [0, 1, 0]  # Medium
    else:
        return [0, 0, 1]  # Large


def normalize_point_count(num_points, target_sizes=[1024, 2025, 4096, 4096]):
    """
    Find the closest valid number of points (perfect square for 2D).
    
    Args:
        num_points: Current number of points
        target_sizes: List of valid sizes (perfect powers)
    
    Returns:
        Closest valid size and subsample ratio needed
    """
    # Use 32x32 = 1024 which works better with U-Net architecture
    target_sizes = [1024, 4096]  # 32^2, 64^2
    closest = min(target_sizes, key=lambda x: abs(x - num_points))
    ratio = closest / num_points
    return closest, ratio


def build_gradient_dataset(
    source_dir,
    output_file="db/gradient_dataset.hdf5",
    max_samples=None,
    threshold=128,
    use_conditioning=True,
    target_point_sizes=[1024, 2025]
):
    """
    Build HDF5 dataset from gradient images with type-based conditioning.
    
    Args:
        source_dir: Directory containing PNG images (gen_gray_<type>_*.png)
        output_file: Output HDF5 file path
        max_samples: Limit number of samples (None = all)
        threshold: Pixel threshold for point detection
        use_conditioning: Whether to add conditioning based on gradient type
        target_point_sizes: Valid point counts (perfect squares for 2D)
    """
    
    # Collect all image files
    image_files = []
    for fname in sorted(os.listdir(source_dir)):
        if fname.endswith('.png') and fname.startswith('gen_gray_'):
            image_files.append((fname, os.path.join(source_dir, fname)))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"Found {len(image_files)} images")
    
    # Count types
    type_counts = {}
    for fname, _ in image_files:
        gtype = extract_gradient_type(fname)
        if gtype:
            type_counts[gtype] = type_counts.get(gtype, 0) + 1
    
    print(f"\nGradient type distribution:")
    for gtype in GRADIENT_TYPES:
        count = type_counts.get(gtype, 0)
        print(f"  {gtype}: {count}")
    
    # Group by number of points (after normalization) and optionally by type
    # Structure: groups[num_points] = [(points, conditioning), ...]
    groups = {}
    
    for fname, image_path in tqdm(image_files, desc="Extracting points"):
        gtype = extract_gradient_type(fname)
        if gtype is None:
            print(f"Warning: Could not extract type from {fname}")
            continue
        
        points = extract_points_from_image(image_path, threshold=threshold)
        num_points = points.shape[0]
        
        if num_points == 0:
            continue
        
        # Normalize to valid point count
        target_count, ratio = normalize_point_count(num_points, target_point_sizes)
        
        if ratio < 1.0:
            # Need to subsample
            n_samples = int(ratio * len(points))
            indices = np.random.choice(len(points), n_samples, replace=False)
            points = points[indices]
        
        num_points = points.shape[0]
        
        if num_points not in groups:
            groups[num_points] = []
        
        # Get conditioning vector for this type
        if use_conditioning:
            conditioning = gradient_type_to_conditioning(gtype)
        else:
            conditioning = np.zeros(NUM_TYPES, dtype=np.float32)
        
        groups[num_points].append((points, conditioning))
    
    print(f"\nPoint set distribution:")
    for num_points in sorted(groups.keys()):
        print(f"  {num_points} points: {len(groups[num_points])} samples")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to HDF5 using QMCDatabaseBuilder format
    import h5py
    
    with h5py.File(output_file, 'w') as f:
        group = f.create_group('gradients')
        
        for num_points in tqdm(sorted(groups.keys()), desc="Writing HDF5"):
            shape_group = group.create_group(f'scale_{num_points}')
            
            # Separate points and conditioning
            point_sets = np.stack([p[0] for p in groups[num_points]])  # (N, N_points, 2)
            conditions = np.stack([p[1] for p in groups[num_points]])  # (N, NUM_TYPES)
            
            # Transform to image space using optimal transport
            transformed = np.stack([
                to_image_optimal_transport(points)
                for points in point_sets
            ])
            
            # Store data
            shape_group.create_dataset('data', data=point_sets)
            shape_group.create_dataset('data_t', data=transformed)
            shape_group.create_dataset('prop', data=conditions)
    
    print(f"\nDataset saved to {output_file}")
    print(f"Conditioning dimensions: {NUM_TYPES} (one-hot encoded gradient types)")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build gradient dataset with type-based conditioning")
    parser.add_argument(
        "--source",
        type=str,
        default="/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target",
        help="Source directory with target PNG images (gen_gray_<type>_*.png)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="db/gradient_dataset.hdf5",
        help="Output HDF5 file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing). None = use all"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Pixel threshold for detecting black points"
    )
    parser.add_argument(
        "--no-conditioning",
        action="store_true",
        help="Disable type-based conditioning (default: enabled)"
    )
    parser.add_argument(
        "--point-sizes",
        type=int,
        nargs="+",
        default=[1024, 4096],
        help="Valid point set sizes (perfect squares for 2D - use 32x32=1024 or 64x64=4096)"
    )
    
    args = parser.parse_args()
    
    build_gradient_dataset(
        args.source,
        args.output,
        max_samples=args.max_samples,
        threshold=args.threshold,
        use_conditioning=not args.no_conditioning,
        target_point_sizes=args.point_sizes
    )
