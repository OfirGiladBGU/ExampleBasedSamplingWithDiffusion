"""
Build a balanced dataset with N samples per gradient type.
This allows quick testing of whether conditioning works.
"""

import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import shutil
from tqdm import tqdm


def get_gradient_type(filename):
    """Extract gradient type from filename."""
    import re
    match = re.match(r'gen_gray_(.+?)_\d+_\d+\.png', filename)
    if match:
        return match.group(1)
    return None


def build_balanced_subset(
    source_dir,
    output_dir,
    samples_per_type=200
):
    """
    Create a balanced subset of images (N per type).
    
    Args:
        source_dir: Source image directory
        output_dir: Output directory for balanced subset
        samples_per_type: Number of samples per gradient type
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Group images by type
    images_by_type = defaultdict(list)
    
    print("Scanning source images...")
    for fname in sorted(os.listdir(source_dir)):
        if fname.endswith('.png') and fname.startswith('gen_gray_'):
            gtype = get_gradient_type(fname)
            if gtype:
                images_by_type[gtype].append(fname)
    
    print(f"\nFound {len(images_by_type)} gradient types:")
    for gtype in sorted(images_by_type.keys()):
        count = len(images_by_type[gtype])
        print(f"  {gtype}: {count} images")
    
    # Copy balanced subset
    print(f"\nCreating balanced subset ({samples_per_type} per type)...")
    total_copied = 0
    
    for gtype in sorted(images_by_type.keys()):
        images = images_by_type[gtype]
        
        # Select random subset
        selected = np.random.choice(
            images,
            min(samples_per_type, len(images)),
            replace=False
        )
        
        print(f"  {gtype}: copying {len(selected)} images...", end='')
        
        for fname in selected:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            total_copied += 1
        
        print(f" ✓")
    
    print(f"\nTotal images copied: {total_copied}")
    return total_copied


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create balanced subset of gradient images"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target",
        help="Source image directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/gradient_balanced_subset",
        help="Output directory for balanced subset"
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=200,
        help="Number of samples per gradient type"
    )
    
    args = parser.parse_args()
    
    build_balanced_subset(
        args.source,
        args.output,
        args.samples_per_type
    )
