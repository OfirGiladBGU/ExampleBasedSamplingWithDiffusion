"""Pre-build the smart-init (and optional SDF) cache to disk, so training epochs run at
full speed instead of computing ot.emd lazily during epoch 1.

Run ONCE per (source, cache_data_dir, seed, grid_size). Every training run that points
--cache-data-dir at the same path then reuses it -- just like the precomputed GT offsets.
Reuses the dataset's own caching code, so the files are guaranteed consistent with training.

Example:
    python -m control_fm.build_smart_init_cache \
        --source S:/ControlNet/training/icons-50_512/source \
        --offsets S:/ControlNet/training/icons-50_512/processed_offsets \
        --cache-data-dir S:/ExampleBasedSamplingWithDiffusion/control_v4/train_archive/train_outputs_icons50_512/smart_init_cache
"""
import argparse
import os

from tqdm import tqdm

from control_fm.DynamicStippleDataset import DynamicStippleDataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Source images dir")
    p.add_argument("--offsets", required=True, help="GT offsets dir (stems must match source)")
    p.add_argument("--cache-data-dir", required=True, help="Persistent cache dir (reused across runs)")
    p.add_argument("--grid-size", type=int, default=32)
    p.add_argument("--smart-init-seed", type=int, default=42)
    p.add_argument("--sdf-features", action="store_true", default=False,
                   help="Also pre-build the SDF cache (only if you train with --sdf-features)")
    args = p.parse_args()

    os.makedirs(args.cache_data_dir, exist_ok=True)
    ds = DynamicStippleDataset(
        args.source,
        args.offsets,
        grid_size=args.grid_size,
        cache_data_dir=args.cache_data_dir,
        smart_init_seed=args.smart_init_seed,
        smart_init_features=False,
        sdf_features=args.sdf_features,
        provide_smart_init_source=True,   # ensures smart-init offsets are built/cached
        preload_ram=False,                # cache to disk, do NOT hold in RAM
    )
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset is empty -- check --source/--offsets stems match.")
    print(f"Pre-building cache for {n} samples -> {args.cache_data_dir}")
    for i in tqdm(range(n), desc="caching"):
        _ = ds[i]   # first touch computes + saves smart-init offsets/grid (+ SDF); later runs load
    print("Cache build complete. Training epochs will now skip ot.emd and run at full speed.")


if __name__ == "__main__":
    main()
