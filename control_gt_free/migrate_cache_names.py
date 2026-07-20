"""Rename legacy smart-init cache files to the explicit naming used by the dataset.

  {stem}_offsets.npy   ->  {stem}_smartinit_offsets.npy   (smart-init offsets)
  {stem}.npy           ->  {stem}_smartinit_grid.npy      (smart-init grid)

Left untouched: *_sdf_raw.npy (SDF), *_points.npy (raw points), and anything already
migrated. GT offsets are not in this dir, so nothing here is a GT file.

Dry-run by default -- prints what it would do. Pass --apply to actually rename.

Usage:
    python migrate_cache_names.py --cache-dir "S:/.../cache_data"            # preview
    python migrate_cache_names.py --cache-dir "S:/.../cache_data" --apply    # do it
"""
import argparse
import os


def target_name(fname):
    """Return the new filename for a legacy cache file, or None to leave it alone."""
    if fname.endswith("_smartinit_grid.npy") or fname.endswith("_smartinit_offsets.npy"):
        return None  # already migrated
    if fname.endswith("_sdf_raw.npy") or fname.endswith("_points.npy"):
        return None  # not smart-init grid/offsets
    if fname.endswith("_offsets.npy"):
        return fname[: -len("_offsets.npy")] + "_smartinit_offsets.npy"
    if fname.endswith(".npy"):
        return fname[: -len(".npy")] + "_smartinit_grid.npy"  # bare grid file
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True, help="Cache dir to migrate (recursed)")
    p.add_argument("--apply", action="store_true", help="Actually rename (default: dry-run)")
    args = p.parse_args()

    if not os.path.isdir(args.cache_dir):
        raise SystemExit(f"Not a directory: {args.cache_dir}")

    renamed = skipped = collisions = 0
    for root, _dirs, files in os.walk(args.cache_dir):
        for f in files:
            new = target_name(f)
            if new is None:
                continue
            src = os.path.join(root, f)
            dst = os.path.join(root, new)
            if os.path.exists(dst):
                collisions += 1
                print(f"SKIP (target exists): {f} -> {new}")
                continue
            if args.apply:
                os.rename(src, dst)
                renamed += 1
            else:
                print(f"would rename: {f} -> {new}")
                renamed += 1

    verb = "Renamed" if args.apply else "Would rename"
    print(f"\n{verb}: {renamed} | collisions skipped: {collisions}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to perform the renames.")


if __name__ == "__main__":
    main()
