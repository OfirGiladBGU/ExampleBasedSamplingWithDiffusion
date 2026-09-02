# Fix script that it needs to create the manifest.json file in the output folder if not exists.

"""Copy per-method stipple results (PNG + NPY) for the validation manifest into
flat target folders, mirroring the layout of target_CN-GBN_1024.

The source trees are nested as  <method_target>/Icons-50/<category>/<stem>.{png,npy}
while the destination folders are FLAT (all files directly inside), exactly like
experiments/outputs/quantitative_advance_metrics/target_CN-GBN_1024.

For each method (WVS, BNOT, GBN) this script:
  * reads validation_manifest.json (a list of "<stem>.png" filenames),
  * indexes every file under the method's source tree by basename (one os.walk),
  * copies each manifest stem's .png AND .npy, FLATTENED, into
    experiments/outputs/quantitative_advance_metrics/target_<METHOD>_1024/

Run (on the cluster, where the /groups filesystem is fast):
    python experiments/quantitative_advance_copy_results.py
    python experiments/quantitative_advance_copy_results.py --dry-run
    python experiments/quantitative_advance_copy_results.py --methods WVS GBN
"""

import argparse
import json
import os
import shutil

OUTPUT_BASE = "/groups/asharf_group/ofirgila/ExampleBasedSamplingWithDiffusion/experiments/outputs/quantitative_advance_metrics"
MANIFEST = os.path.join(OUTPUT_BASE, "validation_manifest.json")

# method -> source "target" root (nested; files live under <root>/Icons-50/<category>/)
SOURCES = {
    "WVS":  "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_WVS/target",
    "BNOT": "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_BNOT/target",
    "GBN":  "/groups/asharf_group/ofirgila/ControlNet/training/Icons-50_1024_GBN/target",
}

# extensions copied for each manifest PNG stem (png = render, npy = point coords)
COPY_EXTS = (".png", ".npy")


def build_index(src_root):
    """Map basename -> full path for every file under src_root (one recursive walk).

    Keeps the first occurrence of each basename and counts duplicates (should be 0
    for this dataset since icon filenames are globally unique).
    """
    index, dups = {}, 0
    for dirpath, _dirs, files in os.walk(src_root):
        for fn in files:
            if fn in index:
                dups += 1
                continue
            index[fn] = os.path.join(dirpath, fn)
    return index, dups


def main():
    ap = argparse.ArgumentParser(description="Flatten-copy manifest PNG+NPY results per method.")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--output-base", default=OUTPUT_BASE)
    ap.add_argument("--methods", nargs="+", default=list(SOURCES.keys()),
                    help="Subset of methods to copy (default: WVS BNOT GBN).")
    ap.add_argument("--dry-run", action="store_true", help="Report only; copy nothing.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recopy files that already exist in the destination.")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    stems = [os.path.splitext(os.path.basename(n))[0] for n in manifest]
    print(f"Manifest: {len(stems)} entries  ({args.manifest})")

    overall_ok = True
    for method in args.methods:
        if method not in SOURCES:
            print(f"\n!! Unknown method '{method}' (known: {list(SOURCES)}); skipping.")
            overall_ok = False
            continue
        src_root = SOURCES[method]
        dest = os.path.join(args.output_base, f"target_{method}_1024")
        print(f"\n=== {method} ===")
        print(f"  source: {src_root}")
        print(f"  dest  : {dest}")
        if not os.path.isdir(src_root):
            print("  !! source directory not found; skipping.")
            overall_ok = False
            continue

        print("  indexing source tree ...", flush=True)
        index, dups = build_index(src_root)
        print(f"  indexed {len(index)} files" + (f"  ({dups} duplicate basenames ignored)" if dups else ""))

        if not args.dry_run:
            os.makedirs(dest, exist_ok=True)

        copied = {e: 0 for e in COPY_EXTS}
        existing = {e: 0 for e in COPY_EXTS}
        missing = {e: [] for e in COPY_EXTS}
        for stem in stems:
            for ext in COPY_EXTS:
                fn = stem + ext
                src = index.get(fn)
                if src is None:
                    missing[ext].append(fn)
                    continue
                dst = os.path.join(dest, fn)
                if os.path.exists(dst) and not args.overwrite:
                    existing[ext] += 1
                    continue
                if not args.dry_run:
                    shutil.copy2(src, dst)
                copied[ext] += 1

        for ext in COPY_EXTS:
            n_miss = len(missing[ext])
            line = (f"  {ext}: copied {copied[ext]}, already-present {existing[ext]}, "
                    f"missing {n_miss} / {len(stems)}")
            if n_miss:
                line += f"  -> e.g. {missing[ext][:5]}" + ("..." if n_miss > 5 else "")
                overall_ok = False
            print(line)

    print("\n" + ("DRY RUN complete (nothing written)." if args.dry_run
                  else "Copy complete." if overall_ok else "Copy finished WITH MISSING files (see above)."))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
