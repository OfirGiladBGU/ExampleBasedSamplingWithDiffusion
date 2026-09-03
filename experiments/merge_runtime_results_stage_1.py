#!/usr/bin/env python3
"""Stage 1: gather the timestamps_<METHOD>_<GRID> folders from each per-method runtime
output folder into ONE common folder, so stage 2 can aggregate them.

The per-method folders and their input images are built by
merge_runtime_results_stage_0.py; each method's own stippling code fills in the
timestamps_<METHOD>_<GRID> folders this stage collects.

Copies (recursively) every folder named ``timestamps_<METHOD>_<GRID>`` from:
    outputs/icons_results_runtimes_cn
    outputs/icons_results_runtimes_gbn
    outputs/icons_results_runtimes_wvs
    outputs/icons_results_runtimes_bnot
into:
    outputs/icons_results_runtimes/

Anything that is not a timestamps_<METHOD>_<GRID> folder is ignored. Re-running merges/
overwrites (dirs_exist_ok), so it is idempotent. Then run merge_runtime_results_stage_2.py.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SOURCE_DIRS = [
    "icons_results_runtimes_wvs",
    "icons_results_runtimes_bnot",
    "icons_results_runtimes_gbn",
    "icons_results_runtimes_cn",
]
DEST = os.path.join(BASE, "icons_results_runtimes")

# Same pattern stage 1 uses, so only real timing folders are copied.
RE_FOLDER = re.compile(r"^timestamps_([A-Za-z0-9]+)_(\d+)$")


def count_txt(folder):
    n = 0
    for _root, _dirs, files in os.walk(folder):
        n += sum(1 for f in files if f.lower().endswith(".txt"))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=DEST, help="Common destination folder.")
    ap.add_argument("--sources", nargs="+", default=SOURCE_DIRS,
                    help="Per-method runtime folders (relative to outputs/, or absolute).")
    ap.add_argument("--dry-run", action="store_true", help="List what would be copied; copy nothing.")
    args = ap.parse_args()

    if not args.dry_run:
        os.makedirs(args.dest, exist_ok=True)

    copied = 0
    for sd in args.sources:
        src_base = sd if os.path.isabs(sd) else os.path.join(BASE, sd)
        if not os.path.isdir(src_base):
            print(f"[WARN] source not found: {src_base}")
            continue
        found = 0
        for name in sorted(os.listdir(src_base)):
            full = os.path.join(src_base, name)
            if not os.path.isdir(full) or not RE_FOLDER.match(name):
                continue
            found += 1
            dst = os.path.join(args.dest, name)
            ntxt = count_txt(full)
            if args.dry_run:
                print(f"  DRY copy {name}  ({ntxt} txt)  {src_base} -> {dst}")
            else:
                shutil.copytree(full, dst, dirs_exist_ok=True)
                print(f"  copied {name}  ({ntxt} txt)")
            copied += 1
        if found == 0:
            print(f"[WARN] {os.path.basename(src_base)}: no timestamps_<METHOD>_<GRID> folders")

    print(f"\n{'Would copy' if args.dry_run else 'Copied'} {copied} timestamps folders -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
