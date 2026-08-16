#!/usr/bin/env python3
"""Aggregate per-image TOTAL runtimes from the ``timestamps_*`` folders.

Each method writes its per-image timing into a different text format:

    BNOT : a line ``total_wall_seconds: <float>``
    CN   : a line ``Total Execution Time: <float> s``
    WVS  : the whole file is just a bare number (seconds)
    GBN  : the whole file is just a bare number (seconds)

Folder layout is ``timestamps_<METHOD>_<GRID>`` and the txt files may sit either
directly inside it (CN) or under ``Icons-50/<category>/`` (BNOT/WVS/GBN); a
recursive glob covers both.  Some folders have fewer files than others because
BNOT occasionally crashes and those images were skipped -- we simply aggregate
whatever totals are present.

Output: ``outputs/icons_results_runtimes/runtimes_avg.json`` ::

    {"WVS": {"256": {"avg": ..., "std": ..., "count": N}, ...}, ...}
"""

from __future__ import annotations

import glob
import json
import math
import os
import re

ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
    "icons_results_runtimes",
)
OUT_JSON = os.path.join(ROOT, "runtimes_avg.json")

# Structured formats (BNOT / CN). WVS & GBN are bare numbers handled below.
RE_BNOT = re.compile(r"total_wall_seconds:\s*([0-9.eE+-]+)")
RE_CN = re.compile(r"Total Execution Time:\s*([0-9.eE+-]+)")
RE_FOLDER = re.compile(r"^timestamps_([A-Za-z0-9]+)_(\d+)$")


def extract_total_seconds(text):
    """Return the TOTAL runtime (seconds) from one timestamp file, or None."""
    m = RE_BNOT.search(text)
    if m:
        return float(m.group(1))
    m = RE_CN.search(text)
    if m:
        return float(m.group(1))
    # WVS / GBN: first non-empty line is the bare total.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return float(line.split()[0])
        except ValueError:
            return None
    return None


def main():
    results = {}  # method -> { size -> stats }

    for name in sorted(os.listdir(ROOT)):
        full = os.path.join(ROOT, name)
        if not os.path.isdir(full):
            continue
        fm = RE_FOLDER.match(name)
        if not fm:
            continue
        method, size = fm.group(1), fm.group(2)

        txts = glob.glob(os.path.join(full, "**", "*.txt"), recursive=True)
        times, skipped = [], []
        for tf in txts:
            try:
                with open(tf, "r", errors="replace") as fh:
                    val = extract_total_seconds(fh.read())
            except OSError as exc:
                skipped.append((tf, str(exc)))
                continue
            if val is None:
                skipped.append((tf, "no total found"))
            else:
                times.append(val)

        if not times:
            print(f"[WARN] {name}: no usable timing files ({len(txts)} txt found)")
            continue

        n = len(times)
        avg = sum(times) / n
        if n > 1:
            std = math.sqrt(sum((t - avg) ** 2 for t in times) / (n - 1))
        else:
            std = 0.0

        results.setdefault(method, {})[size] = {
            "avg": round(avg, 6),
            "std": round(std, 6),
            "count": n,
        }
        flag = f"  [skipped {len(skipped)}]" if skipped else ""
        print(f"{method:5s} {size:>6s}: n={n:2d}  avg={avg:10.4f}s  std={std:9.4f}s{flag}")

    # Sort methods alphabetically and sizes numerically.
    ordered = {
        method: {size: results[method][size]
                 for size in sorted(results[method], key=int)}
        for method in sorted(results)
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(ordered, fh, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
