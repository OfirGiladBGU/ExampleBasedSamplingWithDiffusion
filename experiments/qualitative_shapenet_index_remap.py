"""qualitative_shapenet_index_remap.py

Translate `"shapenet"` sample indices (the old combined ShapeNetRender_Custom-3K_1600 set)
into indices for the per-category sets, and print the SAMPLES_MAP lines to paste.

Why this exists: the combined 300-entry validation set was split into per-category sets of
1000, so every index in a `"shapenet"` entry now points somewhere else. Matching is done by
FILENAME via each set's `validation_manifest.json`, not by arithmetic -- the split reordered
entries, so no offset would work.

No arguments. Edit SAMPLES_MAP below and run it:

    python experiments/qualitative_shapenet_index_remap.py

It accepts either shape, exactly as the showcase scripts do:

    SAMPLES_MAP = {"shapenet": [100, 138]}                  # flat -> one column
    SAMPLES_MAP = {"shapenet": [[-1], [174], [184], [-1]]}  # column-major

and prints, for each replacement key, a padded line covering every column, so dropping them
in keeps the grid shape. HOLE (-1) is passed through untouched.

Airplanes note: the Airplanes set holds the `_01` re-render while the combined set holds
`_00`, so those never match by filename. They are matched on the model hash instead (the
filename with the trailing `_NN` removed) and flagged -- the `_01` view of a model whose
`_00` looked good is not guaranteed to look good itself, so eyeball those picks.
"""
from pathlib import Path
import json

# -- EDIT ME: the entry you want translated --------------------------------
SAMPLES_MAP = {
    "shapenet": [[-1], [174], [184], [-1]]
}

# MAIN - 4 columns
# SAMPLES_MAP = {
#     "shapenet": [[-1], [174], [184], [-1]]
# }

# APPENDIX - 5 columns
# SAMPLES_MAP = {
#     "shapenet": [[250, 260], [252, 274], [261, -1], [276, 93], [225, 266]]
# }
# --------------------------------------------------------------------------

# Where the sets live, relative to the repo root. Mirrors DIR_MAP in the showcase scripts.
COMBINED = "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K_1600"
TARGETS = {
    "cars": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Cars_1600",
    "watercrafts": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Watercrafts_1600",
    "airplanes": "experiments/outputs/z_validation_data/ShapeNetRender_Custom-3K-Airplanes_1600",
}
# Stacking order the replacement keys will have in DIR_MAP. Used only to warn when a split
# would reorder cells inside a column; it does not change the mapping itself.
OUT_ORDER = ["cars", "watercrafts", "airplanes"]

HOLE = -1
ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}


def manifest(rel):
    if rel not in _CACHE:
        p = ROOT / rel / "validation_manifest.json"
        if not p.exists():
            raise SystemExit("missing manifest: %s" % p)
        _CACHE[rel] = json.load(open(p, encoding="utf-8"))
    return _CACHE[rel]


def normalize_cols(entry):
    """Same convention as the showcase scripts: a flat list is one column."""
    if not entry:
        return []
    if isinstance(entry[0], (list, tuple)):
        return [[int(i) for i in col] for col in entry]
    return [[int(i) for i in entry]]


def build_lookup():
    exact, by_hash = {}, {}
    for name, rel in TARGETS.items():
        for i, fn in enumerate(manifest(rel)):
            exact.setdefault(fn, (name, i))
            by_hash.setdefault(fn.rsplit("_", 1)[0], (name, i))
    return exact, by_hash


def main():
    combined = manifest(COMBINED)
    exact, by_hash = build_lookup()

    entry = SAMPLES_MAP.get("shapenet")
    if not entry:
        raise SystemExit('SAMPLES_MAP has no "shapenet" entry to translate.')
    cols = normalize_cols(entry)
    n_cols = len(cols)
    # echo back the shape that was fed in: the comparison showcase uses flat row lists,
    # the ours showcase uses column-major nested lists.
    flat_in = not isinstance(entry[0], (list, tuple))

    print("combined set: %d entries  (%s)" % (len(combined), COMBINED))
    for name, rel in TARGETS.items():
        print("  %-12s %5d entries" % (name, len(manifest(rel))))
    print()

    out = {name: [[] for _ in range(n_cols)] for name in TARGETS}
    problems = []
    print("%3s  %5s  %-14s %5s  %s" % ("col", "old", "category", "new", "filename"))
    print("-" * 78)
    for c, col in enumerate(cols):
        seq = []
        for idx in col:
            if idx == HOLE:
                print("%3d  %5d  %-14s" % (c, idx, "HOLE"))
                continue
            if not (0 <= idx < len(combined)):
                problems.append((c, idx, "out of range"))
                print("%3d  %5d  %-14s" % (c, idx, "OUT OF RANGE"))
                continue
            fn = combined[idx]
            note = ""
            if fn in exact:
                name, new = exact[fn]
            elif fn.rsplit("_", 1)[0] in by_hash:
                name, new = by_hash[fn.rsplit("_", 1)[0]]
                note = "   <- hash-matched, DIFFERENT VIEW: check it looks good"
                problems.append((c, idx, "hash-matched into %s" % name))
            else:
                problems.append((c, idx, "unmatched"))
                print("%3d  %5d  %-14s %5s  %s" % (c, idx, "UNMATCHED", "", fn))
                continue
            out[name][c].append(new)
            seq.append(name)
            print("%3d  %5d  %-14s %5d  %s%s" % (c, idx, name, new, fn, note))

        rank = [OUT_ORDER.index(n) for n in seq if n in OUT_ORDER]
        if rank != sorted(rank):
            problems.append((c, None, "column order will change"))
            print("      ^ column %d: categories appear as %s, not %s order -- "
                  "these cells will swap" % (c, seq, OUT_ORDER))

    print()
    print("=" * 78)
    print("paste into SAMPLES_MAP:")
    print("=" * 78)
    print('    # "shapenet": %s,' % json.dumps(entry))
    width = max(len(n) for n in TARGETS) + 4
    for name in OUT_ORDER:
        colvals = out[name]
        if not any(colvals):
            continue
        label = '"%s":' % name
        if flat_in:
            flat = [x for cell in colvals for x in cell]
            print("    %-*s [%s]," % (width, label, ", ".join(str(x) for x in flat)))
        else:
            cells = [(v if v else [HOLE]) for v in colvals]
            body = ", ".join("[" + ", ".join(str(x) for x in cell) + "]" for cell in cells)
            print("    %-*s [%s]," % (width, label, body))

    print()
    print('DIR_MAP: keep these keys where "shapenet" sat (before "airplanes") so the')
    print("stacking order is unchanged -- DIR_MAP order decides which rows sit on top.")
    if any(p[2] == "column order will change" for p in problems):
        print()
        print("This entry lists its categories in an order DIR_MAP cannot reproduce. In")
        print("qualitative_comparison_showcase.py set STACK_ORDER for that block, e.g.")
        seen = []
        for c, col in enumerate(cols):
            for idx in col:
                if idx == HOLE or not (0 <= idx < len(combined)):
                    continue
                fn = combined[idx]
                hit = exact.get(fn) or by_hash.get(fn.rsplit("_", 1)[0])
                if hit and hit[0] not in seen:
                    seen.append(hit[0])
        print('    STACK_ORDER = ["icons", "faces", %s]'
              % ", ".join('"%s"' % n for n in seen))

    print()
    if problems:
        print("!! review:")
        for c, idx, why in problems:
            where = "col %d" % c + (", old index %d" % idx if idx is not None else "")
            print("   %s: %s" % (where, why))
    else:
        print("all indices matched exactly by filename; nothing to review.")


if __name__ == "__main__":
    main()
