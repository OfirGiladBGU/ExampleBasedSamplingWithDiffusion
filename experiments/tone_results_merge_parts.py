"""Merge tone stage-1 runs that were split with --offset / --limit into one stage-1 folder.

Each part folder holds <stem>.npz files and a summary.json whose "results" cover its own images.
The merged folder gets symlinks to every part's .npz and one summary.json (config of the first part,
results of all parts), so later stages can read it like a single stage-1 run.

Usage:
    python experiments/tone_results_merge_parts.py <merged dir> <part dir> [<part dir> ...]
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Merge split tone stage-1 runs")
    p.add_argument("merged")
    p.add_argument("parts", nargs="+")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.merged)
    out.mkdir(parents=True, exist_ok=True)
    config, results = None, {}
    for part in map(Path, a.parts):
        s = json.loads((part / "summary.json").read_text())
        config = config or s["config"]
        results.update(s["results"])
        for f in part.glob("*.npz"):
            link = out / f.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(f.resolve())
    (out / "summary.json").write_text(json.dumps({"config": config, "results": results}, indent=2))
    print(f"{len(results)} images from {len(a.parts)} parts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
