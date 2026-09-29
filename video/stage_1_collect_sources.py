"""Stage 1 (local): copy every input file the video needs into video/assets/sources/, and point
shots.json at the copies, so the video folder is self-contained (and can be copied to the
server as is for stage_1_trajectories.py).

Collected: each trajectory's condition image and final result, and the tone shot's result,
summary and source image. Files are copied only when missing or changed; paths already inside
video/assets/ are left as they are.

    python video/stage_1_collect_sources.py
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASSETS_DIR, load_shots, save_shots  # noqa: E402

SOURCES_DIR = str(ASSETS_DIR / "sources")
DRY_RUN = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=SOURCES_DIR)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN)
    return p.parse_args()


def collect(path, out_root, dry):
    """Copy `path` under out_root, keeping its path below experiments/outputs/ (or its name)."""
    src = Path(path)
    if Path(ASSETS_DIR) in src.parents:
        return path
    if not src.exists():
        raise SystemExit(f"missing source file: {src}")
    parts = src.parts
    rel = Path(*parts[parts.index("outputs") + 1:]) if "outputs" in parts else Path(src.name)
    dst = Path(out_root) / rel
    if not dry and (not dst.exists() or dst.stat().st_size != src.stat().st_size
                    or dst.stat().st_mtime < src.stat().st_mtime):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst.as_posix()


def main():
    a = parse_args()
    data = load_shots()
    n = 0
    for spec in data["trajectories"].values():
        for key in ("image", "result"):
            if key in spec:
                new = collect(spec[key], a.output, a.dry_run)
                n += new != spec[key]
                spec[key] = new
    for shot in data["shots"]:
        v = shot["visual"]
        if v["type"] != "tone":
            continue
        if "result" in v:        # stage_1_results.py reads the run's summary.json next to the result
            collect(Path(v["result"]).parent / "summary.json", a.output, a.dry_run)
        for key in ("result", "image"):
            if key in v:
                new = collect(v[key], a.output, a.dry_run)
                n += new != v[key]
                v[key] = new
    if a.dry_run:
        print(f"dry run: {n} path(s) would change")
        return
    save_shots(data)
    print(f"{n} path(s) now point into {a.output}")


if __name__ == "__main__":
    main()
