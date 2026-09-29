"""Stage 1 (local): export the paper figures the video shows as PNGs.

Reads the figures from the paper folder (read-only) and writes them to
video/assets/figures/<name>.png. PDFs are rasterized with pdftoppm (poppler; MiKTeX and
TeX Live both ship it), PNGs are copied.

    python video/stage_1_figures.py
    python video/stage_1_figures.py --paper-dir paper4 --dpi 300
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG_DIR  # noqa: E402

PAPER_DIR = "paper4"
OUTPUT_DIR = str(FIG_DIR)
DPI = 300
PDFTOPPM = "pdftoppm"
# output name -> figure path inside the paper folder
FIGURES = {
    "pipeline": "results_v3/flow.png",
    "stress1": "results_v3/stress/stress1.pdf",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper-dir", default=PAPER_DIR)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--dpi", type=int, default=DPI)
    p.add_argument("--pdftoppm", default=PDFTOPPM, help="pdftoppm executable")
    return p.parse_args()


def rasterize(pdf, png, dpi, exe):
    if shutil.which(exe) is None:
        raise SystemExit(f"'{exe}' not found; install poppler or pass --pdftoppm <path>")
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "page"
        subprocess.run([exe, "-png", "-r", str(dpi), "-singlefile", str(pdf), str(stem)], check=True)
        shutil.move(str(stem.with_suffix(".png")), png)


def main():
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in FIGURES.items():
        src = Path(a.paper_dir) / rel
        if not src.exists():
            raise SystemExit(f"missing figure: {src}")
        dst = out / f"{name}.png"
        if src.suffix.lower() == ".pdf":
            rasterize(src, dst, a.dpi, a.pdftoppm)
        else:
            shutil.copyfile(src, dst)
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
