"""Run the local part of the video pipeline: script check, figures, WVS, TTS, render, assemble.

Stage 1 trajectories and tone snapshots need the GPU and the checkpoints, so they run on
the server (video/server/run_stage_1.sbatch); copy video/assets/trajectories/ back before
the final render. Until then those shots render as labeled placeholders.

    python video/make_video.py                  # full quality
    python video/make_video.py --preview        # quick low-resolution pass
    python video/make_video.py --skip tts       # reuse existing narration
"""

import argparse
import subprocess
import sys

STAGES = [
    ("script", ["video/stage_0_parse_script.py"]),
    ("figures", ["video/stage_1_figures.py"]),
    ("sources", ["video/stage_1_collect_sources.py"]),
    ("wvs", ["video/stage_1_wvs.py"]),
    ("results", ["video/stage_1_results.py"]),
    ("tts", ["video/stage_2_tts.py"]),
    ("render", ["video/stage_3_render.py"]),
    ("assemble", ["video/stage_4_assemble.py"]),
]
SKIP = ""
PREVIEW = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip", default=SKIP, help="Comma-separated stage names: " + ",".join(n for n, _ in STAGES))
    p.add_argument("--preview", action=argparse.BooleanOptionalAction, default=PREVIEW)
    return p.parse_args()


def main():
    a = parse_args()
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    for name, cmd in STAGES:
        if name in skip:
            print(f"== {name}: skipped")
            continue
        extra = ["--preview"] if name == "render" and a.preview else []
        print(f"== {name}: {' '.join(cmd + extra)}", flush=True)
        r = subprocess.run([sys.executable] + cmd + extra)
        if r.returncode != 0:
            raise SystemExit(f"stage '{name}' failed (exit {r.returncode})")
    print("done")


if __name__ == "__main__":
    main()
