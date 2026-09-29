"""Stage 3 (local): render every shot with Manim, each exactly as long as its narration.

Reads video/shots.json and the narration lengths from stage 2 (video/build/audio/
durations.json) and writes one silent clip per shot to video/build/shots/<shot_id>.mp4.
A shot's length is  lead-in + narration + tail  (settings in shots.json), or its
"min_seconds" when it has no narration. Missing stage-1 data renders as a labeled
placeholder, so the whole video can be previewed before the server runs.

    python video/stage_3_render.py
    python video/stage_3_render.py --preview                  # 960x540 at 15 fps, fast
    python video/stage_3_render.py --only s05_pipeline,s06_pipeline_initial
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUILD_DIR, SHOT_VIDEO_DIR, line_spans_key, load_durations, load_shots,  # noqa: E402
                    select_shots, shot_duration)

ONLY = ""
PREVIEW = False
PREVIEW_SIZE = (960, 540)
PREVIEW_FPS = 15
MANIM_MEDIA_DIR = str(BUILD_DIR / "manim")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=ONLY, help="Comma-separated shot ids")
    p.add_argument("--preview", action=argparse.BooleanOptionalAction, default=PREVIEW,
                   help=f"Render at {PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}, {PREVIEW_FPS} fps")
    return p.parse_args()


def previous_zoom(shots, index):
    """Where the camera ended in the previous shot, so consecutive pipeline zooms pan smoothly."""
    if index == 0:
        return None
    prev = shots[index - 1]["visual"]
    return prev.get("zoom") if prev["type"] == "pipeline" else None


def render_shot(shot, duration, speech, settings, prev_zoom, size, fps, line_spans=None):
    from manim import tempconfig
    from scenes import ShotScene

    w, h = size
    cfg = {
        "pixel_width": w, "pixel_height": h, "frame_rate": fps,
        "frame_height": 8.0, "frame_width": 8.0 * w / h,
        "background_color": settings["background"],
        "media_dir": MANIM_MEDIA_DIR, "output_file": shot["id"],
        "disable_caching": True, "write_to_movie": True, "format": "mp4",
        "verbosity": "WARNING", "progress_bar": "none",
    }
    with tempconfig(cfg):
        cls = type(f"Shot_{shot['id']}", (ShotScene,),
                   {"shot": shot, "shot_len": duration, "speech_s": speech,
                    "settings": settings, "prev_zoom": prev_zoom, "line_spans": line_spans})
        scene = cls()
        scene.render()
        return Path(scene.renderer.file_writer.movie_file_path)


def main():
    a = parse_args()
    data = load_shots()
    settings = data["settings"]
    durations = load_durations()
    if not durations:
        print("WARNING: no narration durations yet (run stage_2_tts.py); using min_seconds per shot")
    size = PREVIEW_SIZE if a.preview else (settings["width"], settings["height"])
    fps = PREVIEW_FPS if a.preview else settings["fps"]
    SHOT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    all_shots = data["shots"]
    wanted = {s["id"] for s in select_shots(all_shots, a.only)}
    total = 0.0
    for i, shot in enumerate(all_shots):
        if shot["id"] not in wanted:
            continue
        dur = shot_duration(shot, durations, settings)
        t0 = time.time()
        src = render_shot(shot, dur, float(durations.get(shot["id"], 0.0)), settings,
                          previous_zoom(all_shots, i), size, fps,
                          durations.get(line_spans_key(shot["id"])))
        dst = SHOT_VIDEO_DIR / f"{shot['id']}.mp4"
        shutil.copyfile(src, dst)
        total += dur
        print(f"{shot['id']}: {dur:.2f}s rendered in {time.time() - t0:.0f}s -> {dst}", flush=True)
    print(f"rendered {len(wanted)} shot(s), {total:.1f}s of video")


if __name__ == "__main__":
    main()
