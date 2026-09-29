"""Stage 4 (local): add the narration to every shot and stitch them into the final video.

Shots whose clip and narration are unchanged since the last run are reused, so after
re-rendering one shot only that shot is re-encoded before the final (lossless) concatenation.
For each shot, the silent clip from stage 3 is muxed with its narration (delayed by the
lead-in, padded with silence to the clip length), then all shots are concatenated in
shots.json order. Also writes an .srt subtitle file with the same timing as the burned-in
subtitles, for players or upload sites that take a sidecar file.

    python video/stage_4_assemble.py
    python video/stage_4_assemble.py --output video/build/StippleDiffusion.mp4

FFmpeg comes from the imageio-ffmpeg package (a static build), or from PATH.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AUDIO_DIR, BUILD_DIR, MUXED_DIR, SHOT_VIDEO_DIR, load_durations,  # noqa: E402
                    line_spans_key, load_shots, subtitle_timeline)

OUTPUT = str(BUILD_DIR / "StippleDiffusion_video.mp4")
AUDIO_BITRATE = "192k"
VIDEO_CRF = 18                     # x264 quality of the final encode (lower = better)
AUDIO_RATE = 48000
FORCE = False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--crf", type=int, default=VIDEO_CRF)
    p.add_argument("--force", action=argparse.BooleanOptionalAction, default=FORCE,
                   help="Re-mux every shot, not only those whose clip or narration changed")
    return p.parse_args()


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise SystemExit("ffmpeg not found: pip install imageio-ffmpeg, or put ffmpeg on PATH")
        return exe


def probe_duration(ff, path):
    """Clip length in seconds, read from ffmpeg's own report (no ffprobe needed)."""
    r = subprocess.run([ff, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            hh, mm, ss = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    raise RuntimeError(f"cannot read duration of {path}")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("ffmpeg failed:\n" + " ".join(map(str, cmd)) + "\n" + r.stderr[-2000:])


def mux(ff, video, audio, lead_s, out, crf):
    """Video + narration delayed by lead_s, trimmed/padded to the video's length."""
    dur = probe_duration(ff, video)
    cmd = [ff, "-y", "-hide_banner", "-i", str(video)]
    if audio is not None:
        ms = int(round(lead_s * 1000))
        cmd += ["-i", str(audio), "-filter_complex",
                f"[1:a]aresample={AUDIO_RATE},adelay={ms}|{ms},apad[a]", "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=mono", "-map", "0:v", "-map", "1:a"]
    cmd += ["-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "1", str(out)]
    run(cmd)
    return dur


def srt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"


def write_srt(path, entries):
    lines = []
    for i, (t0, t1, text) in enumerate(entries, 1):
        lines += [str(i), f"{srt_time(t0)} --> {srt_time(t1)}", text, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    a = parse_args()
    data = load_shots()
    settings = data["settings"]
    durations = load_durations()
    ff = ffmpeg_exe()
    MUXED_DIR.mkdir(parents=True, exist_ok=True)

    parts, subs, clock = [], [], 0.0
    for shot in data["shots"]:
        video = SHOT_VIDEO_DIR / f"{shot['id']}.mp4"
        if not video.exists():
            raise SystemExit(f"missing {video}; run stage_3_render.py first")
        wav = AUDIO_DIR / f"{shot['id']}.wav"
        audio = wav if wav.exists() and shot["id"] in durations else None
        out = MUXED_DIR / f"{shot['id']}.mp4"
        newest = max(video.stat().st_mtime, wav.stat().st_mtime if audio is not None else 0.0)
        if out.exists() and out.stat().st_mtime > newest and not a.force:
            dur = probe_duration(ff, out)          # unchanged shot: reuse its muxed clip
        else:
            dur = mux(ff, video, audio, settings["lead_in_s"], out, a.crf)
        parts.append(out)

        speech = float(durations.get(shot["id"], 0.0))
        text = (shot.get("narration") or "").strip()
        if text and speech > 0:
            timeline, _ = subtitle_timeline(text, speech, durations.get(line_spans_key(shot["id"])))
            t0 = clock + settings["lead_in_s"]
            subs += [(t0 + a, t0 + b, c.replace("\n", " ")) for a, b, c in timeline]
        print(f"{shot['id']}: {dur:.2f}s {'with' if audio else 'without'} narration")
        clock += dur

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        for p in parts:
            fh.write(f"file '{p.resolve().as_posix()}'\n")
        listing = fh.name
    try:
        run([ff, "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", str(out)])
    finally:
        Path(listing).unlink(missing_ok=True)
    write_srt(out.with_suffix(".srt"), subs)
    print(f"final video {clock:.1f}s -> {out}\nsubtitles -> {out.with_suffix('.srt')}")


if __name__ == "__main__":
    main()
