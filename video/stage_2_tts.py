"""Stage 2 (local): synthesize the narration of every shot with a local text-to-speech engine.

Writes video/build/audio/<shot_id>.wav and video/build/audio/durations.json
({shot_id: seconds}), which stage 3 uses to time each shot. A shot is re-synthesized only
when its text, the engine or the voice changed (tracked in audio/tts_meta.json).

The spoken text is the shot's "speak" field when present (a pronunciation override, e.g.
"x t" for x_t), otherwise its "narration".

Engines, all offline once installed ("auto" takes the first that is available):
    kokoro  neural, best quality (pip install kokoro soundfile; downloads its ~330 MB model once)
    piper   neural, light (pip install piper-tts; needs a voice .onnx, see --piper-model)
    sapi    Windows built-in voices (System.Speech via PowerShell); robotic, nothing to install

    python video/stage_2_tts.py
    python video/stage_2_tts.py --engine kokoro --voice af_heart --speed 0.95
    python video/stage_2_tts.py --only s05_pipeline --force
"""

import argparse
import hashlib
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUDIO_DIR, DURATIONS_PATH, line_spans_key, load_shots, select_shots  # noqa: E402

ENGINE = "auto"
KOKORO_VOICE = "af_heart"          # American English; see the kokoro VOICES list
KOKORO_LANG = "a"
PIPER_MODEL = "video/assets/tts/en_US-lessac-medium.onnx"
SAPI_VOICE = ""                    # "" = system default; substring of a voice name otherwise
SPEED = 1.0
LINE_PAUSE_S = 0.35                # silence between two lines of the same shot
PEAK_DBFS = -1.0                   # every clip is peak-normalized to this level
ONLY = ""
FORCE = False
META_PATH = AUDIO_DIR / "tts_meta.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", default=ENGINE, choices=["auto", "kokoro", "piper", "sapi"])
    p.add_argument("--voice", default="", help="Voice for the chosen engine (default per engine)")
    p.add_argument("--piper-model", default=PIPER_MODEL)
    p.add_argument("--speed", type=float, default=SPEED)
    p.add_argument("--only", default=ONLY, help="Comma-separated shot ids")
    p.add_argument("--force", action=argparse.BooleanOptionalAction, default=FORCE)
    return p.parse_args()


# ── engines: each returns (float32 mono samples, sample_rate) ──────────────────────────

class Kokoro:
    name = "kokoro"

    def __init__(self, voice, speed):
        from kokoro import KPipeline
        self.pipe = KPipeline(lang_code=KOKORO_LANG)
        self.voice, self.speed = voice or KOKORO_VOICE, speed

    def __call__(self, text):
        chunks = [np.asarray(audio, np.float32) for _, _, audio in
                  self.pipe(text, voice=self.voice, speed=self.speed)]
        return np.concatenate(chunks), 24000


class Piper:
    name = "piper"

    def __init__(self, model, speed):
        from piper import PiperVoice
        if not Path(model).exists():
            raise FileNotFoundError(f"piper voice not found: {model}")
        self.voice, self.speed = PiperVoice.load(model), speed

    def __call__(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "piper.wav"
            with wave.open(str(path), "wb") as wf:
                try:                                   # piper-tts >= 1.3
                    from piper import SynthesisConfig
                    self.voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=1.0 / self.speed))
                except ImportError:                    # older API
                    self.voice.synthesize(text, wf, length_scale=1.0 / self.speed)
            return read_wav(path)


class Sapi:
    """Windows' built-in System.Speech voices, driven through PowerShell.

    (pyttsx3 hangs on its second utterance on some Windows setups; this has no Python
    dependency at all and one fresh process per clip.)"""
    name = "sapi"
    SCRIPT = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "if ($env:TTS_VOICE) { $v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Name -like \"*$env:TTS_VOICE*\" } |"
        " Select-Object -First 1; if (-not $v) { throw \"no voice matching $env:TTS_VOICE\" }; $s.SelectVoice($v.VoiceInfo.Name) };"
        "$s.Rate = [int]$env:TTS_RATE;"
        "$s.SetOutputToWaveFile($env:TTS_OUT);"
        "$s.Speak([IO.File]::ReadAllText($env:TTS_TEXT));"
        "$s.Dispose()"
    )

    def __init__(self, voice, speed):
        import shutil
        if shutil.which("powershell") is None:
            raise RuntimeError("PowerShell not found (the sapi engine is Windows-only)")
        self.voice = voice
        # System.Speech rate is -10..10 around the voice's normal pace, roughly 10% per unit
        self.rate = int(np.clip(round((speed - 1.0) * 10), -10, 10))

    def __call__(self, text):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "text.txt", Path(tmp) / "sapi.wav"
            src.write_text(text, encoding="utf-8")
            env = dict(os.environ, TTS_TEXT=str(src), TTS_OUT=str(out), TTS_RATE=str(self.rate),
                       TTS_VOICE=self.voice or "")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", self.SCRIPT],
                               env=env, capture_output=True, text=True, timeout=120)
            if r.returncode != 0 or not out.exists():
                raise RuntimeError(f"System.Speech failed: {r.stderr.strip()[-500:]}")
            return read_wav(out)


def make_engine(a):
    order = ["kokoro", "piper", "sapi"] if a.engine == "auto" else [a.engine]
    errors = []
    for name in order:
        try:
            if name == "kokoro":
                return Kokoro(a.voice, a.speed)
            if name == "piper":
                return Piper(a.piper_model, a.speed)
            return Sapi(a.voice or SAPI_VOICE, a.speed)
        except Exception as e:           # try the next engine, report all if none works
            errors.append(f"  {name}: {type(e).__name__}: {e}")
    raise SystemExit("no text-to-speech engine available:\n" + "\n".join(errors) +
                     "\ninstall one (see video/requirements.txt)")


# ── wav io ────────────────────────────────────────────────────────────────────────────

def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        sr, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype).astype(np.float32)
    x = (x - 128.0) / 128.0 if width == 1 else x / float(np.iinfo(dtype).max)
    return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), sr


def write_wav(path, samples, sr):
    peak = float(np.abs(samples).max()) or 1.0
    x = samples * (10 ** (PEAK_DBFS / 20.0) / peak)
    pcm = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return len(pcm) / sr


def main():
    a = parse_args()
    data = load_shots()
    shots = select_shots(data["shots"], a.only)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    durations = json.loads(DURATIONS_PATH.read_text()) if DURATIONS_PATH.exists() else {}

    engine = None
    for shot in shots:
        text = (shot.get("speak") or shot.get("narration") or "").strip()
        wav = AUDIO_DIR / f"{shot['id']}.wav"
        if not text:
            durations.pop(shot["id"], None)
            print(f"{shot['id']}: no narration")
            continue
        engine = engine or make_engine(a)
        key = hashlib.sha1(f"{engine.name}|{a.voice}|{a.speed}|{text}".encode()).hexdigest()
        if wav.exists() and meta.get(shot["id"]) == key and not a.force:
            print(f"{shot['id']}: unchanged ({durations.get(shot['id'], 0):.2f}s)")
            continue
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        parts, spans, t = [], [], 0.0
        for k, ln in enumerate(lines):
            x, sr = engine(ln)
            if k:
                parts.append(np.zeros(int(LINE_PAUSE_S * sr), np.float32))
                t += LINE_PAUSE_S
            parts.append(np.asarray(x, np.float32))
            spans.append((round(t, 3), round(t + len(x) / sr, 3)))
            t += len(x) / sr
        samples = np.concatenate(parts)
        durations[shot["id"]] = round(write_wav(wav, samples, sr), 3)
        if len(spans) > 1:
            durations[line_spans_key(shot["id"])] = spans
        else:
            durations.pop(line_spans_key(shot["id"]), None)
        meta[shot["id"]] = key
        print(f"{shot['id']}: {durations[shot['id']]:.2f}s  [{engine.name}]  {len(lines)} line(s)  {text[:60]!r}")

    DURATIONS_PATH.write_text(json.dumps(durations, indent=2), encoding="utf-8")
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    total = sum(durations.get(s["id"], 0.0) for s in data["shots"])
    print(f"narration total {total:.1f}s -> {DURATIONS_PATH}")


if __name__ == "__main__":
    main()
