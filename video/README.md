# Paper video

Scripts that turn the narration script into the StippleDiffusion video. Manim draws the
point-set animations and the pipeline visuals. A local text-to-speech engine speaks the
narration. The narration length times each shot, and FFmpeg stitches the shots together.

Run everything from the project root.

## Pipeline

| Stage | Where | Script | Output |
|---|---|---|---|
| 0 | local | `stage_0_parse_script.py` | checks `shots.json` narration against `script/stippling_script.md` |
| 1 | local | `stage_1_figures.py` | `assets/figures/*.png` (pipeline + stress figures from `paper4/`) |
| 1 | local | `stage_1_collect_sources.py` | copies every input file into `assets/sources/` and points `shots.json` at it |
| 1 | local | `stage_1_wvs.py` | `assets/trajectories/wvs_monkey.npz` (Lloyd iterations) |
| 1 | local | `stage_1_results.py` | `assets/results/*.npz`: real final results, from `assets/sources/` |
| 1 | **GPU** | `stage_1_trajectories.py` | `assets/trajectories/<name>.npz` (our reverse process, every 10 steps) |
| 1 | **GPU** | `stage_1_baseline.py` | `assets/results/baseline_stress1.npz` (unconditional baseline: 4 samples, full 1000-step reverse process every 10 steps) |
| 1 | **GPU** | `stage_1_tone.py` | `assets/trajectories/tone_<stem>.npz` (tone optimization) |
| 2 | local | `stage_2_tts.py` | `build/audio/<shot>.wav`, `durations.json` |
| 3 | local | `stage_3_render.py` | `build/shots/<shot>.mp4`, silent, each as long as its narration |
| 4 | local | `stage_4_assemble.py` | `build/StippleDiffusion_video.mp4` + `.srt` |

`make_video.py` runs all the local stages in order. Until the server trajectories exist,
the trajectory shots show the model's real prior and final result (crossfaded and labeled as
such), and the tone shot shows the real uncorrected and optimized states with the real loss
curve. The step-by-step animations replace them automatically once the server files are in place.

Everything the video reads lives under `video/assets/`, so the folder is self-contained.

## Setup (local, Windows)

```
python -m venv video/.venv
video/.venv/Scripts/python -m pip install -r video/requirements.txt
```

## GPU stages (trajectories, baseline, tone snapshots)

Locally, use the conda env `qmcdiffusion` (it has CUDA torch):

```
conda run -n qmcdiffusion python video/stage_1_trajectories.py --only monkey_g24,monkey_g64
conda run -n qmcdiffusion python video/stage_1_baseline.py
```

Our trajectories use the GBN checkpoints at epoch 5000 listed under `checkpoints` in `shots.json` (Icons-50, CelebA-5K 1024 and ShapeNetRender_Custom-3K 1600, in `control_v4/train_outputs_*/checkpoints/`). `stage_1_baseline.py` needs the baseline weights in `assets/weights/GBN_stress1/` (`config.json`, `model.ckpt`, from `config_trained/GBN_stress1/` on the server).

### On the server

```
sbatch video/server/run_stage_1.sbatch          # conda env "stippling", one GPU
```

Then copy `video/assets/trajectories/*.npz` back to the local checkout. To run a subset:
`python video/stage_1_trajectories.py --only monkey_g24,monkey_g64`. To animate from pure
noise instead of the SDEdit start: `--truncation 1.0`.

## Build

```
video/.venv/Scripts/python video/make_video.py --preview     # 960x540, 15 fps
video/.venv/Scripts/python video/make_video.py               # 1920x1080, 30 fps
```

To redo only some shots:

```
video/.venv/Scripts/python video/stage_3_render.py --only s05_pipeline,s06_pipeline_initial
video/.venv/Scripts/python video/stage_4_assemble.py
```

## Editing

Everything is driven by `shots.json`.

- **`narration`:** the subtitle text. A line break in the script's cue makes a separate line: its own subtitle, voiced separately, with exact timing that highlights can follow. Stage 0 keeps it in sync with the .docx; `--update` copies edits over.
- **`speak`:** optional text for the voice when it should differ from the subtitle, e.g. "x t" for x_t.
- **`visual`:** what the shot shows.

| Visual type | Fields |
|---|---|
| `title` | `title`, `subtitle` |
| `wvs` | `data` (npz name), `label`, `counter` |
| `figure` | `image` (in `assets/figures`), `crops`: list of `{box: [x0,y0,x1,y1] (0..1, y down), label}` |
| `pipeline` | `image`, plus either `highlights` (`{at: 0..1 of the shot, box, label, off_after_line}`) or `zoom` (box) with `highlights` (`{box, on_line, color}`: shown while that narration line is spoken) |
| `trajectory` | `panels`: `{traj, label, show_target}`; optional shared `target` |
| `tone` | `data` (npz name) |

Consecutive `pipeline` zoom shots pan from one region to the next. `trajectories` lists
the sampler runs (image, checkpoint key, grid size) that stage 1 produces on the server.
`checkpoints` maps checkpoint keys to paths on the server. `settings` holds the resolution,
fps, colors, font, padding before and after the narration, and whether subtitles are burned in.

## Text to speech

`stage_2_tts.py --engine auto` uses the first engine that is available:

- **kokoro:** neural voice, best quality. Downloads its model once and runs on the CPU.
- **piper:** a lighter neural voice. Needs a voice file in `assets/tts/`.
- **sapi:** Windows' built-in voices. Nothing to install, but it sounds robotic.

Change the voice with `--voice` and the pace with `--speed`. A clip is re-synthesized only
when its text, engine or voice changes.

The video carries no author names, to keep the submission anonymous.
