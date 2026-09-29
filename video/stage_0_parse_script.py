"""Stage 0: read the narration script (.md/.txt or .docx) and check it against video/shots.json.

The script alternates visual cues in square brackets with the lines to be spoken:

    [show WVS on monkey]
    Stipple patterns are traditionally generated ...
    [show figure 4 5-disks baseline - not ours]
    Learned methods focus on ...

This stage splits it into (cue, narration) pairs and writes them to
video/build/script_parsed.json. It then compares every narration with the shot of the same
cue order in shots.json and reports the ones that differ, so an edited script is never
silently out of sync with the video. With --update it copies the script's narration into
shots.json (visual specs are kept; only "narration" changes).

    python video/stage_0_parse_script.py
    python video/stage_0_parse_script.py --update
"""

import argparse
import difflib
import html
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD_DIR, SHOTS_PATH, load_shots, save_shots  # noqa: E402

SCRIPT_PATH = "video/script/stippling_script.md"   # .md/.txt or .docx
OUTPUT_PATH = str(BUILD_DIR / "script_parsed.json")
UPDATE = False
# Shots with these cues are not in the .docx and are left untouched by --update.
UNSCRIPTED_CUES = ["title"]
# Paragraphs that stand for their own shot although the script gives them no [cue].
CUELESS_LINES = {"thank you!": "end"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--script", default=SCRIPT_PATH, help="Narration script (.md, .txt or .docx)")
    p.add_argument("--output", default=OUTPUT_PATH, help="Parsed (cue, narration) pairs")
    p.add_argument("--update", action=argparse.BooleanOptionalAction, default=UPDATE,
                   help="Copy the script narration into shots.json")
    return p.parse_args()


def text_paragraphs(path):
    """Non-empty lines of a plain-text / Markdown script, normalized like the .docx reader."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = re.sub(r"\s+", " ", line.replace("\u2013", "-").replace("\u2014", "-")).strip()
        if text:
            out.append(text)
    return out


def script_paragraphs(path):
    return docx_paragraphs(path) if Path(path).suffix.lower() == ".docx" else text_paragraphs(path)


def docx_paragraphs(path):
    """Plain-text paragraphs of a .docx, read straight from its XML (no python-docx needed)."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    out = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        text = html.unescape(text).replace("–", "-").replace("—", "-")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append(text)
    return out


def split_cues(paragraphs):
    """[(cue, narration)]. Text before the first cue is the title; text after a cue until the
    next one is that cue's narration (several paragraphs are joined)."""
    pairs, cue, lines = [], "title", []
    for p in paragraphs:
        m = re.fullmatch(r"\[(.*)\]", p)
        if m:
            pairs.append((cue, "\n".join(lines)))
            cue, lines = m.group(1).strip(), []
        elif p.lower() in CUELESS_LINES:
            pairs.append((cue, "\n".join(lines)))
            cue, lines = CUELESS_LINES[p.lower()], [p]
        else:
            lines.append(p)
    pairs.append((cue, "\n".join(lines)))
    return [(c, n) for c, n in pairs if n or c != "title"]


def main():
    a = parse_args()
    if not Path(a.script).exists():
        raise SystemExit(f"script not found: {a.script}  (put it there or pass --script)")
    pairs = split_cues(script_paragraphs(a.script))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps([{"cue": c, "narration": n} for c, n in pairs],
                                         indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(pairs)} (cue, narration) pairs -> {a.output}")

    data = load_shots()
    shots = [s for s in data["shots"] if s["cue"] not in UNSCRIPTED_CUES]
    scripted = [(c, n) for c, n in pairs if c != "title"]
    if len(shots) != len(scripted):
        print(f"WARNING: shots.json has {len(shots)} scripted shots, the script has {len(scripted)} cues;"
              f" comparing the first {min(len(shots), len(scripted))} in order.")

    changed = 0
    for shot, (cue, text) in zip(shots, scripted):
        # compare words only: spacing, case, punctuation and x_t vs xt do not count as edits
        # line breaks count (they split subtitles and narration lines), other whitespace does not
        norm = lambda s: " / ".join(" ".join(re.findall(r"[a-z0-9]+", ln.lower().replace("_", "")))  # noqa: E731
                                    for ln in s.split("\n") if ln.strip())
        if norm(shot["narration"]) == norm(text):
            continue
        changed += 1
        ratio = difflib.SequenceMatcher(None, norm(shot["narration"]), norm(text)).ratio()
        print(f"\n[{shot['id']}] narration differs from the script (similarity {ratio:.2f})")
        print(f"  script : {text}")
        print(f"  shots  : {shot['narration']}")
        if a.update:
            shot["narration"] = text
            shot.pop("speak", None)   # a pronunciation override would now be stale
    if a.update and changed:
        save_shots(data)
        print(f"\nupdated {changed} narration(s) in {SHOTS_PATH}; re-check any removed 'speak' overrides")
    elif not changed:
        print("shots.json narration matches the script")


if __name__ == "__main__":
    main()
