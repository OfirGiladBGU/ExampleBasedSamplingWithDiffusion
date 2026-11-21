#!/usr/bin/env python3
"""Mark in blue the text of a rebuttal tex file that differs from the submitted version.

    python3 mark_changes_blue.py <submitted.tex | -> <rebuttal.tex> [-o out.tex]
    python3 mark_changes_blue.py --strip - <rebuttal.tex> [-o out.tex]      # remove the marks

Every piece of text that is new or changed against the submitted file is wrapped in
\\textcolor{blue}{...}. "-" as the submitted file means "everything is new" (a new table).
The script first removes the blue marks it finds in the rebuttal file, so it can be run again
after further edits (it is idempotent, and --strip gives back the unmarked file).

How it works
  * lines are aligned with difflib; an unchanged line stays as it is;
  * inside a changed block, a line is paired with the submitted line that has the same first table
    cell / first words (in order), otherwise with the most similar one; a line with no similar
    submitted line is treated as new;
  * a paired line is compared word by word. A "word" is extended until its braces and $...$ are
    balanced, so a mark never cuts through an argument or a formula;
  * bold / underline around a table value is ignored in the comparison: when columns are added the
    best-value marks move, but the value did not change;
  * structure is never wrapped: comments, \\begin / \\end, \\includegraphics, \\input, \\label,
    \\item, \\multicolumn, \\hline, &, \\\\ and any other command that is not known inline text.
    A new figure therefore shows its caption in blue, not the picture;
  * \\caption{...}, \\section{...}, \\subsection{...} and \\paragraph{...} are opened and their
    text is marked inside the braces;
  * display mathematics and the preamble are left unmarked.
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

MARK = r"\textcolor{blue}{"
# commands whose token may sit inside a blue mark
INLINE = ("cite", "citet", "citep", "ref", "eqref", "textbf", "emph", "textit", "textsc", "underline",
          "texttt", "textrm", "url", "footnote")
WRAPPERS = ("caption", "section", "subsection", "subsubsection", "paragraph")
MATH_ENVS = ("equation", "equation*", "align", "align*", "eqnarray", "eqnarray*", "displaymath", "multline",
             "multline*", "gather", "gather*")
SIMILAR = 0.5      # an unpaired changed line is compared with a submitted line only above this similarity
COMMENT = re.compile(r"(?<!\\)%")
CELL_SEP = re.compile(r"(?<!\\)&")
EMPHASIS = re.compile(r"^\\(?:textbf|underline)\{(.*)\}$", re.S)
WRAP_RE = re.compile(r"^(\s*\\(%s)\*?\{)(.*)$" % "|".join(WRAPPERS), re.S)
GLUED = re.compile(r"(\\\\(?:\[[^\]]*\])?|(?<!\\)&)")


# ── removing existing marks ───────────────────────────────────────────────────

def strip_marks(text):
    out, i = [], 0
    while True:
        j = text.find(MARK, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        line_start = text.rfind("\n", 0, j) + 1
        if COMMENT.search(text[line_start:j]):            # inside a comment: not one of ours
            out.append(text[i:j + len(MARK)])
            i = j + len(MARK)
            continue
        out.append(text[i:j])
        k, depth = j + len(MARK), 1
        while depth:
            c = text[k]
            if c == "\\":
                k += 2
                continue
            depth += (c == "{") - (c == "}")
            k += 1
        out.append(strip_marks(text[j + len(MARK):k - 1]))
        i = k


# ── tokens ────────────────────────────────────────────────────────────────────

def balance(s):
    """(open braces, inside math?) at the end of s, ignoring escaped characters."""
    depth, math, i = 0, False, 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "$":
            math = not math
        i += 1
    return depth, math


def split_comment(line):
    m = COMMENT.search(line)
    return (line[:m.start()], line[m.start():]) if m else (line, "")


def tokens(text):
    """(leading whitespace, [(token, whitespace after it)]); tokens are balanced where possible."""
    parts = [p for p in re.split(r"(\s+)", text) if p]
    lead = ""
    if parts and parts[0].isspace():
        lead, parts = parts[0], parts[1:]
    toks, cur = [], ""
    for p in parts:
        if p.isspace():
            if cur and balance(cur) == (0, False):
                toks.append([cur, p])
                cur = ""
            elif cur:
                cur += p
            elif toks:
                toks[-1][1] += p
            continue
        if not cur and balance(p) == (0, False):
            # a cell separator or a line break glued to a word becomes its own token
            pieces = [x for x in GLUED.split(p) if x]
            if len(pieces) > 1 and all(balance(x) == (0, False) for x in pieces):
                toks.extend([x, ""] for x in pieces)
                continue
        cur += p
    if cur:
        toks.append([cur, ""])
    return lead, [(a, b) for a, b in toks]


def is_barrier(tok):
    """True for a token that must not be wrapped."""
    if balance(tok) != (0, False):
        return True
    if tok in ("&", "}", "{") or tok.startswith("\\\\"):
        return True
    if tok.startswith("\\"):
        m = re.match(r"\\([A-Za-z]+)\*?", tok)
        return bool(m) and m.group(1) not in INLINE      # \, \% \& ... count as text
    return False


def norm(tok):
    """Token as compared: bold / underline around a value is ignored."""
    while True:
        m = EMPHASIS.match(tok)
        if not m or balance(m.group(1)) != (0, False):
            return tok
        tok = m.group(1)


# ── marking one line ──────────────────────────────────────────────────────────

def mark_text(new, old):
    """new with its changed tokens wrapped; old is the submitted text, or None (everything new)."""
    lead, ntoks = tokens(new)
    if not ntoks:
        return new
    changed = [True] * len(ntoks)
    if old is not None:
        _, otoks = tokens(old)
        sm = difflib.SequenceMatcher(None, [norm(t) for t, _ in otoks], [norm(t) for t, _ in ntoks], autojunk=False)
        for tag, _, _, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for j in range(j1, j2):
                    changed[j] = False
    out, j = [lead], 0
    while j < len(ntoks):
        tok, ws = ntoks[j]
        if not changed[j] or is_barrier(tok):
            out.append(tok + ws)
            j += 1
            continue
        k = j
        while k + 1 < len(ntoks) and changed[k + 1] and not is_barrier(ntoks[k + 1][0]):
            k += 1
        body = "".join(t + w for t, w in ntoks[j:k]) + ntoks[k][0]
        out.append(MARK + body + "}" + ntoks[k][1])
        j = k + 1
    return "".join(out)


def peel(code):
    """(prefix, text, suffix, command) for a line that starts with \\caption{ etc.; the closing brace
    may be on this line or on a later one."""
    m = WRAP_RE.match(code)
    if not m:
        return None
    rest = m.group(3)
    if balance(rest) == (0, False):                       # closed on a later line
        return m.group(1), rest, "", m.group(2)
    tail = re.search(r"\}\s*$", rest)
    if tail and balance(rest[:tail.start()]) == (0, False):
        return m.group(1), rest[:tail.start()], rest[tail.start():], m.group(2)
    return None


def mark_line(new_line, old_line):
    nl = "\n" if new_line.endswith("\n") else ""
    code, comment = split_comment(new_line[:len(new_line) - len(nl)])
    if not code.strip():
        return new_line
    old_code = split_comment(old_line.rstrip("\n"))[0] if old_line is not None else None
    p = peel(code)
    if p:
        pre, inner, post, name = p
        old_inner = None
        if old_code is not None:
            q = peel(old_code)
            old_inner = q[1] if q and q[3] == name else old_code
        return pre + mark_text(inner, old_inner) + post + comment + nl
    return mark_text(code, old_code) + comment + nl


# ── whole file ────────────────────────────────────────────────────────────────

def row_key(line):
    """What identifies a line inside a changed block: its first table cell, or its first words."""
    code = split_comment(line)[0].strip()
    if CELL_SEP.search(code):
        return CELL_SEP.split(code)[0].strip()
    return code[:25]


def mark_file(old_text, new_text):
    """(marked text, statistics, unmarked text)."""
    old_lines = old_text.splitlines(keepends=True)
    clean = strip_marks(new_text)
    new_lines = clean.splitlines(keepends=True)
    cand = {}                                              # new line index -> submitted lines to compare with
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
        if tag == "insert":
            for j in range(j1, j2):
                cand[j] = []
        elif tag == "replace":
            block = old_lines[i1:i2]
            for j in range(j1, j2):
                cand[j] = block
            keys = difflib.SequenceMatcher(None, [row_key(x) for x in block],
                                           [row_key(x) for x in new_lines[j1:j2]], autojunk=False)
            for t2, a1, a2, b1, _ in keys.get_opcodes():
                if t2 == "equal":
                    for d in range(a2 - a1):
                        if row_key(block[a1 + d]):
                            cand[j1 + b1 + d] = [block[a1 + d]]
    out, env_math = [], 0
    stats = {"changed_lines": 0, "new_lines": 0, "skipped_math": 0}
    in_document = "\\begin{document}" not in clean         # a table file has no preamble
    for j, line in enumerate(new_lines):
        s = line.strip()
        starts_document = "\\begin{document}" in line
        in_document = in_document or starts_document
        opened = sum(("\\begin{%s}" % e) in line for e in MATH_ENVS)
        closed = sum(("\\end{%s}" % e) in line for e in MATH_ENVS)
        in_math = env_math > 0 or opened > 0 or s in ("\\[", "\\]")
        env_math += opened - closed
        if j not in cand or not in_document or starts_document or not s or s.startswith("%"):
            out.append(line)
            continue
        if in_math:
            stats["skipped_math"] += 1
            out.append(line)
            continue
        best, best_r = None, 0.0
        for o in cand[j]:
            r = difflib.SequenceMatcher(None, o, line, autojunk=False).ratio()
            if r > best_r:
                best, best_r = o, r
        if best is not None and (best_r >= SIMILAR or len(cand[j]) == 1):
            stats["changed_lines"] += 1
            out.append(mark_line(line, best))
        else:
            stats["new_lines"] += 1
            out.append(mark_line(line, None))
    return "".join(out), stats, clean


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submitted", help="submitted tex file, or - if the whole rebuttal file is new")
    ap.add_argument("rebuttal")
    ap.add_argument("-o", "--output", default=None, help="default: overwrite the rebuttal file")
    ap.add_argument("--strip", action="store_true", help="only remove the blue marks")
    a = ap.parse_args()
    with open(a.rebuttal, encoding="utf-8", newline="") as f:
        new_text = f.read()
    if a.strip:
        result, stats = strip_marks(new_text), {}
    else:
        old_text = "" if a.submitted == "-" else Path(a.submitted).read_text(encoding="utf-8")
        result, stats, clean = mark_file(old_text, new_text)
        assert strip_marks(result) == clean, "internal error: removing the marks does not give back the input"
        for n, (x, y) in enumerate(zip(result.splitlines(), clean.splitlines()), 1):
            assert balance(split_comment(x)[0]) == balance(split_comment(y)[0]), f"balance changed on line {n}"
    with open(a.output or a.rebuttal, "w", encoding="utf-8", newline="\n") as f:   # Python 3.9: no newline= in write_text
        f.write(result)
    print(f"{Path(a.rebuttal).name}: {result.count(MARK)} blue marks {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
