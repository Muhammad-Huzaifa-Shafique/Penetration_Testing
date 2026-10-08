#!/usr/bin/env python3
"""
sources.py - Casefile Week 7: make every answer prove its work.

Standard library only (no Ollama, no Chroma), so it can be tested anywhere.
answer.py calls check_answer() once the model has finished; running
`python sources.py --audit` checks the whole index.

What the check CAN tell you
  - every [n] in the answer points at a passage that was really retrieved;
  - which passages the answer cites, and the real file (and line) each came from;
  - whether a command, flag or number written in a cited sentence appears in the
    passage it cites, only in a different passage, or in none of them.

What it CANNOT tell you
  - whether a sentence is TRUE, or means what the passage means. "-sV is an
    alias for -sR" passes a text match when the passage says the reverse.
    Reading the cited lines is your job; Weeks 9-11 measure it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MAX_SCAN_BYTES = 20_000_000   # never read a bigger file than this just to find a line
EXCERPT_CHARS = 100           # how much of the chunk to quote under each source
MAX_SHOWN = 6                 # tokens listed per finding before "+N more"
MAX_FINDINGS = 10             # findings printed under an answer

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

# Corpus files and model output are untrusted text. Before printing any of it,
# drop control characters (ANSI escapes) and bidi overrides, and put it on one line.
_CTRL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f‪-‮⁦-⁩]")
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "−": "-"})


def clean(value, limit=None):
    """One line of terminal-safe text, shortened to `limit` characters."""
    s = _CTRL.sub("", "" if value is None else str(value))
    s = " ".join(s.split())
    if limit and len(s) > limit:
        s = s[: max(1, limit - 3)].rstrip() + "..."
    return s


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _nums(numbers):
    return ", ".join(f"[{n}]" for n in numbers)


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    """One retrieved chunk as the model saw it. `n` is the number it was given: [n]."""
    n: int
    text: str
    source: str = ""            # as stored in the index, e.g. hacktricks/src/.../README.md
    section: str = ""
    similarity: float | None = None
    chunk_index: int | None = None
    source_type: str = ""
    tool: str = ""


@dataclass
class Finding:
    level: str                  # ERROR, WARN, NOTE, INFO or OK
    text: str
    near: str = ""              # the sentence the finding is about, shortened


@dataclass
class SourceRef:
    n: int
    source: str                 # exactly as stored in the index
    path: str                   # what to open (relative to the project when possible)
    exists: bool
    line: int | None = None     # line where the chunk starts
    line_exact: bool = False    # False -> shown as ~LINE (start only approximate)
    section: str = ""
    similarity: float | None = None
    chunk_index: int | None = None
    excerpt: str = ""
    problem: str = ""           # "", "file not found" or "chunk text not found in file"


@dataclass
class CheckResult:
    n_passages: int
    cited: list                                   # valid [n] the answer used, ascending
    invalid: list                                 # [n] that were never given to the model
    not_cited: list                               # retrieved but not cited
    findings: list = field(default_factory=list)
    sources: list = field(default_factory=list)       # SourceRef for each cited passage
    uncited_refs: list = field(default_factory=list)  # short SourceRef for the rest
    checked_tokens: int = 0
    unsupported_tokens: int = 0

    @property
    def errors(self):
        return sum(1 for f in self.findings if f.level == "ERROR")

    @property
    def warnings(self):
        return sum(1 for f in self.findings if f.level == "WARN")

    @property
    def ok(self):
        return self.errors == 0 and self.warnings == 0


# ---------------------------------------------------------------------------
# finding [n] markers in an answer
# ---------------------------------------------------------------------------

_KW = r"(?:passages?|sources?|chunks?|excerpts?|references?|refs?|documents?|docs?)"
_ITEM = rf"(?:{_KW}\s*[:#.]?\s*)?\d{{1,3}}"
_SEP = r"\s*(?:,|;|&|/|\band\b|\bor\b|-|–|—|\bto\b)\s*"
_BODY = rf"{_ITEM}(?:{_SEP}{_ITEM})*"
_SQUARE = re.compile(rf"[\[【]\s*({_BODY})\s*[\]】]", re.I)
_ROUND = re.compile(
    rf"\(\s*({_KW}\s*[:#.]?\s*\d{{1,3}}(?:{_SEP}{_ITEM})*)\s*\)", re.I)
_MARK = (rf"(?:[\[【]\s*{_BODY}\s*[\]】]"
         rf"|\(\s*{_KW}\s*[:#.]?\s*\d{{1,3}}(?:{_SEP}{_ITEM})*\s*\))")
_BARE = re.compile(r"\b(?:passages?|excerpts?|sources?)\s*[:#]?\s*(\d{1,3}(?:\s*(?:,|&|and)\s*\d{1,3})*)\b(?![.,]?\d)",
                   re.I)
_BOUNDARY = re.compile(rf"[.!?]+[\"')]*(?:\s*{_MARK})*(?=\s|$)", re.I)
_LISTPREFIX = re.compile(r"\s*(?:[-*•]|\d+[.)])\s+")


@dataclass
class Marker:
    start: int
    end: int
    numbers: list


def _numbers_from(body):
    """'1, 3' -> [1, 3];  '2-4' -> [2, 3, 4];  'Passage 2 and 5' -> [2, 5]."""
    out = []
    prev = None
    ranged = False
    for t in re.findall(r"\d+|[-–—]|\bto\b", body, re.I):
        if t.isdigit():
            n = int(t)
            if ranged and prev is not None and prev < n <= prev + 20:
                out.extend(range(prev + 1, n + 1))
            else:
                out.append(n)
            prev, ranged = n, False
        else:
            ranged = True
    return out


def find_markers(masked):
    found = []
    for rx in (_SQUARE, _ROUND):
        for m in rx.finditer(masked):
            nums = _numbers_from(m.group(1))
            if nums:
                found.append(Marker(m.start(), m.end(), nums))
    for m in _BARE.finditer(masked):                     # "Passage 2 says ..." without brackets
        if not any(k.start < m.end() and m.start() < k.end for k in found):
            nums = _numbers_from(m.group(1))
            if nums:
                found.append(Marker(m.start(), m.end(), nums))
    found.sort(key=lambda k: k.start)
    return found


# ---------------------------------------------------------------------------
# code spans / fenced blocks (never parsed for citations)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
_SPAN = re.compile(r"`([^`\n]+)`")
_SHELLISH = {"", "bash", "sh", "shell", "console", "zsh", "terminal", "cmd", "bat",
             "powershell", "ps1", "pwsh", "text", "txt", "plaintext", "none", "shell-session"}


@dataclass
class CodeRegion:
    start: int
    end: int
    text: str
    fenced: bool
    lang: str = ""


def split_code(text):
    """Return (masked, regions): `text` with code replaced by spaces (same length,
    newlines kept) and the code regions that were blanked out."""
    out = list(text)
    regions = []

    def blank(a, b):
        for i in range(a, b):
            if out[i] != "\n":
                out[i] = " "

    for m in _FENCE.finditer(text):
        inner = text[m.start() + 3: m.end()]
        if inner.endswith("```"):
            inner = inner[:-3]
        first, sep, rest = inner.partition("\n")
        lang = ""
        if sep and re.fullmatch(r"[\w+.#-]{0,20}\s*", first):
            lang, inner = first.strip(), rest
        regions.append(CodeRegion(m.start(), m.end(), inner, True, lang))
        blank(m.start(), m.end())
    for m in _SPAN.finditer("".join(out)):
        regions.append(CodeRegion(m.start(), m.end(), m.group(1), False))
        blank(m.start(), m.end())
    regions.sort(key=lambda r: r.start)
    return "".join(out), regions


# ---------------------------------------------------------------------------
# technical tokens: the commands, flags and numbers a sentence claims
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    text: str
    kind: str                   # "flag", "name" or "number"

    @property
    def hard(self):             # numbers are 'soft': example ports etc. are common
        return self.kind != "number"


_FLAG_PROSE = re.compile(r"(?<![\w\-])(--?[A-Za-z][A-Za-z0-9_\-]*|-[0-9]{3,})")
_NUM_PROSE = re.compile(r"(?<![\w.,\-])(\d{1,3}(?:,\d{3})+|\d{3,})(?!\w|[.,]\d)")
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?$")
_FILENAME = re.compile(r"^[\w\-]+\.[A-Za-z0-9]{1,5}$")
_RESET = {"|", "||", "&&", ";", "&"}
_PROMPTS = {"$", "#", ">", "%", "PS>", ">>>"}
_WRAPPERS = {"sudo", "time", "nohup", "env", "exec", "doas"}
# generic shell plumbing: using it is never a claim about the topic
_COMMON = {"echo", "cat", "ls", "cd", "grep", "head", "tail", "sort", "uniq", "wc", "awk",
           "sed", "cut", "tr", "xargs", "export", "bash", "sh", "python", "python3", "pip",
           "tee", "mkdir", "rm", "cp", "mv", "touch", "pwd", "man", "less", "more", "true",
           "false", "if", "then", "fi", "for", "do", "done", "while", "set", "unset", "source"}
# stand-ins a model writes for "your value here"
_PLACEHOLDER = {"target", "host", "hostname", "ip", "ipaddress", "user", "username", "password",
                "pass", "domain", "dc", "file", "filename", "path", "dir", "directory", "port",
                "url", "wordlist", "hash", "hashes", "output", "input", "server", "client",
                "network", "interface", "iface", "range", "cidr", "subnet", "name", "value",
                "string", "command", "cmd", "payload", "key", "token", "session", "share",
                "sharename", "remote", "local", "example", "localhost", "rhost", "lhost",
                "rport", "lport", "victim", "attacker", "address", "email", "pattern", "text",
                "root", "admin", "administrator", "guest", "anonymous", "null", "none", "all",
                "any", "yes", "no"}


def _value_like(tok):
    """True for things that are the user's own values, not claims about a tool."""
    if len(tok) < 2 or tok.startswith("."):
        return True
    if any(c in tok for c in "<>{}$*?:@|;=\\\"'/%~#()[]"):
        return True
    if _IPV4.match(tok):
        return True
    if tok.isupper() and len(tok) >= 3:       # TARGET, NTLM, RC4 ...
        return True
    return tok.lower() in _PLACEHOLDER


def code_tokens(s, first=True, flags_only=False):
    """Commands, flags and numbers in one line of code. Placeholders, paths,
    addresses and file names are the user's values and are left out."""
    out = []
    for raw in s.split():
        if raw in _RESET:
            first = True
            continue
        if raw == "\\":
            continue
        if first and raw in _PROMPTS:
            continue
        t = raw.strip("\"'`,;()[]{}")
        if not t:
            continue
        if first and t.lower() in _WRAPPERS:
            continue
        if t.startswith("-") and len(t) >= 2:
            flag = t.split("=", 1)[0].rstrip(".,:")
            if re.fullmatch(r"-{1,2}[A-Za-z0-9][A-Za-z0-9_\-]*", flag) and not re.fullmatch(r"-\d", flag):
                out.append(Token(flag, "flag"))
            first = False
            continue
        if flags_only:
            first = False
            continue
        if re.fullmatch(r"\d{3,}", t):
            out.append(Token(t, "number"))
        elif first:
            if t.lower() not in _COMMON and not _value_like(t):
                out.append(Token(t, "name"))
        elif (("-" in t.strip("-")) or "_" in t) and not _value_like(t) and not _FILENAME.match(t):
            out.append(Token(t, "name"))               # e.g. an nmap script: smb-enum-shares
        first = False
    return out


def _code_block_tokens(region):
    flags_only = region.lang.lower() not in _SHELLISH
    out = []
    continuing = False
    for line in region.text.splitlines():
        out.extend(code_tokens(line, first=not continuing, flags_only=flags_only))
        continuing = line.rstrip().endswith("\\")
    return out


# ---------------------------------------------------------------------------
# splitting an answer into sentences that carry citations
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: int
    end: int
    text: str
    cites: list = field(default_factory=list)
    code: bool = False
    tokens: list = field(default_factory=list)
    region: CodeRegion | None = None


def _only_markers(masked, a, b, markers):
    buf = list(masked[a:b])
    for mk in markers:
        for i in range(max(a, mk.start), min(b, mk.end)):
            buf[i - a] = " "
    return not re.search(r"\w", "".join(buf))


def _push(segs, text, masked, markers, a, b):
    if not masked[a:b].strip():
        return
    a += len(masked[a:b]) - len(masked[a:b].lstrip())
    cites = []
    for mk in markers:
        if a <= mk.start < b:
            for n in mk.numbers:
                if n not in cites:
                    cites.append(n)
    if segs and not segs[-1].code and _only_markers(masked, a, b, markers):
        prev = segs[-1]                         # "...sentence. [2]" belongs to the sentence
        prev.end = b
        prev.text = text[prev.start:b]
        for n in cites:
            if n not in prev.cites:
                prev.cites.append(n)
        return
    segs.append(Segment(a, b, text[a:b], cites=cites))


def analyse(text):
    """Return (segments, markers) for an answer."""
    masked, regions = split_code(text)
    markers = find_markers(masked)
    fences = [r for r in regions if r.fenced]
    pieces, pos = [], 0
    for f in fences:
        if f.start > pos:
            pieces.append((pos, f.start, None))
        pieces.append((f.start, f.end, f))
        pos = f.end
    if pos < len(text):
        pieces.append((pos, len(text), None))

    segs = []
    for a, b, fence in pieces:
        if fence is not None:
            segs.append(Segment(a, b, text[a:b], code=True, region=fence))
            continue
        for lm in re.finditer(r"[^\n]+", masked[a:b]):
            ls, le = a + lm.start(), a + lm.end()
            pm = _LISTPREFIX.match(masked, ls, le)
            cursor = pm.end() if pm else ls
            for bm in _BOUNDARY.finditer(masked, cursor, le):
                _push(segs, text, masked, markers, cursor, bm.end())
                cursor = bm.end()
            _push(segs, text, masked, markers, cursor, le)

    spans = [r for r in regions if not r.fenced]
    for s in segs:
        if s.code:
            s.tokens = _code_block_tokens(s.region)
            continue
        toks = []
        for r in spans:
            if s.start <= r.start and r.end <= s.end:
                toks.extend(code_tokens(r.text))
        seg_masked = masked[s.start:s.end]
        toks.extend(Token(m.group(1), "flag") for m in _FLAG_PROSE.finditer(seg_masked))
        toks.extend(Token(m.group(1), "number") for m in _NUM_PROSE.finditer(seg_masked))
        s.tokens = toks

    # a code block belongs to the citation just before it, else the one just after
    for i, s in enumerate(segs):
        if not s.code:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(segs) and not segs[j].code and segs[j].cites:
                s.cites = list(segs[j].cites)
                break
    return segs, markers


# ---------------------------------------------------------------------------
# does a token appear in a passage?
# ---------------------------------------------------------------------------

class _Hay:
    def __init__(self, text):
        self.text = text.translate(_DASHES)
        self.nocomma = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", self.text)


def _has(token, hay):
    t = token.text
    if token.kind == "flag":
        return re.search(r"(?<![\w\-])" + re.escape(t), hay.text) is not None
    if token.kind == "number":
        return re.search(r"(?<![\d.])" + re.escape(t.replace(",", "")) + r"(?!\d)", hay.nocomma) is not None
    return re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", hay.text, re.I) is not None


def _dedupe(tokens):
    seen, out = set(), []
    for t in tokens:
        key = (t.kind, t.text if t.kind == "flag" else t.text.lower())
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _show(tokens):
    shown = [f"`{t.text}`" for t in tokens[:MAX_SHOWN]]
    extra = len(tokens) - MAX_SHOWN
    return ", ".join(shown) + (f" (+{extra} more)" if extra > 0 else "")


def _are(tokens):
    return "is" if len(tokens) == 1 else "are"


# ---------------------------------------------------------------------------
# turning a chunk into a file and a line you can open
# ---------------------------------------------------------------------------

_WORDY = re.compile(r"\w")


def _norm_line(s):
    return " ".join(s.translate(_DASHES).split())


class FileIndex:
    """A source file, read once, with its lines indexed for lookups."""

    def __init__(self, path):
        self.path = Path(path)
        self.lines = None
        self.index = None
        self.ok = False
        try:
            if self.path.stat().st_size <= MAX_SCAN_BYTES:
                text = self.path.read_text(encoding="utf-8", errors="replace")
                self.lines = [_norm_line(l) for l in text.splitlines()]
                self.index = defaultdict(list)
                for i, l in enumerate(self.lines):
                    if len(l) >= 3 and _WORDY.search(l):
                        self.index[l].append(i)
                self.ok = True
        except OSError:
            pass


def _align(f_lines, c_lines, p, q, span=6):
    """How many neighbouring non-blank lines around file line p / chunk line q agree."""
    score = 0
    for step in (1, -1):
        fp, cq, seen = p, q, 0
        while seen < span:
            cq += step
            while 0 <= cq < len(c_lines) and not c_lines[cq]:
                cq += step
            fp += step
            while 0 <= fp < len(f_lines) and not f_lines[fp]:
                fp += step
            if not (0 <= cq < len(c_lines)) or not (0 <= fp < len(f_lines)):
                break
            if f_lines[fp] != c_lines[cq]:
                break
            score += 1
            seen += 1
    return score


def _start_of(f_lines, c_lines, p, q):
    """Walk back from the matched pair to the chunk's first line. Returns (line, exact)."""
    firsts = [i for i, l in enumerate(c_lines) if l]
    first = firsts[0] if firsts else 0
    fp, cq = p, q
    while cq > 0:
        cq -= 1
        if not c_lines[cq]:
            continue
        nxt = fp - 1
        while nxt >= 0 and not f_lines[nxt]:
            nxt -= 1
        if nxt < 0:
            return fp + 1, False
        fl, cl = f_lines[nxt], c_lines[cq]
        # a chunk that begins with the 200-character overlap may start mid-line
        if fl == cl or (cq == first and fl.endswith(cl)):
            fp = nxt
        else:
            return fp + 1, False
    return fp + 1, True


def locate_chunk(path, chunk_text, cache=None):
    """Find where `chunk_text` sits in the file at `path`.
    Returns (line, exact). (None, False) when the text cannot be found."""
    key = str(path)
    if cache is not None and key in cache:
        fi = cache[key]
    else:
        fi = FileIndex(path)
        if cache is not None:
            if len(cache) >= 24:
                cache.pop(next(iter(cache)))
            cache[key] = fi
    if not fi.ok:
        return None, False
    c_lines = [_norm_line(l) for l in (chunk_text or "").splitlines()]
    anchors = sorted((q for q, l in enumerate(c_lines) if len(l) >= 3 and _WORDY.search(l)),
                     key=lambda q: -len(c_lines[q]))[:6]          # most distinctive lines first
    best = None                                     # (score, -p, p, q)
    for q in anchors:
        for p in fi.index.get(c_lines[q], [])[:200]:
            cand = (_align(fi.lines, c_lines, p, q), -p, p, q)
            if best is None or cand > best:
                best = cand
    if best is not None:
        _, _, p, q = best
        return _start_of(fi.lines, c_lines, p, q)
    for q in anchors:                               # the chunker may have cut a long line
        needle = c_lines[q]
        if len(needle) < 30:
            continue
        for i, l in enumerate(fi.lines):
            if needle in l:
                return i + 1, False
    return None, False


def resolve_source(source, project_root=None, corpus_root=None):
    """Return (found_path_or_None, display_path). The index stores paths relative to
    the corpus folder, so look there first, then relative to the project."""
    s = (source or "").strip().replace("\\", "/")
    if not s:
        return None, ""
    cands = []
    if s.startswith("/") or re.match(r"^[A-Za-z]:/", s):
        cands.append(Path(s))
    else:
        s = re.sub(r"^(?:\./)+", "", s)
        if corpus_root:
            cands.append(Path(corpus_root) / s)
        if project_root:
            cands.append(Path(project_root) / "corpus" / s)
            cands.append(Path(project_root) / s)
        if not cands:
            cands.append(Path(s))
    seen = set()
    chosen = None
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        try:
            if c.is_file():
                chosen = c
                break
        except OSError:
            continue
    shown = chosen or cands[0]
    if project_root:
        try:
            shown = shown.relative_to(Path(project_root))
        except ValueError:
            pass
    return chosen, shown.as_posix()


def _excerpt(text, section=""):
    sec = re.sub(r"[#*`_\s]+", " ", section or "").strip().lower()
    for line in (text or "").splitlines():
        line = line.strip()
        if len(line) < 12 or line.startswith(("{{", "---", "```", "<!--", "#")):
            continue
        if sec and re.sub(r"[#*`_\s]+", " ", line).strip().lower() == sec:
            continue                                  # the heading is shown as 'section'
        return clean(line, EXCERPT_CHARS)
    return clean(text, EXCERPT_CHARS)


def build_source_ref(p, project_root=None, corpus_root=None, locate=True, cache=None):
    found, shown = resolve_source(p.source, project_root, corpus_root)
    ref = SourceRef(n=p.n, source=clean(p.source), path=clean(shown), exists=found is not None,
                    section=clean(p.section, 90), similarity=p.similarity,
                    chunk_index=p.chunk_index, excerpt=_excerpt(p.text, p.section))
    if not p.source:
        ref.path, ref.problem = "(no source recorded in the index)", "no source in index"
    elif found is None:
        ref.problem = "file not found"
    elif locate:
        line, exact = locate_chunk(found, p.text, cache)
        if line is None:
            ref.problem = "chunk text not found in file"
        else:
            ref.line, ref.line_exact = line, exact
    return ref


# ---------------------------------------------------------------------------
# the check itself
# ---------------------------------------------------------------------------

# "The passages do not mention smbmap." is the behaviour we want; never nag about it.
_DISCLAIM = re.compile(
    r"\b(?:passages?|excerpts?|provided\s+(?:text|context|material|information)"
    r"|(?:context|text|material)\s+provided)\b"
    r".{0,50}\b(?:do(?:es)?\s*n[o'\u2019]t|did\s+not|never|not\b|no\b)"
    r"|\b(?:no|nothing|none|neither)\b.{0,40}\b(?:in|from|within)\s+(?:the|these|those|any\s+of\s+the)?\s*"
    r"(?:passages?|excerpts?|provided\s+(?:text|context))"
    r"|\bnot\s+(?:covered|mentioned|included|described|explained|specified|provided|documented|discussed|stated|listed)\b"
    r"|\b(?:no|any)\s+(?:information|mention|details?|guidance|instructions?)\s+(?:about|on|of|regarding|for)\b"
    r"|\b(?:cannot|can[n']t|unable\s+to|couldn[o']t)\s+(?:find|answer|determine|say|tell)\b",
    re.I | re.S)


def _near(seg):
    if seg.code:
        for line in seg.region.text.splitlines():
            if line.strip():
                return clean(line, 90)
    return clean(seg.text, 90)


def check_answer(answer_text, passages, project_root=None, corpus_root=None,
                 locate=True, cache=None):
    """Check one answer against the passages the model was given.

    `passages` is a list of Passage whose `n` is the number the model saw."""
    text = answer_text or ""
    passages = list(passages or [])
    by_n = {p.n: p for p in passages}
    hay = {n: _Hay(p.text or "") for n, p in by_n.items()}
    segs, markers = analyse(text)

    order = []
    for mk in markers:
        for n in mk.numbers:
            if n not in order:
                order.append(n)
    cited = sorted(n for n in order if n in by_n)
    invalid = sorted(n for n in order if n not in by_n)
    not_cited = sorted(n for n in by_n if n not in cited)

    errors, warns, notes, infos = [], [], [], []
    res = CheckResult(n_passages=len(passages), cited=cited, invalid=invalid, not_cited=not_cited)

    # 1. numbers that do not exist
    span = (f"only passages {min(by_n)}-{max(by_n)} were given to the model"
            if by_n else "no passages were given to the model")
    for n in invalid:
        near = ""
        for mk in markers:
            if n in mk.numbers:
                seg = next((s for s in segs if s.start <= mk.start < s.end), None)
                near = _near(seg) if seg else ""
                break
        errors.append(Finding("ERROR", f"[{n}] does not exist - {span}.", near))

    # 2. does each cited sentence's text really appear in what it cites?
    any_hard = _dedupe([t for s in segs for t in s.tokens
                        if t.hard and (s.code or not _DISCLAIM.search(s.text))])
    for s in segs:
        toks = _dedupe(s.tokens)
        if not toks:
            continue
        if not s.code and _DISCLAIM.search(s.text):          # says what the passages lack: not a claim
            continue
        valid = [n for n in s.cites if n in by_n]
        if not valid:
            if s.cites:                                    # only invalid numbers: ERROR covers it
                continue
            hard = [t for t in toks if t.hard]
            if markers and hard:                           # (no markers at all is reported once, below)
                notes.append(Finding(
                    "NOTE",
                    "a code block has no citation" if s.code
                    else f"a sentence with {_show(hard)} has no citation",
                    _near(s)))
            continue
        moved, ghost = [], []
        for t in toks:
            res.checked_tokens += 1
            if any(_has(t, hay[n]) for n in valid):
                continue
            res.unsupported_tokens += 1
            elsewhere = [n for n in by_n if n not in valid and _has(t, hay[n])]
            (moved if elsewhere else ghost).append((t, elsewhere))
        if moved:
            ts = [t for t, _ in moved]
            where = sorted({n for _, el in moved for n in el})
            level = "WARN" if any(t.hard for t in ts) else "NOTE"
            (warns if level == "WARN" else notes).append(Finding(
                level,
                f"cites {_nums(valid)}, but {_show(ts)} {_are(ts)} not in {_nums(valid)}. "
                f"{'It is' if len(ts) == 1 else 'They are'} in {_nums(where)} - wrong number?",
                _near(s)))
        if ghost:
            ts = [t for t, _ in ghost]
            level = "WARN" if any(t.hard for t in ts) else "NOTE"
            (warns if level == "WARN" else notes).append(Finding(
                level,
                f"cites {_nums(valid)}, but {_show(ts)} {_are(ts)} in none of the retrieved passages "
                f"- from the model's memory?",
                _near(s)))

    # 3. nothing cited at all
    if not markers:
        if any_hard:
            warns.append(Finding("WARN", f"the answer cites nothing, yet it contains {_show(any_hard)}"))
        else:
            infos.append(Finding("INFO", "the answer cites no passages"))

    oks = []
    if cited and not invalid:
        oks.append(Finding("OK", f"{_plural(len(cited), 'distinct citation')}, all pointing at passages that were retrieved"))
    if res.checked_tokens and not res.unsupported_tokens:
        oks.append(Finding("OK", f"all {res.checked_tokens} commands/flags/numbers in cited sentences "
                                 f"appear in the passages they cite"))
    res.findings = errors + warns + notes + infos + oks

    # 4. the real files behind each cited passage
    for n in cited:
        res.sources.append(build_source_ref(by_n[n], project_root, corpus_root, locate, cache))
    for n in not_cited:
        res.uncited_refs.append(build_source_ref(by_n[n], project_root, corpus_root, locate=False))
    return res


# ---------------------------------------------------------------------------
# what gets printed under every answer
# ---------------------------------------------------------------------------

def _where(ref):
    if not ref.exists and ref.problem == "file not found":
        return f"{ref.path}   <- FILE NOT FOUND on disk"
    if ref.line is None:
        extra = f"   <- {ref.problem}" if ref.problem else ""
        return f"{ref.path}{extra}"
    return f"{ref.path}:{'' if ref.line_exact else '~'}{ref.line}"


def format_report(r):
    out = ["Checks (a text match only; it cannot tell whether a sentence is true)"]
    for f in r.findings[:MAX_FINDINGS]:
        out.append(f"  {f.level:<6} {f.text}")
        if f.near:
            out.append(f'         near: "{f.near}"')
    if len(r.findings) > MAX_FINDINGS:
        out.append(f"  ...    and {len(r.findings) - MAX_FINDINGS} more checks not shown")
    if r.cited:
        out.append(f"Sources - the answer cites {len(r.cited)} of {r.n_passages} retrieved passages")
        for ref in r.sources:
            out.append(f"  [{ref.n}] {_where(ref)}")
            bits = []
            if ref.section:
                bits.append(f'section "{ref.section}"')
            if ref.similarity is not None:
                bits.append(f"similarity {ref.similarity:.3f}")
            if ref.chunk_index is not None:
                bits.append(f"chunk {ref.chunk_index}")
            if bits:
                out.append("      " + " | ".join(bits))
            if ref.excerpt:
                out.append(f'      begins: "{ref.excerpt}"')
        first = next((x for x in r.sources if x.exists and x.line), None)
        if first is not None:
            out.append(f"  Open one:  less +{first.line} {shlex.quote(first.path)}")
        if r.not_cited:
            out.append("  Retrieved but not cited: " + " ".join(f"[{n}]" for n in r.not_cited))
    elif r.n_passages == 0:
        out.append("Sources - none: no passages were retrieved for this question")
    else:
        out.append(f"Sources - none cited. The {_plural(r.n_passages, 'passage')} the model was given:")
        for ref in r.uncited_refs:
            sim = f"  (similarity {ref.similarity:.3f})" if ref.similarity is not None else ""
            out.append(f"  [{ref.n}] {ref.path}{sim}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# python sources.py --audit : do the paths in the index lead to real text?
# ---------------------------------------------------------------------------

def audit(chunks_path, project_root, corpus_root=None, sample=300, seed=7, out=print):
    chunks_path = Path(chunks_path)
    records, bad = [], 0
    with chunks_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                bad += 1
    by_source = defaultdict(list)
    for rec in records:
        by_source[str(rec.get("source", ""))].append(rec)
    found, missing = {}, []
    for src in by_source:
        f, _ = resolve_source(src, project_root, corpus_root)
        if f is not None:
            found[src] = f
        else:
            missing.append(src)

    def text_of(rec):
        for k in ("text", "document", "content", "chunk"):
            if isinstance(rec.get(k), str):
                return rec[k]
        return None

    pool = [r for r in records if str(r.get("source", "")) in found and text_of(r)]
    rng = random.Random(seed)
    picked = pool if not sample or sample >= len(pool) else rng.sample(pool, sample)
    picked.sort(key=lambda r: str(r.get("source", "")))      # one file read per source
    cache = {}
    exact = approx = 0
    lost = []
    for rec in picked:
        line, ok = locate_chunk(found[str(rec.get("source", ""))], text_of(rec), cache)
        if line is None:
            lost.append(rec)
        elif ok:
            exact += 1
        else:
            approx += 1

    out("Casefile source audit")
    out(f"  chunks file:       {chunks_path}   ({len(records):,} chunks"
        + (f", {bad} unreadable lines" if bad else "") + ")")
    out(f"  distinct sources:  {len(by_source):,}")
    out(f"  found on disk:     {len(found):,}")
    out(f"  NOT found on disk: {len(missing):,}")
    for src in missing[:8]:
        out(f"      {clean(src, 110)}")
    if len(missing) > 8:
        out(f"      ... and {len(missing) - 8} more")
    if not pool:
        out("  line lookup:       skipped (no chunk text in the file, or no source files found)")
    else:
        out(f"  line lookup:       {len(picked):,} chunks checked -> {exact:,} located exactly, "
            f"{approx:,} approximately, {len(lost):,} not found in their file")
        for rec in lost[:8]:
            out(f"      {clean(rec.get('source', ''), 100)}  (chunk {clean(rec.get('chunk_index', '?'), 10)})")
    return {"chunks": len(records), "sources": len(by_source), "found": len(found),
            "missing": missing, "checked": len(picked), "exact": exact, "approx": approx,
            "lost": lost}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Casefile source checks (Week 7).")
    ap.add_argument("--audit", action="store_true",
                    help="check that every source in the chunk file exists, and that chunk text is in it")
    ap.add_argument("--project", default=str(Path(__file__).resolve().parent),
                    help="project root (default: the folder this file is in)")
    ap.add_argument("--corpus", default=None, help="corpus folder (default: <project>/corpus)")
    ap.add_argument("--chunks", default=None, help="chunks file (default: <corpus>/chunks.jsonl)")
    ap.add_argument("--sample", type=int, default=300,
                    help="chunks to look up by line (0 = every chunk; slower)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    if not args.audit:
        ap.print_help()
        return 0
    project = Path(args.project)
    corpus = Path(args.corpus) if args.corpus else project / "corpus"
    chunks = Path(args.chunks) if args.chunks else corpus / "chunks.jsonl"
    if not chunks.is_file():
        print(f"Cannot find the chunks file: {chunks}", file=sys.stderr)
        return 2
    audit(chunks, project, corpus, sample=args.sample, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
