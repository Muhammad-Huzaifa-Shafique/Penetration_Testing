#!/usr/bin/env python3
"""
check_key.py -- Casefile Week 9: check the hand-written answer key

The key is written by hand, with no AI help. This script checks only its FORMAT and the
RULES around it. It never runs retrieve.py, Ollama or any model, and it never suggests a
question or an answer.

    python check_key.py                  check answer_key.txt against corpus/chunks.jsonl
    python check_key.py other_key.txt    check another file
    python check_key.py --format-only    skip the corpus checks (no chunks.jsonl needed)
    python check_key.py --selftest       run the built-in tests (made-up tools only)

The format is explained at the top of answer_key.txt. In short, one block per question:

    id: q01
    question: ...       one line, exactly as you would type it into Casefile
    expect: answer      or decline (the corpus does not hold the answer)
    source: ...         the corpus file that holds the answer (repeatable)
    evidence: ...       optional: a phrase copied from that file (repeatable)
    must: ...           optional: text a correct answer has to contain (repeatable)
    answer:
    ...                 the expected answer, any number of lines

What it reports
  ERROR  a block with no question or answer; a repeated id; an expect: that is not answer or
         decline; an answer question with no source; template text still in place (<...>);
         a source that is not a file in the corpus (the closest names are listed)
  WARN   evidence that is in no chunk, or only in files not listed as sources; evidence in so
         many chunks that it pins down nothing; must: text that is not in the listed sources;
         a question with a long dash (autocorrect may have turned -- into one)
  SMOKE  a question that looks like a smoke-test topic from Weeks 6-8. The key must not use
         them (handoff, Section 6). If a match is wrong, ignore it.

The corpus checks read corpus/chunks.jsonl, the chunks all three indexes were built from.
Dashes, curly quotes, ** and ` are evened out before comparing, so autocorrect or a copy
from rendered Markdown does not break a match.

Weeks 10 and 11 read the key through this file:

    from check_key import load_key
    for q in load_key("answer_key.txt"):
        print(q.id, q.question, q.expect, q.sources, q.evidence, q.must, q.answer)
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

KEY_FILE = "answer_key.txt"
CHUNKS_FILE = os.path.join("corpus", "chunks.jsonl")

FIELDS = ("id", "question", "expect", "source", "evidence", "must", "answer")
REPEATABLE = {"source": "sources", "evidence": "evidence", "must": "must"}   # field -> list attribute
EXPECT_VALUES = ("answer", "decline")
MANY_CHUNKS = 20      # evidence found in more chunks than this does not pin down a passage

# where a line of chunks.jsonl may keep the chunk text and its file (top level or under "metadata")
TEXT_KEYS = ("text", "document", "content", "chunk", "page_content")
SOURCE_KEYS = ("source", "path", "file")
LEVELS = ("ERROR", "SMOKE", "WARN", "NOTE")


# --- evening out text before comparing ----------------------------------------------------
# Phones and Word turn "--" into a long dash and " into curly quotes. Every run of dashes
# becomes one "-", every quote a plain one, ** and ` are dropped, and white space collapses.
# Written with \u escapes so this file stays plain ASCII.
_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_TRANS = str.maketrans({
    **{c: "-" for c in _DASH_CHARS},
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u2026": "...",
})
_MARKUP = re.compile(r"[*`]")
_DASH_RUN = re.compile(r"-{2,}")


def squash(text):
    """Text as the corpus checks compare it."""
    s = _MARKUP.sub("", str(text).translate(_TRANS))
    return " ".join(_DASH_RUN.sub("-", s).split())


_EXT_CUT = re.compile(r"^(.+?\.(?:md|txt|markdown|rst|html?))(?::.*)?$", re.I)
_LINE_CUT = re.compile(r":\d+(?::.*)?$")


def norm_path(path):
    """A corpus path as the checker compares it: / separators, nothing up to and including
    corpus/, no leading ./, and no ':123:...' tail copied from a grep line."""
    p = str(path).translate(_TRANS).strip().strip("\"'").replace("\\", "/")
    m = _EXT_CUT.match(p)
    p = m.group(1) if m else _LINE_CUT.sub("", p)
    i = p.find("corpus/")
    if i != -1:
        p = p[i + len("corpus/"):]
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


# --- the key --------------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    line: int                                     # the line of its id:
    question: str = ""
    expect: str = "answer"
    sources: list = field(default_factory=list)   # as written in the key
    evidence: list = field(default_factory=list)
    must: list = field(default_factory=list)
    answer: str = ""
    files: list = field(default_factory=list)     # the sources as found in the corpus (check_corpus fills it)
    at: dict = field(default_factory=dict)        # field -> line number(s), for messages
    stats: dict = field(default_factory=dict)     # field -> [found, given], from check_corpus
    _answer_lines: list = field(default_factory=list, repr=False)


@dataclass
class Finding:
    level: str          # ERROR, SMOKE, WARN or NOTE
    line: int           # 0 when it is about the file as a whole
    message: str
    q: int = -1         # index of the question it is about, -1 for the file as a whole


_FIELD_LINE = re.compile(r"^([A-Za-z][A-Za-z_-]*)\s*:(.*)$")
_PLACEHOLDER = re.compile(r"\s*<[^<>\n]*>\s*")


def _is_placeholder(value):
    return _PLACEHOLDER.fullmatch(value) is not None


def parse_key(text):
    """-> (questions, findings): the blocks of a key file and its format / rule problems.
    Needs no corpus."""
    questions, findings = [], []
    q, current, last_key = None, None, None

    def add(level, n, message, qi=None):
        findings.append(Finding(level, n, message, len(questions) - 1 if qi is None else qi))

    lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for n, raw in enumerate(lines, 1):
        if raw.startswith("#"):
            continue
        line = raw.rstrip()
        m = None if raw[:1].isspace() else _FIELD_LINE.match(line)
        key = m.group(1).lower() if m else None

        if key in FIELDS:
            value = m.group(2).strip()
            last_key = key
            if key == "id":
                q = Question(id=value, line=n)
                questions.append(q)
                current = None
                if not value:
                    add("ERROR", n, "id: is empty; give the question a short label such as q01")
                continue
            if q is None:
                add("ERROR", n, f"{key}: comes before the first id: line (every block starts with id:)", -1)
                continue
            if key in REPEATABLE:
                current = None
                if value:
                    getattr(q, REPEATABLE[key]).append(value)
                    q.at.setdefault(key, []).append(n)
                continue
            q.at.setdefault(key, []).append(n)
            if len(q.at[key]) > 1:
                add("ERROR", n, f"a second {key}: in this block (the first is on line {q.at[key][0]})")
                current = None
            elif key == "answer":
                q._answer_lines = [value] if value else []
                current = "answer"
            else:
                setattr(q, key, value)
                current = None
            continue

        if not line.strip():
            if current == "answer":
                q._answer_lines.append("")
            continue
        if q is None:
            add("ERROR", n, "text before the first id: line (every block starts with id:)", -1)
            continue
        if current == "answer":
            q._answer_lines.append(line)
        elif m:
            add("ERROR", n, f"'{m.group(1)}:' is not a field (the fields are {', '.join(FIELDS)})")
        elif last_key == "question":
            add("ERROR", n, "the question has to be on one line; join this line to the question: line")
        else:
            add("ERROR", n, "this line belongs to no field (only answer: can run over several lines)")

    seen = {}
    for qi, q in enumerate(questions):
        q.answer = "\n".join(q._answer_lines).strip("\n")
        q.expect = (q.expect or "answer").strip().lower()
        first = {k: v[0] for k, v in q.at.items()}

        def flag(level, n, message):
            findings.append(Finding(level, n, message, qi))

        if q.id:
            if q.id in seen:
                flag("ERROR", q.line, f"id {q.id} is used twice (first on line {seen[q.id]})")
            else:
                seen[q.id] = q.line
        if not q.question:
            flag("ERROR", q.line, "no question: line")
        elif _is_placeholder(q.question):
            flag("ERROR", first["question"], f"the question is still the template text {q.question}")
        if q.expect not in EXPECT_VALUES:
            flag("ERROR", first.get("expect", q.line), f"expect: has to be answer or decline, not '{q.expect}'")
        if not q.answer.strip():
            flag("ERROR", first.get("answer", q.line),
                 "no answer text (write it after answer:, on the same line or the lines below)")
        elif _is_placeholder(q.answer):
            flag("ERROR", first["answer"], f"the answer is still the template text {q.answer.strip()}")
        for key, attr in REPEATABLE.items():
            for value, n in zip(getattr(q, attr), q.at.get(key, [])):
                if _is_placeholder(value):
                    flag("ERROR", n, f"{key}: is still the template text {value}")
        if q.expect == "answer" and not q.sources:
            flag("ERROR", q.line, "no source: line (every question that expects an answer needs one)")
        if q.expect == "decline":
            if q.sources:
                flag("WARN", first["source"],
                     "expect: decline, but a source is given (a decline question is one the corpus does not answer)")
            if q.evidence:
                flag("WARN", first["evidence"], "expect: decline, but evidence is given")
        if any(c in _DASH_CHARS for c in q.question):
            flag("WARN", first.get("question", q.line),
                 "the question has a long or non-ASCII dash (autocorrect?); Casefile is asked exactly "
                 "this text, so retype it as - or --")
        for topic in smoke_topics(q.question):
            flag("SMOKE", first.get("question", q.line),
                 f"looks like the smoke-test topic \"{topic}\"; the key must not use it "
                 "(if the match is wrong, ignore this line)")
    return questions, findings


def read_text(path):
    """-> (text, encoding): UTF-8 with or without a BOM, UTF-16 with a BOM, else Windows-1252."""
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), "cp1252"


def load_key(path=KEY_FILE):
    """The questions in a key file, for Weeks 10 and 11. Raises ValueError if the file has
    format errors (python check_key.py lists them)."""
    text, _ = read_text(path)
    questions, findings = parse_key(text)
    errors = [f for f in findings if f.level == "ERROR"]
    if errors:
        e = errors[0]
        raise ValueError(f"{path} has {len(errors)} format error(s), the first on line {e.line}: "
                         f"{e.message}. Run python check_key.py for the full list.")
    return questions


# --- the smoke-test topics of Weeks 6-8 (handoff Section 6, plus the three from Week 8) ------

def _ci(pattern):
    rx = re.compile(pattern, re.I)
    return lambda q: rx.search(q) is not None


def _cs(pattern):            # case-sensitive: -sV is not -sv
    rx = re.compile(pattern)
    return lambda q: rx.search(q) is not None


def _all(*tests):
    return lambda q: all(t(q) for t in tests)


def _any(*tests):
    return lambda q: any(t(q) for t in tests)


def _not(test):
    return lambda q: not test(q)


_SSH, _NMAP, _FWD = _ci(r"\bssh\b"), _ci(r"\bnmap\b"), _ci(r"forward|tunnel")
_TCP, _UDP, _SCAN, _PORT = _ci(r"\btcp\b"), _ci(r"\budp\b"), _ci(r"\bscan"), _ci(r"\bports?\b")

SMOKE_TOPICS = [
    ("SUID privilege escalation", _ci(r"\bsuid\b|\bsetuid\b")),
    ("nmap: scan every TCP port", _any(_cs(r"(?<![\w-])-p-(?![\w-])"), _ci(r"\b65,?[35]35\b"),
                                       _all(_NMAP, _ci(r"\b(?:every|all)\b"), _PORT))),
    ("Kerberoasting", _any(_ci(r"kerberoast"),
                           _all(_ci(r"\bcrack"), _ci(r"\btgs\b|service ticket|\bspn\b")))),
    ("SSH local port forwarding", _all(_SSH, _any(_all(_ci(r"\blocal\b"), _FWD), _cs(r"(?<![\w-])-L\b")))),
    ("enumerate SMB shares", _all(_ci(r"\bshares?\b"),
                                  _ci(r"\bsmb|samba|cifs|netbios|crackmapexec|netexec|\bnxc\b|net view|windows"))),
    ("sourdough (the off-topic control)", _ci(r"sourdough")),
    ("nmap -sV (version detection)", _any(_cs(r"(?<![\w-])-s[VR]\b"),
                                          _all(_NMAP, _ci(r"\bversions?\b"), _ci(r"detect|service")))),
    ("John the Ripper", _ci(r"\bjohn\b|the ripper")),
    ("ffuf -mc", _any(_cs(r"(?<![\w-])-mc\b"), _all(_ci(r"\bffuf\b"), _ci(r"\bmatch")))),
    ("nmap UDP scan", _any(_cs(r"(?<![\w-])-sU\b"), _all(_UDP, _NMAP), _all(_UDP, _SCAN, _PORT))),
    ("TCP and UDP in one nmap run", _all(_TCP, _UDP, _any(_NMAP, _SCAN))),
    ("SSH remote port forwarding", _all(_SSH, _not(_ci(r"\bagent")),
                                        _any(_all(_ci(r"\b(?:remote|reverse)\b"), _FWD),
                                             _cs(r"(?<![\w-])-R\b")))),
]


def smoke_topics(question):
    """The smoke-test topics a question looks like (a long dash typed for -- still counts)."""
    q = question.translate(_TRANS)
    return [name for name, test in SMOKE_TOPICS if test(q)]


# --- the corpus -----------------------------------------------------------------------------

@dataclass
class Corpus:
    path: str
    texts: list = field(default_factory=list)   # chunk text, squashed
    srcs: list = field(default_factory=list)    # the normalised file of each chunk
    files: dict = field(default_factory=dict)   # normalised file -> chunk numbers
    skipped: int = 0

    def add(self, obj):
        """Add one parsed line of chunks.jsonl. False if no chunk text was found on it."""
        text, source = _pick(obj, TEXT_KEYS), _pick(obj, SOURCE_KEYS)
        if not text:
            self.skipped += 1
            return False
        src = norm_path(source) if source else "?"
        self.files.setdefault(src, []).append(len(self.texts))
        self.texts.append(squash(text))
        self.srcs.append(src)
        return True

    def resolve(self, written):
        """-> (matches, closest): the corpus files a source: line names, or the nearest names."""
        want = norm_path(written)
        if not want:
            return [], []
        if want in self.files:
            return [want], []
        found = sorted(p for p in self.files if p.endswith("/" + want) or want.endswith("/" + p))
        if found:
            return found, []
        close = difflib.get_close_matches(want, list(self.files), n=3, cutoff=0.6)
        if not close:
            base = want.rsplit("/", 1)[-1]
            close = sorted(p for p in self.files if p.rsplit("/", 1)[-1] == base)[:3]
        return [], close

    def find(self, phrase, within=None):
        """Chunk numbers whose text contains the phrase (both squashed), optionally only in `within`."""
        s = squash(phrase)
        if not s:
            return []
        if within is None:
            return [i for i, t in enumerate(self.texts) if s in t]
        nums = sorted(i for p in within for i in self.files.get(p, ()))
        return [i for i in nums if s in self.texts[i]]


def _pick(obj, keys):
    meta = obj.get("metadata")
    if not isinstance(meta, dict):
        meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    for k in keys:
        for c in (obj, meta):
            v = c.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def load_corpus(path=CHUNKS_FILE):
    corpus, keys = Corpus(path), None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                corpus.skipped += 1
                continue
            if not isinstance(obj, dict):
                corpus.skipped += 1
                continue
            if keys is None:
                meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
                keys = sorted(obj) + [f"metadata.{k}" for k in sorted(meta)]
            corpus.add(obj)
    if not corpus.texts:
        raise ValueError(f"found no chunk text in {path} (a line has the keys: {', '.join(keys or [])}). "
                         "Add the key that holds the text to TEXT_KEYS near the top of check_key.py.")
    return corpus


def check_corpus(questions, corpus):
    """Look up each source, evidence phrase and must: text in the corpus. -> findings"""
    findings = []
    for qi, q in enumerate(questions):
        def flag(level, n, message):
            findings.append(Finding(level, n, message, qi))

        q.files, q.stats = [], {"source": [0, 0], "evidence": [0, 0], "must": [0, 0]}
        if q.expect != "answer":
            continue
        for value, n in zip(q.sources, q.at.get("source", [])):
            if _is_placeholder(value):
                continue
            q.stats["source"][1] += 1
            found, close = corpus.resolve(value)
            if len(found) == 1:
                q.stats["source"][0] += 1
                if found[0] not in q.files:
                    q.files.append(found[0])
            elif found:
                flag("ERROR", n, f"source '{value}' matches {len(found)} files; write more of the path "
                                 f"(for example {found[0]})")
            else:
                hint = f"; closest: {', '.join(close)}" if close else ""
                flag("ERROR", n, f"source '{value}' is not a file in {corpus.path}{hint}")

        for value, n in zip(q.evidence, q.at.get("evidence", [])):
            if _is_placeholder(value):
                continue
            q.stats["evidence"][1] += 1
            hits = corpus.find(value)
            if not hits:
                flag("WARN", n, "evidence is in no chunk: check for a typo, or copy a shorter phrase from "
                                "one line of the corpus file itself (not from a web page)")
                continue
            files = sorted({corpus.srcs[i] for i in hits})
            if not q.files:
                continue                   # no source was found to compare with; the source ERROR says why
            if not set(files) & set(q.files):
                more = " ..." if len(files) > 2 else ""
                flag("WARN", n, f"evidence is only in {', '.join(files[:2])}{more}, not in a listed source")
                continue
            q.stats["evidence"][0] += 1
            if len(hits) > MANY_CHUNKS:
                flag("WARN", n, f"evidence is in {len(hits)} chunks, so it does not pin down one passage; "
                                "a longer phrase would")

        for value, n in zip(q.must, q.at.get("must", [])):
            if _is_placeholder(value):
                continue
            q.stats["must"][1] += 1
            if not q.files:
                continue                   # no source was found to look in; the source ERROR says why
            if corpus.find(value, within=q.files):
                q.stats["must"][0] += 1
                continue
            low = squash(value).lower()
            caps = any(low in corpus.texts[i].lower() for p in q.files for i in corpus.files[p])
            flag("WARN", n, f"must text '{value}' is not in the listed source(s)"
                            + (" (it is there with different capitals)" if caps else "; check the spelling"))
    return findings


# --- printing -------------------------------------------------------------------------------

def _clip(text, n=60):
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _frac(pair):
    found, given = pair
    return f"{found}/{given}" if given else "-"


def report(path, questions, findings, corpus=None, out=sys.stdout):
    def say(text=""):
        print(text, file=out)

    where = (f"checked against {corpus.path} ({len(corpus.texts)} chunks, {len(corpus.files)} files)"
             if corpus is not None else "format only (no corpus checks)")
    say(f"{path}: {len(questions)} question(s), {where}")
    rank = {lvl: i for i, lvl in enumerate(LEVELS)}
    groups = {}
    for f in findings:
        groups.setdefault(f.q, []).append(f)
    for f in sorted(groups.get(-1, []), key=lambda f: (rank[f.level], f.line)):
        say(f"  {f.level:<5} " + (f"line {f.line}: " if f.line else "") + f.message)
    say()
    if not questions:
        say("  No questions yet: a block starts with an id: line (see the top of answer_key.txt).")
    w = max([len(q.id or "?") for q in questions] + [4])
    for qi, q in enumerate(questions):
        if corpus is not None and q.expect == "answer" and q.stats:
            cols = (f"src {_frac(q.stats['source'])}  ev {_frac(q.stats['evidence'])}  "
                    f"must {_frac(q.stats['must'])}")
        elif q.expect == "answer":
            cols = f"src {len(q.sources)}  ev {len(q.evidence)}  must {len(q.must)}"
        else:
            cols = ""
        say(f"  {q.id or '?':<{w}}  {q.expect:<7}  {cols:<26}  {_clip(q.question)}")
        for f in sorted(groups.get(qi, []), key=lambda f: (rank[f.level], f.line)):
            say(f"  {'':<{w}}  {f.level:<5} line {f.line}: {f.message}")

    count = Counter(f.level for f in findings)
    declines = sum(q.expect == "decline" for q in questions)
    say()
    say(f"{len(questions)} question(s): {len(questions) - declines} answer, {declines} decline.  "
        f"{count['ERROR']} error(s), {count['SMOKE']} smoke-test match(es), {count['WARN']} warning(s).")
    if corpus is not None and questions:
        say("Columns: src = sources found in the corpus, ev = evidence found in a listed source, "
            "must = must: text found in a listed source.")


# --- self-test (made-up tools and files only) ----------------------------------------------

def _selftest():
    import tempfile

    rows = [
        {"id": "a", "text": "ZORP(1)\n\nNAME\n       zorp - count the widgets",
         "metadata": {"source": "manpages/zorp.man.txt", "source_type": "manpage", "tool": "zorp"}},
        {"id": "b", "text": "OPTIONS\n       -q, --quiet\n              Print nothing but errors. Use it in\n"
                            "              scripts.", "metadata": {"source": "manpages/zorp.man.txt"}},
        {"id": "c", "text": "## Frobnicate the widget\n\nRun **`frob --deep`** to frobnicate every widget "
                            "in the tree. It's \"safe\".", "metadata": {"source": "hacktricks/src/misc/frobnicate.md"}},
        {"id": "d", "text": "# Misc\n\nNothing to see here.", "metadata": {"source": "hacktricks/src/misc/README.md"}},
        {"id": "e", "text": "# Other\n\nPrint nothing but errors.", "metadata": {"source": "hacktricks/src/other/README.md"}},
        {"id": "f", "text": "A flat line keeps its source at the top level.", "source": "hacktricks/src/flat/page.md"},
    ]
    corpus = Corpus("(test corpus)")
    for r in rows:
        corpus.add(r)

    good = (
        "# a comment line\n"
        "id: t1\n"
        "question: What does the zorp -q flag do?\n"
        "source: corpus/manpages/zorp.man.txt\n"
        "evidence: Print nothing but errors.\n"
        "must: -q\n"
        "answer:\n"
        "It makes zorp quiet.\n"
        "Note: errors still print.\n"
        "\n"
        "  source: an indented line stays in the answer\n"
        "\n"
        "id: t2\n"
        "question: How do I bake a cake with frob?\n"
        "expect: Decline\n"
        "answer: The corpus has nothing on cakes.\n"
    )
    template = ("id: q01\nquestion: <your question>\nexpect: answer\nsource: <corpus file>\n"
                "evidence: <a phrase copied from that file>\nmust: <text a correct answer must contain>\n"
                "answer:\n<your expected answer>\n")
    smoke = [  # every phrasing used in Weeks 5-8, and the topic it must be flagged as
        ("Which nmap option scans every TCP port?", "nmap: scan every TCP port"),
        ("How do I set up SSH local port forwarding?", "SSH local port forwarding"),
        ("How does Kerberoasting work, and how do I crack the service ticket offline?", "Kerberoasting"),
        ("How can SUID binaries be used for privilege escalation on Linux?", "SUID privilege escalation"),
        ("linux privilege escalation via SUID", "SUID privilege escalation"),
        ("What does the nmap -sV option do?", "nmap -sV (version detection)"),
        ("How do I crack password hashes with John the Ripper?", "John the Ripper"),
        ("What does the ffuf -mc flag do?", "ffuf -mc"),
        ("how do I enumerate SMB shares", "enumerate SMB shares"),
        ("How do I make sourdough bread?", "sourdough (the off-topic control)"),
        ("What nmap options scan every UDP port?", "nmap UDP scan"),
        ("Can I scan both TCP and UDP ports in one nmap run?", "TCP and UDP in one nmap run"),
        ("What is the syntax for SSH remote port forwarding?", "SSH remote port forwarding"),
        ("How do I set up SSH remote port forwarding?", "SSH remote port forwarding"),
        ("What does the nmap \u2014sV option do?", "nmap -sV (version detection)"),
    ]

    def parse(text):
        return parse_key(text)

    def levels(findings, level):
        return [f for f in findings if f.level == level]

    def corpus_check(text):
        qs, fs = parse_key(text)
        return qs, fs + check_corpus(qs, corpus)

    def one(src_line="source: manpages/zorp.man.txt", extra=""):
        return f"id: x\nquestion: What does zorp do?\n{src_line}\n{extra}answer: It counts widgets.\n"

    def load_key_raises():
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("id: x\nquestion: Q?\n")
            try:
                load_key(p)
            except ValueError:
                return True
            return False

    cases = [
        ("a well-formed key parses with no errors, warnings or smoke matches",
         lambda: (lambda qs, fs: len(qs) == 2 and not [f for f in fs if f.level != "NOTE"])(*parse(good))),
        ("a multi-line answer keeps a 'Note:' line and an indented 'source:' line",
         lambda: (lambda q: q.answer.splitlines()[1] == "Note: errors still print."
                  and q.answer.endswith("an indented line stays in the answer") and q.sources == [
                      "corpus/manpages/zorp.man.txt"])(parse(good)[0][0])),
        ("expect: is read case-insensitively and an answer can sit on the answer: line",
         lambda: (lambda q: q.expect == "decline" and q.answer == "The corpus has nothing on cakes.")(parse(good)[0][1])),
        ("Windows line endings and a BOM give the same result",
         lambda: [(q.id, q.answer) for q in parse("\ufeff" + good.replace("\n", "\r\n"))[0]]
         == [(q.id, q.answer) for q in parse(good)[0]]),
        ("the untouched template gives five template-text errors",
         lambda: len([f for f in levels(parse(template)[1], "ERROR") if "template text" in f.message]) == 5),
        ("a missing answer, a repeated id and a bad expect: are errors",
         lambda: (lambda fs: any("no answer" in f.message for f in fs) and any("used twice" in f.message for f in fs)
                  and any("expect:" in f.message for f in fs))(
             levels(parse("id: a\nquestion: Q?\nsource: s\nexpect: maybe\n\nid: a\nquestion: Q2?\nsource: s\nanswer: A\n")[1],
                    "ERROR"))),
        ("an answer question without a source is an error; a decline needs none",
         lambda: any("no source" in f.message for f in parse("id: a\nquestion: Q?\nanswer: A\n")[1])
         and not parse("id: a\nquestion: Q?\nexpect: decline\nanswer: A\n")[1]),
        ("text before the first id:, an unknown field and a wrapped question are errors",
         lambda: (lambda fs: any("before the first id" in f.message for f in fs)
                  and any("not a field" in f.message for f in fs) and any("one line" in f.message for f in fs))(
             parse("stray text\nid: a\nquestion: Q is\nlong?\nnote: x\nsource: s\nanswer: A\n")[1])),
        ("every smoke-test phrasing of Weeks 5-8 is flagged with the right topic",
         lambda: all(topic in smoke_topics(q) for q, topic in smoke)),
        ("questions about made-up tools are not flagged",
         lambda: not smoke_topics("What does the zorp -q flag do?")
         and not smoke_topics("How do I frobnicate every widget with frob --deep?")),
        ("a long dash in a question is a warning",
         lambda: any("dash" in f.message for f in levels(parse("id: a\nquestion: What does zorp \u2014quiet do?\n"
                                                                "source: s\nanswer: A\n")[1], "WARN"))),
        ("a source is found from corpus/..., an absolute path, a grep -n line and a bare file name",
         lambda: all(corpus.resolve(s)[0] == ["manpages/zorp.man.txt"] for s in (
             "corpus/manpages/zorp.man.txt", "/home/u/p/corpus/manpages/zorp.man.txt",
             "corpus/manpages/zorp.man.txt:12:       -q, --quiet", "manpages/zorp.man.txt", "zorp.man.txt"))),
        ("a mistyped source is an error that names the closest file; an ambiguous one names a match",
         lambda: (lambda fs: "closest: manpages/zorp.man.txt" in fs[0].message)(
             levels(corpus_check(one("source: manpages/zrop.man.txt"))[1], "ERROR"))
         and "matches 2 files" in levels(corpus_check(one("source: README.md"))[1], "ERROR")[0].message),
        ("evidence is found across a line wrap, without ** and `, and with autocorrected dashes and quotes",
         lambda: all(corpus.find(p) for p in ("Use it in scripts.", "Run frob --deep to frobnicate",
                                               "Run frob \u2014deep to frobnicate", "It\u2019s \u201csafe\u201d."))),
        ("evidence in no chunk, or only in an unlisted file, is a warning",
         lambda: any("in no chunk" in f.message for f in corpus_check(one(extra="evidence: Print nothing at all\n"))[1])
         and any("not in a listed source" in f.message
                 for f in corpus_check(one("source: hacktricks/src/misc/README.md",
                                           "evidence: Print nothing but errors.\n"))[1])),
        ("must: text that is not in the listed source is a warning; text that is there is counted",
         lambda: (lambda qs, fs: qs[0].stats["must"] == [1, 2] and any("--loud" in f.message for f in fs))(
             *corpus_check(one(extra="must: -q\nmust: --loud\n")))),
        ("a full good block scores src 1/1, ev 1/1, must 1/1",
         lambda: (lambda qs, fs: qs[0].stats == {"source": [1, 1], "evidence": [1, 1], "must": [1, 1]}
                  and not [f for f in fs if f.level != "NOTE"])(*corpus_check(good))),
        ("chunk text and source are read from the top level or from metadata",
         lambda: corpus.files.get("hacktricks/src/flat/page.md") == [5] and corpus.srcs[0] == "manpages/zorp.man.txt"),
        ("load_key refuses a file with format errors",
         load_key_raises),
    ]
    failed = 0
    for name, test in cases:
        try:
            ok = bool(test())
        except Exception as e:  # noqa: BLE001
            ok, name = False, f"{name}  ({type(e).__name__}: {e})"
        failed += not ok
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return failed


# --- command line ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the hand-written Casefile answer key (Week 9). "
                                             "Format and rules only; it never runs a model.")
    ap.add_argument("key", nargs="?", default=KEY_FILE, help=f"the key file (default {KEY_FILE})")
    ap.add_argument("--chunks", default=CHUNKS_FILE,
                    help=f"the chunk file the indexes were built from (default {CHUNKS_FILE})")
    ap.add_argument("--format-only", action="store_true", help="check the format only; no corpus needed")
    ap.add_argument("--selftest", action="store_true", help="run the built-in tests and exit")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    if args.selftest:
        return 1 if _selftest() else 0
    try:
        text, encoding = read_text(args.key)
    except OSError as e:
        print(f"error: cannot read {args.key} ({e.strerror or e})")
        return 2
    questions, findings = parse_key(text)
    if encoding != "utf-8":
        findings.insert(0, Finding("NOTE", 0, f"the file was read as {encoding}; saving it as UTF-8 is safer"))

    corpus = None
    if not args.format_only:
        if not os.path.isfile(args.chunks):
            print(f"error: {args.chunks} not found. Run this from the project root "
                  "(cd ~/projects/PenetrationTesting), or add --format-only.")
            return 2
        try:
            corpus = load_corpus(args.chunks)
        except (OSError, ValueError) as e:
            print(f"error: {e}")
            return 2
        findings += check_corpus(questions, corpus)

    report(args.key, questions, findings, corpus)
    return 1 if any(f.level in ("ERROR", "SMOKE") for f in findings) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:        # output piped into head or less, which closed early
        sys.exit(0)
