#!/usr/bin/env python3
"""
review_key.py -- Casefile Week 9: read the answer key next to the corpus text it came from

For the owner's review of answer_key.txt. check_key.py proves that every source, evidence
phrase and must: text exists in the corpus; it cannot tell whether an ANSWER is true. This
script puts each answer beside the passage it was written from, so you can compare them.
It only reads: it never changes the key or the corpus and never runs a model.

    python review_key.py                  list the entries
    python review_key.py q02              one entry: question, answer, must: text, source passage
    python review_key.py q02 q03 q07      several entries
    python review_key.py all | less       every entry, paged (space = next page, q = quit)
    python review_key.py q02 --short      only the lines around each evidence phrase
    python review_key.py q12 --around 1   also the chunk before and the chunk after each passage
    python review_key.py --find "ethical hack"
                                          count the chunks that contain this text and show the
                                          first few (use it to re-check a decline question)

Needs check_key.py in the same folder; run it from the project root.
A passage is one chunk of corpus/chunks.jsonl, shown whole; the lines that hold an evidence
phrase are marked >>. A claim in the answer can sit in the chunk before or after it: then use
--around 1, or --find to search the whole corpus.

For each entry ask yourself:
  1. Is every claim in the answer written in the passage shown? (If not, the key is wrong.)
  2. Does the passage say more than the answer, or something different? (Then fix the answer.)
  3. Would a correct answer really have to contain the must: text?
To change an entry, edit answer_key.txt, then run  python check_key.py  again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections import Counter

try:
    from check_key import CHUNKS_FILE, KEY_FILE, TEXT_KEYS, Corpus, _pick, load_key, squash
except ImportError as e:
    sys.exit(f"error: review_key.py needs check_key.py (the Week 9 file) in the same folder ({e}).")

WIDTH = 100
MAX_ROWS = 26       # with --short: rows of source text shown per chunk
CONTEXT = 3         # with --short: lines shown before and after a line that holds a phrase


def load_chunks(path):
    """-> (corpus, raw, section): the corpus as check_key.py reads it, plus each chunk's text with
    its line breaks and its section heading. All three are numbered the same way."""
    corpus, raw, section = Corpus(path), [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and corpus.add(obj):
                raw.append(_pick(obj, TEXT_KEYS))
                section.append(_pick(obj, ("section",)))
    return corpus, raw, section


def fill(text, indent="", first=None):
    return textwrap.fill(text, width=WIDTH, initial_indent=indent if first is None else first,
                         subsequent_indent=indent, break_on_hyphens=False)


def clip(text, n):
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def passage(text, phrases, short=False):
    """Rows of source text. The lines that hold a phrase are marked >>; with short=True only
    the lines around them are shown."""
    lines = text.splitlines() or [text]
    marked = set()
    for phrase in phrases:
        s = squash(phrase)
        found = False
        for span in (1, 2, 3):                   # a phrase may run over a line break
            for i in range(len(lines) - span + 1):
                if s in squash(" ".join(lines[i:i + span])):
                    marked.update(range(i, i + span))
                    found = True
                    break
            if found:
                break

    def rows(show):
        out, last = [], None
        for i in sorted(show):
            line = lines[i].rstrip()
            if short and not line.strip():
                continue
            if last is not None and i > last + 1 and short:
                out.append("       ...")
            last = i
            if i in marked:
                for phrase in phrases:
                    if phrase in line:
                        line = line.replace(phrase, ">>> " + phrase + " <<<", 1)
            pieces = textwrap.wrap(line, WIDTH - 10, break_on_hyphens=False) or [""]
            for j, piece in enumerate(pieces):
                out.append("    " + (">>" if i in marked else "| ") + " " + piece if j == 0
                           else "       " + piece)
        return out

    if not short or not marked:
        return rows(range(len(lines)))
    for ctx in range(CONTEXT, -1, -1):           # shrink the context until it fits
        show = set()
        for i in marked:
            show.update(range(max(0, i - ctx), min(len(lines), i + ctx + 1)))
        out = rows(show)
        if len(out) <= MAX_ROWS:
            return out
    return out[:MAX_ROWS] + ["       ...  (cut; leave out --short to see the whole chunk)"]


def neighbours(corpus, i, n):
    """Chunk i and up to n chunks before and after it in the same file, in order."""
    order = corpus.files[corpus.srcs[i]]
    k = order.index(i)
    return order[max(0, k - n): k + n + 1]


def show(q, corpus, raw, section, short, around=0, hint=False):
    p = print
    p("=" * WIDTH)
    p(f"{q.id}   {q.expect.upper()}   (line {q.line} of the key)")
    p(fill("QUESTION: " + q.question, indent=" " * 10, first=""))
    p("-" * WIDTH)
    p("ANSWER IN THE KEY:")
    for line in q.answer.splitlines() or [""]:
        p(fill(line, indent="  ") if line.strip() else "")
    if q.must:
        p()
        p(fill("A CORRECT ANSWER MUST CONTAIN:  " + "   |   ".join(q.must), indent="    ", first=""))
    p()
    if q.expect == "decline":
        p("This is a DECLINE entry: the corpus should NOT hold the answer. Re-check it yourself with")
        p('    python review_key.py --find "a key word from the question"')
        p()
        return
    files = []
    for written in q.sources:
        found, close = corpus.resolve(written)
        if not found:
            p(f"SOURCE: {written}   <- NOT FOUND in the corpus (python check_key.py says more)")
        for name in found:
            if name not in files:
                files.append(name)
            p(f"SOURCE: {name}")
    if not q.evidence:
        p("(This entry has no evidence: line, so there is no passage to show. Search the source file.)")
    found, marks = [], {}                        # evidence -> its first chunk; chunk -> [(number, phrase)]
    for n, phrase in enumerate(q.evidence, 1):
        hits = corpus.find(phrase, within=files)
        found.append((n, phrase, len(hits) - 1, hits[0] if hits else None))
        if hits:
            marks.setdefault(hits[0], []).append((n, phrase))
    if found:
        p()
    for n, phrase, more, i in found:
        note = ("   <- NOT FOUND in the listed source (python check_key.py says more)" if i is None
                else f"   (the phrase is in {more} more chunk(s) of that source too)" if more else "")
        p(fill(f'EVIDENCE {n}: "{phrase}"{note}', indent="    ", first=""))
    for j in sorted({k for i in marks for k in neighbours(corpus, i, around)}):
        p()
        p("  from " + clip(corpus.srcs[j] + (" > " + section[j] if section[j] else ""), WIDTH - 8))
        if j in marks:
            p("  (this chunk holds evidence " + " and ".join(str(n) for n, _ in marks[j]) + ")")
        else:
            p("  (a neighbouring chunk of the same file, not an evidence chunk)")
        for row in passage(raw[j], [phrase for _, phrase in marks.get(j, [])], short):
            p(row)
    p()
    if hint and marks and not around:
        p("A claim in the answer that you cannot find above may sit in the chunk before or after it:")
        p("    python review_key.py " + q.id + " --around 1")
        p()


def find_text(term, corpus):
    s = squash(term).lower()
    if not s:
        print("Give some text to look for, for example:  python review_key.py --find snort")
        return
    hits = [i for i, t in enumerate(corpus.texts) if s in t.lower()]
    per_file = Counter(corpus.srcs[i] for i in hits)
    print(f'"{term}": found in {len(hits)} chunk(s) of {len(per_file)} file(s) '
          f"(searched {len(corpus.texts)} chunks, ignoring capital letters).")
    if not hits:
        print("No chunk contains it. If a question can only be answered with this, it should be a decline.")
        return
    print("Files (chunks in each):")
    for name, count in per_file.most_common(8):
        print(f"  {count:>4}  {name}")
    if len(per_file) > 8:
        print(f"        ... and {len(per_file) - 8} more file(s)")
    print("First matches:")
    shown = set()
    for i in hits:
        if corpus.srcs[i] in shown:
            continue
        shown.add(corpus.srcs[i])
        t = corpus.texts[i]
        k = t.lower().find(s)
        snippet = t[max(0, k - 90): k + len(s) + 90]
        print(f"  {corpus.srcs[i]}")
        print(fill("..." + snippet + "...", indent="      "))
        if len(shown) >= 4:
            break


def listing(questions, path):
    declines = sum(q.expect == "decline" for q in questions)
    print(f"{path}: {len(questions)} entries ({len(questions) - declines} answer, {declines} decline).")
    print()
    for q in questions:
        print(f"  {q.id:<4}  {q.expect:<7}  {clip(q.question, 80)}")
    print()
    print("Read one entry:   python review_key.py q02")
    print("Read them all:    python review_key.py all | less")
    print('Re-check a decline:   python review_key.py --find "ethical hack"')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Read the Casefile answer key next to the corpus text it came from (read-only).",
        epilog='Text that starts with a dash needs an equals sign:  --find="-Pn"')
    ap.add_argument("what", nargs="*", help="entry ids (q02 q05 ...) or all; nothing lists the entries")
    ap.add_argument("--find", metavar="TEXT", help="count the corpus chunks that contain TEXT and show the first few")
    ap.add_argument("--short", action="store_true", help="show only the lines around each evidence phrase")
    ap.add_argument("--around", type=int, default=0, metavar="N",
                    help="also show the N chunks before and after each evidence chunk (default 0)")
    ap.add_argument("--key", default=KEY_FILE, help=f"the key file (default {KEY_FILE})")
    ap.add_argument("--chunks", default=CHUNKS_FILE, help=f"the chunk file (default {CHUNKS_FILE})")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    chosen = []
    if args.find is None:
        try:
            questions = load_key(args.key)
        except OSError as e:
            print(f"error: cannot read {args.key} ({e.strerror or e})")
            return 2
        except ValueError as e:
            print(f"error: {e}")
            return 2
        if not args.what:
            listing(questions, args.key)
            return 0
        if [w.lower() for w in args.what] == ["all"]:
            chosen = questions
        else:
            by_id = {q.id.lower(): q for q in questions}
            for w in args.what:
                key = w.lower()
                if key.isdigit():
                    key = f"q{int(key):02d}"
                if key not in by_id:
                    print(f"error: no entry {w!r}. The entries are: {', '.join(q.id for q in questions)}")
                    return 2
                chosen.append(by_id[key])

    if not os.path.isfile(args.chunks):
        print(f"error: {args.chunks} not found. Run this from the project root "
              "(cd ~/projects/PenetrationTesting).")
        return 2
    try:
        corpus, raw, section = load_chunks(args.chunks)
    except OSError as e:
        print(f"error: cannot read {args.chunks} ({e.strerror or e})")
        return 2
    if args.find is not None:
        find_text(args.find, corpus)
        return 0
    for q in chosen:
        show(q, corpus, raw, section, args.short, max(0, args.around), hint=len(chosen) <= 3)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:        # piped into less or head, which closed early
        sys.exit(0)
