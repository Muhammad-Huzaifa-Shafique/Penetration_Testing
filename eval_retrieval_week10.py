#!/usr/bin/env python3
"""
eval_retrieval.py -- Casefile Week 10: compare the indexes on the answer key

Runs every question in answer_key.txt through each index, and through a merged list of two
of them, and scores what comes back against the key. It only searches: each question is
embedded and looked up through retrieve.py, the way answer.py does it, and no chat model runs.

    python eval_retrieval.py                  every index, top 10; writes week10_retrieval.txt and .json
    python eval_retrieval.py --show q03 q17   also list what each index returned for these questions
    python eval_retrieval.py -e all-minilm    only this index (repeat -e for more)
    python eval_retrieval.py --selftest       built-in tests (no Ollama, no Chroma, no corpus needed)

What counts as a hit, for a question that expects an answer
  passage  a retrieved chunk from one of the question's source: files (directly, or through
           retrieve.py's "same text also in" list) that contains one of its evidence:
           phrases. That is the passage the answer needs, and the main measure.
  page     a retrieved chunk from one of the source: files, whatever part of the page.
Text and file names are compared the way check_key.py compares them. A question with no
evidence: line is scored by page.

For each index the report counts the answer questions with a passage in the top 1, 3, 5 and
10, and gives the mean reciprocal rank (MRR: 1/rank of the first passage, 0 when there is
none, averaged over the answer questions). Decline questions have no passage. For every
question it also records the top-1 similarity, as data for the Week 12 "nothing found"
cutoff. Similarities are not comparable across indexes.

The merged list combines the result lists of two indexes (default nomic-embed-text-prefixed
and all-minilm) by reciprocal rank fusion: a chunk scores 1/(60 + rank) in each list it is
in, the scores are added, and the best k are kept. It uses ranks only, so the indexes'
different similarity scales do not matter. Chunks with the same text count as one.

Run it from the project root with the venv active and Ollama running. It needs check_key.py,
retrieve.py (and what retrieve.py imports), answer_key.txt and corpus/chunks.jsonl. Re-run it
whenever the key changes; the files it writes are replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field

from check_key import CHUNKS_FILE, KEY_FILE, Corpus, check_corpus, load_corpus, load_key, norm_path, squash

OUT_PREFIX = "week10_retrieval"
K = 10
CUTS = (1, 3, 5, 10)
RRF_K = 60
MERGED = "merged"
MERGE_DEFAULT = ("nomic-embed-text-prefixed", "all-minilm")
ORDER = ("nomic-embed-text", "nomic-embed-text-prefixed", "all-minilm")      # report order
SHORT = {"nomic-embed-text": "nomic", "nomic-embed-text-prefixed": "nomic+pfx", "all-minilm": "minilm"}
DOC_PREFIXES = ("search_document: ", "search_query: ")


def short(index):
    return SHORT.get(index, index)


def cuts_for(k):
    """The ranks the report counts at: 1, 3, 5 and 10 below k, and k itself."""
    return tuple(sorted({c for c in CUTS if c < k} | {k}))


def in_order(indexes):
    """The indexes without repeats, the three known ones first."""
    indexes = list(dict.fromkeys(indexes))
    known = [e for e in ORDER if e in indexes]
    return known + [e for e in indexes if e not in known]


# --- scoring one result list ----------------------------------------------------------------

@dataclass
class Scored:
    """One retrieved chunk, scored against one question of the key."""
    rank: int
    id: str
    source: str                # the chunk's file, named the way check_key.py names corpus files
    also_in: list              # other files that hold the same text
    section: str
    similarity: float | None   # cosine similarity; None in the merged list
    rrf: float | None          # the fusion score; merged list only
    page: bool                 # from a source: file of the question
    evidence: bool             # contains an evidence: phrase of the question (from any file)
    passage: bool              # page and evidence: the passage the answer needs
    known: bool                # its file is in chunks.jsonl


class FileNames:
    """Names a retrieved chunk's file the way check_key.py names the files of chunks.jsonl."""

    def __init__(self, corpus):
        self.corpus, self.cache = corpus, {}

    def __call__(self, source):
        """-> (name, known); known is False when chunks.jsonl has no such file."""
        source = str(source or "")
        if source not in self.cache:
            name = norm_path(source) if source else "?"
            if source and name not in self.corpus.files:
                found, _ = self.corpus.resolve(source)
                if len(found) == 1:
                    name = found[0]
            self.cache[source] = (name, name in self.corpus.files)
        return self.cache[source]


def _num(x, places):
    return None if x is None else round(float(x), places)


def score_hits(q, hits, names):
    """Score one result list (best first) against one question. -> list of Scored"""
    wanted = set(q.files)
    phrases = [p for p in (squash(e) for e in q.evidence) if p]
    scored = []
    for rank, h in enumerate(hits, 1):
        source, known = names(getattr(h, "source", ""))
        also = [a for a in dict.fromkeys(names(s)[0] for s in (getattr(h, "also_in", None) or [])) if a != source]
        page = source in wanted or any(a in wanted for a in also)
        text = squash(getattr(h, "text", "") or "")
        evidence = any(p in text for p in phrases)
        scored.append(Scored(
            rank=rank, id=str(getattr(h, "id", "") or ""), source=source, also_in=also,
            section=str(getattr(h, "section", "") or ""),
            similarity=_num(getattr(h, "similarity", None), 4), rrf=_num(getattr(h, "rrf", None), 6),
            page=page, evidence=evidence, passage=page and (evidence or not phrases), known=known))
    return scored


def first(scored, what):
    """Rank of the first chunk whose `what` ("passage" or "page") is set, or None."""
    return next((s.rank for s in scored if getattr(s, what)), None)


def top1(scored):
    return scored[0].similarity if scored else None


def counts(ranks, cuts):
    """-> ({cut: how many ranks are at or above it}, mean reciprocal rank)"""
    at = {c: sum(1 for r in ranks if r is not None and r <= c) for c in cuts}
    mrr = sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0
    return at, mrr


def cutoff_data(answer_top1, found_top1, decline_top1):
    """The numbers a "nothing found" cutoff for one index would start from (None values are skipped)."""
    a = [x for x in answer_top1 if x is not None]
    f = [x for x in found_top1 if x is not None]
    d = [x for x in decline_top1 if x is not None]
    out = {"answer_lowest": min(a) if a else None,
           "answer_median": round(statistics.median(a), 4) if a else None,
           "found_lowest": min(f) if f else None,
           "decline_highest": max(d) if d else None}
    lo, hi = out["found_lowest"], out["decline_highest"]
    out["gap"] = None if lo is None or hi is None else round(lo - hi, 4)
    return out


# --- the merged list ------------------------------------------------------------------------

@dataclass
class MergedHit:
    """One chunk of the merged list, with the fields score_hits reads."""
    id: str
    text: str
    source: str
    also_in: list
    section: str
    rrf: float
    similarity: float | None = None


def _text_key(h):
    s = " ".join(str(getattr(h, "text", "") or "").split()).lower()
    for p in DOC_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    return s or "id:" + str(getattr(h, "id", ""))


def merge_lists(lists, k, c=RRF_K):
    """Reciprocal rank fusion of result lists (each best first): a chunk scores 1/(c + rank) in
    each list it is in, and the scores are added. Chunks with the same text are one chunk and
    their files are pooled. A tie goes to the better single rank, then to the earlier list.
    -> the best k, as MergedHit"""
    table = {}
    for li, hits in enumerate(lists):
        seen = set()
        for rank, h in enumerate(hits, 1):
            key = _text_key(h)
            if key in seen:
                continue
            seen.add(key)
            e = table.get(key)
            if e is None:
                e = table[key] = {"hit": h, "score": 0.0, "best": (rank, li), "files": []}
            e["score"] += 1.0 / (c + rank)
            e["best"] = min(e["best"], (rank, li))
            for s in [getattr(h, "source", "")] + list(getattr(h, "also_in", None) or []):
                if s and s not in e["files"]:
                    e["files"].append(s)
    best = sorted(table.values(), key=lambda e: (-e["score"], e["best"]))[:k]
    return [MergedHit(id=str(getattr(e["hit"], "id", "") or ""), text=str(getattr(e["hit"], "text", "") or ""),
                      source=e["files"][0] if e["files"] else "", also_in=e["files"][1:],
                      section=str(getattr(e["hit"], "section", "") or ""), rrf=e["score"])
            for e in best]


# --- running the searches -------------------------------------------------------------------

@dataclass
class Run:
    questions: list
    k: int
    indexes: list = field(default_factory=list)     # scored, in report order; the merged list last
    scored: dict = field(default_factory=dict)      # index -> question id -> [Scored]
    seconds: dict = field(default_factory=dict)     # index -> seconds for all its searches
    skipped: dict = field(default_factory=dict)     # index -> why it could not be searched
    merged_from: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    unknown: dict = field(default_factory=dict)     # file not in chunks.jsonl -> chunks retrieved from it
    retrieved: int = 0                              # chunks retrieved by the indexes (not the merged list)


def evaluate(questions, corpus, search, indexes, k, merge=(), log=None):
    """Search every question in every index and score the results.
    search(index, question, k) -> hits, best first (retrieve.py Hit objects or the same fields)."""
    log = log or (lambda message: None)
    run = Run(questions=list(questions), k=k)
    raw = {}
    for e in indexes:
        log(f"  {e}: searching {len(questions)} questions ...")
        t0 = time.time()
        got = {}
        try:
            for q in questions:
                got[q.id] = list(search(e, q.question, k))[:k]
        except (LookupError, RuntimeError) as err:
            run.skipped[e] = str(err)
            log(f"  {e}: skipped: {err}")
            continue
        raw[e] = got
        run.seconds[e] = round(time.time() - t0, 1)
        log(f"  {e}: done in {run.seconds[e]:.1f} s")
    names = FileNames(corpus)
    for e in indexes:
        if e not in raw:
            continue
        run.indexes.append(e)
        run.scored[e] = {}
        for q in questions:
            s = score_hits(q, raw[e][q.id], names)
            run.scored[e][q.id] = s
            run.retrieved += len(s)
            for x in s:
                if not x.known:
                    run.unknown[x.source] = run.unknown.get(x.source, 0) + 1
    if merge:
        missing = [m for m in merge if m not in raw]
        if missing:
            run.notes.append(f"No merged list: {', '.join(missing)} could not be searched.")
        else:
            run.merged_from = list(merge)
            run.indexes.append(MERGED)
            run.scored[MERGED] = {q.id: score_hits(q, merge_lists([raw[m][q.id] for m in merge], k), names)
                                  for q in questions}
    return run


# --- the report -----------------------------------------------------------------------------

def _clip(text, n):
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _sim(x):
    return "-" if x is None else f"{x:.3f}"


def _signed(x):
    return "-" if x is None else f"{x:+.3f}"


def kind_of(q):
    """Where a question's answer is: man page, HackTricks, both, or other."""
    kinds = set()
    for f in q.files:
        low = f.lower()
        if low.startswith("manpages/") or "/manpages/" in low:
            kinds.add("man page")
        elif low.startswith("hacktricks/") or "/hacktricks/" in low:
            kinds.add("HackTricks")
        else:
            kinds.add("other")
    if len(kinds) > 1:
        return "both" if kinds == {"man page", "HackTricks"} else "mixed"
    return kinds.pop() if kinds else "other"


def cell(scored):
    """Rank of the first passage; p<n> when only the right page was found (at rank n); - for neither."""
    r = first(scored, "passage")
    if r:
        return str(r)
    p = first(scored, "page")
    return f"p{p}" if p else "-"


def summaries(run):
    """-> (counts per index, cutoff data per real index)"""
    answers = [q for q in run.questions if q.expect == "answer"]
    declines = [q for q in run.questions if q.expect == "decline"]
    cuts = cuts_for(run.k)
    table, cutoff = {}, {}
    for e in run.indexes:
        sc = run.scored[e]
        pa, mrr = counts([first(sc[q.id], "passage") for q in answers], cuts)
        pg, _ = counts([first(sc[q.id], "page") for q in answers], cuts)
        table[e] = {"passage": pa, "mrr": mrr, "page": pg}
        if e != MERGED:
            cutoff[e] = cutoff_data([top1(sc[q.id]) for q in answers],
                                    [top1(sc[q.id]) for q in answers if first(sc[q.id], "passage")],
                                    [top1(sc[q.id]) for q in declines])
    return table, cutoff


def report(run, info, show=()):
    lines = []
    say = lines.append
    qs = run.questions
    answers = [q for q in qs if q.expect == "answer"]
    declines = [q for q in qs if q.expect == "decline"]
    cols = run.indexes
    real = [e for e in cols if e != MERGED]
    cuts = cuts_for(run.k)
    small = 5 if 5 in cuts else cuts[-1]
    table, cutoff = summaries(run)
    iw = max([len(q.id) for q in qs] + [2])
    qw = 44

    say("Casefile Week 10: the indexes compared on the answer key")
    say(f"Run      {info['made']}, eval_retrieval.py (searches only; no chat model)")
    say(f"Key      {info['key']}: {len(qs)} questions ({len(answers)} answer, {len(declines)} decline), "
        f"sha256 {info['key_sha256'][:12]}")
    say(f"Corpus   {info['chunks']}: {info['n_chunks']} chunks, {info['n_files']} files")
    say("Indexes  " + "; ".join(f"{short(e)} = {e}" for e in real))
    if MERGED in cols:
        say(f"         merged = {' + '.join(short(m) for m in run.merged_from)}, by reciprocal rank fusion "
            f"(each list adds 1/({RRF_K} + rank))")
    say(f"Depth    the top {run.k} chunks per index and question")

    say("")
    say(f"1. Answer questions ({len(answers)}): how many have the passage holding the answer in the top n")
    say("   passage = a chunk from a source: file that contains an evidence: phrase")
    say("   page    = any chunk from a source: file")
    say("   MRR     = mean of 1/rank of the first passage (0 when it is not in the list)")
    if answers:
        w = max([len(short(e)) for e in cols] + [5])
        nums = "".join(f"{'@' + str(c):>5}" for c in cuts)
        say(f"   {'':<{w}}  {'passage':<{5 * len(cuts) + 7}}   page")
        say(f"   {'index':<{w}}  {nums}{'MRR':>7}   {nums}")
        for e in cols:
            t = table[e]
            say(f"   {short(e):<{w}}  " + "".join(f"{t['passage'][c]:>5}" for c in cuts)
                + f"{t['mrr']:>7.3f}   " + "".join(f"{t['page'][c]:>5}" for c in cuts))
    else:
        say("   (the key has no answer questions)")

    groups = {}
    for q in answers:
        groups.setdefault(kind_of(q), []).append(q)
    if groups:
        say("")
        say(f"2. Passage in the top {small}, by where the answer is")
        labels = {g: f"{g} ({len(v)})" for g, v in groups.items()}
        lw = max([len(x) for x in labels.values()] + [len("answer in")])
        cw = [max(len(short(e)), 4) for e in cols]
        say(f"   {'answer in':<{lw}}  " + "  ".join(f"{short(e):>{x}}" for e, x in zip(cols, cw)))
        for g in ("man page", "HackTricks", "both", "mixed", "other"):
            if g in groups:
                row = [sum(1 for q in groups[g] if (first(run.scored[e][q.id], "passage") or run.k + 1) <= small)
                       for e in cols]
                say(f"   {labels[g]:<{lw}}  " + "  ".join(f"{n:>{x}}" for n, x in zip(row, cw)))

    if answers:
        say("")
        say(f"3. Rank of the first passage per answer question (p4 = no passage in the top {run.k}, but")
        say("   a chunk of the right page at rank 4; - = neither)")
        cw = [max(len(short(e)), 4) for e in cols]
        say(f"   {'id':<{iw}} {'question':<{qw}}  " + "  ".join(f"{short(e):>{x}}" for e, x in zip(cols, cw)))
        for q in answers:
            say(f"   {q.id:<{iw}} {_clip(q.question, qw):<{qw}}  "
                + "  ".join(f"{cell(run.scored[e][q.id]):>{x}}" for e, x in zip(cols, cw)))

    if real:
        say("")
        say('4. Top-1 similarity per question: data for the Week 12 "nothing found" cutoff.')
        say("   Each index has its own scale: compare down a column, not across.")
        cw = [max(len(short(e)), 6) for e in real]
        say(f"   {'id':<{iw}} {'kind':<7} {'question':<{qw}}  "
            + "  ".join(f"{short(e):>{x}}" for e, x in zip(real, cw)))
        for q in qs:
            say(f"   {q.id:<{iw}} {q.expect:<7} {_clip(q.question, qw):<{qw}}  "
                + "  ".join(f"{_sim(top1(run.scored[e][q.id])):>{x}}" for e, x in zip(real, cw)))
        lw = iw + 1 + 7 + 1 + qw
        say("   " + "-" * (lw + 2 + sum(cw) + 2 * (len(cw) - 1)))
        rows = [("answer questions: lowest top-1", "answer_lowest", _sim),
                ("answer questions: median top-1", "answer_median", _sim),
                (f"lowest top-1 where the passage was in the top {run.k}", "found_lowest", _sim),
                ("decline questions: highest top-1", "decline_highest", _sim),
                ("gap = lowest found - highest decline", "gap", _signed)]
        for label, key, fmt in rows:
            say(f"   {label:<{lw}}  " + "  ".join(f"{fmt(cutoff[e][key]):>{x}}" for e, x in zip(real, cw)))
        say("   A positive gap means some cutoff would keep every answer question whose passage was found")
        say(f"   and turn away every decline question, in this sample ({len(declines)} decline question(s)).")

    def note(text):
        say(textwrap.fill(text, width=100, initial_indent="   ", subsequent_indent="   ",
                          break_long_words=False, break_on_hyphens=False))

    say("")
    say("5. Notes")
    for e in real:
        note(f"{short(e)}: {len(qs)} searches in {run.seconds.get(e, 0):.1f} s")
    for e, why in run.skipped.items():
        note(f"{e} was not searched: {why}")
    for n in run.notes:
        note(n)
    elsewhere = sorted({q.id for e in real for q in answers for s in run.scored[e][q.id] if s.evidence and not s.page})
    if elsewhere:
        note(f"An evidence phrase also turned up in a chunk from a file the key does not list, for "
             f"{', '.join(elsewhere)}. Those chunks count as misses; --show marks them e.")
    noev = [q.id for q in answers if not any(squash(e) for e in q.evidence)]
    if noev:
        note(f"Scored by page only (no evidence: line): {', '.join(noev)}")
    if run.unknown:
        some = ", ".join(list(run.unknown)[:3])
        note(f"WARNING: {sum(run.unknown.values())} of {run.retrieved} retrieved chunks name a file that is not in "
             f"{info['chunks']} (for example {some}). Page and passage hits cannot be counted for them. "
             "Was the index built from this chunks.jsonl?")
    if info.get("warnings"):
        note(f"check_key.py reports {info['warnings']} warning(s) for this key; run python check_key.py to see them.")
    note(f"Every retrieved chunk, per question and index, is in {info['json']}.")

    if show:
        by_id = {q.id: q for q in qs}
        say("")
        say("6. What each index returned (P = passage, p = right page, e = an evidence phrase in another file)")
        for qid in show:
            q = by_id[qid]
            say("")
            say(f"   {q.id} [{q.expect}] {q.question}")
            if q.files:
                say(f"   source: {', '.join(q.files)}")
            for e in cols:
                say(f"     {short(e)}")
                for s in run.scored[e][qid]:
                    mark = "P" if s.passage else "p" if s.page else "e" if s.evidence else "."
                    score = f"{s.rrf:.4f}" if s.similarity is None and s.rrf is not None else _sim(s.similarity)
                    where = s.source + (f" > {_clip(s.section, 50)}" if s.section else "")
                    say(f"     {s.rank:>3}  {score:>6}  {mark}  {where}")
                    if s.also_in:
                        more = " ..." if len(s.also_in) > 2 else ""
                        say(f"     {'':>3}  {'':>6}     same text also in: {', '.join(s.also_in[:2])}{more}")
    return "\n".join(lines) + "\n"


def as_json(run, info):
    table, cutoff = summaries(run)
    answers = [q for q in run.questions if q.expect == "answer"]
    questions = []
    for q in run.questions:
        results = {}
        for e in run.indexes:
            s = run.scored[e][q.id]
            results[e] = {"passage_rank": first(s, "passage"), "page_rank": first(s, "page"),
                          "top1_similarity": top1(s), "hits": [asdict(x) for x in s]}
        questions.append({"id": q.id, "question": q.question, "expect": q.expect,
                          "kind": kind_of(q) if q.expect == "answer" else None,
                          "files": list(q.files), "evidence": list(q.evidence), "results": results})
    summary = {e: {"passage": {str(c): n for c, n in t["passage"].items()}, "mrr": round(t["mrr"], 4),
                   "page": {str(c): n for c, n in t["page"].items()}} for e, t in table.items()}
    return {"made": info["made"], "script": "eval_retrieval.py",
            "key": {"file": info["key"], "sha256": info["key_sha256"], "answer": len(answers),
                    "decline": len(run.questions) - len(answers)},
            "corpus": {"file": info["chunks"], "chunks": info["n_chunks"], "files": info["n_files"]},
            "k": run.k, "rrf_k": RRF_K, "indexes": run.indexes, "merged_from": run.merged_from,
            "seconds": run.seconds, "skipped": run.skipped, "notes": run.notes, "unknown_files": run.unknown,
            "summary": summary, "cutoff": cutoff, "questions": questions}


def write_results(prefix, text, data):
    with open(prefix + ".txt", "w", encoding="utf-8") as f:
        f.write(text)
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")


def pick_ids(wanted, ids):
    """The question ids --show asks for, in key order ("all" means every one)."""
    if not wanted:
        return []
    if any(w.lower() == "all" for w in wanted):
        return list(ids)
    unknown = [w for w in wanted if w not in ids]
    if unknown:
        raise ValueError(f"--show: the key has no question {', '.join(unknown)} (its ids run {ids[0]} to {ids[-1]})")
    return [i for i in ids if i in wanted]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# --- self-test (made-up files, no Ollama or Chroma) -----------------------------------------

def _selftest():
    import tempfile
    from types import SimpleNamespace as NS

    from check_key import parse_key

    rows = [
        ("manpages/zorp.man.txt", "ZORP(1)\n\nNAME\n       zorp - count the widgets"),
        ("manpages/zorp.man.txt", "OPTIONS\n       -q, --quiet\n              Print nothing but\n              errors."),
        ("hacktricks/src/misc/frob.md", "## Frob\n\nRun **`frob --deep`** to frobnicate every widget."),
        ("hacktricks/src/other/copy.md", "## Frob\n\nRun **`frob --deep`** to frobnicate every widget."),
        ("hacktricks/src/noise.md", "A page that also says: Print nothing but errors."),
        ("hacktricks/src/misc/frob.md", "Frob history, with no command in it."),
    ]
    corpus = Corpus("(test corpus)")
    for src, text in rows:
        corpus.add({"text": text, "metadata": {"source": src}})
    key = ("id: t1\nquestion: what does zorp -q do?\nsource: corpus/manpages/zorp.man.txt\n"
           "evidence: Print nothing but errors.\nanswer: It prints only errors.\n\n"
           "id: t2\nquestion: how do I frobnicate every widget?\nsource: corpus/hacktricks/src/misc/frob.md\n"
           "evidence: Run frob \u2014deep to frobnicate\nanswer: Run frob --deep.\n\n"
           "id: t3\nquestion: what is zorp?\nsource: zorp.man.txt\nanswer: It counts widgets.\n\n"
           "id: t4\nquestion: how do I bake bread?\nexpect: decline\nanswer: Not in the corpus.\n")
    qs, findings = parse_key(key)
    findings += check_corpus(qs, corpus)
    t1, t2, t3, t4 = qs
    names = FileNames(corpus)
    info = {"made": "2026-01-01 00:00", "key": "(test key)", "key_sha256": "0" * 64, "chunks": "(test corpus)",
            "n_chunks": len(corpus.texts), "n_files": len(corpus.files), "warnings": 0, "json": "(test).json"}

    def hit(i, sim=None, source=None, also=(), prefix=""):
        src, text = rows[i]
        return NS(id=f"c{i}", text=prefix + text, source=source or src, also_in=list(also), section="Sec",
                  similarity=sim)

    def doc(name, source="x.md", also=(), text=None):
        return NS(id=name, text=text or f"chunk {name}", source=source, also_in=list(also), section="")

    def ranks(q, hits):
        s = score_hits(q, hits, names)
        return first(s, "passage"), first(s, "page"), [x.evidence for x in s]

    def searcher(table, broken=()):
        def search(index, question, k):
            if index in broken:
                raise RuntimeError("could not embed the question (is Ollama running?)")
            return table[index].get(question, [])[:k]
        return search

    plan = {
        "idx-a": {t1.question: [hit(1, .90), hit(0, .80)], t2.question: [hit(5, .70), hit(2, .60)],
                  t3.question: [hit(4, .50)], t4.question: [hit(5, .40)]},
        "idx-b": {t1.question: [hit(4, .60), hit(0, .55), hit(1, .50)],
                  t2.question: [hit(3, .65, also=["/x/corpus/hacktricks/src/misc/frob.md"], prefix="search_document: ")],
                  t3.question: [hit(0, .45)], t4.question: [hit(2, .30)]},
    }

    def full_run(k=10, broken=()):
        return evaluate(qs, corpus, searcher(plan, broken), ["idx-a", "idx-b"], k, ("idx-a", "idx-b"))

    def cells(run):
        return [cell(run.scored[e][q.id]) for e in run.indexes for q in (t1, t2, t3)]

    def written_files_load():
        run = full_run()
        with tempfile.TemporaryDirectory() as d:
            prefix = os.path.join(d, "out")
            write_results(prefix, report(run, info, ["t2"]), as_json(run, info))
            with open(prefix + ".json", encoding="utf-8") as f:
                data = json.load(f)
            with open(prefix + ".txt", encoding="utf-8") as f:
                text = f.read()
        return (data["summary"]["merged"]["passage"]["1"] == 2 and data["questions"][0]["id"] == "t1"
                and data["questions"][1]["results"]["idx-a"]["passage_rank"] == 2 and "6. What each index" in text)

    def show_lists_hits():
        text = report(full_run(), info, ["t1", "t2"])
        return ("  1   0.600  e  hacktricks/src/noise.md > Sec" in text and "same text also in:" in text
                and "1  0.0325  P" in text and "does not list, for t1." in " ".join(text.split()))

    cases = [
        ("the test key and corpus check clean",
         lambda: not [f for f in findings if f.level in ("ERROR", "WARN", "SMOKE")]),
        ("a passage needs a source: file and an evidence phrase; a source: file alone is a page hit",
         lambda: ranks(t1, [hit(4), hit(0), hit(1)]) == (3, 2, [True, False, True])),
        ("evidence matches across a line wrap, ** and `, a long dash and a search_document: prefix",
         lambda: ranks(t2, [hit(2, prefix="search_document: ")]) == (1, 1, [True])),
        ("the same text kept from another page counts through also_in, and only through it",
         lambda: ranks(t2, [hit(3, also=["hacktricks/src/misc/frob.md"])])[:2] == (1, 1)
         and ranks(t2, [hit(3)]) == (None, None, [True])),
        ("a file is matched however the index names it",
         lambda: all(ranks(t3, [hit(0, source=s)])[1] == 1 and score_hits(t3, [hit(0, source=s)], names)[0].known
                     for s in ("/home/huzai/projects/PenetrationTesting/corpus/manpages/zorp.man.txt",
                               "corpus/manpages/zorp.man.txt", "./corpus/manpages/zorp.man.txt", "zorp.man.txt"))),
        ("a file that is not in chunks.jsonl is flagged as unknown",
         lambda: not score_hits(t1, [hit(1, source="elsewhere/zorp.txt")], names)[0].known),
        ("a question with no evidence: line is scored by page; a decline question gets no hit",
         lambda: ranks(t3, [hit(4), hit(0)])[0] == 2 and ranks(t4, [hit(1), hit(2)])[:2] == (None, None)),
        ("counts at each cut and MRR",
         lambda: counts([1, 3, None, 6], (1, 3, 5, 10)) == ({1: 1, 3: 2, 5: 2, 10: 3}, 0.375)
         and counts([], (1,)) == ({1: 0}, 0.0)),
        ("the cuts follow k, and the indexes come in report order",
         lambda: cuts_for(10) == (1, 3, 5, 10) and cuts_for(4) == (1, 3, 4) and cuts_for(1) == (1,)
         and cuts_for(20) == (1, 3, 5, 10, 20)
         and in_order(["all-minilm", "mxbai", "nomic-embed-text-prefixed", "nomic-embed-text", "mxbai"])
         == ["nomic-embed-text", "nomic-embed-text-prefixed", "all-minilm", "mxbai"]),
        ("fusion: a chunk in both lists beats one in a single list; each list adds 1/(60 + rank)",
         lambda: (lambda m: [h.id for h in m] == ["Y", "X", "W", "Z"] and abs(m[0].rrf - (1 / 62 + 1 / 61)) < 1e-12)(
             merge_lists([[doc("X"), doc("Y"), doc("Z")], [doc("Y"), doc("W")]], 10))),
        ("fusion: a tie goes to the earlier list, and only the best k are kept",
         lambda: [h.id for h in merge_lists([[doc("A")], [doc("B")]], 10)] == ["A", "B"]
         and [h.id for h in merge_lists([[doc("B")], [doc("A")]], 10)] == ["B", "A"]
         and len(merge_lists([[doc(str(i)) for i in range(6)], [doc(str(i)) for i in range(6, 12)]], 4)) == 4),
        ("fusion: the same text in both lists is one chunk, with the files pooled",
         lambda: (lambda m: len(m) == 1 and m[0].source == "p1" and m[0].also_in == ["p2", "p3"]
                  and abs(m[0].rrf - 2 / 61) < 1e-12)(
             merge_lists([[doc("1", "p1", text="Same  text")],
                          [doc("2", "p2", ["p3"], text="search_document: same text")]], 10))),
        ("one file named two ways (relative and absolute) is listed once",
         lambda: score_hits(t3, merge_lists([[hit(0)], [hit(0, source="/h/corpus/manpages/zorp.man.txt")]], 10),
                            names)[0].also_in == []),
        ("cutoff data: lowest, median, gap (positive, negative, or none without decline questions)",
         lambda: abs(cutoff_data([.8, .7, .75], [.8, .7], [.6, .65])["gap"] - .05) < 1e-9
         and cutoff_data([.8, .7, .75], [.8, .7], [.6, .65])["answer_median"] == .75
         and abs(cutoff_data([.8], [.8], [.85])["gap"] + .05) < 1e-9 and cutoff_data([.8], [.8], [])["gap"] is None),
        ("a full run: ranks per index and for the merged list",
         lambda: cells(full_run()) == ["1", "2", "-", "3", "1", "1", "1", "1", "2"]
         and full_run().indexes == ["idx-a", "idx-b", MERGED]),
        ("a full run: counts and MRR leave the decline question out; cutoff data per index",
         lambda: (lambda t, c: t["idx-a"]["passage"] == {1: 1, 3: 2, 5: 2, 10: 2} and abs(t["idx-a"]["mrr"] - .5) < 1e-9
                  and abs(t[MERGED]["mrr"] - 2.5 / 3) < 1e-9 and abs(c["idx-a"]["gap"] - .3) < 1e-9
                  and abs(c["idx-b"]["gap"] - .15) < 1e-9 and MERGED not in c)(*summaries(full_run()))),
        ("an index that cannot be searched is skipped, and so is a merged list that needs it",
         lambda: (lambda r: r.indexes == ["idx-a"] and "Ollama" in r.skipped["idx-b"] and r.notes
                  and "was not searched" in report(r, info))(full_run(broken={"idx-b"}))),
        ("the report follows k (here 2: @1 and @2, no @5)",
         lambda: (lambda text: "   @1   @2" in text and "@5" not in text and "passage in the top 2" in text.lower())(
             report(full_run(k=2), info))),
        ("--show lists each index's chunks with their marks and pooled files",
         show_lists_hits),
        ("--show takes ids in any order, or all; an unknown id is an error",
         lambda: pick_ids(["t2", "t1"], ["t1", "t2", "t3"]) == ["t1", "t2"]
         and pick_ids(["all"], ["t1", "t2"]) == ["t1", "t2"] and pick_ids(None, ["t1"]) == []
         and _raises(ValueError, pick_ids, ["zz"], ["t1", "t2"])),
        ("the .txt and .json are written, and the .json loads back",
         written_files_load),
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


def _raises(kind, fn, *args):
    try:
        fn(*args)
    except kind:
        return True
    return False


# --- command line ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare the Casefile indexes on the answer key (Week 10). "
                                             "It searches only; no chat model runs.")
    ap.add_argument("--key", default=KEY_FILE, help=f"the answer key (default {KEY_FILE})")
    ap.add_argument("--chunks", default=CHUNKS_FILE,
                    help=f"the chunks the indexes were built from (default {CHUNKS_FILE})")
    ap.add_argument("-e", "--embedder", action="append", metavar="INDEX",
                    help="an index to search, by embedder name; repeat for more "
                         "(default: every index retrieve.py finds)")
    ap.add_argument("-k", type=int, default=K, help=f"chunks per index and question (default {K})")
    ap.add_argument("--merge", nargs="+", metavar="INDEX",
                    help=f"indexes to merge into one list (default: {' and '.join(MERGE_DEFAULT)})")
    ap.add_argument("--no-merge", action="store_true", help="make no merged list")
    ap.add_argument("--show", nargs="+", metavar="ID",
                    help="also list what each index returned for these questions (e.g. q03 q17), or all")
    ap.add_argument("--db", help="the Chroma folder (default: retrieve.py's, ./chroma_db)")
    ap.add_argument("--out", default=OUT_PREFIX, help=f"write OUT.txt and OUT.json (default {OUT_PREFIX})")
    ap.add_argument("--selftest", action="store_true", help="run the built-in tests and exit")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    if args.selftest:
        return 1 if _selftest() else 0
    if args.k < 1:
        ap.error("-k must be at least 1")

    def log(message):
        print(message, file=sys.stderr, flush=True)

    try:
        questions = load_key(args.key)
    except OSError as e:
        print(f"error: cannot read {args.key} ({e.strerror or e}). "
              "Run this from the project root (cd ~/projects/PenetrationTesting).")
        return 2
    except ValueError as e:
        print(f"error: {e}")
        return 2
    if not questions:
        print(f"error: {args.key} has no questions.")
        return 2
    try:
        show = pick_ids(args.show, [q.id for q in questions])
    except ValueError as e:
        print(f"error: {e}")
        return 2
    if not os.path.isfile(args.chunks):
        print(f"error: {args.chunks} not found. Run this from the project root (cd ~/projects/PenetrationTesting).")
        return 2
    try:
        corpus = load_corpus(args.chunks)
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 2
    findings = check_corpus(questions, corpus)
    errors = [f for f in findings if f.level == "ERROR"]
    if errors:
        print(f"error: the key has {len(errors)} error(s) against {args.chunks} (the first: {errors[0].message}). "
              "Run python check_key.py and fix them first.")
        return 2

    try:
        import retrieve
    except ImportError as e:
        print(f"error: cannot import retrieve.py ({e}). Run this from the project root with the venv active "
              "(source ~/.venv/bin/activate).")
        return 2
    db = args.db or getattr(retrieve, "DEFAULT_DB", "./chroma_db")
    if args.embedder:
        indexes = in_order(args.embedder)
    else:
        try:
            indexes = in_order(retrieve.available_embedders(db))
        except (LookupError, RuntimeError) as e:
            print(f"error: {e}")
            return 2
        if not indexes:
            print(f"error: no index found in {db}. Run python retrieve.py --list to see what is there.")
            return 2
    if args.no_merge:
        merge = ()
    elif args.merge:
        merge = tuple(dict.fromkeys(args.merge))
        if len(merge) < 2:
            ap.error("--merge needs at least two indexes")
        missing = [m for m in merge if m not in indexes]
        if missing:
            ap.error(f"--merge names {', '.join(missing)}, which is not being searched (add -e for it)")
    else:
        merge = MERGE_DEFAULT if all(m in indexes for m in MERGE_DEFAULT) else ()
    if not merge and not args.no_merge:
        absent = [m for m in MERGE_DEFAULT if m not in indexes]
        merge_note = (f"No merged list: the default merges {' and '.join(MERGE_DEFAULT)}, and "
                      f"{' and '.join(absent)} {'is' if len(absent) == 1 else 'are'} not among the indexes searched.")
    else:
        merge_note = ""

    def search(index, question, k):
        return retrieve.retrieve(question, embedder=index, k=k, db_path=db)

    log(f"Searching {len(questions)} questions in {len(indexes)} index(es), top {args.k} each. No chat model runs.")
    run = evaluate(questions, corpus, search, indexes, args.k, merge, log)
    if merge_note:
        run.notes.append(merge_note)
    if not run.indexes:
        print("error: no index could be searched:")
        for e, why in run.skipped.items():
            print(f"  {e}: {why}")
        return 2
    info = {"made": time.strftime("%Y-%m-%d %H:%M"), "key": args.key, "key_sha256": sha256_of(args.key),
            "chunks": args.chunks, "n_chunks": len(corpus.texts), "n_files": len(corpus.files),
            "warnings": sum(f.level == "WARN" for f in findings), "json": args.out + ".json"}
    text = report(run, info, show)
    sys.stdout.write(text)
    try:
        write_results(args.out, text, as_json(run, info))
    except OSError as e:
        print(f"error: could not write the results ({e})")
        return 2
    print(f"\nWrote {args.out}.txt and {args.out}.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:        # output piped into head or less, which closed early
        sys.exit(0)
