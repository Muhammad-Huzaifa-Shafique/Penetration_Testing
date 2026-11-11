#!/usr/bin/env python3
"""
eval_answers.py -- Casefile Week 11: ask Casefile every key question and collect the answers for grading

Asks each question of answer_key.txt through answer.py's ask(), on its own (no history) and with
answer.py's own defaults (the search index and the number of passages retrieve.py uses), and saves:

  week11_answers.json   everything, saved after every question, so a stopped run can be resumed
  week11_answers.txt    for reading and grading: a summary table, then one block per question with
                        Casefile's answer, the key's expected answer, the passages it was given
                        and the automatic checks

    python eval_answers.py --only q07      one question first, as a quick test (about 2 minutes)
    python eval_answers.py --resume        the rest (or continue a stopped run); saved answers are kept
    python eval_answers.py                 all 20 questions in a new run (about 30-50 minutes on CPU)
    python eval_answers.py --report        rewrite week11_answers.txt from the .json; asks nothing
    python eval_answers.py --selftest      built-in tests (no Ollama needed)

A new run will not overwrite an existing week11_answers.json: use --resume to continue it, or
--out NAME for a separate run. Ctrl-C stops after saving what is done; --resume picks it up.
A question whose ask() failed, or whose wording in the key has changed, is asked again on --resume.

Automatic checks per answer. They help the grading; they do not grade.
  passage   the passage numbers that hold the answer (a chunk from a source: file with an
            evidence: phrase, scored exactly as in Week 10), or - when none was given to the model
  cites     whether the answer cites one of those passages by its number
  must      how many of the key's must: texts the answer contains (case, markup and dashes ignored)
  declined  whether answer.py's citation check reads the answer as "the passages do not cover this"
  check     how many problems the Week 7 citation check found (wording and numbers, not truth)

Needs check_key.py, eval_retrieval.py, answer.py and what it imports, answer_key.txt and
corpus/chunks.jsonl in the project root, the venv active and Ollama running.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from check_key import CHUNKS_FILE, KEY_FILE, Corpus, check_corpus, load_corpus, load_key, squash
from eval_retrieval import FileNames, kind_of, pick_ids, score_hits, sha256_of

OUT_PREFIX = "week11_answers"
STOP_AFTER = 3            # stop when this many questions fail in a row (Ollama not running, ...)
_CITE = re.compile(r"\[(\d{1,3})\]")
RULE = "=" * 20


# --- one answer ----------------------------------------------------------------------------

def must_found(musts, text):
    """The must: texts the answer contains, compared without case, markup or dash differences."""
    low = squash(text).lower()
    return [m for m in musts if squash(m) and squash(m).lower() in low]


def cited_numbers(check, text, n_passages):
    """The passage numbers the answer cites: from the citation check if ask() made one, else from [n] marks."""
    cited = getattr(check, "cited", None) if check is not None else None
    if cited is not None:
        return sorted({int(c) for c in cited})
    return sorted({int(m) for m in _CITE.findall(text or "") if 1 <= int(m) <= n_passages})


def problems_of(check):
    """The citation check's ERROR and WARN lines, as text."""
    out = []
    for f in (getattr(check, "findings", None) or []) if check is not None else []:
        level = str(getattr(f, "level", "") or "")
        if level and level != "NOTE":
            out.append(f"{level}: {getattr(f, 'message', '')}")
    return out


def base_record(q):
    return {"id": q.id, "question": q.question, "expect": q.expect,
            "kind": kind_of(q) if q.expect == "answer" else None, "files": list(q.files),
            "evidence": list(q.evidence), "must": list(q.must), "expected": q.answer}


def record_for(q, ans, seconds, names):
    """The saved record of one answered question."""
    text = str(getattr(ans, "text", "") or "")
    hits = list(getattr(ans, "hits", None) or [])
    scored = score_hits(q, hits, names)
    check = getattr(ans, "check", None)
    cited = cited_numbers(check, text, len(hits))
    is_answer = q.expect == "answer"
    holds = [s.rank for s in scored if s.passage] if is_answer else []
    rec = base_record(q)
    rec.update({
        "seconds": round(seconds, 1),
        "answer": text,
        "truncated": bool(getattr(ans, "truncated", False)),
        "prompt_tokens": getattr(ans, "prompt_tokens", None),
        "answer_tokens": getattr(ans, "answer_tokens", None),
        "passage_numbers": holds,
        "passage_in_context": bool(holds) if is_answer else None,
        "page_in_context": any(s.page for s in scored) if is_answer else None,
        "cited": cited,
        "cites_passage": bool(set(cited) & set(holds)) if is_answer else None,
        "must_found": must_found(q.must, text),
        "declined": None if check is None else bool(getattr(check, "declined", False)),
        "check_problems": problems_of(check),
        "passages": [{"n": s.rank, "source": s.source, "section": s.section, "similarity": s.similarity,
                      "passage": s.passage, "page": s.page, "evidence": s.evidence, "also_in": s.also_in,
                      "text": str(getattr(h, "text", "") or "")} for s, h in zip(scored, hits)],
        "error": None,
    })
    return rec


def ask_one(q, ask, names, embedder=None, k=None, on_token=None):
    """Ask one question. A failure is recorded in the result instead of ending the run."""
    t0 = time.time()
    try:
        ans = ask(q.question, embedder=embedder, k=k, on_token=on_token)
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001  (Ollama down, a bad hit, ...: note it and go on)
        rec = base_record(q)
        rec.update(seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")
        return rec
    return record_for(q, ans, time.time() - t0, names)


def default_k():
    """retrieve()'s default number of passages, which ask() uses when -k is not given (None if unknown)."""
    try:
        import inspect

        import retrieve
        p = inspect.signature(retrieve.retrieve).parameters.get("k")
        return None if p is None or p.default is inspect.Parameter.empty else p.default
    except Exception:  # noqa: BLE001  (only a label for the report)
        return None


def needs_asking(q, records):
    r = records.get(q.id)
    return r is None or bool(r.get("error")) or r.get("question") != q.question


def run_all(questions, ask, names, records, save, log, embedder=None, k=None, on_token=None):
    """Ask every question that has no good record yet; `records` (id -> record) is updated in place and
    save() is called after every question. -> "done", "stopped" (Ctrl-C) or "failing" """
    todo = [q for q in questions if needs_asking(q, records)]
    in_a_row = 0
    for i, q in enumerate(todo, 1):
        log(f"\n[{i}/{len(todo)}] {q.id} ({q.expect}) {q.question}")
        try:
            rec = ask_one(q, ask, names, embedder, k, on_token)
        except KeyboardInterrupt:
            return "stopped"
        records[q.id] = rec
        save()
        if rec["error"]:
            in_a_row += 1
            log(f"  FAILED: {rec['error']}")
            if in_a_row >= STOP_AFTER:
                return "failing"
            continue
        in_a_row = 0
        log(("\n" if on_token else "") + f"  ({rec['seconds']:.0f} s; {short_status(rec)})")
    return "done"


def short_status(r):
    if r["expect"] == "decline":
        return f"declined: {_yes(r['declined'])}"
    where = ",".join(map(str, r["passage_numbers"])) or "not given"
    return (f"answer passage: {where}; cites it: {_yes(r['cites_passage'])}; "
            f"must {len(r['must_found'])}/{len(r['must'])}; declined: {_yes(r['declined'])}")


def _yes(v):
    return "?" if v is None else "yes" if v else "no"


def _options(embedder, k):
    parts = ([f"-e {embedder}"] if embedder else []) + ([f"-k {k}"] if k else [])
    return " ".join(parts) or "answer.py's defaults"


# --- files -----------------------------------------------------------------------------------

def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _indent(text, pad="    "):
    lines = str(text).rstrip().splitlines() or ["(empty)"]
    return "\n".join(pad + ln if ln.strip() else "" for ln in lines)


def transcript(data):
    """week11_answers.txt, built from the saved data alone."""
    recs = data["questions"]
    out = []
    say = out.append
    s = data.get("settings", {})
    answers = [r for r in recs if r["expect"] == "answer" and not r.get("error")]
    declines = [r for r in recs if r["expect"] == "decline" and not r.get("error")]
    failed = [r for r in recs if r.get("error")]

    say("Casefile Week 11: the answer key asked through ask(), for grading")
    say(f"Run      started {data.get('made', '?')}, last saved {data.get('updated', '?')}, eval_answers.py")
    say(f"Key      {data['key']['file']}: sha256 {data['key']['sha256'][:12]}, "
        f"{data['key'].get('questions', '?')} questions")
    say(f"Model    {s.get('chat_model', '?')}, temperature {s.get('temperature', '?')}, "
        f"context {s.get('num_ctx', '?')} tokens; generation is not seeded")
    say(f"Search   {s.get('index', '?')}, {s.get('k_text', '?')}; each question asked on its own (no history)")
    say(f"Answered {len(answers) + len(declines)} of {data['key'].get('questions', len(recs))}"
        + (f"; {len(failed)} failed (asked again on --resume)" if failed else ""))

    say("")
    say("Automatic checks (they help the grading; they do not grade):")
    say("  passage  = numbers of the passages given that hold the answer (- = none was given)")
    say("  cites    = the answer cites one of them;  must = the key's must: texts found in the answer")
    say("  declined = the citation check reads it as 'the passages do not cover this'")
    say("  check    = problems the citation check found (wording and citation numbers, not truth)")
    say("")
    iw = max([len(r["id"]) for r in recs] + [2])
    say(f"  {'id':<{iw}}  {'expect':<7}  {'passage':<8}  {'cites':<5}  {'must':<5}  {'declined':<8}  "
        f"{'check':<5}  time")
    for r in recs:
        if r.get("error"):
            say(f"  {r['id']:<{iw}}  {r['expect']:<7}  FAILED: {r['error'][:70]}")
            continue
        if r["expect"] == "answer":
            passage = ",".join(map(str, r["passage_numbers"])) or "-"
            cites = _yes(r["cites_passage"]) if r["passage_numbers"] else "-"
            must = f"{len(r['must_found'])}/{len(r['must'])}"
        else:
            passage, cites, must = ".", ".", "."
        say(f"  {r['id']:<{iw}}  {r['expect']:<7}  {passage:<8}  {cites:<5}  {must:<5}  "
            f"{_yes(r['declined']):<8}  {len(r['check_problems']):<5}  {r['seconds']:.0f} s")
    if answers:
        given = [r for r in answers if r["passage_in_context"]]
        say("")
        say(f"  Answer questions: the answer passage was given for {len(given)} of {len(answers)}; "
            f"cited in {sum(1 for r in given if r['cites_passage'])} of those {len(given)}.")
        say(f"  All must: texts found: {sum(1 for r in answers if len(r['must_found']) == len(r['must']))} "
            f"of {len(answers)}. Read as declines: {sum(1 for r in answers if r['declined'])} of {len(answers)} "
            f"answer questions.")
    if declines:
        say(f"  Decline questions read as declines: {sum(1 for r in declines if r['declined'])} of {len(declines)}.")
    secs = [r["seconds"] for r in recs if not r.get("error")]
    if secs:
        say(f"  Time: {sum(secs) / 60:.1f} minutes in all, {min(secs):.0f}-{max(secs):.0f} s per question.")

    for r in recs:
        say("")
        say(f"{RULE} {r['id']} [{r['expect']}] {r['question']}")
        if r.get("error"):
            say(f"FAILED after {r['seconds']:.0f} s: {r['error']}")
            continue
        if r["expect"] == "answer":
            if r["passage_numbers"]:
                where = ", ".join(f"[{n}]" for n in r["passage_numbers"])
                say(f"The answer is in passage {where}; the answer {'cites' if r['cites_passage'] else 'does not cite'} it.")
            elif r["page_in_context"]:
                say("No passage holding the answer was given (only other parts of the right page).")
            else:
                say("No passage holding the answer was given, nor any part of the right page.")
            found = ", ".join(r["must_found"]) or "none"
            say(f"must: {len(r['must_found'])} of {len(r['must'])} found ({found}). Declined: {_yes(r['declined'])}.")
        else:
            say(f"Decline question: the corpus does not hold the answer. Declined: {_yes(r['declined'])}.")
        if r["check_problems"]:
            say(f"Citation check: {len(r['check_problems'])} problem(s):")
            for p in r["check_problems"]:
                say(f"  - {p}")
        else:
            say("Citation check: no problems.")
        tokens = ""
        if r.get("prompt_tokens") is not None:
            tokens = f"; prompt {r['prompt_tokens']} tokens, answer {r.get('answer_tokens')} tokens"
        say(f"Time {r['seconds']:.0f} s{tokens}" + ("; the answer was cut off (truncated)" if r.get("truncated") else "")
            + f". Cited: {', '.join(f'[{n}]' for n in r['cited']) or 'nothing'}.")
        say("")
        say("  Casefile's answer:")
        say(_indent(r["answer"]))
        say("")
        say("  The key's expected answer:")
        say(_indent(r["expected"]))
        say("")
        say("  Passages given (* = holds the answer, p = right page, e = an evidence phrase from another file):")
        for p in r["passages"]:
            mark = "*" if p["passage"] else "p" if p["page"] else "e" if p["evidence"] else " "
            sim = "  -  " if p["similarity"] is None else f"{p['similarity']:.3f}"
            where = p["source"] + (f" > {p['section']}" if p["section"] else "")
            say(f"    [{p['n']}] {mark} {sim}  {where}")
    return "\n".join(out) + "\n"


# --- self-test (made-up files and a fake ask(); no Ollama) -----------------------------------

def _selftest():
    import tempfile
    from types import SimpleNamespace as NS

    from check_key import parse_key

    rows = [
        ("manpages/zorp.man.txt", "ZORP(1)\n\nNAME\n       zorp - count the widgets"),
        ("manpages/zorp.man.txt", "OPTIONS\n       -q, --quiet\n              Quiet mode. Print nothing but errors."),
        ("hacktricks/src/noise.md", "A page that also says: Print nothing but errors."),
        ("hacktricks/src/misc/frob.md", "Run frob --deep to frobnicate every widget."),
    ]
    corpus = Corpus("(test corpus)")
    for src, text in rows:
        corpus.add({"text": text, "metadata": {"source": src}})
    key = ("id: t1\nquestion: what does zorp -q do?\nsource: corpus/manpages/zorp.man.txt\n"
           "evidence: Print nothing but errors.\nmust: -q\nmust: Quiet\nanswer: It prints only errors.\n\n"
           "id: t2\nquestion: how do I frobnicate?\nsource: corpus/hacktricks/src/misc/frob.md\n"
           "evidence: Run frob --deep\nanswer: Run frob --deep.\n\n"
           "id: t3\nquestion: how do I bake bread?\nexpect: decline\nanswer: Not in the corpus.\n")
    qs, findings = parse_key(key)
    findings += check_corpus(qs, corpus)
    t1, t2, t3 = qs
    names = FileNames(corpus)

    def hit(i):
        return NS(id=f"c{i}", text=rows[i][1], source=rows[i][0], also_in=[], section="S", similarity=0.7 - i / 100)

    def finding(level, message):
        return NS(level=level, message=message)

    def answer(text, hits, cited=None, declined=False, findings=()):
        check = None if cited is None else NS(cited=cited, declined=declined, findings=list(findings))
        return NS(text=text, hits=hits, prompt_tokens=900, answer_tokens=80, truncated=False, check=check)

    replies = {
        t1.question: answer("Use `-Q`, the QUIET flag [2]. It prints only errors [2].", [hit(0), hit(1), hit(2)],
                            cited=[2], findings=[finding("NOTE", "fine"), finding("WARN", "a flag is in no passage")]),
        t2.question: answer("The passages do not cover this.", [hit(0), hit(2)], cited=[], declined=True),
        t3.question: answer("The passages do not say how to bake bread.", [hit(3)], cited=[], declined=True),
    }

    def fake_ask(query, embedder=None, k=None, on_token=None):
        if on_token:
            on_token("...")
        return replies[query]

    def run(ask, records=None, questions=qs):
        records = {} if records is None else records
        saves = []
        status = run_all(questions, ask, names, records, lambda: saves.append(len(records)), lambda m: None)
        return status, records, saves

    def flaky(fail_on):
        def ask(query, embedder=None, k=None, on_token=None):
            if query in fail_on:
                raise fail_on[query]
            return fake_ask(query)
        return ask

    def ctrl_c_kept():
        try:
            status, records, saves = run(flaky({t2.question: KeyboardInterrupt()}))
        except KeyboardInterrupt:
            return False
        return status == "stopped" and list(records) == ["t1"] and saves == [1]

    def data_for(records):
        return {"made": "2026-01-01 00:00", "updated": "2026-01-01 00:10",
                "key": {"file": "(test key)", "sha256": "0" * 64, "questions": len(qs)},
                "settings": {"chat_model": "test", "temperature": 0.2, "num_ctx": 8192,
                             "index": "(test index)", "k_text": "top 5"},
                "questions": [records[q.id] for q in qs if q.id in records]}

    def round_trip():
        _, records, _ = run(fake_ask)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.json")
            save_json(p, data_for(records))
            with open(p, encoding="utf-8") as f:
                back = json.load(f)
        text = transcript(back)
        return (back["questions"][0]["passage_numbers"] == [2] and "The key's expected answer:" in text
                and "the answer passage was given for 1 of 2; cited in 1 of those 1." in text
                and "Decline questions read as declines: 1 of 1." in text and "    [2] * 0.690" in text
                and "    [3] e 0.680" in text and not os.path.exists(p + ".tmp"))

    cases = [
        ("the test key and corpus check clean",
         lambda: not [f for f in findings if f.level in ("ERROR", "WARN", "SMOKE")]),
        ("the passages that hold the answer are found and numbered as the model saw them",
         lambda: (lambda r: r["passage_numbers"] == [2] and r["passage_in_context"] and r["page_in_context"])(
             record_for(t1, replies[t1.question], 5, names))),
        ("citing that passage is noticed; must: texts are found whatever their case and markup",
         lambda: (lambda r: r["cites_passage"] and r["must_found"] == ["-q", "Quiet"] and r["cited"] == [2])(
             record_for(t1, replies[t1.question], 5, names))),
        ("only ERROR and WARN lines of the citation check are kept as problems",
         lambda: record_for(t1, replies[t1.question], 5, names)["check_problems"] == ["WARN: a flag is in no passage"]),
        ("an answer with no passage given: passage - , page tracked, decline flag kept",
         lambda: (lambda r: r["passage_numbers"] == [] and r["passage_in_context"] is False
                  and r["page_in_context"] is True and r["declined"] is True and r["cites_passage"] is False)(
             record_for(t1, answer("Not covered.", [hit(0)], cited=[], declined=True), 5, names))),
        ("a decline question has no passage fields; with no citation check, [n] marks are read instead",
         lambda: (lambda r: r["passage_in_context"] is None and r["cites_passage"] is None and r["declined"] is None
                  and r["cited"] == [1])(record_for(t3, answer("Bread [1] and [9].", [hit(3)]), 5, names))),
        ("a full run asks every question, streams, and saves after each one",
         lambda: (lambda st, recs, saves: st == "done" and list(recs) == ["t1", "t2", "t3"] and saves == [1, 2, 3])(
             *run(fake_ask))),
        ("a failed question is recorded and the run goes on; three failures in a row stop it",
         lambda: (lambda st, recs, _: st == "done" and recs["t2"]["error"].startswith("RuntimeError")
                  and recs["t3"]["error"] is None)(*run(flaky({t2.question: RuntimeError("Ollama is down")})))
         and run(flaky({q.question: RuntimeError("down") for q in qs}))[0] == "failing"),
        ("Ctrl-C stops the run with what was done kept",
         ctrl_c_kept),
        ("--resume asks only what is missing, failed, or reworded in the key",
         lambda: (lambda recs: [q.id for q in qs if needs_asking(q, recs)] == ["t2", "t3"])(
             {"t1": dict(record_for(t1, replies[t1.question], 5, names)),
              "t2": dict(base_record(t2), error="RuntimeError: x"),
              "t3": dict(record_for(t3, replies[t3.question], 5, names), question="an older wording")})),
        ("the saved .json loads back and the transcript is built from it alone",
         round_trip),
        ("a failed question shows in the transcript",
         lambda: "FAILED: RuntimeError: Ollama is down" in transcript(
             data_for(run(flaky({t2.question: RuntimeError("Ollama is down")}))[1]))),
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
    ap = argparse.ArgumentParser(description="Ask Casefile every question of the answer key and save the answers "
                                             "for grading (Week 11).")
    ap.add_argument("--only", nargs="+", metavar="ID", help="ask only these questions, e.g. q07")
    ap.add_argument("--resume", action="store_true", help="continue the saved run; answers already saved are kept")
    ap.add_argument("--report", action="store_true", help="rewrite the .txt from the saved .json; asks nothing")
    ap.add_argument("--quiet", action="store_true", help="do not show the answers as they are written")
    ap.add_argument("-e", "--embedder", help="search this index instead of answer.py's default")
    ap.add_argument("-k", type=int, help="passages per question instead of answer.py's default")
    ap.add_argument("--key", default=KEY_FILE, help=f"the answer key (default {KEY_FILE})")
    ap.add_argument("--chunks", default=CHUNKS_FILE, help=f"the chunk file (default {CHUNKS_FILE})")
    ap.add_argument("--out", default=OUT_PREFIX, help=f"write OUT.json and OUT.txt (default {OUT_PREFIX})")
    ap.add_argument("--selftest", action="store_true", help="run the built-in tests and exit")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    if args.selftest:
        return 1 if _selftest() else 0
    json_path, txt_path = args.out + ".json", args.out + ".txt"

    def log(message):
        print(message, flush=True)

    def write_txt(data):
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript(data))

    if args.report:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"error: cannot read {json_path} ({e})")
            return 2
        write_txt(data)
        print(f"Wrote {txt_path} from {json_path} ({len(data['questions'])} question(s)).")
        return 0

    if args.k is not None and args.k < 1:
        ap.error("-k must be at least 1")
    exists = os.path.exists(json_path)
    if exists and not args.resume:
        print(f"error: {json_path} already exists. Use --resume to continue that run, or --out NAME "
              "to start a separate one.")
        return 2
    if args.resume and not exists:
        print(f"error: there is no {json_path} to resume. Leave out --resume to start a run.")
        return 2

    try:
        questions = load_key(args.key)
    except OSError as e:
        print(f"error: cannot read {args.key} ({e.strerror or e}). "
              "Run this from the project root (cd ~/projects/PenetrationTesting).")
        return 2
    except ValueError as e:
        print(f"error: {e}")
        return 2
    ids = [q.id for q in questions]
    try:
        only = set(pick_ids(args.only, ids)) if args.only else set(ids)
    except ValueError as e:
        print(f"error: {str(e).replace('--show', '--only')}")
        return 2
    if not os.path.isfile(args.chunks):
        print(f"error: {args.chunks} not found. Run this from the project root (cd ~/projects/PenetrationTesting).")
        return 2
    try:
        corpus = load_corpus(args.chunks)
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 2
    errors = [f for f in check_corpus(questions, corpus) if f.level == "ERROR"]
    if errors:
        print(f"error: the key has {len(errors)} error(s) against {args.chunks} (the first: {errors[0].message}). "
              "Run python check_key.py and fix them first.")
        return 2

    try:
        import answer
    except ImportError as e:
        print(f"error: cannot import answer.py ({e}). Run this from the project root with the venv active "
              "(source ~/.venv/bin/activate).")
        return 2
    index_name = getattr(answer, "index_name", None)
    settings = {
        "chat_model": getattr(answer, "CHAT_MODEL", None), "temperature": getattr(answer, "TEMPERATURE", None),
        "num_ctx": getattr(answer, "NUM_CTX", None),
        "answer_sees_history": getattr(answer, "ANSWER_SEES_HISTORY", None),
        "embedder": args.embedder, "k": args.k,
        "index": index_name(args.embedder) if callable(index_name) else (args.embedder or "answer.py default"),
        "k_text": f"top {args.k}" if args.k else f"top {default_k() or '?'} (retrieve.py's default)",
    }

    if args.resume:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        old = data.get("settings", {})
        if (old.get("embedder"), old.get("k")) != (args.embedder, args.k):
            print(f"error: {json_path} was made with {_options(old.get('embedder'), old.get('k'))}, and this run asks "
                  f"for {_options(args.embedder, args.k)}. Resume it with the same options, or use --out NAME "
                  "for a separate run.")
            return 2
    else:
        data = {"made": time.strftime("%Y-%m-%d %H:%M"), "script": "eval_answers.py"}
    data["settings"] = settings
    data["key"] = {"file": args.key, "sha256": sha256_of(args.key), "questions": len(questions)}
    data["corpus"] = {"file": args.chunks, "chunks": len(corpus.texts), "files": len(corpus.files)}
    records = {r["id"]: r for r in data.get("questions", [])}

    def save():
        data["questions"] = [records[i] for i in ids if i in records]
        data["updated"] = time.strftime("%Y-%m-%d %H:%M")
        save_json(json_path, data)

    def stream(piece):
        sys.stdout.write(piece)
        sys.stdout.flush()

    todo = [q for q in questions if q.id in only and needs_asking(q, records)]
    log(f"{len(todo)} question(s) to ask ({len(records)} already saved in {json_path}). Each takes about "
        "1.5-2.5 minutes; Ctrl-C stops after saving, and --resume continues.")
    try:
        status = run_all(todo, answer.ask, FileNames(corpus), records, save, log, args.embedder, args.k,
                         None if args.quiet else stream)
    finally:                   # also when answer.py itself stops the program
        save()
        write_txt(data)
    left = [q.id for q in questions if needs_asking(q, records)]
    print("")
    if status == "stopped":
        print(f"Stopped. Saved {len(records)} answer(s) to {json_path} and {txt_path}; "
              "run python eval_answers.py --resume to continue.")
        return 130
    if status == "failing":
        errs = [records[q.id]["error"] for q in todo if (records.get(q.id) or {}).get("error")]
        print(f"Stopped: {STOP_AFTER} questions failed in a row (the last: {errs[-1] if errs else '?'}). "
              "Is Ollama running? Fix it, then run python eval_answers.py --resume.")
        return 1
    print(f"Wrote {json_path} and {txt_path}. "
          + (f"Still to ask: {', '.join(left)} (run with --resume)." if left else "All questions are answered."))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:        # output piped into head or less, which closed early
        sys.exit(0)
