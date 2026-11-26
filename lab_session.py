#!/usr/bin/env python3
"""
lab_session.py -- Casefile: log a practice-lab session, then build the report from the log

While you work a TryHackMe room (or an HTB lab, or Juice Shop on your own laptop), ask Casefile
your reference questions through this script instead of answer.py. It calls answer.py's own
run_once(), so you see exactly what you always see, and it saves each question, answer, sources
and timing to a log file. After every answer it asks how it went: did it work, did it need a fix,
was it wrong or made up. Later, the report mode turns the log into a Markdown report.

    python lab_session.py --room basic-pentesting          # a session (run from the project root)
    python lab_session.py --room basic-pentesting -k 8     # answer.py's options: -e -k --no-cutoff
    python lab_session.py report --log logs/basic-pentesting.jsonl --author "Your Name"
    python lab_session.py --selftest                       # built-in tests, no Ollama needed

In a session, type a question, or one of:
    /outcome  relabel the last answer      /note TEXT  add a note to the last answer
    /finding  record a weakness you found  /count      the tally so far
    /history  what the chat remembers      /reset      forget the conversation
    /help     this list                    quit        end the session

What this script does and does not do:
  * It only records what you type and what Casefile printed. Nothing is sent anywhere, no network
    is used, and nothing is run against any machine.
  * Anything you type that looks like a flag (THM{...}, HTB{...}), a long hash or "password: ..."
    is removed before it is saved. Reports must not contain them. The pattern check cannot catch
    everything, so read what you type.
  * The log is append-only (one JSON object per line), so a crash or Ctrl-C loses nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

# --- the outcome labels (the same keys are used in the log and the report) -----------------

OUTCOMES = {
    "w": ("worked", "I used it and it did what the answer said"),
    "f": ("needed a fix", "partly right: I had to change something"),
    "x": ("wrong or made up", "a flag, command or fact in it was false"),
    "c": ("correct decline", "it said nothing found / not covered, and that was right"),
    "m": ("missed", "it declined or found nothing, but the books do cover it"),
    "u": ("not tried", "I read it but did not or could not try it"),
}
SEVERITIES = ("low", "medium", "high", "critical")

MENU = """How did it go?
  w worked          f needed a fix        x wrong or made up
  c correct decline m missed (the books do cover it)    u not tried
  Enter = decide later (use /outcome)"""

HELP = """Type a question, or:
  /outcome        relabel the last answer
  /note TEXT      add a note to the last answer (what was wrong, what you changed)
  /finding        record a weakness you found in the lab (with how to fix it)
  /count          the tally so far
  /history        what the chat remembers      /reset   forget the conversation
  quit            end the session
Never type flags, passwords or cracked hashes: they are removed, and reports must not have them."""

# --- keeping secrets out of the log --------------------------------------------------------

_FLAG = re.compile(r"\b(?:THM|HTB|FLAG|CTF)\{[^}\n]*\}", re.I)
_HASH = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_PASS = re.compile(r"(?i)\b(pass(?:word)?|passwd|pwd)\b(\s*[:=]\s*)\S+")


def redact(text):
    """Remove anything that looks like a flag, a long hash or 'password: ...'.
    Returns (clean text, how many things were removed)."""
    count = 0
    out = str(text or "")
    for rx, repl in ((_FLAG, "[flag removed]"), (_HASH, "[hash removed]"),
                     (_PASS, r"\1\2[removed]")):
        out, k = rx.subn(repl, out)
        count += k
    return out, count


# --- small helpers --------------------------------------------------------------------------

def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "session"


def sha256_of(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def count_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def git_head(folder):
    try:
        out = subprocess.run(["git", "-C", folder, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        s = out.stdout.strip()
        return s if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", s) else None
    except (OSError, subprocess.SubprocessError):
        return None


def append_log(path, record):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(path):
    """-> (records, number of unreadable lines). A damaged line is skipped, never fatal."""
    records, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(rec, dict):
                records.append(rec)
            else:
                bad += 1
    return records, bad


def _g(obj, name, default=None):
    return default if obj is None else getattr(obj, name, default)


# --- reading answer.py's results ------------------------------------------------------------

def _hit_label(mod, hit):
    fn = getattr(mod, "hit_label", None)
    try:
        return str(fn(hit)) if callable(fn) else str(hit)
    except Exception:  # noqa: BLE001
        return str(hit)


def _hit_sim(mod, hit):
    fn = getattr(mod, "hit_similarity", None)
    try:
        v = fn(hit) if callable(fn) else None
        return None if v is None else round(float(v), 4)
    except Exception:  # noqa: BLE001
        return None


def _hit_source(mod, hit):
    fn = getattr(mod, "hit_get", None)
    try:
        return str(fn(hit, "source", default="")) if callable(fn) else ""
    except Exception:  # noqa: BLE001
        return ""


def qa_record(n, query, ans, seconds, mod, error=None):
    """One question and what Casefile gave back, as a log record. No chunk text is stored."""
    hits = list(_g(ans, "hits", None) or [])
    check = _g(ans, "check")
    cited = sorted({int(c) for c in (_g(check, "cited", None) or [])})
    sources = [{"n": i, "label": _hit_label(mod, h), "source": _hit_source(mod, h),
                "similarity": _hit_sim(mod, h), "cited": i in cited}
               for i, h in enumerate(hits, 1)]
    sims = [s["similarity"] for s in sources if s["similarity"] is not None]
    top = _g(ans, "top_similarity")
    if top is None and sims:
        top = max(sims)
    problems = []
    for f in (_g(check, "findings", None) or []):
        level = str(_g(f, "level", "") or "")
        if level and level != "NOTE":
            problems.append(f"{level}: {_g(f, 'message', '')}")
    clean_q, _ = redact(query)
    return {
        "type": "qa", "n": n, "time": now_iso(), "question": clean_q,
        "search_query": str(_g(ans, "search_query", "") or ""),
        "answer": str(_g(ans, "text", "") or ""),
        "nothing_found": bool(_g(ans, "nothing_found", False)),
        "top_similarity": top, "cutoff": _g(ans, "cutoff"),
        "seconds": round(seconds, 1),
        "prompt_tokens": _g(ans, "prompt_tokens"), "answer_tokens": _g(ans, "answer_tokens"),
        "truncated": bool(_g(ans, "truncated", False)),
        "cited": cited,
        "declined": None if check is None else bool(_g(check, "declined", False)),
        "check_problems": problems, "sources": sources, "error": error,
    }


def session_header(args, mod):
    index = None
    fn = getattr(mod, "index_name", None)
    if callable(fn):
        try:
            index = fn(args.embedder)
        except Exception:  # noqa: BLE001
            pass
    cutoffs = getattr(mod, "CUTOFFS", None) or {}
    return {
        "type": "session", "time": now_iso(), "room": args.room,
        "chat_model": getattr(mod, "CHAT_MODEL", None),
        "temperature": getattr(mod, "TEMPERATURE", None),
        "num_ctx": getattr(mod, "NUM_CTX", None),
        "index": index, "k": args.k,
        "cutoff": None if args.no_cutoff else cutoffs.get(index),
        "history": not args.no_history,
        "answer_py_sha256": sha256_of(getattr(mod, "__file__", None) or "answer.py"),
        "chunks": count_lines(os.path.join("corpus", "chunks.jsonl")),
        "hacktricks_commit": git_head(os.path.join("corpus", "hacktricks")),
    }


# --- collating a log ------------------------------------------------------------------------

def collate(records):
    headers = [r for r in records if r.get("type") == "session"]
    qas = sorted((r for r in records if r.get("type") == "qa"), key=lambda r: r.get("n", 0))
    labels, notes, findings = {}, {}, []
    for r in records:
        t = r.get("type")
        if t == "label":
            labels[r.get("n")] = r            # the last label for a question wins
        elif t == "note":
            notes.setdefault(r.get("n") or 0, []).append(str(r.get("text", "")))
        elif t == "finding":
            findings.append(r)
    return {"header": headers[-1] if headers else None, "sittings": len(headers),
            "qa": qas, "labels": labels, "notes": notes, "findings": findings}


def tally(c):
    """-> (Counter of outcome keys, number of answers with no label yet). Questions that produced
    no answer are not counted."""
    counts, unlabelled = Counter(), 0
    for q in c["qa"]:
        if q.get("error"):
            continue
        lab = c["labels"].get(q.get("n"))
        if lab and lab.get("outcome") in OUTCOMES:
            counts[lab["outcome"]] += 1
        else:
            unlabelled += 1
    return counts, unlabelled


def summary_lines(records):
    c = collate(records)
    counts, unlabelled = tally(c)
    total = sum(counts.values()) + unlabelled
    lines = [f"{total} answer(s) so far: "
             + (", ".join(f"{counts[k]} {OUTCOMES[k][0]}" for k in OUTCOMES if counts[k]) or "none labelled")
             + (f"; {unlabelled} not labelled yet" if unlabelled else "")]
    if c["findings"]:
        lines.append(f"{len(c['findings'])} finding(s) recorded")
    return lines


# --- the interactive session ----------------------------------------------------------------

def typed(read, say, prompt):
    """A line typed by the user, with anything that looks like a secret removed."""
    clean, k = redact(read(prompt).strip())
    if k:
        say(f"  (removed {k} flag, hash or password from that line: reports must not contain them)")
    return clean


def ask_choice(read, say):
    say(MENU)
    for _ in range(3):
        raw = read("Outcome> ").strip().lower()
        if raw == "":
            return None
        if raw in OUTCOMES:
            return raw
        say("  please type one of: " + " ".join(OUTCOMES) + " (or just Enter to decide later)")
    return None


def label_flow(n, log_path, read, say):
    key = ask_choice(read, say)
    if key is None:
        say("  (not labelled yet: /outcome sets it)")
        return None
    command = "" if key in ("c", "u") else typed(read, say, "What did you run or do? (Enter to skip)> ")
    hint = {"x": "What was wrong", "m": "Where in the books was it", "f": "What did you change"}.get(key, "Note")
    note = typed(read, say, f"{hint}? (Enter to skip)> ")
    append_log(log_path, {"type": "label", "n": n, "outcome": key, "command": command,
                          "note": note, "time": now_iso()})
    return key


def finding_flow(last, log_path, read, say):
    say("Record a weakness you found in the lab. No flags or passwords, please.")
    title = typed(read, say, "Title (e.g. 'Login form allows SQL injection')> ")
    if not title:
        say("  (cancelled)")
        return False
    where = typed(read, say, "Where (page, service or port)> ")
    severity = ""
    for _ in range(2):
        severity = read(f"Severity [{'/'.join(SEVERITIES)}]> ").strip().lower()
        if severity in SEVERITIES:
            break
        severity = ""
    what = typed(read, say, "What it is and how you found it> ")
    fix = typed(read, say, "How to fix it> ")
    append_log(log_path, {"type": "finding", "time": now_iso(), "n": last or 0, "title": title,
                          "where": where, "severity": severity or "unrated", "what": what, "fix": fix})
    say("  (finding saved)")
    return True


def session(args, mod, read=input, say=None):
    """The interactive loop. `mod` is answer.py (or a stand-in); read/say can be replaced in tests."""
    say = say or (lambda s="": print(s, flush=True))
    log_path = args.log or os.path.join("logs", slug(args.room) + ".jsonl")
    existing = read_log(log_path)[0] if os.path.exists(log_path) else []
    n = max([r.get("n", 0) for r in existing if r.get("type") == "qa"] + [0])
    append_log(log_path, session_header(args, mod))
    say(f"Casefile lab session: {args.room}\nLog: {log_path}"
        + (f" ({n} earlier question(s) kept)" if n else ""))
    say("Only test the machine your lab gives you. After each answer, say how it went; type /help for more.")
    history = None if args.no_history else []
    last = None
    while True:
        try:
            line = read("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            say()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "q", "/quit"):
            break
        if low == "/help":
            say(HELP)
        elif low == "/count":
            for s in summary_lines(read_log(log_path)[0]):
                say(s)
        elif low == "/history":
            show = getattr(mod, "show_history", None)
            if callable(show) and history is not None:
                show(history)
            else:
                say("(the chat is not remembering anything)")
        elif low in ("/reset", "/clear"):
            if history is not None:
                history.clear()
            say("(conversation forgotten)")
        elif low == "/outcome":
            if last is None:
                say("(ask a question first)")
            else:
                label_flow(last, log_path, read, say)
        elif low.startswith("/note"):
            text, k = redact(line[5:].strip())
            if not text:
                say("usage: /note TEXT")
            else:
                append_log(log_path, {"type": "note", "n": last or 0, "text": text, "time": now_iso()})
                say("  (note saved)" + (f" (removed {k} flag, hash or password)" if k else ""))
        elif low == "/finding":
            finding_flow(last, log_path, read, say)
        elif line.startswith("/"):
            say("(unknown command; /help lists them)")
        else:
            n += 1
            started = time.time()
            ans = mod.run_once(line, args.embedder, args.k, False, history=history,
                               rewrite=True, cutoff=not args.no_cutoff)
            seconds = time.time() - started
            last = n
            if not ans:
                append_log(log_path, qa_record(
                    n, line, None, seconds, mod,
                    error="no answer was produced (failed or interrupted; see the message above)"))
                say("  (no answer was produced, so there is nothing to label)")
                continue
            rec = qa_record(n, line, ans, seconds, mod)
            append_log(log_path, rec)
            if rec["nothing_found"]:
                say("  Casefile found nothing close enough. Was that right?  c = yes   m = no, the books do cover it")
            label_flow(n, log_path, read, say)
    records = read_log(log_path)[0]
    say("\n" + "\n".join(summary_lines(records)))
    say(f"Build the report with:  python lab_session.py report --log {log_path} --author \"Your Name\"")
    return 0


# --- the report -----------------------------------------------------------------------------

def pct(a, b):
    return "n/a" if not b else f"{100 * a / b:.0f}%"


def cell(s, limit=None):
    s = " ".join(str(s if s is not None else "").split()).replace("|", "\\|")
    if limit and len(s) > limit:
        s = s[: limit - 3].rstrip() + "..."
    return s


def fenced(text):
    text = str(text or "").rstrip() or "(empty)"
    fence = "~~~~"
    while fence in text:
        fence += "~"
    return f"{fence}text\n{text}\n{fence}"


def build_report(records, author="[Your name]", course="[Course]", title=None, bad_lines=0):
    """Markdown report from a session log. -> (text, stats). Only what the log holds is stated."""
    c = collate(records)
    hdr = c["header"] or {}
    room = hdr.get("room") or "(unknown room)"
    title = title or f"Casefile in a practice lab: {room}"
    qas = c["qa"]
    answered = [q for q in qas if not q.get("error")]
    counts, unlabelled = tally(c)
    labelled = sum(counts.values())
    tried = counts["w"] + counts["f"] + counts["x"]
    no_answer = len(qas) - len(answered)
    times = [q["seconds"] for q in answered if not q.get("nothing_found") and q.get("seconds") is not None]
    stamps = sorted(r["time"] for r in records if r.get("time"))
    first, last = (stamps[0][:10], stamps[-1][:10]) if stamps else ("?", "?")
    redactions = [0]

    def user_text(s):
        clean, k = redact(s)
        redactions[0] += k
        return clean

    out = []

    def add(*lines):
        out.extend(lines)

    add(f"# {title}", "")
    add(f"**Author:** {author}  ")
    add(f"**Course:** {course}  ")
    add(f"**Lab:** {room}  ")
    add(f"**Dates:** {first} to {last} ({c['sittings']} session(s))  ")
    add("**Tool tested:** Casefile, a local retrieval-augmented study assistant (HackTricks and man pages)")
    add("")

    add("## 1. Scope and rules", "")
    add(f"- Everything here was done against the machine or application the lab deployed for me (`{room}`). "
        "Nothing else was scanned or attacked. *(Edit this line if that is not true.)*")
    add("- This report contains no flags, passwords or cracked hashes, which is TryHackMe's rule for writeups. "
        "Anything I typed that looked like one was removed when it was logged. A pattern check cannot catch "
        "everything, so I read the file again before handing it in.")
    add("- AI assistance: Casefile is a local assistant (an 8B language model) that I built with AI help. I used it "
        "as a reference book while working and labelled every result myself. This file was assembled from my "
        "session log by `lab_session.py`; section 11 is my own writing.")
    add("")

    add("## 2. What I tested and why", "")
    add("Casefile answers penetration-testing questions only from the passages it retrieves from HackTricks and "
        "the man pages of 20 tools, cites them, and says so when they do not cover the question. The risk with any "
        "such assistant is a confident answer that is wrong (a made-up flag or command). During the lab I asked it "
        "the how-do-I questions that came up, tried what it said where I could, and recorded whether it held up.")
    add("")

    add("## 3. Setup", "")
    rows = [("Chat model", hdr.get("chat_model")), ("Temperature", hdr.get("temperature")),
            ("Context window (tokens)", hdr.get("num_ctx")), ("Search index", hdr.get("index")),
            ("Passages per question", hdr.get("k") or "answer.py default"),
            ("'Nothing found' cutoff", hdr.get("cutoff") if hdr.get("cutoff") is not None else "off or not set"),
            ("Conversation memory", "on" if hdr.get("history") else "off" if hdr else None),
            ("Chunks in the corpus", f"{hdr['chunks']:,}" if hdr.get("chunks") else None),
            ("HackTricks commit", (hdr.get("hacktricks_commit") or "")[:12] or None),
            ("answer.py sha256", (hdr.get("answer_py_sha256") or "")[:12] or None)]
    add("| Setting | Value |", "|---|---|")
    for k, v in rows:
        if v not in (None, ""):
            add(f"| {k} | {cell(v)} |")
    add("")

    add("## 4. Results at a glance", "")
    if not qas:
        add("No questions were logged.")
    else:
        add(f"I asked {len(qas)} question(s). {len(answered)} produced an answer"
            + (f"; {no_answer} did not (a failure or an interrupted answer)" if no_answer else "")
            + f". {labelled} are labelled"
            + (f"; {unlabelled} are not labelled yet" if unlabelled else "") + ".")
        add("")
        add("| Outcome | Meaning | Count | Share of labelled |", "|---|---|---|---|")
        for k, (name, meaning) in OUTCOMES.items():
            add(f"| {name} | {meaning} | {counts[k]} | {pct(counts[k], labelled)} |")
        add("")
        if tried:
            add(f"Of the {tried} answer(s) I actually tried, {counts['w']} ({pct(counts['w'], tried)}) worked as "
                f"written, {counts['f']} ({pct(counts['f'], tried)}) needed a fix and {counts['x']} "
                f"({pct(counts['x'], tried)}) were wrong or made up.")
        else:
            add("None of the answers has been marked as tried yet, so there is no worked-or-wrong rate.")
        nf = sum(1 for q in answered if q.get("nothing_found"))
        if nf:
            add(f"Casefile answered \"nothing found\" without asking the model {nf} time(s).")
        if times:
            add(f"Answering took {sum(times) / len(times):.0f} s on average ({min(times):.0f} to {max(times):.0f} s).")
    add("")

    add("## 5. Every question", "")
    if qas:
        add("| # | Question | Outcome | Top similarity | Cited | Time |", "|---|---|---|---|---|---|")
        for q in qas:
            lab = c["labels"].get(q["n"])
            outcome = ("no answer" if q.get("error") else
                       OUTCOMES[lab["outcome"]][0] if lab and lab.get("outcome") in OUTCOMES else "not labelled")
            sim = "" if q.get("top_similarity") is None else f"{q['top_similarity']:.3f}"
            cited = ", ".join(f"[{i}]" for i in q.get("cited", [])) or "none"
            add(f"| {q['n']} | {cell(q['question'], 70)} | {outcome} | {sim} | {cited} | {q.get('seconds', 0):.0f} s |")
    add("")

    def listing(heading, key, intro):
        add(f"### {heading}", "")
        items = [q for q in answered if (c["labels"].get(q["n"]) or {}).get("outcome") == key]
        if not items:
            add("None.", "")
            return
        add(intro, "")
        for q in items:
            lab = c["labels"][q["n"]]
            top = ", ".join(f"{s['label']} ({s['similarity']:.2f})" if s.get("similarity") is not None
                            else s["label"] for s in q.get("sources", [])[:3])
            add(f"- **Q{q['n']}: {cell(q['question'], 100)}**")
            if lab.get("command"):
                add(f"  - What I ran or did: {user_text(lab['command'])}")
            note = " ".join(user_text(t) for t in c["notes"].get(q["n"], []))
            add(f"  - What happened: {user_text(lab.get('note', '')) or note or '(no note was written)'}")
            if lab.get("note") and note:
                add(f"  - Added later: {note}")
            add(f"  - Closest passages: {top or 'none'}")
            for p in q.get("check_problems", [])[:3]:
                add(f"  - Citation check: {cell(p, 160)}")
        add("")

    add("## 6. What went wrong", "")
    listing("Wrong or made up", "x", "The answers whose content was false when I tried it:")
    listing("Missed", "m", "Questions where the books do cover the answer but Casefile did not find or use it:")
    listing("Needed a fix", "f", "Partly right:")
    listing("Correct declines", "c", "Nothing found or not covered, and that was right:")

    add("## 7. Similarity and outcome", "")
    by = {}
    for q in answered:
        lab = c["labels"].get(q["n"])
        if lab and lab.get("outcome") in OUTCOMES and q.get("top_similarity") is not None:
            by.setdefault(lab["outcome"], []).append(q["top_similarity"])
    if by:
        add("| Outcome | Questions | Lowest | Mean | Highest |", "|---|---|---|---|---|")
        for k in OUTCOMES:
            if k in by:
                v = by[k]
                add(f"| {OUTCOMES[k][0]} | {len(v)} | {min(v):.3f} | {sum(v) / len(v):.3f} | {max(v):.3f} |")
        add("")
        add("This is only a hint: with so few questions, a gap between rows may be chance. If wrong answers "
            "tend to have lower similarity, a \"low confidence\" label would catch some of them.")
    else:
        add("Not enough labelled answers with a similarity score yet.")
    add("")

    add("## 8. Weaknesses found in the lab", "")
    if not c["findings"]:
        add("No findings were recorded (use `/finding` during a session).")
    else:
        add("These are weaknesses in a deliberately vulnerable practice machine.", "")
        add("| # | Finding | Severity | Where |", "|---|---|---|---|")
        for i, f in enumerate(c["findings"], 1):
            add(f"| {i} | {cell(user_text(f.get('title')), 70)} | {f.get('severity', '')} | "
                f"{cell(user_text(f.get('where')), 50)} |")
        add("")
        for i, f in enumerate(c["findings"], 1):
            add(f"**{i}. {user_text(f.get('title'))}** ({f.get('severity', 'unrated')})", "")
            add(f"- Where: {user_text(f.get('where')) or '(not recorded)'}")
            add(f"- What and how found: {user_text(f.get('what')) or '(not recorded)'}")
            add(f"- How to fix it: {user_text(f.get('fix')) or '(not recorded)'}")
            add("")

    add("## 9. Compared with my earlier test (Week 11)", "")
    add("*From my own Week 11 grading; edit if it is out of date.* I asked Casefile a 20-question answer key "
        "(18 answerable). Grades: 9 correct, 5 partly correct, 1 wrong, 3 honestly declined. When the passage "
        "holding the answer reached the model it got 8 of 10 right and none wrong; when it did not, it got 1 of 8 "
        "right. The lab results above show whether the same pattern holds on questions I did not write.")
    add("")

    add("## 10. Limits", "")
    add("- One lab and a small number of questions" + (
        f": each question moves a percentage by about {100 / tried:.0f} points." if tried else "."))
    add("- The labels are my own judgement and I could be wrong about some of them.")
    add("- The model's wording is not seeded, so the same question can give a different answer.")
    add("- Casefile knows one HackTricks snapshot and 20 man pages. Tools that are not in them (ffuf, gobuster "
        "and others) are covered only where HackTricks mentions them.")
    add("- Answers take about two minutes on my CPU.")
    add("- The citation check compares wording, flags and numbers; it cannot tell whether a claim is true.")
    add("")

    add("## 11. In my own words", "")
    add("> **TODO: write this section yourself before handing the report in.** Answer in a few sentences each.")
    add("")
    add("- The hardest question I asked, and what Casefile did with it: **TODO**")
    add("- Where it saved me time, and where it wasted it: **TODO**")
    add("- A time it sounded sure and was wrong, and what gave it away: **TODO**")
    add("- The first thing I would change, and why: **TODO**")
    add("")

    add("## Appendix: every answer in full", "")
    for q in qas:
        lab = c["labels"].get(q["n"]) or {}
        add(f"### Q{q['n']}. {cell(q['question'], 120)}", "")
        if q.get("error"):
            add(f"No answer: {q['error']}", "")
            continue
        if q.get("search_query") and q["search_query"] != q["question"]:
            add(f"Searched for: {cell(q['search_query'], 160)}  ")
        add(f"Outcome: {OUTCOMES[lab['outcome']][0] if lab.get('outcome') in OUTCOMES else 'not labelled'}  ")
        if lab.get("command"):
            add(f"What I ran or did: {user_text(lab['command'])}  ")
        if lab.get("note"):
            add(f"My note: {user_text(lab['note'])}  ")
        for t in c["notes"].get(q["n"], []):
            add(f"Later note: {user_text(t)}  ")
        add("")
        add(fenced(q.get("answer")), "")
        if q.get("sources"):
            add("Passages given to the model (* = cited in the answer):", "")
            for s in q["sources"]:
                sim = "" if s.get("similarity") is None else f" ({s['similarity']:.3f})"
                add(f"- [{s['n']}]{'*' if s.get('cited') else ''} {cell(s['label'], 120)}{sim}")
            add("")
        for p in q.get("check_problems", []):
            add(f"- Citation check: {cell(p, 200)}")
        if q.get("check_problems"):
            add("")
    for n in sorted(k for k in c["notes"] if k == 0):
        add("### General notes", "")
        for t in c["notes"][n]:
            add(f"- {user_text(t)}")
        add("")

    text = "\n".join(out).rstrip() + "\n"
    flags_left = len(_FLAG.findall(text))
    missing = [q["n"] for q in answered
               if (c["labels"].get(q["n"]) or {}).get("outcome") in ("x", "m")
               and not (c["labels"][q["n"]].get("note") or c["notes"].get(q["n"]))]
    stats = {"questions": len(qas), "answered": len(answered), "labelled": labelled,
             "unlabelled": unlabelled, "no_answer": no_answer, "tried": tried,
             "missing_notes": missing, "redactions": redactions[0], "flags_in_report": flags_left,
             "unreadable_lines": bad_lines, "todo_left": text.count("**TODO**")}
    return text, stats


def report_cli(argv):
    ap = argparse.ArgumentParser(prog="lab_session.py report",
                                 description="Build the Markdown report from a session log.")
    ap.add_argument("--log", required=True, help="the .jsonl log written by a session")
    ap.add_argument("--out", help="the report file (default: the log's name with .md)")
    ap.add_argument("--author", default="[Your name]")
    ap.add_argument("--course", default="[Course]")
    ap.add_argument("--title")
    args = ap.parse_args(argv)
    try:
        records, bad = read_log(args.log)
    except OSError as e:
        print(f"error: cannot read {args.log} ({e.strerror or e}). Run a session first, from the project root.")
        return 2
    text, st = build_report(records, args.author, args.course, args.title, bad)
    out = args.out or os.path.splitext(args.log)[0] + ".md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {out}: {st['questions']} question(s), {st['labelled']} labelled.")
    if st["unlabelled"]:
        print(f"  - {st['unlabelled']} answer(s) have no label: start a session and use /outcome, or label them next time.")
    if st["missing_notes"]:
        print("  - wrong or missed answers with no note: " + ", ".join(f"Q{n}" for n in st["missing_notes"])
              + " (add one with /note so the report can say what went wrong)")
    if st["unreadable_lines"]:
        print(f"  - {st['unreadable_lines']} damaged line(s) in the log were skipped")
    if st["flags_in_report"]:
        print(f"  - WARNING: {st['flags_in_report']} flag-like text(s) are in the report. Remove them before handing it in.")
    if st["todo_left"]:
        print(f"  - section 11 still has {st['todo_left']} TODO line(s): write them yourself before handing it in")
    return 0


# --- command line ---------------------------------------------------------------------------

def session_cli(argv):
    ap = argparse.ArgumentParser(
        description="Casefile lab session: ask questions, say how each answer went, keep a log "
                    "(then: python lab_session.py report --log LOG).")
    ap.add_argument("--room", help="a short name for the lab, e.g. basic-pentesting (names the log file)")
    ap.add_argument("--log", help="the log file (default: logs/ROOM.jsonl)")
    ap.add_argument("-e", "--embedder", help="which index to search (default: retrieve.py's own)")
    ap.add_argument("-k", type=int, help="how many passages per question (default: retrieve.py's own)")
    ap.add_argument("--no-cutoff", action="store_true", help="ask the chat model even when nothing is close")
    ap.add_argument("--no-history", action="store_true", help="ask every question on its own")
    ap.add_argument("--no-checks", action="store_true", help="skip the start-up checks")
    ap.add_argument("--selftest", action="store_true", help="run the built-in tests and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        return 1 if _selftest() else 0
    if not args.room:
        ap.error("--room is required, e.g. --room basic-pentesting")
    if args.k is not None and args.k < 1:
        ap.error("-k must be at least 1")
    try:
        import answer
    except ImportError as e:
        print(f"error: cannot import answer.py ({e}). Run this from the project root with the venv active "
              "(source ~/.venv/bin/activate).")
        return 2
    if not args.no_checks:
        problems = answer.preflight(args.embedder) if hasattr(answer, "preflight") else []
        if problems:
            print("Casefile cannot start yet:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 2
    return session(args, answer)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "report":
        return report_cli(argv[1:])
    return session_cli(argv)


# --- self-test (a stand-in for answer.py; no Ollama) ----------------------------------------

def _selftest():
    import tempfile
    from types import SimpleNamespace as NS

    def hit(label, sim):
        return NS(label=label, source=label.replace(" ", "/"), sim=sim)

    def answer_obj(text, hits, cited=(1,), declined=False, nothing=False, top=None):
        check = NS(cited=list(cited), declined=declined, findings=[NS(level="NOTE", message="fine")]
                   + ([NS(level="WARN", message="a flag is in no passage")] if not declined and not nothing else []))
        return NS(text=text, hits=hits, check=check, search_query="", nothing_found=nothing,
                  top_similarity=top, cutoff=0.62 if nothing else None, prompt_tokens=1200,
                  answer_tokens=80, truncated=False)

    script = {
        "scan": answer_obj("Use `nmap -p-` [1].", [hit("man nmap > PORT SPEC", 0.83), hit("hacktricks/nmap", 0.8)], top=0.83),
        "flag": answer_obj("Use `ffuf --magic` [1].", [hit("hacktricks/web", 0.66)], top=0.66),
        "bread": answer_obj("Nothing found: no passage is close enough.", [hit("hacktricks/seimpersonate", 0.58)],
                            cited=(), declined=True, nothing=True, top=0.58),
        "snort": answer_obj("The passages do not cover Snort.", [hit("man nmap > X", 0.7)], cited=(), declined=True, top=0.7),
    }
    mod = NS(CHAT_MODEL="test-model", TEMPERATURE=0.2, NUM_CTX=8192, CUTOFFS={"idx": 0.62},
             index_name=lambda e: "idx", hit_label=lambda h: h.label, hit_get=lambda h, n, default=None: h.source,
             hit_similarity=lambda h: h.sim, show_history=lambda h: None, __file__="answer.py")
    asked = []

    def run_once(query, embedder, k, show, history=None, rewrite=True, cutoff=True):
        asked.append((query, history is not None))
        for key, ans in script.items():
            if key in query:
                if history is not None:
                    history.append((query, ans.text))
                return ans
        return False                    # a failed question

    mod.run_once = run_once

    def feed(lines):
        it = iter(lines)

        def read(prompt=""):
            try:
                return next(it)
            except StopIteration:
                raise EOFError from None
        return read

    results = []

    def check(name, ok):
        results.append(bool(ok))
        print(("PASS  " if ok else "FAIL  ") + name)

    # redaction
    clean, k = redact("got THM{abc_123} and hash 8743b52063cd84097a65d1633f5c74f5 and password: hunter2 ok")
    check("redact removes a flag, a long hash and a password, and counts them",
          k == 3 and "THM{" not in clean and "8743b5" not in clean and "hunter2" not in clean and clean.endswith("ok"))
    check("redact leaves ordinary text and short hex alone",
          redact("nmap -p- -sV 10.10.1.5 port 0x1f 22/tcp")[0] == "nmap -p- -sV 10.10.1.5 port 0x1f 22/tcp")

    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "logs", "t.jsonl")
        args = argparse.Namespace(room="Test Room", log=log, embedder=None, k=None, no_cutoff=False, no_history=False)
        lines = [
            "scan all ports", "w", "nmap -p- 10.10.10.10", "worked first time",
            "flag question", "x", "ffuf --magic", "no such flag, saw THM{secret_value} in my terminal",
            "/note the real flag is -mc",
            "bread question", "c", "",
            "unknown thing",                      # run_once returns False -> no answer
            "snort question", "m", "", "it is in the HackTricks IDS page",
            "/outcome", "f", "nmap -p-", "needed -Pn too",
            "/finding", "Weak login", "/login", "high", "Default credentials work (password: letmein)", "Change them",
            "/count", "/help", "/bogus", "quit",
        ]
        said = []
        rc = session(args, mod, read=feed(lines), say=lambda s="": said.append(str(s)))
        recs, bad = read_log(log)
        c = collate(recs)
        check("a session returns 0 and writes a header, 5 questions and the labels",
              rc == 0 and bad == 0 and c["header"]["chat_model"] == "test-model" and len(c["qa"]) == 5
              and c["header"]["cutoff"] == 0.62)
        check("the last label for a question wins (/outcome relabelled Q5 from 'missed' to 'needed a fix')",
              c["labels"][5]["outcome"] == "f" and c["labels"][1]["outcome"] == "w" and c["labels"][3]["outcome"] == "c")
        check("a question with no answer is logged as such and is not labelled",
              c["qa"][3]["n"] == 4 and c["qa"][3]["error"] and 4 not in c["labels"])
        raw = open(log, encoding="utf-8").read()
        check("the flag and the password never reach the log file",
              "secret_value" not in raw and "letmein" not in raw and "[flag removed]" in raw)
        check("a /note attaches to the last question that was asked",
              c["notes"].get(2) == ["the real flag is -mc"])
        check("the finding is saved with severity and fix",
              len(c["findings"]) == 1 and c["findings"][0]["severity"] == "high" and c["findings"][0]["fix"] == "Change them")
        check("history is on by default and each question is asked through run_once",
              all(h for _, h in asked) and len(asked) == 5)

        # numbering continues in a second session
        args2 = argparse.Namespace(room="Test Room", log=log, embedder=None, k=None, no_cutoff=False, no_history=True)
        session(args2, mod, read=feed(["scan again", "u", "", "quit"]), say=lambda s="": None)
        recs2, _ = read_log(log)
        check("a second session keeps the old questions and continues the numbering",
              [q["n"] for q in collate(recs2)["qa"]] == [1, 2, 3, 4, 5, 6] and collate(recs2)["sittings"] == 2)

        # damaged line is skipped
        with open(log, "a", encoding="utf-8") as f:
            f.write("{not json\n")
        recs3, bad3 = read_log(log)
        check("a damaged log line is skipped and counted", bad3 == 1 and len(recs3) == len(recs2))

        text, st = build_report(recs3, "Test Author", "Test Course", bad_lines=bad3)
        check("the report has all eleven sections and the appendix",
              all(f"## {h}" in text for h in ("1. Scope", "2. What I tested", "3. Setup", "4. Results", "5. Every",
                                              "6. What went wrong", "7. Similarity", "8. Weaknesses",
                                              "9. Compared", "10. Limits", "11. In my own words", "Appendix")))
        check("the report counts the outcomes from the labels (1 worked, 1 fix, 1 wrong, 1 correct decline, 1 not tried)",
              "| worked | I used it and it did what the answer said | 1 | 20% |" in text
              and "| wrong or made up | a flag, command or fact in it was false | 1 | 20% |" in text
              and "| not tried | I read it but did not or could not try it | 1 | 20% |" in text)
        check("the wrong answer is listed under 'Wrong or made up' with its note, and the flag is gone",
              "**Q2: flag question**" in text and "THM{" not in text and "secret_value" not in text
              and st["flags_in_report"] == 0)
        check("the report says which wrong or missed answers lack a note",
              st["missing_notes"] == [] and st["no_answer"] == 1 and st["unlabelled"] == 0)
        check("section 11 stays a visible TODO for the author", st["todo_left"] == 4 and "Test Author" in text)
        check("tables are not broken by a pipe in a question",
              cell("a | b") == "a \\| b" and "\n" not in cell("a\nb"))
        check("a fence longer than the answer's own tildes is used", fenced("x ~~~~ y").startswith("~~~~~"))

        synth = [{"type": "session", "room": "r", "time": "2026-01-01 10:00:00"},
                 {"type": "qa", "n": 1, "question": "q", "answer": "a", "seconds": 5, "top_similarity": 0.7,
                  "cited": [], "sources": [], "check_problems": [], "error": None, "time": "2026-01-01 10:01:00"},
                 {"type": "label", "n": 1, "outcome": "x", "command": "", "note": ""},
                 {"type": "qa", "n": 2, "question": "q2", "answer": "a", "seconds": 5, "top_similarity": 0.7,
                  "cited": [], "sources": [], "check_problems": [], "error": None, "time": "2026-01-01 10:02:00"}]
        _, sst = build_report(synth)
        check("a wrong answer with no note is flagged, and an unlabelled answer is counted",
              sst["missing_notes"] == [1] and sst["unlabelled"] == 1 and sst["tried"] == 1)

        empty_text, est = build_report([], "A", "C")
        check("an empty log still gives a report instead of crashing",
              "No questions were logged." in empty_text and est["questions"] == 0)

        out = os.path.join(d, "r.md")
        rc2 = report_cli(["--log", log, "--out", out, "--author", "A"])
        check("the report command writes the file", rc2 == 0 and os.path.getsize(out) > 1000)
        check("the report command reports a missing log clearly", report_cli(["--log", os.path.join(d, "none.jsonl")]) == 2)

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return failed


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
