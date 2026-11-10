#!/usr/bin/env python3
"""
conversation.py -- Casefile Week 8: "Remember the conversation"

The memory side of the chat, in plain Python (no Ollama, no Chroma), so it runs instantly and
can be tested on its own:    python conversation.py --selftest

A conversation is a list of (question, answer) pairs, oldest first. answer.py keeps that list and
adds to it after every answer; this file decides which part of it travels with the next question.

    recent(history)                    the latest turns that fit the budget, answers without [n]
    history_messages(history)          those turns as chat messages for the model
    rewrite_messages(question, turns)  the prompt that turns "what about UDP?" into a standalone question
    clean_rewrite(reply, question)     the model's rewrite, or the question as typed if the reply looks wrong

Choices worth knowing about
  * Remembered answers lose their [n] markers. Passage numbers start again at [1] for every
    question, so an old [2] would point at the wrong passage of the new question.
  * Old passages are never sent again, only the questions and answers. Facts must come from the
    passages retrieved for the newest question; the earlier turns only say what a follow-up means.
  * Only the newest turns are kept (oldest dropped first), so the prompt stays well inside the
    context window.
"""
from __future__ import annotations

import re
import sys


# --- the rewrite prompt --------------------------------------------------------

# The examples are made-up everyday topics on purpose: none of them is a question you might
# later put in the answer key.
REWRITE_SYSTEM = """You prepare a student's question for a search engine. The student is in \
the middle of a conversation and has just typed a follow-up that may not make sense on its \
own. Rewrite the follow-up as ONE standalone question: replace words such as "it", "that", \
"both", "the other one" or "what about X" with the tool, option or topic they refer to.

Rules:
- Keep the student's meaning. Do not answer the question and do not add anything new.
- Keep every command, flag and name exactly as written (for example -sV or --script).
- If the follow-up already makes sense by itself, or starts a new topic, repeat it unchanged.
- Reply with the question only, on one line.

Examples (made-up topics, only to show the pattern):

Conversation so far:
Student: How do I boil an egg?
Assistant: Lower it into boiling water for 7 minutes.

Follow-up: What about a soft one?

Standalone question: How do I boil a soft egg?

Conversation so far:
Student: How do I change a flat tire?
Assistant: Loosen the nuts, jack up the car and swap the wheel.

Follow-up: How long does it take?

Standalone question: How long does it take to change a flat tire?

Conversation so far:
Student: How do I change a flat tire?
Assistant: Loosen the nuts, jack up the car and swap the wheel.

Follow-up: How do I tune a guitar?

Standalone question: How do I tune a guitar?"""


# --- citations -----------------------------------------------------------------

_N = r"\[\d{1,3}(?:\s*[,;]\s*\d{1,3})*\]"          # [2]  [2, 4]: the shape citecheck.py reads
_LEAD = re.compile(                                  # "According to [2], ..." at the start of a sentence
    r"(^|[.!?]\s+)(?:according to|as (?:stated|described|noted|shown|mentioned) in)\s*"
    + _N + r"(?:\s*(?:,|and)?\s*" + _N + r")*\s*,?\s*([a-z]?)",
    re.I | re.M,
)
_MARK = re.compile(r"[ \t]*" + _N)                  # a marker and the space before it


def _tidy(prose):
    prose = _LEAD.sub(lambda m: m.group(1) + m.group(2).upper(), prose)
    return _MARK.sub("", prose)


def strip_citations(text):
    """`text` without its [n] markers. Code is left alone: a [1] inside ``` or `backticks` is code."""
    parts = str(text or "").split("```")
    for i in range(0, len(parts), 2):                         # even parts are prose, odd parts are code blocks
        pieces = re.split(r"(`[^`\n]*`)", parts[i])           # odd pieces are `inline code`
        parts[i] = "".join(p if j % 2 else _tidy(p) for j, p in enumerate(pieces))
    return "```".join(parts)


# --- what travels with the next question ---------------------------------------

def clip(text, n):
    """One line, at most n characters."""
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def recent(history, max_turns=3, max_chars=4000):
    """The latest (question, answer) pairs that fit, oldest first, each answer without its [n] markers.

    At most `max_turns` pairs and about `max_chars` characters; older turns are dropped first.
    The newest turn always goes in (cut down to max_chars if it is longer than that). A turn
    whose answer is empty is skipped."""
    if max_turns <= 0:
        return []
    kept, used = [], 0
    for question, answer in reversed(list(history or [])[-max_turns:]):
        question, answer = str(question).strip(), strip_citations(answer).strip()
        if not answer:
            continue
        if not kept:
            if len(answer) > max_chars:
                answer = answer[:max_chars].rstrip() + " ..."
        elif used + len(question) + len(answer) > max_chars:
            break
        kept.append((question, answer))
        used += len(question) + len(answer)
    return kept[::-1]


def history_messages(history, max_turns=3, max_chars=4000):
    """The remembered turns as chat messages: user question, assistant answer, user question, ..."""
    messages = []
    for question, answer in recent(history, max_turns, max_chars):
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return messages


# --- turning a follow-up into a standalone question ----------------------------

_LABEL = re.compile(r"^(?:(?:the\s+)?(?:standalone|rewritten|reworded)(?:\s+question)?|question)\s*:\s*", re.I)
_FLAG = re.compile(r"(?<![\w-])--?[A-Za-z0-9][A-Za-z0-9-]*")      # -sV  --script  -p-  (same as citecheck.py)


def rewrite_messages(question, turns, max_turns=2, answer_chars=300):
    """The prompt for the rewrite step. `turns` is recent()'s output; only its last `max_turns`
    are shown, and each answer is shortened, so this extra model call stays small."""
    shown = []
    for q, a in (turns[-max_turns:] if max_turns > 0 else []):
        shown += [f"Student: {clip(q, 300)}", f"Assistant: {clip(a, answer_chars)}"]
    user = ("Conversation so far:\n" + "\n".join(shown)
            + f"\n\nFollow-up: {question}\n\nStandalone question:")
    return [{"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": user}]


def clean_rewrite(reply, question, max_chars=300):
    """The rewritten question from the model's reply, or `question` as typed when the reply does
    not look like a rewrite: empty, far too long (probably an answer), or missing a flag the
    student typed (a search for -sV must not turn into a search for 'the version option')."""
    lines = [ln.strip() for ln in str(reply or "").splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.endswith(":")] or lines     # skip a lead-in such as "Here is the question:"
    text = _LABEL.sub("", lines[0]).strip().strip("\"'`").strip() if lines else ""
    if (not text or len(text) > max_chars or len(text) > 3 * len(question) + 80
            or any(f not in text for f in _FLAG.findall(question))):
        return question
    return text


# --- self-test: python conversation.py --selftest ------------------------------

def _selftest():
    turns = [(f"question {i}", f"answer {i} " + "x" * 90) for i in range(1, 6)]   # each pair is 109 characters
    cases = [
        ("a [n] marker and the space before it are removed",
         lambda: strip_citations("Use `-p-` [2].") == "Use `-p-`."),
        ("several markers in a row, and a [2, 4] pair, are removed",
         lambda: strip_citations("Both agree [1][3] and also [2, 4].") == "Both agree and also."),
        ("'According to [2],' opening a sentence is dropped and the next word capitalised",
         lambda: strip_citations("According to [2], you can use `-p-`. As noted in [1] and [3], it is fast.")
                 == "You can use `-p-`. It is fast."),
        ("code is left alone: a [1] inside a code block or `backticks` stays",
         lambda: strip_citations("Run:\n```bash\nls a[1] [2]\n```\nThen `arr[1]` [3].")
                 == "Run:\n```bash\nls a[1] [2]\n```\nThen `arr[1]`."),
        ("recent() keeps the newest turns, oldest first",
         lambda: [q for q, _ in recent(turns, max_turns=3, max_chars=10 ** 6)]
                 == ["question 3", "question 4", "question 5"]),
        ("recent() drops the oldest turns first when the character budget is full",
         lambda: [q for q, _ in recent(turns, max_turns=5, max_chars=250)] == ["question 4", "question 5"]),
        ("the newest turn is always kept, cut down if it alone is over the budget",
         lambda: len(recent([("q", "y" * 500)], max_turns=3, max_chars=100)[0][1]) <= 110),
        ("zero turns of memory means nothing is sent",
         lambda: recent(turns, max_turns=0) == [] and history_messages(None) == [] and history_messages([]) == []),
        ("a turn with an empty answer is skipped",
         lambda: recent([("q1", "a1"), ("q2", " [1] ")], max_turns=3) == [("q1", "a1")]),
        ("history_messages() gives user/assistant pairs without [n] markers",
         lambda: history_messages([("What does -p- do?", "It scans all ports [2].")])
                 == [{"role": "user", "content": "What does -p- do?"},
                     {"role": "assistant", "content": "It scans all ports."}]),
        ("the rewrite prompt shows the follow-up and shortens an old answer",
         lambda: (lambda m: m[0]["role"] == "system"
                  and "Follow-up: What about UDP?" in m[1]["content"]
                  and "Student: Which nmap option scans every TCP port?" in m[1]["content"]
                  and len(m[1]["content"]) < 700)(
             rewrite_messages("What about UDP?", [("Which nmap option scans every TCP port?", "word " * 200)]))),
        ("clean_rewrite() strips a label and quotes",
         lambda: clean_rewrite('Standalone question: "Which nmap option scans UDP ports?"', "What about UDP?")
                 == "Which nmap option scans UDP ports?"),
        ("clean_rewrite() skips a lead-in line that ends with a colon",
         lambda: clean_rewrite("Here is the rewritten question:\nWhich nmap option scans UDP ports?", "What about UDP?")
                 == "Which nmap option scans UDP ports?"),
        ("clean_rewrite() falls back to the question as typed when the reply is empty or far too long",
         lambda: clean_rewrite("", "What about UDP?") == "What about UDP?"
                 and clean_rewrite("  \n ", "What about UDP?") == "What about UDP?"
                 and clean_rewrite("x" * 400, "What about UDP?") == "What about UDP?"),
        ("clean_rewrite() refuses a rewrite that lost a flag the student typed",
         lambda: clean_rewrite("What does the version option do?", "What does -sV do?") == "What does -sV do?"
                 and clean_rewrite("What does the nmap -sV option do?", "What does -sV do?")
                 == "What does the nmap -sV option do?"),
        ("clean_rewrite() leaves a question that already stands alone as it was",
         lambda: clean_rewrite("How do I make sourdough bread?", "How do I make sourdough bread?")
                 == "How do I make sourdough bread?"),
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


if __name__ == "__main__":
    if len(sys.argv) == 1 or "--selftest" in sys.argv[1:]:
        sys.exit(1 if _selftest() else 0)
    print(__doc__)
