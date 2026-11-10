#!/usr/bin/env python3
"""
patch_answer_w8.py -- Casefile Week 8: give answer.py a memory

Run once from the project root (next to answer.py and conversation.py):

    python patch_answer_w8.py

It saves answer.py as answer.py.w7.bak, then makes the edits listed in EDITS below. Every edit
must find its text exactly once in answer.py. If one does not, nothing is written. The patched
file is also compiled before it is saved, so a broken answer.py is never left behind.
"""
import os
import shutil
import sys

PATH = "answer.py"

# (name, text to find, text to put there). Blank lines at the start and end of each text are ignored.
EDITS = [
    ("docstring: title", r'''
answer.py -- Casefile Weeks 6-7: "Get a real answer out" + "Make it prove its work"
''', r'''
answer.py -- Casefile Weeks 6-8: "Get a real answer out", "Make it prove its work", "Remember the conversation"
'''),

    ("docstring: what Week 8 adds", r'''
Later weeks can import it:   from answer import ask
(Week 8 extends build_messages() with history; Week 11 calls ask() in a loop
over the answer key.)
''', r'''
Week 8 (conversation.py): the interactive chat remembers the last few turns. A follow-up such
as "what about UDP?" is first rewritten into a standalone question for the search, and the
earlier turns travel with the new question to the model. '/reset' forgets, '/history' shows
what is remembered. A single question (-q) is still asked on its own, exactly as in Week 7.

Later weeks can import it:   from answer import ask
(Week 11 calls ask() in a loop over the answer key, without history, so every question is
asked on its own.)
'''),

    ("import time", r'''
import sys
from dataclasses import dataclass, field
''', r'''
import sys
import time
from dataclasses import dataclass, field
'''),

    ("import conversation", r'''
try:
    import citecheck  # Week 7: checks the [n] citations in an answer
except Exception as e:  # noqa: BLE001
    sys.exit(f"Could not import citecheck.py ({e}). It must sit next to answer.py.")
''', r'''
try:
    import citecheck  # Week 7: checks the [n] citations in an answer
except Exception as e:  # noqa: BLE001
    sys.exit(f"Could not import citecheck.py ({e}). It must sit next to answer.py.")

try:
    import conversation  # Week 8: what the chat remembers, and the follow-up rewrite
except Exception as e:  # noqa: BLE001
    sys.exit(f"Could not import conversation.py ({e}). It must sit next to answer.py.")
'''),

    ("settings: how much is remembered", r'''
# Keep the model loaded between questions (Ollama unloads it after 5 minutes).
KEEP_ALIVE = "30m"
''', r'''
# Keep the model loaded between questions (Ollama unloads it after 5 minutes).
KEEP_ALIVE = "30m"

# Week 8: how much of the conversation the chat remembers. The oldest turns are dropped first.
HISTORY_TURNS = 3      # earlier question/answer pairs sent with a new question (0 = no memory)
HISTORY_CHARS = 4000   # ...and never more than about this many characters of them
'''),

    ("prompt: the history rule", r'''
- Be concise and practical; the student is at a terminal."""
''', r'''
- Be concise and practical; the student is at a terminal."""

# Week 8: added to the rules only when earlier turns are sent along. A question asked on its own
# gets exactly the Week 7 prompt, so Week 11's grading runs are not affected by this.
HISTORY_RULE = """
- Earlier turns of this conversation are there only so you can tell what a short follow-up \
(for example "what about UDP?") refers to. Take no facts or commands from them: your only \
source is the numbered passages sent with the latest question, and those numbers start \
again at [1] for every question."""
'''),

    ("build_messages: history", r'''
def build_messages(query, hits):
    """The whole Week 6 idea: system rules + numbered chunks + the question.
    (Week 8 will insert conversation history here.)"""
    user = (f"Context passages:\n\n{build_context(hits)}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the passages above, and cite them like [1].")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
''', r'''
def build_messages(query, hits, history=None):
    """The whole Week 6 idea: system rules + numbered chunks + the question.
    Week 8: `history` is a list of earlier (question, answer) pairs. They go between the system
    rules and the new question as plain chat turns, with no old passages and no old [n]. With no
    history the messages are exactly the Week 7 ones."""
    user = (f"Context passages:\n\n{build_context(hits)}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the passages above, and cite them like [1].")
    earlier = conversation.history_messages(history, HISTORY_TURNS, HISTORY_CHARS)
    system = SYSTEM_PROMPT + (HISTORY_RULE if earlier else "")
    return ([{"role": "system", "content": system}] + earlier
            + [{"role": "user", "content": user}])
'''),

    ("Answer: search_query", r'''
    truncated: bool = False
    check: object = None  # Week 7: a citecheck.Check, filled in by ask()
''', r'''
    truncated: bool = False
    check: object = None  # Week 7: a citecheck.Check, filled in by ask()
    search_query: str = ""  # Week 8: what the search ran with (a follow-up made standalone)
'''),

    ("ask: rewrite step and history", r'''
def ask(query, embedder=None, k=None, on_hits=None, on_token=None):
    """Retrieve chunks, build the prompt, ask the chat model, return an Answer.

    on_hits(hits)  is called once retrieval is done, before generation starts.
    on_token(text) is called with each streamed piece of the answer.
    Leave embedder / k as None to use retrieve.py's own defaults.
    """
    hits = list(call_retrieve(query, embedder, k) or [])
    if not hits:
        return Answer(text="")
''', r'''
def rewrite_query(query, earlier):
    """Week 8: turn a follow-up such as "what about UDP?" into a standalone question for the
    search. `earlier` is conversation.recent()'s list. One short extra call to the chat model; if
    anything goes wrong, or the reply does not look like a question, the words as typed are used."""
    try:
        reply = ollama.chat(
            model=CHAT_MODEL,
            messages=conversation.rewrite_messages(query, earlier),
            options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": 100},
            keep_alive=KEEP_ALIVE,
        )
    except Exception:  # noqa: BLE001  (a dead Ollama is reported by the answer call that follows)
        return query
    return conversation.clean_rewrite(_get(_get(reply, "message"), "content", ""), query)


def ask(query, embedder=None, k=None, on_hits=None, on_token=None,
        history=None, rewrite=True, on_search=None):
    """Retrieve chunks, build the prompt, ask the chat model, return an Answer.

    on_hits(hits)  is called once retrieval is done, before generation starts.
    on_token(text) is called with each streamed piece of the answer.
    Leave embedder / k as None to use retrieve.py's own defaults.

    Week 8:
    history        earlier (question, answer) pairs, oldest first. ask() only reads it; the caller
                   adds the new pair. None (the default) asks the question on its own, which is
                   how Week 11 should call it.
    rewrite        with history, first turn the follow-up into a standalone question for the
                   search (False searches with the words as typed).
    on_search(q)   is called with the question the search will use, before retrieval starts.
    """
    earlier = conversation.recent(history, HISTORY_TURNS, HISTORY_CHARS)
    search_query = rewrite_query(query, earlier) if earlier and rewrite else query
    if on_search:
        on_search(search_query)
    hits = list(call_retrieve(search_query, embedder, k) or [])
    if not hits:
        return Answer(text="", search_query=search_query)
'''),

    ("ask: history goes into the prompt", r'''
            messages=build_messages(query, hits),
''', r'''
            messages=build_messages(query, hits, history),
'''),

    ("ask: report the search query", r'''
    return Answer(text, hits, prompt_tokens, answer_tokens, truncated, check)
''', r'''
    return Answer(text, hits, prompt_tokens, answer_tokens, truncated, check, search_query)
'''),

    ("run_once: head, /history helper", r'''
def run_once(query, embedder, k, show_chunks):
    """Ask one question and print everything. Returns True if an answer was printed."""

    def show_hits(hits):
        print(f"\nSearching {index_name(embedder)}: {len(hits)} chunk(s)")
''', r'''
def show_history(history):
    """'/history': what the chat remembers, and how much of it the next question will carry."""
    if not history:
        print("(nothing remembered yet)")
        return
    sent = conversation.recent(history, HISTORY_TURNS, HISTORY_CHARS)
    print(f"{len(history)} turn(s) so far; the next question carries the last {len(sent)}.")
    for i, (q, a) in enumerate(history, 1):
        print(f"  {i}. {clip(q, 60)}  ->  {clip(a, 60)}")


def run_once(query, embedder, k, show_chunks, history=None, rewrite=True):
    """Ask one question and print everything. Returns the Answer if one was printed, else False.

    Week 8: with a `history` list the question is asked in context, and once it has been
    answered, (query, answer) is added to that list. history=None asks it on its own."""
    started = time.time()
    earlier = conversation.recent(history, HISTORY_TURNS, HISTORY_CHARS)
    searched = [query]    # what the search ran with; ask() reports it through on_search

    def note_search(search_query):
        searched[0] = search_query

    def show_hits(hits):
        print(f"\nSearching {index_name(embedder)}: {len(hits)} chunk(s)")
        if searched[0] != query:
            print(f'  searched for: "{searched[0]}"   (your follow-up, made standalone)')
'''),

    ("run_once: ask with history", r'''
    try:
        ans = ask(query, embedder, k, on_hits=show_hits,
                  on_token=lambda t: print(t, end="", flush=True))
''', r'''
    if earlier and rewrite:
        print("\nReading your follow-up against the conversation (one short extra model call)...",
              flush=True)
    try:
        ans = ask(query, embedder, k, on_hits=show_hits,
                  on_token=lambda t: print(t, end="", flush=True),
                  history=history, rewrite=rewrite, on_search=note_search)
'''),

    ("run_once: time and memory in the stats line", r'''
    if stats:
        print(f"[{', '.join(stats)}]")
''', r'''
    stats.append(f"{time.time() - started:.0f}s")
    if earlier:
        stats.append(f"{len(earlier)} earlier turn(s) remembered")
    if stats:
        print(f"[{', '.join(stats)}]")
'''),

    ("run_once: remember the turn", r'''
    print_verification(ans)
    print()
    return True
''', r'''
    print_verification(ans)
    print()
    if history is not None and ans.text.strip():
        history.append((query, ans.text))    # remembered for the next question
    return ans
'''),

    ("main: head", r'''
def main():
    ap = argparse.ArgumentParser(
        description="Casefile Week 6 -- grounded answers from your local RAG index.")
''', r'''
def main():
    global HISTORY_TURNS
    ap = argparse.ArgumentParser(
        description="Casefile -- grounded answers from your local RAG index (Weeks 6-8).")
'''),

    ("main: --turns and --no-rewrite", r'''
    ap.add_argument("--show-chunks", action="store_true",
                    help="also print the full chunk text sent to the model")
    args = ap.parse_args()
''', r'''
    ap.add_argument("--show-chunks", action="store_true",
                    help="also print the full chunk text sent to the model")
    ap.add_argument("--turns", type=int,
                    help=f"earlier turns the chat remembers (default {HISTORY_TURNS}; 0 = none)")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="search a follow-up with the words as typed (skip the rewrite step)")
    args = ap.parse_args()
    if args.turns is not None:
        HISTORY_TURNS = max(0, args.turns)
'''),

    ("main: the chat loop with memory", r'''
    print(f"Casefile -- searching {index_name(args.embedder)}. "
          "Ask a pentesting question; 'quit' or Ctrl-D to exit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            run_once(q, args.embedder, args.k, args.show_chunks)
''', r'''
    print(f"Casefile -- searching {index_name(args.embedder)}. Ask a pentesting question; "
          "follow-ups are remembered.\n"
          "'/reset' forgets the conversation, '/history' shows it, 'quit' or Ctrl-D exits.")
    history = []    # Week 8: (question, answer) pairs, oldest first
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not sys.stdin.isatty():
            print(q)    # input() does not echo questions that are piped in from a file, so show them
        if q.lower() in ("quit", "exit", "q"):
            break
        if q.lower() in ("/reset", "/clear"):
            history.clear()
            print("(conversation forgotten)")
        elif q.lower() == "/history":
            show_history(history)
        elif q and not q.startswith("#"):
            run_once(q, args.embedder, args.k, args.show_chunks, history,
                     rewrite=not args.no_rewrite)
'''),
]


def main():
    if not os.path.isfile(PATH):
        sys.exit(f"{PATH} not found. Run this from the project root (the folder that holds answer.py).")
    with open(PATH, "rb") as f:
        raw = f.read().decode("utf-8")
    crlf = "\r\n" in raw
    src = raw.replace("\r\n", "\n")
    if "import conversation" in src:
        sys.exit("answer.py already has the Week 8 changes (it imports conversation). Nothing to do.")

    spots = []                                   # (where it starts, how long it is, replacement, name)
    for name, old, new in EDITS:
        old, new = old.strip("\n"), new.strip("\n")
        n = src.count(old)
        if n != 1:
            sys.exit(f"Edit '{name}': its text was found {n} times in {PATH}, not once.\n"
                     "answer.py differs from the Week 7 version this patch was written for, so nothing "
                     "was written. Put the Week 7 answer.py back and run this again.")
        spots.append((src.index(old), len(old), new, name))
    spots.sort()
    for (a, length, _, name_a), (b, _, _, name_b) in zip(spots, spots[1:]):
        if a + length > b:
            sys.exit(f"Edits '{name_a}' and '{name_b}' overlap, so nothing was written.")
    for start, length, new, _ in reversed(spots):   # from the bottom up, so the earlier positions stay valid
        src = src[:start] + new + src[start + length:]

    try:
        compile(src, PATH, "exec")
    except SyntaxError as e:
        sys.exit(f"The patched file would not compile ({e}), so nothing was written.")

    shutil.copy2(PATH, PATH + ".w7.bak")
    with open(PATH, "wb") as f:
        f.write((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
    print(f"Patched {PATH} with {len(EDITS)} edits. The Week 7 version is saved as {PATH}.w7.bak.")


if __name__ == "__main__":
    main()
