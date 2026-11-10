#!/usr/bin/env python3
"""
patch_answer_w8b.py -- answer.py: ask the model the standalone question, not the typed follow-up

Found in the Week 8 A/B test (SSH remote port forwarding, 3 runs each way): on a follow-up the chat
searched with the rewritten question ("What is the syntax for SSH remote port forwarding?") but still
asked the model the words as typed ("And the remote version?") and showed it the earlier answer.
The model then copied from that answer: it swapped -R into the earlier -L command template (adding
-N -f, which no passage has), and once it began by repeating the earlier answer word for word. The same
question typed on its own was answered cleanly 3 times out of 3, with the same five passages.

This patch makes a rewritten follow-up reach the model exactly as if it had been typed on its own:

  * the "Question:" line of the prompt is the standalone question, and
  * the earlier turns are no longer sent to the answering call (they still help write the rewrite).
    With --no-rewrite they are still sent, because then they are the only context the model has.

A new setting, ANSWER_SEES_HISTORY, near the top of answer.py switches the old behaviour back on
(standalone question plus earlier turns) if you want to compare.

Run once from the project root (next to answer.py, after patch_answer_w8.py):

    python patch_answer_w8b.py

It saves answer.py as answer.py.w8.bak. Every edit must find its text exactly once; if one does not,
nothing is written. The patched file is compiled before it is saved.
"""
import os
import shutil
import sys

PATH = "answer.py"

# (name, text to find, text to put there). Blank lines at the start and end of each text are ignored.
EDITS = [
    ("docstring: how a follow-up reaches the model", r'''
Week 8 (conversation.py): the interactive chat remembers the last few turns. A follow-up such
as "what about UDP?" is first rewritten into a standalone question for the search, and the
earlier turns travel with the new question to the model. '/reset' forgets, '/history' shows
what is remembered. A single question (-q) is still asked on its own, exactly as in Week 7.
''', r'''
Week 8 (conversation.py): the interactive chat remembers the last few turns. A follow-up such
as "what about UDP?" is first rewritten into a standalone question, using the earlier turns to
work out what it means. The search and the model are then given that question, as if it had
been typed on its own (ANSWER_SEES_HISTORY, below, says why the model is not also shown the
earlier turns). '/reset' forgets, '/history' shows what is remembered. A single question (-q)
is still asked on its own, exactly as in Week 7.
'''),

    ("setting: ANSWER_SEES_HISTORY", r'''
HISTORY_TURNS = 3      # earlier question/answer pairs sent with a new question (0 = no memory)
HISTORY_CHARS = 4000   # ...and never more than about this many characters of them
''', r'''
HISTORY_TURNS = 3      # earlier question/answer pairs the chat remembers (0 = no memory)
HISTORY_CHARS = 4000   # ...and never more than about this many characters of them

# Week 8, after the first real test: once a follow-up has been rewritten into a standalone
# question, should the model that writes the answer also be shown the earlier turns?
#   False (default)  No. It is asked the standalone question and gets the same prompt as a
#                    question typed on its own. In a real test it copied from the earlier
#                    answer (a command template, once a whole sentence) when it could see it.
#   True             Yes, as in the first Week 8 version.
# With --no-rewrite the earlier turns are always shown: then they are the only way the model
# can tell what the follow-up means.
ANSWER_SEES_HISTORY = False
'''),

    ("ask(): docstring for rewrite", r'''
    rewrite        with history, first turn the follow-up into a standalone question for the
                   search (False searches with the words as typed).
''', r'''
    rewrite        with history, first turn the follow-up into a standalone question. The search
                   and the model are both given that question (False: both get the words as
                   typed, and the model is shown the earlier turns instead).
'''),

    ("ask(): what the model is asked and shown", r'''
    search_query = rewrite_query(query, earlier) if earlier and rewrite else query
''', r'''
    search_query = rewrite_query(query, earlier) if earlier and rewrite else query
    # The model is asked the question the search used, so a follow-up reaches it already
    # standalone. It is shown the earlier turns only when nothing was rewritten (--no-rewrite)
    # or when ANSWER_SEES_HISTORY is switched on.
    asked = search_query
    shown = history if ANSWER_SEES_HISTORY or not (earlier and rewrite) else None
'''),

    ("ask(): the call to the model", r'''
            messages=build_messages(query, hits, history),
''', r'''
            messages=build_messages(asked, hits, shown),
'''),
]


def main():
    if not os.path.isfile(PATH):
        sys.exit(f"{PATH} not found. Run this from the project root (the folder that holds answer.py).")
    with open(PATH, "rb") as f:
        raw = f.read().decode("utf-8")
    crlf = "\r\n" in raw
    src = raw.replace("\r\n", "\n")
    if all(new.strip("\n") in src for _, _, new in EDITS):
        sys.exit("answer.py already has this change. Nothing to do.")

    spots = []                                   # (where it starts, how long it is, replacement, name)
    for name, old, new in EDITS:
        old, new = old.strip("\n"), new.strip("\n")
        n = src.count(old)
        if n != 1:
            sys.exit(f"Edit '{name}': its text was found {n} times in {PATH}, not once.\n"
                     "answer.py is not the Week 8 version this patch was written for (the one that "
                     "patch_answer_w8.py makes), so nothing was written.")
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

    shutil.copy2(PATH, PATH + ".w8.bak")
    with open(PATH, "wb") as f:
        f.write((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
    print(f"Patched {PATH} with {len(EDITS)} edits. The earlier Week 8 version is saved as {PATH}.w8.bak.")


if __name__ == "__main__":
    main()
