"""Casefile practice run. Builds practice files in ~/practice_key and walks through one answer-key
block, step by step, with your real check_key.py. Nothing in your real project is touched."""
import json
import os
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
CHECKER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "projects", "PenetrationTesting", "check_key.py")
DEST = os.path.join(HOME, "practice_key")

KEY = """id: q01
question: How do I make sort print only one copy of each duplicate line?
expect: answer
source: corpus/manpages/sort.txt
evidence: output only the first of an equal run
must: -u
answer:
Use sort -u (long form --unique). It outputs only the first line of each run of equal lines, so duplicates are dropped.
"""

SLIPS = """id: q02
question: How do I make sort print only one copy of each duplicate line?
expect: answer
source: corpus/manpages/sort.txt
evidence: outputs only the first of an equal run
answer:
Use sort -u.

id: q03
question: How do I make sort print only one copy of each duplicate line?
expect: answer
source: corpus/manpages/sorts.txt
evidence: output only the first of an equal run
answer:
Use sort -u.
"""


def say(text=""):
    print(text, flush=True)


def pause():
    try:
        input("\n[Press Enter for the next step] ")
    except EOFError:
        pass
    say()


def run(cmd, shown):
    """Show a command the way a terminal does, run it in the practice folder, show what it prints."""
    say("$ " + shown)
    r = subprocess.run(cmd, cwd=DEST, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).rstrip("\n").splitlines():
        say("    " + line)
    return r


def frame(title, text):
    say("---- " + title + " ----")
    say(text.rstrip("\n"))
    say("-" * (len(title) + 10))


def build():
    os.makedirs(os.path.join(DEST, "corpus", "manpages"), exist_ok=True)
    env = dict(os.environ, LC_ALL="C")
    chunks = []
    for tool in ("sort", "ls", "cut"):
        text = subprocess.run([tool, "--help"], capture_output=True, text=True, env=env).stdout
        rel = "manpages/" + tool + ".txt"
        with open(os.path.join(DEST, "corpus", rel), "w", encoding="utf-8") as f:
            f.write(text)
        lines, start, n = text.splitlines(keepends=True), 0, 0
        while start < len(lines):
            end, size = start, 0
            while end < len(lines) and size < 1500:
                size += len(lines[end])
                end += 1
            body = "".join(lines[start:end])
            meta = {"chunk_index": n, "n_chars": len(body), "section": tool + " --help",
                    "source": "corpus/" + rel, "source_type": "manpage", "tool": tool}
            chunks.append({"id": "%s-%d" % (tool, n), "text": body, "metadata": meta})
            start, n = end, n + 1
    with open(os.path.join(DEST, "corpus", "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    shutil.copy(CHECKER, os.path.join(DEST, "check_key.py"))
    with open(os.path.join(DEST, "answer_key.txt"), "w", encoding="utf-8") as f:
        f.write(KEY)
    with open(os.path.join(DEST, "slips.txt"), "w", encoding="utf-8") as f:
        f.write(SLIPS)


def main():
    if not os.path.isfile(CHECKER):
        sys.exit("Can't find check_key.py at " + CHECKER + "\nRun: python3 ~/practice_demo.py /full/path/to/check_key.py")
    build()
    with open(os.path.join(DEST, "corpus", "manpages", "sort.txt"), encoding="utf-8") as f:
        if "output only the first of an equal run" not in f.read():
            sys.exit("Your 'sort --help' text differs from the one this practice was written for. Stop here and tell Claude.")

    say("PRACTICE RUN. Nothing here touches your real project or your real answer_key.txt.")
    say("Practice files are in " + DEST + " (the --help text of sort, ls and cut, laid out like your corpus folder).")
    say()
    say("Three kinds of text will appear:")
    say("  $ lines        what you TYPE at the terminal (leave out the $)")
    say("      indented   what the screen PRINTS back; you never type it")
    say("  framed text    what goes INSIDE answer_key.txt (typed in nano or Notepad, never at the terminal)")
    pause()

    say("STEP 1 of 4: find the passage with grep (terminal)")
    say()
    r = run(["grep", "-rniI", "unique", "corpus/manpages"], 'grep -rniI "unique" corpus/manpages')
    hit = next((x for x in r.stdout.splitlines() if x.startswith("corpus/manpages/sort.txt:")), None)
    say()
    say("Each line is   path : line number : text of that line.")
    if hit:
        n = int(hit.split(":")[1])
        say("The text stops mid-sentence, so read around it:")
        say()
        run(["sed", "-n", "%d,%dp" % (n, n + 2), "corpus/manpages/sort.txt"],
            "sed -n '%d,%dp' corpus/manpages/sort.txt" % (n, n + 2))
    pause()

    say("STEP 2 of 4: write the block (inside answer_key.txt)")
    say()
    frame("inside answer_key.txt", KEY)
    say()
    say("source:    the path from the grep line (everything before the first :number:)")
    say("evidence:  a short stretch of that line's own words, copied exactly")
    say("must:      text a correct answer has to contain")
    say("answer:    your expected answer, in your own words")
    pause()

    say("STEP 3 of 4: check it (terminal)")
    say()
    run(["python3", "check_key.py"], "python3 check_key.py")
    say()
    say("src 1/1  ev 1/1  must 1/1  means: the file was found, the evidence phrase was found in it,")
    say("and the must: text was found in it.")
    pause()

    say("STEP 4 of 4: two typical slips")
    say()
    say("In q02 the evidence has one extra letter (outputs instead of output).")
    say("In q03 the file name has one extra letter (sorts.txt instead of sort.txt).")
    say()
    run(["python3", "check_key.py", "slips.txt"], "python3 check_key.py slips.txt")
    say()
    say("WARN = fix it, but the check still ran. ERROR = must be fixed. Each message names the line to fix.")
    say()
    say("That is the whole loop: grep to find, copy a path and a phrase into the block, run the check.")
    say("To play: cd ~/practice_key, change a word in answer_key.txt (notepad.exe answer_key.txt),")
    say("then run python3 check_key.py again.")


main()
