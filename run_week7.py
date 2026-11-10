#!/usr/bin/env python3
"""
run_week7.py -- Casefile Week 7: run the regression questions through answer.py

Asks the three Week 7 regression questions with  python answer.py -q "..." --show-chunks
(about 3-4 minutes each on CPU), shows the output as it arrives, and saves it to
week7_answers.txt in the same layout as week6_answers.txt, so the two can be compared.
A short digest of each citation check is printed and saved at the end.

    python run_week7.py          all three questions
    python run_week7.py 3        only question 3
    python run_week7.py 1 2      questions 1 and 2 (saved to week7_answers_q1_2.txt)

Run it from the project root with the venv active, after patch_answer_w7.py has been run.

What to look for
  1  Week 6 said "65,335" and cited [2]. If [2] does not contain that number, the citation
     check should say so and name the passage that does.
  2  Week 6 cited only [3] and was right to. A clean check is expected.
  3  The off-topic control: a decline with no citations, and a NOTE saying none are expected.
     (week6_answers.txt has no control run, so the wording of this question is new.)
"""
import codecs
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

QUESTIONS = [
    "Which nmap option scans every TCP port?",
    "What does the ffuf -mc flag do?",
    "How do I make sourdough bread?",
]
RULE = "=" * 20


def clock():
    return time.strftime("%H:%M:%S")


def mins(seconds):
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def emit(text, *streams):
    for s in streams:
        s.write(text)
        s.flush()


def run_one(question, sink):
    """Run answer.py on one question: show its output as it arrives, copy it to `sink`, return it."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "answer.py", "-q", question, "--show-chunks"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    decode = codecs.getincrementaldecoder("utf-8")(errors="replace").decode
    parts = []
    try:
        while True:
            data = proc.stdout.read1(1024)  # whatever has arrived, so the answer appears as it is written
            if not data:
                break
            text = decode(data)
            parts.append(text)
            emit(text, sys.stdout, sink)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
    if code:
        note = f"\n[answer.py stopped with exit code {code}]\n"
        parts.append(note)
        emit(note, sys.stdout, sink)
    return "".join(parts)


def digest(output):
    """The verification block of one run (what was cited, what the check found), trimmed to the essentials."""
    lines = output.splitlines()
    rules = [i for i, ln in enumerate(lines) if re.fullmatch(r"-{20,}", ln.strip())]
    if not rules:
        return ["   (no verification block in the output; answer.py may have stopped early, see above)"]
    keep = []
    for ln in lines[rules[-1] + 1:]:
        s = ln.strip()
        if s and not s.startswith(("file ", "claim ", "passage ", "same text")):
            keep.append("   " + ln.rstrip())
    return keep


def summary(results):
    out = [f"{RULE} SUMMARY: citation checks", ""]
    for n, question, seconds, output in results:
        out.append(f"{n}. {question}   ({mins(seconds)})")
        out.extend(digest(output))
        out.append("")
    return "\n".join(out) + "\n"


def main(argv):
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    picks = []
    for a in argv:
        if not a.isdigit() or not 1 <= int(a) <= len(QUESTIONS):
            sys.exit(f"Unknown question '{a}'. Use a number from 1 to {len(QUESTIONS)}, or nothing to run all of them.")
        if int(a) not in picks:
            picks.append(int(a))
    everything = not picks or set(picks) == set(range(1, len(QUESTIONS) + 1))
    picks = sorted(picks) if picks else list(range(1, len(QUESTIONS) + 1))
    out_name = "week7_answers.txt" if everything else "week7_answers_q" + "_".join(map(str, picks)) + ".txt"

    answer_path = os.path.join(HERE, "answer.py")
    try:
        with open(answer_path, encoding="utf-8") as f:
            patched = "citecheck" in f.read()
    except OSError as e:
        sys.exit(f"Cannot read {answer_path} ({e}). Put run_week7.py in the project root, next to answer.py.")
    if not patched:
        sys.exit("answer.py has no citation check yet. Run  python patch_answer_w7.py  first.")
    if not os.path.isfile(os.path.join(HERE, "citecheck.py")):
        sys.exit("citecheck.py is missing from the project root.")

    results, stopped = [], False
    with open(os.path.join(HERE, out_name), "w", encoding="utf-8") as sink:
        emit(f"started {clock()}\n\n", sink)
        try:
            for n in picks:
                question = QUESTIONS[n - 1]
                emit(f"{RULE} {question}\n", sys.stdout, sink)
                t0 = time.time()
                output = run_one(question, sink)
                results.append((n, question, time.time() - t0, output))
                emit("\n", sys.stdout, sink)
        except KeyboardInterrupt:
            stopped = True
            emit("\n[stopped early with Ctrl-C]\n\n", sys.stdout, sink)
        if results:
            emit(summary(results), sys.stdout, sink)
        emit(f"finished {clock()}\n", sys.stdout, sink)

    print(f"Saved to {os.path.join(HERE, out_name)}")
    return 130 if stopped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
