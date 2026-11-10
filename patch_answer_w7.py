#!/usr/bin/env python3
"""
patch_answer_w7.py -- adds the Week 7 citation check to answer.py

Run it once, from the project root (next to answer.py and citecheck.py):

    python patch_answer_w7.py

It swaps a few exact snippets in answer.py for new ones. If any snippet is not found exactly
once, it stops and changes nothing. To undo it:   git checkout answer.py
"""
import ast
import sys

PATH = "answer.py"

EDITS = [
    ("docstring title",
     r'''answer.py -- Casefile Week 6: "Get a real answer out"''',
     r'''answer.py -- Casefile Weeks 6-7: "Get a real answer out" + "Make it prove its work"'''),

    ("docstring note",
     r'''Later weeks can import it:   from answer import ask''',
     r'''Week 7 (citecheck.py): under every answer it prints only the passages the answer actually
cited, with the file each came from, and flags citations that do not check out.

Later weeks can import it:   from answer import ask'''),

    ("import citecheck",
     r'''Try: pip install ollama")''',
     r'''Try: pip install ollama")

try:
    import citecheck  # Week 7: checks the [n] citations in an answer
except Exception as e:  # noqa: BLE001
    sys.exit(f"Could not import citecheck.py ({e}). It must sit next to answer.py.")'''),

    ("Answer.check field",
     r'''    truncated: bool = False''',
     r'''    truncated: bool = False
    check: object = None  # Week 7: a citecheck.Check, filled in by ask()'''),

    ("ask() runs the check",
     r'''    return Answer("".join(pieces), hits, prompt_tokens, answer_tokens, truncated)''',
     r'''    text = "".join(pieces)
    check = citecheck.check(text, [hit_text(h) for h in hits])
    return Answer(text, hits, prompt_tokens, answer_tokens, truncated, check)'''),

    ("print_verification()",
     r'''def run_once(query, embedder, k, show_chunks):''',
     r'''# --- Week 7: show the sources the answer cited, and check them ----------------

def source_file(source):
    """(path, found): the full path of a chunk's source file if it can be found from the
    project root (as stored, under corpus/, or anywhere below corpus/), else the path as stored."""
    s = str(source).replace("\\", "/")
    for p in (s, os.path.join("corpus", s)):
        if os.path.isfile(p):
            return os.path.abspath(p), True
    base = os.path.basename(s)
    for root, _dirs, files in os.walk("corpus"):
        if base in files:
            p = os.path.join(root, base)
            if p.replace("\\", "/").endswith(s):
                return os.path.abspath(p), True
    return s, False


def clip(text, n=110):
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def print_verification(ans):
    """Under the answer: only the passages it cited (the file each came from, and the wording
    of that passage closest to each claim), then what the citation check found."""
    chk, hits = ans.check, ans.hits
    if chk is None or not hits:
        return
    print("\n" + "-" * 78)
    if chk.cited:
        print(f"Sources cited in the answer ({len(chk.cited)} of {len(hits)} retrieved):")
        for n in chk.cited:
            h = hits[n - 1]
            score = hit_score(h)
            print(f"  [{n}] {hit_label(h)}" + (f"   ({score})" if score else ""))
            path, found = source_file(hit_get(h, "source", default="?"))
            print(f"      file    : {path}" + ("" if found else "   (path as stored; not found from here)"))
            also = hit_get(h, "also_in", default=None) or []
            if also:
                print(f"      same text also in: {', '.join(map(str, also[:2]))}")
            for claim in [c for c in chk.claims if n in c.cites][:3]:
                near, _ = citecheck.closest(claim.text, hit_text(h))
                print(f"      claim   : {clip(claim.text)}")
                print(f"      passage : {clip(near) if near else '(nothing close to it in this passage)'}")
        unused = [i for i in range(1, len(hits) + 1) if i not in chk.cited]
        if unused:
            print("Retrieved but not cited: " + " ".join(f"[{i}]" for i in unused))
    else:
        print("Sources cited in the answer: none")
    if chk.problems:
        print(f"Citation check: {len(chk.problems)} thing(s) to look at")
    else:
        print("Citation check: no problems found (it checks wording, not truth)")
    for f in chk.findings:
        print(f"  {f.level:<5} {f.message}")
        if f.excerpt:
            print(f'        in: "{clip(f.excerpt, 90)}"')


def run_once(query, embedder, k, show_chunks):'''),

    ("call it from run_once()",
     r'''    print()
    return True
''',
     r'''    print_verification(ans)
    print()
    return True
'''),
]

src = open(PATH, encoding="utf-8").read()
if "citecheck" in src:
    sys.exit("answer.py already mentions citecheck, so it looks patched already. Nothing changed.")
for name, old, new in EDITS:
    found = src.count(old)
    if found != 1:
        sys.exit(f"Stopped: the snippet for '{name}' was found {found} times in answer.py "
                 "(expected exactly 1). answer.py was not changed.")
    src = src.replace(old, new)
ast.parse(src)  # refuse to write anything that is not valid Python
with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print(f"answer.py patched ({len(EDITS)} edits). To undo: git checkout answer.py")
