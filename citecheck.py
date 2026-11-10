#!/usr/bin/env python3
"""
citecheck.py -- Casefile Week 7: "Make it prove its work"

Checks the [n] citations in a model's answer against the passages the model was given.
Plain Python (no Ollama, no Chroma), so it runs instantly and Week 11 can import it too.

    from citecheck import check
    result = check(answer_text, [passage_1_text, passage_2_text, ...])   # [n] means passage n
    result.cited      # the valid numbers the answer cites, e.g. [1, 2]
    result.claims     # every sentence that cites something, with its numbers
    result.findings   # what looks wrong, as ERROR / WARN / NOTE lines
    result.problems   # just the ERROR and WARN lines

What it looks for
  ERROR  a citation number that does not exist (below 1 or above the number of passages)
  WARN   a sentence cites [n], but a flag or number it mentions is not in passage n
         (the message says which passage does contain it, and if passage n has a number that is
         one digit away, such as 65335 against 65535, it says that too)
  WARN   a sentence shares very little of its wording with the passage it cites
  WARN   a sentence with no citation whose flags / numbers / words are in no passage
         (it may have come from the model's memory)
  WARN   a code block that uses a flag found in no passage
  WARN   an answer that cites nothing and does not decline
  NOTE   uncited sentences that do match a passage, a declined answer, an empty code block

What it can NOT do: judge meaning. A sentence that cites the right passage but says the
opposite (nmap -sV vs -sR), or repeats a typo that is in the passage itself, passes. This
is a check on wording and citation numbers, not on truth. Hand grading (Week 11) covers the rest.

    python citecheck.py --selftest      run the built-in tests
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

# --- numbers you can tune ------------------------------------------------------
LOW_SUPPORT = 0.30      # cited sentence: flag it if less than this share of its words is in the passage(s) it cites
BETTER_BY = 0.20        # ...and name another passage if that one matches at least this much better
LOW_ANY = 0.30          # uncited sentence: flag it if less than this share of its words is in ANY passage
MIN_WORDS = 4           # sentences with fewer content words than this are not word-checked
DECLINE_MAX_WORDS = 90  # a decline is short; a long answer that merely contains "not mentioned" is not one


# --- what comes back -----------------------------------------------------------

@dataclass
class Finding:
    level: str            # "ERROR", "WARN" or "NOTE"
    message: str
    excerpt: str = ""     # the sentence (or code line) the message is about


@dataclass
class Claim:
    text: str             # one sentence of the answer that cites something
    cites: list           # the valid passage numbers it cites


@dataclass
class Check:
    cited: list = field(default_factory=list)      # valid passage numbers the answer cites, ascending
    claims: list = field(default_factory=list)     # Claim objects, in answer order
    findings: list = field(default_factory=list)   # Finding objects, ERROR first
    declined: bool = False                         # the answer says the passages do not cover the question

    @property
    def problems(self):
        return [f for f in self.findings if f.level != "NOTE"]

    @property
    def ok(self):
        return not self.problems


# --- words, flags and numbers --------------------------------------------------

_STOP = set("""
a about above after again all also although always among an and any are as at be because been before being
between both but by can could did do does doing done during each else for from had has have having how however
if in into is it its itself just like may might more most must no nor not now of on only or other our out over
per same should since so some such than that the their them then there these they this those through to too
under until up use used uses using very was we were what when where which while who whom why will with within
would you your according additionally following mention mentions mentioned note passage passages state states
stated text
""".split())

_CITE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")                    # [2]  [2][4]  [2, 4]
_INLINE = re.compile(r"`([^`\n]+)`")                                            # `code`
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")                            # 65,535 -> 65535
_WORD = re.compile(r"(?<![a-z0-9])-{1,2}[a-z0-9][a-z0-9-]*|[a-z0-9]+(?:[._:/][a-z0-9]+)*")
_FLAG = re.compile(r"(?<![\w-])--?[A-Za-z0-9][A-Za-z0-9-]*")                  # -sV  --script  -p-
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?!\w)")                         # 7.94  2011  8080
_IP = re.compile(r"\d+(?:\.\d+){3}")


def tokens(text):
    """The content words of a text: lower case, no stop words, plurals trimmed. Flags keep their dash."""
    out = set()
    for t in _WORD.findall(_THOUSANDS.sub("", text.lower())):
        if t in _STOP:
            continue
        is_flag = t.startswith("-")
        if not (is_flag or len(t) >= 3 or (len(t) == 2 and t.isdigit())):
            continue
        if not is_flag and len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t = t[:-1]
        out.add(t)
    return out


def _support(words, have):
    """The share of `words` that also appear in `have` (1.0 when there is nothing to check)."""
    return len(words & have) / len(words) if words else 1.0


def _specifics(text):
    """The flags and numbers a text mentions: [("flag", "-p-"), ("number", "65535"), ...]."""
    text = _THOUSANDS.sub("", text)
    found = [("flag", f) for f in _FLAG.findall(text)]
    found += [("number", n) for n in _NUMBER.findall(text) if len(n) >= 2 and not _IP.fullmatch(n)]
    return list(dict.fromkeys(found))


def _has(spec, passage):
    """Is this flag / number written in the passage? (Flags are case-sensitive: -sV is not -sv.)"""
    kind, value = spec
    if kind == "flag":
        return re.search(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", passage) is not None
    plain = _THOUSANDS.sub("", passage)
    return re.search(r"(?<![\w.])" + re.escape(value) + r"(?!\w|\.\d)", plain) is not None


def _brackets(numbers):
    return "".join(f"[{n}]" for n in numbers)


def _one_off(value, others):
    """The first of `others` that is `value` with exactly one digit changed ("65335" -> "65535"), else "".
    Only for whole numbers of 4+ digits: a slip in a long number is a typo, a slip in "80" is another port."""
    if not (value.isdigit() and len(value) >= 4):
        return ""
    for other in others:
        if other.isdigit() and len(other) == len(value) and sum(a != b for a, b in zip(value, other)) == 1:
            return other
    return ""


# --- taking the answer apart ---------------------------------------------------

_FENCE = "```"
_ROUND_BULLET = chr(0x2022)   # written as chr() so this file stays plain ASCII
_ELLIPSIS = chr(0x2026)
# A list item starts with a SHORT number ("1." or "12)"). A man page wraps its lines, so a line can
# also start with a long number and a full stop ("65535. Scanning port zero ..."): that is not a list.
_BULLET = re.compile(r"^\s*(?:[*" + _ROUND_BULLET + r"-]|\d{1,2}[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(*])")
_DECLINE = re.compile(
    r"\b(?:do(?:es)?\s+not|don'?t|doesn'?t|cannot|can'?t|could\s+not|couldn'?t|unable\s+to|fail(?:s|ed)?\s+to)\b"
    r"[^.]{0,60}?\b(?:contain|mention|cover|include|provide|explain|describe|say|specify|address|answer|have|find|discuss|state|offer)"
    r"|\bno\s+(?:relevant\s+|specific\s+|direct\s+)?(?:information|mention|details?|passages?|content|answer)\b"
    r"|\bnot\s+(?:covered|mentioned|provided|explained|included|addressed|discussed)\b"
    r"|\bnothing\s+(?:about|on|in|relevant)\b"
    r"|\b(?:unrelated|irrelevant|off[- ]topic|outside\s+the\s+scope)\b",
    re.I,
)
_PLACEHOLDER_LINE = re.compile(r"\s*(?:#.*|<[^>]*>|\.\.\.|" + _ELLIPSIS + r"|\(.*\)|\[.*\])?\s*")


def _split_answer(answer):
    """-> (prose, code_blocks): the text outside the ``` fences, and the text inside each fence."""
    if answer.count(_FENCE) % 2:              # the model never closed a fence
        answer += "\n" + _FENCE
    parts = answer.split(_FENCE)
    blocks = []
    for part in parts[1::2]:
        label, _, rest = part.partition("\n")
        blocks.append(rest if re.fullmatch(r"\s*[\w+.#-]*\s*", label) else part)   # drop a "bash" label
    return "\n".join(parts[0::2]), blocks


def _sentences(prose):
    """Split prose into sentences, one bullet or line at a time. A "[4]" after the full stop stays put."""
    out = []
    for line in prose.splitlines():
        line = _BULLET.sub("", line).strip()
        out.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return out


def _cites(sentence):
    """The passage numbers a sentence cites, in order. (A [1] inside `code` is not a citation.)"""
    probe = _INLINE.sub(" ", sentence)
    nums = (int(n) for group in _CITE.findall(probe) for n in re.split(r"[,;]", group))
    return list(dict.fromkeys(nums))


def _body(sentence):
    """The sentence without its [n] markers and backticks: what words, flags and numbers are read from."""
    return _CITE.sub("", _INLINE.sub(r"\1", sentence))


def _is_placeholder(block):
    """True for a code block with nothing real in it: empty, "...", a # comment, a <placeholder>."""
    return all(_PLACEHOLDER_LINE.fullmatch(line) for line in block.splitlines())


# --- the check -----------------------------------------------------------------

def _weight(token):
    """Flags and numbers say more about a claim than ordinary words do."""
    return 3 if token.startswith("-") or any(c.isdigit() for c in token) else 1


def closest(sentence, passage, floor=0.15):
    """The sentence of `passage` that is nearest in wording to `sentence`: (text, score).
    Shown next to a claim so you can read what the cited passage actually says."""
    want = tokens(_body(sentence))
    in_passage = tokens(passage)
    for t in [t for t in want if t not in in_passage]:     # 65335 written, 65535 in the passage: that is the sentence to show
        near = _one_off(t, in_passage)
        if near:
            want.discard(t)
            want.add(near)
    want_weight = sum(_weight(t) for t in want)
    best, best_score = "", 0.0
    for para in re.split(r"\n\s*\n|\n(?=\s*(?:[-*" + _ROUND_BULLET + r"]|\d{1,2}[.)])\s)", passage):
        for cand in re.split(r"(?<=[.!?])\s+", " ".join(para.split())):
            cand = _BULLET.sub("", cand)
            have = tokens(cand)
            if want and have:
                shared = sum(_weight(t) for t in want & have)
                score = shared / (want_weight * sum(_weight(t) for t in have)) ** 0.5
                if score > best_score:
                    best, best_score = cand, score
    return (best, best_score) if best_score >= floor else ("", best_score)


def check(answer, passages):
    """Check the [n] citations in `answer` against `passages` (passage n is passages[n - 1])."""
    k = len(passages)
    res = Check()
    prose, blocks = _split_answer(answer or "")
    sentences = _sentences(prose)
    if not sentences and not any(b.strip() for b in blocks):
        res.findings.append(Finding("NOTE", "the answer is empty"))
        return res

    ptoks = [tokens(p) for p in passages]
    said = " ".join(sentences)
    res.declined = bool(_DECLINE.search(said)) and len(said.split()) <= DECLINE_MAX_WORDS

    def add(level, message, excerpt=""):
        res.findings.append(Finding(level, message, excerpt))

    cited, uncited_match = set(), []
    for sent in sentences:
        ns = _cites(sent)
        good = [n for n in ns if 1 <= n <= k]
        for n in ns:
            if n not in good:
                add("ERROR", f"cites [{n}], but only {k} passage(s) were given (valid: 1 to {k})", sent)
        cited.update(good)
        if good:
            res.claims.append(Claim(sent, good))
        elif _DECLINE.search(sent):
            continue          # "the passages do not mention --foo" is about the passages; nothing to check it against

        label = _brackets(good)
        pool = [n - 1 for n in good] or list(range(k))    # look in the cited passages, else in all of them
        body = _body(sent)

        # 1) every flag and number the sentence mentions should be written in the passage it cites
        missing = []                                       # [(value, [other passages that have it], a number one digit off)]
        for spec in _specifics(body):
            if not any(_has(spec, passages[i]) for i in pool):
                elsewhere = [i + 1 for i in range(k) if i not in pool and _has(spec, passages[i])]
                near = ""
                if spec[0] == "number":                    # 65335 written, 65535 in the cited passage: a typo is the likely story
                    near = _one_off(spec[1], [v for i in pool for kind, v in _specifics(passages[i]) if kind == "number"])
                missing.append((spec[1], elsewhere, near))
        if missing and good:
            items = ", ".join(
                v + (f" (it is in {_brackets(w)}" if w else " (in no passage")
                + (f"; {label} has {near}, one digit different" if near else "") + ")"
                for v, w, near in missing)
            add("WARN", f"{label} does not contain {items}", sent)
        elif missing:
            add("WARN", "no citation, and not in any passage: " + ", ".join(v for v, _, _ in missing), sent)

        # 2) most of the sentence's wording should be in the passage it cites
        words = tokens(body)
        if missing or len(words) < MIN_WORDS or not k:
            continue
        scores = [_support(words, t) for t in ptoks]
        if good:
            have = set().union(*(ptoks[i] for i in pool))
            s = _support(words, have)
            if s < LOW_SUPPORT:
                msg = f"only {s:.0%} of this sentence's wording is in {label}"
                absent = sorted(words - have)          # name the words that are not there: they are the invented part
                if absent:
                    msg += f"; not in {label}: " + ", ".join(absent[:8]) + (", ..." if len(absent) > 8 else "")
                others = [i for i in range(k) if i not in pool]
                if others:
                    b = max(others, key=lambda i: scores[i])
                    if scores[b] >= s + BETTER_BY:
                        msg += f"; [{b + 1}] matches better ({scores[b]:.0%})"
                add("WARN", msg, sent)
        elif not res.declined:
            b = max(range(k), key=lambda i: scores[i])
            if scores[b] < LOW_ANY:
                add("WARN", f"no citation, and little of it is in any passage "
                            f"(best: [{b + 1}], {scores[b]:.0%}); maybe from memory", sent)
            else:
                uncited_match.append(b + 1)

    res.cited = sorted(cited)
    if not cited:
        if res.declined:
            add("NOTE", "the answer says the passages do not cover this, so no citations are expected")
        else:
            add("WARN", "the answer cites no passage, so none of it is backed by a source")
    if uncited_match:
        near = ", ".join(f"[{n}]" for n in sorted(set(uncited_match)))
        add("NOTE", f"{len(uncited_match)} sentence(s) have no valid [n] (closest passage: {near})")

    for block in blocks:
        if _is_placeholder(block):
            add("NOTE", "the answer has an empty or placeholder code block")
            continue
        flags = [f for f in dict.fromkeys(_FLAG.findall(block))
                 if not any(_has(("flag", f), p) for p in passages)]
        if flags:
            first = next((ln.strip() for ln in block.splitlines() if ln.strip()), "")
            add("WARN", "a code block uses flag(s) found in no passage: " + ", ".join(flags), first)

    rank = {"ERROR": 0, "WARN": 1, "NOTE": 2}
    res.findings.sort(key=lambda f: rank[f.level])
    return res


# --- self-test: python citecheck.py --selftest ---------------------------------

def _selftest():
    P = [
        "-sR is an alias for -sV. Prior to March 2011, it was used to activate the RPC grinder "
        "separately from version detection, but now these options are always combined.",
        "To scan every port use nmap -p- <target>. That scans all 65,535 TCP ports. "
        "Add --open to show only the open ports.",
        "Local port forwarding: ssh -L <local_port>:<victim>:<remote_port> user@host. "
        "Add -N -f to keep it quiet in the background.",
    ]
    R = [  # trimmed from a real Casefile run: man nmap, then the HackTricks nmap page, then a HackTricks ffuf entry
        "-p port ranges (Only scan specified ports)\n"
        "           This option specifies which ports you want to scan and overrides\n"
        "           the default. Individual port numbers are OK, as are ranges\n"
        "           separated by a hyphen (e.g.  1-1023). The beginning and/or end\n"
        "           values of a range may be omitted, causing Nmap to use 1 and 65535,\n"
        "           respectively. So you can specify -p- to scan ports from 1 through\n"
        "           65535. Scanning port zero is allowed if you specify it explicitly.",
        "**-p:** Used to specify ports to scan. To select all 65,335 ports: **-p-** or **-p all**. "
        "Nmap has an internal classification based on popularity. By default, it uses the top 1000 ports.",
        "Entry_12:\n  Name: Ffuf Vhost\n  Description: Simple Scan with Ffuf for discovering additional vhosts\n"
        "  Command: ffuf -w {Subdomain_List}:FUZZ -u {Web_Proto}://{Domain_Name} "
        "-H \"Host:FUZZ.{Domain_Name}\" -c -mc all {Ffuf_Filters}",
    ]
    cases = [
        ("a correct, cited answer has no problems",
         "To scan every port, use `-p-` [2].",
         lambda r: r.ok and r.cited == [2]),
        ("a citation number that does not exist is an ERROR",
         "To scan every port, use `-p-` [9].",
         lambda r: any(f.level == "ERROR" and "[9]" in f.message for f in r.findings)),
        ("a number cited to the wrong passage is caught, and the right passage is named",
         "Nmap scans all 65,535 ports with `-p-` [1].",
         lambda r: any("does not contain" in f.message and "it is in [2]" in f.message for f in r.findings)),
        ("an invented flag is caught even when the sentence cites nothing",
         "Add `--fast-mode` to make the scan quicker.",
         lambda r: any("--fast-mode" in f.message for f in r.problems)),
        ("an invented flag inside a code block is caught",
         "Use this:\n```bash\nnmap -p- --fast-mode 10.0.0.5\n```",
         lambda r: any("code block" in f.message and "--fast-mode" in f.message for f in r.findings)),
        ("a factual answer with no citation at all is flagged",
         "Run nmap against the target and wait for the scan to finish on every single port.",
         lambda r: any("cites no passage" in f.message for f in r.problems)),
        ("a decline needs no citation",
         "The passages do not contain any information about baking bread.",
         lambda r: r.ok and r.declined and not r.cited),
        ("a sentence cited to a passage that says nothing like it is flagged",
         "Kerberos tickets can be cracked with a dictionary attack offline [3].",
         lambda r: any("wording" in f.message for f in r.problems)),
        ("closest() prefers the passage sentence that contains the claim's flag",
         "Use `-sV` for version detection [1].",
         lambda r: closest(r.claims[0].text, P[0])[0].startswith("-sR is an alias")),
        ("KNOWN LIMIT: an inverted claim passes, so closest() is how you spot it",
         "`-sV` is an alias for `-sR` [1].",
         lambda r: r.ok and "-sR is an alias for -sV" in closest(r.claims[0].text, P[0])[0]),
        ("odd input does not crash it",
         "```\n[1] [2][3] `[4]` ``` `",
         lambda r: isinstance(r.findings, list)),
        # the next three use passages (R) from a real run
        ("REAL RUN, nmap -p-: a number copied from one passage but cited to another is caught",
         "According to [1], you can specify all 65,335 ports by using the option `-p-` or `-p all`.",
         lambda r: any("does not contain 65335" in f.message and "it is in [2]" in f.message for f in r.findings),
         R),
        ("REAL RUN: a man-page line that wraps right before '65535.' is not cut in two",
         "Use `-p-` to scan ports from 1 through 65535 [1].",
         lambda r: closest(r.claims[0].text, R[0])[0].endswith("65535."),
         R),
        ("REAL RUN, ffuf -mc: an invented sentence is flagged, and the made-up words are named",
         "According to [3], the ffuf -mc flag is used to specify the match condition, "
         "and the possible values are all, partial, or none.",
         lambda r: any("partial" in f.message and "none" in f.message for f in r.problems),
         R),
        ("REAL RUN, nmap -p- reworded: the passage line still shows the sentence with 65535",
         "According to [1], you can use the option `-p-` or `-p all` to scan all 65,335 ports.",
         lambda r: closest(r.claims[0].text, R[0])[0].endswith("65535."),
         R),
        ("REAL RUN, nmap -p-: the warning says the cited passage has 65535, one digit different",
         "According to [1], you can specify all 65,335 ports by using the option `-p-` or `-p all`.",
         lambda r: any("65535" in f.message and "one digit different" in f.message for f in r.findings),
         R),
        ("a number two digits off, or a short one, is not called a typo",
         "x",
         lambda r: _one_off("65335", ["65555"]) == "" and _one_off("443", ["444"]) == ""
                   and _one_off("65335", ["65535"]) == "65535"),
    ]
    failed = 0
    for name, answer, test, *own in cases:
        ok = bool(test(check(answer, own[0] if own else P)))
        failed += not ok
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return failed


if __name__ == "__main__":
    if len(sys.argv) == 1 or "--selftest" in sys.argv[1:]:
        sys.exit(1 if _selftest() else 0)
    print(__doc__)
