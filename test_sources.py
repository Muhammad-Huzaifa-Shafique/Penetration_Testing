"""Tests for sources.py (Casefile Week 7).  Run:  python -m unittest test_sources -v

Standard library only. These use small made-up passages and files, plus answers
modelled on the Week 6 results, so they do not need Ollama or Chroma.
"""
import contextlib
import io
import json
import random
import re
import tempfile
import unittest
from pathlib import Path

import sources
from sources import (Passage, analyse, audit, build_source_ref, check_answer, clean,
                     code_tokens, find_markers, format_report, locate_chunk,
                     resolve_source, split_code)


def cites(text):
    masked, _ = split_code(text)
    return [m.numbers for m in find_markers(masked)]


def levels(result):
    return [f.level for f in result.findings]


def texts(result, level):
    return [f.text for f in result.findings if f.level == level]


def P(n, text, source="", **kw):
    return Passage(n=n, text=text, source=source, **kw)


# ---------------------------------------------------------------------------

class MarkerTests(unittest.TestCase):
    def test_plain_and_adjacent(self):
        self.assertEqual(cites("A [1]. B [2][3]."), [[1], [2], [3]])

    def test_lists_ranges_and_words(self):
        self.assertEqual(cites("x [1, 3] y [4,5] z [2-4] w [2–4]"),
                         [[1, 3], [4, 5], [2, 3, 4], [2, 3, 4]])
        self.assertEqual(cites("[Passage 2] [Source 2, 4] [passages 1 and 3] [Passage 1 to 3]"),
                         [[2], [2, 4], [1, 3], [1, 2, 3]])
        self.assertEqual(cites("see (Passage 3) and (Source 2)"), [[3], [2]])
        self.assertEqual(cites("fullwidth 【2】"), [[2]])

    def test_things_that_are_not_citations(self):
        self.assertEqual(cites("(1) first, (2) second"), [])          # bare round brackets
        self.assertEqual(cites("the year [2024] and a task [x]"), [])
        self.assertEqual(cites("a link [text](http://x)"), [])
        self.assertEqual(cites("reference [^1]"), [])

    def test_code_is_ignored(self):
        self.assertEqual(cites("use `argv[1]` here [2]"), [[2]])
        self.assertEqual(cites("```bash\necho ${a[1]}  # [3]\n```\nthen [4]"), [[4]])
        self.assertEqual(cites("```\nunterminated [5]"), [])

    def test_zero_and_huge_ranges(self):
        self.assertEqual(cites("[0]"), [[0]])
        self.assertEqual(cites("[1-99]"), [[1, 99]])                   # not expanded to 99 numbers


class SegmentTests(unittest.TestCase):
    def segs(self, text):
        return [(s.text, s.cites) for s in analyse(text)[0]]

    def test_each_sentence_keeps_its_own_citation(self):
        self.assertEqual(self.segs("Use -p- [1]. Then run -sV [2]."),
                         [("Use -p- [1].", [1]), ("Then run -sV [2].", [2])])

    def test_marker_after_the_full_stop_belongs_to_the_sentence_before(self):
        self.assertEqual(self.segs("Use -p-. [1] Then stop."),
                         [("Use -p-. [1]", [1]), ("Then stop.", [])])
        self.assertEqual(self.segs("Use -p-.[1] Then stop."),
                         [("Use -p-.[1]", [1]), ("Then stop.", [])])

    def test_marker_alone_on_a_line_joins_the_line_above(self):
        segs = self.segs("Use -p- to scan everything.\n[2]")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0][1], [2])

    def test_list_items_and_prefixes(self):
        text = "1. Run `nmap -sV` [1].\n2) Run `nmap -sU` [2].\n- Done [3]"
        self.assertEqual([c for _, c in self.segs(text)], [[1], [2], [3]])

    def test_decimals_and_filenames_do_not_split(self):
        segs = self.segs("Version 1.5 of README.md is in [1]. Next.")
        self.assertEqual(segs[0][1], [1])
        self.assertEqual(len(segs), 2)

    def test_code_block_takes_the_citation_before_it_else_after_it(self):
        before = analyse("Run this [2]:\n\n```\nnmap -p-\n```")[0]
        self.assertEqual([s.cites for s in before if s.code], [[2]])
        after = analyse("```\nnmap -p-\n```\n\nThat is from [3].")[0]
        self.assertEqual([s.cites for s in after if s.code], [[3]])
        none = analyse("```\nnmap -p-\n```")[0]
        self.assertEqual([s.cites for s in none if s.code], [[]])


class TokenTests(unittest.TestCase):
    def tok(self, line, **kw):
        return [(t.text, t.kind) for t in code_tokens(line, **kw)]

    def test_commands_flags_numbers(self):
        self.assertEqual(self.tok("nmap -p- <target>"), [("nmap", "name"), ("-p-", "flag")])
        self.assertEqual(self.tok("smbclient -L //192.168.1.10 -N"),
                         [("smbclient", "name"), ("-L", "flag"), ("-N", "flag")])
        self.assertEqual(self.tok("hashcat -m 13100 hash.txt rockyou.txt"),
                         [("hashcat", "name"), ("-m", "flag"), ("13100", "number")])

    def test_your_own_values_are_not_claims(self):
        self.assertEqual(self.tok("sudo nmap -sV 10.0.0.5 TARGET example"),
                         [("nmap", "name"), ("-sV", "flag")])
        self.assertEqual(self.tok("$ nmap -p-"), [("nmap", "name"), ("-p-", "flag")])
        self.assertEqual(self.tok("john --wordlist=rockyou.txt hashes.txt"),
                         [("john", "name"), ("--wordlist", "flag")])
        self.assertEqual(self.tok("echo hi | grep x"), [])           # plain shell plumbing

    def test_hyphenated_script_names_are_claims(self):
        self.assertEqual(self.tok("nmap --script smb-enum-shares -p 445 host"),
                         [("nmap", "name"), ("--script", "flag"), ("smb-enum-shares", "name"),
                          ("-p", "flag"), ("445", "number")])

    def test_pipes_start_a_new_command(self):
        self.assertEqual(self.tok("cat x | sqlmap -u url"),
                         [("sqlmap", "name"), ("-u", "flag")])

    def test_flags_only_mode_for_other_languages(self):
        self.assertEqual(self.tok("gcc -shared -fPIC int main", flags_only=True),
                         [("-shared", "flag"), ("-fPIC", "flag")])

    def test_prose_flags_and_numbers(self):
        toks = [(t.text, t.kind) for s in analyse("Use -h or --help on all 65,335 ports, "
                                                  "not well-known ports 1-1024 or e-mail.")[0]
                for t in s.tokens]
        self.assertEqual(toks, [("-h", "flag"), ("--help", "flag"), ("65,335", "number")])

    def test_fence_language_decides_what_is_checked(self):
        shell = analyse("```bash\ngcc -shared x\n```")[0][0]
        self.assertEqual([t.text for t in shell.tokens], ["gcc", "-shared"])
        c = analyse("```c\ngcc -shared x\n```")[0][0]
        self.assertEqual([t.text for t in c.tokens], ["-shared"])

    def test_backslash_continuation(self):
        seg = analyse("```\nnmap -p- \\\n  192.168.1.1 -oN out\n```")[0][0]
        self.assertEqual([t.text for t in seg.tokens], ["nmap", "-p-", "-oN"])


class CheckTests(unittest.TestCase):
    NMAP = [
        P(1, "Service detection\n\nnmap -sV <target> probes open ports.", "hacktricks/a.md"),
        P(2, "-sS (TCP SYN scan)\nSYN scan is the default.\n-sU (UDP scans)", "manpages/nmap.txt"),
        P(3, "Scan all ports\n\nnmap -p- <ip>   # scans all 65,335 ports\nnmap -p all <ip>", "hacktricks/b.md"),
    ]

    def test_supported_claim_is_ok(self):
        r = check_answer("Use `nmap -p-` to scan every port [3].", self.NMAP)
        self.assertTrue(r.ok)
        self.assertEqual(r.cited, [3])
        self.assertEqual(r.not_cited, [1, 2])
        self.assertIn("OK", levels(r))

    def test_right_claim_wrong_number(self):                         # the Week 6 nmap case
        r = check_answer("Use `-p-` to scan all 65,335 ports [2].", self.NMAP)
        self.assertEqual(r.warnings, 1)
        msg = texts(r, "WARN")[0]
        self.assertIn("`-p-`", msg)
        self.assertIn("[3]", msg)
        self.assertIn("wrong number", msg)
        self.assertEqual(r.findings[0].near, "Use `-p-` to scan all 65,335 ports [2].")

    def test_claim_in_no_passage_is_flagged_as_memory(self):
        r = check_answer("Try `smbmap -H host` [1].", self.NMAP)
        self.assertEqual(r.warnings, 1)
        self.assertIn("none of the retrieved passages", texts(r, "WARN")[0])

    def test_numbers_alone_are_only_a_note(self):
        r = check_answer("That is port 4455 [1].", self.NMAP)
        self.assertEqual((r.warnings, levels(r).count("NOTE")), (0, 1))

    def test_flags_are_case_sensitive_names_are_not(self):
        p = [P(1, "Use smbclient -L to list shares.")]
        self.assertTrue(check_answer("Run `smbclient -L` [1].", p).ok)
        self.assertEqual(check_answer("Run `smbclient -l` [1].", p).warnings, 1)
        self.assertTrue(check_answer("Run `SMBCLIENT -L` [1].", p).ok)

    def test_matching_ignores_dash_style_and_thousands_separators(self):
        p = [P(1, "The option ‐sV (version detection) and 65535 ports, 1,024 too.")]
        self.assertTrue(check_answer("Use `-sV` [1] for 65,535 ports and 1024.", p).ok)

    def test_longer_flag_does_not_count_as_the_short_one(self):
        p = [P(1, "--port selects the port.")]
        self.assertEqual(check_answer("Use `-p` [1].", p).warnings, 1)

    def test_invalid_numbers_are_errors(self):
        r = check_answer("A [7]. B [0]. C [2].", self.NMAP)
        self.assertEqual(r.invalid, [0, 7])
        self.assertEqual(r.errors, 2)
        self.assertIn("only passages 1-3", texts(r, "ERROR")[0])
        self.assertEqual(r.cited, [2])
        self.assertFalse(r.ok)
        self.assertNotIn("OK  ", " ".join(f.level + "  " + f.text for f in r.findings if "distinct" in f.text))

    def test_decline_cites_nothing_and_is_not_a_warning(self):
        r = check_answer("The passages do not contain anything about baking bread.", self.NMAP)
        self.assertEqual(levels(r), ["INFO"])
        self.assertTrue(r.ok)
        self.assertEqual(r.cited, [])

    def test_commands_without_any_citation_warn(self):
        r = check_answer("Run `nmap -p-` and then `smbclient -L`.", self.NMAP)
        self.assertEqual(r.warnings, 1)
        self.assertIn("cites nothing", texts(r, "WARN")[0])

    def test_one_uncited_sentence_among_cited_ones_is_a_note(self):
        r = check_answer("Use `nmap -p-` [3]. Then run `smbclient -L` for shares.", self.NMAP)
        self.assertEqual(r.warnings, 0)
        notes = texts(r, "NOTE")
        self.assertEqual(len(notes), 1)
        self.assertIn("`smbclient`", notes[0])

    def test_placeholders_only_means_nothing_to_check(self):
        r = check_answer("Run `<command> TARGET` [1].", self.NMAP)
        self.assertEqual(r.checked_tokens, 0)
        self.assertTrue(r.ok)

    def test_code_block_uses_the_citation_before_it(self):
        ans = "Scan every port [3]:\n\n```bash\nnmap -p- 10.0.0.1\n```"
        self.assertTrue(check_answer(ans, self.NMAP).ok)
        wrong = "Scan every port [2]:\n\n```bash\nnmap -p- 10.0.0.1\n```"
        self.assertEqual(check_answer(wrong, self.NMAP).warnings, 1)

    def test_citing_two_passages_is_supported_by_either(self):
        self.assertTrue(check_answer("Use `-p-` and `-sS` [2][3].", self.NMAP).ok)

    def test_robust_to_empty_and_odd_input(self):
        for ans in ("", None, "   \n", "[", "```", "`", "[1", "((", "[1-]"):
            r = check_answer(ans, self.NMAP)
            self.assertIsNotNone(format_report(r))
        r = check_answer("A [1].", [])
        self.assertEqual(r.invalid, [1])
        self.assertIn("no passages were given", texts(r, "ERROR")[0])
        self.assertIsNotNone(format_report(r))

    def test_fuzz_never_raises(self):
        rng = random.Random(1)
        alphabet = list("[]()`.!? \n-,;&0123456789abc") + ["and", "to", "Passage", "```", "–"]
        for _ in range(600):
            ans = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
            format_report(check_answer(ans, self.NMAP))

    def test_unsafe_text_is_cleaned_before_printing(self):
        evil = P(1, "\x1b[31mred\x1b[0m payload ‮evil", "a\x1b]0;pwn\x07/b.md", section="s\x1b[2J")
        r = check_answer("Use `payload` [1].", [evil])
        out = format_report(r)
        for bad in ("\x1b", "\x07", "‮"):
            self.assertNotIn(bad, out)
        self.assertEqual(clean("a\tb\nc\x00d  e"), "a b cd e")
        self.assertEqual(clean("abcdefghij", 8), "abcde...")


class RefinementTests(unittest.TestCase):
    PS = [P(1, "Use nmap -sS for a SYN scan.\nsmbclient -L lists shares.", "a.md", section="Scans"),
          P(2, "Port forwarding with ssh -L.", "b.md")]

    def test_saying_what_the_passages_lack_is_not_a_claim(self):
        for line in ("The passages do not mention `smbmap` or `enum4linux`.",
                     "Passage [2] does not cover `smbmap -H host`.",
                     "There is no mention of `smbmap` in the passages.",
                     "`smbmap` is not covered.",
                     "I cannot find `smbmap` in the provided context."):
            r = check_answer("Use `nmap -sS` [1]. " + line, self.PS)
            self.assertTrue(r.ok, (line, [f.text for f in r.findings]))
            self.assertEqual(levels(r).count("NOTE"), 0, line)

    def test_a_real_claim_that_merely_contains_not_is_still_checked(self):
        r = check_answer("Do not forget `smbmap -H host` [1]. The source IP is not needed: `rpcclient -U` [1].", self.PS)
        self.assertEqual(r.warnings, 2)

    def test_a_plain_decline_that_names_a_flag_is_info_only(self):
        r = check_answer("The passages do not explain what the `-mc` flag does.", self.PS)
        self.assertEqual(levels(r), ["INFO"])
        self.assertTrue(r.ok)

    def test_no_passages_at_all(self):
        out = format_report(check_answer("I do not know.", []))
        self.assertIn("no passages were retrieved", out)

    def test_bare_passage_references_count(self):
        self.assertEqual(cites("Passage 2 says so. Passages 1, 2 and 3 agree."), [[2], [1, 2, 3]])
        self.assertEqual(cites("traffic from source 10.0.0.1 and source port 80"), [])
        self.assertEqual(cites("see [Passage 2] only"), [[2]])               # not counted twice

    def test_uncited_code_block_is_named_as_such(self):
        r = check_answer("Use `nmap -sS` [1].\n\nSomething else entirely.\n\n```\nsmbmap -H host\n```\n\nDone.", self.PS)
        notes = [f for f in r.findings if f.level == "NOTE"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].text, "a code block has no citation")
        self.assertEqual(notes[0].near, "smbmap -H host")

    def test_excerpt_skips_the_heading(self):
        p = Passage(1, "### Scans\n\n{{#include x.md}}\n\nUse nmap -sS for a SYN scan.", "a.md", section="Scans")
        self.assertEqual(build_source_ref(p, locate=False).excerpt, "Use nmap -sS for a SYN scan.")
        q = Passage(1, "SCAN TECHNIQUES\n   -sS (TCP SYN scan) is the default.", "a.md", section="SCAN TECHNIQUES")
        self.assertEqual(build_source_ref(q, locate=False).excerpt, "-sS (TCP SYN scan) is the default.")


class RealisticAnswerTests(unittest.TestCase):
    """Answers shaped like the Week 6 ones (SMB, sourdough, ffuf)."""

    SMB = [
        P(1, "ShareHound traverses shares.\n\nnxc smb 10.10.10.1 -u user -p pass --shares", "hacktricks/smb.md"),
        P(2, "NTLM and RC4 material for Active Directory.", "hacktricks/ad.md"),
        P(3, "Service identification\n\nnmap -sV -p 445 <target>", "hacktricks/wifi.md"),
        P(4, "List shares with an NT hash\n\nnxc smb <ip> -u <user> -H <hash> --shares", "hacktricks/wifi.md"),
        P(5, "MSSQL file system function xp_dirtree.", "hacktricks/mssql.md"),
    ]

    def test_good_smb_answer(self):
        ans = ("To enumerate SMB shares you can use `nxc smb <target> -u <user> -p <pass> --shares` [1][4]. "
               "To identify the service first, run `nmap -sV -p 445 <target>` [3]. "
               "The passages do not include the ShareQL rules.")
        r = check_answer(ans, self.SMB)
        self.assertTrue(r.ok, [f.text for f in r.findings])
        self.assertEqual(r.cited, [1, 3, 4])
        self.assertEqual(r.not_cited, [2, 5])

    def test_invented_tool_is_caught(self):
        ans = "Use `enum4linux -a <target>` to list shares [1]."
        r = check_answer(ans, self.SMB)
        self.assertEqual(r.warnings, 1)
        self.assertIn("`enum4linux`", texts(r, "WARN")[0])

    def test_sourdough_control(self):
        r = check_answer("The passages contain nothing about baking or cooking.", self.SMB)
        self.assertEqual(levels(r), ["INFO"])
        out = format_report(r)
        self.assertIn("none cited", out)
        self.assertIn("hacktricks/mssql.md", out)           # still shows what the model had

    def test_semantic_errors_are_beyond_a_text_match(self):
        """-sV described backwards passes: the checker says so, and so does its header."""
        p = [P(1, "-sR is an alias for -sV. -sV enables version detection.", "manpages/nmap.txt")]
        r = check_answer("`-sV` is an alias for `-sR` [1].", p)
        self.assertTrue(r.ok)
        self.assertIn("cannot tell whether a sentence is true", format_report(r))

    def test_models_own_source_list_is_still_checked(self):
        ans = "Use `nmap -sV` [3].\n\nSources: [3], [9]"
        r = check_answer(ans, self.SMB)
        self.assertEqual(r.invalid, [9])


# ---------------------------------------------------------------------------

def mimic_chunks(text, size=420, overlap=60):
    """Paragraph packer with an overlap that starts mid-line, like the Week 3 chunker."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > size:
            out.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        out.append(cur)
    return out


DOC = "\n".join([
    "# Pentesting SMB",
    "",
    "{{#include ../../banners/hacktricks-training.md}}",
    "",
] + [f"## Section {i}\n\nParagraph {i} explains topic {i} with enough words to be a real line of text.\n"
     f"A second line for section {i} that mentions smbclient -L and shares.\n\n"
     f"{{{{#include ../../banners/hacktricks-training.md}}}}\n" for i in range(1, 15)])


class LocateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "corpus" / "hacktricks" / "src").mkdir(parents=True)
        (self.root / "corpus" / "manpages").mkdir()
        self.doc = self.root / "corpus" / "hacktricks" / "src" / "smb.md"
        self.doc.write_text(DOC, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def line_of(self, needle):
        for i, l in enumerate(DOC.splitlines(), 1):
            if needle in l:
                return i

    def test_every_chunk_is_located_at_its_true_start(self):
        chunks = mimic_chunks(DOC)
        self.assertGreater(len(chunks), 5)
        lines = [" ".join(l.split()) for l in DOC.splitlines()]
        exact_count = 0
        for c in chunks:
            line, exact = locate_chunk(self.doc, c)
            self.assertIsNotNone(line, c[:60])
            first = next(" ".join(l.split()) for l in c.splitlines() if l.strip())
            if exact:                       # 'exact' means the chunk's first line IS that file line
                exact_count += 1
                self.assertTrue(lines[line - 1].endswith(first), (line, first, lines[line - 1]))
        self.assertGreaterEqual(exact_count, len(chunks) - 1)

    def test_repeated_boilerplate_does_not_confuse_the_lookup(self):
        chunk = "## Section 9\n\nParagraph 9 explains topic 9 with enough words to be a real line of text."
        line, exact = locate_chunk(self.doc, chunk)
        self.assertEqual(line, self.line_of("## Section 9"))
        self.assertTrue(exact)

    def test_blank_lines_may_differ(self):
        chunk = ("Paragraph 3 explains topic 3 with enough words to be a real line of text.\n\n\n\n"
                 "A second line for section 3 that mentions smbclient -L and shares.")
        line, exact = locate_chunk(self.doc, chunk)
        self.assertEqual(line, self.line_of("Paragraph 3 explains"))

    def test_crlf_files(self):
        self.doc.write_bytes(DOC.replace("\n", "\r\n").encode("utf-8"))
        line, _ = locate_chunk(self.doc, "## Section 5\n\nParagraph 5 explains topic 5 with enough words to be a real line of text.")
        self.assertEqual(line, self.line_of("## Section 5"))

    def test_file_changed_since_indexing(self):
        line, exact = locate_chunk(self.doc, "This sentence was never in the file at all, honestly.\nNeither was this one, not a word.")
        self.assertIsNone(line)

    def test_unreadable_or_missing_file(self):
        self.assertEqual(locate_chunk(self.root / "nope.md", "some long enough line of text here"), (None, False))

    def test_resolve_source_candidates(self):
        found, shown = resolve_source("hacktricks/src/smb.md", self.root)
        self.assertEqual(found, self.doc)
        self.assertEqual(shown, "corpus/hacktricks/src/smb.md")
        found, _ = resolve_source("corpus/hacktricks/src/smb.md", self.root)      # already prefixed
        self.assertEqual(found, self.doc)
        found, _ = resolve_source("hacktricks\\src\\smb.md", self.root)           # Windows separators
        self.assertEqual(found, self.doc)
        found, _ = resolve_source(str(self.doc), self.root)                        # absolute
        self.assertEqual(found, self.doc)
        found, _ = resolve_source("src/smb.md", self.root, corpus_root=self.root / "corpus" / "hacktricks")
        self.assertEqual(found, self.doc)
        found, shown = resolve_source("hacktricks/src/gone.md", self.root)
        self.assertIsNone(found)
        self.assertEqual(shown, "corpus/hacktricks/src/gone.md")                   # what it expected

    def test_source_ref_and_report(self):
        chunk = mimic_chunks(DOC)[2]
        p = Passage(1, chunk, "hacktricks/src/smb.md", "Section 4", 0.7, 2)
        gone = Passage(2, "text", "hacktricks/src/gone.md", "x", 0.6, 1)
        changed = Passage(3, "A line that is not in the file at all, sorry.\nAnother one that is not in it.", "hacktricks/src/smb.md")
        other = Passage(4, "unrelated", "manpages/x.txt")
        r = check_answer("Use `smbclient -L` [1][2][3].", [p, gone, changed, other], self.root)
        out = format_report(r)
        refs = {s.n: s for s in r.sources}
        self.assertTrue(refs[1].exists and refs[1].line)
        self.assertIn("FILE NOT FOUND", out)
        self.assertEqual(refs[3].problem, "chunk text not found in file")
        self.assertIn(f"corpus/hacktricks/src/smb.md:{'' if refs[1].line_exact else '~'}{refs[1].line}", out)
        self.assertIn(f"less +{refs[1].line} corpus/hacktricks/src/smb.md", out)
        self.assertIn("Retrieved but not cited: [4]", out)

    def test_paths_with_spaces_are_quoted_in_the_open_hint(self):
        d = self.root / "corpus" / "my notes"
        d.mkdir()
        (d / "a b.md").write_text("line one is long enough\nline two is also long enough\n", encoding="utf-8")
        p = Passage(1, "line one is long enough\nline two is also long enough", "my notes/a b.md")
        out = format_report(check_answer("see [1]", [p], self.root))
        self.assertIn("less +1 'corpus/my notes/a b.md'", out)


class AuditTests(unittest.TestCase):
    def test_audit_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus" / "hacktricks" / "src").mkdir(parents=True)
            (root / "corpus" / "hacktricks" / "src" / "smb.md").write_text(DOC, encoding="utf-8")
            recs = [{"id": f"c{i}", "source": "hacktricks/src/smb.md", "text": c}
                    for i, c in enumerate(mimic_chunks(DOC))]
            recs.append({"id": "bad", "source": "hacktricks/src/smb.md",
                         "text": "This text is not in the file anywhere, I promise.\nNot even this second line of it."})
            recs.append({"id": "gone", "source": "hacktricks/src/gone.md", "text": "whatever text goes here"})
            with open(root / "corpus" / "chunks.jsonl", "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
                fh.write("not json\n")
            lines = []
            stats = audit(root / "corpus" / "chunks.jsonl", root, sample=0, out=lines.append)
            self.assertEqual(stats["chunks"], len(recs))
            self.assertEqual(stats["sources"], 2)
            self.assertEqual(stats["found"], 1)
            self.assertEqual(stats["missing"], ["hacktricks/src/gone.md"])
            self.assertEqual([r["id"] for r in stats["lost"]], ["bad"])
            self.assertEqual(stats["exact"] + stats["approx"], len(recs) - 2)
            self.assertTrue(any("NOT found on disk: 1" in l for l in lines))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sources.main(["--audit", "--project", str(root), "--sample", "5"]), 0)
                self.assertEqual(sources.main(["--audit", "--project", str(root / "corpus" / "nothing")]), 2)


if __name__ == "__main__":
    unittest.main()
