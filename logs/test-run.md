# Casefile in a practice lab: test-run

**Author:** Your Name  
**Course:** [Course]  
**Lab:** test-run  
**Dates:** 2026-11-25 to 2026-11-25 (1 session(s))  
**Tool tested:** Casefile, a local retrieval-augmented study assistant (HackTricks and man pages)

## 1. Scope and rules

- Everything here was done against the machine or application the lab deployed for me (`test-run`). Nothing else was scanned or attacked. *(Edit this line if that is not true.)*
- This report contains no flags, passwords or cracked hashes, which is TryHackMe's rule for writeups. Anything I typed that looked like one was removed when it was logged. A pattern check cannot catch everything, so I read the file again before handing it in.
- AI assistance: Casefile is a local assistant (an 8B language model) that I built with AI help. I used it as a reference book while working and labelled every result myself. This file was assembled from my session log by `lab_session.py`; section 11 is my own writing.

## 2. What I tested and why

Casefile answers penetration-testing questions only from the passages it retrieves from HackTricks and the man pages of 20 tools, cites them, and says so when they do not cover the question. The risk with any such assistant is a confident answer that is wrong (a made-up flag or command). During the lab I asked it the how-do-I questions that came up, tried what it said where I could, and recorded whether it held up.

## 3. Setup

| Setting | Value |
|---|---|
| Chat model | llama3.1:8b |
| Temperature | 0.2 |
| Context window (tokens) | 8192 |
| Search index | nomic-embed-text-prefixed |
| Passages per question | answer.py default |
| 'Nothing found' cutoff | 0.62 |
| Conversation memory | on |
| Chunks in the corpus | 22,720 |
| HackTricks commit | a5fcd4f418a7 |
| answer.py sha256 | 7434457d6dbd |

## 4. Results at a glance

I asked 1 question(s). 1 produced an answer. 1 are labelled.

| Outcome | Meaning | Count | Share of labelled |
|---|---|---|---|
| worked | I used it and it did what the answer said | 0 | 0% |
| needed a fix | partly right: I had to change something | 0 | 0% |
| wrong or made up | a flag, command or fact in it was false | 0 | 0% |
| correct decline | it said nothing found / not covered, and that was right | 0 | 0% |
| missed | it declined or found nothing, but the books do cover it | 0 | 0% |
| not tried | I read it but did not or could not try it | 1 | 100% |

None of the answers has been marked as tried yet, so there is no worked-or-wrong rate.
Answering took 157 s on average (157 to 157 s).

## 5. Every question

| # | Question | Outcome | Top similarity | Cited | Time |
|---|---|---|---|---|---|
| 1 | Which nmap option scans every TCP port? | not tried | 0.846 | [2] | 157 s |

## 6. What went wrong

### Wrong or made up

None.

### Missed

None.

### Needed a fix

None.

### Correct declines

None.

## 7. Similarity and outcome

| Outcome | Questions | Lowest | Mean | Highest |
|---|---|---|---|---|
| not tried | 1 | 0.846 | 0.846 | 0.846 |

This is only a hint: with so few questions, a gap between rows may be chance. If wrong answers tend to have lower similarity, a "low confidence" label would catch some of them.

## 8. Weaknesses found in the lab

No findings were recorded (use `/finding` during a session).
## 9. Compared with my earlier test (Week 11)

*From my own Week 11 grading; edit if it is out of date.* I asked Casefile a 20-question answer key (18 answerable). Grades: 9 correct, 5 partly correct, 1 wrong, 3 honestly declined. When the passage holding the answer reached the model it got 8 of 10 right and none wrong; when it did not, it got 1 of 8 right. The lab results above show whether the same pattern holds on questions I did not write.

## 10. Limits

- One lab and a small number of questions.
- The labels are my own judgement and I could be wrong about some of them.
- The model's wording is not seeded, so the same question can give a different answer.
- Casefile knows one HackTricks snapshot and 20 man pages. Tools that are not in them (ffuf, gobuster and others) are covered only where HackTricks mentions them.
- Answers take about two minutes on my CPU.
- The citation check compares wording, flags and numbers; it cannot tell whether a claim is true.

## 11. In my own words

> **TODO: write this section yourself before handing the report in.** Answer in a few sentences each.

- The hardest question I asked, and what Casefile did with it: **TODO**
- Where it saved me time, and where it wasted it: **TODO**
- A time it sounded sure and was wrong, and what gave it away: **TODO**
- The first thing I would change, and why: **TODO**

## Appendix: every answer in full

### Q1. Which nmap option scans every TCP port?

Outcome: not tried  

~~~~text
According to [2], you can specify all 65,335 ports by using the option `-p-` or `-p all`.
~~~~

Passages given to the model (* = cited in the answer):

- [1] man nmap > EXAMPLES (0.846)
- [2]* man nmap > PORT SPECIFICATION AND SCAN ORDER (0.833)
- [3] generic-methodologies-and-resources/pentesting-network/nmap-summary-esp > **Focus Analysis** (0.833)
- [4] man nmap > PORT SCANNING BASICS (0.823)
- [5] man nmap > HOST DISCOVERY (0.816)

- Citation check: WARN: [2] does not contain 65335 (it is in [3]; [2] has 65535, one digit different)
