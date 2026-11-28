# Casefile

Casefile is a local study assistant for penetration-testing coursework. You ask a question; it
searches HackTricks and the man pages of 20 security tools, answers only from the passages it
found, cites them by number, and then checks its own citations. Everything runs on your own
machine through Ollama and Chroma. Nothing is sent to a cloud service; only the setup downloads
things (packages, models and the HackTricks pages).

> **Authorized use only.** Casefile is a study aid for labs, CTFs and coursework. Use what it
> tells you only on systems you own or have written permission to test.

## What this project shows

Casefile is a retrieval-augmented generation (RAG) system. It first finds the passages in a document collection that match a question, then has a language model answer from those passages only, and it runs entirely on a local machine.

| What an AI/ML team looks for | Where it shows up here |
|---|---|
| A full RAG pipeline, built stage by stage | Collect, split into chunks of about 1,500 characters with overlap, embed, index 22,720 chunks in a Chroma vector database, search, answer with llama3.1:8b through Ollama, then check the citations. Each stage is its own script |
| Comparing embedding setups instead of picking one | The same 22,720 chunks are indexed three ways: nomic-embed-text, the same model with its recommended prefixes, and all-minilm. On 7 test questions, the number of top-5 passages that all three returned ranged from 0 to 4 and was 1 or fewer on four of the seven (see `week5_compare_3way.txt`). The choice of embedding model changes what the language model gets to read |
| Guardrails against made-up answers | `sources.py` checks every [n] citation against the passages the model was shown, and checks that commands, flags and numbers in a cited sentence appear in the passage it cites. It catches a citation to a passage that was never retrieved, a flag that sits in a different passage, and a tool that appears in no passage. 58 unit tests cover it |
| Knowing what a check cannot prove | The header of `sources.py` says the check cannot tell whether a sentence is true. A test gives it a Week 6-style answer that describes `-sV` backwards, and it still passes a text match, so the limit is documented and tested rather than hidden |
| Private by design | The embeddings and the answer model both run through Ollama on your own computer, so no question goes to a cloud AI service |

**Not done yet.** The 7 questions in `my_questions.txt` are a smoke test, as the first line of that file says, not a benchmark. There is no answer key yet, so there are no recall@k or answer-accuracy figures, and no claim that one embedding setup beats another. An off-topic control question (sourdough bread) got the lowest top-1 similarity of the seven under all three setups, between 0.29 and 0.58 depending on the model. That hints a similarity cutoff could flag questions the collection cannot answer, but no cutoff has been tested, and scores from different models cannot be compared with each other.

**Next step.** Build the answer key (the "Week 9 answer key" named in `my_questions.txt`): questions paired with the passages that really answer them. Then measure recall@5 for each embedding setup and score the written answers for how well they stick to their sources.

## How it works

```
question -> embed it (Ollama) -> 5 closest chunks (Chroma) -> prompt -> llama3.1:8b -> answer with [n] citations
                                                                                  -> citation check
```

- **Corpus:** HackTricks at commit `a5fcd4f4` (1,068 Markdown files) and the man pages of 20
  tools, split into 22,720 chunks (median 608 characters, at most 1,701).
- **Indexes:** three Chroma collections of the same chunks, one per embedding setup:
  `nomic-embed-text`, `nomic-embed-text-prefixed` (the default) and `all-minilm`.
- **Answers:** `llama3.1:8b` (temperature 0.2, 8,192-token context), told to use only the
  numbered passages and to say so when they do not hold the answer.
- **Checks:** under every answer, the passages it cited, the file each came from, and a citation
  check of wording, flags and numbers. If no passage is close enough to the question, Casefile
  answers "Nothing found" at once, without asking the model.
- **Follow-ups:** in the chat, a short follow-up such as "what about UDP?" is rewritten into a
  standalone question before the search.

## Files

| File | What it does |
|---|---|
| `collect_corpus.sh` | gets HackTricks into `corpus/hacktricks/`, saves man pages to `corpus/manpages/`, writes `corpus/MANIFEST.txt` |
| `chunk_corpus.py` | splits the corpus into chunks: `corpus/chunks.jsonl` |
| `build_index.py` | embeds the chunks with one Ollama model into `chroma_db/` (resumable) |
| `build_nomic_prefixed.py` | builds the prefixed nomic index from the plain one (resumable) |
| `retrieve.py` | search only, from the command line or as `from retrieve import retrieve` |
| `answer.py` | the chat: search, grounded answer, cited sources, citation check |
| `citecheck.py` | the citation check (plain Python) |
| `conversation.py` | what the chat remembers, and the follow-up rewrite (plain Python) |
| `answer_key.txt`, `check_key.py`, `review_key.py` | the evaluation key (20 questions) and its checker |
| `eval_retrieval.py` | compares the indexes on the key (search only) |
| `eval_answers.py` | asks the key through `answer.py` and collects the answers for grading |
| `requirements.txt` | the Python packages, pinned |

Other files in the folder are week-by-week helpers and saved outputs.

## Reference setup

The corpus was collected in September 2026; the indexes and the evaluation below were run on:

| | |
|---|---|
| System | Ubuntu 24.04.3 LTS on WSL2 (Windows), kernel 6.6.87.2 |
| Hardware | 8 CPUs, 7.6 GiB RAM, CPU only |
| Python | 3.12.3, with ollama 0.6.2, chromadb 1.5.9, numpy 2.3.3 |
| Ollama | 0.40.2 |
| Models | `nomic-embed-text:latest` (ID `0a109f422b47`), `all-minilm:latest` (`1b226e2802db`), `llama3.1:8b` (`46e0c10c039e`) |
| Disk | about 6 GB for the models and HackTricks, plus `chroma_db/` |

## Setup, from a fresh Ubuntu 24.04

On Windows, install Ubuntu under WSL first (PowerShell: `wsl --install -d Ubuntu-24.04`) and run
everything below in the Ubuntu terminal, from the project root (the folder that holds `answer.py`).
If your copy already has `corpus/` and `chroma_db/`, do steps 1, 3 and 4, then go to "Using it".

### 1. System packages

```bash
sudo apt update
sudo apt install -y git python3-venv man-db bsdextrautils
```

### 2. The tools whose man pages go into the corpus

```bash
sudo apt install -y nmap netcat-openbsd nikto sqlmap whatweb dirb curl wget smbclient snmp \
  ldap-utils hydra john hashcat libcap2-bin iproute2
```

These give the 20 man pages in the corpus: nmap, nc, nikto, sqlmap, whatweb, dirb, curl, wget,
smbclient, rpcclient, snmpwalk, ldapsearch, hydra, john, hashcat, sudo, find, chmod, getcap, ss.
`nikto` is in Ubuntu's multiverse component; if apt cannot find it, run
`sudo add-apt-repository multiverse` and `sudo apt update`.

### 3. Ollama and the three models

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.2 sh
ollama pull nomic-embed-text
ollama pull all-minilm
ollama pull llama3.1:8b
ollama list
```

`ollama list` should show the IDs in the table above. If it says it cannot connect, start the
server: `ollama serve` in a second terminal (or `sudo systemctl start ollama` if the installer
set it up as a service). A different ID means a different version of that model: indexes you
build with it still work, but similarity scores (and the right cutoffs) may differ from the
numbers below.

### 4. Python environment

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the three packages Casefile imports. `requirements-lock.txt`, if present,
records every package of the reference environment (`pip install -r requirements-lock.txt` for an
exact copy). Activate the venv (`source ~/.venv/bin/activate`) in every new terminal.

### 5. The corpus: HackTricks at a fixed commit, plus the man pages

`collect_corpus.sh` on its own would download the newest HackTricks. To get the exact pages the
indexes were built from, fetch the pinned commit first:

```bash
mkdir -p corpus
git init -q corpus/hacktricks
git -C corpus/hacktricks remote add origin https://github.com/HackTricks-wiki/hacktricks.git
git -C corpus/hacktricks fetch --depth 1 origin a5fcd4f418a7edb7c508fccecfe5d42625c88533
git -C corpus/hacktricks checkout -q --detach FETCH_HEAD
git -C corpus/hacktricks remote remove origin
git -C corpus/hacktricks rev-parse HEAD
```

The last line must print `a5fcd4f418a7edb7c508fccecfe5d42625c88533` (commit of 2026-09-24
01:33 +0200, "Merge pull request #2845"). Removing the remote keeps the checkout pinned. Then:

```bash
bash collect_corpus.sh
```

It finds the HackTricks copy and tries to update it, which fails on purpose (a git message, then
`(pull skipped; keeping existing copy)`); then it saves the man pages and writes
`corpus/MANIFEST.txt`. The MANIFEST should say
`markdown files: 1068`. If it lists more than 20 man pages (tools that happen to be installed on
your machine), remove the extra ones so the corpus matches:

```bash
keep=" nmap nc nikto sqlmap whatweb dirb curl wget smbclient rpcclient snmpwalk ldapsearch hydra john hashcat sudo find chmod getcap ss "
for f in corpus/manpages/*; do t=$(basename "$f"); t=${t%%.*}; case "$keep" in *" $t "*) ;; *) echo "removing $f"; rm -f "$f" ;; esac; done
ls corpus/manpages | wc -l
```

The count must be 20.

### 6. Chunks

```bash
python chunk_corpus.py
```

Expected output:

```
==> Chunking complete
    documents chunked : 1088
    total chunks      : 22720
      from hacktricks : 21826
      from man pages  : 894
    chunk chars       : min 50 / median 608 / avg 691 / max 1701
```

If every number matches, you have the chunks the indexes were built from. (On the reference
laptop in November 2026, running the chunker again on this corpus gave exactly the chunks indexed
in September, and the 20 man pages still rendered identically.) Chunks follow the
Markdown headings and the man-page sections, aim at 1,500 characters with a 200-character
overlap, and keep their file, section and tool as metadata.

### 7. The indexes

```bash
python build_index.py --embedder nomic-embed-text
python build_index.py --embedder all-minilm
python build_nomic_prefixed.py
python retrieve.py --list
```

Each build prints its progress and can be stopped with Ctrl-C and started again with the same
command; it carries on where it stopped. The nomic-embed-text index took about 3 h 40 min on the
reference laptop; `build_nomic_prefixed.py` embeds every chunk again with the same model, so plan
for a similar time; all-minilm is a much smaller model (45 MB against 274 MB).
`build_nomic_prefixed.py` first re-embeds three stored chunks and compares them with the stored
vectors (`check : ... lowest similarity`); it stops if they differ.

`python retrieve.py --list` should show three collections, each with `vectors : 22720 (same as
chunks.jsonl)` and `metric : cosine`: `casefile_nomic-embed-text` and
`casefile_nomic-embed-text-prefixed` with 768 dimensions, `casefile_all-minilm` with 384.

## Using it

```bash
python retrieve.py "how do I enumerate SMB shares"            # search only, in seconds
python answer.py                                               # the chat
python answer.py -q "Which nmap option scans every TCP port?"  # one question
```

`answer.py` first checks that the index is there, that Ollama answers and that the models are
pulled. For each question it lists the passages it found with their similarity, streams the
answer, and then shows the passages the answer cited, the file each came from, the passage line
closest to each claim, and what the citation check found. In the chat, follow-ups are
remembered; `/history` shows what is remembered, `/reset` forgets it, `quit` or Ctrl-D exits.
A failed question (Ollama stopped, say) is reported in a few lines and the chat carries on.

| Option | Effect |
|---|---|
| `-e INDEX` | search another index: `nomic-embed-text`, `nomic-embed-text-prefixed` (default) or `all-minilm` |
| `-k N` | passages per question (default 5) |
| `--show-chunks` | also print the full text of the passages sent to the model |
| `--turns N` | earlier turns the chat remembers (default 3; 0 = none) |
| `--no-rewrite` | search a follow-up as typed, without the rewrite step |
| `--no-cutoff` | ask the model even when no passage is close enough |
| `--no-checks` | skip the start-up checks |

Answers are slow on a CPU: 48 s to 3.7 min each (mean 111 s) in the evaluation run on the
reference laptop. The first question also loads the model, which then stays loaded for 30 minutes.

**"Nothing found" cutoff.** When the closest passage scores below the cutoff for the index
searched, `answer.py` does not ask the model. Each index has its own value, because each
embedding model has its own similarity scale. Each cutoff is the lowest top-1 similarity seen for
a question the corpus answers, minus about 0.04, rounded down:

| Index | Cutoff | Lowest answered | Made-up off-topic questions turned away |
|---|---|---|---|
| `nomic-embed-text-prefixed` | 0.62 | 0.667 | 7 of 8 |
| `nomic-embed-text` | 0.60 | 0.648 | 7 of 8 |
| `all-minilm` | 0.40 | 0.445 | 8 of 8 |

None of the 20 key questions falls below these values. They are set in `CUTOFFS` near the top of
`answer.py`.

**Settings.** `CASEFILE_EMBEDDER=all-minilm` makes another index the default and
`CASEFILE_DB=PATH` moves the database (both read by `retrieve.py`). The chat model, context size,
temperature, memory and cutoffs are constants near the top of `answer.py`.

## Evaluation

```bash
python check_key.py        # checks answer_key.txt against the corpus: 20 questions, 0 errors
python eval_retrieval.py   # every index on the key, search only: week10_retrieval.txt and .json
python eval_answers.py     # the key through answer.py: week11_answers.txt and .json (about 40 minutes)
```

`eval_answers.py` will not overwrite an earlier run: use `--resume` to continue it or
`--out NAME` for a new one. Each script also has `--selftest`.

Results on the reference setup (2026-11-10), on the 18 key questions the corpus answers:

- **Search:** the page that holds the answer was in the top 5 for 17 or 18 of the 18 questions
  in every index; the passage that holds the answer for 10 to 12 of 18 (best: a merged list of
  `nomic-embed-text-prefixed` and `all-minilm`, 12).
- **Answers** (`nomic-embed-text-prefixed`, top 5; grades proposed with AI help, reviewed and
  confirmed by the author): 9 correct, 5 partly correct, 1 wrong, 3 declined. Of the 2 key
  questions the corpus cannot answer, one was declined and one answered anyway. The 20 questions
  took 37 minutes.

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| `Casefile cannot start yet:` `the index: No Chroma database at ...` | Run from the project root (the folder with `chroma_db/`), or build the indexes (step 7). |
| `the index: No index for 'X'` | That index was not built; `python retrieve.py --list` shows the ones that exist. |
| `Ollama does not answer` | Start Ollama (`ollama serve` in a second terminal) and check with `ollama list`. |
| `the model 'X' is not pulled` | `ollama pull X` |
| `The search failed, so no answer was generated` | Usually Ollama stopped; start it and ask again. |
| `man nmap` says "No manual entry" although nmap is installed | This Ubuntu was installed without man pages (a minimized system); Ubuntu's `unminimize` command restores them. Then run `bash collect_corpus.sh` again. |
| `chunk_corpus.py` does not give 22,720 chunks | Check `git -C corpus/hacktricks rev-parse HEAD` and `ls corpus/manpages`. Man-page text can also differ with the installed package versions. |
| `build_nomic_prefixed.py` stops after its check | The nomic model now embeds the stored text differently from when the index was built (another model version). Rebuild the nomic index, or use `--force` knowing the comparison is unfair. |

## Known limits

- Casefile knows only HackTricks at one commit and 20 man pages. Tools whose man pages were not
  installed when the corpus was built (ffuf, gobuster, enum4linux, netexec and others) are covered
  only where HackTricks mentions them.
- The search misses often: for 6 to 8 of the 18 key questions, the passage with the answer was
  not in the top 5. The model is told to say so then, but it sometimes answers from a nearby
  passage instead (one wrong answer in the evaluation).
- Section headings are stored with each chunk but not embedded, so a heading's words do not help
  the search.
- The citation check compares wording, flags and numbers; it cannot tell whether a claim is true.
- The "nothing found" cutoff turns away only clearly off-topic questions. A question close to the
  corpus whose answer is missing (Snort rules, for example) still goes to the model, which is told
  to decline it.
