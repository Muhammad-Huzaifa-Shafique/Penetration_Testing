# Casefile: a local study assistant for penetration-testing notes

Casefile answers penetration-testing questions using public documentation (the HackTricks wiki and the man pages of common security tools) and shows exactly which passages each answer came from. Everything runs on your own computer, so no questions or documents are sent to a cloud AI service.

It was built week by week as a capstone project and works in six steps: collect documents, split them into pieces, turn the pieces into searchable vectors, search them, write an answer with a local language model, and check the answer's citations.

> Intended for learning, authorized labs, CTFs, and coursework. Casefile only retrieves and summarizes existing public documentation. Always check an answer against the sources it cites.

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
question -> search the index -> best-matching passages -> local chat model -> answer with [1], [2] citations -> citation check
```

| Step | What happens | Main files |
|---|---|---|
| 1. Collect | Clones HackTricks and saves man pages (or help text) for common security tools into `corpus/` | `collect_corpus.sh` |
| 2. Chunk | Splits the documents into overlapping pieces of about 1,500 characters and labels each with its source and tool | `chunk_corpus.py` |
| 3. Index | Turns each chunk into a vector with an embedding model and stores it in a local Chroma database (`chroma_db/`) | `build_index.py`, `run_week4.py`, `build_nomic_prefixed.py` |
| 4. Search | Finds the chunks closest in meaning to a question; can compare several embedding models side by side | `retrieve.py` |
| 5. Answer | Sends the question and the best chunks to a local chat model (llama3.1:8b through Ollama) | `answer.py` |
| 6. Check | Confirms that every [n] citation points at a passage that was really retrieved, and that commands, flags, and numbers in cited sentences appear in that passage | `sources.py` |

## What you need

- A Linux machine (developed on Ubuntu, including under WSL) with Python 3 (developed with 3.12)
- [Ollama](https://ollama.com), installed and running
- The Python packages in `requirements.txt` (ollama, chromadb, numpy)

Pull the models with Ollama:

```bash
ollama pull nomic-embed-text      # embeddings (used for the default index)
ollama pull llama3.1:8b           # writes the answers
ollama pull mxbai-embed-large     # optional: a second embedding model to compare
```

The code also supports `all-minilm` and `qwen3-embedding` as embedding models. `inotify-tools` or `watchdog` are only needed for the auto-save scripts.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash collect_corpus.sh                              # 1. collect the documents (HackTricks is large)
python chunk_corpus.py                              # 2. split them into corpus/chunks.jsonl
python build_index.py --embedder nomic-embed-text   # 3. build the first index
python build_nomic_prefixed.py                      # 3b. build the default index (nomic with its recommended prefixes)

python answer.py -q "Which nmap option scans every TCP port?"   # ask a question
```

The default index is `nomic-embed-text-prefixed`, which is why step 3b is needed. Pick another index with `-e`.

## Using it

```bash
python answer.py                                      # interactive: type questions, "quit" to exit
python answer.py -q "how do I enumerate SMB shares"   # one question
python answer.py -q "..." -e all-minilm -k 8          # choose the index and how many passages to use
python answer.py -q "..." --show-chunks               # also print the passages sent to the model

python retrieve.py "ssh local port forwarding"                          # search only, no chat model
python retrieve.py --compare --file my_questions.txt                    # run questions through every index side by side
python retrieve.py --source-type manpage --tool nmap "scan all ports"   # narrow the search
python retrieve.py --list                                               # show what is in the database

python sources.py --audit                  # check citations across the whole index
python -m unittest test_sources -v         # run the tests for sources.py
```

## Settings

- `CASEFILE_EMBEDDER` chooses the index to search (default `nomic-embed-text-prefixed`). `CASEFILE_DB` moves the database (default `./chroma_db`).
- Available indexes: `nomic-embed-text`, `nomic-embed-text-prefixed`, `mxbai-embed-large`, `all-minilm`, `qwen3-embedding`. Build each one first with `build_index.py --embedder NAME` (or `build_nomic_prefixed.py` for the prefixed one).
- At the top of `answer.py`: the chat model (`llama3.1:8b`), the context window (8192 tokens), the temperature (0.2), and the instructions the model follows.

## Project files

| File | Purpose |
|---|---|
| `collect_corpus.sh`, `chunk_corpus.py` | Collect and split the documents |
| `build_index.py`, `build_nomic_prefixed.py`, `run_week4.py` | Build the searchable indexes |
| `retrieve.py`, `query_index.py` | Search the indexes |
| `answer.py` | Ask a question and get a cited answer |
| `sources.py`, `test_sources.py` | Check citations, and the tests for that check |
| `inspect_index.py` | Read-only report on the indexes |
| `test_embed.py`, `cosine_similarity_test.py` | Early checks that embeddings work |
| `patch_retrieve_v3.py`, `retrieve.py.v2.bak` | One-time patch that taught `retrieve.py` the prefixed index, and the backup it made |
| `my_questions.txt`, `week6_questions.txt` | Test questions |
| `week4_*.txt`, `week5_*.txt`, `week6_answers.txt` | Saved reports and results from each week |
| `demo_week5.sh` | Screen-recording demo for Week 5 |
| `setup_github.sh`, `autosave.sh`, `autosave_watchdog.py` | GitHub setup and automatic commit-and-push helpers |

## Weekly progress

| Week | Focus |
|---|---|
| 2 | Collect the corpus |
| 3 | Split it into chunks |
| 4 | Build indexes with two embedding models |
| 5 | Search, compare embedding models, add the prefixed nomic index |
| 6 | Get real answers from a local chat model |
| 7 | Make every answer prove its work with source checks |

## Good to know

- `corpus/` and `chroma_db/` are created on your machine and are not stored in this repository (they are listed in `.gitignore`).
- Answers come from the retrieved passages only. If the passages do not contain the answer, Casefile says so.
- Source checks can show that a command or flag appears in a cited passage, but they cannot prove a sentence is true. Read the cited text.
- On a computer without a GPU, the first words of an answer can take a while.

## Author

Made by [Muhammad Huzaifa Shafique](https://github.com/Muhammad-Huzaifa-Shafique).
