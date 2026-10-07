#!/usr/bin/env python3
"""
retrieve.py v2 -> v3: teach it the prefixed nomic index. Run once, from the project root:

    python patch_retrieve_v3.py

It adds the name "nomic-embed-text-prefixed" to the known indexes and puts
"search_query: " in front of questions asked of that index (and only that index).
A copy of your file is kept as retrieve.py.v2.bak.
"""

import hashlib
import pathlib
import shutil
import sys

PATH = pathlib.Path("retrieve.py")

DOC_OLD = "--list      shows what is in the database: collections, sizes, metric, metadata\n"
DOC_NEW = DOC_OLD + '''
nomic-embed-text-prefixed is the same chunks embedded the way nomic's model card asks
("search_document: " in front of chunks, "search_query: " in front of questions).
python build_nomic_prefixed.py builds it; after that --compare includes it automatically.
'''

SET_OLD = '''KNOWN_EMBEDDERS = ["nomic-embed-text", "qwen3-embedding", "mxbai-embed-large", "all-minilm"]
'''
SET_NEW = '''KNOWN_EMBEDDERS = ["nomic-embed-text", "nomic-embed-text-prefixed", "qwen3-embedding",
                   "mxbai-embed-large", "all-minilm"]
# An index can be built from the same Ollama model but fed differently. For those, this says
# which model to call and what to put in front of every question (it must match the index).
VARIANTS = {"nomic-embed-text-prefixed": ("nomic-embed-text", "search_query: ")}
'''

EMB_OLD = '''def _embed(question: str, embedder: str) -> list[float]:
    try:
        vec = embed_texts([question], embedder)[0]
    except Exception as e:  # Ollama not running, model not pulled, ...
        raise RuntimeError(
            f"could not embed the question with '{embedder}' ({type(e).__name__}: {e}). "
'''
EMB_NEW = '''def _embed(question: str, embedder: str) -> list[float]:
    model, prefix = VARIANTS.get(embedder, (embedder, ""))
    try:
        vec = embed_texts([prefix + question], model)[0]
    except Exception as e:  # Ollama not running, model not pulled, ...
        raise RuntimeError(
            f"could not embed the question with '{model}' ({type(e).__name__}: {e}). "
'''

EDITS = [("docstring", DOC_OLD, DOC_NEW), ("settings", SET_OLD, SET_NEW), ("_embed", EMB_OLD, EMB_NEW)]


def main() -> None:
    if not PATH.exists():
        sys.exit("retrieve.py is not in this folder. Run this from the project root.")
    text = PATH.read_text(encoding="utf-8")
    if "VARIANTS" in text:
        sys.exit("retrieve.py already knows the prefixed index. Nothing to do.")
    for name, old, _ in EDITS:
        if text.count(old) != 1:
            sys.exit(f"Could not patch: the '{name}' part of retrieve.py is not the version I expected "
                     f"(found {text.count(old)} matches). Nothing was changed.")
    for _, old, new in EDITS:
        text = text.replace(old, new)
    try:
        compile(text, str(PATH), "exec")
    except SyntaxError as e:
        sys.exit(f"The patched file would not compile ({e}). Nothing was changed.")
    backup = PATH.with_name("retrieve.py.v2.bak")
    if not backup.exists():
        shutil.copy2(PATH, backup)
    PATH.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    print(f"patched OK: retrieve.py now knows 'nomic-embed-text-prefixed' (checksum {sha}, backup {backup.name})")


if __name__ == "__main__":
    main()
