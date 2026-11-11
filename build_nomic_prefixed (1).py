#!/usr/bin/env python3
"""
Casefile - Week 5 follow-up: build a third index, nomic-embed-text WITH its prefixes.

nomic-embed-text's model card asks for "search_document: " in front of every chunk and
"search_query: " in front of every question. The Week 4 build sent plain text for both.

This copies every chunk (same id, text and metadata) out of the existing nomic collection
and embeds it again WITH the document prefix, into a new collection. Nothing that exists is
changed or deleted. retrieve.py adds the matching "search_query: " to your questions.

    python build_nomic_prefixed.py

It prints progress and an estimate. Ctrl+C is safe: run it again and it carries on.
Do not run searches while it builds. When it says done:

    python retrieve.py --list
    python retrieve.py --compare --file my_questions.txt | tee week5_compare_3way.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

try:
    import chromadb
except ImportError:
    sys.exit("chromadb is not installed. Activate your venv first: source ~/.venv/bin/activate")

# the same helpers Week 4 used
from build_index import default_collection_name, embed_texts

DB = "./chroma_db"
MODEL = "nomic-embed-text"  # the Ollama model, and the existing index to copy from
NEW_NAME = "nomic-embed-text-prefixed"  # the name retrieve.py uses for the new index
DOC_PREFIX = "search_document: "
BATCH = 64
MIN_MATCH = 0.98  # the old vectors must be reproducible from the stored text (see check below)


def _names(client) -> list[str]:
    return [getattr(c, "name", c) for c in client.list_collections()]


def _get(client, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return client.get_collection(name, embedding_function=None)


def _ids_in(collection) -> set[str]:
    ids: set[str] = set()
    offset = 0
    while True:
        res = collection.get(include=["metadatas"], limit=5000, offset=offset)
        if not res["ids"]:
            return ids
        ids.update(res["ids"])
        offset += len(res["ids"])


def _fmt(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the prefixed nomic-embed-text index.")
    ap.add_argument("--force", action="store_true", help="skip the safety check on the existing vectors")
    args = ap.parse_args(argv)

    if not os.path.exists(os.path.join(DB, "chroma.sqlite3")):
        sys.exit(f"No Chroma database at {os.path.abspath(DB)}. Run this from the project root.")
    client = chromadb.PersistentClient(path=DB)
    have = _names(client)
    src_name = default_collection_name(MODEL)
    dst_name = default_collection_name(NEW_NAME)
    if src_name not in have:
        sys.exit(f"No collection '{src_name}' to copy from. Collections that exist: {', '.join(have) or 'none'}.")
    src = _get(client, src_name)
    total = src.count()
    if total == 0:
        sys.exit(f"'{src_name}' is empty.")

    # Safety check. If the stored text, embedded plainly again, does not give back the stored
    # vectors, the Week 4 build embedded something other than the chunk text (a heading plus
    # the text, say). The new index would then differ in more than the prefix and the
    # comparison would be unfair.
    try:
        sample = src.get(limit=3, include=["documents", "embeddings"])
        stored = np.asarray(sample["embeddings"], dtype=float)
        fresh = np.asarray(embed_texts(list(sample["documents"]), MODEL), dtype=float)
    except Exception as e:
        sys.exit(f"could not embed with '{MODEL}' ({type(e).__name__}: {e}). Is Ollama running (try: ollama list)?")
    if stored.shape != fresh.shape:
        sys.exit(f"stored vectors have shape {stored.shape} but '{MODEL}' now gives {fresh.shape}. Stopping.")
    match = (stored * fresh).sum(axis=1) / (np.linalg.norm(stored, axis=1) * np.linalg.norm(fresh, axis=1))
    print(f"check  : stored vs freshly embedded vectors for {len(match)} chunks, lowest similarity {match.min():.4f}")
    if match.min() < MIN_MATCH and not args.force:
        sys.exit(
            "Stopped, so you do not get a misleading comparison. The vectors in "
            f"'{src_name}' are not what Ollama gives for the stored text now. Either the Week 4 "
            "build embedded something other than the chunk text, or the model changed since. "
            "Look at how build_index.py builds the text it embeds, or run again with --force."
        )

    meta = dict(src.metadata or {}) or {"hnsw:space": "cosine"}
    dst = client.get_or_create_collection(dst_name, metadata=meta, embedding_function=None)
    done = _ids_in(dst)
    todo = total - len(done)
    print(f"source : '{src_name}'  ({total} chunks)")
    print(f"target : '{dst_name}'  ({len(done)} already there, {todo} to embed)")
    if todo <= 0:
        print("nothing to do.")
        return

    start = last = time.time()
    added = offset = 0
    try:
        while offset < total:
            got = src.get(limit=BATCH, offset=offset, include=["documents", "metadatas"])
            ids = list(got["ids"])
            if not ids:
                break
            offset += len(ids)
            keep = [i for i, cid in enumerate(ids) if cid not in done]
            if keep:
                docs = [got["documents"][i] for i in keep]
                vectors = embed_texts([DOC_PREFIX + d for d in docs], MODEL)
                dst.add(
                    ids=[ids[i] for i in keep],
                    documents=docs,
                    metadatas=[got["metadatas"][i] for i in keep],
                    embeddings=vectors,
                )
                added += len(keep)
            now = time.time()
            if added and (last == start or now - last >= 15 or added >= todo):
                last = now
                rate = added / max(now - start, 1e-9)
                print(f"  {added}/{todo} embedded   {rate:.1f} chunks/s   about {_fmt((todo - added) / rate)} left",
                      flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped after {added} chunks. Run the same command again to carry on.")
        sys.exit(130)

    count = dst.count()
    print(f"done   : '{dst_name}' now holds {count} chunks ({added} added this run) in {_fmt(time.time() - start)}")
    if count != total:
        print(f"warning: the source has {total} chunks. Run this again to fill the gap.")


if __name__ == "__main__":
    main()
