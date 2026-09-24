#!/usr/bin/env python3
r"""
Casefile — Week 4: build a persistent Chroma index from the Week 3 chunks.

RESUMABLE: this version checks which chunks are already stored and skips them,
so if the build is interrupted (you stop it, the PC restarts, WSL shuts down),
just run the SAME command again and it continues from where it left off instead
of starting over. Safe to re-run as many times as you like.

This script is parameterized by embedder so it can be run once per embedder
(Week 4's goal: build the index per embedder and compare retrieval later). Each
embedder gets its OWN Chroma collection inside one shared on-disk database.

Usage
-----
    python build_index.py --embedder nomic-embed-text
    python build_index.py --embedder mxbai-embed-large

Full options:
    python build_index.py --embedder mxbai-embed-large --chunks corpus/chunks.jsonl --db ./chroma_db --batch-size 64
    (add --reset to wipe the collection and rebuild from scratch)

Chunk file format (Week 3 output): a .jsonl file (one JSON object per line) or a
JSON array. Each chunk should have a text field and ideally an id + metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Iterator

import ollama

try:
    import chromadb
except ImportError:
    sys.exit("chromadb is not installed. Activate your venv and run: pip install chromadb")


# --------------------------------------------------------------------------- #
# Chunk loading
# --------------------------------------------------------------------------- #
TEXT_KEYS = ("text", "document", "content", "chunk", "body")
ID_KEYS = ("id", "chunk_id", "_id", "uid")
META_KEYS = ("metadata", "meta")


def _extract_text(obj: dict) -> str:
    for k in TEXT_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            return obj[k]
    raise KeyError(f"No text field found in chunk (looked for {TEXT_KEYS}). Keys: {list(obj)}")


def _extract_id(obj: dict, fallback_index: int) -> str:
    for k in ID_KEYS:
        if k in obj and obj[k] not in (None, ""):
            return str(obj[k])
    return f"chunk-{fallback_index:06d}"


def _extract_metadata(obj: dict) -> dict:
    for k in META_KEYS:
        if k in obj and isinstance(obj[k], dict):
            meta = dict(obj[k])
            break
    else:
        used = set(TEXT_KEYS) | set(ID_KEYS) | set(META_KEYS)
        meta = {k: v for k, v in obj.items() if k not in used}
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is not None:
            clean[k] = json.dumps(v, ensure_ascii=False)
    # Chroma rejects an empty metadata dict, so give it a harmless marker.
    if not clean:
        clean = {"_": ""}
    return clean


def load_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"Chunk file not found: {path}")
    raw: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                sys.exit("Expected a JSON array of chunk objects.")
            raw = data
        else:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError as e:
                    sys.exit(f"Bad JSON on line {line_no} of {path}: {e}")

    chunks = []
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            sys.exit(f"Chunk #{i} is not a JSON object: {type(obj).__name__}")
        chunks.append({"id": _extract_id(obj, i), "text": _extract_text(obj),
                       "metadata": _extract_metadata(obj)})
    if not chunks:
        sys.exit(f"No chunks loaded from {path}.")

    seen: dict[str, int] = {}
    for c in chunks:
        if c["id"] in seen:
            seen[c["id"]] += 1
            c["id"] = f"{c['id']}-dup{seen[c['id']]}"
        else:
            seen[c["id"]] = 0
    return chunks


# --------------------------------------------------------------------------- #
# Embedding via Ollama
# --------------------------------------------------------------------------- #
def _batched(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    try:
        resp = ollama.embed(model=model, input=texts)
        vecs = resp["embeddings"]
        if vecs and len(vecs) == len(texts):
            return [list(v) for v in vecs]
    except (TypeError, KeyError, ollama.ResponseError):
        pass
    out: list[list[float]] = []
    for t in texts:
        r = ollama.embeddings(model=model, prompt=t)
        out.append(list(r["embedding"]))
    return out


# --------------------------------------------------------------------------- #
# Index build (resumable)
# --------------------------------------------------------------------------- #
def build(embedder: str, chunks_path: str, db_path: str, collection_name: str,
          batch_size: int, reset: bool) -> None:
    print(f"[week4] embedder      : {embedder}")
    print(f"[week4] chunks        : {chunks_path}")
    print(f"[week4] chroma db     : {db_path}")
    print(f"[week4] collection    : {collection_name}")

    chunks = load_chunks(chunks_path)
    total = len(chunks)
    print(f"[week4] loaded {total} chunks")

    client = chromadb.PersistentClient(path=db_path)
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[week4] --reset: dropped existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedder": embedder,
                  "source_chunks": os.path.basename(chunks_path)},
    )

    already = collection.count()
    if already and not reset:
        print(f"[week4] RESUME: {already}/{total} chunks already embedded — skipping those "
              f"and continuing with the rest.")

    t0 = time.time()
    dim: int | None = None
    n_done = already          # count already-stored ones as done
    n_new = 0

    for batch in _batched(chunks, batch_size):
        ids = [c["id"] for c in batch]

        # RESUME CHECK: which of these ids are already in the index?
        try:
            existing = set(collection.get(ids=ids, include=[])["ids"])
        except Exception:
            existing = set()
        todo = [c for c in batch if c["id"] not in existing]

        if not todo:
            continue  # whole batch already done — skip, no embedding needed

        texts = [c["text"] for c in todo]
        vectors = embed_texts(texts, embedder)

        if dim is None:
            dim = len(vectors[0])
            print(f"[week4] embedding dimension: {dim}")

        collection.upsert(
            ids=[c["id"] for c in todo],
            documents=texts,
            embeddings=vectors,
            metadatas=[c["metadata"] for c in todo],
        )

        n_done = collection.count()
        n_new += len(todo)
        pct = 100 * n_done / total
        elapsed = time.time() - t0
        rate = n_new / elapsed if elapsed else 0
        eta = (total - n_done) / rate if rate else 0
        print(f"\r[week4] {n_done}/{total} ({pct:5.1f}%)  "
              f"new this run: {n_new}  {rate:4.1f}/s  ETA {eta/3600:4.1f}h",
              end="", flush=True)

    print()
    elapsed = time.time() - t0
    final = collection.count()
    print(f"[week4] DONE  {final}/{total} vectors  dim={dim}  "
          f"(embedded {n_new} this run in {elapsed:.0f}s)")
    print(f"[week4] persisted to: {os.path.abspath(db_path)}  collection='{collection_name}'")
    if final < total:
        print(f"[week4] NOTE: {total - final} chunks still remain — re-run the same "
              f"command to continue.")


def default_collection_name(embedder: str) -> str:
    safe = embedder.replace(":", "_").replace("/", "_")
    return f"casefile_{safe}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build a Chroma index for Casefile (Week 4, resumable).")
    p.add_argument("--embedder", required=True)
    p.add_argument("--chunks", default="corpus/chunks.jsonl")
    p.add_argument("--db", default="./chroma_db")
    p.add_argument("--collection", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reset", action="store_true", help="Drop the collection and rebuild from scratch")
    args = p.parse_args(argv)
    build(args.embedder, args.chunks, args.db,
          args.collection or default_collection_name(args.embedder),
          args.batch_size, args.reset)


if __name__ == "__main__":
    main()
