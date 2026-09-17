#!/usr/bin/env python3
"""
Casefile — Week 4: build a persistent Chroma index from the Week 3 chunks.

This script is parameterized by embedder so it can be run once per embedder
(that is Week 4's whole point: build the index twice and compare retrieval
quality in later weeks). Each embedder gets its OWN Chroma collection inside
one shared on-disk database, so the two never mix.

Design note — why we precompute embeddings instead of letting Chroma call the
model: we embed every chunk ourselves with Ollama and hand the raw vectors to
Chroma via `collection.add(embeddings=...)`. This keeps everything local, gives
us batching + a progress readout, avoids Chroma trying to download its own
embedding model, and is robust across Chroma versions. The trade-off is that we
must always QUERY with `query_embeddings=` (never `query_texts=`), which
query_index.py does. Distance is set to cosine to match the Week 1 mechanic.

Usage
-----
    python build_index.py --embedder nomic-embed-text
    python build_index.py --embedder mxbai-embed-large

Full options:
    python build_index.py \
        --embedder nomic-embed-text \
        --chunks corpus/chunks.jsonl \
        --db ./chroma_db \
        --collection casefile_nomic-embed-text \   # auto-derived if omitted
        --batch-size 64 \
        --reset                                     # drop & rebuild collection

Chunk file format (Week 3 output)
---------------------------------
Either a JSONL file (one JSON object per line) or a JSON array. Each chunk
object should have some text field and ideally an id + metadata. The loader is
tolerant about names:
    text     : "text" | "document" | "content" | "chunk"
    id       : "id" | "chunk_id" | "_id"   (auto-generated if missing)
    metadata : "metadata" | "meta"         (or any leftover flat fields)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Iterable, Iterator

import ollama

try:
    import chromadb
except ImportError:
    sys.exit(
        "chromadb is not installed in this environment.\n"
        "Activate your venv and run:  pip install chromadb"
    )


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
    raise KeyError(
        f"No text field found in chunk (looked for {TEXT_KEYS}). Keys present: {list(obj)}"
    )


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
        # No explicit metadata dict: treat leftover scalar fields as metadata.
        used = set(TEXT_KEYS) | set(ID_KEYS) | set(META_KEYS)
        meta = {k: v for k, v in obj.items() if k not in used}
    # Chroma only accepts str/int/float/bool metadata values; coerce the rest.
    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = json.dumps(v, ensure_ascii=False)
    return clean


def load_chunks(path: str) -> list[dict]:
    """Load chunks from a .jsonl (one object per line) or .json array file."""
    if not os.path.exists(path):
        sys.exit(f"Chunk file not found: {path}\n(Point --chunks at your Week 3 output.)")

    raw: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":  # JSON array
            data = json.load(f)
            if not isinstance(data, list):
                sys.exit("Expected a JSON array of chunk objects.")
            raw = data
        else:  # JSONL
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
        chunks.append(
            {
                "id": _extract_id(obj, i),
                "text": _extract_text(obj),
                "metadata": _extract_metadata(obj),
            }
        )
    if not chunks:
        sys.exit(f"No chunks loaded from {path}.")

    # Guard against duplicate ids (Chroma requires unique ids per collection).
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
    """
    Embed a batch of texts with Ollama. Prefers the batch API
    (ollama.embed(input=[...])) and falls back to per-item calls for older
    versions of the ollama python library.
    """
    try:
        resp = ollama.embed(model=model, input=texts)
        vecs = resp.get("embeddings") if isinstance(resp, dict) else resp["embeddings"]
        if vecs and len(vecs) == len(texts):
            return [list(v) for v in vecs]
    except (TypeError, KeyError, ollama.ResponseError):
        pass  # fall through to per-item

    out: list[list[float]] = []
    for t in texts:
        r = ollama.embeddings(model=model, prompt=t)  # legacy single-item API
        out.append(list(r["embedding"]))
    return out


# --------------------------------------------------------------------------- #
# Index build
# --------------------------------------------------------------------------- #
def build(
    embedder: str,
    chunks_path: str,
    db_path: str,
    collection_name: str,
    batch_size: int,
    reset: bool,
) -> None:
    print(f"[week4] embedder      : {embedder}")
    print(f"[week4] chunks        : {chunks_path}")
    print(f"[week4] chroma db     : {db_path}")
    print(f"[week4] collection    : {collection_name}")

    chunks = load_chunks(chunks_path)
    print(f"[week4] loaded {len(chunks)} chunks")

    client = chromadb.PersistentClient(path=db_path)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[week4] --reset: dropped existing collection '{collection_name}'")
        except Exception:
            pass  # didn't exist yet

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",   # cosine distance, matching Week 1's mechanic
            "embedder": embedder,     # provenance: which model built this index
            "source_chunks": os.path.basename(chunks_path),
        },
    )

    existing = collection.count()
    if existing and not reset:
        print(
            f"[week4] collection already holds {existing} vectors; upserting "
            f"(pass --reset to rebuild from scratch)."
        )

    t0 = time.time()
    dim: int | None = None
    n_done = 0
    total = len(chunks)

    for batch in _batched(chunks, batch_size):
        texts = [c["text"] for c in batch]
        vectors = embed_texts(texts, embedder)

        if dim is None:
            dim = len(vectors[0])
            print(f"[week4] embedding dimension: {dim}")

        # upsert = safe to re-run without duplicating ids
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=vectors,
            metadatas=[c["metadata"] for c in batch],
        )

        n_done += len(batch)
        pct = 100 * n_done / total
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed else 0
        print(f"\r[week4] embedded {n_done}/{total} ({pct:5.1f}%)  "
              f"{rate:5.1f} chunks/s", end="", flush=True)

    print()  # newline after progress line
    elapsed = time.time() - t0
    final_count = collection.count()
    print(f"[week4] DONE  {final_count} vectors  dim={dim}  in {elapsed:.1f}s "
          f"({total/elapsed:.1f} chunks/s)")
    print(f"[week4] persisted to: {os.path.abspath(db_path)}  "
          f"collection='{collection_name}'")


def default_collection_name(embedder: str) -> str:
    # e.g. "nomic-embed-text" -> "casefile_nomic-embed-text"
    safe = embedder.replace(":", "_").replace("/", "_")
    return f"casefile_{safe}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build a Chroma index for Casefile (Week 4).")
    p.add_argument("--embedder", required=True,
                   help="Ollama embedding model, e.g. nomic-embed-text or mxbai-embed-large")
    p.add_argument("--chunks", default="corpus/chunks.jsonl",
                   help="Path to Week 3 chunk file (.jsonl or .json array)")
    p.add_argument("--db", default="./chroma_db",
                   help="Directory for the persistent Chroma database")
    p.add_argument("--collection", default=None,
                   help="Collection name (default: casefile_<embedder>)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="How many chunks to embed per Ollama call")
    p.add_argument("--reset", action="store_true",
                   help="Drop the collection first and rebuild from scratch")
    args = p.parse_args(argv)

    collection_name = args.collection or default_collection_name(args.embedder)
    build(
        embedder=args.embedder,
        chunks_path=args.chunks,
        db_path=args.db,
        collection_name=collection_name,
        batch_size=args.batch_size,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
