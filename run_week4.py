#!/usr/bin/env python3
"""
Casefile — Week 4 runner: build the index TWICE (once per embedder) and run the
same sanity query against each so you can eyeball the difference in retrieval.

This is the deliverable that captures the week's goal in one command:
    python run_week4.py

It will, for each embedder in EMBEDDERS:
  1. build (or upsert into) its own Chroma collection from the Week 3 chunks
  2. run a few sanity queries and print the top hit from each index side by side

Edit EMBEDDERS below if you want a different second model. Both must be pulled
in Ollama first, e.g.:
    ollama pull nomic-embed-text
    ollama pull mxbai-embed-large
"""

from __future__ import annotations

import argparse
import time

from build_index import build, default_collection_name, embed_texts

try:
    import chromadb
except ImportError:
    raise SystemExit("chromadb is not installed. Activate your venv and run: pip install chromadb")


# The two embedders to compare. nomic-embed-text (768-dim) was validated in
# Week 1; mxbai-embed-large (1024-dim) is a strong, larger local alternative.
# Swap the second one for whatever you want to benchmark against.
EMBEDDERS = ["nomic-embed-text", "mxbai-embed-large"]

# A few representative pentest-flavoured probes for the sanity check.
SANITY_QUERIES = [
    "linux privilege escalation via SUID binaries",
    "enumerate SMB shares on a windows host",
    "extract password hashes from /etc/shadow",
]


def build_all(chunks_path: str, db_path: str, batch_size: int, reset: bool) -> None:
    for embedder in EMBEDDERS:
        print("\n" + "=" * 78)
        print(f"BUILD  {embedder}")
        print("=" * 78)
        build(
            embedder=embedder,
            chunks_path=chunks_path,
            db_path=db_path,
            collection_name=default_collection_name(embedder),
            batch_size=batch_size,
            reset=reset,
        )


def compare(db_path: str, top_k: int) -> None:
    client = chromadb.PersistentClient(path=db_path)
    collections = {}
    for embedder in EMBEDDERS:
        name = default_collection_name(embedder)
        try:
            collections[embedder] = client.get_collection(name)
        except Exception:
            print(f"[compare] collection for {embedder} not found — skipping")

    for q in SANITY_QUERIES:
        print("\n" + "-" * 78)
        print(f"QUERY: {q!r}")
        print("-" * 78)
        for embedder, coll in collections.items():
            q_vec = embed_texts([q], embedder)[0]
            res = coll.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
            print(f"\n  [{embedder}]  (dim via {coll.count()} vectors)")
            if not docs:
                print("    (no results)")
                continue
            for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
                sim = 1.0 - dist
                source = (meta or {}).get("source") or (meta or {}).get("path") or "?"
                snippet = " ".join(doc.split())[:120]
                print(f"    #{rank} sim={sim:.4f}  {source}")
                print(f"        {snippet}…")


def main() -> None:
    p = argparse.ArgumentParser(description="Build both Casefile indexes and compare (Week 4).")
    p.add_argument("--chunks", default="corpus/chunks.jsonl", help="Week 3 chunk file")
    p.add_argument("--db", default="./chroma_db", help="Persistent Chroma directory")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reset", action="store_true", help="Rebuild both collections from scratch")
    p.add_argument("--top-k", type=int, default=3, help="Results per query in the comparison")
    p.add_argument("--build-only", action="store_true", help="Build the indexes, skip the comparison")
    p.add_argument("--compare-only", action="store_true", help="Skip building, just compare existing indexes")
    args = p.parse_args()

    t0 = time.time()
    if not args.compare_only:
        build_all(args.chunks, args.db, args.batch_size, args.reset)
    if not args.build_only:
        print("\n" + "=" * 78)
        print("SIDE-BY-SIDE RETRIEVAL COMPARISON")
        print("=" * 78)
        compare(args.db, args.top_k)
    print(f"\n[week4] all done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
