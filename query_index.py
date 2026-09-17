#!/usr/bin/env python3
"""
Casefile — Week 4 sanity check: query one Chroma index and print the top hits.

Because build_index.py stores precomputed vectors (no embedding function is
attached to the collection), we must embed the query ourselves and search with
`query_embeddings=` rather than `query_texts=`. This script does that.

Usage
-----
    python query_index.py --embedder nomic-embed-text "linux privilege escalation via SUID"
    python query_index.py --embedder mxbai-embed-large -k 3 "how to enumerate SMB shares"

Options mirror build_index.py so the collection resolves to the same place:
    --db          persistent Chroma dir           (default ./chroma_db)
    --collection  collection name                 (default casefile_<embedder>)
    -k / --top-k  how many results to show         (default 5)
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import ollama

try:
    import chromadb
except ImportError:
    sys.exit("chromadb is not installed. Activate your venv and run: pip install chromadb")

from build_index import default_collection_name, embed_texts  # reuse helpers


def query(
    embedder: str,
    text: str,
    db_path: str,
    collection_name: str,
    top_k: int,
) -> None:
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        sys.exit(
            f"Collection '{collection_name}' not found in {db_path}. "
            f"Build it first:  python build_index.py --embedder {embedder}"
        )

    q_vec = embed_texts([text], embedder)[0]
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print(f"\nQuery      : {text!r}")
    print(f"Embedder   : {embedder}   (collection '{collection_name}', {collection.count()} vectors)")
    print(f"Top {len(docs)} results (cosine distance — lower = closer):\n")

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        similarity = 1.0 - dist  # cosine similarity, echoing the Week 1 numbers
        source = (meta or {}).get("source") or (meta or {}).get("path") or "?"
        print(f"  #{rank}  distance={dist:.4f}  similarity={similarity:.4f}  source={source}")
        snippet = " ".join(doc.split())
        for line in textwrap.wrap(snippet[:400], width=88):
            print(f"        {line}")
        if len(snippet) > 400:
            print("        …")
        print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Query a Casefile Chroma index (Week 4 sanity check).")
    p.add_argument("query", help="The natural-language question to search for")
    p.add_argument("--embedder", required=True, help="Ollama embedding model used to build the index")
    p.add_argument("--db", default="./chroma_db", help="Persistent Chroma directory")
    p.add_argument("--collection", default=None, help="Collection name (default casefile_<embedder>)")
    p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to show")
    args = p.parse_args(argv)

    collection_name = args.collection or default_collection_name(args.embedder)
    query(args.embedder, args.query, args.db, collection_name, args.top_k)


if __name__ == "__main__":
    main()
