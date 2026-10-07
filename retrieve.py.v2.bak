#!/usr/bin/env python3
"""
Casefile - Week 5: search the index with either embedding model.

Takes a typed question, embeds it with the embedding model you pick, and pulls
back the closest-matching chunks from THAT model's index.

Pick the model with a simple setting (first one found wins):
    -e / --embedder NAME        on the command line
    CASEFILE_EMBEDDER=NAME      environment variable
    DEFAULT_EMBEDDER            the default near the top of this file
(CASEFILE_DB=PATH likewise moves the database; the default is ./chroma_db.)

Usage
-----
    python retrieve.py "how do I enumerate SMB shares"
    python retrieve.py -e mxbai-embed-large -k 3 "linux privilege escalation via SUID"
    python retrieve.py --compare "how do I enumerate SMB shares"
    python retrieve.py --compare --file my_questions.txt
    python retrieve.py --source-type manpage --tool nmap "scan all ports"
    python retrieve.py --list

--compare   runs the same question through every index and shows them side by side
            (or name two or more with -e, e.g. -e nomic-embed-text -e mxbai-embed-large)
--file      reads one question per line (blank lines and lines starting with # are skipped)
--no-dedupe keeps chunks whose text is identical. HackTricks repeats some sections on
            several pages; by default only the best-ranked copy is shown
--source-type / --tool
            narrow the search using the metadata from Week 3; run --list to see
            which values exist
--list      shows what is in the database: collections, sizes, metric, metadata

Week 6's chat loop can import this instead of shelling out:
    from retrieve import retrieve
    for hit in retrieve("how do I enumerate SMB shares", embedder="nomic-embed-text", k=5):
        print(hit.source, hit.distance, hit.text)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field

try:
    import chromadb
except ImportError:
    sys.exit("chromadb is not installed. Activate your venv and run: pip install chromadb")

from build_index import default_collection_name, embed_texts  # same helpers Week 4 used

# --- the simple setting ------------------------------------------------------
DEFAULT_EMBEDDER = os.environ.get("CASEFILE_EMBEDDER", "nomic-embed-text")
DEFAULT_DB = os.environ.get("CASEFILE_DB", "./chroma_db")
# --compare and --list look for an index for each of these. Add a name if you build another.
KNOWN_EMBEDDERS = ["nomic-embed-text", "qwen3-embedding", "mxbai-embed-large", "all-minilm"]
CHUNKS_FILE = "corpus/chunks.jsonl"
OVERFETCH = 3  # with dedupe on, fetch k * OVERFETCH results so k different chunks survive


@dataclass
class Hit:
    """One search result. Week 6's chat loop will build its prompt from these."""

    rank: int
    id: str
    text: str
    distance: float
    similarity: float | None  # only for cosine / inner-product indexes
    source: str
    source_type: str
    tool: str
    section: str
    metadata: dict = field(default_factory=dict)
    also_in: list = field(default_factory=list)  # other pages holding this exact text


# --- opening an index --------------------------------------------------------
def _client(db_path: str):
    # Check first: PersistentClient quietly creates an empty database in the wrong folder.
    if not os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
        raise LookupError(
            f"No Chroma database at {os.path.abspath(db_path)}. "
            "Run this from the project root, or pass --db PATH."
        )
    return chromadb.PersistentClient(path=db_path)


def _names(client) -> list[str]:
    # Some Chroma versions return names, others return Collection objects.
    return [getattr(c, "name", c) for c in client.list_collections()]


def _get(client, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return client.get_collection(name, embedding_function=None)


def open_collection(embedder: str, db_path: str = DEFAULT_DB):
    client = _client(db_path)
    name = default_collection_name(embedder)
    have = _names(client)
    if name not in have:
        raise LookupError(
            f"No index for '{embedder}': there is no collection '{name}' in {db_path}. "
            f"Collections that exist: {', '.join(have) or 'none'}."
        )
    return _get(client, name)


def metric_of(collection) -> str:
    """'cosine', 'l2' or 'ip': how Chroma measures distance in this index."""
    meta = collection.metadata or {}
    if meta.get("hnsw:space"):
        return meta["hnsw:space"]
    cfg = getattr(collection, "configuration", None)
    try:
        hnsw = cfg.get("hnsw") if hasattr(cfg, "get") else getattr(cfg, "hnsw", None)
        space = hnsw.get("space") if hasattr(hnsw, "get") else getattr(hnsw, "space", None)
        if space:
            return space
    except Exception:
        pass
    return "l2"  # Chroma's default


# --- searching ---------------------------------------------------------------
def _embed(question: str, embedder: str) -> list[float]:
    try:
        vec = embed_texts([question], embedder)[0]
    except Exception as e:  # Ollama not running, model not pulled, ...
        raise RuntimeError(
            f"could not embed the question with '{embedder}' ({type(e).__name__}: {e}). "
            "Is Ollama running (try: ollama list) and is the model pulled?"
        ) from e
    return [float(x) for x in vec]


def _where(source_type: str | None, tool: str | None):
    clauses = []
    if source_type:
        clauses.append({"source_type": source_type})
    if tool:
        clauses.append({"tool": tool})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _search(collection, embedder, question, k, source_type, tool, dedupe=True) -> list[Hit]:
    q_vec = _embed(question, embedder)
    where = _where(source_type, tool)
    extra = {"where": where} if where else {}
    # With dedupe on, ask for extra results so k different chunks are left after copies are dropped.
    fetch = max(k, min(k * OVERFETCH, collection.count())) if dedupe else k
    try:
        res = collection.query(
            query_embeddings=[q_vec],
            n_results=fetch,
            include=["documents", "metadatas", "distances"],
            **extra,
        )
    except Exception as e:
        hint = " (was this index built with a different model?)" if "dimension" in str(e).lower() else ""
        raise RuntimeError(f"search in '{collection.name}' failed: {e}{hint}") from e

    uses_similarity = metric_of(collection) in ("cosine", "ip")  # both are 1 - similarity
    hits: list[Hit] = []
    seen: dict[str, Hit] = {}  # chunk text -> the hit that kept it
    rows = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    for cid, doc, meta, dist in rows:
        meta = meta or {}
        source = str(meta.get("source") or meta.get("path") or "?")
        key = " ".join((doc or "").split()).lower()
        if dedupe and key and key in seen:  # same text as a better-ranked hit: note where else it lives
            kept = seen[key]
            if source != kept.source and source not in kept.also_in:
                kept.also_in.append(source)
            continue
        if len(hits) >= k:  # enough hits; keep scanning only to note copies of the ones we have
            continue
        hit = Hit(
            rank=len(hits) + 1,
            id=cid,
            text=doc or "",
            distance=float(dist),
            similarity=(1.0 - float(dist)) if uses_similarity else None,
            source=source,
            source_type=str(meta.get("source_type") or ""),
            tool=str(meta.get("tool") or ""),
            section=str(meta.get("section") or ""),
            metadata=dict(meta),
        )
        hits.append(hit)
        seen[key] = hit
    return hits


def retrieve(
    question: str,
    embedder: str = DEFAULT_EMBEDDER,
    k: int = 5,
    db_path: str = DEFAULT_DB,
    source_type: str | None = None,
    tool: str | None = None,
    dedupe: bool = True,
) -> list[Hit]:
    """The k chunks closest to `question` in `embedder`'s index, best first.

    Chunks with identical text (HackTricks repeats some sections on several pages) are
    returned once, with the other pages listed in `hit.also_in`. Pass dedupe=False to keep them.
    """
    collection = open_collection(embedder, db_path)
    return _search(collection, embedder, question, k, source_type, tool, dedupe)


def available_embedders(db_path: str) -> list[str]:
    have = set(_names(_client(db_path)))
    candidates = list(dict.fromkeys([DEFAULT_EMBEDDER] + KNOWN_EMBEDDERS))
    return [m for m in candidates if default_collection_name(m) in have]


# --- printing ----------------------------------------------------------------
def _snippet(text: str, limit: int | None) -> str:
    s = " ".join(text.split())
    if limit is None or len(s) <= limit:
        return s
    return s[:limit].rstrip() + " ..."


def _where_text(h: Hit) -> str:
    kind = h.source_type + (f"/{h.tool}" if h.tool else "")
    path = h.source + (f" > {h.section}" if h.section else "")
    return f"[{kind}] {path}" if kind else path


def _score(h: Hit) -> str:
    if h.similarity is None:
        return f"dist={h.distance:.4f}"
    return f"sim={h.similarity:.4f}  dist={h.distance:.4f}"


def _also(h: Hit) -> str:
    shown = ", ".join(h.also_in[:2]) + (" ..." if len(h.also_in) > 2 else "")
    return f"(same text also in: {shown})"


def _describe(collection) -> str:
    return f"collection '{collection.name}', {collection.count()} vectors, {metric_of(collection)}"


def _filter_text(source_type, tool) -> str:
    parts = [f"source_type={source_type}" if source_type else "", f"tool={tool}" if tool else ""]
    return ", ".join(p for p in parts if p)


def _no_results(filters: str) -> str:
    if filters:
        return "No chunks match that filter. Run `python retrieve.py --list` to see which values exist."
    return "No results."


def show_single(question, embedder, collection, hits, snippet_limit, filters) -> None:
    print(f"\nQuestion : {question!r}")
    print(f"Embedder : {embedder}   ({_describe(collection)})")
    if filters:
        print(f"Filter   : {filters}")
    print()
    if not hits:
        print(f"  {_no_results(filters)}\n")
        return
    for h in hits:
        print(f"  #{h.rank}  {_score(h)}  {_where_text(h)}")
        for line in textwrap.wrap(_snippet(h.text, snippet_limit), width=88):
            print(f"        {line}")
        if h.also_in:
            print(f"        {_also(h)}")
        print()


def show_compare(question, results, filters) -> None:
    print(f"\nQuestion : {question!r}")
    if filters:
        print(f"Filter   : {filters}")
    for embedder, collection, hits, err in results:
        print(f"\n  == {embedder}" + (f"   ({_describe(collection)})" if collection else ""))
        if err:
            print(f"     skipped: {err}")
        elif not hits:
            print(f"     {_no_results(filters)}")
        for h in hits:
            print(f"   #{h.rank}  {_score(h)}  {_where_text(h)}")
            print(f"        {_snippet(h.text, 140)}")
            if h.also_in:
                print(f"        {_also(h)}")
    id_sets = [{h.id for h in hits} for _, _, hits, err in results if not err and hits]
    if len(id_sets) > 1:
        shared = set.intersection(*id_sets)
        print(f"\n  Chunks returned by every model: {len(shared)} of {min(len(s) for s in id_sets)}")
    print()


def _chunks_file_count() -> int | None:
    if not os.path.exists(CHUNKS_FILE):
        return None
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def list_indexes(db_path: str) -> None:
    client = _client(db_path)
    names = _names(client)
    print(f"Database : {os.path.abspath(db_path)}")
    n_chunks = _chunks_file_count()
    if n_chunks is not None:
        print(f"{CHUNKS_FILE} : {n_chunks} chunks")
    if not names:
        print("\nNo collections yet. Build one with build_index.py.")
        return
    by_name = {default_collection_name(m): m for m in dict.fromkeys([DEFAULT_EMBEDDER] + KNOWN_EMBEDDERS)}
    for name in names:
        col = _get(client, name)
        n = col.count()
        types, tools, keys = Counter(), Counter(), Counter()
        offset = 0
        while offset < n:
            res = col.get(include=["metadatas"], limit=5000, offset=offset)
            if not res["ids"]:
                break
            for m in res["metadatas"] or []:
                m = m or {}
                keys.update(m.keys())
                types[str(m.get("source_type", "?"))] += 1
                if m.get("tool"):
                    tools[str(m["tool"])] += 1
            offset += len(res["ids"])
        emb = col.get(limit=1, include=["embeddings"])["embeddings"]
        dim = len(emb[0]) if emb is not None and len(emb) > 0 else "?"
        if n_chunks is None:
            match = ""
        elif n == n_chunks:
            match = "   (same as chunks.jsonl)"
        else:
            match = f"   (chunks.jsonl has {n_chunks}; differs by {abs(n_chunks - n)})"

        print(f"\nCollection '{name}'   <- {by_name.get(name, 'not one of the known embedders')}")
        print(f"    vectors     : {n}{match}")
        print(f"    dimensions  : {dim}")
        print(f"    metric      : {metric_of(col)}")
        print(f"    metadata    : {', '.join(sorted(keys)) or 'none'}")
        print(f"    source_type : {', '.join(f'{k}={v}' for k, v in types.most_common())}")
        if tools:
            top = ", ".join(f"{k}={v}" for k, v in tools.most_common(12))
            print(f"    tools       : {len(tools)} tools tagged. Most chunks: {top}")
    print()


# --- command line ------------------------------------------------------------
def read_questions(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
    except OSError as e:
        sys.exit(f"error: cannot read {path}: {e}")
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Search the Casefile index (Week 5).")
    p.add_argument("question", nargs="?", help="what to search for")
    p.add_argument("-e", "--embedder", action="append",
                   help=f"embedding model to search with (default {DEFAULT_EMBEDDER}); repeat to compare")
    p.add_argument("--compare", action="store_true", help="run the question through every index")
    p.add_argument("--file", help="text file with one question per line")
    p.add_argument("-k", "--top-k", type=int, default=5, help="results per question (default 5)")
    p.add_argument("--source-type", help="only search chunks with this source_type (see --list)")
    p.add_argument("--tool", help="only search chunks tagged with this tool, e.g. nmap (see --list)")
    p.add_argument("--db", default=DEFAULT_DB, help="persistent Chroma directory (default ./chroma_db)")
    p.add_argument("--full", action="store_true", help="print the whole chunk, not a snippet (single-model mode)")
    p.add_argument("--no-dedupe", action="store_true",
                   help="keep chunks whose text is identical (default: show only the best-ranked copy)")
    p.add_argument("--list", action="store_true", help="show what is in the database and exit")
    args = p.parse_args(argv)

    try:
        if args.list:
            list_indexes(args.db)
            return

        questions = ([args.question] if args.question else []) + (read_questions(args.file) if args.file else [])
        if not questions:
            p.error("give a question, or --file FILE, or --list")
        if args.top_k < 1:
            p.error("-k must be at least 1")

        named = args.embedder or []
        compare = args.compare or len(named) > 1
        if compare:
            embedders = named if len(named) > 1 else available_embedders(args.db)
            if not embedders:
                raise LookupError("no index found for any known embedder. Run `python retrieve.py --list`.")
        else:
            embedders = [named[0] if named else DEFAULT_EMBEDDER]

        filters = _filter_text(args.source_type, args.tool)
        dedupe = not args.no_dedupe
        for question in questions:
            if not compare:
                collection = open_collection(embedders[0], args.db)
                hits = _search(collection, embedders[0], question, args.top_k, args.source_type, args.tool, dedupe)
                show_single(question, embedders[0], collection, hits, None if args.full else 400, filters)
                continue
            results = []
            for emb in embedders:
                try:
                    col = open_collection(emb, args.db)
                    hits = _search(col, emb, question, args.top_k, args.source_type, args.tool, dedupe)
                    results.append((emb, col, hits, None))
                except (LookupError, RuntimeError) as e:
                    results.append((emb, None, [], str(e)))
            if all(err for _, _, _, err in results):
                raise RuntimeError("; ".join(err for _, _, _, err in results))
            show_compare(question, results, filters)
    except (LookupError, RuntimeError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
