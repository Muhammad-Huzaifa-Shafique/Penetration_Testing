#!/usr/bin/env python3
"""Read-only report on the Chroma indexes built in Week 4 (Casefile).

Run from the project root, with the venv active that you used for the index build:
    python inspect_index.py | tee week4_index_report.txt
Optional: pass folders to look in instead:  python inspect_index.py path/to/db
It only reads from your databases (no adds, updates or deletes).
"""
import json
import os
import subprocess
import sys
from collections import Counter

SKIP = {"venv", ".venv", "env", ".git", "node_modules", "__pycache__",
        "site-packages", "hacktricks"}
NOTE_EXT = {".py", ".sh", ".md", ".txt", ".pdf", ".docx", ".ipynb", ".toml",
            ".yaml", ".yml", ".cfg"}


def hr(title):
    print("\n==== " + title + " " + "=" * max(4, 66 - len(title)))


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def walk(root, max_depth, skip_hidden=False, extra_skip=()):
    """os.walk with pruning and a depth limit."""
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base
        dirnames[:] = [d for d in dirnames if d not in SKIP and d not in extra_skip
                       and not (skip_hidden and d.startswith("."))]
        if depth >= max_depth:
            dirnames[:] = []
        yield dirpath, filenames


def find_dbs(root, skip_hidden=False):
    return sorted(d for d, files in walk(root, 5, skip_hidden) if "chroma.sqlite3" in files)


def dir_size(path):
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def load_chunk_ids(root):
    path = os.path.join(root, "corpus", "chunks.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line).get("id") for line in f if line.strip()]


def space_of(col):
    meta = col.metadata or {}
    if meta.get("hnsw:space"):
        return meta["hnsw:space"], "from collection metadata"
    cfg = getattr(col, "configuration", None)
    try:
        hnsw = cfg.get("hnsw") if hasattr(cfg, "get") else getattr(cfg, "hnsw", None)
        space = hnsw.get("space") if hasattr(hnsw, "get") else getattr(hnsw, "space", None)
        if space:
            return space, "from collection configuration"
    except Exception:
        pass
    return "l2", "Chroma's default; nothing else was found"


def inspect_collection(client, name, chunk_ids, rows, db):
    try:
        col = client.get_collection(name)
    except Exception:
        col = client.get_collection(name, embedding_function=None)
    n = col.count()
    print(f'\n-- collection "{name}"')
    print(f"   count: {n}")
    peek = col.get(limit=1, include=["documents", "metadatas", "embeddings"])
    emb, docs = peek.get("embeddings"), peek.get("documents")
    dim = len(emb[0]) if emb is not None and len(emb) > 0 else None
    space, why = space_of(col)
    print(f"   vector size: {dim}")
    print(f"   distance metric: {space} ({why})")
    print(f"   raw metadata: {col.metadata}")
    cfg = getattr(col, "configuration", None)
    if cfg is not None:
        print("   raw configuration:", str(cfg)[:300])
    print(f"   chunk text stored: {'yes' if docs and docs[0] else 'NO (no documents saved)'}")

    keys, stypes, tools, no_tool, ids, offset = Counter(), Counter(), Counter(), 0, [], 0
    while offset < n:
        res = col.get(include=["metadatas"], limit=5000, offset=offset)
        if not res["ids"]:
            break
        ids.extend(res["ids"])
        for m in res.get("metadatas") or []:
            m = m or {}
            keys.update(m.keys())
            stypes[str(m.get("source_type"))] += 1
            if m.get("tool"):
                tools[str(m["tool"])] += 1
            else:
                no_tool += 1
        offset += len(res["ids"])
    print("   metadata keys:", ", ".join(f"{k} ({v})" for k, v in keys.most_common()))
    print("   by source_type:", ", ".join(f"{k}={v}" for k, v in stypes.most_common()))
    print(f"   tool tag: {sum(tools.values())} chunks across {len(tools)} tools; none on {no_tool}")
    if tools:
        print("   top tools:", ", ".join(f"{k}={v}" for k, v in tools.most_common(12)))
    if chunk_ids is not None:
        s_idx, s_cor = set(ids), set(chunk_ids)
        both = len(s_idx & s_cor)
        print(f"   vs corpus/chunks.jsonl: {both} of {len(s_cor)} chunk ids are in this index; "
              f"{len(s_idx - s_cor)} index ids are not in the file")
        if both == 0 and ids and chunk_ids:
            print(f"   (no overlap: index uses its own id scheme, e.g. {ids[0]!r} vs chunk id {chunk_ids[0]!r})")
    if docs and docs[0]:
        m = (peek.get("metadatas") or [{}])[0] or {}
        print(f"   sample: {m.get('source')} | {m.get('section')} | {docs[0][:80]!r}")
    rows.append((os.path.relpath(db), name, n, dim, space))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = [os.path.abspath(os.path.expanduser(a)) for a in sys.argv[1:]]
    root = os.getcwd()

    hr("Environment")
    print("python:", sys.version.split()[0])
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
        print("ollama models:\n" + (out.stdout.strip() or out.stderr.strip() or "(no output)"))
    except FileNotFoundError:
        print("ollama: command not found in this shell")
    except Exception as e:
        print(f"ollama: could not run 'ollama list' ({e})")

    hr("Project folder: " + root)
    print("top level:", "  ".join(n + "/" if os.path.isdir(n) else n for n in sorted(os.listdir(root))[:40]))
    found = [os.path.relpath(os.path.join(d, f), root)
             for d, files in walk(root, 2, extra_skip=("corpus",)) for f in files
             if os.path.splitext(f)[1].lower() in NOTE_EXT or "plan" in f.lower() or "week" in f.lower()]
    print("scripts and notes (2 levels deep, corpus/ skipped):")
    for p in sorted(found)[:60]:
        print("  ", p)
    if os.path.isdir("corpus"):
        print("corpus/:", "  ".join(sorted(os.listdir("corpus"))[:20]))

    hr("Chroma")
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("chromadb is not importable here. Activate the venv you used for Week 4 and re-run.")
        return
    print("chromadb:", chromadb.__version__)

    if args:
        dbs = []
        for a in args:
            dbs += [a] if os.path.exists(os.path.join(a, "chroma.sqlite3")) else find_dbs(a)
    else:
        dbs = find_dbs(root)
        if not dbs:
            print("No chroma.sqlite3 under the project folder; searching your home folder...")
            dbs = find_dbs(os.path.expanduser("~"), skip_hidden=True)
    if not dbs:
        print("No Chroma database found. If you know where it is: python inspect_index.py /path/to/it")
        return

    chunk_ids = next((c for c in (load_chunk_ids(r) for r in [root] + args) if c is not None), None)
    print("corpus/chunks.jsonl:", f"{len(chunk_ids)} chunks" if chunk_ids is not None else "not found")
    rows = []
    for db in dbs:
        hr("Chroma DB: " + db)
        print("size on disk:", human(dir_size(db)))
        try:
            client = chromadb.PersistentClient(path=db, settings=Settings(anonymized_telemetry=False))
            names = [getattr(c, "name", c) for c in client.list_collections()]
        except Exception as e:
            print(f"could not open: {type(e).__name__}: {e}")
            continue
        if not names:
            print("no collections in this database")
        for name in names:
            try:
                inspect_collection(client, name, chunk_ids, rows, db)
            except Exception as e:
                print(f'   could not read "{name}": {type(e).__name__}: {e}')

    hr("SUMMARY")
    print(f"{'database':30} {'collection':28} {'count':>7} {'dim':>5}  metric")
    for db, name, n, dim, space in rows:
        print(f"{db[-30:]:30} {name[:28]:28} {n:>7} {str(dim):>5}  {space}")
    print("\nDone. The script only read from your indexes.")


if __name__ == "__main__":
    main()
