#!/usr/bin/env python3
"""chunk_corpus.py -- Casefile capstone, Week 3. Splits corpus/ into
retrieval-ready overlapping chunks with metadata -> corpus/chunks.jsonl."""
import argparse, json, os, re, sys

TARGET_CHARS = 1500
OVERLAP_CHARS = 200
MIN_CHARS = 200
FLOOR_CHARS = 50

HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')
MAN_SECTION_RE = re.compile(r'^([A-Z][A-Z0-9 /,\-]{2,40})\s*$')

def split_into_sections_md(text):
    lines = text.splitlines(); sections = []; cur_title = "(intro)"; cur = []
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if cur: sections.append((cur_title, "\n".join(cur).strip())); cur = []
            cur_title = m.group(2).strip() or "(untitled)"
        else:
            cur.append(line)
    if cur: sections.append((cur_title, "\n".join(cur).strip()))
    return [(t, b) for t, b in sections if b]

def split_into_sections_man(text):
    lines = text.splitlines(); sections = []; cur_title = "(header)"; cur = []
    for line in lines:
        m = MAN_SECTION_RE.match(line)
        if m and len(line) == len(line.rstrip()):
            if cur: sections.append((cur_title, "\n".join(cur).strip())); cur = []
            cur_title = m.group(1).strip()
        else:
            cur.append(line)
    if cur: sections.append((cur_title, "\n".join(cur).strip()))
    return [(t, b) for t, b in sections if b]

def pack(body, target, overlap, min_chars):
    paras = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
    chunks = []; buf = ""
    for p in paras:
        if len(p) > target:
            if buf: chunks.append(buf); buf = ""
            for i in range(0, len(p), target - overlap): chunks.append(p[i:i + target])
            continue
        if buf and len(buf) + 2 + len(p) > target:
            chunks.append(buf); tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n\n" + p).strip()
        else:
            buf = (buf + "\n\n" + p).strip() if buf else p
    if buf: chunks.append(buf)
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        chunks[-2] = (chunks[-2] + "\n\n" + chunks[-1]).strip(); chunks.pop()
    return chunks

def iter_documents(corpus):
    ht = os.path.join(corpus, "hacktricks")
    for root, _, files in os.walk(ht):
        for f in files:
            if f.endswith(".md"): yield os.path.join(root, f), "hacktricks", None
    man = os.path.join(corpus, "manpages")
    if os.path.isdir(man):
        for f in sorted(os.listdir(man)):
            if f.endswith(".txt"): yield os.path.join(man, f), "manpage", f.split(".")[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=int, default=TARGET_CHARS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_CHARS)
    ap.add_argument("--min", type=int, default=MIN_CHARS)
    ap.add_argument("--floor", type=int, default=FLOOR_CHARS)
    args = ap.parse_args()
    if not os.path.isdir(args.corpus):
        sys.exit(f"error: no '{args.corpus}/' here. Run from your project root.")
    out_path = args.out or os.path.join(args.corpus, "chunks.jsonl")
    n_docs = n_chunks = 0; by_type = {"hacktricks": 0, "manpage": 0}; sizes = []
    with open(out_path, "w", encoding="utf-8") as out:
        for path, stype, tool in iter_documents(args.corpus):
            try: text = open(path, encoding="utf-8", errors="replace").read()
            except OSError: continue
            if not text.strip(): continue
            n_docs += 1; rel = os.path.relpath(path, args.corpus)
            sections = (split_into_sections_md(text) if stype == "hacktricks"
                        else split_into_sections_man(text))
            if not sections: sections = [("(document)", text.strip())]
            ci = 0
            for title, body in sections:
                for chunk in pack(body, args.target, args.overlap, args.min):
                    if len(chunk) < args.floor: continue
                    rec = {"id": f"{rel}::{ci}", "source": rel, "source_type": stype,
                           "tool": tool, "section": title, "chunk_index": ci,
                           "n_chars": len(chunk), "text": chunk}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ci += 1; n_chunks += 1; by_type[stype] += 1; sizes.append(len(chunk))
    sizes.sort(); avg = sum(sizes)//len(sizes) if sizes else 0
    med = sizes[len(sizes)//2] if sizes else 0
    print("==> Chunking complete")
    print(f"    documents chunked : {n_docs}")
    print(f"    total chunks      : {n_chunks}")
    print(f"      from hacktricks : {by_type['hacktricks']}")
    print(f"      from man pages  : {by_type['manpage']}")
    print(f"    chunk chars       : min {sizes[0] if sizes else 0} / median {med} / avg {avg} / max {sizes[-1] if sizes else 0}")
    print(f"    output            : {out_path}")

if __name__ == "__main__":
    main()
