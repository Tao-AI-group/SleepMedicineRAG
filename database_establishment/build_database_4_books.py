# -*- coding: utf-8 -*-
"""
Build hybrid RAG index for sleep textbooks with minimal-section splitting:
- For NUMBERED_TITLE_BOOKS, level-1 headings are matched only by '# Chapter <N>' prefix (not full-text).
- Minimal sections: within each Chapter (L1) span, boundaries are L1 itself and any L2/L3 headings
  that are listed in the TOC for that Chapter and actually appear in the Markdown. Boundaries are in
  the order they occur in the Markdown. Extra MD headings not in TOC are ignored for boundaries.
- Each minimal section is then split by token budget with small overlap.
- Outputs: chunks.parquet, bm25.pkl, faiss.index, meta.json
"""

import os
import re
import json
import time
import math
import pickle
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer
import tiktoken


# -------------------- CONFIG --------------------
BOOK_DIR = Path("PATH_TO/RAG_PAPER/all_textbooks_1105")

PURE_TITLE_BOOK = "1-ICSD-3-TR"
NUMBERED_TITLE_BOOKS = [
    "2-Essentials of Sleep Medicine",
    "4-Principles_and_Practice_of_Sleep_Medicine_7th",
    "5-Fundamentals of sleep medicine",
]

OUT_DIR = BOOK_DIR / "rag_index_1105"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# chunking params
TARGET_TOKENS = 512
MIN_TOKENS = 300
MAX_TOKENS = 800
OVERLAP_TOKENS = 80

# embedding model
EMB_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
EMB_BATCH = 128


# -------------------- UTILS --------------------
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[_\u2010\u2011\u2012\u2013\u2014\u2212\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\s:;,\.\|\-–—·]+", "", s)
    s = re.sub(r"[\s:;,\.\|\-–—·]+$", "", s)
    return s

def load_md(md_path: Path) -> str:
    return md_path.read_text(encoding="utf-8", errors="ignore")

def get_md_heading_positions(md_text: str):
    """
    Return (positions, total_len)
    positions: list of {raw, norm, start_char}, for lines starting with '#'
    """
    lines = md_text.splitlines(keepends=True)
    offs = [0]
    for l in lines[:-1]:
        offs.append(offs[-1] + len(l))
    positions = []
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("#"):
            continue
        m = re.match(r"^\s*#\s*(.*?)\s*$", line)
        if not m:
            continue
        raw = m.group(1)
        positions.append({"raw": raw, "norm": normalize_text(raw), "start_char": offs[i]})
    total_len = sum(len(l) for l in lines)
    return positions, total_len

def extract_md_chapter_l1_positions(md_text: str):
    """
    Only match level-1 chapters by '# Chapter <num>' prefix (case-insensitive).
    Returns (chapters, total_len):
      chapters: [{'num': int, 'raw': 'Chapter 1: ...', 'start_char': int}, ...]
    """
    lines = md_text.splitlines(keepends=True)
    offs = [0]
    for l in lines[:-1]:
        offs.append(offs[-1] + len(l))
    pat = re.compile(r"^\s*#\s*chapter\s+(\d+)\b.*$", flags=re.IGNORECASE)
    out = []
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m:
            n = int(m.group(1))
            raw = line.strip()[2:].strip()  # after '# '
            out.append({"num": n, "raw": raw, "start_char": offs[i]})
    out.sort(key=lambda x: x["start_char"])
    return out, sum(len(l) for l in lines)

def parse_toc(json_path: Path):
    """
    Return:
      paths: list of paths, each path is [L1, L2?, L3?]
      title_to_paths: dict norm(title) -> list[path]
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    paths = []
    def rec(obj, prefix):
        if isinstance(obj, dict):
            if not obj and prefix:
                paths.append(prefix)
            for k, v in obj.items():
                rec(v, prefix + [k])
        elif isinstance(obj, list):
            if not obj and prefix:
                paths.append(prefix)
            for x in obj:
                paths.append(prefix + [x])
        else:
            if prefix:
                paths.append(prefix)
    rec(data, [])
    title_to_paths = defaultdict(list)
    for p in paths:
        for t in p:
            title_to_paths[normalize_text(t)].append(p)
    return paths, title_to_paths, data

def parse_toc_levels(json_path: Path):
    """
    Return:
      l1_list: [(raw, norm)] in TOC order
      l1_to_l2: norm(L1) -> [norm(L2), ...]
      l1_to_l3: norm(L1) -> [norm(L3), ...]
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    l1_list = []
    l1_to_l2 = defaultdict(list)
    l1_to_l3 = defaultdict(list)
    for l1_raw, l2_obj in data.items():
        n1 = normalize_text(l1_raw)
        l1_list.append((l1_raw, n1))
        if isinstance(l2_obj, dict):
            for l2_raw, l3_list in l2_obj.items():
                n2 = normalize_text(l2_raw)
                l1_to_l2[n1].append(n2)
                if isinstance(l3_list, list):
                    for l3_raw in l3_list:
                        l1_to_l3[n1].append(normalize_text(l3_raw))
        elif isinstance(l2_obj, list):
            for l2_raw in l2_obj:
                l1_to_l2[n1].append(normalize_text(l2_raw))
    return l1_list, l1_to_l2, l1_to_l3

def parse_toc_l1_chapter_numbers(json_path: Path):
    """
    From L1 keys in TOC, parse 'Chapter <num>'.
    Returns list [(raw_l1, chap_num)], logs warnings for non-matching ones.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    lst = []
    miss = []
    for l1_raw in data.keys():
        m = re.search(r"\bchapter\s+(\d+)\b", l1_raw, flags=re.IGNORECASE)
        if m:
            lst.append((l1_raw, int(m.group(1))))
        else:
            miss.append(l1_raw)
    if miss:
        print(f"[WARN] L1 not matched as 'Chapter N' in TOC ({len(miss)}): {miss[:5]}{' ...' if len(miss)>5 else ''}")
    return lst

def simple_keywords(title: str, text: str, k: int = 8) -> List[str]:
    combo = (title or "") + "\n" + (text or "")
    combo = re.sub(r"[^\w\s\-]", " ", combo)
    tokens = [w.lower() for w in re.split(r"\s+", combo) if w]
    stop = set("""
    the a an and or of for in on to from with without within into by as is are was were be been being it this that these those
    we you they he she i at not no yes can could may might should would will than then thus hence therefore about above below
    """.split())
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    from collections import Counter
    c = Counter(tokens)
    return [w for w, _ in c.most_common(k)]

# tokenization
_enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(s: str) -> int:
    try:
        return len(_enc.encode(s))
    except Exception:
        return max(1, len(s) // 4)

def split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    sents = []
    for p in paragraphs:
        if re.search(r"(^|\n)\s*[\-\*\d]+\.", p) or "|" in p:
            sents.append(p.strip())
            continue
        parts = re.split(r"(?<=[。！？!?。\.])\s+", p.strip())
        for a in parts:
            if a:
                sents.append(a)
    return sents

def sentence_aware_chunk(text: str, title: str, target=TARGET_TOKENS, max_tokens=MAX_TOKENS,
                         min_tokens=MIN_TOKENS, overlap=OVERLAP_TOKENS) -> List[Tuple[str, int, int, int]]:
    if not text.strip():
        return []
    sents = split_sentences(text)
    if not sents:
        return []

    tok_lens = [count_tokens(s) for s in sents]
    pref = [0]
    for t in tok_lens:
        pref.append(pref[-1] + t)
    def window_tokens(i, j):  # [i, j)
        return pref[j] - pref[i]

    chunks = []
    n = len(sents)
    i = 0
    while i < n:
        
        if tok_lens[i] > max_tokens:
            toks = _enc.encode(sents[i])
            pos = 0
            while pos < len(toks):
                sub = toks[pos:pos+max_tokens]
                chunks.append(_enc.decode(sub))
                pos += max_tokens - overlap if overlap > 0 else max_tokens
            i += 1
            continue

        
        lo, hi = i, i
        while hi < n and window_tokens(i, hi+1) <= max_tokens:
            hi += 1
            if window_tokens(i, hi) >= target:
                break
        if hi == i:
            hi = min(i+1, n)

        ch_txt = "\n".join(sents[i:hi]).strip()
        chunks.append(ch_txt)

        if overlap > 0:
            k = hi
            while k > i and window_tokens(k, hi) < overlap:
                k -= 1
            i = k
        else:
            i = hi

    
    merged = []
    buf = None
    for ch in chunks:
        if buf is None:
            buf = ch
            continue
        if count_tokens(buf) < min_tokens:
            comb = buf + "\n" + ch
            if count_tokens(comb) <= int(max_tokens * 1.2):
                buf = comb
            else:
                merged.append(buf); buf = ch
        else:
            merged.append(buf); buf = ch
    if buf:
        merged.append(buf)

    out = []
    st = 0
    for ch in merged:
        tl = count_tokens(ch)
        out.append((ch, st, st + tl, tl))
        st += max(0, tl - overlap) if overlap > 0 else tl
    return out

# -------------------- MINIMAL SECTION BUILDER --------------------
def build_minimal_sections(book_name: str) -> List[Dict]:
    md_path = BOOK_DIR / f"{book_name}.md"
    json_path = BOOK_DIR / f"{book_name}.json"
    assert md_path.exists() and json_path.exists(), f"Missing files for {book_name}"

    md_text = load_md(md_path)
   
    all_md_heads, total_len = get_md_heading_positions(md_text)

    
    from collections import defaultdict
    norm2pos = defaultdict(list)
    for h in sorted(all_md_heads, key=lambda x: x["start_char"]):
        norm2pos[h["norm"]].append(h)

    l1_list, l1_to_l2, l1_to_l3 = parse_toc_levels(json_path)

    sections = []

    if book_name in NUMBERED_TITLE_BOOKS:
        
        md_chapters, _ = extract_md_chapter_l1_positions(md_text)
        num2md = {c["num"]: c for c in md_chapters}
        l1_chaps = parse_toc_l1_chapter_numbers(json_path)

        l1_occ = []
        for raw1, num in l1_chaps:
            if num in num2md:
                l1_occ.append((raw1, num2md[num]["raw"], num2md[num]["start_char"], normalize_text(raw1)))
            else:
                log(f"[WARN] L1 '{raw1}' (num={num}) not found in MD by '# Chapter {num}'")
        l1_occ.sort(key=lambda x: x[2])

        
        l1_spans = []
        for i, (raw1, md_raw1, s, norm1) in enumerate(l1_occ):
            e = l1_occ[i+1][2] if i+1 < len(l1_occ) else total_len
            l1_spans.append((raw1, md_raw1, norm1, s, e))

        
        for raw1, md_raw1, n1, s, e in l1_spans:
            cand_norms = set(l1_to_l2.get(n1, []) + l1_to_l3.get(n1, []))

            local_boundaries = [{"raw": md_raw1, "breadcrumb": [raw1], "start_char": s}]
            for nm in cand_norms:
                for h in norm2pos.get(nm, []):
                    if s < h["start_char"] < e:
                        local_boundaries.append({
                            "raw": h["raw"],
                            "breadcrumb": [raw1, h["raw"]],
                            "start_char": h["start_char"],
                        })

            local_boundaries.sort(key=lambda x: x["start_char"])
            for i in range(len(local_boundaries)):
                bs = local_boundaries[i]["start_char"]
                be = local_boundaries[i+1]["start_char"] if i+1 < len(local_boundaries) else e
                if be <= bs:
                    continue
                sections.append({
                    "book_id": book_name,
                    "heading_text": local_boundaries[i]["raw"],
                    "breadcrumb": local_boundaries[i]["breadcrumb"],
                    "breadcrumb_str": " > ".join(local_boundaries[i]["breadcrumb"]),
                    "start_char": int(bs),
                    "end_char": int(be),
                })
        return sections

    # ------- PURE_TITLE_BOOK（1-ICSD-3-TR）--------
    paths, title_to_paths, toc_obj = parse_toc(json_path)
    toc_norm_titles = set(normalize_text(t) for t in title_to_paths.keys())
    log(f"[ICSD] MD headings: {len(all_md_heads)}; TOC titles: {len(toc_norm_titles)}")

    assigned = []
    for h in all_md_heads:
        if h["norm"] in toc_norm_titles:
            bc = title_to_paths[h["norm"]][0]
            assigned.append({"raw": h["raw"], "breadcrumb": bc, "start_char": h["start_char"]})

    log(f"[ICSD] Assigned (MD∩TOC): {len(assigned)}")
    if not assigned:
        
        log("[ICSD][FALLBACK] No TOC-matched headings; using all MD headings as boundaries.")
        assigned = [{"raw": h["raw"], "breadcrumb": [h["raw"]], "start_char": h["start_char"]}
                    for h in all_md_heads]

    assigned.sort(key=lambda x: x["start_char"])
    for i in range(len(assigned)):
        s = assigned[i]["start_char"]
        e = assigned[i+1]["start_char"] if i+1 < len(assigned) else total_len
        if e <= s:
            continue
        bc = assigned[i]["breadcrumb"]
        sections.append({
            "book_id": book_name,
            "heading_text": assigned[i]["raw"],
            "breadcrumb": bc,
            "breadcrumb_str": " > ".join(bc if isinstance(bc, list) else [str(bc)]),
            "start_char": int(s),
            "end_char": int(e),
        })
    log(f"[ICSD] Minimal sections built: {len(sections)}")
    return sections

def token_window_chunk(text: str, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    if not text.strip():
        return []
    toks = _enc.encode(text)
    n = len(toks)
    if n == 0:
        return []
    step = max(1, max_tokens - overlap if overlap > 0 else max_tokens)
    chunks = []
    pos = 0
    while pos < n:
        sub = toks[pos:pos+max_tokens]
        chunks.append(_enc.decode(sub))
        pos += step
    out = []
    st = 0
    for ch in chunks:
        tl = count_tokens(ch)
        out.append((ch, st, st + tl, tl))
        st += max(0, tl - overlap) if overlap > 0 else tl
    return out


def robust_chunk_section(body: str, heading: str):
   
    punct = len(re.findall(r"[。！？!?\.]", body))
    density = punct / max(1, len(body))
    BIG_CHARS = 12000  # > 12k 字符走快速通道；你可按机器情况调

    if len(body) > BIG_CHARS or density < 0.0015:
        
        return token_window_chunk(body, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS)
    else:
        
        return sentence_aware_chunk(
            body, heading,
            target=TARGET_TOKENS, max_tokens=MAX_TOKENS,
            min_tokens=MIN_TOKENS, overlap=OVERLAP_TOKENS
        )

def char_based_chunk(
    text: str,
    target_chars: int = 2000,   
    max_chars: int = 3000,      
    overlap_chars: int = 400   
) -> List[Tuple[str, int, int, int]]:
    """
    完全基于字符的快速切分。
    不使用 tiktoken，也不做句子识别。
    返回 (chunk_text, start_pos, end_pos, pseudo_token_len)
    """
    text = text or ""
    n = len(text)
    if n == 0:
        return []

    step = max_chars - overlap_chars if overlap_chars > 0 else max_chars
    chunks = []
    pos = 0
    pseudo_pos = 0
    while pos < n:
        sub = text[pos:pos + max_chars]
        tlen = len(sub) // 4  
        chunks.append((sub, pseudo_pos, pseudo_pos + tlen, tlen))
        pseudo_pos += tlen
        pos += step
    return chunks


# -------------------- BOOK -> CHUNKS --------------------
def process_book(book_name: str) -> pd.DataFrame:
    md_path = BOOK_DIR / f"{book_name}.md"
    if not md_path.exists() or not (BOOK_DIR / f"{book_name}.json").exists():
        log(f"[WARN] Missing files for {book_name}")
        return pd.DataFrame()

    log(f"[Process] Loading MD/TOC for {book_name} ...")
    md_text = load_md(md_path)

    log(f"[Process] Building minimal sections for {book_name} ...")
    min_sections = build_minimal_sections(book_name)
    log(f"[Process] Sections ready: {len(min_sections)} for {book_name}")
    if not min_sections:
        log(f"[WARN] No minimal sections for {book_name}")
        return pd.DataFrame()

    records = []
    for si, sec in enumerate(tqdm(min_sections, desc=f"{book_name} sections", unit="sec", mininterval=0.4)):
        s, e = sec["start_char"], sec["end_char"]
        heading = sec["heading_text"]
        breadcrumb = sec["breadcrumb"]
        breadcrumb_str = sec["breadcrumb_str"]
        parent_id = f"{book_name}::{breadcrumb_str}"

        body = md_text[s:e]
        body = re.sub(r"^\s*#\s*.*?$", "", body, flags=re.MULTILINE).strip()

        if si < 5 or (si % 50 == 0):
            log(f"[Chunk] Section {si+1}/{len(min_sections)} len={len(body)} heading='{heading[:60]}'")

        chunks = char_based_chunk(body, target_chars=2000, max_chars=3000, overlap_chars=400)

        for ci, (ch_text, t0, t1, tlen) in enumerate(chunks):
            records.append(dict(
                book_id=book_name,
                heading_text=heading,
                breadcrumb=breadcrumb,
                breadcrumb_str=breadcrumb_str,
                parent_section_id=parent_id,
                chunk_index_in_section=ci,
                content=ch_text,
                chunk_token_len=int(tlen), 
                start_char=s, end_char=e,
                keywords=simple_keywords(heading, ch_text),
                source_path=str(md_path),
            ))

    df = pd.DataFrame.from_records(records)
    df["chunk_global_index"] = np.arange(len(df))
    cols = [
        "chunk_global_index", "book_id", "breadcrumb_str", "breadcrumb",
        "heading_text", "parent_section_id", "chunk_index_in_section",
        "chunk_token_len", "content", "keywords",
        "source_path", "start_char", "end_char",
    ]
    return df[cols]


# -------------------- INDEX BUILDERS --------------------
def build_bm25_index(texts: List[str]):
    def tokfun(s):
        s = s.lower()
        s = re.sub(r"[^\w\s\-]", " ", s)
        return [w for w in s.split() if w]
    corpus_tokens = []
    for t in tqdm(texts, desc="BM25 tokenizing", unit="doc"):
        corpus_tokens.append(tokfun(t))
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, corpus_tokens

def embed_corpus(model: SentenceTransformer, texts: List[str], batch=EMB_BATCH) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs, dtype="float32")

def build_faiss_index(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # normalized -> cosine
    index.add(embs)
    return index


# -------------------- MAIN --------------------
def main():
    all_books = [PURE_TITLE_BOOK] + NUMBERED_TITLE_BOOKS
    dfs = []

    log(f"[Build] Processing {len(all_books)} books ...")
    for b in tqdm(all_books, desc="Books", unit="book"):
        log(f"[Build] Processing book: {b}")
        dfb = process_book(b)
        if not dfb.empty:
            dfs.append(dfb)

    if not dfs:
        print("[ERROR] No data built.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Compose retrieval text
    def retr_str(row):
        head = row["heading_text"] or ""
        bc = row["breadcrumb_str"] or ""
        kw = " ".join(row["keywords"] or [])
        first_para = row["content"].split("\n\n")[0] if row["content"] else ""
        return f"{head}\n{bc}\n{kw}\n\n{row['content']}\n\n{first_para}"

    log("[Build] Composing retrieval_text ...")
    df["retrieval_text"] = df.progress_apply(retr_str, axis=1)

    # Save chunks
    chunks_path = OUT_DIR / "chunks.parquet"
    df.to_parquet(chunks_path, index=False)
    log(f"[Save] chunks -> {chunks_path}  (rows={len(df)})")

    # BM25
    log("[Index] Building BM25 ...")
    bm25, corpus_tokens = build_bm25_index(df["retrieval_text"].tolist())
    with open(OUT_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "corpus_tokens": corpus_tokens}, f)
    log(f"[Save] bm25 -> {OUT_DIR/'bm25.pkl'}")

    # Dense + FAISS
    log("[Index] Building Dense (SentenceTransformer + FAISS) ...")
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    embs = embed_corpus(emb_model, df["retrieval_text"].tolist(), batch=EMB_BATCH)

    log("[Index] Building FAISS index ...")
    faiss_index = build_faiss_index(embs)
    faiss.write_index(faiss_index, str(OUT_DIR / "faiss.index"))

    meta = {
        "embedding_model": EMB_MODEL_NAME,
        "vector_dim": int(embs.shape[1]),
        "normalize": True,
        "index_type": "IndexFlatIP",
        "books": [PURE_TITLE_BOOK] + NUMBERED_TITLE_BOOKS,
        "params": dict(
            target_tokens=TARGET_TOKENS,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
        )
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[Save] faiss -> {OUT_DIR/'faiss.index'} ; meta -> {OUT_DIR/'meta.json'}")
    log("[Done] Hybrid index is ready.")


# -------------------- OPTIONAL: Hybrid search helpers --------------------
def rrf_fuse(ranklists: List[List[Tuple[int, float]]], k: float = 60.0, topn: int = 20) -> List[Tuple[int, float]]:
    scores = defaultdict(float)
    for rl in ranklists:
        for r, (idx, _) in enumerate(rl):
            scores[idx] += 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]

def search_hybrid(query: str, topn: int = 12):
    chunks_path = OUT_DIR / "chunks.parquet"
    df = pd.read_parquet(chunks_path)

    with open(OUT_DIR / "bm25.pkl", "rb") as f:
        bm25_pack = pickle.load(f)
    bm25 = bm25_pack["bm25"]

    faiss_index = faiss.read_index(str(OUT_DIR / "faiss.index"))
    meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    emb_model = SentenceTransformer(meta["embedding_model"])

    def tokfun(s):
        s = s.lower()
        s = re.sub(r"[^\w\s\-]", " ", s)
        return [w for w in s.split() if w]

    q_tokens = tokfun(query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top = np.argsort(-bm25_scores)[:100]
    bm25_list = [(int(i), float(bm25_scores[i])) for i in bm25_top]

    q_emb = emb_model.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(np.asarray(q_emb, dtype="float32"), 100)
    dense_list = [(int(i), float(d)) for i, d in zip(I[0], D[0])]

    fused = rrf_fuse([bm25_list, dense_list], k=60, topn=topn)
    out = []
    for idx, s in fused:
        row = df.iloc[idx]
        out.append(dict(
            score=s,
            chunk_id=int(row["chunk_global_index"]),
            book=row["book_id"],
            breadcrumb=row["breadcrumb_str"],
            heading=row["heading_text"],
            preview=row["content"][:300].replace("\n", " "),
        ))
    return out


if __name__ == "__main__":
    print(search_hybrid("diagnostic criteria for insomnia disorder"))
