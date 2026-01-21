# -*- coding: utf-8 -*-
"""
Build hybrid RAG index for sleep textbooks (VLM markdowns) with fixed-window splitting.

Data layout:
- ROOT_DIR = PATH_TO/sleep/RAG/text_books/extracted_content
  - <book_dir_1>/
      vlm/
        *.md   <-- use these markdown files
  - <book_dir_2>/
      vlm/
        *.md
  - ...

Special rules:
1) Ignore this folder completely (skip any markdown inside it):
   PATH_TO/sleep/RAG/text_books/extracted_content/
       3-Competencies in Sleep Medicine An Assessment Guide

2) Do NOT use TOC. We treat each markdown as a whole document and split it
   with a fixed-size window (char-based), without minimal sections.

Outputs:
- chunks.parquet  (per-chunk metadata + content + retrieval_text)
- bm25.pkl        (BM25Okapi index + corpus tokens)
- faiss.index     (dense index on retrieval_text)
- meta.json       (index metadata)
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
ROOT_DIR = Path(
    "PATH_TO/sleep/RAG/text_books/extracted_content"
)

OUT_DIR = Path("PATH_TO/RAG_PAPER/all_textbooks_1105/rag_index_1105_unleaned_textbooks")
OUT_DIR.mkdir(exist_ok=True, parents=True)

TARGET_TOKENS = 512
MIN_TOKENS = 300
MAX_TOKENS = 800
OVERLAP_TOKENS = 80


CHAR_TARGET = 2000      # ~ 500 tokens
CHAR_MAX = 3000         # ~ 750 tokens
CHAR_OVERLAP = 400      # ~ 80 tokens

# embedding model
EMB_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
EMB_BATCH = 128


EXCLUDE_DIR_NAME = "3-Competencies in Sleep Medicine An Assessment Guide"


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


def simple_keywords(title: str, text: str, k: int = 8) -> List[str]:
    combo = (title or "") + "\n" + (text or "")
    combo = re.sub(r"[^\w\s\-]", " ", combo)
    tokens = [w.lower() for w in re.split(r"\s+", combo) if w]
    stop = set(
        """
        the a an and or of for in on to from with without within into by as is are was were be been being it this that these those
        we you they he she i at not no yes can could may might should would will than then thus hence therefore about above below
        """.split()
    )
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    c = Counter(tokens)
    return [w for w, _ in c.most_common(k)]


# ---- tokenization for statistics only ----
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(s: str) -> int:
    try:
        return len(_enc.encode(s))
    except Exception:
        return max(1, len(s) // 4)


def char_based_chunk(
    text: str,
    target_chars: int = CHAR_TARGET,
    max_chars: int = CHAR_MAX,
    overlap_chars: int = CHAR_OVERLAP,
) -> List[Tuple[str, int, int, int]]:
    """
    完全基于字符的快速切分。
    不使用句子/TOC，固定窗口划分。
    返回 (chunk_text, pseudo_token_start, pseudo_token_end, pseudo_token_len)
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
        sub = text[pos : pos + max_chars]
        tlen = len(sub) // 4  
        chunks.append((sub, pseudo_pos, pseudo_pos + tlen, tlen))
        pseudo_pos += tlen
        pos += step
    return chunks


# -------------------- DOC -> CHUNKS --------------------
def find_vlm_markdown_files(root_dir: Path) -> List[Tuple[str, Path, Path]]:

    docs = []
    for book_dir in sorted(root_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        if book_dir.name == EXCLUDE_DIR_NAME:
            log(f"[Skip] Excluding dir: {book_dir}")
            continue

        vlm_dir = book_dir / "vlm"
        if not vlm_dir.exists() or not vlm_dir.is_dir():
            log(f"[WARN] No 'vlm' subfolder in {book_dir}")
            continue

        md_files = sorted(vlm_dir.glob("*.md"))
        if not md_files:
            log(f"[WARN] No markdown in {vlm_dir}")
            continue

        if len(md_files) > 1:
            log(
                f"[INFO] Multiple markdown files in {vlm_dir}, using first one: {md_files[0].name}"
            )

        md_path = md_files[0]
        doc_id = book_dir.name  
        docs.append((doc_id, book_dir, md_path))

    log(f"[Scan] Found {len(docs)} markdown docs (after exclusion).")
    return docs


def extract_doc_title(md_text: str, fallback: str) -> str:

    for line in md_text.splitlines():
        if line.strip().startswith("#"):
            # strip leading '#'
            t = re.sub(r"^\s*#+\s*", "", line).strip()
            if t:
                return t
    return fallback


def process_doc(doc_id: str, book_dir: Path, md_path: Path) -> pd.DataFrame:

    log(f"[Process] doc_id={doc_id}  md={md_path}")
    md_text = load_md(md_path)
    total_len = len(md_text)
    log(f"[Process]  length={total_len} chars")

    if not md_text.strip():
        log(f"[WARN] Empty markdown: {md_path}")
        return pd.DataFrame()

    doc_title = extract_doc_title(md_text, fallback=doc_id)
    breadcrumb = [doc_title]
    breadcrumb_str = " > ".join(breadcrumb)
    parent_id = f"{doc_id}::{breadcrumb_str}"

   
    chunks = char_based_chunk(
        md_text, target_chars=CHAR_TARGET, max_chars=CHAR_MAX, overlap_chars=CHAR_OVERLAP
    )

    records = []
    start_char = 0
    end_char = total_len

    for ci, (ch_text, t0, t1, tlen) in enumerate(chunks):
        records.append(
            dict(
                book_id=doc_id, 
                heading_text=doc_title,
                breadcrumb=breadcrumb,
                breadcrumb_str=breadcrumb_str,
                parent_section_id=parent_id,
                chunk_index_in_section=ci,
                content=ch_text,
                chunk_token_len=int(tlen), 
                start_char=start_char,
                end_char=end_char,
                keywords=simple_keywords(doc_title, ch_text),
                source_path=str(md_path),
            )
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["chunk_global_index"] = np.arange(len(df))
    cols = [
        "chunk_global_index",
        "book_id",
        "breadcrumb_str",
        "breadcrumb",
        "heading_text",
        "parent_section_id",
        "chunk_index_in_section",
        "chunk_token_len",
        "content",
        "keywords",
        "source_path",
        "start_char",
        "end_char",
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


def embed_corpus(
    model: SentenceTransformer, texts: List[str], batch: int = EMB_BATCH
) -> np.ndarray:
    embs = model.encode(
        texts, batch_size=batch, show_progress_bar=True, normalize_embeddings=True
    )
    return np.asarray(embs, dtype="float32")


def build_faiss_index(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # normalized -> cosine
    index.add(embs)
    return index


# -------------------- HYBRID SEARCH HELPERS --------------------
def rrf_fuse(
    ranklists: List[List[Tuple[int, float]]], k: float = 60.0, topn: int = 20
) -> List[Tuple[int, float]]:
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

    # BM25
    q_tokens = tokfun(query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top = np.argsort(-bm25_scores)[:100]
    bm25_list = [(int(i), float(bm25_scores[i])) for i in bm25_top]

    # Dense
    q_emb = emb_model.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(np.asarray(q_emb, dtype="float32"), 100)
    dense_list = [(int(i), float(d)) for i, d in zip(I[0], D[0])]

    # RRF fuse
    fused = rrf_fuse([bm25_list, dense_list], k=60, topn=topn)
    out = []
    for idx, s in fused:
        row = df.iloc[idx]
        out.append(
            dict(
                score=s,
                chunk_id=int(row["chunk_global_index"]),
                book=row["book_id"],
                breadcrumb=row["breadcrumb_str"],
                heading=row["heading_text"],
                preview=row["content"][:300].replace("\n", " "),
            )
        )
    return out


# -------------------- MAIN --------------------
def main():
    log(f"[Build] Scanning markdown docs under: {ROOT_DIR}")
    doc_list = find_vlm_markdown_files(ROOT_DIR)
    if not doc_list:
        log("[ERROR] No markdown docs found. Abort.")
        return

    dfs = []
    log(f"[Build] Processing {len(doc_list)} docs ...")
    for doc_id, book_dir, md_path in tqdm(doc_list, desc="Docs", unit="doc"):
        df_doc = process_doc(doc_id, book_dir, md_path)
        if not df_doc.empty:
            dfs.append(df_doc)

    if not dfs:
        log("[ERROR] No data built (all docs empty?). Abort.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df["chunk_global_index"] = np.arange(len(df))

    # Compose retrieval text
    def retr_str(row):
        head = row["heading_text"] or ""
        bc = row["breadcrumb_str"] or ""
        kw = " ".join(row["keywords"] or [])
        first_para = (
            row["content"].split("\n\n")[0] if isinstance(row["content"], str) else ""
        )
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
        "root_dir": str(ROOT_DIR),
        "exclude_dir": EXCLUDE_DIR_NAME,
        "params": dict(
            target_tokens=TARGET_TOKENS,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
            char_target=CHAR_TARGET,
            char_max=CHAR_MAX,
            char_overlap=CHAR_OVERLAP,
        ),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"[Save] faiss -> {OUT_DIR/'faiss.index'} ; meta -> {OUT_DIR/'meta.json'}")
    log("[Done] Hybrid VLM index is ready.")


if __name__ == "__main__":

    main()

    print(search_hybrid("diagnostic criteria for insomnia disorder"))
