import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


CASE_DATA = "PATH_TO/RAG_PAPER/dataset/vignette_cases_structured.json"


KB_CONFIG: Dict[str, Dict[str, Any]] = {
    "TOC_hybrid": {
        "rag_dir": "PATH_TO/RAG_PAPER/all_textbooks_1105/rag_index_1105",
        "retriever": "hybrid",  # BM25 + FAISS
    },
    "TOC_single": {
        "rag_dir": "PATH_TO/RAG_PAPER/all_textbooks_1105/rag_index_1105_dense_only",
        "retriever": "dense",  # FAISS only
    },
    "cleaned_hybrid": {
        "rag_dir": "PATH_TO/RAG_PAPER/all_textbooks_1105/rag_index_1113_textbooks_simple",
        "retriever": "hybrid",
    },
    "cleaned_single": {
        "rag_dir": "PATH_TO/RAG_PAPER/all_textbooks_1105/rag_index_1113_textbooks_dense_only_simple",
        "retriever": "dense",
    },
    "uncleaned_hybrid": {
        "rag_dir": "PATH_TO/RAG_PAPER/all_textbooks_1105/rag_index_1105_unleaned_textbooks",
        "retriever": "hybrid",
    },
    "uncleaned_single": {
        "rag_dir": "PATH_TO/RAG_PAPER/all_textbooks_1105/rag_index_1105_unleaned_textbooks_dense_only",
        "retriever": "dense",
    },
}

TOP_K = 16

OUT_ROOT = "PATH_TO/RAG_PAPER/exp_code_Nov/case/results"

os.makedirs(OUT_ROOT, exist_ok=True)


user_prompt_template_with_rag_case = """You are given the following information:

--- RETRIEVED CONTEXT ---
{retrieved_context}
---

--- CLINICAL VIGNETTE ---
{CASE_TEXT}
---

Task:
Return EXACTLY one JSON object with the schema:
{{"top_diagnoses": ["<dx1>", "<dx2>", "<dx3>", "<dx4>", "<dx5>"]}}

Rules:
1) Provide all the possible diagnoses, sorted by likelihood (most likely first).
2) Use short labels (ICSD-3 preferred terms when applicable).
3) If fewer than 5 diagnoses are appropriate, you may output 3 or 4, but preserve the order by likelihood.
4) No extra commentary before or after the JSON.
"""

SYSTEM_INSTRUCTION = (
    "You are a Sleep Medicine board-exam assistant. "
    "Follow ICSD-3 and AASM guidelines. "
    "Given a clinical vignette and retrieved textbook context, "
    "return EXACTLY one JSON object with the schema:\n"
    "{\"top_diagnoses\": [\"<dx1>\", \"<dx2>\", \"<dx3>\", \"<dx4>\", \"<dx5>\"]}\n"
    "No extra text."
)

# ---------------- Models ----------------
MODELS: Dict[str, str] = {
    "MedGemma-27B-it": "/dgx1data/aii/tao/m327768/model/google/medgemma-27b-text-it/models--google--medgemma-27b-text-it/snapshots/6b08c481126ff65a9b8fa5ab4d691b152b8edb5d",
    "MedGemma-4B-it":  "/dgx1data/aii/tao/m327768/model/google/medgemma-4b-it/models--google--medgemma-4b-it/snapshots/698f7911b8e0569ff4ebac5d5552f02a9553063c",
    "LLaMA-70B":       "/dgx1data/aii/tao/tools/models/meta-llama/Llama-3.3-70B-Instruct",
    "LLaMA-8B":        "/dgx1data/aii/tao/m327768/model/meta-llama/Llama-3.1-8B-Instruct/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    "Qwen-14B":        "/dgx1data/aii/tao/m327768/model/Qwen/Qwen3-14B/models--Qwen--Qwen3-14B/snapshots/8268fe3026cb304910457689366670e803a6fd56",
    "Qwen-235B":       "/dgx1data/aii/tao/m327768/model/Qwen/Qwen3-235B-A22B-Instruct-2507/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/8b0a01458495f9e2c57148603702a8057b4a265d",
}


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "CASE_DATA must be a JSON list"
    return data

def safe_get_str(x: Dict[str, Any], k: str) -> str:
    v = x.get(k, "")
    return v if isinstance(v, str) else ""

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def rrf(scores_lists: List[List[Tuple[int, float]]], k: int = 60) -> Dict[int, float]:
    agg: Dict[int, float] = {}
    for lst in scores_lists:
        for rank, (idx, _) in enumerate(lst):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return agg

def load_embedding_model(model_name: str) -> SentenceTransformer:

    return SentenceTransformer(model_name, device="cuda")

def encode_query_embedding(q: str, embedder: SentenceTransformer) -> np.ndarray:
    v = embedder.encode(q, convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")

def safe_name(s: str) -> str:
    s = s.rstrip("/\\")
    base = os.path.basename(s)
    return base or s

def truncate_chars(text: str, max_len: int) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= max_len else s[:max_len]

def maybe_add_no_think(model_name: str, content: str) -> str:

    return (content + " /no_think") if ("qwen" in model_name.lower()) else content


def retrieve_hybrid(q: str,
                    embedder,
                    index,
                    kb_records: List[Dict[str, Any]],
                    bm25: BM25Okapi,
                    top_k: int = 16):
    # BM25
    q_tokens = tokenize(q)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top = sorted(list(enumerate(bm25_scores)), key=lambda x: x[1], reverse=True)[:50]

    # FAISS（IndexFlatIP）
    qv = encode_query_embedding(q, embedder)
    D, I = index.search(qv.reshape(1, -1), 50)
    dense_top = list(zip(I[0].tolist(), D[0].tolist()))

    # RRF 
    fused = rrf([bm25_top, dense_top])
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "text": kb_records[i]["content"],
            "meta": kb_records[i],
            "score": float(s),
            "kb_idx": int(i),
        }
        for i, s in ranked
    ]

def retrieve_dense(q: str,
                   embedder,
                   index,
                   kb_records: List[Dict[str, Any]],
                   top_k: int = 16):
    qv = encode_query_embedding(q, embedder)
    D, I = index.search(qv.reshape(1, -1), top_k)
    ranked = list(zip(I[0].tolist(), D[0].tolist()))
    return [
        {
            "text": kb_records[i]["content"],
            "meta": kb_records[i],
            "score": float(s),
            "kb_idx": int(i),
        }
        for i, s in ranked
    ]

# ========= vLLM Chat / JSON =========
def _chat_once(llm: LLM, system: str, user: str, sp: SamplingParams, model_name: str) -> str:
    msg_user = maybe_add_no_think(model_name, user)
    outputs = llm.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": msg_user},
        ],
        sampling_params=sp,
    )
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    return text.replace("<think>", "").replace("</think>", "").strip()

def _extract_json_obj(s: str) -> Optional[dict]:
    if not s:
        return None
    s = re.sub(r"^```(json)?", "", s.strip(), flags=re.I).strip()
    s = re.sub(r"```$", "", s.strip())
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    end_idx = -1
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx == -1:
        return None
    candidate = s[start:end_idx]
    try:
        return json.loads(candidate)
    except Exception:
        cand2 = candidate.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        try:
            return json.loads(cand2)
        except Exception:
            return None

def run_chat_json(
    llm: LLM,
    prompt: str,
    sp_json: SamplingParams,
    model_name: str,
    system: str = "Return valid JSON only.",
) -> Tuple[Optional[dict], str]:
    raw = _chat_once(llm, system, prompt, sp_json, model_name)
    js = _extract_json_obj(raw)
    return js, raw


def llm_rewrite_query(vignette: str,
                      llm: LLM,
                      sp_json: SamplingParams,
                      model_name: str) -> Tuple[str, str]:
    prompt = f"""Query Rewriting for Retrieval:

Clinical vignette:
{vignette}

Please reformulate the above vignette into a concise retrieval query using precise medical terminology.
Focus on the core sleep-related disorders, key symptoms, and comorbidities.
Be concise (<=30 words).

Return JSON only: {{"rewritten":"<rewritten query>"}}"""
    js, out = run_chat_json(
        llm,
        prompt,
        sp_json,
        model_name,
        system="You are a concise assistant that returns valid JSON only.",
    )
    rewritten = (js or {}).get("rewritten", "").strip() if js else ""
    if not rewritten:
        # 回退：简单清洗 vignette
        rewritten = re.sub(r"\s+", " ", vignette).strip()
    return rewritten, out

def llm_screen(passages: List[str],
               q: str,
               llm: LLM,
               sp_json: SamplingParams,
               model_name: str) -> Tuple[List[str], List[int], str]:
    short_psgs = [truncate_chars(p, 1800) for p in passages]
    joined = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(short_psgs)])
    prompt = f"""Passage Screening (merge relevance & usefulness):

Query (summarized vignette):
{q}

Passages:
{joined}

For each passage, label exactly one of:
- Irrelevant: off-topic or generic noise.
- Relevant-Context: on-topic background but not directly answer-enabling.
- Evidential: contains criteria, thresholds, rules, or facts that directly support diagnosis.

Return JSON only with two arrays of equal length = {len(passages)}:
{{
  "labels": ["Evidential"|"Relevant-Context"|"Irrelevant", ...],
  "scores": [2|1|0, ...]
}}
No explanations."""
    js, out = run_chat_json(
        llm,
        prompt,
        sp_json,
        model_name,
        system="Return valid JSON only.",
    )
    labels: List[str] = []
    scores: List[int] = []
    if js:
        labels = js.get("labels", []) or []
        scores = js.get("scores", []) or []
    if len(labels) != len(passages) or len(scores) != len(passages):
        labels = ["Relevant-Context"] * len(passages)
        scores = [1] * len(passages)

    norm = {"evidential": "Evidential", "relevant-context": "Relevant-Context", "irrelevant": "Irrelevant"}
    labels = [norm.get(str(l).strip().lower(), "Relevant-Context") for l in labels]
    scores = [
        int(s) if str(s).isdigit() else (2 if lb == "Evidential" else 1 if lb == "Relevant-Context" else 0)
        for s, lb in zip(scores, labels)
    ]
    return labels, scores, out


def build_case_core_text(case_obj: Dict[str, Any]) -> str:

    parts = []
    cid = case_obj.get("id", None)
    if cid is not None:
        parts.append(f"Case ID: {cid}")
    h = safe_get_str(case_obj, "history")
    p = safe_get_str(case_obj, "physical_examination")
    t = safe_get_str(case_obj, "testing")

    if h:
        parts.append("\nHistory:\n" + h.strip())
    if p:
        parts.append("\nPhysical Examination:\n" + p.strip())
    if t:
        parts.append("\nTesting:\n" + t.strip())

    return "\n".join(parts).strip()

def build_case_prompt_with_context(case_obj: Dict[str, Any],
                                   context_text: str) -> str:
    core = build_case_core_text(case_obj)
    if not context_text:
        ctx = "[No additional retrieved context.]"
    else:
        ctx = context_text
    return user_prompt_template_with_rag_case.format(
        retrieved_context=ctx,
        CASE_TEXT=core,
    )

JSON_TOP_PAT = re.compile(
    r'\{[^{}]*"top_diagnoses"\s*:\s*\[[^\]]*\][^{}]*\}',
    re.IGNORECASE | re.DOTALL,
)

def _clean_detoken(s: str) -> str:
    # Remove stray think tags or formatting noise
    return s.replace("<think>", "").replace("</think>", "").strip()

def parse_top_diagnoses(text: str) -> Optional[List[str]]:
    text = _clean_detoken(text)

    # Case 1: direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "top_diagnoses" in obj and isinstance(obj["top_diagnoses"], list):
            return [str(x).strip() for x in obj["top_diagnoses"] if str(x).strip()][:5]
    except Exception:
        pass

    # Case 2: extract first JSON-like block containing "top_diagnoses"
    m = JSON_TOP_PAT.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "top_diagnoses" in obj and isinstance(obj["top_diagnoses"], list):
                return [str(x).strip() for x in obj["top_diagnoses"] if str(x).strip()][:5]
        except Exception:
            return None

    # Case 3: tolerant fallback — bullet/numbered lines
    lines = [ln.strip(" -*\t") for ln in text.splitlines()]
    picks = []
    for ln in lines:
        ln = re.sub(r'^\d+[\.\)\:]?\s*', '', ln)  # strip leading "1) ", "2. ", etc.
        if 1 <= len(ln) <= 128 and re.search(r'[A-Za-z]', ln):
            picks.append(ln)
        if len(picks) >= 5:
            break
    return picks or None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name, one of: {list(MODELS.keys())}")
    parser.add_argument(
        "--kb",
        type=str,
        required=True,
        choices=list(KB_CONFIG.keys()),
        help="which RAG DB to use",
    )
    parser.add_argument(
        "--rag_mode",
        type=str,
        required=True,
        choices=["comprehensive", "plain"],
        help=(
            "RAG pipeline type: "
            "'comprehensive' = rewrite + screen (no summarization); "
            "'plain' = no rewrite/screen, directly stuff retrieved passages."
        ),
    )
    args = parser.parse_args()

    model_name = args.model
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")
    model_path = MODELS[model_name]

    kb_name = args.kb
    kb_conf = KB_CONFIG[kb_name]
    rag_dir = Path(kb_conf["rag_dir"])
    retriever_type = kb_conf["retriever"]  # 'hybrid' or 'dense'

    INDEX_PATH = rag_dir / "faiss.index"
    META_PATH = rag_dir / "meta.json"
    CHUNKS_PARQUET = rag_dir / "chunks.parquet"

    # Output dir: OUT_ROOT / "<kb>__<rag_mode>"
    out_dir = os.path.join(OUT_ROOT, f"{kb_name}__{args.rag_mode}")
    os.makedirs(out_dir, exist_ok=True)

    # ===== Embedding / Index / KB / BM25=====
    # 1) FAISS
    index = faiss.read_index(str(INDEX_PATH))

    # 2) meta -> embedder
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta_conf = json.load(f)
    embedder = load_embedding_model(meta_conf["embedding_model"])

    # 3) chunks.parquet
    df_chunks = pd.read_parquet(CHUNKS_PARQUET)

    # 4) kb_records
    kb_records: List[Dict[str, Any]] = []
    for _, row in df_chunks.iterrows():
        kb_records.append(
            {
                "content": row["retrieval_text"],
                "book_id": row["book_id"],
                "breadcrumb": row["breadcrumb_str"],
                "heading": row["heading_text"],
                "chunk_id": int(row["chunk_global_index"]),
            }
        )

    # 5) BM25 for hybrid
    bm25 = None
    if retriever_type == "hybrid":
        def _bm25_tok(s: str) -> List[str]:
            return re.findall(r"[a-z0-9]+", s.lower())
        bm25_tokens = [_bm25_tok(t) for t in df_chunks["retrieval_text"].tolist()]
        bm25 = BM25Okapi(bm25_tokens)

    # ===== Init LLM =====
    if "235" in model_name:
        llm = LLM(model=model_path, tensor_parallel_size=8)
    elif "70" in model_name:
        llm = LLM(model=model_path, tensor_parallel_size=2)
    else:
        llm = LLM(model=model_path, tensor_parallel_size=1)

    # Sampling params
    sp_answer = SamplingParams(temperature=0, max_tokens=512)   # final JSON answer
    sp_json   = SamplingParams(temperature=0, max_tokens=196)   # rewrite & screen JSON

    # Load cases
    cases = load_json_list(CASE_DATA)

    rows_out: List[Dict[str, Any]] = []
    full_records: List[Dict[str, Any]] = []

    for case in tqdm(cases, desc=f"{safe_name(model_name)} | {kb_name} | {args.rag_mode} | CASES"):
        case_id = case.get("id", None)
        core_text = build_case_core_text(case)

        rewrite_out = ""
        screen_out = ""
        labels: List[str] = []
        scores: List[int] = []
        selected_indices: List[int] = []
        ctx_text: str = ""
        rewritten_q: str = ""

        if args.rag_mode == "comprehensive":

            rewritten_q, rewrite_out = llm_rewrite_query(core_text, llm, sp_json, model_name)
            effective_q = rewritten_q if rewritten_q else core_text
        else:

            effective_q = core_text


        if retriever_type == "hybrid":
            retrieved = retrieve_hybrid(effective_q, embedder, index, kb_records, bm25, TOP_K)
        else:
            retrieved = retrieve_dense(effective_q, embedder, index, kb_records, TOP_K)

        passages = [r["text"] for r in retrieved]

        if args.rag_mode == "comprehensive" and len(passages) > 0:

            labels, scores, screen_out = llm_screen(passages, effective_q, llm, sp_json, model_name)
            selected_indices = [i for i, s in enumerate(scores) if s >= 1]
            if not selected_indices:
                selected_indices = [0]
            selected_passages = [passages[i] for i in selected_indices]

            ctx_text = "\n\n".join(
                [f"[{rank+1}] {truncate_chars(p, 2000)}" for rank, p in enumerate(selected_passages)]
            )
        else:

            if len(passages) > 0:
                ctx_text = "\n\n".join(
                    [f"[{i+1}] {truncate_chars(p, 2000)}" for i, p in enumerate(passages)]
                )
            else:
                ctx_text = ""


        user_prompt = build_case_prompt_with_context(case, ctx_text)


        raw_output = _chat_once(llm, SYSTEM_INSTRUCTION, user_prompt, sp_answer, model_name)
        top_dx = parse_top_diagnoses(raw_output) or []

        top_dx = top_dx[:5]
        padded = top_dx + [""] * (5 - len(top_dx))

        row = {
            "model": safe_name(model_name),
            "kb": kb_name,
            "rag_mode": args.rag_mode,
            "retriever_type": retriever_type,
            "case_id": case_id,
            "top1": padded[0],
            "top2": padded[1],
            "top3": padded[2],
            "top4": padded[3],
            "top5": padded[4],
            "top_json": json.dumps(top_dx, ensure_ascii=False),
            "raw_output": raw_output,
            # "context_text": ctx_text,
            "rewritten_query": rewritten_q,
        }
        rows_out.append(row)

        full_records.append(
            {
                **row,
                "case_obj": case,
                "retrieved": retrieved,
                "screen_labels": labels,
                "screen_scores": scores,
                "selected_indices": selected_indices,
                "rewrite_out_raw": rewrite_out,
                "screen_out_raw": screen_out,
            }
        )

    df = pd.DataFrame(rows_out)
    out_csv = os.path.join(out_dir, f"{safe_name(model_name)}__{kb_name}__{args.rag_mode}__CASES_top5.csv")
    out_json = os.path.join(out_dir, f"{safe_name(model_name)}__{kb_name}__{args.rag_mode}__CASES_full.json")

    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_records, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] CSV  : {out_csv}")
    print(f"[SAVED] JSON : {out_json}")


if __name__ == "__main__":
    main()
