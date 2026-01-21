# vllm_mcq.py
# -*- coding: utf-8 -*-
"""
Unified MCQ RAG evaluation with vLLM.

Supports:
- 6 RAG DBs (TOC / cleaned / uncleaned × hybrid / single)
- Retrieval backend: hybrid (BM25 + FAISS) or dense-only (FAISS)
- RAG mode:
    * comprehensive: query rewrite + passage screening
    * plain: no rewrite, no screening; directly stuff top-K passages

Usage example:
    python vllm_mcq.py --model LLaMA-70B --kb cleaned_hybrid --rag_mode comprehensive
"""

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


AASM_MCQ = "PATH_TO/RAG_PAPER/dataset/AASM_MCQ.jsonl"
BOARD_VITALS_MCQ = "PATH_TO/RAG_PAPER/dataset/BOARD_VITALS.jsonl"
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

OUT_ROOT = "PATH_TO/RAG_PAPER/exp_code_Nov/mcq/comprehensive_code/k16"


user_prompt_template_with_rag = """You are given the following information:

--- CONTEXT ---
{retrieved_context}
---

--- MULTIPLE CHOICE QUESTION ---
{MCQ}
---

Answer:"""

SYSTEM_INSTRUCTION = (
    "You are a Sleep Medicine board-exam assistant. "
    "Select ONLY one best answer from the provided options (A–H). "
    "Output must be a single capital letter with no explanation."
)

MODELS: Dict[str, str] = {
    "MedGemma-27B-it": "/dgx1data/aii/tao/m327768/model/google/medgemma-27b-text-it/models--google--medgemma-27b-text-it/snapshots/6b08c481126ff65a9b8fa5ab4d691b152b8edb5d",
    "MedGemma-4B-it": "/dgx1data/aii/tao/m327768/model/google/medgemma-4b-it/models--google--medgemma-4b-it/snapshots/698f7911b8e0569ff4ebac5d5552f02a9553063c",
    "LLaMA-70B": "/dgx1data/aii/tao/tools/models/meta-llama/Llama-3.3-70B-Instruct",
    "LLaMA-8B": "/dgx1data/aii/tao/m327768/model/meta-llama/Llama-3.1-8B-Instruct/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    "Qwen-14B": "/dgx1data/aii/tao/m327768/model/Qwen/Qwen3-14B/models--Qwen--Qwen3-14B/snapshots/8268fe3026cb304910457689366670e803a6fd56",
    "Qwen-235B": "/dgx1data/aii/tao/m327768/model/Qwen/Qwen3-235B-A22B-Instruct-2507/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/8b0a01458495f9e2c57148603702a8057b4a265d",
}

LETTER2IDX = {chr(65 + i): i for i in range(8)}  # A–H
IDX2LETTER = {v: k for k, v in LETTER2IDX.items()}
ANS_PAT = re.compile(r"\b([A-H])\b", flags=re.I)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

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

def _to_list_choices(item: Any) -> List[str]:
    if item is None:
        return []
    if isinstance(item, list):
        return [str(x) for x in item]
    if isinstance(item, dict):
        return [str(item[k]) for k in sorted(item.keys())]
    if isinstance(item, str):
        lines = [l.strip() for l in item.splitlines() if l.strip()]
        pairs = []
        for l in lines:
            m = re.match(r"^([A-H])[\.\)\:]\s*(.*)$", l, flags=re.I)
            if m:
                pairs.append((m.group(1).upper(), m.group(2).strip()))
        if pairs:
            return [p[1] for p in sorted(pairs, key=lambda x: x[0])]
        if "|" in item:
            return [s.strip() for s in item.split("|")]
        if "; " in item:
            return [s.strip() for s in item.split("; ")]
        return [item.strip()]
    return [str(item)]

def normalize_mcq(raw: Dict[str, Any]) -> Dict[str, Any]:
    q = raw.get("question") or raw.get("stem") or raw.get("Question") or raw.get("prompt") or ""
    options = raw.get("options") or raw.get("choices") or raw.get("Options") or raw.get("Choices")
    choices = _to_list_choices(options)

    ans_letter, ans_idx = None, None

    key_letter = raw.get("answer_key") or raw.get("Answer_key") or raw.get("key") or raw.get("Key")
    if isinstance(key_letter, str):
        m = re.match(r"^([A-H])$", key_letter.strip(), flags=re.I)
        if m:
            ans_letter = m.group(1).upper()
            ans_idx = LETTER2IDX.get(ans_letter)

    if ans_idx is None:
        ans_text = raw.get("answer_text") or raw.get("Answer_text")
        if isinstance(ans_text, str) and choices:
            try:
                ans_idx = choices.index(ans_text.strip())
                ans_letter = IDX2LETTER.get(ans_idx)
            except ValueError:
                pass

    if ans_idx is None:
        ans_raw = raw.get("answer") or raw.get("Answer") or raw.get("gold") or raw.get("correct")
        if isinstance(ans_raw, (int, float)) and not (isinstance(ans_raw, float) and math.isnan(ans_raw)):
            idx = int(ans_raw)
            if 0 <= idx < len(choices):
                ans_idx = idx
                ans_letter = IDX2LETTER.get(idx)
        elif isinstance(ans_raw, str):
            s = ans_raw.strip()
            m = re.match(r"^([A-H])$", s, flags=re.I)
            if m:
                ans_letter = m.group(1).upper()
                ans_idx = LETTER2IDX.get(ans_letter)
            if ans_idx is None:
                m = re.match(r"^([A-H])[\.\)\:]\s*(.*)$", s, flags=re.I)
                if m:
                    ans_letter = m.group(1).upper()
                    ans_idx = LETTER2IDX.get(ans_letter)
            if ans_idx is None and choices:
                try:
                    ans_idx = choices.index(s)
                    ans_letter = IDX2LETTER.get(ans_idx)
                except ValueError:
                    pass

    return {
        "question": str(q).strip(),
        "choices": choices,
        "answer_idx": ans_idx,
        "answer_letter": ans_letter,
        "meta": raw,
    }

def truncate_chars(text: str, max_len: int) -> str:
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= max_len else s[:max_len]

def parse_letter(text: str) -> Optional[str]:
    m = ANS_PAT.search(text.strip())
    return m.group(1).upper() if m else None

def safe_name(s: str) -> str:
    s = s.rstrip("/\\")
    base = os.path.basename(s)
    return base or s

def maybe_add_no_think(model_name: str, content: str) -> str:
    return (content + " /no_think") if ("qwen" in model_name.lower()) else content

# ========= vLLM Chat / JSON  =========
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

# ========= 两步（给 comprehensive 用：rewrite + screen）=========
def llm_rewrite_query(question: str, llm: LLM, sp_json: SamplingParams, model_name: str) -> Tuple[str, str]:
    prompt = f"""Query Rewriting:

Question:
{question}

Please reformulate the given question by employing precise medical terminology.
Focus on capturing the essence of the patient’s symptoms and conditions in a generalized
form that reflects common clinical descriptions. Avoid colloquial language and be concise (<=30 words).

Return JSON only: {{"rewritten":"<rewritten question>"}}"""
    js, out = run_chat_json(
        llm,
        prompt,
        sp_json,
        model_name,
        system="You are a concise assistant that returns valid JSON only.",
    )
    rewritten = (js or {}).get("rewritten", "").strip() if js else ""
    if not rewritten:
        rewritten = re.sub(r"\b(pt|patient)\b", "patient", question, flags=re.I)
    return rewritten, out

def llm_screen(passages: List[str], x: str, llm: LLM, sp_json: SamplingParams, model_name: str) -> Tuple[List[str], List[int], str]:
    short_psgs = [truncate_chars(p, 1800) for p in passages]
    joined = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(short_psgs)])
    prompt = f"""Passage Screening (merge relevance & usefulness):

Question:
{x}

Passages:
{joined}

For each passage, label exactly one of:
- Irrelevant: off-topic or generic noise.
- Relevant-Context: on-topic background but not directly answer-enabling.
- Evidential: contains criteria, thresholds, rules, or facts that directly support answering.

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

def build_prompt_with_context(example: Dict[str, Any], context_text: str) -> str:
    lines = [example["question"], ""]
    for i, opt in enumerate(example["choices"]):
        lines.append(f"{chr(65 + i)}. {opt}")
    prompt = user_prompt_template_with_rag.format(
        retrieved_context=context_text,
        MCQ="\n".join(lines),
    )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="logical model name in MODELS")
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

    if args.model not in MODELS:
        raise ValueError(f"Unknown model: {args.model}")
    model_name = args.model
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

    # ===== Embedding / Index / KB / BM25（如果需要）=====
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
    sp = SamplingParams(temperature=0, max_tokens=128)      # final answer
    sp_json = SamplingParams(temperature=0, max_tokens=196) # JSON short outputs

    datasets = {"AASM_MCQ": AASM_MCQ, "BOARD_VITALS_MCQ": BOARD_VITALS_MCQ}

    for dname, path in datasets.items():
        rows = load_jsonl(path) if path.endswith(".jsonl") else json.load(open(path, "r", encoding="utf-8"))
        results: List[Dict[str, Any]] = []

        for raw in tqdm(rows, desc=f"{safe_name(model_name)} | {kb_name} | {args.rag_mode} | {dname}"):
            ex = normalize_mcq(raw)
            if not ex["choices"] or len(ex["choices"]) < 2:
                continue

            original_q = ex["question"]


            rewrite_out = screen_out = ""
            labels: List[str] = []
            scores: List[int] = []
            selected_indices: List[int] = []
            ctx_text: str = ""
            rewritten_q: str = ""

            if args.rag_mode == "comprehensive":

                rewritten_q, rewrite_out = llm_rewrite_query(original_q, llm, sp_json, model_name)
                effective_q = rewritten_q if rewritten_q else original_q
            else:

                effective_q = original_q

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


            prompt = build_prompt_with_context(ex, ctx_text)


            raw_answer = _chat_once(llm, SYSTEM_INSTRUCTION, prompt, sp, model_name)
            pred_letter = parse_letter(raw_answer)

            gold_letter = ex["answer_letter"]
            correct = (pred_letter == gold_letter) if gold_letter else None

            results.append(
                {
                    "model": safe_name(model_name),
                    "dataset": dname,
                    "kb": kb_name,
                    "rag_mode": args.rag_mode,
                    "retriever_type": retriever_type,
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "gold_letter": gold_letter,
                    "pred_letter": pred_letter,
                    "correct": correct,
                    "raw_output": raw_answer,
                    "retrieved": retrieved,
                    "rewritten_query": rewritten_q,
                    "screen_labels": labels,
                    "screen_scores": scores,
                    "selected_indices": selected_indices,
                    "context_text": ctx_text,
                    "rewrite_out_raw": rewrite_out,
                    "screen_out_raw": screen_out,
                }
            )

        df = pd.DataFrame.from_records(results)
        sub = df.dropna(subset=["correct"]) if len(df) else df
        if len(sub):
            acc = float(sub["correct"].mean())
            n = int(sub["correct"].notna().sum())
            print(
                f"[RESULT] {safe_name(model_name)} | {kb_name} | {args.rag_mode} | {dname}: "
                f"Acc={acc:.3%} (n={n})"
            )
        else:
            print(
                f"[RESULT] {safe_name(model_name)} | {kb_name} | {args.rag_mode} | {dname}: no valid samples"
            )

        csv_path = os.path.join(
            out_dir,
            f"{safe_name(model_name)}__{kb_name}__{args.rag_mode}__{dname}.csv",
        )
        json_path = os.path.join(
            out_dir,
            f"{safe_name(model_name)}__{kb_name}__{args.rag_mode}__{dname}.json",
        )

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
