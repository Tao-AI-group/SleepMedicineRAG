# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import requests
from tqdm import tqdm
import pandas as pd


PAPERS_ROOT_DIR = Path(
    "/dgx1data/aii/tao/m327768/workdir/projects/RAG_PAPER/exp_code_Nov/case/results_temp"
)

OUTPUT_ROOT = Path(
    "/dgx1data/aii/tao/m327768/workdir/projects/RAG_PAPER/exp_code_Nov/case/diag_match_llm"
)

CASES_JSON_PATH = Path(
    "/dgx1data/aii/tao/m327768/workdir/projects/RAG_PAPER/dataset/vignette_cases_structured_pipe.json"
)


apigee_base = ""
engine = "gpt-4o"
api_version = "2024-10-21"
apigee_token = ""  

endpoint = (
    f"{apigee_base}/llm-azure-openai/openai/deployments/"
    f"{engine}/chat/completions?api-version={api_version}"
)

max_completion_tokens = 512  
top_p = 1
presence_penalty = 0
frequency_penalty = 0

# ----------------- System Prompt -----------------
SYSTEM_PROMPT = r"""
You are an expert sleep medicine clinician tasked with evaluating model-predicted diagnoses.

You will be given:
- a list of ground-truth diagnoses for one case (ICSD-3 / sleep medicine style labels)
- a list of up to 5 model predictions (in ranked order, 1 = most likely)

Your job:
For each ground-truth diagnosis, determine whether any of the model’s predictions are semantically equivalent to the same disorder.

Matching rules:
- Treat synonyms, abbreviations, and wording variants as the same disorder.
  Examples:
  * "obstructive sleep apnea" == "OSA" == "obstructive sleep apnea syndrome"
  * "restless legs syndrome" == "RLS"
  * "periodic limb movement disorder" == "PLMD"
  * "narcolepsy type 1" == "narcolepsy with cataplexy"
- Severity variations (mild/moderate/severe) count as matches to the base disorder.
- Ignore capitalization, punctuation, and suffixes like "disorder", "syndrome".
- If multiple predictions match the same ground-truth diagnosis, report all matched ranks in a sorted list.
- Use the lowest (best) rank among them as best_rank.
- If no prediction matches, matched=false, matched_ranks=[], matched_predictions=[], best_rank=null.

Output JSON only:
{
  "matches": [
    {
      "gt": "<ground_truth_diagnosis>",
      "matched": true or false,
      "matched_ranks": [<rank1>, <rank2>, ...],   // sorted ascending
      "best_rank": integer or null,
      "matched_predictions": ["<text1>", "<text2>", ...]
    },
    ...
  ]
}

Constraints:
- The 'matches' array must have the same length and order as the provided ground_truth list.
- Do NOT include explanations or extra fields.
"""


def build_case_eval_prompt(
    case_id,
    gt_labels,
    pred_labels,
    model_name=None,
    kb=None,
    rag_mode=None,
) -> str:

    lines = []
    if case_id is not None:
        lines.append(f"Case ID: {case_id}")
    if model_name:
        lines.append(f"Model: {model_name}")
    if kb:
        lines.append(f"KB: {kb}")
    if rag_mode:
        lines.append(f"RAG mode: {rag_mode}")

    lines.append("\nGround truth diagnoses (this case):")
    for i, gt in enumerate(gt_labels, 1):
        lines.append(f"{i}. {gt}")

    lines.append("\nModel top-5 predictions (ranked, 1 is highest):")
    for i, pred in enumerate(pred_labels, 1):
        lines.append(f"{i}. {pred}")

    lines.append(
        "\nTask:\n"
        "For each ground truth diagnosis above, decide whether any of the predictions 1-5 "
        "are semantically equivalent to the same disorder, following the matching rules. "
        "Return a JSON object exactly in the specified format."
    )

    return "\n".join(lines)



def call_llm(prompt_text: str) -> dict:
    headers = {
        "Authorization": f"Bearer {apigee_token}",
        "Content-Type": "application/json",
    }

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ],
        },
    ]

    payload = {
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_completion_tokens,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=600)

    if resp.status_code != 200:
        print("[ERROR] LLM HTTP", resp.status_code)
        print(resp.text[:2000])
        resp.raise_for_status()

    data = resp.json()

    content = data["choices"][0]["message"]["content"]

    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except Exception as e:
        print("[WARN] Failed to json.loads(content), returning raw content with error:", e)
        return {"raw_content": content, "parse_error": str(e)}

def debug_print_final_diagnosis_counts(id2gt, max_cases: int | None = 20):

    def _sort_key(item):
        cid, _ = item
        try:
            return int(cid)
        except ValueError:
            return cid

    items = sorted(id2gt.items(), key=_sort_key)
    sums = 0
    print("\n[DEBUG] final_diagnosis count per case:")
    for i, (cid, labels) in enumerate(items):
        if max_cases is not None and i >= max_cases:
            print(f"... (only showing first {max_cases} cases)")
            break
        print(f"  case_id={cid}: {len(labels)} diagnosis -> {labels}")
        sums += len(labels)
    print(f"[DEBUG] Total diagnoses per case: {sums}")
    print(f"[DEBUG] Total cases with GT: {len(id2gt)}\n")


def load_case_ground_truth(path: Path):

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id2gt = {}

    for obj in data:
        cid = obj.get("id")
        if cid is None:
            continue

        val = obj.get("final_diagnosis")
        if not isinstance(val, str):
            continue

        s = val.strip()
        if not s:
            continue


        parts = [p.strip() for p in s.split("|") if p.strip()]

        if parts:
            id2gt[str(cid)] = parts

    return id2gt



def main():

    print(f"[INFO] Loading ground truth from {CASES_JSON_PATH}")
    id2gt = load_case_ground_truth(CASES_JSON_PATH)
    print(f"[INFO] Loaded GT for {len(id2gt)} cases")

    example_gt = id2gt.get("19")
    if example_gt:
        print(f"[DEBUG] Example GT for case_id=1: {example_gt}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


    subdirs = [p for p in PAPERS_ROOT_DIR.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"[WARN] No subdirectories found under {PAPERS_ROOT_DIR}")
        return

    for subdir in sorted(subdirs):
        print(f"[INFO] Entering subdir: {subdir}")
        csv_files = sorted(subdir.glob("*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files in subdir {subdir}, skip.")
            continue

        for csv_path in csv_files:
            print(f"[INFO] Processing result file: {csv_path}")
            df = pd.read_csv(csv_path)


            required_cols = ["case_id", "top1", "top2", "top3", "top4", "top5"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                print(
                    f"[WARN] File {csv_path} missing required columns: {missing_cols}, skip this file."
                )
                continue


            has_model_col = "model" in df.columns
            has_kb_col = "kb" in df.columns
            has_rag_mode_col = "rag_mode" in df.columns
            has_raw_output_col = "raw_output" in df.columns

            rel_path = csv_path.relative_to(PAPERS_ROOT_DIR)
            out_path = OUTPUT_ROOT / rel_path
            out_path = out_path.with_suffix(".llm_eval.jsonl")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with out_path.open("w", encoding="utf-8") as fout:
                for idx, row in tqdm(
                    df.iterrows(), total=len(df), desc=f"{rel_path}"
                ):
                    case_id = row.get("case_id")
                    if pd.isna(case_id):
                        continue
                    case_id_str = str(case_id)

                    gt_labels = id2gt.get(case_id_str)
                    if not gt_labels:

                        continue

                    pred_labels = []
                    for col in ["top1", "top2", "top3", "top4", "top5"]:
                        val = row.get(col)
                        if isinstance(val, str) and val.strip():
                            pred_labels.append(val.strip())
                    if not pred_labels:
                        continue

                    model_name = row["model"] if has_model_col else None

                    if model_name and "medgemma" in model_name.lower():
                        continue

                    kb = row["kb"] if has_kb_col else None
                    rag_mode = row["rag_mode"] if has_rag_mode_col else None
                    raw_output = row["raw_output"] if has_raw_output_col else None

                    prompt = build_case_eval_prompt(
                        case_id=case_id_str,
                        gt_labels=gt_labels,
                        pred_labels=pred_labels,
                        model_name=model_name,
                        kb=kb,
                        rag_mode=rag_mode,
                    )

                    try:
                        llm_result = call_llm(prompt)
                        error = None
                    except Exception as e:
                        llm_result = None
                        error = repr(e)
                        print(
                            f"[ERROR] LLM call failed for case_id={case_id_str} "
                            f"in {csv_path.name}: {e}"
                        )

                    out_obj = {
                        "source_file": str(csv_path),
                        "row_index": int(idx),
                        "model": model_name,
                        "kb": kb,
                        "rag_mode": rag_mode,
                        "case_id": case_id_str,
                        "ground_truth": gt_labels,
                        "predictions": pred_labels,
                        "raw_output": raw_output,
                        "llm_result": llm_result,
                        "error": error,
                    }

                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            print(f"[INFO] Saved LLM eval results for {rel_path} -> {out_path}")

    print(f"[DONE] All evaluations saved under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
