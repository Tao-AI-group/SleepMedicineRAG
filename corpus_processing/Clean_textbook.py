from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI


# -------------------------
# Page detection / splitting
# -------------------------
DEFAULT_PAGE_MARK_RE = r"(?m)^\s*(?:<!--\s*)?(?:page|p)\s*[:#]?\s*\d+\s*(?:-->)?\s*$"

def split_markdown_pages(md: str, page_mark_re: str, require_pages: bool) -> Tuple[List[str], str, int]:
    """
    Split Markdown into pages using explicit page markers.

    Returns (pages, mode, n_markers).
    mode is "pages" if markers found; otherwise "none".
    """
    md = md.replace("\r\n", "\n")

    page_re = re.compile(page_mark_re)
    marks = list(page_re.finditer(md))

    if len(marks) >= 2:
        sentinel = "\n<<<PAGE_MARK>>>\n"
        tmp = page_re.sub(sentinel, md)
        pages = [u.strip("\n") for u in tmp.split(sentinel) if u.strip()]
        return pages, "pages", len(marks)

    if require_pages:
        raise ValueError(
            "Page markers were not detected (need at least 2 markers).\n"
            "Fix by providing the correct --page-mark-re for your Markdown.\n"
            f"Current --page-mark-re: {page_mark_re}"
        )

    return [md.strip("\n")], "none", len(marks)


# -------------------------
# Fence-aware helpers
# -------------------------
FENCE_RE = re.compile(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|\$\$[\s\S]*?\$\$)", re.M)

def fence_aware_sub(text: str, pattern: re.Pattern, repl: str) -> str:
    out, idx = [], 0
    for m in FENCE_RE.finditer(text):
        before = text[idx:m.start()]
        out.append(pattern.sub(repl, before))
        out.append(m.group(0))
        idx = m.end()
    out.append(pattern.sub(repl, text[idx:]))
    return "".join(out)


# -------------------------
# Pass 1 — Boundary repair with strict continuity constraint
# -------------------------
@dataclass
class WindowResult:
    finalized: str
    carryover_anchor: str  
    notes: str
    raw_model: str


BOUNDARY_SYSTEM = """You are performing boundary repair on Markdown for retrieval indexing.

Input:
- A 3-page window (Page i, Page i+1, Page i+2).

Tasks (BOUNDARY REPAIR ONLY):
1) Fix paragraph fragmentation introduced at boundaries (join paragraphs split across page breaks).
2) Remove duplication caused by overlap across windows.

Strict continuity constraint:
- If a paragraph begins on Page i+1 and continues into Page i+2,
  consolidate that entire paragraph into the FINALIZED output for this window (i),
  and ensure it will be excluded from the leading portion of the subsequent window (i+1).

Rules:
- Perform ONLY boundary repair and deduplication. Do NOT paraphrase, summarize, reorder, or add content.
- Preserve Markdown structure (headings, lists, emphasis) as much as possible.
- Do NOT delete references/citations/questions in this pass (that is Pass 2).

Output MUST be valid JSON with exactly these keys:
{
  "finalized": "<text to append for Page i>",
  "carryover_anchor": "<first ~160 characters of the moved paragraph (exactly as in finalized), or empty string>",
  "notes": "<brief note>"
}

Important:
- "finalized" should include repaired Page i content plus any moved paragraph(s) from Page i+1 that continue into Page i+2.
- "carryover_anchor" must be an exact prefix of the moved paragraph from your finalized text if non-empty.
- Do not wrap output in code fences.
"""

def call_llm_json(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    backoff_base: float,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": BOUNDARY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            out = resp.choices[0].message.content or ""
            return out.strip()
        except Exception as e:
            last_err = e
            wait = backoff_base ** (attempt - 1)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}") from last_err


def safe_parse_window_json(raw: str) -> WindowResult:
    try:
        obj = json.loads(raw)
        finalized = str(obj.get("finalized", "")).strip()
        anchor = str(obj.get("carryover_anchor", "")).strip()
        notes = str(obj.get("notes", "")).strip()
        return WindowResult(finalized=finalized, carryover_anchor=anchor, notes=notes, raw_model=raw)
    except Exception:
        return WindowResult(finalized=raw.strip(), carryover_anchor="", notes="parse_failed_fallback", raw_model=raw)


def drop_leading_anchor(text: str, anchor: str) -> str:
    if not anchor:
        return text
    hay = text.lstrip()
    idx = hay.find(anchor)
    if idx != -1 and idx < 3000:
        rest = hay[idx:]
        m = re.search(r"\n\s*\n", rest)
        end = idx + (m.start() if m else len(rest))
        new_hay = hay[:idx].rstrip() + "\n\n" + hay[end:].lstrip()
        return new_hay.strip("\n")
    return text


def build_window_prompt(p_i: str, p_i1: str, p_i2: str, i: int) -> str:
    return (
        f"PAGE i (index {i}):\n-----\n{p_i}\n\n"
        f"PAGE i+1 (index {i+1}):\n-----\n{p_i1}\n\n"
        f"PAGE i+2 (index {i+2}):\n-----\n{p_i2}\n"
    )


def boundary_repair_pass(
    pages: List[str],
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    rate_limit_sleep: float,
    max_retries: int,
    backoff_base: float,
    audit_path: Optional[Path],
) -> str:
    repaired_chunks: List[str] = []
    prev_anchor = ""

    if audit_path:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            audit_path.unlink()

    n = len(pages)
    if n == 0:
        return ""

    padded = pages + ["", ""]
    for i in range(n):
        p_i = padded[i]
        p_i1 = drop_leading_anchor(padded[i + 1], prev_anchor)
        p_i2 = padded[i + 2]

        prompt = build_window_prompt(p_i, p_i1, p_i2, i)
        raw = call_llm_json(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        res = safe_parse_window_json(raw)

        finalized = res.finalized.strip()
        if finalized:
            repaired_chunks.append(finalized.rstrip())

        if audit_path:
            record = {
                "page_index": i,
                "carryover_anchor": res.carryover_anchor,
                "notes": res.notes,
                "raw_model": res.raw_model,
            }
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        prev_anchor = res.carryover_anchor.strip()
        time.sleep(rate_limit_sleep)

    repaired = "\n\n".join([c for c in repaired_chunks if c.strip()])
    repaired = re.sub(r"\n{4,}", "\n\n\n", repaired).strip() + "\n"
    return repaired


# -------------------------
# Pass 2 — Deletion-only cleaning (Markdown-preserving)
# -------------------------
NUMERIC_BRACKETS = re.compile(r"\s*\[(?:\d+(?:\s*[–-]\s*|\s*,\s*)?)+\]")
AUTHOR_YEAR = re.compile(
    r"\((?:[A-Z][A-Za-z-]+(?:\s+et al\.)?(?:\s*&\s*[A-Z][A-Za-z-]+)?(?:,\s*\d{4})(?:;\s*[A-Z][A-Za-z-]+(?:\s+et al\.)?,\s*\d{4})*)\)"
)
SUPERSCRIPTS = re.compile(r"[\u00b9\u00b2\u00b3\u2070-\u2079]+")
CARET_NUM = re.compile(r"\^\s*\d+")

REF_SECTION_MD_RE = re.compile(
    r"(?ims)^#{1,6}\s*(references|selected readings|further reading|suggested reading)\b[\s\S]*?(?=^#{1,6}\s|\Z)"
)
QA_SECTION_MD_RE = re.compile(
    r"(?ims)^#{1,6}\s*(review questions|self-assessment|quiz|questions|answers|answer key)\b[\s\S]*?(?=^#{1,6}\s|\Z)"
)
KEYWORDS_LINE_RE = re.compile(r"(?im)^\s*keywords?\s*[:\-].*$")

TOC_LIKE_LINE_RE = re.compile(r"(?m)^\s*\d+\s+.+?\s+\d+\s*(?:[A-Z][A-Za-z\-\s]+)?\s*$")
ISOLATED_PAGE_NUM_RE = re.compile(r"(?m)^\s*\d{1,4}\s*$")

DOI_URL_LINE_RE = re.compile(r"(?im)^\s*(?:doi\s*:|https?://\S+|www\.\S+)\s*$")

FRONT_MATTER_HEADS = {
    "foreword", "preface", "contents", "contributors", "index",
    "acknowledgments", "acknowledgements", "disclaimer", "copyright",
}

def deletion_only_clean_markdown(md: str, drop_front_matter: bool = True) -> str:
    s = md.replace("\r\n", "\n")

    if drop_front_matter:
        for h in sorted(FRONT_MATTER_HEADS, key=len, reverse=True):
            s = re.sub(
                rf"(?ims)^#{1,6}\s*{re.escape(h)}\b[\s\S]*?(?=^#{1,6}\s|\Z)",
                "",
                s,
            )

    s = REF_SECTION_MD_RE.sub("", s)
    s = QA_SECTION_MD_RE.sub("", s)
    s = KEYWORDS_LINE_RE.sub("", s)

    s = fence_aware_sub(s, NUMERIC_BRACKETS, "")
    s = fence_aware_sub(s, AUTHOR_YEAR, "")
    s = fence_aware_sub(s, SUPERSCRIPTS, "")
    s = fence_aware_sub(s, CARET_NUM, "")

    s = TOC_LIKE_LINE_RE.sub("", s)
    s = ISOLATED_PAGE_NUM_RE.sub("", s)
    s = DOI_URL_LINE_RE.sub("", s)

    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"\[\s*\]", "", s)
    s = re.sub(r"\s+([,.;:)\]])", r"\1", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip() + "\n"


# -------------------------
# CLI / main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-pass preprocessing for Markdown (boundary repair + deletion-only cleaning).")
    p.add_argument("--input-md", required=True, help="Input Markdown file (.md).")
    p.add_argument("--out-dir", default=".", help="Output directory.")
    p.add_argument("--model", default="gpt-4o", help="OpenAI model (e.g., gpt-4o).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--rate-sleep", type=float, default=0.6)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--backoff-base", type=float, default=1.6)

    p.add_argument("--page-mark-re", default=DEFAULT_PAGE_MARK_RE,
                   help="Regex for page marker lines used to split pages.")
    p.add_argument("--require-pages", action="store_true",
                   help="Fail if page markers are not detected (recommended when your files have markers).")

    p.add_argument("--keep-front-matter", action="store_true",
                   help="If set, do NOT drop common front-matter sections (Foreword/Preface/etc.).")

    return p.parse_args()


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    args = parse_args()
    in_path = Path(args.input_md)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = in_path.stem
    repaired_path = out_dir / f"{stem}.repaired.md"
    cleaned_path = out_dir / f"{stem}.cleaned.md"
    audit_path = out_dir / f"{stem}.audit.jsonl"

    md = in_path.read_text(encoding="utf-8", errors="replace")

    try:
        pages, mode, n_markers = split_markdown_pages(md, page_mark_re=args.page_mark_re, require_pages=args.require_pages)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"[info] input: {in_path.name}")
    print(f"[info] split mode: {mode} | pages: {len(pages)} | markers detected: {n_markers}")
    print(f"[info] model: {args.model}")

    client = OpenAI(api_key=api_key)

    repaired = boundary_repair_pass(
        pages=pages,
        client=client,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        rate_limit_sleep=args.rate_sleep,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
        audit_path=audit_path,
    )
    repaired_path.write_text(repaired, encoding="utf-8")
    print(f"[ok] wrote: {repaired_path}")

    cleaned = deletion_only_clean_markdown(repaired, drop_front_matter=(not args.keep_front_matter))
    cleaned_path.write_text(cleaned, encoding="utf-8")
    print(f"[ok] wrote: {cleaned_path}")

    print(f"[ok] audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
