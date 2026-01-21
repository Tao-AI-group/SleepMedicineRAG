# Optimizing Retrieval-Augmented Generation (RAG) in Clinical Medicine: Methods and Performance Evaluation

This repository contains the official implementation and experimental pipelines for the research paper: **"Optimizing Retrieval-Augmented Generation (RAG) in Clinical Medicine: Methods and Performance Evaluation."**

Our work investigates the optimization of RAG systems specifically for the high-stakes domain of sleep medicine, focusing on corpus preprocessing, hybrid retrieval strategies, and specialized evaluation for clinical vignettes and multiple-choice questions (MCQs).

---

## 🚀 Overview

Clinical RAG systems often struggle with fragmented textbook data and complex medical terminology. This project provides a robust framework to:

* **Preprocess medical corpora** using an LLM-assisted two-pass pipeline (Boundary Repair & Deletion-only Cleaning).
* **Establish multi-modal indices** (TOC-aware vs. Plain, Hybrid BM25+FAISS vs. Dense-only).
* **Evaluate clinical reasoning** via automated diagnosis matching and standardized medical exam (AASM, BoardVitals) benchmarking.

---

## 📂 Project Structure

```bash
├── corpus_processing/
│   ├── Clean_textbook.py          # Two-pass LLM-assisted Markdown cleaning
│   └── instructions_to_clean.txt  # Detailed preprocessing logic
├── database_establishment/
│   ├── build_database_4_books.py  # Builds TOC-aware hybrid RAG indices
│   ├── build_single_database.py   # Builds dense-only RAG indices
│   └── ...                        # Variants for uncleaned/plain corpora
├── RAG_part/
│   ├── case.py                    # Clinical Vignette (Case) RAG pipeline
│   ├── mcq.py                     # MCQ evaluation pipeline (AASM/BoardVitals)
│   └── eval_case_diagnosis_matches_llm.py # LLM-based diagnosis evaluation
├── dataset/                       # (Not included) Requires licensed medical data
└── README.md

```

---

## 🛠 Methodology

### 1. Corpus Preprocessing Pipeline

To reduce non-clinical noise while preserving narrative continuity:

* 
**Pass 1 (Boundary Repair):** Uses a sliding three-page window to repair paragraph fragmentation and remove duplicates introduced by page breaks.


* 
**Pass 2 (Deletion-only Cleaning):** Conservatively removes non-clinical artifacts like reference sections, inline citations, and isolated page numbers without rewriting clinical content.



### 2. Retrieval Strategy

We support multiple knowledge base (KB) configurations:

* **Hybrid Retrieval:** Combines **BM25** (keyword matching) with **FAISS** (dense vector embeddings) using Reciprocal Rank Fusion (**RRF**).
* **Comprehensive Mode:** Includes LLM-driven query rewriting and passage screening to filter irrelevant context before generation.

### 3. Models Supported

The pipeline integrates several state-of-the-art models via **vLLM**:

* Llama-3.1/3.3 (8B, 70B)
* Qwen series (14B, 235B)

---

## 📊 Evaluation

### Clinical Vignettes (Case Studies)

The system extracts top diagnoses from clinical cases and compares them against ground-truth ICSD-3 labels. We utilize a specialized GPT-4o evaluator to determine semantic equivalence for medical synonyms (e.g., "OSA" vs. "Obstructive Sleep Apnea").

### Multiple Choice Questions (MCQ)

Evaluates zero-shot and RAG-augmented performance on:

* **AASM MCQ Dataset**
* **BoardVitals Sleep Medicine**

---

## 💻 Usage

### Preprocessing

```bash
export OPENAI_API_KEY="your_api_key"
python corpus_processing/Clean_textbook.py --input-md textbook.md --out-dir ./cleaned_data --require-pages

```

### Building Index

```bash
python database_establishment/build_database_4_books.py

```

### Running Evaluation

```bash
# Run MCQ evaluation with Llama-70B using the cleaned hybrid KB
python RAG_part/mcq.py --model LLaMA-70B --kb cleaned_hybrid --rag_mode comprehensive

```

---

## ⚖️ License

This project is intended for research purposes. Medical datasets (AASM, textbook corpora) are subject to their respective licenses and are not included in this repository.

