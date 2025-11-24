# 📘 **SciRAG-Eval**

*A basic scientific Retrieval-Augmented Generation (RAG) system built from scratch using two astrophysics research papers.*

---

## 🧭 **Overview**

**SciRAG-Eval** is a **basic RAG prototype** I built to understand how real-world scientific question-answering systems work.
Rather than using large datasets, I intentionally restricted the scope to **just two scientific PDFs**:

1. **paper1.pdf** – Neutron star–black hole (NSBH) gravitational-wave events
2. **paper2.pdf** – Dust formation in the supernova remnant G54.1+0.3

This allowed me to focus entirely on the **end-to-end mechanics** of a RAG pipeline:

* Extracting text from PDFs
* Cleaning and chunking the text
* Embedding chunks using Sentence Transformers
* Building a FAISS vector index
* Retrieving relevant chunks for each query
* Answering questions using a local LLM (FLAN-T5)
* Comparing answers to gold references using cosine similarity

Even though the system is small, the core architecture mirrors real production RAG systems.

---

## 🌟 **Why I Built This**

I built this project as a practical way to learn:

* How vector search works
* How RAG systems break (and how to fix them)
* How scientific PDFs differ from normal text sources
* How to evaluate QA systems automatically
* How FAISS, Sentence Transformers, and local LLMs interact

This project helped me build a real AI pipeline on my own, understand common failure modes, and gain the confidence to build larger systems in the future.

---

# 🧩 **Project Architecture**

```
📄 PDFs (paper1.pdf, paper2.pdf)
        │
        ▼
🔧 ingest_papers.py
        - Extract text (PyMuPDF)
        - Normalize text, repair characters
        - Save as structured JSONL
        │
        ▼
🔍 build_index.py
        - Sentence-level chunking
        - Embedding with all-MiniLM-L6-v2
        - FAISS index creation
        │
        ▼
❓ rag_qa.py
        - Retrieve top-K similar chunks
        - Return context to answer module
        │
        ▼
🤖 rag_answer_local.py
        - Local LLM (FLAN-T5) answers using retrieved context
        │
        ▼
📊 evaluate.py
        - Compare predictions to gold answers
        - Cosine similarity scoring
        ▼
📁 eval_results.csv
```

---

# 📁 **Repository Structure**

```
sci-rag-eval/
│
├── data/
│   ├── raw/           # uploaded PDFs
│   ├── processed/     # extracted text, chunks, embeddings, FAISS index
│
├── src/
│   ├── ingest_papers.py
│   ├── build_index.py
│   ├── rag_qa.py
│   ├── rag_answer_local.py
│   ├── evaluate.py
│
├── eval/
│   ├── qa_set.jsonl   # gold question-answer dataset
│   ├── eval_results.csv
│
├── requirements.txt
├── README.md
└── app.py (optional CLI)
```

---

# 🔬 **How It Works**

### 1️⃣ **Ingestion**

`ingest_papers.py` extracts text from PDFs using PyMuPDF and converts each page to structured JSON.

### 2️⃣ **Chunking + Embeddings**

`build_index.py` chunks the extracted text into overlapping windows, embeds them using `all-MiniLM-L6-v2`, and builds a FAISS index.

### 3️⃣ **Retrieval**

`rag_qa.py` retrieves the top-K most similar chunks to the question.

### 4️⃣ **Answer Generation**

`rag_answer_local.py` uses a **local FLAN-T5 model** (no API cost!) to generate an answer using ONLY the retrieved context.

### 5️⃣ **Evaluation**

`evaluate.py` compares the model’s answer with gold answers using cosine similarity and writes the results to `eval_results.csv`.

---

# 🐛 **What Didn't Work (My Mistakes)**

This section is important for recruiters.
I am openly documenting all failure modes and lessons.

### ❗ Mistake 1 — Assuming PDF extraction “just works”

Scientific PDFs are formatted in complex ways:

* two-column layout
* ligatures (ﬂ, ﬁ)
* misaligned text blocks
* missing abstract block

PyMuPDF skipped several important sections, including:

* abstracts
* conclusions
* modelling results

This made retrieval impossible for many questions.

---

### ❗ Mistake 2 — Using naive fixed-length chunking

I initially chunked by character count (800 chars).
This caused:

* broken sentences
* unrelated sentences being merged
* meaningless fragments entering FAISS
* poor retrieval even with good embeddings

---

### ❗ Mistake 3 — Over-aggressive reference filtering

My back-matter heuristic accidentally removed:

* modelling sections
* parts of introduction
* some captions and text blocks

This further reduced the amount of usable content.

---

### ❗ Mistake 4 — No re-ranking

FAISS similarity search alone is not enough, especially on scientific PDFs.
It retrieves:

* citation blocks
* numerical tables
* noise
* figure captions

A re-ranker is needed to improve chunk selection.

---

### ❗ Mistake 5 — Evaluating before validating retrieval

I ran evaluation before checking that:

* chunks contained the answers
* ingestion was complete
* abstract text existed

This led to early similarity scores near **0.0**, which initially confused me.

---

# 📉 **Current Results (Honest)**

* Some questions achieve high similarity (e.g., scientific dust composition → 0.76)
* Many questions still fail with “Not enough information…”
* Evaluation CSV is **not fully accurate yet**
* Bottleneck: **retrieval**, not the model

This is expected for a **basic two-PDF RAG**.

---

# 🚀 **Future Improvements (Clear Roadmap)**

### 🔧 1. Rewrite ingestion using block-level extraction

Sort blocks by coordinates to handle 2-column PDFs correctly.

### 🔧 2. Add ligature and Unicode repair (`ftfy`)

Fixes “ﬁ”, “ﬂ”, “ﬀ”, “–”, “—”, etc.

### 🔧 3. Switch to sentence-level chunking

Use sliding windows of 3–7 sentences.

### 🔧 4. Add a cross-encoder re-ranking step

Re-rank top-20 FAISS hits using a BERT re-ranker.

### 🔧 5. Add section-based boost

Prioritize:

* Abstract
* Introduction
* Conclusion
* Methods

### 🔧 6. Visualize retrieval

Save a debug file showing text used for each answer.

### 🔧 7. Expand to more PDFs

A larger corpus will improve retrieval, recall, and evaluation.

---

# ▶️ **How to Run the Project**

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ingest your papers

```bash
python src/ingest_papers.py
```

### 3. Build the index

```bash
python src/build_index.py
```

### 4. Ask a question

```bash
python src/rag_answer_local.py "What is the aim of this study?"
```

### 5. Run evaluation

```bash
python src/evaluate.py
```

Results will appear in:

```
eval/eval_results.csv
```

---

# 🎓 **What I Learned**

This project taught me:

* How RAG systems are built end-to-end
* Why ingestion is the hardest part of RAG
* How scientific PDF extraction works
* Why chunking strategy determines retrieval accuracy
* How to debug embeddings and FAISS indexes
* Why evaluation requires careful thought
* The importance of transparent documentation and error analysis

---

# 🏁 **Final Notes**

Even though this is a **basic two-PDF RAG system**, it represents a meaningful engineering milestone for me.
It shows:

* my ability to build an end-to-end AI system
* my debugging problem-solving skills
* my realistic understanding of RAG failure modes
* my honesty about limitations and plan for improvement

This is the foundation for a full-scale scientific literature assistant I plan to build next.

---
