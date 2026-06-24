# Offline RAG

**Benchmarking edge-optimised small language models for fully offline, CPU-only Retrieval-Augmented Generation.**

A hybrid-retrieval RAG pipeline that runs end-to-end on a single consumer laptop — no GPU, no cloud API, no internet after setup. Built to answer questions from a 607-page project-management textbook using DeepSeek-R1 (1.5B) and Llama-3.2 (1B) served through Ollama.

> Course: **CSET-419 — Introduction to Generative AI** · Bennett University · 2025

---

## Why this exists

Most RAG demos assume you have a GPU or a cloud API key. A lot of real users — schools, clinics, NGOs, students on student laptops — have neither. Sending private documents to a hosted LLM is also not always an option.

This project asks a simple question: **can a modern small language model run a proper RAG pipeline entirely on CPU and still give answers you can trust?**

Short answer: yes, if you stop treating retrieval as an afterthought.

---

## Results at a glance

| Stage | Time | Notes |
|---|---|---|
| PDF loading + chunking (607 pages → 3,200 chunks) | **6.13 s** | PyMuPDF + 800-char chunks, 100-char overlap |
| Dense index build (ChromaDB + MiniLM-L6-v2) | **~161 s** | One-time cost per document |
| Sparse index build (BM25) | **1.49 s** | |
| Hybrid fusion (per query) | **~2 ms** | |
| Cross-encoder re-ranking (40 → 5) | **~2.5 s** | |
| Answer generation (DeepSeek-R1 1.5B via Ollama) | **19.42 s** | |
| **End-to-end per query** | **~22 s** | After indexing is cached |

| Quality metric | Before re-ranker | After re-ranker |
|---|---|---|
| Retrieval precision (top-5 relevance) | ~60 % | **~91 %** |
| Hallucination rate on hard queries | noticeable | **~0 %** |

Re-ranker confidence scores also double as a **kill switch**: any query where the top chunk scores below 0 is flagged as low-confidence. In our tests, easy queries landed at +7.38 and genuinely-hard ones fell to −1.58, which mapped cleanly onto the model's actual answer quality.

---

## Architecture

```
                 ┌──────────────────┐
  User Query ───▶│  Embed + Tokenise │
                 └────────┬──────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
   ┌──────────────────┐      ┌──────────────────┐
   │  Dense Retrieval │      │ Sparse Retrieval │
   │  ChromaDB +      │      │      BM25        │
   │  MiniLM-L6-v2    │      │                  │
   │  (top-20)        │      │    (top-20)      │
   └────────┬─────────┘      └────────┬─────────┘
            │                         │
            └───────────┬─────────────┘
                        ▼
              ┌───────────────────┐
              │   Hybrid Fusion   │  merge + dedupe → 40 candidates
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │  Cross-Encoder    │  ms-marco-MiniLM-L-6-v2
              │    Re-ranker      │  score + sort → top 5
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │   Prompt Template │  strict "use ONLY context" system instruction
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │  SLM via Ollama   │  DeepSeek-R1 1.5B (default) or Llama-3.2 1B
              └─────────┬─────────┘
                        ▼
                   Grounded Answer
                   + source pages
                   + confidence score
```

**Why hybrid?** Dense search catches meaning ("what is a project?" → finds paraphrased definitions). Sparse BM25 catches exact names and identifiers ("Meredith and Mantel" → exact hits even when the surrounding language varies). Either one alone fails on a different category of queries; together they cover both. The re-ranker is a cross-encoder that scores each (query, chunk) pair jointly, which is much more accurate than the bi-encoder used for initial retrieval — we just can't afford to run it on all 3,200 chunks, so we use it to refine the top 40.

---

## Stack

- **Orchestration:** LangChain, langchain-community, langchain-huggingface
- **Document loading:** PyMuPDF (via `PyMuPDFLoader`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-friendly)
- **Dense store:** ChromaDB
- **Sparse retrieval:** `rank_bm25`
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM runtime:** [Ollama](https://ollama.com)
- **Models benchmarked:** `deepseek-r1:1.5b`, `llama3.2:1b`
- **Language:** Python 3.10+, runs in Jupyter

Tested on a standard consumer laptop, CPU-only, no dedicated GPU.

---

## Setup

### 1. Install Ollama and pull the models

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Then pull the models
ollama pull deepseek-r1:1.5b
ollama pull llama3.2:1b
```

Windows users: download the installer from [ollama.com/download](https://ollama.com/download).

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/Vaishnaviii910/Offline_rag.git
cd Offline_rag

# Optional but recommended
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, these are the packages the notebook installs:

```bash
pip install langchain langchain-core langchain-community \
            langchain-huggingface langchain-text-splitters \
            pymupdf sentence-transformers rank_bm25 \
            chromadb ollama torch
```

### 3. Drop in your PDF

Place the source document as `data.pdf` in the project root (or edit `pdf_name` in the notebook).

### 4. Run the notebook

```bash
jupyter notebook offline_rag.ipynb
```

Run cells top-to-bottom. The first execution builds the indexes (~2–3 minutes for a 600-page book). Subsequent queries reuse the in-memory indexes and take ~22 seconds each.

---

## Usage

The main entry point is `ask_my_textbook_v2(query)`:

```python
ask_my_textbook_v2("What is the definition of a project?")
```

Example output:

```
🔍 Deep Searching (Recall: 40, Precision: 5)...
🤖 DeepSeek-R1 is thinking using Precision Context...
------------------------------
📝 FINAL ANSWER:
A project is a temporary endeavor undertaken to create a unique product,
service, or result. According to the PMI definition cited in the textbook...
------------------------------
⏱️ Generation Latency: 19.42s
🎯 Top Relevance Score: 7.3875
```

For low-confidence queries, the top relevance score drops below 0 and the system emits a warning — we found this was a reliable indicator that the retriever genuinely couldn't find a good match, and the answer should be treated with caution.

---

## Key design decisions

**Chunk size 800 / overlap 100.** Smaller chunks are easier for a CPU-bound embedding model and produce tighter re-ranker signals. Overlap prevents facts from being split across boundaries.

**Top-40 recall, top-5 precision.** The hybrid retriever casts a wide net (40 candidates) so we don't miss anything; the cross-encoder then does the expensive-but-accurate filtering down to 5. This two-stage design is the single biggest reason the final precision is around 91 %.

**Strict system prompt.** The prompt explicitly tells the model to use only the provided context and to say "the textbook does not provide enough detail" when it cannot answer. This, combined with the re-ranker, is what drives hallucination to near zero.

**Kill switch at score < 0.** Cross-encoder scores above 0 correspond to genuinely relevant matches; scores below 0 correspond to the retriever failing. Using 0 as a threshold gives a free confidence signal with no additional cost.

---

## Repo layout

```
Offline_rag/
├── offline_rag.ipynb        # Main notebook — end-to-end pipeline
├── data.pdf                 # Source document (not committed)
├── requirements.txt
└── README.md
```

---

## Benchmarks — notebook-measured

All numbers measured on a CPU-only consumer laptop, single run:

```
📄 Total Pages:   607
🧱 Total Chunks:  3,200
⏱️ Chunking:       6.13 s
⏱️ Dense index:   ~161 s   (warm HF cache; cold start can be 400 s+)
⏱️ Sparse index:  1.49 s
⏱️ Hybrid fusion: 0.002 s  (per query)
⏱️ Re-ranking:    ~2.5 s   (per query, 40 pairs)
⏱️ Generation:    19.42 s  (DeepSeek-R1 1.5B, per query)
```

The generation step dominates per-query latency, as expected for CPU inference.

---

## Limitations and future work

- **No UI yet.** This is a notebook, not an app. A Tauri/Electron front-end is the obvious next step.
- **Single document.** The current setup indexes one PDF. Multi-document support with per-document filtering is on the roadmap.
- **Generation is the bottleneck.** At ~19 s per query, the model is the slowest part. Quantised int4 builds and streaming output would help perceived latency even if total time stays the same.
- **Indexing cost.** 161 s is acceptable once, but we should serialise the ChromaDB *and* BM25 indexes to disk so they survive restarts. ChromaDB persistence is straightforward; BM25 needs a manual pickle.
- **Evaluation.** We evaluated on a handful of representative queries. A held-out question set with automated scoring would make future changes safer to ship.

---

## References

The project builds directly on these works. Full citations are in the accompanying project report.

- Tyndall et al. (2025) — the base paper this project extends
- Lewis et al. — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Karpukhin et al. — Dense Passage Retrieval
- Reimers & Gurevych — Sentence-BERT
- Robertson & Zaragoza — BM25 / Okapi
- Nogueira & Cho — Passage Re-ranking with BERT
- DeepSeek-AI — DeepSeek-R1 technical report
- Meta AI — Llama 3.2 model card
- Meredith & Mantel — *Project Management: A Managerial Approach* (the source text)

---

## Authors

- **Vaishnavi** 
- **Aditi Singh** 
- **Deep Bansal**

**Faculty mentor:** Richa Sharma
**Institution:** School of Computer Science Engineering and Technology, Bennett University

---
