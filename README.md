# Investing RAG Assistant

A production-shaped **Retrieval-Augmented Generation (RAG)** chatbot that answers
investing questions over a knowledge base and **cites its sources**. Built with a Python
(FastAPI) backend and a JavaScript chat frontend, with semantic vector search, a
grounded-generation layer, and an automated retrieval-evaluation harness.

The knowledge base covers investing fundamentals — order types, risk management, options,
diversification, taxes, valuation metrics (P/E, EPS, beta), a terms glossary, notable
large-cap stocks, financial research sites, and technical, fundamental, and macro analysis.
It answers only from these documents and cites them, and refuses questions outside the
knowledge base (or requests for advice) instead of guessing. The docs are swappable, so the
same pipeline works for any domain.

Video of the RAG Investing Asistant
https://www.loom.com/share/26e86e21065449369159bd949ac083e8
---

## Why this project

It demonstrates the core skills these AI/GenAI engineering roles ask for:

- **RAG pipeline** — document chunking → embeddings → vector retrieval → grounded, cited answers.
- **Conversational AI** — a chat interface backed by an LLM, answering only from retrieved context.
- **Python + JavaScript** — FastAPI backend, vanilla-JS frontend, clean service boundaries.
- **Semantic search / embeddings** — sentence-transformers vectors with cosine retrieval (TF-IDF fallback).
- **Testing & reliability of AI systems** — an eval harness measuring retrieval accuracy, CI-ready.
- **Cloud-ready architecture** — maps cleanly onto GCP (Vertex AI, Cloud Run, Dialogflow) — see below.

---

## Architecture

```
          ┌──────────────┐     POST /api/chat      ┌───────────────────────────┐
  User ── │  Chat UI     │ ──────────────────────▶ │  FastAPI backend          │
          │  (JS/HTML)   │ ◀────────────────────── │                           │
          └──────────────┘   answer + citations    │  1. embed the question    │
                                                    │  2. cosine search (top-k) │
                                                    │  3. build grounded prompt │
                                                    │  4. LLM generates + cites │
                                                    └───────────┬───────────────┘
                                                                │
                              ┌─────────────────────────────────┴───────────┐
                              │  Vector index (embeddings of doc chunks)     │
                              │  built offline by ingest.py from data/       │
                              └──────────────────────────────────────────────┘
```

---

## Quickstart (local, ~2 minutes)

```bash
# 1. install core deps
pip install -r requirements.txt

# 2. (optional) real semantic embeddings instead of TF-IDF
pip install sentence-transformers

# 3. (optional) generated cited answers instead of demo mode
cp .env.example .env        # then set ANTHROPIC_API_KEY

# 4. build the vector index from data/sample_docs
python backend/ingest.py

# 5. run the server, then open http://localhost:8000
cd backend && uvicorn app:app --reload --port 8000
```

**It runs with zero API keys and zero model downloads** — TF-IDF embeddings plus a
transparent "demo mode" that returns the top retrieved passage. Add
`sentence-transformers` for semantic vectors and an `ANTHROPIC_API_KEY` for fully
generated, cited answers. Nothing else changes.

---

## Evaluation

```bash
python eval/eval.py
```

Runs a set of question → expected-source cases and reports **top-3 retrieval
accuracy** and **keyword grounding**. It exits non-zero below an 80% threshold, so it
plugs straight into CI (GitHub Actions) as a quality gate on the RAG pipeline.

---

## Configuration

| Variable            | Purpose                                        | Default                     |
|---------------------|------------------------------------------------|-----------------------------|
| `EMBED_BACKEND`     | `auto` \| `st` (semantic) \| `tfidf`           | `auto`                      |
| `ANTHROPIC_API_KEY` | enables generated, cited answers               | *(unset → demo mode)*       |
| `ANTHROPIC_MODEL`   | generation model                               | `claude-3-5-sonnet-latest`  |

To use your own knowledge base, drop `.md`/`.txt` files into `data/sample_docs/`
and re-run `python backend/ingest.py`.

---

## Deploying on Google Cloud Platform

The code is structured so each piece maps onto a managed GCP service:

| Local component            | GCP production equivalent                                        |
|----------------------------|------------------------------------------------------------------|
| FastAPI app                | **Cloud Run** (containerize, autoscale to zero)                  |
| sentence-transformers      | **Vertex AI** `text-embedding` models                            |
| Anthropic generation call  | **Vertex AI** (Gemini) or Anthropic on Vertex                    |
| in-memory cosine index     | **Vertex AI Vector Search** for large corpora                    |
| chat UI                    | static hosting on Cloud Run or **Cloud Storage + CDN**           |
| eval harness               | **Cloud Build / GitHub Actions** quality gate                    |
| logging & analytics        | **BigQuery** sink for questions, sources, latency                |
| managed conversational front-end | **Dialogflow CX** webhook calling `/api/chat`             |

Swapping the embedding and generation functions in `backend/rag.py` for their Vertex
AI equivalents is the only code change needed; the retrieval and API layers stay the
same.

---

## Project layout

```
gcp-rag-assistant/
├── backend/
│   ├── app.py        FastAPI server: serves UI + /api/chat
│   ├── rag.py        chunking, embeddings, retrieval, cited generation
│   └── ingest.py     builds the vector index
├── frontend/
│   └── index.html    chat UI (vanilla JS)
├── data/sample_docs/ knowledge base (swap in your own)
├── eval/eval.py      retrieval-accuracy test harness
├── requirements.txt
└── .env.example
```

Built by Dylan Ackerman — github.com/dylan2007trader
