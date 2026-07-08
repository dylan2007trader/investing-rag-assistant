# Your 2–3 Hour Plan — Make This Yours & Ship It

The repo already runs. These steps turn it from "Claude wrote this" into "I built and
deployed this," which is what actually impresses a hiring manager. Do them in order.

## 0. Run it locally first (15 min)
- `pip install -r requirements.txt`
- `pip install sentence-transformers` (turns on real semantic embeddings)
- `python backend/ingest.py`
- `cd backend && uvicorn app:app --reload --port 8000` → open http://localhost:8000
- Ask a few questions. Confirm answers + source citations show up.

## 1. Add your Anthropic key (10 min)
- `cp .env.example .env`, paste your `ANTHROPIC_API_KEY` (you already have one from
  Ackerman Tools).
- Restart the server. Answers are now fully generated and cited — much stronger in a demo.

## 2. Make the knowledge base yours (30 min)
- Replace the files in `data/sample_docs/` with a domain YOU can speak to in an
  interview. Good options: a topic you know well, docs for one of your own projects,
  or a public dataset's documentation.
- Re-run `python backend/ingest.py`.
- Update the three example questions in `frontend/index.html` to match.
- **Understand every function in `backend/rag.py`** — you must be able to explain
  chunking, embeddings, cosine retrieval, and the grounded prompt. This is the part
  interviewers probe.

## 3. Tighten the eval (20 min)
- Edit `eval/eval.py` test cases to match your new docs.
- Run `python eval/eval.py` and get it to 100%. Being able to say "I wrote an eval
  harness that gates retrieval quality in CI" is a standout line.

## 4. Push to GitHub (20 min)
- `git init && git add . && git commit -m "RAG conversational assistant"`
- Create a repo on github.com/dylan2007trader and push.
- Confirm `.env` is gitignored (it is) — never commit your key.

## 5. Deploy so it has a live URL (30–45 min)
Pick the fastest path you're comfortable with:
- **Easiest:** deploy the FastAPI app to **Render** or **Railway** (free tier, connects
  to your GitHub repo, gives a public URL in minutes).
- **GCP version (best for these job descriptions):** containerize and deploy to
  **Cloud Run** — then you can literally say "deployed on GCP," which several of these
  listings ask for. The README's GCP table is your guide.

## 6. Record a 60–90s demo (20 min)
- Screen-record: ask 2–3 questions, show the cited sources, show the eval passing.
- Put the video link + live URL + repo link at the top of your README.

## What you can now honestly say
- "I built a RAG conversational assistant with semantic retrieval and source-grounded,
  cited answers."
- "Python/FastAPI backend, JS frontend, embeddings + vector search, and a CI eval
  harness that gates retrieval accuracy."
- "It's architected to deploy on GCP — Cloud Run, Vertex AI embeddings, and a
  Dialogflow CX front end." (and if you did step 5's GCP path: "and I deployed it there.")

## How this maps to the AI/GenAI job descriptions
- RAG pipelines for enterprise use cases ✓
- Conversational agents ✓
- Python + JavaScript backend services ✓
- Integrating AI/ML models into APIs ✓
- Semantic search / embeddings ✓
- Functional testing of AI/ML applications ✓
- GCP end-to-end delivery ✓ (architecture; real deploy if you do step 5's GCP path)
