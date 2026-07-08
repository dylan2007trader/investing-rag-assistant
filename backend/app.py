"""FastAPI server: serves the chat UI and exposes a /api/chat RAG endpoint.

    uvicorn app:app --reload --port 8000    (run from the backend/ directory)

Then open http://localhost:8000
"""

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import RagIndex, answer_question, INDEX_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(os.path.dirname(BASE_DIR), "frontend", "index.html")

app = FastAPI(title="RAG Conversational Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_index = None


def get_index():
    global _index
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise RuntimeError(
                "Index not found. Run `python backend/ingest.py` first."
            )
        _index = RagIndex.load(INDEX_PATH)
    return _index


class ChatRequest(BaseModel):
    question: str
    k: int = 4


class Source(BaseModel):
    source: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int
    embed_backend: str


@app.get("/")
def home():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    idx = get_index()
    return {"status": "ok", "chunks": len(idx.chunks), "embed_backend": idx.embedder.backend}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    idx = get_index()
    t0 = time.time()
    result = answer_question(idx, req.question, k=req.k)
    return ChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        latency_ms=int((time.time() - t0) * 1000),
        embed_backend=idx.embedder.backend,
    )
