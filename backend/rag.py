"""
Core RAG pipeline: chunking, embeddings, semantic retrieval, and grounded
generation with source citations.

Embedding backend is pluggable:
  - "st"    -> sentence-transformers (true semantic embeddings, recommended)
  - "tfidf" -> scikit-learn TF-IDF (zero-download fallback, always works)
  - "auto"  -> use sentence-transformers if installed, else fall back to TF-IDF

Set EMBED_BACKEND in your environment to force one. Generation uses the
Anthropic API when ANTHROPIC_API_KEY is set, and a transparent "demo mode"
(extractive answer) otherwise, so the app runs even with no keys.
"""

import os
import re
import glob
import pickle

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.pkl")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "sample_docs")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text, max_chars=800, overlap=150):
    """Pack paragraphs into ~max_chars chunks; hard-split anything oversized."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
                cur = ""
            if len(p) > max_chars:
                step = max_chars - overlap
                for i in range(0, len(p), step):
                    chunks.append(p[i : i + max_chars])
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class Embedder:
    def __init__(self, backend=None):
        self.backend = backend or os.getenv("EMBED_BACKEND", "auto")
        self.model = None
        self.vectorizer = None

        if self.backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend = "st"
                return
            except Exception:
                if self.backend == "st":
                    raise
                self.backend = "tfidf"

        # tfidf fallback
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.backend = "tfidf"

    def fit_transform(self, texts):
        if self.backend == "st":
            return self._normalize(self.model.encode(texts, convert_to_numpy=True))
        X = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        return self._normalize(X)

    def transform(self, texts):
        if self.backend == "st":
            return self._normalize(self.model.encode(texts, convert_to_numpy=True))
        X = self.vectorizer.transform(texts).toarray().astype("float32")
        return self._normalize(X)

    @staticmethod
    def _normalize(X):
        X = np.asarray(X, dtype="float32")
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms


# --------------------------------------------------------------------------- #
# Vector index (cosine similarity over a small in-memory matrix)
# --------------------------------------------------------------------------- #
class RagIndex:
    def __init__(self, embedder, chunks, metadata, vectors):
        self.embedder = embedder
        self.chunks = chunks
        self.metadata = metadata
        self.vectors = vectors

    def search(self, query, k=4):
        q = self.embedder.transform([query])[0]
        sims = self.vectors @ q
        order = np.argsort(-sims)[:k]
        return [
            {
                "text": self.chunks[i],
                "source": self.metadata[i]["source"],
                "score": round(float(sims[i]), 4),
            }
            for i in order
        ]

    def save(self, path=INDEX_PATH):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "backend": self.embedder.backend,
                    "vectorizer": self.embedder.vectorizer,
                    "chunks": self.chunks,
                    "metadata": self.metadata,
                    "vectors": self.vectors,
                },
                f,
            )

    @staticmethod
    def load(path=INDEX_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        emb = Embedder(backend=data["backend"])
        emb.vectorizer = data["vectorizer"]
        return RagIndex(emb, data["chunks"], data["metadata"], data["vectors"])


def build_index(data_dir=DATA_DIR):
    """Read all .md/.txt files in data_dir, chunk, embed, and return a RagIndex."""
    chunks, metadata = [], []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.md")) + glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No documents found in {data_dir}")
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        for c in chunk_text(text):
            chunks.append(c)
            metadata.append({"source": source})

    embedder = Embedder()
    vectors = embedder.fit_transform(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(paths)} docs "
          f"using '{embedder.backend}' embeddings.")
    return RagIndex(embedder, chunks, metadata, vectors)


# --------------------------------------------------------------------------- #
# Generation (grounded, cited)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a helpful investing-education assistant. "
    "Answer the user's question using ONLY the numbered context passages provided. "
    "Cite the passages you use with their bracket numbers, e.g. [1], [2]. "
    "If the answer is not contained in the context, say you don't have that "
    "information and suggest consulting a licensed financial professional. "
    "This is general educational information, not financial advice. Keep answers concise."
)


def generate_answer(query, retrieved):
    context = "\n\n".join(
        f"[{i + 1}] (source: {r['source']})\n{r['text']}"
        for i, r in enumerate(retrieved)
    )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
            msg = client.messages.create(
                model=model,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    }
                ],
            )
            return msg.content[0].text
        except Exception as e:  # fall through to demo mode on any API error
            return (
                f"(LLM call failed: {e}. Showing top retrieved passage instead.)\n\n"
                f"{retrieved[0]['text']}"
            )

    # Demo mode — no API key required so the app always runs.
    top = retrieved[0]
    return (
        "(Demo mode — set ANTHROPIC_API_KEY to get a fully generated, cited answer.)\n\n"
        f"Most relevant passage [1] from {top['source']}:\n\n{top['text']}"
    )


def answer_question(index, query, k=4):
    retrieved = index.search(query, k=k)
    answer = generate_answer(query, retrieved)
    return {"answer": answer, "sources": retrieved}
