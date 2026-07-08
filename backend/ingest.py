"""Build the vector index from the documents in data/sample_docs and save it.

Run this once before starting the API (or any time you change the docs):

    python backend/ingest.py
"""

from rag import build_index, INDEX_PATH


def main():
    index = build_index()
    index.save(INDEX_PATH)
    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    main()
