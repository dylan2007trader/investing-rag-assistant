"""Tiny evaluation harness for the RAG pipeline.

Demonstrates functional testing of an AI/ML system:
  - retrieval quality: does the correct source document rank in the top-k?
  - keyword grounding: does the retrieved context contain the expected fact?

Run:  python eval/eval.py
Exits non-zero if retrieval accuracy drops below the threshold, so it can be
wired into CI (GitHub Actions).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from rag import build_index  # noqa: E402

# (question, expected source doc, a keyword that should appear in retrieved text)
TEST_CASES = [
    ("What is a stop-loss order?", "order-types.md", "stop"),
    ("How long must I hold an asset for long-term capital gains?", "taxes.md", "one year"),
    ("What is the wash sale rule?", "taxes.md", "30 days"),
    ("What is a call option?", "options-basics.md", "right to buy"),
    ("How does diversification reduce risk?", "diversification.md", "correlation"),
    ("What is position sizing?", "risk-management.md", "position"),
    ("What is the P/E ratio?", "valuation-metrics.md", "earnings"),
    ("What is the ticker symbol for Apple?", "notable-stocks.md", "AAPL"),
    ("Where can I find official company filings?", "research-tools.md", "EDGAR"),
    ("What is an ETF?", "glossary.md", "exchange"),
    ("What does the RSI indicator measure?", "technical-analysis.md", "overbought"),
    ("What is on a company's balance sheet?", "fundamental-analysis.md", "assets"),
    ("How do interest rates affect the market?", "economic-indicators.md", "Federal Reserve"),
]

THRESHOLD = 0.80  # require 80% top-3 retrieval accuracy


def main():
    index = build_index()
    hits, grounded, total = 0, 0, len(TEST_CASES)
    print(f"\nRunning {total} eval cases (embeddings: {index.embedder.backend})\n" + "-" * 60)

    for q, expected_source, keyword in TEST_CASES:
        results = index.search(q, k=3)
        top_sources = [r["source"] for r in results]
        joined = " ".join(r["text"] for r in results)
        hit = expected_source in top_sources
        has_kw = keyword.lower() in joined.lower()
        hits += hit
        grounded += has_kw
        flag = "PASS" if hit else "MISS"
        print(f"[{flag}] {q}")
        print(f"       expected={expected_source:22s} top3={top_sources} keyword={'ok' if has_kw else 'MISSING'}")

    acc = hits / total
    ground = grounded / total
    print("-" * 60)
    print(f"Retrieval top-3 accuracy: {acc:.0%}   |   Keyword grounding: {ground:.0%}")

    if acc < THRESHOLD:
        print(f"FAIL: accuracy {acc:.0%} below threshold {THRESHOLD:.0%}")
        sys.exit(1)
    print("OK: retrieval accuracy meets threshold.")


if __name__ == "__main__":
    main()
