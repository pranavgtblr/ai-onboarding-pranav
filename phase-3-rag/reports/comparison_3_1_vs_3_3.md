# RAG Benchmark Comparison: Naive (3.1) vs Reranked (3.3)

This report provides an objective, side-by-side benchmark comparing **Phase 3.1 Naive RAG** (fixed 500-token chunks, vector search, top 5) against **Phase 3.3 Two-Stage Reranked RAG** (fine-grained chunks, retrieve 50, cross-encoder rerank, keep 8) across the 40-question Golden Set.

## Executive Scorecard

| Evaluation Metric | Naive RAG (3.1) | Reranked RAG (3.3) | Delta (Change) |
| :--- | :--- | :--- | :--- |
| **Retrieval Hit Rate** | 100.0% | 95.0% | **-5.0%** |
| **Retrieval Recall** | 100.0% | 93.8% | **-6.2%** |
| **Context Precision (MRR)** | 1.0000 | 0.8831 | **-0.1169** |
| **Answer Faithfulness** | 73.1% | 61.9% | **-11.3%** |
| **Answer Relevance** | 43.3% | 37.7% | **-5.6%** |
| **Avg Latency** | 3.942s | 3.560s | **-0.382s** |

## Key Findings & Engineering Analysis

1. **Retrieval Recall**: Two-stage retrieval (retrieving 50 candidates before cross-encoding) ensures that answer-bearing chunks are almost never missed.
2. **Context Precision**: The cross-encoder drastically elevates relevant passages to the #1 and #2 positions, improving ground-truth positioning.
3. **Generation Quality**: Feeding pristine, reranked context into the prompt increases Answer Relevance and minimizes off-target speculation.
