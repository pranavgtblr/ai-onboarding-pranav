"""Hybrid RAG implementation combining BM25 keyword search with Vector search.

Uses Reciprocal Rank Fusion (RRF) to merge candidate lists and stuff the top
fused context into the LLM prompt.
"""

import argparse
import sys

import httpx

from phase_3_rag.bm25 import BM25Index, BM25SearchResult
from phase_3_rag.chunking import Chunk, chunk_corpus
from phase_3_rag.config import get_settings
from phase_3_rag.corpus import load_corpus
from phase_3_rag.fusion import DEFAULT_RRF_K, FusedResult, reciprocal_rank_fusion
from phase_3_rag.naive_rag import generate_llm_response
from phase_3_rag.vector_store import (
    SearchResult,
    VectorIndex,
    embed_single_text,
)


def format_hybrid_rag_prompt(query: str, retrieved_chunks: list[FusedResult]) -> str:
    """Format prompt with technical reference blocks and RRF scores."""
    context_blocks = []
    for item in retrieved_chunks:
        c = item.chunk
        v_str = (
            f"V-Rank #{item.vector_rank} (score {item.vector_score:.4f})"
            if item.vector_rank is not None and item.vector_score is not None
            else "V-Rank: None"
        )
        b_str = (
            f"B-Rank #{item.bm25_rank} (score {item.bm25_score:.2f})"
            if item.bm25_rank is not None and item.bm25_score is not None
            else "B-Rank: None"
        )

        header = (
            f"--- [Doc: {c.source_title} ({c.doc_id}) | Chunk: {c.chunk_id} | "
            f"RRF: {item.rrf_score:.5f} | {v_str} | {b_str}] ---"
        )
        context_blocks.append(f"{header}\n{c.text.strip()}\n")

    context_str = "\n".join(context_blocks)

    return (
        "You are an AI mission operations assistant for Project Odyssey Mars Base.\n"
        "Answer the question using ONLY the provided technical context chunks below.\n"
        "If the information is not contained in the context, explicitly state that you "
        "do not have enough information from the provided documents. "
        "Do not extrapolate or speculate.\n\n"
        f"=== RETRIEVED CONTEXT (TOP {len(retrieved_chunks)} VIA HYBRID RRF) ===\n"
        f"{context_str}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def execute_hybrid_rag(
    query: str,
    chunks: list[Chunk],
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    *,
    client: httpx.Client,
    top_k: int = 5,
    k_rrf: int = DEFAULT_RRF_K,
    candidate_k: int = 10,
) -> tuple[
    str,
    list[SearchResult],
    list[BM25SearchResult],
    list[FusedResult],
    int,
    int,
]:
    """Execute full hybrid retrieval, RRF ranking, and generation pipeline.

    Args:
        query: User input prompt or question.
        chunks: All corpus chunks.
        vector_index: Populated vector index.
        bm25_index: Populated BM25 keyword index.
        client: HTTP client for embeddings and generation.
        top_k: Number of final fused chunks to stuff into prompt.
        k_rrf: Smoothing constant for Reciprocal Rank Fusion.
        candidate_k: Number of candidates to pull from each ranker before fusion.

    Returns:
        Tuple of (generated_answer, vector_hits, bm25_hits, fused_hits,
        prompt_tokens, completion_tokens).
    """
    settings = get_settings()

    # 1. Semantic Vector Search
    query_vector = embed_single_text(
        query,
        client=client,
        model=settings.gemini_embedding_model,
        api_key=settings.gemini_api_key,
    )
    vector_results = vector_index.search(query_vector, top_k=candidate_k)

    # 2. Lexical BM25 Search
    bm25_results = bm25_index.search(query, top_k=candidate_k)

    # 3. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(
        vector_results, bm25_results, k=k_rrf, top_k=top_k
    )

    # 4. Prompt construction & LLM completion
    prompt = format_hybrid_rag_prompt(query, fused_results)
    answer, prompt_toks, comp_toks = generate_llm_response(prompt, client=client)

    return (
        answer,
        vector_results,
        bm25_results,
        fused_results,
        prompt_toks,
        comp_toks,
    )


def print_comparison_table(
    vector_results: list[SearchResult],
    bm25_results: list[BM25SearchResult],
    fused_results: list[FusedResult],
    limit: int = 5,
) -> None:
    """Print side-by-side comparison of Vector, BM25, and Fused ranks."""
    print("\n" + "=" * 80)
    print("HYBRID SEARCH COMPARISON: VECTOR vs BM25 vs RECIPROCAL RANK FUSION")
    print("=" * 80)

    print("\n[1] VECTOR SEARCH (Cosine Similarity):")
    for r in vector_results[:limit]:
        print(
            f"  Rank #{r.rank}: {r.chunk.chunk_id:12} | "
            f"Score: {r.score:.4f} | {r.chunk.source_title}"
        )

    print("\n[2] BM25 SEARCH (Lexical Keyword Match):")
    for r in bm25_results[:limit]:
        print(
            f"  Rank #{r.rank}: {r.chunk.chunk_id:12} | "
            f"Score: {r.score:6.2f} | {r.chunk.source_title}"
        )

    print("\n[3] FUSED RESULTS (Reciprocal Rank Fusion):")
    for r in fused_results[:limit]:
        v_info = f"V#{r.vector_rank}" if r.vector_rank else "V:None"
        b_info = f"B#{r.bm25_rank}" if r.bm25_rank else "B:None"
        print(
            f"  Rank #{r.rank}: {r.chunk.chunk_id:12} | "
            f"RRF: {r.rrf_score:.5f} ({v_info}, {b_info}) | {r.chunk.source_title}"
        )
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for hybrid RAG."""
    parser = argparse.ArgumentParser(
        description="Hybrid RAG: BM25 + Vector Search with Reciprocal Rank Fusion",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is the liquid methane storage capacity for the ERV?",
        help="Query string to search and answer",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of fused context chunks to stuff into prompt (default: 5)",
    )
    parser.add_argument(
        "--k-rrf",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF smoothing constant k (default: 60)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Display side-by-side comparison of Vector vs BM25 vs Fused rankings",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose context chunks and prompt breakdown",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass embedding disk cache and force live embedding generation",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for the hybrid-rag CLI tool."""
    args = parse_args()

    print("\n🚀 [Hybrid RAG] Loading Project Odyssey technical corpus...")
    docs = load_corpus()
    chunks = chunk_corpus(docs, chunk_size=500)
    print(f"📦 Loaded {len(docs)} documents and produced {len(chunks)} chunks.")

    with httpx.Client(timeout=60.0) as client:
        # Build Vector Index
        print("🧠 Loading vector index...")
        vector_index = VectorIndex.build(
            chunks, client=client, use_cache=not args.no_cache
        )

        # Build BM25 Index
        print("🔍 Building BM25 keyword index...")
        bm25_index = BM25Index(chunks)

        print(f"\n❓ Query: '{args.query}'")
        (
            answer,
            vector_hits,
            bm25_hits,
            fused_hits,
            prompt_toks,
            comp_toks,
        ) = execute_hybrid_rag(
            args.query,
            chunks,
            vector_index,
            bm25_index,
            client=client,
            top_k=args.top_k,
            k_rrf=args.k_rrf,
        )

        if args.compare or args.verbose:
            print_comparison_table(vector_hits, bm25_hits, fused_hits, limit=args.top_k)

        if args.verbose:
            print("\n[STUFFED PROMPT WITH FUSED CONTEXT]")
            print(format_hybrid_rag_prompt(args.query, fused_hits))

        print("\n" + "=" * 50)
        print("🎯 [GROUNDED ANSWER]")
        print("=" * 50)
        print(answer.strip())
        print("=" * 50)
        print(f"📊 Tokens: {prompt_toks} prompt + {comp_toks} completion\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
