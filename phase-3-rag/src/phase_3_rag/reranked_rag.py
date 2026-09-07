"""Two-stage RAG pipeline with Cross-Encoder Reranker.

Stage 1: Retrieve 50 candidates using fast first-stage retriever (BM25 / Hybrid).
Stage 2: Cross-Encoder evaluates all (query, chunk) pairs simultaneously.
Stage 3: Keep top 8 highest-scoring chunks and stuff into LLM generation prompt.
"""

import argparse
import sys

import httpx

from phase_3_rag.bm25 import BM25Index
from phase_3_rag.chunking import Chunk, chunk_corpus_fine
from phase_3_rag.corpus import load_corpus
from phase_3_rag.naive_rag import generate_llm_response
from phase_3_rag.reranker import (
    BaseReranker,
    GeminiCrossEncoder,
    HeuristicCrossEncoder,
    RerankResult,
)


def format_reranked_prompt(query: str, top_chunks: list[RerankResult]) -> str:
    """Format final generation prompt with top 8 reranked chunks."""
    context_blocks = []
    for item in top_chunks:
        c = item.chunk
        header = (
            f"--- [Doc: {c.source_title} ({c.doc_id}) | Chunk: {c.chunk_id} | "
            f"Cross-Score: {item.relevance_score:.4f} | "
            f"Rank #{item.reranked_rank} (was #{item.original_rank})] ---"
        )
        context_blocks.append(f"{header}\n{c.text.strip()}\n")

    context_str = "\n".join(context_blocks)

    return (
        "You are an AI mission operations assistant for Project Odyssey Mars Base.\n"
        "Answer the question using ONLY the provided technical context chunks below.\n"
        "If the information is not contained in the context, explicitly state that you "
        "do not have enough information from the provided documents. "
        "Do not extrapolate or speculate.\n\n"
        f"=== TOP {len(top_chunks)} CONTEXT CHUNKS (RERANKED VIA CROSS-ENCODER) ===\n"
        f"{context_str}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def execute_reranked_rag(
    query: str,
    chunks: list[Chunk],
    bm25_index: BM25Index,
    reranker: BaseReranker,
    *,
    client: httpx.Client,
    candidate_k: int = 50,
    top_k: int = 8,
) -> tuple[str, list[Chunk], list[RerankResult], int, int]:
    """Execute two-stage retrieve-then-rerank RAG pipeline.

    Args:
        query: User input prompt or question.
        chunks: All corpus chunks.
        bm25_index: Populated BM25 index for Stage 1.
        reranker: Cross-Encoder reranker instance for Stage 2.
        client: HTTP client for generation.
        candidate_k: Number of candidates to retrieve in Stage 1 (default: 50).
        top_k: Number of elite candidates to retain in Stage 2 (default: 8).

    Returns:
        Tuple of (answer, stage1_candidates, stage2_results,
        prompt_tokens, answer_tokens).
    """
    # Stage 1: Fast high-recall retrieval of 50 candidates
    stage1_results = bm25_index.search(query, top_k=candidate_k)
    candidates = [r.chunk for r in stage1_results]

    # Stage 2: Deep cross-attention reranking
    reranked = reranker.rerank(query, candidates, top_k=top_k)

    # Stage 3: Generation with Top 8 elite chunks
    prompt = format_reranked_prompt(query, reranked)
    answer, prompt_toks, comp_toks = generate_llm_response(prompt, client=client)

    return answer, candidates, reranked, prompt_toks, comp_toks


def print_rerank_comparison(
    candidates: list[Chunk],
    reranked: list[RerankResult],
    display_limit: int = 8,
) -> None:
    """Print side-by-side comparison showing rank movements after cross-encoding."""
    print("\n" + "=" * 80)
    print("STAGE 1 vs STAGE 2 COMPARISON: RETRIEVE 50 -> RERANK -> KEEP 8")
    print("=" * 80)

    print(f"\n[1] STAGE 1 RETRIEVAL (Top {display_limit} of {len(candidates)}):")
    for i, chunk in enumerate(candidates[:display_limit], start=1):
        print(f"  Rank #{i:2d}: {chunk.chunk_id:14} | {chunk.source_title}")

    print(f"\n[2] STAGE 2 CROSS-ENCODER RERANKED (Top {len(reranked)} Kept):")
    for r in reranked:
        shift = r.original_rank - r.reranked_rank
        shift_str = f"+{shift}" if shift > 0 else f"{shift}"
        print(
            f"  Rank #{r.reranked_rank:2d}: {r.chunk.chunk_id:14} | "
            f"Score: {r.relevance_score:6.4f} | (Original: #{r.original_rank:2d} "
            f"[{shift_str:>3s}]) | {r.chunk.source_title}"
        )
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reranked RAG."""
    parser = argparse.ArgumentParser(
        description="Two-Stage RAG: Retrieve 50 candidates, Rerank, Keep 8",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is the emergency backup battery runtime?",
        help="Query to search and answer",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Number of candidates to retrieve in Stage 1 (default: 50)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=8,
        help="Number of elite candidates to keep after reranking (default: 8)",
    )
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Force use of local heuristic cross-encoder",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison between Stage 1 retrieval and Stage 2 reranking",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose context chunks and prompt breakdown",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for the reranked-rag CLI tool."""
    args = parse_args()

    print("\n🚀 [Reranked RAG] Loading Project Odyssey technical corpus...")
    docs = load_corpus()
    chunks = chunk_corpus_fine(docs, chunk_size=120, overlap=20)
    print(f"📦 Loaded {len(docs)} documents ({len(chunks)} fine-grained chunks).")

    print(f"🔍 Building Stage 1 BM25 Index across all {len(chunks)} chunks...")
    bm25_index = BM25Index(chunks)

    with httpx.Client(timeout=60.0) as client:
        reranker: BaseReranker
        if args.heuristic_only:
            print("⚡ Using local Heuristic Cross-Encoder...")
            reranker = HeuristicCrossEncoder()
        else:
            print("🧠 Initializing Gemini Cross-Encoder Reranker...")
            reranker = GeminiCrossEncoder(client=client)

        print(f"\n❓ Query: '{args.query}'")
        print(f"🎯 Stage 1: Retrieving {args.candidate_k} candidates...")
        print(f"🎯 Stage 2: Cross-Encoder reranking ({args.top_k} elite)...")

        (
            answer,
            candidates,
            reranked,
            prompt_toks,
            comp_toks,
        ) = execute_reranked_rag(
            args.query,
            chunks,
            bm25_index,
            reranker,
            client=client,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )

        if args.compare or args.verbose:
            print_rerank_comparison(candidates, reranked, display_limit=args.top_k)

        if args.verbose:
            print("\n[STUFFED PROMPT WITH TOP 8 RERANKED CHUNKS]")
            print(format_reranked_prompt(args.query, reranked))

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
