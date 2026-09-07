"""Cited RAG pipeline ensuring verifiable source URLs and chunk IDs.

Every synthesized answer maps each claim to its exact source chunk,
document title, and canonical URL.
"""

import argparse
import sys

import httpx

from phase_3_rag.bm25 import BM25Index
from phase_3_rag.chunking import Chunk, chunk_corpus
from phase_3_rag.citations import (
    CitedAnswer,
    extract_and_validate_citations,
    format_cited_prompt,
)
from phase_3_rag.corpus import load_corpus
from phase_3_rag.naive_rag import generate_llm_response


def execute_cited_rag(
    query: str,
    chunks: list[Chunk],
    bm25_index: BM25Index,
    *,
    client: httpx.Client,
    top_k: int = 5,
) -> tuple[CitedAnswer, int, int]:
    """Retrieve relevant context, generate answer, and extract citations.

    Args:
        query: User input question.
        chunks: All corpus chunks.
        bm25_index: Populated BM25 keyword index.
        client: HTTP client for API completion.
        top_k: Number of reference chunks to stuff into prompt.

    Returns:
        Tuple of (CitedAnswer, prompt_tokens, completion_tokens).
    """
    # 1. Retrieve top-k chunks
    search_results = bm25_index.search(query, top_k=top_k)
    retrieved_chunks = [r.chunk for r in search_results]

    # 2. Format prompt with strict citation rules
    prompt = format_cited_prompt(query, retrieved_chunks)

    # 3. Call LLM for generation
    raw_answer, prompt_toks, comp_toks = generate_llm_response(prompt, client=client)

    # 4. Extract, cross-reference, and validate citations against retrieved context
    cited_answer = extract_and_validate_citations(
        raw_answer, retrieved_chunks, query=query
    )

    return cited_answer, prompt_toks, comp_toks


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cited RAG."""
    parser = argparse.ArgumentParser(
        description="Cited RAG: Grounded answers with verified chunk IDs and URLs",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What are the liquid methane and oxygen storage specs for the ERV?",
        help="Query string to answer with citations",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display prompt and raw LLM response with citation tags",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for the cited-rag CLI tool."""
    args = parse_args()

    print("\n🚀 [Cited RAG] Loading Project Odyssey technical corpus...")
    docs = load_corpus()
    chunks = chunk_corpus(docs, chunk_size=500)
    print(f"📦 Loaded {len(docs)} documents and produced {len(chunks)} chunks.")

    print(f"🔍 Building BM25 index across all {len(chunks)} chunks...")
    bm25_index = BM25Index(chunks)

    with httpx.Client(timeout=60.0) as client:
        print(f"\n❓ Query: '{args.query}'")
        print(f"🎯 Retrieving top {args.top_k} chunks and generating cited answer...")

        cited_answer, p_toks, c_toks = execute_cited_rag(
            args.query, chunks, bm25_index, client=client, top_k=args.top_k
        )

        if args.verbose:
            search_results = bm25_index.search(args.query, top_k=args.top_k)
            retrieved = [r.chunk for r in search_results]
            print("\n[STUFFED PROMPT WITH CITATION RULES]")
            print(format_cited_prompt(args.query, retrieved))
            print("\n[RAW LLM RESPONSE]")
            print(cited_answer.raw_answer)

        print("\n" + "=" * 60)
        print("🎯 [GROUNDED ANSWER WITH VERIFIED CITATIONS]")
        print("=" * 60)
        print(cited_answer.format_display())
        print("=" * 60)
        print(f"📊 Tokens: {p_toks} prompt + {c_toks} completion")
        print(f"✅ Citations Verified: {cited_answer.is_fully_verified}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
