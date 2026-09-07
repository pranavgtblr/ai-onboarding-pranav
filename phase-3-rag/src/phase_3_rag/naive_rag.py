"""Task 3.1: Naive RAG Baseline Control Pipeline.

Components:
- 20 documents in the corpus.
- Fixed 500-token chunks.
- Vector search with cosine similarity (gemini-embedding-001).
- Top 5 chunks retrieved and stuffed into the prompt.
- Generation grounded strictly in retrieved context.
- Serves as the control benchmark for all subsequent Phase 3 tasks.
"""

import argparse
import sys
import time

import httpx
from pydantic import BaseModel, Field

from phase_3_rag.chunking import chunk_corpus
from phase_3_rag.config import get_settings
from phase_3_rag.corpus import load_corpus
from phase_3_rag.vector_store import (
    SearchResult,
    VectorIndex,
    embed_single_text,
)


class RAGResponse(BaseModel):
    """Result of a Naive RAG execution."""

    query: str
    answer: str
    retrieved_chunks: list[SearchResult] = Field(
        description="Top-K chunks retrieved by vector search"
    )
    total_chunks_in_index: int
    prompt_tokens: int = 0
    answer_tokens: int = 0
    latency_seconds: float = 0.0


def format_rag_prompt(query: str, retrieved_chunks: list[SearchResult]) -> str:
    """Format prompt with top-k context chunks stuffed into the instructions."""
    context_blocks: list[str] = []
    for res in retrieved_chunks:
        c = res.chunk
        context_blocks.append(
            f"--- [Doc: {c.source_title} ({c.chunk_id}) | Score: {res.score:.4f}] ---\n"
            f"{c.text.strip()}\n"
        )

    context_str = "\n".join(context_blocks)

    return (
        "You are an AI mission operations assistant for Project Odyssey Mars Base.\n"
        "Answer the question using ONLY the provided technical context chunks below.\n"
        "If the information is not contained in the context, explicitly state that you "
        "do not have enough information from the provided documents. "
        "Do not extrapolate or speculate.\n\n"
        f"=== RETRIEVED CONTEXT (TOP {len(retrieved_chunks)}) ===\n"
        f"{context_str}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_llm_response(
    augmented_prompt: str,
    *,
    client: httpx.Client,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, int, int]:
    """Send augmented prompt to Gemini REST API and return response tokens."""
    settings = get_settings()
    effective_key = api_key or settings.gemini_api_key
    effective_model = model or settings.gemini_model

    if not effective_key:
        raise ValueError("Gemini API key is required.")

    url = f"{settings.gemini_base_url}/models/{effective_model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": effective_key,
    }
    payload = {"contents": [{"parts": [{"text": augmented_prompt}]}]}

    for attempt in range(5):
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code in (429, 503) and attempt < 4:
            retry_after = resp.headers.get("retry-after")
            wait_time = (
                float(retry_after) if retry_after else min(30.0, 3.0 * (2**attempt))
            )
            print(
                f"  ⚠️  [Gemini Rate Limit] {resp.status_code} received. "
                f"Backing off for {wait_time:.1f}s (attempt {attempt + 1}/5)..."
            )
            time.sleep(wait_time)
            continue
        resp.raise_for_status()
        data = resp.json()
        break
    else:
        raise RuntimeError("Failed to obtain response after retries")

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No response generated: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    answer_text = "".join(p.get("text", "") for p in parts if "text" in p)

    usage = data.get("usageMetadata", {})
    prompt_tokens = int(usage.get("promptTokenCount", 0))
    answer_tokens = int(usage.get("candidatesTokenCount", 0))

    return answer_text.strip(), prompt_tokens, answer_tokens


def execute_naive_rag(
    query: str,
    *,
    top_k: int = 5,
    use_cache: bool = True,
    client: httpx.Client | None = None,
    api_key: str | None = None,
    model: str | None = None,
    index: VectorIndex | None = None,
) -> RAGResponse:
    """Execute complete Naive RAG pipeline: retrieval + stuffing + generation.

    Args:
        query: User question to answer.
        top_k: Number of chunks to retrieve and stuff (default 5).
        use_cache: Whether to use cached vector embeddings on disk.
        client: Optional shared httpx.Client.
        api_key: Optional API key override.
        model: Optional model identifier override.
        index: Optional pre-built VectorIndex (useful for tests and benchmarks).

    Returns:
        RAGResponse: Synthesized answer, retrieved chunks, and metadata.
    """
    settings = get_settings()
    start_time = time.perf_counter()

    should_close = False
    if client is None:
        client = httpx.Client(timeout=settings.timeout_seconds)
        should_close = True

    try:
        # Step 1: Ensure corpus, chunking, and index are ready
        if index is None:
            docs = load_corpus()
            chunks = chunk_corpus(docs, chunk_size=500)
            index = VectorIndex.build(chunks, use_cache=use_cache, client=client)

        # Step 2: Embed user query
        query_vector = embed_single_text(query, client=client, api_key=api_key)

        # Step 3: Vector search - Top K nearest neighbor chunks
        retrieved = index.search(query_vector, top_k=top_k)

        # Step 4: Stuff retrieved chunks into augmented prompt
        augmented_prompt = format_rag_prompt(query, retrieved)

        # Step 5: Generate answer from LLM
        answer, p_tokens, a_tokens = generate_llm_response(
            augmented_prompt,
            client=client,
            model=model,
            api_key=api_key,
        )

        latency = time.perf_counter() - start_time

        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved,
            total_chunks_in_index=index.size(),
            prompt_tokens=p_tokens,
            answer_tokens=a_tokens,
            latency_seconds=round(latency, 3),
        )

    finally:
        if should_close:
            client.close()


def main() -> None:
    """CLI entrypoint for Naive RAG demonstration."""
    parser = argparse.ArgumentParser(
        description="Phase 3.1: Naive RAG Baseline (Vector Search + Stuffing)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="When was Protocol Omega established and what triggers it?",
        help="Query to ask the RAG system.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve and stuff into prompt (default 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed retrieved chunk snippets and similarity scores.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass disk embedding cache and re-compute embeddings.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 78)
    print(" 🚀 TASK 3.1: NAIVE RAG BASELINE (CONTROL)")
    print("=" * 78)
    print(f"Query         : '{args.query}'")
    print(f"Top-K Chunks  : {args.top_k}")
    print("Chunk Size    : 500 tokens (fixed)")
    print("Corpus Size   : 20 technical documents")
    print("-" * 78)

    try:
        print("🔍 Indexing & Retrieving top chunks...")
        result = execute_naive_rag(
            query=args.query,
            top_k=args.top_k,
            use_cache=not args.no_cache,
        )

        print("\n" + "=" * 78)
        print(
            f" 📑 RETRIEVED TOP {len(result.retrieved_chunks)} CHUNKS (VECTOR SEARCH)"
        )
        print("=" * 78)
        for res in result.retrieved_chunks:
            c = res.chunk
            print(
                f"[Rank {res.rank}] Score: {res.score:.4f} | "
                f"ID: {c.chunk_id} ({c.token_count} tok) | Source: '{c.source_title}'"
            )
            if args.verbose:
                preview = c.text.strip()[:180].replace("\n", " ")
                print(f"       Preview: {preview}...")

        print("\n" + "=" * 78)
        print(" 🎯 SYNTHESIZED GROUNDED ANSWER")
        print("=" * 78)
        print(result.answer)
        print("-" * 78)
        print(f"Prompt Tokens  : {result.prompt_tokens}")
        print(f"Answer Tokens  : {result.answer_tokens}")
        print(f"Total Latency  : {result.latency_seconds}s")
        print(f"Indexed Chunks : {result.total_chunks_in_index}")
        print("=" * 78 + "\n")

    except Exception as exc:
        print(f"\n❌ Error executing Naive RAG: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
