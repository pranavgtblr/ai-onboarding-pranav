"""Web Search RAG Interactive Tool (Project D - Task 3.17).

Demonstrates:
1. Query Rewriting: LLM-powered and heuristic transformation of conversational
   human questions into search-optimized keyword queries.
2. Search API Retrieval: Querying DuckDuckGo or Mock providers.
3. Grounded Synthesis: Generating answers with inline markdown citations.
"""

import argparse
import sys

from phase_3_rag.web_search import (
    DuckDuckGoSearchProvider,
    MockSearchProvider,
    WebSearchRAG,
    heuristic_rewrite_query,
    rewrite_search_query,
)


def run_single_query(
    question: str,
    *,
    use_mock: bool = False,
    num_results: int = 5,
) -> None:
    """Execute Web Search RAG for a single question and display rich output."""
    provider = MockSearchProvider() if use_mock else DuckDuckGoSearchProvider()
    rag = WebSearchRAG(search_provider=provider)

    print("\n" + "=" * 70)
    print(" PROJECT D — WEB SEARCH RAG (TASK 3.17)")
    print("=" * 70)

    # 1. Show query rewriting breakdown
    print(f'\n[?] Original User Question:\n    "{question}"')

    heuristic = heuristic_rewrite_query(question)
    rewritten = rewrite_search_query(question)

    print("\n[>] Query Rewriting Comparison:")
    print(f"    - Heuristic Search Query : '{heuristic.search_query}'")
    print(f"    - LLM Rewritten Query    : '{rewritten.search_query}'")
    if rewritten.alternative_queries:
        print(f"    - Alternative Queries    : {rewritten.alternative_queries}")
    print(f"    - Detected Intent        : {rewritten.intent}")
    print(f"    - Transformation Rationale: {rewritten.explanation}")

    # 2. Execute RAG
    provider_name = provider.__class__.__name__
    print(f"\n[Retrieving top {num_results} search hits from {provider_name}...]")
    response = rag.query(question, num_results=num_results)

    # 3. Show retrieved search results
    print(f"\n[+] Retrieved Web Search Hits ({len(response.search_results)} found):")
    if not response.search_results:
        print("    (No results returned)")
    for r in response.search_results:
        print(f"    [{r.rank}] {r.title}")
        print(f"        URL    : {r.url}")
        print(f"        Snippet: {r.snippet[:140]}...")

    # 4. Show synthesized answer
    print("\n" + "-" * 70)
    print("[=] Grounded Answer with Citations:")
    print("-" * 70)
    print(response.answer)
    print("-" * 70)

    print("\n[*] Sources Cited:")
    for url in response.citations:
        print(f"    * {url}")
    print("=" * 70 + "\n")


def interactive_loop(*, use_mock: bool = False, num_results: int = 5) -> None:
    """Interactive question-answering CLI."""
    provider = MockSearchProvider() if use_mock else DuckDuckGoSearchProvider()
    rag = WebSearchRAG(search_provider=provider)

    print("\n" + "=" * 70)
    print(" Web Search RAG - Interactive CLI (Project D)")
    print(" Type your question below, or 'exit' / 'quit' to end.")
    print("=" * 70 + "\n")

    while True:
        try:
            prompt = input("Search Question> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not prompt or prompt.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            resp = rag.query(prompt, num_results=num_results)
            print(f"\n-> Search Engine Query : '{resp.rewritten_query.search_query}'")
            print(f"-> Hits Retrieved      : {len(resp.search_results)}")
            print("\nAnswer:\n" + resp.answer + "\n")
            if resp.citations:
                print("Citations:")
                for url in resp.citations:
                    print(f"  - {url}")
            print("-" * 50)
        except Exception as exc:
            print(f"\n[!] Error processing query: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project D: Web Search RAG CLI (Task 3.17)"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="One-shot question to ask the Web Search RAG pipeline.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive multi-turn question answering loop.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock search provider instead of live DuckDuckGo.",
    )
    parser.add_argument(
        "--num-results",
        "-n",
        type=int,
        default=5,
        help="Number of web search results to retrieve (default: 5).",
    )
    args = parser.parse_args()

    if args.query:
        run_single_query(args.query, use_mock=args.mock, num_results=args.num_results)
    elif args.interactive or len(sys.argv) == 1:
        interactive_loop(use_mock=args.mock, num_results=args.num_results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
