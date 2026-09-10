"""Omni-Router Chatbot CLI (Project E - Task 3.21).

One unified chatbot that:
1. Routes across all four sources (DIRECT_LLM, LOCAL_CORPUS, STRUCTURED_DB, WEB_SEARCH).
2. Synthesizes answers grounded with citations.
3. Explicitly tells the user which knowledge source it used and why.
"""

from __future__ import annotations

import argparse
import sys

from phase_3_rag.router import (
    AdaptiveRAGRouter,
)


def run_interactive(router: AdaptiveRAGRouter) -> None:
    """Multi-turn interactive terminal chatbot loop."""
    print("\n" + "=" * 75)
    print(" 🪐 OMNI-ROUTER CHATBOT — 4-SOURCE INTELLIGENT ROUTING (TASK 3.21)")
    print("=" * 75)
    print(" Available Knowledge Modalities:")
    print("  [1] DIRECT_LLM    - Code generation, math, logic, greetings")
    print("  [2] LOCAL_CORPUS  - Project Odyssey Mars Base engineering & ECLSS")
    print("  [3] STRUCTURED_DB - Customers, orders, products, appointments (SQL)")
    print("  [4] WEB_SEARCH    - Live web search, latest releases, real-time news")
    print("-" * 75)
    print(" Type your question below, or 'exit' / 'quit' to end.\n")

    while True:
        try:
            prompt = input("Omni-Bot> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not prompt or prompt.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            resp = router.route_and_execute(prompt)
            print("\n" + "─" * 75)
            print(f"🧭 SOURCE USED : [{resp.source_used.value}] ({resp.source_label})")
            print(f"⏱️  LATENCY     : {resp.retrieval_time_ms} ms")
            print(f"🎯 CONFIDENCE  : {resp.decision.confidence * 100:.1f}%")
            print(f"💡 RATIONALE   : {resp.decision.reasoning}")
            print("─" * 75)
            print(f"\n{resp.answer}\n")

            if resp.citations:
                print("📚 CITATIONS:")
                for c in resp.citations:
                    print(f"  • {c}")
            else:
                print("📚 CITATIONS: None (Parametric memory)")
            print("=" * 75 + "\n")
        except Exception as exc:
            print(f"\n[!] Error processing query: {exc}\n")


def run_single(query: str, router: AdaptiveRAGRouter) -> None:
    """One-shot query answering with source transparency and citations."""
    resp = router.route_and_execute(query)

    print("\n" + "=" * 75)
    print(f"QUESTION: {resp.question}")
    print("=" * 75)
    print(
        f"SOURCE USED : [{resp.source_used.value}] ({resp.source_label})\n"
        f"LATENCY     : {resp.retrieval_time_ms} ms\n"
        f"CONFIDENCE  : {resp.decision.confidence * 100:.1f}%\n"
        f"RATIONALE   : {resp.decision.reasoning}"
    )
    print("-" * 75)
    print(f"ANSWER:\n{resp.answer}")
    print("-" * 75)
    if resp.citations:
        print("CITATIONS:")
        for c in resp.citations:
            print(f" • {c}")
    else:
        print("CITATIONS: None (Parametric memory)")
    print("=" * 75 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Omni-Router Chatbot across 4 sources (Task 3.21)"
    )
    parser.add_argument("--query", "-q", type=str, help="One-shot question.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock search provider for web route.",
    )
    args = parser.parse_args()

    router = AdaptiveRAGRouter(use_mock_search=args.mock)

    if args.query:
        run_single(args.query, router)
    elif len(sys.argv) == 1 or not args.query:
        run_interactive(router)


if __name__ == "__main__":
    main()
