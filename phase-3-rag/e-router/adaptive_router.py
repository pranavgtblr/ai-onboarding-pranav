"""Adaptive Query Router Interactive CLI (Project E - Task 3.19).

Demonstrates intelligent multi-target query routing:
1. DIRECT_LLM: Code generation, math, greetings, general knowledge.
2. LOCAL_CORPUS: Project Odyssey Mars Base engineering, ECLSS, telemetry.
3. WEB_SEARCH: Current events, breaking news, latest software releases.
"""

import argparse
import sys
import time

from phase_3_rag.router import (
    AdaptiveRAGRouter,
    RouteTarget,
    RoutingDecision,
    classify_route_heuristic,
)

BENCHMARK_SUITE = [
    # Category 1: Direct LLM
    (
        "Write a Python function to check if a word is a palindrome.",
        RouteTarget.DIRECT_LLM,
        "Coding task solvable directly by LLM parametric memory",
    ),
    (
        "Calculate 128 * 4 + 56 - 12.",
        RouteTarget.DIRECT_LLM,
        "Pure math evaluation; no retrieval required",
    ),
    (
        "Hello! How are you doing today?",
        RouteTarget.DIRECT_LLM,
        "Conversational greeting; zero retrieval needed",
    ),
    # Category 2: Local Corpus (Proprietary Mars Engineering)
    (
        "What is the nominal cabin atmospheric pressure limit for the ECLSS?",
        RouteTarget.LOCAL_CORPUS,
        "Internal Project Odyssey ECLSS engineering documentation",
    ),
    (
        "What propellant mixture does the MDAS propulsion system use?",
        RouteTarget.LOCAL_CORPUS,
        "Internal Mars descent and ascent propulsion specifications",
    ),
    (
        "What is the maximum safe operating voltage of the primary power bus?",
        RouteTarget.LOCAL_CORPUS,
        "Internal telemetry matrix and power bus limits",
    ),
    # Category 3: Web Search
    (
        "What are the latest features introduced in the newest Python 3.13 release?",
        RouteTarget.WEB_SEARCH,
        "Recent external software release requiring live web search",
    ),
    (
        "What is the current weather forecast for Tokyo today?",
        RouteTarget.WEB_SEARCH,
        "Volatile real-time weather query",
    ),
    (
        "Who won the latest Arsenal football match yesterday?",
        RouteTarget.WEB_SEARCH,
        "Recent sports score requiring external search retrieval",
    ),
]


def run_benchmark_demo(router: AdaptiveRAGRouter) -> None:
    """Run evaluation benchmark across 9 diverse questions."""
    print("\n" + "=" * 75)
    print(" PROJECT E — ADAPTIVE RAG ROUTER BENCHMARK (TASK 3.19)")
    print(" Evaluating Query Classification Across 3 Target Routes")
    print("=" * 75)

    correct = 0
    total = len(BENCHMARK_SUITE)

    for idx, (question, expected_route, note) in enumerate(BENCHMARK_SUITE, start=1):
        t0 = time.perf_counter()
        decision: RoutingDecision = router.route_query(question)
        latency_ms = (time.perf_counter() - t0) * 1000

        is_match = decision.route == expected_route
        if is_match:
            correct += 1

        status_icon = "✓ PASS" if is_match else "✗ FAIL"
        print(f"\n[{idx}/{total}] {status_icon} | Latency: {latency_ms:.1f}ms")
        print(f'    Question  : "{question}"')
        print(f"    Expected  : {expected_route.value} ({note})")
        pred_str = f"{decision.route.value} (Confidence: {decision.confidence:.2f})"
        print(f"    Predicted : {pred_str}")
        print(f"    Reasoning : {decision.reasoning}")

    accuracy = (correct / total) * 100
    print("\n" + "=" * 75)
    print(f" Benchmark Summary: {correct}/{total} Correct ({accuracy:.1f}% Accuracy)")
    print("=" * 75 + "\n")


def run_single_query(
    question: str,
    router: AdaptiveRAGRouter,
) -> None:
    """Execute routing and dispatch for a single query."""
    print("\n" + "=" * 75)
    print(" ADAPTIVE RAG ROUTER (TASK 3.19)")
    print("=" * 75)

    # 1. Show heuristic vs active decision
    heuristic = classify_route_heuristic(question)
    print(f'\n[?] Question: "{question}"')
    print(f"    Heuristic Route : {heuristic.route.value} ({heuristic.reasoning})")

    # 2. Run router dispatch
    resp = router.route_and_execute(question)

    print(f"\n[>] Chosen Route   : [{resp.decision.route.value}]")
    print(f"    Confidence     : {resp.decision.confidence:.2f}")
    print(f"    Needs Retrieval: {'YES' if resp.decision.needs_retrieval else 'NO'}")
    print(f"    Reasoning      : {resp.decision.reasoning}")
    print(f"    Retrieval Time : {resp.retrieval_time_ms} ms")

    print("\n" + "-" * 75)
    print(f"[=] Synthesized Answer ({resp.decision.route.value}):")
    print("-" * 75)
    print(resp.answer)
    print("-" * 75)

    if resp.sources:
        print("\n[*] Sources / References:")
        for s in resp.sources:
            print(f"    * {s}")
    else:
        print("\n[*] Sources: None (Answered directly without retrieval)")
    print("=" * 75 + "\n")


def interactive_loop(router: AdaptiveRAGRouter) -> None:
    """Interactive CLI loop."""
    print("\n" + "=" * 75)
    print(" Adaptive RAG Router - Interactive CLI (Project E)")
    print(" Routes to: [DIRECT_LLM] | [LOCAL_CORPUS] | [WEB_SEARCH]")
    print(" Type your question, or 'exit' / 'quit' to quit.")
    print("=" * 75 + "\n")

    while True:
        try:
            user_q = input("Router Question> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_q or user_q.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        try:
            resp = router.route_and_execute(user_q)
            badge = f"[{resp.decision.route.value}]"
            print(f"\n-> Route: {badge} | Latency: {resp.retrieval_time_ms}ms")
            print(f"-> Why  : {resp.decision.reasoning}")
            print("\nAnswer:\n" + resp.answer + "\n")
            if resp.sources:
                print("Sources:")
                for s in resp.sources:
                    print(f"  - {s}")
            print("-" * 50)
        except Exception as exc:
            print(f"\n[!] Error: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Query Router CLI (Project E - Task 3.19)"
    )
    parser.add_argument("--query", "-q", type=str, help="Question to route and answer.")
    parser.add_argument(
        "--demo", action="store_true", help="Run routing benchmark evaluation."
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive terminal mode."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock search provider for web search route.",
    )
    args = parser.parse_args()

    router = AdaptiveRAGRouter(use_mock_search=args.mock)

    if args.demo:
        run_benchmark_demo(router)
    elif args.query:
        run_single_query(args.query, router)
    elif args.interactive or len(sys.argv) == 1:
        interactive_loop(router)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
