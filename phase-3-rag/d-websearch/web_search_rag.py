"""Web Search RAG Interactive Tool (Project D - Task 3.17 & 3.18).

Demonstrates:
1. Query Rewriting: Transforms conversational human questions into search keywords.
2. Search API Retrieval: Querying DuckDuckGo or Mock providers.
3. Resilient Page Extraction (Task 3.18): Fetches destination URLs, handles dead
   links (404/500), paywalls (403/subscription gates), and junk/bot challenge pages
   without crashing, falling back gracefully to search snippets.
4. Grounded Synthesis: Generating answers with inline citations.
"""

import argparse
import sys

from phase_3_rag.web_search import (
    DuckDuckGoSearchProvider,
    FetchStatus,
    MockSearchProvider,
    PageContentExtractor,
    SearchResult,
    WebSearchRAG,
    heuristic_rewrite_query,
    rewrite_search_query,
    synthesize_web_answer,
)


def run_resilience_demo() -> None:
    """Demonstrate handling dead links, paywalls, and junk pages without crashing."""
    print("\n" + "=" * 75)
    print(" PROJECT D — TASK 3.18 RESILIENCE DEMONSTRATION")
    print(" Testing: Dead Links, Paywalls, and Junk Pages")
    print("=" * 75)

    extractor = PageContentExtractor()

    # 1. Test Dead Link (Simulated 404 / 500)
    print("\n[Case 1: Dead Link (HTTP 404 / Server Failure)]")
    dead_html = "<html><body><h1>404 Not Found</h1></body></html>"
    text, status, err = extractor.extract_html_text(dead_html)
    print(f"    - Extracted Text : {text}")
    print(f"    - Status Code    : [{status.value}]")
    print(f"    - Note           : {err}")

    # 2. Test Paywall Page
    print("\n[Case 2: Paywall Barrier (Subscription Gated)]")
    paywall_html = """
    <html>
      <body>
        <h1>Breaking Research: Quantum Computing Breakthrough</h1>
        <p>Researchers today announced a novel superconducting qubit topology...</p>
        <div class="paywall-banner">
          <h2>Subscribe to continue reading</h2>
          <p>This article is for subscribers only. Join now to read the full story.</p>
        </div>
      </body>
    </html>
    """
    text, status, err = extractor.extract_html_text(paywall_html)
    print(f"    - Extracted Text : {text}")
    print(f"    - Status Code    : [{status.value}]")
    print(f"    - Note           : {err}")

    # 3. Test Junk / Bot Challenge Page
    print("\n[Case 3: Junk / Cloudflare Bot Challenge Wall]")
    bot_html = """
    <html>
      <head><title>Just a moment...</title></head>
      <body>
        <h1>Attention Required! | Cloudflare</h1>
        <p>Please complete the security check to verify you are human.</p>
        <p>Cloudflare Ray ID: 8c34f9a12b7d002</p>
      </body>
    </html>
    """
    text, status, err = extractor.extract_html_text(bot_html)
    print(f"    - Extracted Text : {text}")
    print(f"    - Status Code    : [{status.value}]")
    print(f"    - Note           : {err}")

    # 4. Test Valid Page
    print("\n[Case 4: Valid Clean Article]")
    valid_html = """
    <html>
      <header><nav>Home | Products | Contact</nav></header>
      <main>
        <article>
          <h1>Python 3.13 Free-Threaded Build Guide</h1>
          <p>Python 3.13 introduces experimental free-threaded execution.
          This mode allows multiple threads to run concurrently across CPU cores
          without the Global Interpreter Lock (GIL).</p>
          <p>To enable free-threading, install python3.13t.</p>
        </article>
      </main>
      <footer>Copyright 2026 Python Foundation. Cookie Preferences.</footer>
    </html>
    """
    text, status, err = extractor.extract_html_text(valid_html)
    print(f"    - Extracted Text : {text[:100]}...")
    extracted_len = len(text or "")
    print(f"    - Status Code    : [{status.value}] ({extracted_len} chars)")

    # 5. Mixed Pipeline Execution
    print("\n[Case 5: End-to-End Pipeline with Mixed Sources]")
    mixed_results = [
        SearchResult(
            title="Official PEP 703 Specification",
            url="https://peps.python.org/pep-0703/",
            snippet="PEP 703 proposes making the GIL optional in CPython.",
            rank=1,
            page_content=(
                "PEP 703 proposes making the Global Interpreter Lock optional in "
                "the CPython implementation. This provides scalable multithreading."
            ),
            fetch_status=FetchStatus.SUCCESS,
        ),
        SearchResult(
            title="Dead Link Tech News Blog",
            url="https://example-dead-news.org/python-gil",
            snippet="Python removes the GIL in version 3.13 milestone release.",
            rank=2,
            page_content=None,
            fetch_status=FetchStatus.DEAD_LINK,
            fetch_error="HTTP 404 Not Found",
        ),
        SearchResult(
            title="Paywalled Wall Street Journal Article",
            url="https://example-paywall.com/tech/python-parallel",
            snippet="Major companies adopt free-threaded Python for parallel tasks.",
            rank=3,
            page_content=None,
            fetch_status=FetchStatus.PAYWALL,
            fetch_error="Paywall barrier detected",
        ),
    ]

    answer, citations = synthesize_web_answer(
        "What is free-threaded execution in Python 3.13?",
        mixed_results,
    )
    print("-" * 75)
    print("Synthesized Output with Mixed / Flaky Sources:")
    print(answer)
    print("-" * 75)
    print("Citations:")
    for c in citations:
        print(f" - {c}")
    print("=" * 75 + "\n")


def run_single_query(
    question: str,
    *,
    use_mock: bool = False,
    num_results: int = 5,
    fetch_pages: bool = True,
) -> None:
    """Execute Web Search RAG for a single question and display rich output."""
    provider = MockSearchProvider() if use_mock else DuckDuckGoSearchProvider()
    rag = WebSearchRAG(search_provider=provider, fetch_pages=fetch_pages)

    print("\n" + "=" * 70)
    print(" PROJECT D — WEB SEARCH RAG (TASK 3.17 & 3.18)")
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
    mode_desc = "with full-page extraction" if fetch_pages else "snippets only"
    print(
        f"\n[Retrieving top {num_results} search hits from {provider_name} "
        f"({mode_desc})...]"
    )
    response = rag.query(question, num_results=num_results, fetch_pages=fetch_pages)

    # 3. Show retrieved search results with fetch statuses
    print(f"\n[+] Retrieved Web Search Hits ({len(response.search_results)} found):")
    if not response.search_results:
        print("    (No results returned)")
    for r in response.search_results:
        status_tag = f"[{r.fetch_status.value}]"
        body_len = (
            f"{len(r.page_content)} chars"
            if r.page_content
            else f"snippet ({len(r.snippet)} chars)"
        )
        print(f"    [{r.rank}] {status_tag} {r.title} ({body_len})")
        print(f"        URL    : {r.url}")
        if r.fetch_error:
            print(f"        Notice : {r.fetch_error}")
        preview = r.effective_content[:140].replace("\n", " ")
        print(f"        Excerpt: {preview}...")

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


def interactive_loop(
    *,
    use_mock: bool = False,
    num_results: int = 5,
    fetch_pages: bool = True,
) -> None:
    """Interactive question-answering CLI."""
    provider = MockSearchProvider() if use_mock else DuckDuckGoSearchProvider()
    rag = WebSearchRAG(search_provider=provider, fetch_pages=fetch_pages)

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
            resp = rag.query(prompt, num_results=num_results, fetch_pages=fetch_pages)
            print(f"\n-> Search Engine Query : '{resp.rewritten_query.search_query}'")
            print(f"-> Hits Retrieved      : {len(resp.search_results)}")
            for r in resp.search_results:
                print(f"   [{r.fetch_status.value}] {r.title} -> {r.url}")
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
        description="Project D: Web Search RAG CLI (Task 3.17 & 3.18)"
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
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Disable page content fetching and use search snippets only.",
    )
    parser.add_argument(
        "--demo-resilience",
        action="store_true",
        help="Run live resilience demonstration on dead links, paywalls, and junk.",
    )
    args = parser.parse_args()

    if args.demo_resilience:
        run_resilience_demo()
    elif args.query:
        run_single_query(
            args.query,
            use_mock=args.mock,
            num_results=args.num_results,
            fetch_pages=not args.no_fetch,
        )
    elif args.interactive or len(sys.argv) == 1:
        interactive_loop(
            use_mock=args.mock,
            num_results=args.num_results,
            fetch_pages=not args.no_fetch,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
