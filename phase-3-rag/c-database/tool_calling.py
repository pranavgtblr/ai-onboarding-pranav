"""Interactive CLI runner for Database RAG via Structured Tool Calling (Task 3.15).

Demonstrates typed function exposure for live customer and appointment data.
The LLM only fills in arguments—it never writes raw SQL.

Usage:
    uv run python c-database/tool_calling.py \
        --question "What is Alice's next appointment?"
    uv run python c-database/tool_calling.py \
        --question "Where is Bob's latest order?"
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to pythonpath if running directly
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import httpx  # noqa: E402

from phase_3_rag.config import get_settings  # noqa: E402
from phase_3_rag.database import DEFAULT_DB_PATH, init_database  # noqa: E402
from phase_3_rag.database_tools import (  # noqa: E402
    run_database_tool_loop,
)


def main() -> None:
    """Run CLI demonstration of structured tool calling for Database RAG."""
    parser = argparse.ArgumentParser(
        description="Run Database RAG using Structured Tool Calling (Task 3.15)."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="What is Alice's next appointment and is it confirmed?",
        help="Customer support question to answer using structured tool calls.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Gemini model override (e.g. gemini-2.5-flash).",
    )
    parser.add_argument(
        "--session-customer-id",
        type=int,
        default=1,
        help="Authenticated customer ID (default: 1 for Alice).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database file.",
    )
    args = parser.parse_args()

    # Ensure database exists
    if not args.db_path.exists():
        print(f"Initializing database at: {args.db_path}")
        init_database(args.db_path)

    settings = get_settings()
    if not settings.gemini_api_key:
        print("ERROR: GEMINI_API_KEY is not set. Please set it in .env.")
        sys.exit(1)

    print("=" * 70)
    print("Database RAG Approach 2: Structured Tool Calling (Task 3.15/3.16)")
    print("=" * 70)
    print(f"Authenticated User: Customer ID {args.session_customer_id}")
    print(f"Question: {args.question}")
    print(f"Database: {args.db_path.resolve()}\n")

    with httpx.Client(timeout=30.0) as client:
        try:
            result = run_database_tool_loop(
                args.question,
                client=client,
                session_customer_id=args.session_customer_id,
                model=args.model,
                db_path=args.db_path,
            )
        except Exception as exc:
            print(f"Execution failed: {exc}")
            sys.exit(1)

    print("-" * 70)
    print(f"Tool Calls Executed ({len(result.tool_calls)}):")
    for idx, tc in enumerate(result.tool_calls, 1):
        print(f"  [{idx}] Tool: {tc.tool_name}")
        print(f"      Arguments: {json.dumps(tc.arguments)}")
        print(f"      Output:    {json.dumps(tc.output, indent=6)}")

    print("-" * 70)
    print(f"Final Answer:\n{result.final_answer}")
    print("=" * 70)


if __name__ == "__main__":
    main()
