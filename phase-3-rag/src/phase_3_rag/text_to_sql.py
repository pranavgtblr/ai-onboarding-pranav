"""Hardened Text-to-SQL pipeline for Project C (Database RAG).

Enforces:
1. Pure SELECT statements (blocks INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA).
2. Allowlisted tables only (customers, orders, order_items, products, appointments).
3. Mandatory LIMIT enforcement (injects or clamps to MAX_LIMIT).
4. Strict read-only database credentials (PRAGMA query_only = ON, mode=ro).
5. AST validation via sqlglot prior to execution (never string interpolation).
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import sqlglot
import sqlglot.expressions as exp
from pydantic import BaseModel, Field

from phase_3_rag.config import get_settings
from phase_3_rag.database import DEFAULT_DB_PATH, get_readonly_connection, init_database

ALLOWLISTED_TABLES: set[str] = {
    "customers",
    "orders",
    "order_items",
    "products",
    "appointments",
}

DEFAULT_MAX_LIMIT: int = 50

SCHEMA_PROMPT = """
You are a precision SQL analyst. Write SQLite queries against the schema
below to answer user questions.

### Database Schema (SQLite):
1. customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT NOT NULL,
    city TEXT NOT NULL,
    created_at TIMESTAMP
);

2. products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL
);

3. orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL, -- ('pending', 'processing', 'shipped', 'delivered')
    total_amount REAL NOT NULL,
    shipping_address TEXT NOT NULL
);

4. order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);

5. appointments (
    appointment_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    service_type TEXT NOT NULL,
    scheduled_time TEXT NOT NULL,
    status TEXT NOT NULL, -- ('scheduled', 'confirmed', 'completed', 'cancelled')
    notes TEXT
);

### Strict Rules:
- For customer names, always use substring matching:
  customers.name LIKE '%Name%' (e.g. '%Alice%') because customer records
  store full names.
- Join tables using foreign keys where needed
  (e.g. appointments.customer_id = customers.customer_id).
- Output ONLY the raw SQL query. No explanations, no markdown fences,
  no conversational prose.
"""


class SecurityValidationError(ValueError):
    """Raised when an incoming SQL query violates security policies."""


class TextToSqlResult(BaseModel):
    """Structured container for Text-to-SQL generation and execution results."""

    question: str
    generated_sql: str
    sanitized_sql: str
    row_count: int
    query_results: list[dict[str, Any]]
    answer: str
    execution_time_ms: float = Field(default=0.0)


def extract_sql_from_text(raw_text: str) -> str:
    """Strip markdown code fences and extraneous text from LLM response."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    # Remove leading/trailing semicolons and whitespace
    return cleaned.strip(" ;")


def validate_and_sanitize_sql(
    raw_sql: str,
    *,
    allowlisted_tables: set[str] = ALLOWLISTED_TABLES,
    max_limit: int = DEFAULT_MAX_LIMIT,
) -> str:
    """Validate SQL query against security policies and inject mandatory LIMIT.

    Security checks:
    1. Multi-statement injection rejection (semicolon chaining).
    2. Pure SELECT statement AST check (blocks INSERT, UPDATE, DELETE, DROP, etc.).
    3. Allowlisted tables verification (blocks sqlite_master, admin logs, etc.).
    4. Mandatory LIMIT enforcement (injects or clamps to max_limit).
    """
    clean_sql = extract_sql_from_text(raw_sql)
    if not clean_sql:
        raise SecurityValidationError("Empty SQL query provided.")

    # 1. Multi-statement injection check
    # Check if there are multiple non-comment statements separated by semicolons
    statements = [s.strip() for s in clean_sql.split(";") if s.strip()]
    if len(statements) > 1:
        raise SecurityValidationError(
            f"Multi-statement execution rejected "
            f"({len(statements)} statements detected)."
        )

    # 2. Parse via sqlglot AST
    try:
        parsed = sqlglot.parse_one(clean_sql, read="sqlite")
    except Exception as exc:
        raise SecurityValidationError(f"SQL syntax / parsing error: {exc}") from exc

    if not isinstance(parsed, exp.Select):
        raise SecurityValidationError(
            f"Only SELECT queries are permitted; received {type(parsed).__name__}."
        )

    # 3. Verify that no modifying expressions or dangerous functions exist
    prohibited_exp_types = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Command,
        exp.Pragma,
    )
    for node in parsed.walk():
        if isinstance(node, prohibited_exp_types):
            raise SecurityValidationError(
                f"Prohibited SQL expression detected: {type(node).__name__}."
            )

    # 4. Table allowlist check
    referenced_tables: set[str] = set()
    for table_exp in parsed.find_all(exp.Table):
        tbl_name = table_exp.name.lower()
        # Exclude subquery aliases or CTE aliases
        referenced_tables.add(tbl_name)

    disallowed_tables = referenced_tables - {t.lower() for t in allowlisted_tables}
    if disallowed_tables:
        raise SecurityValidationError(
            f"Access to non-allowlisted table(s) rejected: {sorted(disallowed_tables)}."
        )

    # 5. Mandatory LIMIT enforcement
    limit_exp = parsed.args.get("limit")
    if limit_exp is None:
        # Inject mandatory limit
        parsed = parsed.limit(max_limit)
    else:
        # Inspect existing limit expression
        try:
            limit_val_str = limit_exp.expression.name
            limit_val = int(limit_val_str)
            if limit_val > max_limit or limit_val <= 0:
                parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))
        except Exception:
            # If complex limit expression, clamp to max_limit
            parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))

    # Return clean, transpiled SQLite SQL
    return parsed.sql(dialect="sqlite")


def execute_readonly_sql(
    sql_query: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    allowlisted_tables: set[str] = ALLOWLISTED_TABLES,
    max_limit: int = DEFAULT_MAX_LIMIT,
) -> tuple[str, list[dict[str, Any]]]:
    """Validate, sanitize, and execute query using a strictly read-only connection."""
    sanitized_sql = validate_and_sanitize_sql(
        sql_query,
        allowlisted_tables=allowlisted_tables,
        max_limit=max_limit,
    )

    conn = get_readonly_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sanitized_sql)
        rows = [dict(r) for r in cursor.fetchall()]
        return sanitized_sql, rows
    finally:
        conn.close()


def generate_sql_for_question(
    question: str,
    *,
    client: httpx.Client,
    model: str | None = None,
) -> str:
    """Use Gemini LLM to generate a precision SQL query from natural language."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required for Text-to-SQL generation.")

    active_model = model or settings.gemini_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={settings.gemini_api_key}"
    )

    prompt = (
        f"{SCHEMA_PROMPT}\n\n"
        f"User Question: {question}\n\n"
        f"Generate the exact SQLite SQL query:"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 300},
    }

    resp = client.post(url, json=payload, timeout=30.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")

    data = resp.json()
    raw_sql = data["candidates"][0]["content"]["parts"][0]["text"]
    return extract_sql_from_text(raw_sql)


def synthesize_natural_answer(
    question: str,
    sql_query: str,
    results: list[dict[str, Any]],
    *,
    client: httpx.Client,
    model: str | None = None,
) -> str:
    """Synthesize a concise, natural language response based on verified
    query records.
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        return f"Found {len(results)} records matching your query."

    active_model = model or settings.gemini_model
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={settings.gemini_api_key}"
    )

    results_json = json.dumps(results, indent=2, default=str)
    prompt = (
        "You are a helpful database assistant. Answer the user's question accurately "
        "using ONLY the SQL query results provided below.\n\n"
        f"User Question: {question}\n"
        f"Executed SQL: {sql_query}\n"
        f"Database Results:\n{results_json}\n\n"
        "Instructions:\n"
        "- Answer directly and concisely.\n"
        "- Mention specific details like status, timestamps, prices, and IDs.\n"
        "- If no records were found, clearly state that no matching records were found."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
    }

    resp = client.post(url, json=payload, timeout=30.0)
    if resp.status_code != 200:
        return f"Executed query successfully. Found {len(results)} matching records."

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def run_text_to_sql_pipeline(
    question: str,
    *,
    client: httpx.Client,
    db_path: Path = DEFAULT_DB_PATH,
) -> TextToSqlResult:
    """End-to-end Text-to-SQL pipeline: Generate, Validate, Execute, Synthesize."""
    start_time = time.perf_counter()

    # 1. Generate SQL from question
    raw_sql = generate_sql_for_question(question, client=client)

    # 2. Validate, enforce LIMIT, and execute against read-only DB
    sanitized_sql, rows = execute_readonly_sql(raw_sql, db_path=db_path)

    # 3. Synthesize natural language answer
    answer = synthesize_natural_answer(question, sanitized_sql, rows, client=client)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return TextToSqlResult(
        question=question,
        generated_sql=raw_sql,
        sanitized_sql=sanitized_sql,
        row_count=len(rows),
        query_results=rows,
        answer=answer,
        execution_time_ms=round(elapsed_ms, 2),
    )


def parse_args() -> argparse.Namespace:
    """CLI argument parser for Text-to-SQL tool."""
    parser = argparse.ArgumentParser(
        description="Secure Text-to-SQL runner for live database queries."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=True,
        help="Natural language question to query against the database",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database file (default: c-database/ecommerce.db)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for executing Text-to-SQL queries."""
    args = parse_args()
    if not args.db_path.exists():
        print(f"📦 Initializing database at {args.db_path}...")
        init_database(args.db_path)

    with httpx.Client(timeout=30.0) as client:
        print(f"❓ Question: {args.question}")
        result = run_text_to_sql_pipeline(
            args.question, client=client, db_path=args.db_path
        )
        print(f"🔍 Generated SQL: {result.generated_sql}")
        print(f"🛡️ Sanitized SQL: {result.sanitized_sql}")
        print(f"📊 Rows Returned: {result.row_count}")
        print(f"💬 Answer: {result.answer}")
        print(f"⏱️ Execution Time: {result.execution_time_ms} ms")


if __name__ == "__main__":
    main()
