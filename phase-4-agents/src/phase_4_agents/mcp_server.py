"""Model Context Protocol (MCP) Server for Phase 3 Ecommerce Knowledge Base.

Exposes 3 read-only tools and a schema resource over the SQLite ecommerce
database:
1. `query_products`: Filter catalog products by category, price, and stock.
2. `get_customer_orders`: Lookup order history and line items for a customer.
3. `execute_read_only_sql`: Safely execute read-only SELECT queries with security
   checks and limit enforcement.
Resource:
- `ecommerce://schema`: DDL and table structure for all ecommerce tables.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

# Default path to Phase 3 ecommerce SQLite database
DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[3] / "phase-3-rag" / "c-database" / "ecommerce.db"
)

# Forbidden keywords in read-only SQL queries
FORBIDDEN_SQL_KEYWORDS = {
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "ALTER",
    "ATTACH",
    "DETACH",
    "CREATE",
    "REPLACE",
    "TRUNCATE",
    "PRAGMA",
    "GRANT",
    "REVOKE",
}

# Forbidden sensitive tables
FORBIDDEN_TABLES = {"sensitive_admin_logs", "sqlite_master", "sqlite_sequence"}


def resolve_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve the path to the ecommerce database with environment fallbacks."""
    if db_path is not None:
        path = Path(db_path).resolve()
        if path.exists():
            return path

    env_path = os.getenv("ECOMMERCE_DB_PATH")
    if env_path:
        path = Path(env_path).resolve()
        if path.exists():
            return path

    if DEFAULT_DB_PATH.exists():
        return DEFAULT_DB_PATH

    # Fallback to local workspace database if present
    workspace_db = Path.cwd() / "ecommerce.db"
    if workspace_db.exists():
        return workspace_db

    # Return default path even if not yet created (caller can initialize)
    return DEFAULT_DB_PATH


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Create a sqlite3 connection with Row factory enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def create_ecommerce_mcp_server(
    db_path: Path | str | None = None,
    server_name: str = "ecommerce-knowledge-base",
) -> MCPServer:
    """Construct and configure the Model Context Protocol (MCP) server.

    Args:
        db_path: Optional path to the SQLite database file.
        server_name: MCP server name identifier.

    Returns:
        Configured MCPServer instance exposing 3 read-only tools and 1 resource.
    """
    resolved_path = resolve_db_path(db_path)
    server = MCPServer(
        name=server_name,
        instructions=(
            "Model Context Protocol (MCP) server providing read-only tools "
            "and catalog resources for the Phase 3 ecommerce knowledge base."
        ),
    )

    # -------------------------------------------------------------------------
    # Resource: Database Schema
    # -------------------------------------------------------------------------
    @server.resource("ecommerce://schema")
    def get_database_schema() -> str:
        """Expose the DDL schema and table definitions for all ecommerce tables."""
        if not resolved_path.exists():
            return "Ecommerce database not found."

        conn = get_db_connection(resolved_path)
        try:
            cur = conn.cursor()
            tables = cur.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT IN "
                "('sensitive_admin_logs', 'sqlite_sequence');"
            ).fetchall()

            lines = ["# Ecommerce Knowledge Base Database Schema\n"]
            for row in tables:
                table_name = row["name"]
                sql = row["sql"] or ""
                count = cur.execute(
                    f"SELECT COUNT(*) FROM {table_name}"  # noqa: S608
                ).fetchone()[0]
                lines.append(f"### Table: `{table_name}` ({count} records)")
                lines.append(f"```sql\n{sql}\n```\n")

            return "\n".join(lines)
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Tool 1: Query Products Catalog
    # -------------------------------------------------------------------------
    @server.tool(
        name="query_products",
        description=(
            "Query products in the ecommerce catalog (READ-ONLY). Supports filtering "
            "by category name, minimum price, maximum price, and stock availability."
        ),
    )
    def query_products(
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        in_stock_only: bool = False,
        limit: int = 10,
    ) -> str:
        """Filter products in the catalog based on optional search criteria.

        Args:
            category: Optional category filter (e.g. 'Electronics', 'Hardware').
            min_price: Minimum price threshold in USD.
            max_price: Maximum price threshold in USD.
            in_stock_only: If True, only returns items where stock_quantity > 0.
            limit: Maximum number of products to return (default: 10, max: 50).
        """
        if not resolved_path.exists():
            return f"Database error: Database file not found at {resolved_path}."

        capped_limit = max(1, min(limit, 50))
        clauses: list[str] = []
        params: list[Any] = []

        if category:
            clauses.append("LOWER(category) = LOWER(?)")
            params.append(category.strip())
        if min_price is not None:
            clauses.append("price >= ?")
            params.append(float(min_price))
        if max_price is not None:
            clauses.append("price <= ?")
            params.append(float(max_price))
        if in_stock_only:
            clauses.append("stock_quantity > 0")

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT product_id, name, category, price, stock_quantity "
            f"FROM products {where_clause} "
            f"ORDER BY product_id ASC LIMIT ?"
        )
        params.append(capped_limit)

        conn = get_db_connection(resolved_path)
        try:
            cur = conn.cursor()
            rows = cur.execute(query, params).fetchall()

            if not rows:
                return (
                    "No products matched the given criteria. Available categories "
                    "in catalog: 'Laptops', 'Accessories', 'Monitors', 'Audio'."
                )

            lines = [f"Found {len(rows)} product(s):"]
            for r in rows:
                lines.append(
                    f"• [ID: {r['product_id']}] {r['name']} | "
                    f"Category: {r['category']} | Price: ${r['price']:.2f} | "
                    f"Stock: {r['stock_quantity']}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Error querying products catalog: {exc}"
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Tool 2: Get Customer Orders
    # -------------------------------------------------------------------------
    @server.tool(
        name="get_customer_orders",
        description=(
            "Retrieve order history and purchased line items for a specific "
            "customer ID (READ-ONLY). Optional status filter "
            "('pending', 'completed', 'shipped')."
        ),
    )
    def get_customer_orders(
        customer_id: int,
        status: str | None = None,
        limit: int = 5,
    ) -> str:
        """Retrieve customer orders and associated items.

        Args:
            customer_id: Numeric identifier of the customer (e.g. 1, 2).
            status: Optional order status filter (e.g. 'completed', 'pending').
            limit: Maximum number of orders to return (default: 5, max: 20).
        """
        if not resolved_path.exists():
            return f"Database error: Database file not found at {resolved_path}."

        capped_limit = max(1, min(limit, 20))
        conn = get_db_connection(resolved_path)
        try:
            cur = conn.cursor()

            # Check customer existence
            cust = cur.execute(
                "SELECT customer_id, name, email, city FROM customers "
                "WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()

            if not cust:
                return f"Customer ID {customer_id} not found in customer registry."

            # Fetch orders
            order_query = (
                "SELECT order_id, order_date, status, total_amount "
                "FROM orders WHERE customer_id = ?"
            )
            order_params: list[Any] = [customer_id]

            if status:
                order_query += " AND LOWER(status) = LOWER(?)"
                order_params.append(status.strip())

            order_query += " ORDER BY order_date DESC LIMIT ?"
            order_params.append(capped_limit)

            orders = cur.execute(order_query, order_params).fetchall()
            if not orders:
                return (
                    f"Customer {cust['name']} (ID: {customer_id}) has no orders "
                    f"matching status='{status or 'all'}'."
                )

            lines = [
                f"Customer Orders for {cust['name']} ({cust['email']}, {cust['city']}):"
            ]

            for o in orders:
                lines.append(
                    f"\n• Order #{o['order_id']} [{o['status'].upper()}] - "
                    f"Date: {o['order_date']} | Total: ${o['total_amount']:.2f}"
                )
                # Fetch line items
                items = cur.execute(
                    "SELECT oi.quantity, oi.unit_price, p.name "
                    "FROM order_items oi "
                    "JOIN products p ON oi.product_id = p.product_id "
                    "WHERE oi.order_id = ?",
                    (o["order_id"],),
                ).fetchall()
                for it in items:
                    qty = it["quantity"]
                    pname = it["name"]
                    uprice = it["unit_price"]
                    lines.append(f"    - {qty}x {pname} @ ${uprice:.2f}")

            return "\n".join(lines)
        except Exception as exc:
            return f"Error retrieving customer orders: {exc}"
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Tool 3: Execute Read-Only SQL Query
    # -------------------------------------------------------------------------
    @server.tool(
        name="execute_read_only_sql",
        description=(
            "Execute arbitrary read-only SELECT SQL queries against the ecommerce "
            "database (READ-ONLY). Mutation statements (INSERT, UPDATE, DELETE, DROP) "
            "and sensitive tables are strictly forbidden."
        ),
    )
    def execute_read_only_sql(
        sql_query: str,
        limit: int = 20,
    ) -> str:
        """Safely execute a read-only SELECT query against the ecommerce database.

        Args:
            sql_query: SQL query starting with SELECT or WITH.
            limit: Maximum rows to return (default: 20, max: 50).
        """
        if not resolved_path.exists():
            return f"Database error: Database file not found at {resolved_path}."

        clean_query = sql_query.strip().rstrip(";")
        capped_limit = max(1, min(limit, 50))

        # 1. Reject empty queries
        if not clean_query:
            return "SQL Error: Query string cannot be empty."

        # 2. Enforce SELECT / WITH prefix
        normalized = clean_query.strip().upper()
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            return (
                "Security Error: Only read-only SELECT queries are allowed. "
                "Queries modifying data are strictly forbidden."
            )

        # 3. Disallow multiple statements (semicolon chaining)
        if ";" in clean_query:
            return "Security Error: Multiple SQL statements are not permitted."

        # 4. Keyword safety check
        tokens = set(re.findall(r"\b[A-Za-z_]+\b", normalized))
        for forbidden in FORBIDDEN_SQL_KEYWORDS:
            if forbidden in tokens:
                return (
                    f"Security Error: Keyword '{forbidden}' is forbidden "
                    "in read-only mode."
                )

        # 5. Sensitive table access check
        for table in FORBIDDEN_TABLES:
            if table.upper() in tokens:
                return (
                    f"Security Error: Access to sensitive table '{table}' is forbidden."
                )

        # 6. Enforce LIMIT clause
        if not re.search(r"\bLIMIT\b", normalized):
            clean_query += f" LIMIT {capped_limit}"

        conn = get_db_connection(resolved_path)
        try:
            cur = conn.cursor()
            cur.execute(clean_query)
            rows = cur.fetchall()

            if not rows:
                return "Query executed successfully. Rows returned: 0."

            col_names = [d[0] for d in cur.description]
            lines = [f"SQL Query Results ({len(rows)} rows):"]
            lines.append(" | ".join(col_names))
            lines.append("-" * min(80, len(" | ".join(col_names)) + 5))

            for row in rows:
                row_vals = [
                    str(row[c]) if row[c] is not None else "NULL" for c in col_names
                ]
                lines.append(" | ".join(row_vals))

            return "\n".join(lines)
        except Exception as exc:
            return (
                f"SQL Execution Error: {exc}. Valid tables: "
                "'customers', 'products', 'orders', 'order_items', 'appointments'."
            )
        finally:
            conn.close()

    return server


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the MCP server."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4 Model Context Protocol (MCP) Server for Ecommerce Knowledge Base"
        )
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the SQLite ecommerce database file",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    return parser.parse_args()


async def run_server(args: argparse.Namespace) -> None:
    """Run the MCP server over standard input/output."""
    server = create_ecommerce_mcp_server(db_path=args.db_path)
    if args.transport == "stdio":
        await server.run_stdio_async()


def main() -> None:
    """Entrypoint function for CLI and pyproject.toml script execution."""
    args = parse_args()
    try:
        asyncio.run(run_server(args))
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)


if __name__ == "__main__":
    main()
