"""Tests for Task 4.11: Model Context Protocol (MCP) Server for Knowledge Base.

Verifies:
1. Tool listing and registration on the MCPServer (3 read-only tools).
2. Resource exposure (ecommerce://schema DDL and table metadata).
3. Tool execution: query_products with category/price/stock filtering.
4. Tool execution: get_customer_orders with customer ID and status filter.
5. Tool execution: execute_read_only_sql with safety checks and limit enforcement.
6. Security guards: forbidding DROP, INSERT, UPDATE, DELETE, and sensitive tables.
7. LangChain tool adapter: converting MCP tools to LangChain BaseTool instances.
8. End-to-end stdio transport communication via Stdio client session.
"""

from __future__ import annotations

import sqlite3
import sys
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from phase_4_agents.mcp_client import (
    convert_mcp_to_langchain_tools,
    open_mcp_stdio_client,
)
from phase_4_agents.mcp_server import (
    create_ecommerce_mcp_server,
    resolve_db_path,
)


def _get_text(res: Any) -> str:
    """Extract string text safely from a tool call result."""
    if isinstance(res, CallToolResult) and res.content:
        first = res.content[0]
        if isinstance(first, TextContent):
            return first.text
    return str(getattr(res, "text", "") or "")


@pytest.fixture
def test_db(tmp_path: Any) -> str:
    """Create a temporary SQLite database with ecommerce tables for testing."""
    db_file = tmp_path / "test_ecommerce.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            city TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_quantity INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            total_amount REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE sensitive_admin_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_user TEXT NOT NULL,
            action TEXT NOT NULL
        );

        INSERT INTO customers (name, email, city) VALUES
            ('Alice Smith', 'alice@example.com', 'San Francisco'),
            ('Bob Jones', 'bob@example.com', 'New York');

        INSERT INTO products (name, category, price, stock_quantity) VALUES
            ('MacBook Pro 16', 'Laptops', 2499.00, 10),
            ('Keychron K2 Keyboard', 'Accessories', 89.00, 50),
            ('Sony WH-1000XM5 Headphones', 'Audio', 399.00, 0);

        INSERT INTO orders (customer_id, order_date, status, total_amount) VALUES
            (1, '2026-03-01', 'completed', 2499.00),
            (1, '2026-03-05', 'pending', 89.00);

        INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
            (1, 1, 1, 2499.00),
            (2, 2, 1, 89.00);

        INSERT INTO sensitive_admin_logs (admin_user, action) VALUES
            ('superadmin', 'purged_database');
        """
    )
    conn.commit()
    conn.close()
    return str(db_file)


@pytest.fixture
def mcp_server(test_db: str) -> MCPServer:
    """Instantiate an MCPServer backed by the temporary SQLite database."""
    return create_ecommerce_mcp_server(db_path=test_db)


# =============================================================================
# 1. MCP Tool Registration & Metadata Tests
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_server_lists_all_tools(mcp_server: MCPServer) -> None:
    """Verify MCPServer exposes the 3 read-only tools and 1 write action."""
    tools = await mcp_server.list_tools()
    tool_names = [t.name for t in tools]

    assert "query_products" in tool_names
    assert "get_customer_orders" in tool_names
    assert "execute_read_only_sql" in tool_names
    assert "add_to_cart" in tool_names
    assert len(tool_names) == 4

    read_tools = [t for t in tools if "READ-ONLY" in (t.description or "")]
    assert len(read_tools) == 3

    write_tool = next(t for t in tools if t.name == "add_to_cart")
    assert "WRITE ACTION" in (write_tool.description or "")
    assert "HUMAN APPROVAL" in (write_tool.description or "")


# =============================================================================
# 2. MCP Resource Tests
# =============================================================================


@pytest.mark.asyncio
async def test_mcp_server_exposes_database_schema_resource(
    mcp_server: MCPServer,
) -> None:
    """Verify MCPServer exposes the ecommerce://schema resource."""
    resources = await mcp_server.list_resources()
    resource_uris = [r.uri for r in resources]
    assert "ecommerce://schema" in resource_uris

    contents = list(await mcp_server.read_resource("ecommerce://schema"))
    assert len(contents) == 1
    content_text = getattr(contents[0], "content", "")
    assert "Table: `products`" in content_text
    assert "Table: `customers`" in content_text
    assert "Table: `orders`" in content_text
    assert "sensitive_admin_logs" not in content_text


# =============================================================================
# 3. Tool 1 Tests: query_products
# =============================================================================


@pytest.mark.asyncio
async def test_query_products_all(mcp_server: MCPServer) -> None:
    """Verify query_products returns all products when no filters are set."""
    res = await mcp_server.call_tool("query_products", {})
    text = _get_text(res)
    assert "Found 3 product(s):" in text
    assert "MacBook Pro 16" in text
    assert "Keychron K2 Keyboard" in text
    assert "Sony WH-1000XM5 Headphones" in text


@pytest.mark.asyncio
async def test_query_products_category_filter(mcp_server: MCPServer) -> None:
    """Verify query_products filters by category case-insensitively."""
    res = await mcp_server.call_tool("query_products", {"category": "laptops"})
    text = _get_text(res)
    assert "Found 1 product(s):" in text
    assert "MacBook Pro 16" in text
    assert "Keychron K2 Keyboard" not in text


@pytest.mark.asyncio
async def test_query_products_price_filter(mcp_server: MCPServer) -> None:
    """Verify query_products respects min_price and max_price."""
    res = await mcp_server.call_tool(
        "query_products", {"min_price": 50.0, "max_price": 500.0}
    )
    text = _get_text(res)
    assert "Keychron K2 Keyboard" in text
    assert "Sony WH-1000XM5 Headphones" in text
    assert "MacBook Pro 16" not in text


@pytest.mark.asyncio
async def test_query_products_in_stock_only(mcp_server: MCPServer) -> None:
    """Verify query_products in_stock_only excludes zero-inventory items."""
    res = await mcp_server.call_tool("query_products", {"in_stock_only": True})
    text = _get_text(res)
    assert "Found 2 product(s):" in text
    assert "Sony WH-1000XM5 Headphones" not in text


@pytest.mark.asyncio
async def test_query_products_no_matches_actionable_hint(
    mcp_server: MCPServer,
) -> None:
    """Verify query_products returns available categories when no match found."""
    res = await mcp_server.call_tool("query_products", {"category": "Unicorns"})
    text = _get_text(res)
    assert "No products matched the given criteria" in text
    assert "Available categories in catalog" in text


# =============================================================================
# 4. Tool 2 Tests: get_customer_orders
# =============================================================================


@pytest.mark.asyncio
async def test_get_customer_orders_success(mcp_server: MCPServer) -> None:
    """Verify get_customer_orders retrieves orders and line items."""
    res = await mcp_server.call_tool("get_customer_orders", {"customer_id": 1})
    text = _get_text(res)
    assert "Customer Orders for Alice Smith" in text
    assert "Order #1 [COMPLETED]" in text
    assert "Order #2 [PENDING]" in text
    assert "1x MacBook Pro 16" in text


@pytest.mark.asyncio
async def test_get_customer_orders_status_filter(mcp_server: MCPServer) -> None:
    """Verify get_customer_orders filters orders by status."""
    res = await mcp_server.call_tool(
        "get_customer_orders", {"customer_id": 1, "status": "completed"}
    )
    text = _get_text(res)
    assert "Order #1 [COMPLETED]" in text
    assert "Order #2" not in text


@pytest.mark.asyncio
async def test_get_customer_orders_missing_customer(mcp_server: MCPServer) -> None:
    """Verify get_customer_orders returns clear message for unknown customer."""
    res = await mcp_server.call_tool("get_customer_orders", {"customer_id": 999})
    text = _get_text(res)
    assert "Customer ID 999 not found" in text


# =============================================================================
# 5. Tool 3 Tests: execute_read_only_sql & Security Guards
# =============================================================================


@pytest.mark.asyncio
async def test_execute_read_only_sql_valid_query(mcp_server: MCPServer) -> None:
    """Verify execute_read_only_sql executes valid SELECT queries with columns."""
    res = await mcp_server.call_tool(
        "execute_read_only_sql",
        {"sql_query": "SELECT name, price FROM products WHERE price > 100"},
    )
    text = _get_text(res)
    assert "SQL Query Results" in text
    assert "MacBook Pro 16" in text
    assert "Sony WH-1000XM5 Headphones" in text


@pytest.mark.asyncio
async def test_execute_read_only_sql_appends_limit(mcp_server: MCPServer) -> None:
    """Verify queries without a LIMIT clause automatically have LIMIT enforced."""
    res = await mcp_server.call_tool(
        "execute_read_only_sql",
        {"sql_query": "SELECT * FROM customers", "limit": 1},
    )
    text = _get_text(res)
    assert "SQL Query Results (1 rows):" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forbidden_query",
    [
        "DROP TABLE customers;",
        "DELETE FROM orders WHERE order_id = 1;",
        "INSERT INTO products (name, category, price) VALUES ('X', 'Y', 10);",
        "UPDATE products SET price = 0 WHERE product_id = 1;",
        "ALTER TABLE customers ADD COLUMN balance REAL;",
        "SELECT 1; DROP TABLE products;",
    ],
)
async def test_execute_read_only_sql_rejects_mutations(
    mcp_server: MCPServer, forbidden_query: str
) -> None:
    """Verify execute_read_only_sql rejects SQL mutation operations."""
    res = await mcp_server.call_tool(
        "execute_read_only_sql", {"sql_query": forbidden_query}
    )
    text = _get_text(res)
    assert "Security Error" in text


@pytest.mark.asyncio
async def test_execute_read_only_sql_rejects_sensitive_tables(
    mcp_server: MCPServer,
) -> None:
    """Verify execute_read_only_sql blocks access to sensitive_admin_logs."""
    res = await mcp_server.call_tool(
        "execute_read_only_sql",
        {"sql_query": "SELECT * FROM sensitive_admin_logs;"},
    )
    text = _get_text(res)
    assert "Security Error: Access to sensitive table 'sensitive_admin_logs'" in text


# =============================================================================
# 6. LangChain Tool Adapter Tests
# =============================================================================


def test_convert_mcp_to_langchain_tools(mcp_server: MCPServer) -> None:
    """Verify convert_mcp_to_langchain_tools returns valid LangChain tools."""
    lc_tools = convert_mcp_to_langchain_tools(mcp_server)
    assert len(lc_tools) == 4

    tool_map = {t.name: t for t in lc_tools}
    assert "query_products" in tool_map
    assert "get_customer_orders" in tool_map
    assert "execute_read_only_sql" in tool_map
    assert "add_to_cart" in tool_map

    # Test sync invocation of query_products via LangChain
    prod_res = tool_map["query_products"].invoke({"category": "Accessories"})
    assert "Keychron K2 Keyboard" in prod_res

    # Test sync invocation of execute_read_only_sql via LangChain
    sql_res = tool_map["execute_read_only_sql"].invoke(
        {"sql_query": "SELECT name FROM customers"}
    )
    assert "Alice Smith" in sql_res


@pytest.mark.asyncio
async def test_add_to_cart_tool_mutates_database(
    mcp_server: MCPServer, test_db: str
) -> None:
    """Verify add_to_cart tool checks stock and inserts into cart_items."""
    # 1. Successful addition
    res = await mcp_server.call_tool(
        "add_to_cart",
        {"product_name": "Keychron K2 Keyboard", "quantity": 2, "customer_id": 1},
    )
    text = _get_text(res)
    assert "Successfully added" in text
    assert "Keychron K2 Keyboard" in text

    # Verify directly in the database
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()
    cur.execute("SELECT quantity, unit_price FROM cart_items WHERE customer_id = 1")
    row = cur.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 2
    assert row[1] == 89.00

    # 2. Out of stock rejection
    res_oos = await mcp_server.call_tool(
        "add_to_cart",
        {"product_name": "Sony WH-1000XM5 Headphones", "quantity": 1, "customer_id": 1},
    )
    text_oos = _get_text(res_oos)
    assert "Inventory Warning" in text_oos

    # 3. Product not found
    res_missing = await mcp_server.call_tool(
        "add_to_cart",
        {"product_name": "Nonexistent Gizmo", "quantity": 1, "customer_id": 1},
    )
    text_missing = _get_text(res_missing)
    assert "not found" in text_missing


# =============================================================================
# 7. End-to-End Stdio Transport Client Integration
# =============================================================================


@pytest.mark.asyncio
async def test_open_mcp_stdio_client_integration() -> None:
    """Verify end-to-end communication with the MCP server over stdio transport."""
    db_path = str(resolve_db_path())

    async with open_mcp_stdio_client(
        command=sys.executable,
        args=["-m", "phase_4_agents.mcp_server", "--db-path", db_path],
    ) as session:
        # 1. List tools
        tools_result = await session.list_tools()
        names = [t.name for t in tools_result.tools]
        assert "query_products" in names
        assert "get_customer_orders" in names
        assert "execute_read_only_sql" in names
        assert "add_to_cart" in names

        # 2. List resources
        resources_result = await session.list_resources()
        uris = [r.uri for r in resources_result.resources]
        assert "ecommerce://schema" in uris

        # 3. Call tool
        tool_call_result = await session.call_tool("query_products", {"limit": 2})
        text = _get_text(tool_call_result)
        assert "Found" in text or "product" in text
