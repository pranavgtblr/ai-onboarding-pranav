"""Structured tool calls for database access in Project C (Task 3.15).

Exposes strictly typed, parameterized functions for customer and appointment queries.
The LLM only fills in arguments—it never writes SQL. All database operations use
read-only connections and parameterized SQL bindings, making SQL injection impossible
and enforcing strict tenant access controls.
"""

import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from phase_3_rag.config import get_settings
from phase_3_rag.database import DEFAULT_DB_PATH, get_readonly_connection

logger = logging.getLogger("database_tools")


# ---------------------------------------------------------------------------
# Typed Database Functions (Strictly Parameterized, Read-Only)
# ---------------------------------------------------------------------------


def get_customer(
    *,
    customer_id: int | None = None,
    name: str | None = None,
    email: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Look up customer records by ID, name substring, or email.

    Args:
        customer_id: Optional exact customer ID.
        name: Optional substring to match against customer name.
        email: Optional exact or substring match for customer email.
        db_path: Path to SQLite database.

    Returns:
        Dictionary containing matched customer records.
    """
    conn = get_readonly_connection(db_path)
    cursor = conn.cursor()

    conditions: list[str] = []
    params: list[Any] = []

    if customer_id is not None:
        conditions.append("customer_id = ?")
        params.append(customer_id)
    if name is not None and name.strip():
        conditions.append("name LIKE ?")
        params.append(f"%{name.strip()}%")
    if email is not None and email.strip():
        conditions.append("email LIKE ?")
        params.append(f"%{email.strip()}%")

    if not conditions:
        conn.close()
        return {
            "error": (
                "At least one search filter (customer_id, name, or email) required."
            )
        }

    where_clause = " AND ".join(conditions)
    query = (
        f"SELECT customer_id, name, email, phone, city, created_at "
        f"FROM customers WHERE {where_clause} LIMIT 10"
    )

    try:
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        return {"count": len(rows), "customers": rows}
    finally:
        conn.close()


def get_appointments(
    *,
    customer_id: int,
    from_date: str | None = None,
    to_date: str | None = None,
    status: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Retrieve appointments for a specific customer within an optional date range.

    Args:
        customer_id: The ID of the customer (required for strict tenant scoping).
        from_date: Optional ISO date/timestamp string (e.g. '2026-09-01').
        to_date: Optional ISO date/timestamp string (e.g. '2026-09-30').
        status: Optional status filter ('scheduled', 'confirmed',
            'completed', 'cancelled').
        db_path: Path to SQLite database.

    Returns:
        Dictionary containing matched appointments and customer metadata.
    """
    conn = get_readonly_connection(db_path)
    cursor = conn.cursor()

    conditions: list[str] = ["customer_id = ?"]
    params: list[Any] = [customer_id]

    if from_date is not None and from_date.strip():
        conditions.append("scheduled_time >= ?")
        params.append(from_date.strip())
    if to_date is not None and to_date.strip():
        conditions.append("scheduled_time <= ?")
        params.append(to_date.strip())
    if status is not None and status.strip():
        conditions.append("status = ?")
        params.append(status.strip().lower())

    where_clause = " AND ".join(conditions)
    query = (
        f"SELECT appointment_id, customer_id, service_type, scheduled_time, "
        f"status, notes FROM appointments WHERE {where_clause} "
        f"ORDER BY scheduled_time ASC LIMIT 50"
    )

    try:
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        return {
            "customer_id": customer_id,
            "count": len(rows),
            "appointments": rows,
        }
    finally:
        conn.close()


def get_orders(
    *,
    customer_id: int,
    status: str | None = None,
    limit: int = 10,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Retrieve recent orders for a customer with optional status filtering.

    Args:
        customer_id: The customer ID.
        status: Optional order status filter.
        limit: Maximum number of orders to return (clamped to 50).
        db_path: Path to SQLite database.

    Returns:
        Dictionary containing order summaries.
    """
    conn = get_readonly_connection(db_path)
    cursor = conn.cursor()

    safe_limit = max(1, min(limit, 50))
    conditions: list[str] = ["customer_id = ?"]
    params: list[Any] = [customer_id]

    if status is not None and status.strip():
        conditions.append("status = ?")
        params.append(status.strip().lower())

    where_clause = " AND ".join(conditions)
    query = (
        f"SELECT order_id, customer_id, order_date, status, total_amount, "
        f"shipping_address FROM orders WHERE {where_clause} "
        f"ORDER BY order_date DESC LIMIT {safe_limit}"
    )

    try:
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        return {
            "customer_id": customer_id,
            "count": len(rows),
            "orders": rows,
        }
    finally:
        conn.close()


def get_order_details(
    *,
    order_id: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Retrieve comprehensive details of an order including line items and products.

    Args:
        order_id: The primary key of the order.
        db_path: Path to SQLite database.

    Returns:
        Dictionary containing order header, line items, and product details.
    """
    conn = get_readonly_connection(db_path)
    cursor = conn.cursor()

    try:
        # Fetch order header
        cursor.execute(
            "SELECT o.order_id, o.customer_id, c.name AS customer_name, "
            "o.order_date, o.status, o.total_amount, o.shipping_address "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE o.order_id = ?",
            (order_id,),
        )
        order_row = cursor.fetchone()
        if not order_row:
            return {"error": f"Order #{order_id} not found."}

        order_dict = dict(order_row)

        # Fetch line items
        cursor.execute(
            "SELECT oi.item_id, oi.product_id, p.name AS product_name, "
            "oi.quantity, oi.unit_price, (oi.quantity * oi.unit_price) AS line_total "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
            "WHERE oi.order_id = ?",
            (order_id,),
        )
        items = [dict(row) for row in cursor.fetchall()]
        order_dict["items"] = items
        return order_dict
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Gemini Tool Declarations (REST API Format)
# ---------------------------------------------------------------------------


def get_database_tool_declarations() -> list[dict[str, Any]]:
    """Return JSON Schema tool declarations formatted for the Gemini REST API."""
    return [
        {
            "function_declarations": [
                {
                    "name": "get_customer",
                    "description": (
                        "Look up a customer record by their name, email, or ID. "
                        "Always call this first if you only have a customer's name "
                        "and need their customer_id."
                    ),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "customer_id": {
                                "type": "INTEGER",
                                "description": "The exact numeric customer ID.",
                            },
                            "name": {
                                "type": "STRING",
                                "description": (
                                    "Customer name or partial name (e.g. 'Alice')."
                                ),
                            },
                            "email": {
                                "type": "STRING",
                                "description": "Customer email address.",
                            },
                        },
                    },
                },
                {
                    "name": "get_appointments",
                    "description": (
                        "Retrieve upcoming or past appointments for a specific "
                        "customer. Requires customer_id. Optionally filter by "
                        "date range or status."
                    ),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "customer_id": {
                                "type": "INTEGER",
                                "description": "Numeric ID of the customer.",
                            },
                            "from_date": {
                                "type": "STRING",
                                "description": (
                                    "Filter appointments on or after this ISO "
                                    "date/time."
                                ),
                            },
                            "to_date": {
                                "type": "STRING",
                                "description": (
                                    "Filter appointments on or before this ISO "
                                    "date/time."
                                ),
                            },
                            "status": {
                                "type": "STRING",
                                "enum": [
                                    "scheduled",
                                    "confirmed",
                                    "completed",
                                    "cancelled",
                                ],
                                "description": "Status filter for appointments.",
                            },
                        },
                        "required": ["customer_id"],
                    },
                },
                {
                    "name": "get_orders",
                    "description": (
                        "Retrieve a list of orders placed by a specific customer. "
                        "Requires customer_id. Returns recent orders sorted newest "
                        "first."
                    ),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "customer_id": {
                                "type": "INTEGER",
                                "description": "Numeric ID of the customer.",
                            },
                            "status": {
                                "type": "STRING",
                                "enum": [
                                    "pending",
                                    "processing",
                                    "shipped",
                                    "delivered",
                                    "cancelled",
                                ],
                                "description": "Optional order status filter.",
                            },
                            "limit": {
                                "type": "INTEGER",
                                "description": "Max orders to retrieve (default 10).",
                            },
                        },
                        "required": ["customer_id"],
                    },
                },
                {
                    "name": "get_order_details",
                    "description": (
                        "Retrieve detailed line items and shipping info for a "
                        "specific order."
                    ),
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "order_id": {
                                "type": "INTEGER",
                                "description": "The exact numeric order ID.",
                            },
                        },
                        "required": ["order_id"],
                    },
                },
            ]
        }
    ]


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------


TOOL_REGISTRY = {
    "get_customer": get_customer,
    "get_appointments": get_appointments,
    "get_orders": get_orders,
    "get_order_details": get_order_details,
}


def execute_database_tool(
    tool_name: str,
    args: dict[str, Any],
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Execute a registered database tool safely with typed arguments."""
    handler = TOOL_REGISTRY.get(tool_name)
    if handler is None:
        return {"error": f"Tool '{tool_name}' is not recognized."}

    # Pass db_path to the handler along with model arguments
    call_args = dict(args)
    call_args["db_path"] = db_path

    try:
        return handler(**call_args)
    except TypeError as e:
        return {"error": f"Invalid argument types for tool '{tool_name}': {e}"}
    except Exception as e:
        logger.exception("Unexpected error executing tool %s", tool_name)
        return {"error": f"Tool execution failed: {e}"}


# ---------------------------------------------------------------------------
# Tool Calling Loop & Result Container
# ---------------------------------------------------------------------------


class ToolCallLog(BaseModel):
    """Structured record of a tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    output: dict[str, Any]


class DatabaseToolResult(BaseModel):
    """Complete result from running structured database tool calling."""

    question: str
    final_answer: str
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    iterations: int = 0
    hit_max_guard: bool = False


def run_database_tool_loop(
    prompt: str,
    *,
    client: httpx.Client,
    model: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    max_iterations: int = 5,
) -> DatabaseToolResult:
    """Execute a multi-turn tool calling loop for customer database inquiries.

    The model only fills in arguments for typed functions. All queries execute
    via parameterized read-only database connections.
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required to run database tool calling.")

    active_model = model or settings.gemini_model
    endpoint_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{active_model}:generateContent?key={settings.gemini_api_key}"
    )

    tools_payload = get_database_tool_declarations()
    system_instruction = (
        "You are an assistant answering customer support inquiries using live data. "
        "Use the provided tools to inspect customers, appointments, and orders. "
        "If a user asks about a customer by name, use 'get_customer' to look up their "
        "customer_id first, then query their appointments or orders with that ID. "
        "Never make up customer IDs or order statuses. Answer clearly and concisely."
    )

    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": prompt}]}]

    all_tool_logs: list[ToolCallLog] = []
    final_text = ""
    hit_guard = False

    for _ in range(1, max_iterations + 1):
        payload: dict[str, Any] = {
            "contents": contents,
            "tools": tools_payload,
            "system_instruction": {"parts": [{"text": system_instruction}]},
        }

        response = client.post(
            endpoint_url, json=payload, timeout=settings.timeout_seconds
        )
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            final_text = "No response generated by model."
            break

        candidate = candidates[0]
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])

        # Check for function calls
        function_calls: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for part in parts:
            if "functionCall" in part:
                function_calls.append(part["functionCall"])
            if "text" in part:
                text_parts.append(part["text"])

        current_text = "".join(text_parts).strip()

        # If model did not request any tools, it's done
        if not function_calls:
            final_text = current_text
            break

        # Append model's thought / tool request to conversation
        contents.append(content_obj)

        # Execute each function call
        function_response_parts: list[dict[str, Any]] = []
        for fc in function_calls:
            fn_name = fc.get("name", "")
            fn_args = fc.get("args", {})

            result = execute_database_tool(fn_name, fn_args, db_path=db_path)
            all_tool_logs.append(
                ToolCallLog(
                    tool_name=fn_name,
                    arguments=fn_args,
                    output=result,
                )
            )

            function_response_parts.append(
                {
                    "functionResponse": {
                        "name": fn_name,
                        "response": result,
                    }
                }
            )

        # Append function responses as user turn
        contents.append({"role": "user", "parts": function_response_parts})
    else:
        hit_guard = True
        final_text = "Reached maximum tool execution iterations without a final answer."

    return DatabaseToolResult(
        question=prompt,
        final_answer=final_text,
        tool_calls=all_tool_logs,
        iterations=len(all_tool_logs),
        hit_max_guard=hit_guard,
    )
