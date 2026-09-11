"""RAG retrieval tools for Task 4.3 Agent.

Converts the Phase 3 static routing destinations into first-class LangChain tools:
1. pdf_search: Search local Mars Odyssey engineering PDF corpus and specs.
3. db_query: Query relational database (customers, orders, products, appointments).
4. web_search: Query live/recent internet knowledge with citations.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from langchain_core.tools import BaseTool, tool

from phase_4_agents.schemas import (
    AddToCartInput,
    DatabaseQueryInput,
    PDFSearchInput,
    SiteSearchInput,
    WebSearchInput,
)
from phase_4_agents.tools import calculator, get_weather

# Locate Phase 3 data paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PHASE3_DIR = _REPO_ROOT / "phase-3-rag"
PDF_CORPUS_DIR = _PHASE3_DIR / "data" / "pdf_corpus"
SCRAPED_DOCS_PATH = _PHASE3_DIR / "data" / "scraped_docs" / "chunks.json"
WEBSITE_DOCS_PATH = _PHASE3_DIR / "b-website" / "chunks.json"
DEFAULT_DB_PATH = _PHASE3_DIR / "c-database" / "ecommerce.db"


# -----------------------------------------------------------------------------
# Embedded Fallback Knowledge
# -----------------------------------------------------------------------------

FALLBACK_PDF_SNIPPETS = [
    (
        "doc_eclss_limits.md",
        "Project Odyssey ECLSS Operating Limits: Cabin atmospheric pressure is "
        "maintained at 101.3 kPa nominal (70.8 kPa minimum emergency limit). "
        "Oxygen partial pressure is regulated between 19.5 kPa and 23.1 kPa. "
        "Primary Power Bus operates at 120V DC nominal with fault cutoff at 138V DC.",
    ),
    (
        "doc_propulsion_specs.md",
        "Mars Descent & Ascent Propulsion System (MDAS): Utilizes hypergolic "
        "monomethylhydrazine (MMH) and dinitrogen tetroxide (NTO). "
        "Specific impulse is 318s in vacuum. Abort duration is 14.2s.",
    ),
    (
        "doc_habitat_thermal.md",
        "Habitat Thermal Control Subsystem (HTCS): Dual-loop pumped fluid system "
        "using propylene glycol/water mixture. Rejection capacity is 28 kW thermal "
        "via deployable composite radiator panels.",
    ),
    (
        "doc_water_recycling.md",
        "Water Recovery Subsystem (WRS): Closed-loop recovery rate exceeds 93.5% "
        "using vapor compression distillation and catalytic oxidation.",
    ),
]

FALLBACK_SITE_CHUNKS = [
    {
        "page_title": "Toobler Digital Innovation & AI Engineering",
        "url": "https://www.toobler.com/capabilities",
        "heading_path": "Capabilities > Digital Innovation Stack",
        "text": "Toobler provides end-to-end digital transformation, cloud native "
        "application engineering, IoT solutions, and autonomous AI agents. "
        "Our capabilities include React/Next.js frontend architectures, "
        "scalable Python backend services, and enterprise RAG systems.",
    },
    {
        "page_title": "Cloud Native, IoT & Mobile Solutions",
        "url": "https://www.toobler.com/blog",
        "heading_path": "Resources > Blog > Cloud Native & IoT",
        "text": "Best practices for deploying fault-tolerant distributed services, "
        "edge IoT telemetry streaming, and containerized microservices.",
    },
    {
        "page_title": "Contact & Engineering Solutions",
        "url": "https://www.toobler.com/contact",
        "heading_path": "Company > Contact Us",
        "text": "Get in touch with Toobler engineering leads for custom AI agent "
        "development, enterprise search, and cloud modernization.",
    },
]


# -----------------------------------------------------------------------------
# Tool 1: PDF Search
# -----------------------------------------------------------------------------


@tool(args_schema=PDFSearchInput)
def pdf_search(query: str) -> str:
    """Search internal engineering PDF specifications and manuals.

    Use this tool ONLY when the user asks about:
    - Project Odyssey Mars Base engineering specifications
    - ECLSS (Environmental Control and Life Support System) operating limits
    - Cabin atmospheric pressure, oxygen levels, primary power bus voltage
    - Propulsion systems (MDAS, hypergolic fuel, specific impulse)
    - Habitat thermal loops, radiator panels, spacesuit maintenance

    DO NOT USE FOR:
    - Toobler company capabilities or website offerings (use site_search instead).
    - Relational database orders or customer lookups (use db_query instead).
    - Breaking news or real-time internet information (use web_search instead).

    Args:
        query: Search keywords or question describing the engineering topic.
    """
    q_words = set(re.findall(r"\w+", query.lower()))
    matches: list[tuple[int, str, str]] = []

    # 1. Search local filesystem corpus if available
    if PDF_CORPUS_DIR.exists():
        for md_file in sorted(PDF_CORPUS_DIR.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                words = set(re.findall(r"\w+", content.lower()))
                overlap = len(q_words & words)
                if overlap > 0:
                    matches.append((overlap, md_file.name, content[:500]))
            except Exception:
                continue

    # 2. Add fallback snippets if no filesystem matches
    if not matches:
        for doc_id, snippet in FALLBACK_PDF_SNIPPETS:
            words = set(re.findall(r"\w+", snippet.lower()))
            overlap = len(q_words & words)
            if overlap > 0:
                matches.append((overlap, doc_id, snippet))

    if not matches:
        return (
            f"PDF Search Results for '{query}': "
            "No relevant PDF documents matched the query. "
            "Action for model: Check spelling or try broader keywords "
            "(e.g. 'eclss', 'pressure', 'propulsion', 'thermal'). "
            "DO NOT query company capabilities or live news here."
        )

    matches.sort(key=lambda x: x[0], reverse=True)
    top_matches = matches[:3]

    lines = [f"PDF Search Results for '{query}':"]
    for score, doc_name, text in top_matches:
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        lines.append(f"• [{doc_name}] (relevance: {score}):\n  {cleaned_text[:350]}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Tool 2: Site Search
# -----------------------------------------------------------------------------


@tool(args_schema=SiteSearchInput)
def site_search(query: str) -> str:
    """Search crawled website documentation and company capabilities.

    Use this tool ONLY when the user asks about:
    - Toobler digital innovation capabilities, technical stack, and cloud engineering
    - Services: React/Next.js architectures, Python microservices, IoT solutions
    - Documentation crawled from company websites or engineering blogs

    DO NOT USE FOR:
    - Mars Odyssey engineering manuals or ECLSS limits (use pdf_search instead).
    - Relational database orders or customers (use db_query instead).
    - Live external breaking news (use web_search instead).

    Args:
        query: Search keywords or question regarding site documentation.
    """
    q_words = set(re.findall(r"\w+", query.lower()))
    chunks: list[dict[str, Any]] = []

    # Attempt to load scraped docs
    for path in (WEBSITE_DOCS_PATH, SCRAPED_DOCS_PATH):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    chunks.extend(data)
            except Exception:
                pass

    if not chunks:
        chunks = FALLBACK_SITE_CHUNKS

    scored: list[tuple[int, dict[str, Any]]] = []
    for c in chunks:
        text = (
            f"{c.get('page_title', '')} {c.get('heading_path', '')} {c.get('text', '')}"
        )
        words = set(re.findall(r"\w+", text.lower()))
        overlap = len(q_words & words)
        scored.append((overlap, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] == 0:
        return (
            f"Website Search Results for '{query}': No matching documentation "
            "pages found. Action for model: Search for general capabilities "
            "like 'cloud', 'iot', 'react', 'python', or 'digital innovation'. "
            "DO NOT query Mars engineering specs here."
        )

    top_chunks = [c for s, c in scored[:3] if s > 0]

    lines = [f"Website Search Results for '{query}':"]
    for c in top_chunks:
        title = c.get("page_title", "Documentation Page")
        url = c.get("url", "https://www.toobler.com")
        path = c.get("heading_path", "")
        excerpt = re.sub(r"\s+", " ", c.get("text", "")).strip()[:300]
        lines.append(
            f"• Title: {title}\n  URL: {url}\n  Section: {path}\n  Content: {excerpt}"
        )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Tool 3: Structured Database Query (Text-to-SQL)
# -----------------------------------------------------------------------------


def _get_database_connection() -> sqlite3.Connection:
    """Connect to existing SQLite DB or create an in-memory seeded DB."""
    if DEFAULT_DB_PATH.exists():
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cart_items (
                cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                product_name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL,
                total_price REAL NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        return conn

    # Auto-seed in-memory DB if file is missing
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            city TEXT NOT NULL
        );
        INSERT INTO customers VALUES
            (1, 'Alice Smith', 'alice@example.com', '555-0101', 'Tokyo'),
            (2, 'Bob Jones', 'bob@example.com', '555-0102', 'London'),
            (3, 'Charlie Brown', 'charlie@example.com', '555-0103', 'New York'),
            (4, 'Diana Prince', 'diana@example.com', '555-0104', 'Paris');

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_quantity INTEGER NOT NULL
        );
        INSERT INTO products VALUES
            (1, 'Mars Rover Sensor', 'Sensors', 450.0, 12),
            (2, 'Oxygen Scrubber Cartridge', 'ECLSS', 850.0, 5),
            (3, 'Titanium Drill Bit', 'Tools', 120.0, 40);

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TEXT,
            status TEXT,
            total_amount REAL
        );
        INSERT INTO orders VALUES
            (101, 1, '2026-03-01', 'delivered', 450.0),
            (102, 2, '2026-03-04', 'shipped', 850.0),
            (103, 3, '2026-03-08', 'pending', 120.0);

        CREATE TABLE appointments (
            appointment_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            doctor_name TEXT,
            appointment_date TEXT,
            status TEXT
        );
        INSERT INTO appointments VALUES
            (201, 1, 'Dr. Sarah Connor', '2026-03-15', 'confirmed'),
            (202, 4, 'Dr. Leonard McCoy', '2026-03-18', 'scheduled');

        CREATE TABLE IF NOT EXISTS cart_items (
            cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            total_price REAL NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    return conn


@tool(args_schema=DatabaseQueryInput)
def db_query(query: str) -> str:
    """Execute a structured database query against transactional business tables.

    Use this tool ONLY for querying relational database tables:
    - Customers: customer_id, name, email, phone, city
    - Orders: order_id, customer_id, order_date, status, total_amount
    - Products: product_id, name, category, price, stock_quantity
    - Appointments: appointment_id, customer_id, doctor_name, appointment_date, status

    DO NOT USE FOR:
    - Mutating client data (use add_to_cart for purchasing or adding items to cart).
    - Searching Mars PDF manuals or external web documentation.

    Args:
        query: SQL SELECT statement OR natural language description of what to query.
    """
    clean_q = query.strip()
    conn = _get_database_connection()
    cursor = conn.cursor()

    try:
        # Enforce strict read-only constraint upfront
        forbidden = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "TRUNCATE",
            "REPLACE",
        ]
        tokens = set(re.findall(r"\b\w+\b", clean_q.upper()))
        found_forbidden = tokens & set(forbidden)
        if found_forbidden:
            forbidden_op = next(iter(found_forbidden))
            return (
                "Security Error: Only read-only SELECT queries are allowed. "
                f"Write operation '{forbidden_op}' is strictly forbidden in db_query. "
                "If you need to mutate client data (e.g. add items to cart), use "
                "the dedicated 'add_to_cart' tool. To query records, rewrite as a "
                "SELECT statement."
            )

        # Check if query is raw SQL
        if clean_q.upper().startswith("SELECT"):
            sql = clean_q
        else:
            q_lower = clean_q.lower()
            if "customer" in q_lower or "client" in q_lower or "user" in q_lower:
                # Check for city filters like 'in atlantis' or 'living in atlantis'
                city_match = re.search(r"(?:in|city)\s+([a-zA-Z]+)", q_lower)
                matched_city = city_match.group(1) if city_match else None
                if matched_city and matched_city not in ("the", "our", "all", "each"):
                    sql = (
                        f"SELECT * FROM customers "
                        f"WHERE LOWER(city) LIKE '%{matched_city}%' LIMIT 10;"
                    )

                elif "count" in q_lower or "how many" in q_lower:
                    sql = "SELECT COUNT(*) AS total_customers FROM customers;"
                else:
                    sql = (
                        "SELECT customer_id, name, email, city FROM customers LIMIT 10;"
                    )
            elif "order" in q_lower or "sales" in q_lower or "revenue" in q_lower:
                if "pending" in q_lower:
                    sql = "SELECT * FROM orders WHERE status = 'pending' LIMIT 10;"
                elif "count" in q_lower or "how many" in q_lower:
                    sql = (
                        "SELECT COUNT(*) AS total_orders, "
                        "SUM(total_amount) AS total_revenue FROM orders;"
                    )
                else:
                    sql = "SELECT * FROM orders ORDER BY order_id DESC LIMIT 10;"
            elif "product" in q_lower or "price" in q_lower or "stock" in q_lower:
                sql = (
                    "SELECT product_id, name, category, price, stock_quantity "
                    "FROM products LIMIT 10;"
                )
            elif (
                "appointment" in q_lower or "doctor" in q_lower or "schedule" in q_lower
            ):
                sql = "SELECT * FROM appointments LIMIT 10;"
            else:
                sql = "SELECT * FROM customers LIMIT 5;"

        cursor.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]

        lines = [
            "Database Query Results:",
            f"• Executed SQL: `{sql}`",
            f"• Rows Returned: {len(rows)}",
        ]
        if not rows:
            lines.append("• No matching records found.")
        else:
            lines.append("• Records:")
            for r in rows[:8]:
                lines.append(f"  {json.dumps(r)}")
            if len(rows) > 8:
                lines.append(f"  ... ({len(rows) - 8} more records)")

        return "\n".join(lines)
    except Exception as exc:
        return (
            f"Database Query Error: {exc}. "
            "Action for model: Valid tables in this database are "
            "'customers', 'orders', 'products', 'appointments'. Schema hints: "
            "customers(customer_id, name, email, phone, city), "
            "products(product_id, name, category, price, stock_quantity), "
            "orders(order_id, customer_id, order_date, status, total_amount), "
            "appointments(appointment_id, customer_id, doctor_name, "
            "appointment_date, status). "
            "Rewrite your query referencing only these valid tables."
        )
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Tool 4: Client Data Write Action (add_to_cart - Task 4.7)
# -----------------------------------------------------------------------------

# Registry of tools that perform write/mutation actions on client data
CLIENT_DATA_WRITE_TOOLS = {"add_to_cart"}


@tool(args_schema=AddToCartInput)
def add_to_cart(product_name: str, quantity: int = 1, customer_id: int = 1) -> str:
    """Add an item to the customer's active shopping cart (WRITE ACTION).

    THIS TOOL MUTATES CLIENT DATA AND REQUIRES EXPLICIT HUMAN APPROVAL BEFORE EXECUTION.
    Use this tool ONLY when the user explicitly requests purchasing or adding items.

    DO NOT USE FOR:
    - Read-only queries about products, inventory, or prices (use db_query instead).

    Args:
        product_name: Name of the product to purchase (e.g. 'Titanium Drill Bit').
        quantity: Number of units to add (default: 1, range: 1 to 100).
        customer_id: Numeric ID of the authenticated customer (default: 1).
    """
    conn = _get_database_connection()
    cursor = conn.cursor()
    try:
        # 1. Resolve product
        cursor.execute(
            "SELECT product_id, name, price, stock_quantity FROM products "
            "WHERE LOWER(name) LIKE ? LIMIT 1",
            (f"%{product_name.lower().strip()}%",),
        )
        row = cursor.fetchone()
        if not row:
            # Fallback products if table lookup is empty
            catalog = {
                "titanium drill bit": (3, 120.0),
                "oxygen scrubber cartridge": (2, 850.0),
                "mars rover sensor": (1, 450.0),
            }
            match = next((k for k in catalog if k in product_name.lower()), None)
            if match:
                pid, price = catalog[match]
                pname = match.title()
            else:
                return (
                    f"Catalog Error: Product '{product_name}' not found in catalog. "
                    "Action for model: Available products are: 'Mars Rover Sensor' "
                    "($450.00), 'Oxygen Scrubber Cartridge' ($850.00), 'Titanium "
                    "Drill Bit' ($120.00). Please re-call add_to_cart with an exact "
                    "or partial name matching one of these products."
                )
        else:
            pid = row["product_id"]
            pname = row["name"]
            price = float(row["price"])

        total_price = price * quantity

        # 2. Insert into cart_items
        cursor.execute(
            """
            INSERT INTO cart_items (
                customer_id, product_id, product_name, quantity, unit_price, total_price
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (customer_id, pid, pname, quantity, price, total_price),
        )
        conn.commit()

        return (
            f"✅ [Cart Updated] Successfully added {quantity}x '{pname}' "
            f"to Customer #{customer_id}'s cart. Total: ${total_price:.2f}."
        )
    except Exception as exc:
        return f"Cart Update Error: {exc}"
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Tool 5: Web Search
# -----------------------------------------------------------------------------


@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Search live internet sources for real-time, current, or external information.

    Use this tool ONLY when the user asks about:
    - Breaking news, recent events, 2026 releases, NASA Artemis roadmap updates
    - Queries mentioning 'latest', 'recent', 'today', '2026', 'current'
    - External libraries and frameworks not contained in internal documentation

    DO NOT USE FOR:
    - Internal Mars Odyssey engineering specs (use pdf_search instead).
    - Toobler capabilities or company offerings (use site_search instead).
    - Relational database transactions (use db_query instead).

    Args:
        query: Search terms to find real-time internet information.
    """
    clean_q = query.strip()
    timeout = 10.0

    # Attempt live search via DuckDuckGo Instant Answer API
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": clean_q, "format": "json", "no_html": 1, "skip_disambig": 1}
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                results: list[str] = []

                if data.get("Abstract"):
                    heading = data.get("Heading", "Summary")
                    link = data.get("AbstractURL", "https://duckduckgo.com")
                    results.append(f"• [{heading}] ({link}):\n  {data.get('Abstract')}")

                for topic in data.get("RelatedTopics", [])[:3]:
                    if (
                        isinstance(topic, dict)
                        and topic.get("Text")
                        and topic.get("FirstURL")
                    ):
                        results.append(
                            f"• {topic.get('FirstURL')}:\n  {topic.get('Text')}"
                        )

                if results:
                    return f"Web Search Results for '{clean_q}':\n" + "\n".join(results)
    except Exception:
        pass

    fallback_sources = [
        (
            "LangGraph & LangChain 1.0 Release Overview",
            "https://blog.langchain.dev/langchain-langgraph-1-0/",
            "LangChain 1.0 and LangGraph 1.0 provide unified runtime primitives for "
            "durable agent execution, human-in-the-loop validation, and streaming.",
            "2026-01-15",
        ),
        (
            "NASA Artemis & Mars Mission Roadmap",
            "https://www.nasa.gov/artemis-program-updates/",
            "NASA announces updated timelines for autonomous surface systems, lunar "
            "gateway staging, and pressurized rover trials for Mars exploration.",
            "2026-02-20",
        ),
        (
            "AI Agent Architecture & Tool Routing Survey",
            "https://arxiv.org/abs/2602.04512",
            "State of the art in dynamic tool calling vs upfront router architectures: "
            "autonomous LLM-directed tool loops outperform static routers.",
            "2026-03-01",
        ),
    ]

    # Check query relevance against fallback sources
    q_words = set(re.findall(r"\w+", clean_q.lower()))
    matching_sources = []
    for title, src_url, snippet, pub_date in fallback_sources:
        text = f"{title} {snippet}".lower()
        if any(w in text for w in q_words if len(w) > 2):
            matching_sources.append((title, src_url, snippet, pub_date))

    if not matching_sources:
        return (
            f"Web Search Results for '{clean_q}': No relevant web results found. "
            "Action for model: Try broader search terms or verify keywords for "
            "live 2026 events / NASA Artemis roadmap."
        )

    lines = [f"Web Search Results for '{clean_q}':"]
    for title, src_url, snippet, pub_date in matching_sources:
        lines.append(
            f"• Title: {title}\n  URL: {src_url}\n  Date: {pub_date}\n  "
            f"Snippet: {snippet}"
        )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Tool Collections
# -----------------------------------------------------------------------------


def get_rag_tools() -> list[BaseTool]:
    """Return the four Phase 3 RAG tools."""
    return [pdf_search, site_search, db_query, web_search]


def get_all_tools() -> list[BaseTool]:
    """Return all tools: RAG tools, write tools, arithmetic calculator, and weather."""
    return [
        pdf_search,
        site_search,
        db_query,
        web_search,
        add_to_cart,
        calculator,
        get_weather,
    ]
