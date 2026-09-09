# Architectural Comparison: Database RAG vs. Vector Search (Project C)

When building client chatbots that answer live, user-specific questions like *"What is my next appointment?"* or *"Where is my order?"*, AI engineers have three distinct architectural approaches:
1. **Text-to-SQL (Task 3.14)**: The model writes raw SQL queries against a schema.
2. **Structured Tool Calling (Task 3.15)**: The model fills arguments into predefined, typed functions.
3. **Vector / Semantic Search (Tasks 3.1 - 3.13)**: Chunking, embeddings, and similarity retrieval over unstructured text.

Each approach addresses fundamentally different data patterns, security profiles, and query complexities.

---

## 1. Comprehensive Three-Way Decision Matrix

| Dimension | Approach 1: Text-to-SQL (3.14) | Approach 2: Structured Tool Calls (3.15) | Approach 3: Vector Search (3.1-3.13) |
| :--- | :--- | :--- | :--- |
| **Primary Use Case** | Ad-hoc analytics, aggregations, arbitrary joins across multiple tables | **Customer transactional data & PII** (orders, appointments, account balances) | Unstructured text, manuals, policies, troubleshooting guides |
| **Model Responsibility** | Writes full SQL query text | **Extracts typed arguments only** (`customer_id`, `from_date`, `to_date`) | Embeds question query for vector similarity match |
| **Security Risk** | **High**: SQL injection, schema exfiltration, complex AST validation required | **Minimal**: Parameterized queries only (`?`), zero SQL injection possible | Low: Context leakage if tenant partitioning is absent |
| **Tenant Isolation** | Hard to enforce reliably in generated SQL (model might omit `WHERE customer_id = ?`) | **Trivial & strict**: Tenant ID can be injected/verified directly from session tokens | Moderate: Requires metadata filtering on vector chunks |
| **Flexibility** | **Maximum**: Can answer unexpected questions joining any allowlisted tables | **Constrained**: Only answers questions supported by exposed tools | High: Matches broad semantic concepts and synonyms |
| **Predictability** | Lower: Model might write slow, unindexed queries or syntax errors | **Highest**: 100% deterministic queries pre-indexed and tuned by developers | Medium: Probabilistic ranking (top-k semantic neighbors) |
| **Latency** | 1x LLM call (generate SQL) + DB query + 1x synthesis call | 1x LLM call (tool call) + DB query + 1x synthesis call (or single turn) | Vector embedding latency + vector index KNN search + synthesis call |
| **Data Freshness** | Instant real-time (reads live DB tables) | Instant real-time (reads live DB tables) | Stale until chunks are re-embedded and synchronized |

---

## 2. Approach 1: Text-to-SQL (When to Use)

### Ideal Scenarios:
1. **Internal Business Intelligence & Analytics**:
   - *"What was our total revenue from MacBook sales in August 2026?"*
   - *"How many customers in Chicago have placed orders exceeding $500?"*
2. **Unpredictable, Multi-Table Reporting**:
   - Queries requiring ad-hoc joins across orders, items, products, and categories where developers cannot anticipate every possible combination.
3. **Internal Administrative Dashboards**:
   - Where the user is a trusted analyst or engineer, and the system needs to support open-ended analytical exploration.

### Mandatory Safeguards Required:
- **Pure `SELECT` AST verification**: Reject any DDL/DML node (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `PRAGMA`).
- **Table Allowlist**: Parse AST and reject any table not explicitly permitted.
- **Mandatory LIMIT Clamping**: Enforce a strict upper bound (e.g. `LIMIT 50`) to prevent denial-of-service memory exhaustion.
- **Read-Only Database Connection**: `file:...mode=ro` and `PRAGMA query_only = ON`.

---

## 3. Approach 2: Structured Tool Calls (When to Use)

> **Golden Rule**: Default to **Structured Tool Calls** for anything touching customer data, personal information, or customer-facing production chatbots.

### Ideal Scenarios:
1. **Customer-Facing Self-Service Chatbots**:
   - *"What is my next appointment?"*
   - *"Where is my order #104?"*
   - *"Cancel my consultation on Friday."*
2. **Strict Multi-Tenant Customer Data (PII)**:
   - In production, you never trust an LLM to remember to include `WHERE customer_id = 42`.
   - With structured tool calling, the backend framework (FastAPI/Express) can automatically bind the authenticated `customer_id` from the user's JWT/session cookie directly into `get_appointments(customer_id=session.user_id)`. The model cannot tamper with or bypass this argument!
3. **High-Throughput, Low-Latency Endpoints**:
   - Pre-written SQL queries use database indexes (`INDEX ON appointments(customer_id, scheduled_time)`), guaranteeing sub-millisecond query execution without runaway query plans.
4. **Zero SQL Injection Threat Surface**:
   - Since the LLM never generates SQL text, SQL injection is architecturally impossible.

### Inherent Trade-Offs:
- **Less Flexible**: If the customer asks *"Show me appointments where the technician's favorite color is blue"*, the tool cannot answer unless an engineer explicitly coded that parameter.
- **Schema Evolution**: Adding new query dimensions requires writing new tool schemas and Python handler functions.

---

## 4. Approach 3: Vector / Semantic Search (When to Use)

### Ideal Scenarios:
1. **Fuzzy, Descriptive, or Ambiguous Queries**:
   - *"Do you have any quiet wireless peripherals suitable for open offices?"*
   - *"Show me products similar to the Sony noise-cancelling headphones."*
2. **Unstructured Knowledge & Documentation**:
   - Troubleshooting manuals, return policies, onboarding guides, long customer feedback transcripts.
3. **Multi-Modal or Freeform Text**:
   - Searching across customer service chat transcripts or product reviews.

### Inherent Limitations:
- **Cannot do arithmetic**: Cannot calculate counts, averages, sums, or min/max dates.
- **Staleness**: Live appointment schedule changes are not immediately reflected without re-indexing.

---

## 5. The Production Hybrid Architecture (The Unified Stack)

In production enterprise AI systems, the three approaches combine into a tiered query router:

```
                  [User Prompt]
                        │
                        ▼
               ┌────────────────┐
               │  Query Router  │
               └───────┬────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
[Customer PII /  [Analytical /   [Documentation /
 Live Status]     Ad-hoc BI]      Knowledge]
       │               │               │
       ▼               ▼               ▼
 ┌───────────┐   ┌───────────┐   ┌───────────┐
 │Structured │   │Text-to-SQL│   │Vector RAG │
 │Tool Calls │   │ (5-layer  │   │  (Hybrid  │
 │ (Task 3.15│   │ AST guard)│   │  Search)  │
 └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
            [Synthesized Final Answer]
```

### Example Hybrid Routing Scenarios:
- **Customer Query**: *"Can I return order #104 under your policy?"*
  1. **Structured Tool Call**: `get_order_details(order_id=104)` -> Returns delivered date: Sept 1, 2026.
  2. **Vector RAG**: Searches `return_policy.md` -> Returns 30-day return window.
  3. **LLM Synthesis**: Calculates today is Sept 9, 2026 (8 days later) -> *"Yes, your order was delivered on Sept 1, 2026, which is within our 30-day return window."*
