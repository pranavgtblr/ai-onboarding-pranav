# Architectural Comparison: Text-to-SQL vs. Vector Search for Live Data (Project C)

When building client chatbots that answer live, user-specific questions like *"What is my next appointment?"* or *"Where is my order?"*, AI engineers have two primary architectural approaches: **Text-to-SQL** and **Vector / Embedding Search**. 

Each approach addresses fundamentally different data patterns, security profiles, and query complexities.

---

## 1. Quick Decision Matrix

| Dimension | Approach A: Text-to-SQL | Approach B: Vector / Semantic Search |
| :--- | :--- | :--- |
| **Data Nature** | Highly structured, relational, transactional (PostgreSQL, MySQL, SQLite) | Unstructured or semi-structured text, prose, manuals, product catalogs |
| **Exactness** | 100% deterministic mathematical precision (exact dates, sums, counts) | Probabilistic fuzzy relevance (semantic similarity, nearest neighbors) |
| **Aggregations** | Native SQL: `COUNT()`, `SUM()`, `AVG()`, `GROUP BY`, `HAVING` | Incapable of reliable mathematical aggregations or arithmetic |
| **Latency** | 1x LLM call (generate SQL) + 1-5ms DB query + 1x synthesis call | Embedding vectorization + vector DB search (KNN) + synthesis call |
| **Freshness** | Instant real-time zero-lag (reads active live DB rows) | Requires continuous index synchronization / embedding re-indexing |
| **Security Stakes** | **Highest**: Risk of SQL injection, data leakage, drop/write attacks | Lower: Risk of retrieving unauthorized context if tenancy is unpartitioned |
| **Cost** | Low: Small token prompt describing schema; no vector database hosting | Higher: Vector database hosting costs + embedding API calls per update |

---

## 2. Approach A: Text-to-SQL (When to Use)

### Ideal Scenarios:
1. **Live Transactional Status**:
   - *"Where is order #8841?"*
   - *"Is my appointment tomorrow confirmed or cancelled?"*
2. **Mathematical Calculations & Summaries**:
   - *"What was our total revenue from MacBook sales in August 2026?"*
   - *"How many customers in Seattle have more than 3 completed orders?"*
3. **Complex Relational Joins**:
   - Joining `customers` $\rightarrow$ `orders` $\rightarrow$ `order_items` $\rightarrow$ `products`.
4. **Zero-Staleness Requirements**:
   - The user updated their delivery address 2 seconds ago. Text-to-SQL queries the live database immediately with zero indexing latency.

### Core Security Mandates:
1. **Read-Only Database Role**:
   - Configure credentials at the database level with only `SELECT` privileges (e.g. `PRAGMA query_only = ON;`, `mode=ro`).
2. **Table Allowlisting**:
   - Parse the query AST before execution and verify every referenced table is on a strict permit list. Prevent access to internal tables (`sqlite_master`, audit logs, credential vaults).
3. **Mandatory LIMIT Clamping**:
   - If an LLM generates `SELECT * FROM orders;`, prevent runaway memory exhaustion by injecting or clamping `LIMIT 50`.
4. **Zero String Interpolation**:
   - Never interpolate user input directly into SQL strings (`f"SELECT * WHERE id = '{input}'"`). Let the model write the whole query structure or use parameterized values.

---

## 3. Approach B: Vector / Semantic Search (When to Use)

### Ideal Scenarios:
1. **Fuzzy, Descriptive, or Ambiguous Queries**:
   - *"Do you have any quiet wireless peripherals suitable for open offices?"*
   - *"Show me products similar to the Sony noise-cancelling headphones."*
2. **Unstructured Knowledge & Documentation**:
   - Troubleshooting manuals, return policies, onboarding guides, long customer feedback transcripts.
3. **Multi-Modal or Freeform Text**:
   - Searching across customer service chat transcripts or product reviews.

### Inherent Limitations:
- Cannot reliably calculate arithmetic: Vector embeddings cannot answer *"How many orders were placed between 2 PM and 4 PM?"*
- Stale Data: If a product's price drops from 99 USD to 79 USD, the vector database will return stale data until the record is re-embedded and synchronized.

---

## 4. The Production Hybrid Pattern (Best of Both Worlds)

For modern enterprise chatbots, the gold standard is **Query Routing**:
- **Router Layer**: Inspects the user's intent:
  - If the question involves live order status, account balances, appointments, or numbers $\rightarrow$ Route to **Text-to-SQL**.
  - If the question involves policy explanations, product recommendations, or troubleshooting guides $\rightarrow$ Route to **Vector Search**.
  - If both are needed (*"Can I return order #104 under your policy?"*) $\rightarrow$ Run Text-to-SQL to fetch order delivery date, fetch Return Policy via Vector RAG, and synthesize the final answer.
