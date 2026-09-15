# PG Recommends: System Specifications & Test Matrix

## Epic Overview
**PG Recommends** is an enterprise-grade, multi-tenant AI film curator and personalized movie recommendation platform. Built around a core human taste profile derived from Pranav's authentic Letterboxd reviews (`reviews.csv`), live Letterboxd RSS syndication (`https://letterboxd.com/pranavg/rss/`), persistent multi-turn conversational intelligence, and a Letterboxd-inspired frosted-glass React application.

---

## User Stories & Acceptance Criteria

### Story 1: Multi-Source Film Ingestion & Live Sync
* **As a** curator (PG),
* **I want to** ingest historical Letterboxd reviews (`reviews.csv`) and continuously sync live reviews from my RSS feed,
* **So that** the knowledge base stays up to date without duplicate entries.
* **Acceptance Criteria**:
  1. `reviews.csv` (5,600+ records) parses into structured `MovieReview` objects (title, year, rating, review text, Letterboxd URI, watched date).
  2. Live RSS feed (`https://letterboxd.com/pranavg/rss/`) parses XML items with titles, ratings, review CDATA, and poster images.
  3. Incremental sync is idempotent: duplicate GUIDs or URIs are updated or skipped, never duplicated.
* **Test File**: `tests/test_ingestion.py`

### Story 2: Hybrid Retrieval & Citation Engine
* **As a** film enthusiast,
* **I want to** search for films by genre, director, mood, or thematic tropes,
* **So that** I get curated suggestions vetted by PG with authentic star ratings and Letterboxd citations.
* **Acceptance Criteria**:
  1. Combines BM25 keyword matching with dense semantic embeddings via Reciprocal Rank Fusion (RRF).
  2. Every returned result includes verifiable citations (`movie_id`, `title`, `year`, `rating`, `letterboxd_url`, and review excerpt).
  3. Supports filtering by minimum star rating and genres.
* **Test File**: `tests/test_retrieval.py`

### Story 3: Dynamic User Taste Learning & Tenant Isolation
* **As an** end user,
* **I want the** system to learn my personal film taste across dialogue turns while preserving privacy,
* **So that** recommendations evolve to match my likes without leaking my data to other tenants.
* **Acceptance Criteria**:
  1. User taste profile extracts liked/disliked directors, genres, and mood preferences.
  2. All queries and taste profiles enforce `tenant_id` and `user_id` scoping.
  3. Cross-tenant access attempts are strictly rejected.
* **Test File**: `tests/test_taste_engine.py`

### Story 4: Stateful LangGraph Agent & Human Escalation
* **As a** user needing bespoke curation or customer service,
* **I want to** converse with a stateful agent and escalate to PG directly when needed,
* **So that** complex requests receive human attention.
* **Acceptance Criteria**:
  1. Agent state is checkpointed per conversation.
  2. Automatic retry/rewriting cycle if initial search yields 0 matches.
  3. "Ask PG Directly" triggers an escalation node, creates a `human_escalations` ticket, and hands off gracefully.
* **Test File**: `tests/test_agent.py`

### Story 5: Production Guardrails & Output Safety
* **As a** platform administrator,
* **I want** prompt injection defense, SQL query parameterization, and PII log scrubbing,
* **So that** untrusted movie reviews cannot hijack the agent and sensitive tokens are not logged.
* **Acceptance Criteria**:
  1. External reviews wrapped in `<untrusted_context>` XML envelopes.
  2. Model outputs safely escaped against XSS.
  3. 0.0% PII in telemetry logs.
* **Test File**: `tests/test_guardrails.py`

### Story 6: Streaming API Gateway & Frosted Glass UI
* **As a** web user,
* **I want a** responsive Letterboxd-inspired frosted glass interface with real-time streaming tokens,
* **So that** chatting with PG Recommends feels fluid and immersive.
* **Acceptance Criteria**:
  1. FastAPI `POST /api/chat/stream` emits Server-Sent Events (SSE) tokens and citation events.
  2. React 19 UI renders star ratings, review quote cards, and live taste radar.
* **Test File**: `tests/test_api.py`

---

## Test Execution Matrix

| Test Module | Target Functionality | Passing Metric |
| :--- | :--- | :--- |
| `test_ingestion.py` | CSV parsing, RSS XML parsing, deduplication | 100% pass, 0 duplicates |
| `test_retrieval.py` | Hybrid RRF, citation generation, rating filter | Recall@5 >= 0.80, valid URIs |
| `test_taste_engine.py` | Preference extraction, tenant isolation | Zero cross-tenant leakage |
| `test_agent.py` | LangGraph cycles, escalation ticket creation | Persistent checkpoint reload |
| `test_guardrails.py` | Injection neutralizing, PII redactor | Zero prompt hijack, zero PII |
| `test_api.py` | SSE streaming endpoint, healthcheck | HTTP 200, valid SSE event stream |
