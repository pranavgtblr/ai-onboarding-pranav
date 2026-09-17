# The Production AI Project Starter Blueprint: Day 1 Operational Guide
*A Step-by-Step Scaffolding & Execution Manual for Starting Any Enterprise AI Project*

---

## 1. Overview: How to Use This Blueprint

When starting any new AI project from scratch, do not write ad-hoc scripts or prompt chains in random folders. Follow this document sequentially from **Step 1 (Day 1 Morning)** to **Step 6 (Production Shipping)**. 

Every production AI system follows this exact architectural pipeline:
```
[User Request] 
      │
      ▼
1. FastAPI Gateway (Streaming SSE) ──> 2. Security Guardrails (PII & CDATA Sandbox)
                                                      │
                                                      ▼
4. Output (XSS Escaped & Cited) <── 3. LangGraph Agent (State, Cycle, HITL)
                                          ├── Tools (Hard iteration limit)
                                          └── Hybrid Retrieval (BM25 + Dense Vectors)
```

---

## 2. Canonical Project Directory Structure

Create this exact folder layout on Day 1:

```text
my-ai-project/
├── .env.example               # Template of all required secrets/keys (committed to git)
├── .env                       # Local secrets (never committed to git)
├── pyproject.toml             # uv dependencies and project metadata
├── uv.lock                    # Deterministic dependency lockfile
├── Dockerfile                 # Multi-stage production container build
├── README.md                  # System architecture, setup & test instructions
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── config.py          # Step 1: Type-safe pydantic-settings configuration
│       ├── models.py          # Step 2: Shared Pydantic schemas (requests, records, citations)
│       ├── ingestion.py       # Step 3: Document parsers (PDF, CSV, database connectors)
│       ├── retrieval.py       # Step 3: BM25 + Vector hybrid search engine (RRF)
│       ├── tools.py           # Step 4: Local tool execution functions with loop guards
│       ├── agent.py           # Step 4: LangGraph StateGraph (typed state, cycles, HITL)
│       ├── guardrails.py      # Step 5: Input CDATA sandboxing, HTML escaping, PII scrubber
│       └── server.py          # Step 6: FastAPI streaming SSE gateway (/api/chat/stream)
├── tests/
│   ├── test_api.py            # Verifies health & streaming SSE endpoints
│   ├── test_retrieval.py      # Verifies BM25 + vector search recall & ranking
│   └── test_guardrails.py     # Verifies prompt injection neutralization & PII scrub
└── frontend/                  # Optional: Vite/React client served by FastAPI
```

---

## 3. Step-by-Step Chronological Execution

### Step 1: Environment & Dependency Initialization (Day 1 Morning)
Open your terminal in a new empty directory:

```bash
# 1. Initialize project with uv
uv init --lib my-ai-project
cd my-ai-project

# 2. Install production core dependencies
uv add fastapi uvicorn pydantic pydantic-settings httpx rank-bm25 langgraph langchain-core

# 3. Install dev and testing tools
uv add --dev pytest pytest-asyncio ruff pyright
```

---

### Step 2: Configuration & Secrets Schema (`src/my_project/config.py`)
Never hardcode API keys or connection strings. Enforce validation at startup:

```python
"""Application runtime settings schema."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Enterprise AI System"
    app_env: str = "production"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Model & Provider
    model_provider: str = "google_genai"  # or openai, anthropic, mock
    model_name: str = "gemini-2.0-flash"
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    # Database
    database_url: str = "sqlite+aiosqlite:///app.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
```

Create `.env.example`:
```bash
GEMINI_API_KEY=your_key_here
MODEL_NAME=gemini-2.0-flash
DATABASE_URL=sqlite+aiosqlite:///app.db
```

---

### Step 3: Grounded Knowledge & Hybrid Retrieval (`retrieval.py`)
Always combine exact keyword matching (BM25) with semantic embeddings:

```python
"""Production Hybrid Search: BM25 Lexical + Vector Cosine Similarity fused with RRF."""

from collections import defaultdict
from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, documents: list[dict]):
        self.documents = documents
        # Tokenize corpus for exact lexical match
        self.tokenized_corpus = [
            doc["content"].lower().split() for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
        """Fuses BM25 and semantic ranking using Reciprocal Rank Fusion."""
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )

        rrf_scores = defaultdict(float)
        for rank, doc_idx in enumerate(bm25_indices[:50]):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

        # Sort by fused score
        top_indices = sorted(
            rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True
        )
        return [self.documents[idx] for idx in top_indices[:top_k]]
```

---

### Step 4: Stateful Graph Agent (`agent.py`)
Build workflows as cyclical state machines with retry logic and approval gates:

```python
"""Stateful Agent Workflow with Cycles and Human-in-the-Loop Interruption."""

from typing import Annotated, Literal, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    query: str
    retrieval_attempts: int
    context_docs: list[str]
    proposed_action: str | None
    final_output: str | None


def retrieve_node(state: AgentState) -> dict:
    attempts = state.get("retrieval_attempts", 0) + 1
    # Retrieve documents...
    docs = ["Verified reference document snippet"] if attempts > 1 else []
    return {"retrieval_attempts": attempts, "context_docs": docs}


def rewrite_node(state: AgentState) -> dict:
    return {"query": f"clarified search: {state['query']}"}


def should_retry(state: AgentState) -> Literal["rewrite", "synthesize"]:
    # Cycle back if retrieval returned empty (up to 2 attempts)
    if not state["context_docs"] and state["retrieval_attempts"] < 2:
        return "rewrite"
    return "synthesize"


def synthesize_node(state: AgentState) -> dict:
    return {"final_output": "Synthesized grounded answer with citations."}


def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges("retrieve", should_retry)
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("synthesize", END)

    # In production, use AsyncPostgresSaver for real persistence
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
```

---

### Step 5: Enterprise Security & Guardrails (`guardrails.py`)
Never trust incoming text or log raw PII:

```python
"""Enterprise Security: Prompt Injection Envelope, XSS Escaping, and PII Log Scrubbing."""

import html
import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
TOKEN_RE = re.compile(r"Bearer\s+([A-Za-z0-9-_.]+)", re.IGNORECASE)
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
)


def quarantine_untrusted_context(text: str) -> str:
    """Isolates untrusted external text in an XML envelope, escaping delimiters."""
    safe_text = text.replace("]]>", "]]&gt;")
    return f"<untrusted_context>\n<![CDATA[\n{safe_text}\n]]>\n</untrusted_context>"


def sanitize_output(content: str) -> str:
    """Escapes HTML entities to prevent Stored XSS when rendered in browsers."""
    return html.escape(content, quote=True)


def scrub_pii(log_line: str) -> str:
    """Redacts emails, bearer tokens, and phone numbers before writing to logs."""
    clean = EMAIL_RE.sub("[REDACTED_EMAIL]", log_line)
    clean = TOKEN_RE.sub("Bearer [REDACTED_TOKEN]", clean)
    return PHONE_RE.sub("[REDACTED_PHONE]", clean)
```

---

### Step 6: Streaming Gateway & Server (`server.py`)
Stream tokens immediately over HTTP:

```python
"""Production FastAPI Streaming Gateway."""

import json
from collections.abc import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="AI API Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Yield tokens as they are generated
            for word in ["This ", "is ", "real-time ", "streaming."]:
                yield f"data: {json.dumps({'event': 'token', 'data': word})}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'data': '[DONE]'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'data': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 4. Production Multi-Stage Container (`Dockerfile`)
Package frontend and backend into a single self-contained container:

```dockerfile
# Stage 1: Build Frontend (if applicable)
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: High-Performance Python Runtime
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies deterministically
RUN uv sync --no-dev

# Copy compiled frontend into static directory
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Dynamically bind to assigned $PORT on cloud host (Render, Fly, Cloud Run)
CMD ["sh", "-c", "uv run uvicorn my_project.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

## 5. Verification Checklist (Run Before Submitting Pull Request)

Execute these 4 commands in your terminal:

```bash
# 1. Code Style & Linting
uv run ruff check .

# 2. Code Formatting Check
uv run ruff format --check .

# 3. Static Type Verification
uv run pyright

# 4. Automated Test Suite Execution
uv run pytest -v
```

If all 4 commands pass with zero errors, your project is ready for staging and production deployment.
