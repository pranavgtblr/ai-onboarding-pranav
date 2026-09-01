# Phase 0: Baseline Setup & FastAPI Service

This directory contains the baseline Python development environment, a FastAPI service implementing `/echo` and `/stream` endpoints, and an asynchronous concurrency benchmark comparing `asyncio.gather` vs. sequential execution.

---

## Features

- **Package Management:** Managed with [uv](https://docs.astral.sh/uv/) and locked in `uv.lock`.
- **Code Quality:** Linting & formatting via `ruff`, static type analysis via `pyright`.
- **Testing:** Automated tests via `pytest` and `httpx` (`TestClient`).
- **FastAPI Endpoints:**
  - `POST /echo`: Accepts a Pydantic-validated payload (`message`, optional `metadata`) and returns a typed `EchoResponse` with timestamp.
  - `GET /stream`: Emits 20 chunks spaced 100ms apart using `StreamingResponse`.
  - `GET /delay/{ms}`: Simulates endpoint latency for benchmarking.
- **Async Concurrency Benchmark (Task 0.4):** Evaluates `asyncio.gather` concurrent HTTP GETs vs. sequential execution.

---

## Prerequisites

- Python `>= 3.10`
- [uv](https://docs.astral.sh/uv/) package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation

```bash
cd phase-0-baseline
uv sync
```

---

## Concurrency Benchmark (Task 0.4)

### Running the Benchmark

```bash
uv run concurrency-demo
```

### Benchmark Results (10 Requests @ 100ms latency each)

```text
==================================================
Total Requests:        10
Sequential Duration:   1.0134 seconds
Concurrent Duration:   0.1163 seconds
Speedup Factor:        8.71x faster
==================================================
```

### Understanding the Gap

#### 1. Sequential Execution ($O(N \times \text{Latency})$)
* Each HTTP request is sent and awaited individually in a synchronous loop.
* The application spends almost all of its time idle, waiting for the server to reply before dispatching the next request.
* **Duration:** $\approx 10 \times 100\text{ms} = 1000\text{ms}$ ($1.01\text{s}$).

#### 2. Concurrent Execution via `asyncio.gather` ($O(\max(\text{Latency}))$)
* All 10 requests are initiated immediately in the Python event loop.
* When each request reaches an I/O wait (`await client.get(...)`), control yields back to the event loop to fire the next request without waiting for the response.
* The operating system handles network socket I/O concurrently.
* Once the 100ms latency elapses, all responses arrive nearly simultaneously.
* **Duration:** $\approx \max(\text{latencies}) + \text{event loop overhead} = 116\text{ms}$ ($0.116\text{s}$).

#### 3. Why This Matters for LLMs & AI Systems
* LLM operations (calling external tool APIs, querying vector databases, retrieving web search results, or generating embeddings for chunks) are **I/O-bound**.
* Firing multiple LLM or retrieval calls sequentially accumulates round-trip latencies, severely degrading user experience. Using `asyncio.gather` ensures that overall latency is bounded by the slowest single request rather than the sum of all requests.

---

## Running the FastAPI Service

Start the local development server:

```bash
uv run uvicorn phase_0_baseline.app:app --reload --port 8000
```
*(Alternatively: `uv run phase-0-baseline`)*

The interactive API documentation is available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Testing Endpoints via cURL

### 1. Test `POST /echo`
```bash
curl -X POST http://localhost:8000/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello from client!", "metadata": {"env": "dev"}}'
```

### 2. Test `GET /stream` (Emits 20 chunks with 100ms delay)
```bash
curl -N http://localhost:8000/stream
```

---

## Quality Checks & Automated Tests

### Run All Tests
```bash
uv run pytest
```

### Run Linter
```bash
uv run ruff check .
```

### Run Format Check
```bash
uv run ruff format --check .
```

### Run Type Checker
```bash
uv run pyright
```

---

## CI/CD Integration

All checks (`ruff`, `pyright`, `pytest`) run automatically across all project phases on every pull request and push via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
