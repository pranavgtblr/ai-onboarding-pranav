# Phase 0: Baseline Setup & FastAPI Service

This directory contains the complete baseline Python environment, application configuration management with `pydantic-settings`, a containerized FastAPI service (`/echo`, `/stream`, `/health`), and asynchronous benchmarking.

---

## Features

- **Package & Dependency Management:** Managed with [uv](https://docs.astral.sh/uv/) and locked in `uv.lock`.
- **Environment & Configuration:** Type-safe settings loaded from `.env` via `pydantic-settings`.
- **Code Quality:** Linting & formatting via `ruff`, static type analysis via `pyright`.
- **Testing:** 11 automated unit and integration tests via `pytest` and `httpx` (`TestClient`).
- **Containerization:** Production-ready `Dockerfile` optimized with `uv` multi-stage layer caching.
- **FastAPI Endpoints:**
  - `POST /echo`: Accepts a Pydantic-validated payload (`message`, optional `metadata`) and returns a typed `EchoResponse` with timestamp.
  - `GET /stream`: Emits 20 chunks spaced 100ms apart using `StreamingResponse`.
  - `GET /health`: Returns service health status and sanitized non-secret configuration.
  - `GET /delay/{ms}`: Simulates endpoint latency for benchmarking.
- **Async Concurrency Benchmark:** Evaluates `asyncio.gather` concurrent HTTP GETs vs. sequential execution.

---

## Prerequisites

- Python `>= 3.10`
- [uv](https://docs.astral.sh/uv/) package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Installation & Environment Setup

1. Synchronize the virtual environment:
   ```bash
   cd phase-0-baseline
   uv sync
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   > **Note:** `.env` is ignored by `.gitignore`. Never commit secrets or API keys to git.

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

### 1. Test `POST /echo` (Valid Payload)
```bash
curl -X POST http://localhost:8000/echo \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello from client!", "metadata": {"env": "dev"}}'
```

### 2. Test `POST /echo` (Invalid Payload -> 422 Error)
```bash
curl -i -X POST http://localhost:8000/echo \
  -H "Content-Type: application/json" \
  -d '{"message": ""}'
```

### 3. Test `GET /stream` (Progressive Stream)
```bash
curl -N http://localhost:8000/stream
```

### 4. Test `GET /health`
```bash
curl http://localhost:8000/health
```

---

## Docker Support

### Build the Image
```bash
docker build -t phase-0-baseline:latest .
```

### Run the Container
```bash
docker run -p 8000:8000 --env-file .env phase-0-baseline:latest
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

All checks (`ruff`, `pyright`, `pytest`, and `docker build`) run automatically across all project phases on every pull request and push via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
