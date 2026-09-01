# Phase 0: Baseline Setup & FastAPI Service

This directory contains the baseline Python development environment and a FastAPI service implementing `/echo` and `/stream` endpoints.

---

## Features

- **Package Management:** Managed with [uv](https://docs.astral.sh/uv/) and locked in `uv.lock`.
- **Code Quality:** Linting & formatting via `ruff`, static type analysis via `pyright`.
- **Testing:** Automated tests via `pytest` and `httpx` (`TestClient`).
- **FastAPI Endpoints:**
  - `POST /echo`: Accepts a Pydantic-validated payload (`message`, optional `metadata`) and returns a typed `EchoResponse` with timestamp.
  - `GET /stream`: Emits 20 chunks spaced 100ms apart using `StreamingResponse`.

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

**Sample Response:**
```json
{
  "message": "Hello from client!",
  "received_at": "2026-09-01T11:18:00Z",
  "metadata": {
    "env": "dev"
  }
}
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
