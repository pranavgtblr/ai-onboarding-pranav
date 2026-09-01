# Phase 0: Baseline Setup

This directory sets up the baseline Python development environment with `uv`, linting via `ruff`, static type checking via `pyright`, and testing via `pytest`.

---

## Prerequisites

- Python `>= 3.10`
- [uv](https://docs.astral.sh/uv/) package manager

### Installing `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Getting Started

1. Navigate to the `phase-0-baseline` directory:
   ```bash
   cd phase-0-baseline
   ```

2. Synchronize virtual environment and install all dependencies:
   ```bash
   uv sync
   ```

---

## Running Quality Checks and Tests

### 1. Run Linter (Ruff)
```bash
uv run ruff check .
```

### 2. Check Code Formatting (Ruff)
```bash
uv run ruff format --check .
```
*(To auto-format, run `uv run ruff format .`)*

### 3. Run Static Type Checking (Pyright)
```bash
uv run pyright
```

### 4. Run Test Suite (Pytest)
```bash
uv run pytest
```

### 5. Run the Application
```bash
uv run phase-0-baseline
```

---

## CI/CD Integration

All checks (`ruff`, `pyright`, and `pytest`) are automated via GitHub Actions in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) and run automatically on every pull request and push.
