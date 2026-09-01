FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

# Optimize python and uv runtime
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Copy dependency definition and lockfile
COPY pyproject.toml uv.lock ./

# Install locked dependencies (excluding dev dependencies for lean production image)
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source and documentation
COPY README.md ./
COPY src/ ./src/

# Install project package
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8000

# Default command to run FastAPI via uvicorn
CMD ["uv", "run", "uvicorn", "phase_0_baseline.app:app", "--host", "0.0.0.0", "--port", "8000"]
