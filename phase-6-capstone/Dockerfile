# Multi-stage Dockerfile for PG Recommends
# Stage 1: Frontend Build
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend Runtime
FROM python:3.10-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml README.md reviews.csv ./
COPY src/ ./src/

# Copy sibling phase dependencies needed by editable paths
COPY ../phase-3-rag /phase-3-rag
COPY ../phase-4-agents /phase-4-agents
COPY ../phase-5-production /phase-5-production

# Install Python dependencies
RUN uv venv && uv sync --frozen || uv sync

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uv", "run", "uvicorn", "phase_6_capstone.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
