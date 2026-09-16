# Stage 1: Frontend Build
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend Runtime
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Copy project files
COPY pyproject.toml README.md reviews.csv ./
COPY src/ ./src/

# Install Python dependencies
RUN uv sync --no-dev

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

CMD ["sh", "-c", "uv run uvicorn phase_6_capstone.server:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
