import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from phase_0_baseline.config import Settings, get_settings

settings: Settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="FastAPI baseline service with echo, stream, and config endpoints",
    version="0.1.0",
    debug=settings.debug,
)


class EchoRequest(BaseModel):
    message: str = Field(..., description="Message to echo back", min_length=1)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional key-value metadata"
    )


class EchoResponse(BaseModel):
    message: str
    received_at: datetime
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_env: str
    debug: bool


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check() -> HealthResponse:
    """Return health status and sanitized non-secret environment config."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        app_env=settings.app_env,
        debug=settings.debug,
    )


@app.post("/echo", response_model=EchoResponse, summary="Echo request payload")
async def echo(request: EchoRequest) -> EchoResponse:
    """Validate request body and return a typed echo response."""
    return EchoResponse(
        message=request.message,
        received_at=datetime.now(timezone.utc),
        metadata=request.metadata,
    )


async def generate_chunks(
    total_chunks: int = 20, delay_seconds: float = 0.1
) -> AsyncGenerator[str, None]:
    """Generate stream chunks with specified delay."""
    for i in range(1, total_chunks + 1):
        yield f"chunk {i}\n"
        if i < total_chunks:
            await asyncio.sleep(delay_seconds)


@app.get("/stream", summary="Stream 20 chunks with 100ms delay")
async def stream() -> StreamingResponse:
    """Stream 20 chunks emitted 100ms apart."""
    return StreamingResponse(
        generate_chunks(total_chunks=20, delay_seconds=0.1),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/delay/{ms}", summary="Simulate delayed endpoint response")
async def delay_endpoint(ms: int) -> dict[str, Any]:
    """Simulate server processing latency in milliseconds."""
    await asyncio.sleep(ms / 1000.0)
    return {"status": "ok", "delayed_ms": ms}


def start() -> None:
    """Entry point to run the FastAPI service using configured host/port."""
    import uvicorn

    active_settings = get_settings()
    uvicorn.run(
        "phase_0_baseline.app:app",
        host=active_settings.host,
        port=active_settings.port,
        reload=active_settings.debug or active_settings.app_env == "development",
    )
