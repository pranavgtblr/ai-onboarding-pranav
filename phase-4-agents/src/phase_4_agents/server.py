"""FastAPI Streaming Server for Task 4.9: Progressive intermediate step streaming.

Reuses the Phase 0 /stream endpoint pattern:
- GET /stream without query: returns Phase 0 baseline 20 chunks (text/plain).
- GET /stream with query: streams real-time intermediate agent steps (SSE).
- POST /stream: JSON payload streaming endpoint.
- GET /: Interactive Web UI for visualizing intermediate agent steps.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from phase_4_agents.graph_agent import (
    astream_agent_steps,
    build_state_graph_agent,
    handle_human_approval,
    list_state_history,
)
from phase_4_agents.web_ui import INDEX_HTML

app = FastAPI(
    title="Odyssey Agent Progressive Stream Service",
    description="Reuses Phase 0 /stream endpoint to stream agent steps to UI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared memory checkpointer for streaming sessions
_SHARED_CHECKPOINTER = MemorySaver()
_AGENT_CACHE: dict[str, Any] = {}


def get_cached_agent(provider: str = "mock") -> Any:
    """Get or instantiate cached agent with checkpointer."""
    if provider not in _AGENT_CACHE:
        _AGENT_CACHE[provider] = build_state_graph_agent(
            provider=provider,
            checkpointer=_SHARED_CHECKPOINTER,
        )
    return _AGENT_CACHE[provider]


# -----------------------------------------------------------------------------
# Phase 0 Baseline Generator (Backward Compatibility)
# -----------------------------------------------------------------------------


async def generate_phase_0_chunks(
    total_chunks: int = 20, delay_seconds: float = 0.1
) -> AsyncGenerator[str, None]:
    """Generate stream chunks with specified delay (Phase 0 baseline)."""
    for i in range(1, total_chunks + 1):
        yield f"chunk {i}\n"
        if i < total_chunks:
            await asyncio.sleep(delay_seconds)


# -----------------------------------------------------------------------------
# Request Models
# -----------------------------------------------------------------------------


class StreamRequest(BaseModel):
    """Payload for POST /stream."""

    query: str = Field(..., description="User query to process through agent.")
    provider: str = Field(default="mock", description="Model provider.")
    thread_id: str = Field(default="default-stream", description="Session thread ID.")


class ApprovalRequest(BaseModel):
    """Payload for POST /approve."""

    thread_id: str
    approved: bool
    rejection_reason: str = "Declined from web interface."
    provider: str = "mock"


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, summary="Agent Streaming Web UI")
@app.get("/ui", response_class=HTMLResponse, summary="Agent Streaming Web UI")
async def get_ui() -> HTMLResponse:
    """Serve the interactive real-time agent streaming web interface."""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/health", summary="Service health check")
async def health_check() -> dict[str, Any]:
    """Health check confirming service status and streaming capability."""
    return {
        "status": "healthy",
        "service": "phase-4-agents-stream",
        "phase_0_streaming_reused": True,
    }


@app.get("/stream", summary="Stream intermediate agent steps (reused Phase 0 endpoint)")
async def stream_endpoint(
    query: str | None = Query(default=None, description="Query to execute"),
    provider: str = Query(default="mock", description="Model provider"),
    thread_id: str = Query(default="default-stream", description="Session thread ID"),
) -> StreamingResponse:
    """Reuses Phase 0 /stream endpoint.

    If query is absent -> emits Phase 0 baseline 20 chunks (text/plain).
    If query is present -> streams real-time intermediate agent steps (SSE).
    """
    if not query:
        # Phase 0 baseline behavior preserved!
        return StreamingResponse(
            generate_phase_0_chunks(total_chunks=20, delay_seconds=0.1),
            media_type="text/plain; charset=utf-8",
        )

    agent = get_cached_agent(provider=provider)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in astream_agent_steps(agent, query, config=config):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/stream", summary="POST streaming endpoint for agent steps")
async def post_stream_endpoint(payload: StreamRequest) -> StreamingResponse:
    """Stream intermediate agent execution steps via POST request body."""
    agent = get_cached_agent(provider=payload.provider)
    config: RunnableConfig = {"configurable": {"thread_id": payload.thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event in astream_agent_steps(agent, payload.query, config=config):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/approve", summary="Submit human approval for paused write actions")
async def approve_endpoint(payload: ApprovalRequest) -> dict[str, Any]:
    """Approve or reject a paused write action on a given thread."""
    agent = get_cached_agent(provider=payload.provider)
    config: RunnableConfig = {"configurable": {"thread_id": payload.thread_id}}

    result = handle_human_approval(
        agent,
        config,
        approved=payload.approved,
        rejection_reason=payload.rejection_reason,
    )
    return {
        "status": "approved" if payload.approved else "rejected",
        "thread_id": payload.thread_id,
        "completed": True,
        "final_messages_count": len(result.get("messages", [])),
    }


@app.get("/history/{thread_id}", summary="Get checkpoint history for thread")
async def get_history_endpoint(
    thread_id: str, provider: str = "mock"
) -> dict[str, Any]:
    """Return checkpoint snapshots for a session thread."""
    agent = get_cached_agent(provider=provider)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    history = list_state_history(agent, config)
    return {"thread_id": thread_id, "checkpoints": history}


def main() -> None:
    """Run the FastAPI server via CLI."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Odyssey Agent Streaming Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port (8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (0.0.0.0)")
    args = parser.parse_args()

    print(f"🚀 Starting Odyssey Agent Streaming UI on http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
