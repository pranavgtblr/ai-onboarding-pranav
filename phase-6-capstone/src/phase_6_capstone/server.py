"""FastAPI Streaming Server and API Gateway for PG Recommends."""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from phase_6_capstone.agent import CapstoneAgent
from phase_6_capstone.config import settings
from phase_6_capstone.db import DatabaseManager, HumanEscalationModel
from phase_6_capstone.ingestion import (
    fetch_live_letterboxd_feed,
    parse_reviews_csv,
    sync_movies_to_store,
)
from phase_6_capstone.models import MovieRecord
from phase_6_capstone.retrieval import HybridMovieRetriever
from phase_6_capstone.taste_engine import TasteProfileManager


class ChatStreamRequest(BaseModel):
    tenant_id: str = "default_tenant"
    user_id: str = "guest_user"
    conversation_id: str = Field(default_factory=lambda: f"conv_{uuid4().hex[:8]}")
    message: str


class EscalateRequest(BaseModel):
    tenant_id: str = "default_tenant"
    user_id: str = "guest_user"
    conversation_id: str
    reason: str
    transcript_summary: str = ""


def create_app(
    db: DatabaseManager | None = None,
    initial_catalog: list[MovieRecord] | None = None,
    agent: CapstoneAgent | None = None,
) -> FastAPI:
    """Application factory for PG Recommends FastAPI backend."""
    db_manager = db or DatabaseManager(settings.database_url)
    movie_store: dict[str, MovieRecord] = {}

    if initial_catalog:
        for m in initial_catalog:
            key = m.letterboxd_url or m.movie_id
            movie_store[key] = m

    retriever = HybridMovieRetriever(list(movie_store.values()))
    taste_manager = TasteProfileManager(db_manager)
    curator_agent = agent or CapstoneAgent(
        retriever=retriever,
        taste_manager=taste_manager,
        db=db_manager,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Initialize DB tables
        await db_manager.init_models()

        # 2. Ingest CSV if store is empty
        if not movie_store:
            csv_path = Path(__file__).parent.parent.parent / "reviews.csv"
            if csv_path.exists():
                records = parse_reviews_csv(file_path=csv_path)
                sync_movies_to_store(records, movie_store)
                retriever.catalog = list(movie_store.values())
                retriever.__init__(retriever.catalog)

        yield

    app = FastAPI(
        title="PG Recommends API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "app": "PG Recommends",
            "curator": "Pranav G (@pranavg)",
            "movies_count": len(movie_store),
        }

    @app.get("/api/movies")
    async def list_movies(query: str = "", limit: int = 50):
        if query.strip():
            results = retriever.search(query, top_k=limit)
            return {"movies": [r.record.model_dump() for r in results]}

        sorted_movies = sorted(
            movie_store.values(),
            key=lambda x: (x.rating or 0.0, x.watched_date or ""),
            reverse=True,
        )
        return {"movies": [m.model_dump() for m in sorted_movies[:limit]]}

    @app.get("/api/taste-profile")
    async def get_taste_profile(tenant_id: str, user_id: str):
        profile = await taste_manager.get_profile(tenant_id, user_id)
        if not profile:
            return {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "liked_directors": [],
                "liked_genres": [],
                "disliked_elements": [],
                "mood_tags": [],
            }
        return profile.model_dump()

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatStreamRequest):
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for chunk in curator_agent.stream_turn(
                    tenant_id=req.tenant_id,
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    message=req.message,
                ):
                    payload = json.dumps(chunk)
                    yield f"data: {payload}\n\n"
            except Exception as exc:
                logger.error("Unhandled error in stream_turn: %s", exc, exc_info=True)
                error_payload = json.dumps({"event": "error", "data": str(exc)})
                yield f"data: {error_payload}\n\n"
                done_payload = json.dumps({"event": "done", "data": "[DONE]"})
                yield f"data: {done_payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.post("/api/escalate")
    async def escalate(req: EscalateRequest):
        ticket_id = f"TICK-{uuid4().hex[:8].upper()}"
        async with db_manager.session_factory() as session:
            escalation = HumanEscalationModel(
                ticket_id=ticket_id,
                tenant_id=req.tenant_id,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                reason=req.reason,
                transcript_summary=req.transcript_summary,
                status="pending",
            )
            session.add(escalation)
            await session.commit()

        return {
            "status": "escalated",
            "ticket_id": ticket_id,
            "message": "Escalated to PG directly.",
        }

    @app.post("/api/sync")
    async def sync_live_feed():
        try:
            live_records = await fetch_live_letterboxd_feed()
            stats = sync_movies_to_store(live_records, movie_store)
            retriever.catalog = list(movie_store.values())
            retriever.__init__(retriever.catalog)
            return {"status": "success", "stats": stats}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    dist_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
    index_file = dist_path / "index.html"
    assets_dir = dist_path / "assets"

    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def root():
        if index_file.exists():
            return FileResponse(index_file)
        return {
            "message": "PG Recommends API is running.",
            "status": "healthy",
            "docs": "/docs",
        }

    return app


def run_server():
    """CLI runner for FastAPI server."""
    uvicorn.run(
        "phase_6_capstone.server:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=True,
    )
