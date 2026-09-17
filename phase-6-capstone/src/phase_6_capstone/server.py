"""FastAPI Streaming Server and API Gateway for PG Recommends."""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select

from phase_6_capstone.agent import CapstoneAgent
from phase_6_capstone.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from phase_6_capstone.cache import SemanticQueryCache
from phase_6_capstone.config import settings
from phase_6_capstone.db import DatabaseManager, HumanEscalationModel, UserModel
from phase_6_capstone.ingestion import (
    fetch_live_letterboxd_feed,
    parse_reviews_csv,
    sync_movies_to_store,
)
from phase_6_capstone.models import MovieRecord
from phase_6_capstone.retrieval import HybridMovieRetriever
from phase_6_capstone.taste_engine import (
    TasteProfileManager,
    ingest_user_letterboxd_feed,
)

logger = logging.getLogger(__name__)


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


class RegisterRequest(BaseModel):
    email: str
    password: str
    letterboxd_handle: str | None = None
    tenant_id: str = "default_tenant"


class LoginRequest(BaseModel):
    email: str
    password: str
    tenant_id: str = "default_tenant"


class SyncLetterboxdRequest(BaseModel):
    username: str
    tenant_id: str = "default_tenant"


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
    query_cache = SemanticQueryCache(db=db_manager)
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

    @app.post("/api/auth/register")
    async def register(req: RegisterRequest):
        email_clean = req.email.strip().lower()
        if not email_clean or "@" not in email_clean:
            raise HTTPException(status_code=400, detail="Invalid email address.")
        if len(req.password) < 6:
            raise HTTPException(
                status_code=400, detail="Password must be at least 6 characters."
            )

        user_id = f"user_{uuid4().hex[:12]}"
        hashed = hash_password(req.password)
        raw_handle = req.letterboxd_handle
        handle_clean = raw_handle.strip().lstrip("@") if raw_handle else None

        async with db_manager.session_factory() as session:
            stmt = select(UserModel).where(
                UserModel.email == email_clean,
                UserModel.tenant_id == req.tenant_id,
            )
            res = await session.execute(stmt)
            if res.scalar_one_or_none():
                raise HTTPException(
                    status_code=400, detail="An account with this email already exists."
                )

            new_user = UserModel(
                id=user_id,
                tenant_id=req.tenant_id,
                email=email_clean,
                password_hash=hashed,
                letterboxd_handle=handle_clean,
                taste_match_pct=72.0,
            )
            session.add(new_user)
            await session.commit()

        taste_match = 72.0
        if handle_clean:
            try:
                _, taste_match = await ingest_user_letterboxd_feed(
                    username=handle_clean,
                    db=db_manager,
                    user_id=user_id,
                    tenant_id=req.tenant_id,
                )
            except Exception as e:
                logger.warning("Could not sync letterboxd on register: %s", e)

        token = create_access_token(
            user_id=user_id,
            email=email_clean,
            tenant_id=req.tenant_id,
            letterboxd_handle=handle_clean,
        )

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "email": email_clean,
                "letterboxd_handle": handle_clean,
                "taste_match_pct": taste_match,
            },
        }

    @app.post("/api/auth/login")
    async def login(req: LoginRequest):
        email_clean = req.email.strip().lower()
        async with db_manager.session_factory() as session:
            stmt = select(UserModel).where(
                UserModel.email == email_clean,
                UserModel.tenant_id == req.tenant_id,
            )
            res = await session.execute(stmt)
            user = res.scalar_one_or_none()
            if not user or not verify_password(req.password, user.password_hash):
                raise HTTPException(
                    status_code=401, detail="Invalid email or password."
                )

            token = create_access_token(
                user_id=user.id,
                email=user.email,
                tenant_id=user.tenant_id,
                letterboxd_handle=user.letterboxd_handle,
            )
            return {
                "access_token": token,
                "token_type": "bearer",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "letterboxd_handle": user.letterboxd_handle,
                    "taste_match_pct": user.taste_match_pct or 72.0,
                },
            }

    @app.get("/api/auth/me")
    async def get_me(request: Request):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return {"authenticated": False, "user": None}
        token = auth_header[7:].strip()
        payload = decode_access_token(token)
        if not payload or "sub" not in payload:
            return {"authenticated": False, "user": None}

        user_id = payload["sub"]
        async with db_manager.session_factory() as session:
            stmt = select(UserModel).where(UserModel.id == user_id)
            res = await session.execute(stmt)
            user = res.scalar_one_or_none()
            if not user:
                return {"authenticated": False, "user": None}

            return {
                "authenticated": True,
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "letterboxd_handle": user.letterboxd_handle,
                    "taste_match_pct": user.taste_match_pct or 72.0,
                },
            }

    @app.post("/api/user/sync-letterboxd")
    async def sync_user_letterboxd(req: SyncLetterboxdRequest, request: Request):
        user_id = "guest_user"
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            payload = decode_access_token(token)
            if payload and "sub" in payload:
                user_id = payload["sub"]

        handle = req.username.strip().lstrip("@")
        try:
            profile, match_score = await ingest_user_letterboxd_feed(
                username=handle,
                db=db_manager,
                user_id=user_id,
                tenant_id=req.tenant_id,
            )
            return {
                "status": "success",
                "username": handle,
                "taste_match_pct": match_score,
                "taste_profile": profile.model_dump(),
            }
        except Exception as e:
            logger.error("Failed to sync Letterboxd for %s: %s", handle, e)
            raise HTTPException(
                status_code=500, detail=f"Failed to sync Letterboxd feed: {e}"
            )

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatStreamRequest, request: Request):
        resolved_user_id = req.user_id
        resolved_tenant_id = req.tenant_id
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            payload = decode_access_token(token)
            if payload and "sub" in payload:
                resolved_user_id = payload["sub"]
                resolved_tenant_id = payload.get("tenant_id", req.tenant_id)

        # 1. Fast sub-10ms Semantic Query Cache Check
        cached_entry = await query_cache.get(req.message)
        if cached_entry:

            async def cached_event_generator() -> AsyncGenerator[str, None]:
                yield f"data: {json.dumps({'event': 'cache_hit', 'data': True})}\n\n"
                citations = cached_entry.get("citations", [])
                if citations:
                    cit_payload = json.dumps({"event": "citations", "data": citations})
                    yield f"data: {cit_payload}\n\n"
                critic_citations = cached_entry.get("critic_citations", [])
                if critic_citations:
                    cc_payload = json.dumps(
                        {"event": "critic_citations", "data": critic_citations}
                    )
                    yield f"data: {cc_payload}\n\n"

                cached_text = cached_entry.get("response", "")
                words = cached_text.split(" ")
                for word in words:
                    tok_payload = json.dumps({"event": "token", "data": word + " "})
                    yield f"data: {tok_payload}\n\n"
                yield f"data: {json.dumps({'event': 'done', 'data': '[DONE]'})}\n\n"

            return StreamingResponse(
                cached_event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # 2. Cache Miss: Execute agent pipeline and populate cache
        async def event_generator() -> AsyncGenerator[str, None]:
            accumulated_tokens: list[str] = []
            captured_citations: list[dict] = []
            captured_critic_citations: list[dict] = []

            try:
                async for chunk in curator_agent.stream_turn(
                    tenant_id=resolved_tenant_id,
                    user_id=resolved_user_id,
                    conversation_id=req.conversation_id,
                    message=req.message,
                ):
                    event_type = chunk.get("event")
                    if event_type == "token":
                        accumulated_tokens.append(chunk.get("data", ""))
                    elif event_type == "citations":
                        captured_citations = chunk.get("data", [])
                    elif event_type == "critic_citations":
                        captured_critic_citations = chunk.get("data", [])

                    payload = json.dumps(chunk)
                    yield f"data: {payload}\n\n"

                # Persist to cache
                full_response = "".join(accumulated_tokens).strip()
                if full_response:
                    await query_cache.set(
                        query=req.message,
                        response=full_response,
                        citations=captured_citations,
                        critic_citations=captured_critic_citations,
                    )

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
