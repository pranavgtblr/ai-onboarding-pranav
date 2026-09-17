"""Database manager and SQLAlchemy models for multi-tenant isolation."""

from collections.abc import AsyncGenerator
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TenantModel(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default_tenant", index=True
    )
    email: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    letterboxd_handle: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    taste_match_pct: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UserTasteProfileModel(Base):
    __tablename__ = "user_taste_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    liked_directors: Mapped[str] = mapped_column(Text, default="[]")
    liked_genres: Mapped[str] = mapped_column(Text, default="[]")
    disliked_elements: Mapped[str] = mapped_column(Text, default="[]")
    mood_tags: Mapped[str] = mapped_column(Text, default="[]")
    taste_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class HumanEscalationModel(Base):
    __tablename__ = "human_escalations"

    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(256))
    transcript_summary: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SemanticCacheModel(Base):
    __tablename__ = "semantic_query_cache"

    query_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_text: Mapped[str] = mapped_column(Text)
    cached_response: Mapped[str] = mapped_column(Text)
    cached_citations: Mapped[str] = mapped_column(Text, default="[]")
    cached_critic_citations: Mapped[str] = mapped_column(Text, default="[]")
    hit_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class DatabaseManager:
    """Manages async database engine, sessions, and multi-tenant queries."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: AsyncEngine = create_async_engine(
            self.database_url,
            echo=False,
            future=True,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_models(self) -> None:
        """Initializes database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provides an async database session context."""
        async with self.session_factory() as session:
            yield session
