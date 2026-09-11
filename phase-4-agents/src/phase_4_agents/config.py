"""Configuration settings and provider-agnostic model factory for Phase 4 Agents."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime application settings loaded from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider Selection: "google_genai" | "openai" | "mock"
    model_provider: str = "google_genai"
    model_name: str = "gemini-3.5-flash-lite"

    # API Keys
    google_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str | None = None

    # Global options
    temperature: float = 0.0
    request_timeout_seconds: float = 30.0

    # Persistence / Checkpointer Configuration (Task 4.6)
    postgres_uri: str = (
        "postgresql://postgres:postgres@localhost:5433/langgraph?sslmode=disable"
    )

    @property
    def effective_google_api_key(self) -> str:
        """Return the available Google API key."""
        return (
            self.google_api_key
            or self.gemini_api_key
            or os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", "")
        )


class MockToolChatModel(BaseChatModel):
    """Deterministic offline mock model for provider-agnostic testing."""

    model_name: str = "mock-agent-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_message = messages[-1]
        content_str = str(last_message.content).lower()

        # If the last message is a ToolMessage, synthesize the final answer
        if messages[-1].type == "tool":
            tool_outputs = [str(msg.content) for msg in messages if msg.type == "tool"]
            combined = " ".join(tool_outputs)
            ai_msg = AIMessage(
                content=(
                    "Mock Model synthesized answer based on retrieved tool "
                    f"results: {combined[:150]}..."
                )
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # Extract target query (handling self-correction rewrite attempts)
        target_query = str(last_message.content)
        if "[self-correction attempt" in content_str:
            m = re.search(r"rewritten query:\s*['\"]([^'\"]+)['\"]", target_query)
            if m:
                target_query = m.group(1)
                content_str = target_query.lower()

        # 1. Check for Greetings / Chit-Chat (direct parametric response, no tools)
        greetings = ("hello", "hi", "hey", "good morning", "how are you", "who are you")
        if any(w in content_str for w in greetings):
            clean_input = str(last_message.content)[:40]
            ai_msg = AIMessage(
                content=f"Hello! I am ready to help. (Direct: {clean_input})"
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # 2. Check for PDF / Engineering Specs Search
        pdf_keywords = (
            "pdf",
            "eclss",
            "mars",
            "pressure",
            "propulsion",
            "manual",
            "spec",
        )
        if any(w in content_str for w in pdf_keywords):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "pdf_search",
                        "args": {"query": target_query},
                        "id": "mock_call_pdf",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # 3. Check for Website Documentation Search
        site_keywords = (
            "site",
            "website",
            "toobler",
            "capability",
            "capabilities",
            "company",
            "portal",
        )
        if any(w in content_str for w in site_keywords):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "site_search",
                        "args": {"query": target_query},
                        "id": "mock_call_site",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # 4. Check for Database Queries (Customers, Orders, Appointments)
        db_keywords = (
            "customer",
            "order",
            "appointment",
            "doctor",
            "sql",
            "database",
            "stock",
            "table",
            "atlantis",
        )
        if any(w in content_str for w in db_keywords):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "db_query",
                        "args": {"query": target_query},
                        "id": "mock_call_db",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # 5. Check for Web Search (Live info, latest, breaking news, 2026)
        web_keywords = (
            "news",
            "latest",
            "recent",
            "2026",
            "web",
            "internet",
            "artemis",
        )
        if any(w in content_str for w in web_keywords):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": target_query},
                        "id": "mock_call_web",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # 5. Check for Weather
        if any(w in content_str for w in ("weather", "temperature")):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "Tokyo"},
                        "id": "mock_call_tokyo",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # 6. Check for Math / Calculator
        math_keywords = ("+", "*", "multiply", "add", "calculate", "math", "divide")
        if any(w in content_str for w in math_keywords):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"operation": "multiply", "a": 25, "b": 4},
                        "id": "mock_call_calc",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # 7. Check for general search or non-existent query (for retries)
        search_keywords = (
            "search",
            "information",
            "query",
            "find",
            "lookup",
            "unmatchable",
        )
        if any(w in content_str for w in search_keywords):
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": target_query},
                        "id": "mock_call_search",
                        "type": "tool_call",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        # 8. Conversational / Direct Answer (no tools required)
        ai_msg = AIMessage(
            content=f"Direct parametric response for: {last_message.content}"
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> MockToolChatModel:
        """Bind tools to the mock model (no-op for offline testing)."""
        return self

    @property
    def _llm_type(self) -> str:
        return "mock_chat_model"


def get_chat_model(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Initialize a chat model from any provider using configuration only.

    Supports 'google_genai', 'openai' (and OpenAI-compatible endpoints), or 'mock'.
    """
    cfg = settings or get_settings()
    active_provider = (provider or cfg.model_provider).strip().lower()
    active_model = model or cfg.model_name
    active_temp = temperature if temperature is not None else cfg.temperature

    if active_provider == "mock":
        return MockToolChatModel(model_name=active_model)

    if active_provider in ("google_genai", "google", "gemini"):
        key = api_key or cfg.effective_google_api_key
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY is required for "
                "google_genai provider."
            )

        return init_chat_model(
            active_model,
            model_provider="google_genai",
            api_key=key,
            temperature=active_temp,
        )

    if active_provider in ("openai", "azure_openai"):
        key = api_key or cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        kwargs: dict[str, Any] = {"api_key": key or "mock-key-for-local"}
        if cfg.openai_base_url:
            kwargs["base_url"] = cfg.openai_base_url
        return init_chat_model(
            active_model,
            model_provider="openai",
            temperature=active_temp,
            **kwargs,
        )

    # Generic fallback to any other provider supported by init_chat_model
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    return init_chat_model(
        active_model,
        model_provider=active_provider,
        temperature=active_temp,
        **kwargs,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()


def get_postgres_checkpointer(
    uri: str | None = None,
    *,
    auto_setup: bool = True,
):
    """Context manager yielding an initialized PostgresSaver instance.

    Usage:
        with get_postgres_checkpointer() as checkpointer:
            graph = build_state_graph_agent(checkpointer=checkpointer)
            res = graph.invoke(..., config={"configurable": {"thread_id": "..."}})
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    cfg = get_settings()
    target_uri = uri or cfg.postgres_uri

    class _CheckpointerContext:
        def __init__(self, conn_uri: str) -> None:
            self.conn_uri = conn_uri
            self._ctx = PostgresSaver.from_conn_string(self.conn_uri)
            self._saver: PostgresSaver | None = None

        def __enter__(self) -> PostgresSaver:
            self._saver = self._ctx.__enter__()
            if auto_setup:
                self._saver.setup()
            return self._saver

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)

    return _CheckpointerContext(target_uri)
