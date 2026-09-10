"""Configuration settings and provider-agnostic model factory for Phase 4 Agents."""

from __future__ import annotations

import os
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

        # If user asks about weather, simulate calling get_weather
        if any(w in content_str for w in ("weather", "temperature")) and not any(
            msg.type == "tool" for msg in messages
        ):
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

        # If user asks math or after weather, simulate calling calculator
        if any(w in content_str for w in ("+", "*", "multiply", "add", "calculate")):
            if not any(msg.type == "tool" for msg in messages):
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

        # Final answer
        ai_msg = AIMessage(
            content=f"Mock Model response processed for query: {last_message.content}"
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
