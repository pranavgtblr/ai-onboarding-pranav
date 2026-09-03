from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Phase 2 Raw API loaded from .env and environment."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(
        default="",
        alias="GEMINI_API_KEY",
        description="Google Gemini API Key",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        alias="GEMINI_MODEL",
        description="Gemini Model Identifier",
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        alias="GEMINI_BASE_URL",
        description="Base URL for Google Gemini REST API",
    )
    timeout_seconds: float = Field(
        default=30.0,
        alias="REQUEST_TIMEOUT_SECONDS",
        description="HTTP request timeout in seconds",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
