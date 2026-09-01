from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(
        default="Phase 0 Baseline API",
        description="Name of the application",
    )
    app_env: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Application environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the service",
    )
    port: int = Field(
        default=8000,
        description="Port to bind the service",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key for authenticated operations",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
