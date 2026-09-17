"""Configuration settings for PG Recommends."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application runtime settings."""

    app_name: str = "PG Recommends"
    app_env: str = "production"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = "sqlite+aiosqlite:///pg_recommends.db"
    model_provider: str = "google_genai"
    model_name: str = "gemini-3.5-flash-lite"
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    tmdb_api_key: str | None = None
    letterboxd_rss_url: str = "https://letterboxd.com/pranavg/rss/"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def effective_api_key(self) -> str | None:
        """Returns active Google/Gemini API key from settings or env."""
        import os

        return (
            self.gemini_api_key
            or self.google_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )


settings = Settings()
