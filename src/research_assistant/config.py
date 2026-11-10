"""Settings loaded from .env (and real environment variables, which win)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr  # SecretStr: prints as '**********', never the real key
    model: str
    timeout_s: float = 30.0
    max_output_tokens: int = 2000  # includes REASONING tokens; 1000 truncated ask (D3 M5)
    runs_dir: Path = Path("runs")
    max_attempts: int = 3  # total tries, including the first
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    database_url: str = "postgresql://localhost/research_assistant"
    embedding_model: str = "text-embedding-3-small"  # EMBEDDING_MODEL in .env wins
    embedding_dims: int = 1536  # must match the model; checked on every embed call


@lru_cache
def get_settings() -> Settings:
    """Load settings once; tests can call get_settings.cache_clear()."""
    return Settings()
