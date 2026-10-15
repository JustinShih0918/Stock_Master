"""Runtime settings from environment variables and the untracked .env file.

Secrets (API keys, database password) are read here and nowhere else. Never log a Settings object
directly; use ``Settings.redacted()``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore",
        # compose passes unset keys as empty strings; let .env values win over those
        env_ignore_empty=True,
    )

    database_url: str = "postgresql://stock_master:stock_master_dev@db:5432/stock_master"

    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = ""

    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = ""

    finmind_api_token: SecretStr | None = None

    # Comma-separated provider:model chain, tried in order before the deterministic fallback.
    # Example: "nvidia:nvidia/nemotron-3-ultra-550b-a55b,nvidia:z-ai/glm-5.3,gemini:gemini-2.5-pro"
    llm_chain: str = ""

    config_dir: Path = PROJECT_ROOT / "config"
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    reference_dir: Path = PROJECT_ROOT / "reference"

    alert_webhook_url: SecretStr | None = Field(default=None)

    def redacted(self) -> dict:
        out = self.model_dump()
        for key, value in out.items():
            if isinstance(getattr(self, key), SecretStr) and value is not None:
                out[key] = "***"
        if "@" in self.database_url:
            scheme, rest = self.database_url.split("://", 1)
            out["database_url"] = f"{scheme}://***@{rest.split('@', 1)[1]}"
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
