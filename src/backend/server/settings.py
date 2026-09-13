from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
# In the repo: <root>/src/backend. In the Docker image the backend is /app with no repo around it,
# so fall back to BACKEND_DIR (compose sets DATA_DIR / ARTIFACTS_DIR explicitly there).
REPO_ROOT = BACKEND_DIR.parents[1] if len(BACKEND_DIR.parents) >= 2 else BACKEND_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: SecretStr | None = None
    default_arm: str = "play-gemini"
    google_rpm: int = 10
    model_timeout_s: float = 120.0
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str | None = None

    checkpoint_path: Path = BACKEND_DIR / "var" / "checkpoints.db"
    app_db_path: Path = BACKEND_DIR / "var" / "app.db"
    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    config_dir: Path = BACKEND_DIR / "config"
    experiments_dir: Path = BACKEND_DIR / "experiments"

    runlog_enabled: bool = True
    git_sha: str | None = None
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
