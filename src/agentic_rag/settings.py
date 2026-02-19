from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import tomllib  # Python 3.11+
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    env: str = "dev"
    log_level: str = "INFO"


class ModelConfig(BaseModel):
    # We will use Mistral for router/graders/generation
    chat_model: str = "mistral-small-latest"
    # Optional, used later in Phase 2+ if you choose Mistral embeddings
    embed_model: str = "mistral-embed"


class PathsConfig(BaseModel):
    data_dir: str = "./data"
    vectorstore_dir: str = "./data/vectorstore"


class RagConfig(BaseModel):
    top_k: int = 4
    max_loops: int = 3


class WebConfig(BaseModel):
    provider: str = "tavily"
    max_results: int = 5


# ---------- Main Settings ----------
class Settings(BaseSettings):
    """
    Priority (high -> low):
      1) Environment variables (including .env)
      2) configs/settings.toml
      3) defaults in this class
    """

    model_config = SettingsConfigDict(
        env_file=".env",                # load .env 
        env_file_encoding="utf-8",
        extra="ignore",                 # ignore unrelated env vars
    )

    # Secrets / keys (from env)
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # Nested config blocks (from TOML / env overrides)
    app: AppConfig = AppConfig()
    models: ModelConfig = ModelConfig()
    paths: PathsConfig = PathsConfig()
    rag: RagConfig = RagConfig()
    web: WebConfig = WebConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ) -> Tuple[Any, ...]:
        def toml_source() -> Dict[str, Any]:
            return load_toml_settings()
        """
        This lets us add a TOML config file as a settings source.

        Order matters: earlier sources win.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            toml_source,
            file_secret_settings,
        )


def load_toml_settings() -> Dict[str, Any]:
    """
    Load defaults from configs/settings.toml (if exists).
    """
    root = Path(__file__).resolve().parents[2]  # project root if using src-layout
    toml_path = root / "configs" / "settings.toml"
    if not toml_path.exists():
        return {}

    with toml_path.open("rb") as f:
        data = tomllib.load(f)

    # Expect sections like [app], [models], [paths]... already matching field names.
    return data


def get_settings() -> Settings:
    return Settings()
