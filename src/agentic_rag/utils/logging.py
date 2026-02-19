from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml

from agentic_rag.settings import get_settings


def setup_logging(config_path: Optional[str] = None) -> None:
    """
    Load logging configuration from YAML and apply it using logging.config.dictConfig.

    - If config_path is None, uses configs/logging.yaml at project root.
    - Falls back to basicConfig if YAML file is missing.
    """
    settings = get_settings()

    # Project root 
    root = Path(__file__).resolve().parents[3]
    path = Path(config_path) if config_path else (root / "configs" / "logging.yaml")

    if not path.exists():
        logging.basicConfig(level=settings.app.log_level)
        logging.getLogger("agentic_rag").warning(
            "Logging config not found at %s. Falling back to basicConfig.", path
        )
        return

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # safer than yaml.load for untrusted input

    # Apply config dict (official logging configuration mechanism)
    logging.config.dictConfig(config)


def get_logger(name: str = "agentic_rag") -> logging.Logger:
    return logging.getLogger(name)
