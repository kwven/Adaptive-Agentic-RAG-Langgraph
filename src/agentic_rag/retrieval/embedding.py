from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.settings import get_settings


@dataclass(frozen=True)
class EmbeddingConfig:
    # Default local embedding model
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # keep CPU for your machine
    normalize_embeddings: bool = True  # helps cosine similarity stability
    cache_folder: Optional[str] = None  # None = default HF cache (~/.cache)


def get_embeddings(cfg: Optional[EmbeddingConfig] = None) -> Embeddings:
    """
    Returns a LangChain Embeddings object backed by sentence-transformers (local).
    """
    settings = get_settings()
    cfg = cfg or EmbeddingConfig()

    # If you want the model name controlled by settings.toml later, you can map it there.
    model_name = cfg.model_name

    encode_kwargs = {"normalize_embeddings": cfg.normalize_embeddings}
    model_kwargs = {"device": cfg.device}

    kwargs = dict(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    if cfg.cache_folder:
        kwargs["cache_folder"] = cfg.cache_folder

    return HuggingFaceEmbeddings(**kwargs)