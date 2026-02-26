from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class ChunkingConfig:
    # v0: character-based chunking (simple + reliable)
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Default separators used by RecursiveCharacterTextSplitter:
    # ["\n\n", "\n", " ", ""]
    separators: Optional[List[str]] = None


def build_text_splitter(cfg: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """
    Recursive chunking tries to split on natural boundaries first (paragraphs, lines, spaces),
    then falls back to smaller separators, while respecting size limits and adding overlap.
    """
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    kwargs = dict(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
    )
    if cfg.separators is not None:
        kwargs["separators"] = cfg.separators

    return RecursiveCharacterTextSplitter(**kwargs)


def split_documents(
    documents: Iterable[Document],
    cfg: ChunkingConfig = ChunkingConfig(),
) -> List[Document]:
    """
    Split documents into chunks while preserving metadata.

    Returns a flat list of chunked Documents.
    """
    splitter = build_text_splitter(cfg)
    return splitter.split_documents(list(documents))