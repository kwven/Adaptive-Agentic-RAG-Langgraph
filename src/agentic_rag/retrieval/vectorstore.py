from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from agentic_rag.settings import get_settings


DEFAULT_COLLECTION = "agentic_rag"


def _ensure_dir(path: str) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def get_vectorstore(
    embedding: Embeddings,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Create or load a persistent Chroma vector store.

    - If persist_directory is provided, Chroma persists the collection to disk.
    - Uses langchain_chroma.Chroma integration.
    """
    settings = get_settings()
    db_dir = persist_directory or settings.paths.vectorstore_dir
    db_dir = _ensure_dir(db_dir)

    # When persist_directory is set, Chroma works in persistent mode.
    # (No need to call persist() in newer integrations; persistence is handled by Chroma itself.)
    return Chroma(
        collection_name=collection_name,
        persist_directory=db_dir,
        embedding_function=embedding,
    )


def get_retriever(
    embedding: Embeddings,
    top_k: Optional[int] = None,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: Optional[str] = None,
):
    """
    Return a retriever for similarity search over Chroma.
    """
    settings = get_settings()
    k = top_k or settings.rag.top_k
    vs = get_vectorstore(
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return vs.as_retriever(search_kwargs={"k": k})