from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from agentic_rag.settings import get_settings
from agentic_rag.retrieval.chunking import ChunkingConfig, split_documents
from agentic_rag.retrieval.embedding import get_embeddings
from agentic_rag.retrieval.vectorstore import get_vectorstore


def load_pdfs(pdf_paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()  # typically one Document per page
        # enrich metadata
        for d in pdf_docs:
            d.metadata = d.metadata or {}
            d.metadata["source_file"] = pdf_path.name
        docs.extend(pdf_docs)
    return docs


def ingest_books() -> None:
    settings = get_settings()
    books_dir = Path(settings.paths.data_dir) / "raw" / "books"
    books_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(books_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {books_dir}. Put your books there as *.pdf"
        )

    print(f"[INGEST] Found {len(pdf_paths)} PDF(s). Loading...")
    docs = load_pdfs(pdf_paths)
    print(f"[INGEST] Loaded {len(docs)} page-documents. Chunking...")

    chunk_cfg = ChunkingConfig(chunk_size=1000, chunk_overlap=150)
    chunks = split_documents(docs, cfg=chunk_cfg)
    print(f"[INGEST] Produced {len(chunks)} chunks. Embedding + storing...")

    embeddings = get_embeddings()
    vs = get_vectorstore(embedding=embeddings)

    # Add chunks to Chroma (persisted)
    vs.add_documents(chunks)

    print(f"[INGEST] Done. Stored chunks in: {settings.paths.vectorstore_dir}")


if __name__ == "__main__":
    ingest_books()