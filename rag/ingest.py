"""
rag/ingest.py — PDF ingestion pipeline
Loads PDFs from rag/documents/, chunks them, embeds with sentence-transformers,
and stores in ChromaDB. Run once (or re-run to refresh the index).

Usage:
    python -m rag.ingest
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config.settings import settings

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400       # words per chunk
CHUNK_OVERLAP = 80     # words overlap between consecutive chunks
COLLECTION_NAME = "musa_guidelines"


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def _doc_id(source: str, idx: int) -> str:
    h = hashlib.md5(source.encode()).hexdigest()[:8]
    return f"{h}_{idx}"


def build_index(force: bool = False) -> chromadb.Collection:
    docs_dir = Path(settings.rag_docs_dir)
    db_dir = Path(settings.chroma_db_dir)

    client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection for rebuild.")
        except Exception:
            pass

    # Skip if already indexed and not forced
    try:
        col = client.get_collection(COLLECTION_NAME)
        if col.count() > 0 and not force:
            logger.info(f"Index already exists ({col.count()} chunks). Skipping ingestion.")
            return col
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = SentenceTransformer(settings.embedding_model)
    pdfs = list(docs_dir.glob("*.pdf"))
    logger.info(f"Ingesting {len(pdfs)} PDFs from {docs_dir}")

    all_chunks, all_ids, all_meta = [], [], []

    for pdf_path in pdfs:
        logger.info(f"  Processing: {pdf_path.name}")
        text = _load_pdf(pdf_path)
        chunks = _chunk_text(text)
        logger.info(f"    → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(_doc_id(pdf_path.name, i))
            all_meta.append({"source": pdf_path.name, "chunk_idx": i})

    logger.info(f"Embedding {len(all_chunks)} total chunks...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, batch_size=32).tolist()

    # Upsert in batches of 500
    batch = 500
    for i in range(0, len(all_chunks), batch):
        col.upsert(
            ids=all_ids[i : i + batch],
            documents=all_chunks[i : i + batch],
            embeddings=embeddings[i : i + batch],
            metadatas=all_meta[i : i + batch],
        )

    logger.success(f"Index built: {col.count()} chunks stored in {db_dir}")
    return col


if __name__ == "__main__":
    build_index(force=False)
