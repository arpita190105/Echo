"""
Knowledge base ingestion: chunks study notes/textbook text files and stores
them as embeddings in a local Chroma vector DB (free, self-hosted, no API).

Usage:
    1. Drop .txt files (notes/textbook content) into data/notes/
    2. Run: python -m rag.knowledge_base
    3. This builds/updates the persistent vector DB in data/chroma_db/
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from config import config
from logger import get_logger

logger = get_logger(__name__)

NOTES_DIR = "data/notes"
CHUNK_SIZE = 300      # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Simple word-count based chunking with overlap, to preserve context across chunk boundaries."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_knowledge_base():
    """Reads all .txt files in data/notes/, chunks them, and stores in Chroma."""
    os.makedirs(NOTES_DIR, exist_ok=True)
    os.makedirs(config.rag.persist_dir, exist_ok=True)

    txt_files = glob.glob(os.path.join(NOTES_DIR, "*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {NOTES_DIR}. Add study notes there first.")
        print(f"No .txt files found in {NOTES_DIR}. Add some notes/textbook text files and re-run.")
        return

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.rag.embedding_model
    )

    client = chromadb.PersistentClient(path=config.rag.persist_dir)
    collection = client.get_or_create_collection(
        name=config.rag.collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},  # explicit — Chroma defaults to L2 otherwise,
                                             # which breaks the distance_threshold filtering
                                             # in retriever.py
    )

    total_chunks = 0
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = _chunk_text(text)
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
        logger.info(f"Ingested {filename}: {len(chunks)} chunks")

    print(f"Knowledge base built: {total_chunks} chunks from {len(txt_files)} file(s).")
    logger.info(f"Knowledge base build complete: {total_chunks} total chunks")


if __name__ == "__main__":
    build_knowledge_base()