"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        try:
            Path(self._settings.chroma_db_path).mkdir(
                parents=True,
                exist_ok=True,
            )

            self._client = chromadb.PersistentClient(
                path=self._settings.chroma_db_path,
            )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )

            logger.info(
                "Initialised collection '{}' with {} chunks",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )

        except Exception as exc:
            raise RuntimeError(
                "Could not initialise ChromaDB at "
                f"{self._settings.chroma_db_path}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(
            ids=[chunk_id],
            include=[],
        )
        return chunk_id in result["ids"]

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """Embed and store new chunks, skipping duplicate IDs."""
        result = IngestionResult()
        pending = []
        seen_ids = set()

        for chunk in chunks:
            try:
                if (
                    chunk.chunk_id in seen_ids
                    or self.check_duplicate(chunk.chunk_id)
                ):
                    result.skipped += 1
                    continue

                seen_ids.add(chunk.chunk_id)
                pending.append(chunk)

            except Exception as exc:
                result.errors.append(
                    f"Chunk {chunk.chunk_id}: {exc}"
                )

        batch_size = 100

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]

            try:
                embeddings = self._embeddings.embed_documents(
                    [chunk.chunk_text for chunk in batch]
                )

                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings,
                    documents=[chunk.chunk_text for chunk in batch],
                    metadatas=[
                        chunk.metadata.to_dict() for chunk in batch
                    ],
                )

                result.ingested += len(batch)

                for chunk in batch:
                    source = chunk.metadata.source
                    if source not in result.document_ids:
                        result.document_ids.append(source)

            except Exception as exc:
                for chunk in batch:
                    result.errors.append(
                        f"Chunk {chunk.chunk_id}: {exc}"
                    )

                logger.error("Ingestion batch failed: {}", exc)

        logger.info(
            "Ingestion complete: {} added, {} skipped, {} errors",
            result.ingested,
            result.skipped,
            len(result.errors),
        )

        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using cosine similarity."""
        if not query_text.strip():
            return []

        count = self._collection.count()
        if count == 0:
            return []

        requested_k = self._settings.retrieval_k if k is None else k
        if requested_k <= 0:
            return []

        filters = []

        if topic_filter:
            filters.append({"topic": topic_filter})

        if difficulty_filter:
            filters.append({"difficulty": difficulty_filter})

        where_filter = None

        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        query_embedding = self._embeddings.embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(requested_k, count),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []

        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if text is None or metadata is None or distance is None:
                continue

            score = 1.0 - float(distance)

            if score < self._settings.similarity_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )

        return sorted(
            retrieved,
            key=lambda chunk: chunk.score,
            reverse=True,
        )

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """List source documents and their chunk counts."""
        results = self._collection.get(include=["metadatas"])
        documents = {}

        for metadata in results["metadatas"] or []:
            if metadata is None:
                continue

            source = metadata["source"]

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": metadata["topic"],
                    "chunk_count": 0,
                }

            documents[source]["chunk_count"] += 1

        return sorted(
            documents.values(),
            key=lambda document: document["source"],
        )

    def get_document_chunks(
        self,
        source: str,
    ) -> list[DocumentChunk]:
        """Retrieve all stored chunks for a source document."""
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        chunks = []

        for chunk_id, text, metadata in zip(
            results["ids"],
            results["documents"] or [],
            results["metadatas"] or [],
        ):
            if text is None or metadata is None:
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata),
                )
            )

        return chunks

    def get_collection_stats(self) -> dict:
        """Return chunk count, topics, sources, and bonus status."""
        results = self._collection.get(include=["metadatas"])

        topics = set()
        sources = set()
        bonus_topics_present = False

        for metadata in results["metadatas"] or []:
            if metadata is None:
                continue

            topics.add(metadata["topic"])
            sources.add(metadata["source"])

            if str(metadata.get("is_bonus", "false")).lower() == "true":
                bonus_topics_present = True

        return {
            "total_chunks": self._collection.count(),
            "topics": sorted(topics),
            "sources": sorted(sources),
            "bonus_topics_present": bonus_topics_present,
        }

    def delete_document(self, source: str) -> int:
        """Delete all chunks for a source and return their count."""
        results = self._collection.get(
            where={"source": source},
            include=[],
        )

        chunk_ids = results["ids"]

        if not chunk_ids:
            return 0

        self._collection.delete(ids=chunk_ids)

        logger.info(
            "Deleted {} chunks from '{}'",
            len(chunk_ids),
            source,
        )

        return len(chunk_ids)
