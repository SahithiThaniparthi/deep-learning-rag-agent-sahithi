"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """Load a Markdown file and create chunks with metadata."""
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be nonnegative and less than chunk_size."
            )

        if file_path.suffix.lower() != ".md":
            raise ValueError(
                "This workshop supports Markdown (.md) files only."
            )

        metadata = self._infer_metadata(
            file_path,
            metadata_overrides,
        )

        raw_chunks = self._chunk_markdown(
            file_path,
            chunk_size,
            chunk_overlap,
        )

        chunks = []

        for raw_chunk in raw_chunks:
            text = raw_chunk["text"].strip()

            if not text:
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=VectorStoreManager.generate_chunk_id(
                        metadata.source,
                        text,
                    ),
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata.to_dict()),
                )
            )

        logger.info(
            "Created {} chunks from '{}'",
            len(chunks),
            file_path.name,
        )

        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        # TODO: implement — iterate and collect, handle per-file errors
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        # TODO: implement using langchain_community.document_loaders.PyPDFLoader
        # and langchain.text_splitter.RecursiveCharacterTextSplitter
        raise NotImplementedError

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """Split Markdown by headings, then split oversized sections."""
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        text = file_path.read_text(encoding="utf-8-sig")

        if not text.strip():
            return []

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,
        )

        sections = header_splitter.split_text(text)

        character_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        documents = character_splitter.split_documents(sections)

        raw_chunks = []

        for document in documents:
            header = " > ".join(
                document.metadata[key]
                for key in ("h1", "h2", "h3", "h4", "h5", "h6")
                if key in document.metadata
            )

            raw_chunks.append(
                {
                    "text": document.page_content,
                    "header": header,
                }
            )

        return raw_chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """Infer topic and difficulty from the filename."""
        topic_map = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "som": "SOM",
            "boltzmannmachine": "BoltzmannMachine",
            "gan": "GAN",
        }

        difficulty_levels = {
            "beginner",
            "intermediate",
            "advanced",
        }

        stem = file_path.stem.lower()
        topic_name, separator, difficulty_name = stem.rpartition("_")

        if not separator or difficulty_name not in difficulty_levels:
            topic_name = stem
            difficulty_name = "beginner"

        topic_key = (
            topic_name.replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        metadata_values = {
            "topic": topic_map.get(topic_key, "ANN"),
            "difficulty": difficulty_name,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
        }

        if overrides:
            metadata_values.update(overrides)

        if not overrides or "is_bonus" not in overrides:
            metadata_values["is_bonus"] = metadata_values["topic"] in {
                "SOM",
                "BoltzmannMachine",
                "GAN",
            }

        return ChunkMetadata(**metadata_values)
