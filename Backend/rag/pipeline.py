"""End-to-end ingestion and query pipeline for ASHA Sahaayak RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypedDict

from langchain_core.documents import Document

import config
from rag.chunker import DocumentChunker
from rag.generator import (
    ClinicalResponseGenerator,
    INSUFFICIENT_EVIDENCE_RESPONSE,
)
from rag.loader import PDFDocumentLoader
from rag.retriever import ChromaRetriever
from utils.helpers import get_logger

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


logger = get_logger("RAG.Pipeline")
_DELETE_BATCH_SIZE = 1_000


class SourceResult(TypedDict):
    """One retrieved evidence chunk returned to an API caller."""

    content: str
    metadata: dict[str, Any]


class PipelineResponse(TypedDict):
    """JSON-serializable response from the online RAG pipeline."""

    answer: str
    retrieved_sources: list[SourceResult]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Summary of a completed knowledge-base indexing operation."""

    loaded_pages: int
    generated_chunks: int
    stored_chunks: int
    removed_stale_chunks: int
    collection_name: str

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class PipelineIndexingError(RuntimeError):
    """Raised when the knowledge base cannot be indexed safely."""


class PipelineQueryError(RuntimeError):
    """Raised when retrieval or response generation cannot be completed."""


def _create_collection(
    *,
    persist_directory: str | Path | None,
    collection_name: str,
) -> Collection:
    from rag.vector_store import create_vector_store

    return create_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def _save_to_collection(
    documents: Sequence[Document],
    collection: Collection,
    *,
    ids: Sequence[str],
) -> list[str]:
    from rag.vector_store import save_documents

    return save_documents(documents, collection, ids=ids)


def _document_ids(documents: Sequence[Document]) -> list[str]:
    """Build stable content-derived IDs for idempotent Chroma upserts."""
    occurrences: dict[str, int] = {}
    record_ids: list[str] = []

    for document in documents:
        source = str(
            document.metadata.get("source")
            or document.metadata.get("file_path")
            or document.metadata.get("document_name")
            or "unknown"
        )
        page = str(document.metadata.get("page", ""))
        identity = "\x1f".join((source, page, document.page_content.strip()))
        digest = sha256(identity.encode("utf-8")).hexdigest()
        occurrence = occurrences.get(digest, 0)
        occurrences[digest] = occurrence + 1
        suffix = f":{occurrence}" if occurrence else ""
        record_ids.append(f"rag:{digest}{suffix}")

    return record_ids


def _existing_ids(collection: Collection) -> set[str]:
    """Read collection IDs without loading document bodies when supported."""
    try:
        result = collection.get(include=[])
    except (TypeError, ValueError):
        result = collection.get()
    return {str(record_id) for record_id in (result.get("ids") or [])}


def _delete_ids(collection: Collection, ids: Iterable[str]) -> int:
    stale_ids = sorted(set(ids))
    for start in range(0, len(stale_ids), _DELETE_BATCH_SIZE):
        collection.delete(ids=stale_ids[start : start + _DELETE_BATCH_SIZE])
    return len(stale_ids)


class RAGPipeline:
    """Coordinate PDF ingestion, Chroma retrieval, and grounded generation.

    Dependencies may be injected for application lifecycle management and
    testing. Default components and the persistent collection are initialized
    lazily.
    """

    def __init__(
        self,
        *,
        loader: PDFDocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
        vector_store: Collection | None = None,
        retriever: ChromaRetriever | None = None,
        generator: ClinicalResponseGenerator | None = None,
        top_k: int = config.TOP_K_RESULTS,
        persist_directory: str | Path | None = None,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
    ) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")

        self._loader = loader
        self._chunker = chunker
        self._vector_store = vector_store
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._lock = RLock()

    @property
    def loader(self) -> PDFDocumentLoader:
        if self._loader is None:
            self._loader = PDFDocumentLoader()
        return self._loader

    @property
    def chunker(self) -> DocumentChunker:
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker

    @property
    def vector_store(self) -> Collection:
        if self._vector_store is None:
            self._vector_store = _create_collection(
                persist_directory=self._persist_directory,
                collection_name=self._collection_name,
            )
        return self._vector_store

    @property
    def retriever(self) -> ChromaRetriever:
        if self._retriever is None:
            self._retriever = ChromaRetriever(
                top_k=self._top_k,
                vector_store=self.vector_store,
            )
        return self._retriever

    @property
    def generator(self) -> ClinicalResponseGenerator:
        if self._generator is None:
            self._generator = ClinicalResponseGenerator()
        return self._generator

    def build_index(self, *, remove_stale: bool = True) -> IndexingResult:
        """Load, chunk, embed, and synchronize all knowledge-base PDFs.

        Stale records are removed only after every current chunk has been
        embedded and upserted successfully, preserving the previous index if
        generation fails midway.
        """
        with self._lock:
            try:
                pages = self.loader.load_all_documents()
                if not pages:
                    logger.warning("No PDF pages were loaded; the existing index was unchanged.")
                    return IndexingResult(0, 0, 0, 0, self.vector_store.name)

                chunks = [
                    document
                    for document in self.chunker.split_documents(pages)
                    if document.page_content.strip()
                ]
                if not chunks:
                    logger.warning("No non-empty chunks were generated; the existing index was unchanged.")
                    return IndexingResult(len(pages), 0, 0, 0, self.vector_store.name)

                collection = self.vector_store
                old_ids = _existing_ids(collection) if remove_stale else set()
                record_ids = _document_ids(chunks)
                stored_ids = _save_to_collection(chunks, collection, ids=record_ids)
                if stored_ids != record_ids:
                    raise PipelineIndexingError(
                        "Vector storage returned IDs that do not match the indexed chunks."
                    )

                removed_count = (
                    _delete_ids(collection, old_ids.difference(record_ids))
                    if remove_stale
                    else 0
                )
                logger.info(
                    "Indexed %d pages into %d chunks; removed %d stale chunks.",
                    len(pages),
                    len(chunks),
                    removed_count,
                )
                return IndexingResult(
                    loaded_pages=len(pages),
                    generated_chunks=len(chunks),
                    stored_chunks=len(stored_ids),
                    removed_stale_chunks=removed_count,
                    collection_name=collection.name,
                )
            except PipelineIndexingError:
                raise
            except Exception as exc:
                logger.exception("Knowledge-base indexing failed.")
                raise PipelineIndexingError("Failed to build the RAG index.") from exc

    def query(self, user_query: str) -> PipelineResponse:
        """Retrieve evidence and generate a grounded maternal-health answer."""
        if not isinstance(user_query, str):
            raise TypeError("user_query must be a string.")
        normalized_query = user_query.strip()
        if not normalized_query:
            raise ValueError("user_query must not be empty or whitespace-only.")

        with self._lock:
            try:
                documents = self.retriever.retrieve(normalized_query, top_k=self._top_k)
                answer = self.generator.generate(normalized_query, documents)
                sources: list[SourceResult] = [
                    {
                        "content": document.page_content,
                        "metadata": dict(document.metadata),
                    }
                    for document in documents
                ]
                return {
                    "answer": answer,
                    "retrieved_sources": sources,
                    "metadata": {
                        "retrieved_count": len(sources),
                        "top_k": self._top_k,
                        "collection_name": self._collection_name,
                        "insufficient_evidence": (
                            answer == INSUFFICIENT_EVIDENCE_RESPONSE
                        ),
                    },
                }
            except Exception as exc:
                logger.exception("RAG query processing failed.")
                raise PipelineQueryError("Failed to process the RAG query.") from exc

    def run(
        self,
        user_query: str,
        *,
        rebuild_index: bool = False,
        remove_stale: bool = True,
    ) -> PipelineResponse:
        """Run the complete pipeline, indexing when requested or when empty."""
        if not isinstance(user_query, str):
            raise TypeError("user_query must be a string.")
        if not user_query.strip():
            raise ValueError("user_query must not be empty or whitespace-only.")

        with self._lock:
            if rebuild_index or self.vector_store.count() == 0:
                self.build_index(remove_stale=remove_stale)
            return self.query(user_query)


def run_pipeline(
    user_query: str,
    *,
    rebuild_index: bool = False,
) -> PipelineResponse:
    """Execute the default end-to-end pipeline with one function call."""
    return RAGPipeline().run(user_query, rebuild_index=rebuild_index)


__all__ = [
    "IndexingResult",
    "PipelineIndexingError",
    "PipelineQueryError",
    "PipelineResponse",
    "RAGPipeline",
    "SourceResult",
    "run_pipeline",
]
