"""Hugging Face embedding utilities and an in-memory vector store.

This module deliberately keeps vectors in process memory.  It provides the
embedding and similarity-search boundary needed by the RAG pipeline without
coupling the application to ChromaDB (or any other vector database).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from threading import RLock
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import config
from utils.helpers import get_logger


logger = get_logger("RAG.Embeddings")


def _validate_texts(texts: Iterable[str]) -> list[str]:
    """Materialize and validate text input before model inference."""
    materialized = list(texts)
    if any(not isinstance(text, str) for text in materialized):
        raise TypeError("Every value to embed must be a string.")
    if any(not text.strip() for text in materialized):
        raise ValueError("Empty or whitespace-only text cannot be embedded.")
    return materialized


def _default_device() -> str:
    """Return the configured device, defaulting to CPU for portability."""
    return os.getenv("EMBEDDING_DEVICE", "cpu")


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = config.EMBEDDING_MODEL,
    device: str | None = None,
) -> HuggingFaceEmbeddings:
    """Create and cache a reusable Hugging Face embedding model.

    Model weights are loaded lazily on the first call. Embeddings are
    normalized so cosine similarity can be calculated efficiently with a dot
    product while remaining compatible with future vector-store integrations.
    """
    selected_device = device or _default_device()
    logger.info(
        "Loading Hugging Face embedding model '%s' on %s",
        model_name,
        selected_device,
    )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": selected_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def generate_embeddings(
    texts: Iterable[str],
    *,
    model: HuggingFaceEmbeddings | None = None,
) -> list[list[float]]:
    """Generate one normalized embedding for each supplied text."""
    validated_texts = _validate_texts(texts)
    if not validated_texts:
        return []
    embedding_model = model or get_embedding_model()
    return embedding_model.embed_documents(validated_texts)


def generate_query_embedding(
    query: str,
    *,
    model: HuggingFaceEmbeddings | None = None,
) -> list[float]:
    """Generate a normalized embedding for a retrieval query."""
    validated_query = _validate_texts([query])[0]
    embedding_model = model or get_embedding_model()
    return embedding_model.embed_query(validated_query)


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    """A snapshot of one document and its in-memory embedding."""

    id: str
    document: Document
    embedding: tuple[float, ...]


class InMemoryEmbeddingStore:
    """Thread-safe, process-local storage for documents and embeddings.

    The store is intentionally ephemeral: all data is discarded when the
    process exits. It is suitable for development and for preparing the RAG
    pipeline before a persistent vector database is introduced.
    """

    def __init__(self, model: HuggingFaceEmbeddings | None = None) -> None:
        self._model = model
        self._ids: list[str] = []
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None
        self._lock = RLock()

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Return the model, loading the shared default model lazily."""
        if self._model is None:
            self._model = get_embedding_model()
        return self._model

    def __len__(self) -> int:
        with self._lock:
            return len(self._documents)

    @property
    def dimension(self) -> int | None:
        """Return the vector dimension, or ``None`` while the store is empty."""
        with self._lock:
            if self._embeddings is None:
                return None
            return int(self._embeddings.shape[1])

    def add_documents(
        self,
        documents: Iterable[Document],
        *,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Embed documents and retain them in memory.

        Returns the generated or caller-supplied identifiers in input order.
        """
        materialized = list(documents)
        if not materialized:
            return []
        if any(not isinstance(document, Document) for document in materialized):
            raise TypeError("Every item must be a langchain Document.")

        texts = _validate_texts(document.page_content for document in materialized)
        record_ids = list(ids) if ids is not None else [str(uuid4()) for _ in texts]
        if len(record_ids) != len(materialized):
            raise ValueError("The number of ids must match the number of documents.")
        if any(not isinstance(record_id, str) or not record_id.strip() for record_id in record_ids):
            raise ValueError("Every document id must be a non-empty string.")
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Document ids must be unique.")

        vectors = np.asarray(
            generate_embeddings(texts, model=self.model),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(materialized):
            raise RuntimeError("The embedding model returned an invalid vector matrix.")

        # Copy documents so later caller mutations cannot silently alter the
        # indexed content or metadata.
        stored_documents = [
            Document(page_content=document.page_content, metadata=dict(document.metadata))
            for document in materialized
        ]

        with self._lock:
            duplicates = set(record_ids).intersection(self._ids)
            if duplicates:
                duplicate_list = ", ".join(sorted(duplicates))
                raise ValueError(f"Document ids already exist: {duplicate_list}")
            if self._embeddings is not None and vectors.shape[1] != self._embeddings.shape[1]:
                raise ValueError("Embedding dimension does not match existing vectors.")

            self._ids.extend(record_ids)
            self._documents.extend(stored_documents)
            self._embeddings = (
                vectors.copy()
                if self._embeddings is None
                else np.concatenate((self._embeddings, vectors), axis=0)
            )

        logger.info("Stored %d embeddings in memory (total: %d)", len(vectors), len(self))
        return record_ids

    def add_texts(
        self,
        texts: Iterable[str],
        *,
        metadatas: Sequence[Mapping[str, object]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Create documents from text and add their embeddings to the store."""
        validated_texts = _validate_texts(texts)
        if not validated_texts:
            return []
        metadata_values = (
            list(metadatas)
            if metadatas is not None
            else [{} for _ in validated_texts]
        )
        if len(metadata_values) != len(validated_texts):
            raise ValueError("The number of metadata entries must match the texts.")

        documents = [
            Document(page_content=text, metadata=dict(metadata))
            for text, metadata in zip(validated_texts, metadata_values)
        ]
        return self.add_documents(documents, ids=ids)

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = config.TOP_K_RESULTS,
    ) -> list[tuple[Document, float]]:
        """Return the most similar documents and cosine scores."""
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        # Avoid loading the model just to query an empty store.
        with self._lock:
            if self._embeddings is None:
                return []

        query_vector = np.asarray(
            generate_query_embedding(query, model=self.model),
            dtype=np.float32,
        )
        if query_vector.ndim != 1:
            raise RuntimeError("The embedding model returned an invalid query vector.")

        with self._lock:
            # The store may have been cleared while inference was running.
            if self._embeddings is None:
                return []
            if query_vector.shape[0] != self._embeddings.shape[1]:
                raise ValueError("Query embedding dimension does not match stored vectors.")

            scores = self._embeddings @ query_vector
            limit = min(k, len(self._documents))
            best_indices = np.argsort(scores)[::-1][:limit]
            return [
                (
                    Document(
                        page_content=self._documents[index].page_content,
                        metadata=dict(self._documents[index].metadata),
                    ),
                    float(scores[index]),
                )
                for index in best_indices
            ]

    def similarity_search(
        self,
        query: str,
        *,
        k: int = config.TOP_K_RESULTS,
    ) -> list[Document]:
        """Return the most similar documents without their scores."""
        return [
            document
            for document, _score in self.similarity_search_with_score(query, k=k)
        ]

    def snapshot(self) -> list[StoredEmbedding]:
        """Return a defensive snapshot of all records currently in memory."""
        with self._lock:
            if self._embeddings is None:
                return []
            return [
                StoredEmbedding(
                    id=record_id,
                    document=Document(
                        page_content=document.page_content,
                        metadata=dict(document.metadata),
                    ),
                    embedding=tuple(float(value) for value in vector),
                )
                for record_id, document, vector in zip(
                    self._ids,
                    self._documents,
                    self._embeddings,
                )
            ]

    def clear(self) -> None:
        """Remove all documents and vectors from memory."""
        with self._lock:
            self._ids.clear()
            self._documents.clear()
            self._embeddings = None


@lru_cache(maxsize=1)
def get_default_embedding_store() -> InMemoryEmbeddingStore:
    """Return the process-wide in-memory embedding store."""
    return InMemoryEmbeddingStore()


def store_documents_in_memory(
    documents: Iterable[Document],
    *,
    store: InMemoryEmbeddingStore | None = None,
) -> InMemoryEmbeddingStore:
    """Embed documents into a supplied store or the process-wide store."""
    target_store = store if store is not None else get_default_embedding_store()
    target_store.add_documents(documents)
    return target_store


__all__ = [
    "InMemoryEmbeddingStore",
    "StoredEmbedding",
    "generate_embeddings",
    "generate_query_embedding",
    "get_default_embedding_store",
    "get_embedding_model",
    "store_documents_in_memory",
]
