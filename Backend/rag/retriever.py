"""Reusable retrieval service for the ASHA Sahaayak RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from langchain_core.documents import Document

import config
from utils.helpers import get_logger

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


logger = get_logger("RAG.Retriever")


def _load_collection(
    *,
    persist_directory: str | Path | None,
    collection_name: str,
) -> Collection:
    """Load Chroma lazily so importing this module has no database side effects."""
    from rag.vector_store import load_vector_store

    return load_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def _search_collection(
    query: str,
    *,
    k: int,
    vector_store: Collection,
    where: Mapping[str, Any] | None,
) -> list[Document]:
    """Delegate similarity search to the vector-store boundary."""
    from rag.vector_store import similarity_search

    return similarity_search(
        query,
        k=k,
        vector_store=vector_store,
        where=where,
    )


class ChromaRetriever:
    """Retrieve semantically similar chunks from a ChromaDB collection.

    A collection can be injected by callers that already manage its lifecycle.
    Otherwise, the configured persistent collection is loaded lazily on the
    first query.
    """

    def __init__(
        self,
        *,
        top_k: int = config.TOP_K_RESULTS,
        vector_store: Collection | None = None,
        persist_directory: str | Path | None = None,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
    ) -> None:
        self._top_k = self._validate_top_k(top_k)
        self._vector_store = vector_store
        self._persist_directory = persist_directory
        self._collection_name = collection_name

        if vector_store is None and not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        return top_k

    @property
    def top_k(self) -> int:
        """Return the default maximum number of chunks per query."""
        return self._top_k

    @property
    def vector_store(self) -> Collection:
        """Return the collection, loading the configured store when needed."""
        if self._vector_store is None:
            self._vector_store = _load_collection(
                persist_directory=self._persist_directory,
                collection_name=self._collection_name,
            )
        return self._vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """Return the most relevant chunks, including their metadata.

        Results are ordered from most to least similar. The vector-store layer
        includes ``_chroma_id`` and ``_chroma_distance`` in each document's
        metadata in addition to the original source metadata.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty or whitespace-only.")

        limit = self._top_k if top_k is None else self._validate_top_k(top_k)
        matches = _search_collection(
            normalized_query,
            k=limit,
            vector_store=self.vector_store,
            where=where,
        )
        logger.info(
            "Retrieved %d chunks for query from collection '%s'",
            len(matches),
            self.vector_store.name,
        )
        return matches

    def similarity_search(
        self,
        query: str,
        *,
        k: int | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """LangChain-style alias for :meth:`retrieve`."""
        return self.retrieve(query, top_k=k, where=where)


def retrieve_documents(
    query: str,
    *,
    top_k: int = config.TOP_K_RESULTS,
    vector_store: Collection | None = None,
    where: Mapping[str, Any] | None = None,
) -> list[Document]:
    """Retrieve relevant chunks without explicitly creating a retriever."""
    return ChromaRetriever(top_k=top_k, vector_store=vector_store).retrieve(
        query,
        where=where,
    )


__all__ = ["ChromaRetriever", "retrieve_documents"]
