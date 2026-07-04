"""Unit tests for the end-to-end RAG pipeline coordinator."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from langchain_core.documents import Document


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag.pipeline import RAGPipeline, _document_ids


class FakeLoader:
    def __init__(self, pages: list[Document]) -> None:
        self.pages = pages
        self.calls = 0

    def load_all_documents(self) -> list[Document]:
        self.calls += 1
        return self.pages


class FakeChunker:
    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks

    def split_documents(self, _pages: list[Document]) -> list[Document]:
        return self.chunks


class FakeCollection:
    name = "test_collection"

    def __init__(self, ids: list[str] | None = None) -> None:
        self.ids = list(ids or [])
        self.deleted: list[str] = []

    def count(self) -> int:
        return len(self.ids)

    def get(self, include=None):
        return {"ids": list(self.ids)}

    def delete(self, *, ids: list[str]) -> None:
        self.deleted.extend(ids)


class FakeRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def retrieve(self, _query: str, *, top_k: int) -> list[Document]:
        return self.documents[:top_k]


class FakeGenerator:
    def generate(self, _query: str, _documents: list[Document]) -> str:
        return "Check blood pressure. [Evidence 1]"


class RAGPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.page = Document(page_content="Full page", metadata={"source": "anc.pdf", "page": 1})
        self.chunk = Document(
            page_content="Check blood pressure.",
            metadata={"source": "anc.pdf", "page": 1, "category": "WHO"},
        )

    @patch("rag.pipeline._save_to_collection")
    def test_build_index_stores_chunks_and_removes_stale_records(self, save_mock) -> None:
        current_ids = _document_ids([self.chunk])
        save_mock.return_value = current_ids
        collection = FakeCollection(["stale-id"])
        pipeline = RAGPipeline(
            loader=FakeLoader([self.page]),
            chunker=FakeChunker([self.chunk]),
            vector_store=collection,
        )

        result = pipeline.build_index()

        self.assertEqual(result.stored_chunks, 1)
        self.assertEqual(result.removed_stale_chunks, 1)
        self.assertEqual(collection.deleted, ["stale-id"])
        save_mock.assert_called_once_with([self.chunk], collection, ids=current_ids)

    def test_query_returns_answer_sources_and_metadata(self) -> None:
        pipeline = RAGPipeline(
            vector_store=FakeCollection(["existing"]),
            retriever=FakeRetriever([self.chunk]),
            generator=FakeGenerator(),
        )

        response = pipeline.query("What should be checked?")

        self.assertEqual(response["answer"], "Check blood pressure. [Evidence 1]")
        self.assertEqual(response["retrieved_sources"][0]["content"], self.chunk.page_content)
        self.assertEqual(response["retrieved_sources"][0]["metadata"]["category"], "WHO")
        self.assertEqual(response["metadata"]["retrieved_count"], 1)

    def test_run_does_not_reindex_a_nonempty_collection(self) -> None:
        loader = FakeLoader([self.page])
        pipeline = RAGPipeline(
            loader=loader,
            vector_store=FakeCollection(["existing"]),
            retriever=FakeRetriever([self.chunk]),
            generator=FakeGenerator(),
        )

        pipeline.run("What should be checked?")

        self.assertEqual(loader.calls, 0)

    def test_document_ids_are_stable(self) -> None:
        self.assertEqual(_document_ids([self.chunk]), _document_ids([self.chunk]))

    def test_run_rejects_empty_query_before_indexing(self) -> None:
        loader = FakeLoader([self.page])
        pipeline = RAGPipeline(
            loader=loader,
            vector_store=FakeCollection(),
            retriever=FakeRetriever([]),
            generator=FakeGenerator(),
        )

        with self.assertRaises(ValueError):
            pipeline.run("   ")

        self.assertEqual(loader.calls, 0)


if __name__ == "__main__":
    unittest.main()
