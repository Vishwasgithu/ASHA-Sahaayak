"""Unit tests for the Chroma retrieval service."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.documents import Document


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag.retriever import ChromaRetriever


class ChromaRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = SimpleNamespace(name="test_collection")

    @patch("rag.retriever._search_collection")
    def test_retrieve_returns_documents_with_metadata(self, search_mock) -> None:
        expected = [
            Document(
                page_content="Antenatal care guidance",
                metadata={"source": "who.pdf", "page": 2},
            )
        ]
        search_mock.return_value = expected
        retriever = ChromaRetriever(vector_store=self.collection)

        result = retriever.retrieve("  antenatal care  ")

        self.assertEqual(result, expected)
        search_mock.assert_called_once_with(
            "antenatal care",
            k=5,
            vector_store=self.collection,
            where=None,
        )

    @patch("rag.retriever._search_collection", return_value=[])
    def test_custom_limit_and_metadata_filter(self, search_mock) -> None:
        retriever = ChromaRetriever(vector_store=self.collection)

        retriever.similarity_search(
            "safe motherhood",
            k=3,
            where={"category": "NHM_India"},
        )

        search_mock.assert_called_once_with(
            "safe motherhood",
            k=3,
            vector_store=self.collection,
            where={"category": "NHM_India"},
        )

    def test_empty_query_is_rejected(self) -> None:
        retriever = ChromaRetriever(vector_store=self.collection)

        with self.assertRaises(ValueError):
            retriever.retrieve("   ")


if __name__ == "__main__":
    unittest.main()
