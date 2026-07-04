"""Unit tests for evidence-grounded clinical generation."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from langchain_core.documents import Document


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag.generator import (
    ClinicalResponseGenerator,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    build_clinical_prompt,
)


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


class ClinicalResponseGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(
                page_content="Measure blood pressure during antenatal visits.",
                metadata={"document_name": "ANC.pdf", "page": 4},
            )
        ]

    def test_empty_evidence_returns_fallback_without_calling_llm(self) -> None:
        llm = FakeLLM("should not be used")

        answer = ClinicalResponseGenerator(llm).generate("What should be checked?", [])

        self.assertEqual(answer, INSUFFICIENT_EVIDENCE_RESPONSE)
        self.assertEqual(llm.prompts, [])

    def test_valid_cited_response_is_returned(self) -> None:
        expected = "Measure blood pressure during antenatal visits. [Evidence 1]"
        llm = FakeLLM(
            json.dumps(
                {
                    "has_sufficient_evidence": True,
                    "answer": expected,
                    "citations": [1],
                }
            )
        )

        answer = ClinicalResponseGenerator(llm).generate(
            "What should be checked?",
            self.documents,
        )

        self.assertEqual(answer, expected)

    def test_uncited_response_is_rejected(self) -> None:
        llm = FakeLLM(
            json.dumps(
                {
                    "has_sufficient_evidence": True,
                    "answer": "Take an unspecified medicine.",
                    "citations": [],
                }
            )
        )

        answer = ClinicalResponseGenerator(llm).generate("What should I take?", self.documents)

        self.assertEqual(answer, INSUFFICIENT_EVIDENCE_RESPONSE)

    def test_prompt_contains_query_evidence_and_safety_rules(self) -> None:
        prompt = build_clinical_prompt("What should be checked?", self.documents)

        self.assertIn("What should be checked?", prompt)
        self.assertIn("Measure blood pressure", prompt)
        self.assertIn("Answer only", prompt)
        self.assertIn(INSUFFICIENT_EVIDENCE_RESPONSE, prompt)


if __name__ == "__main__":
    unittest.main()
