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

    def test_duplicate_citation_is_accepted(self) -> None:
        # Models often cite the same evidence more than once; the deduped
        # citation set must not be compared by length to the raw list.
        expected = "Check BP [Evidence 1]. Refer if high [Evidence 1]."
        llm = FakeLLM(
            json.dumps(
                {
                    "has_sufficient_evidence": True,
                    "answer": expected,
                    "citations": [1, 1],
                }
            )
        )

        answer = ClinicalResponseGenerator(llm).generate("What now?", self.documents)

        self.assertEqual(answer, expected)

    def test_citations_without_inline_markers_is_accepted(self) -> None:
        # The citations array is a valid grounding signal even when the model
        # does not mirror it with inline [Evidence N] markers.
        expected = "Measure blood pressure and refer to PHC."
        llm = FakeLLM(
            json.dumps(
                {
                    "has_sufficient_evidence": True,
                    "answer": expected,
                    "citations": [1],
                }
            )
        )

        answer = ClinicalResponseGenerator(llm).generate("What now?", self.documents)

        self.assertEqual(answer, expected)

    def test_inline_markers_without_citations_array_is_accepted(self) -> None:
        expected = "Measure blood pressure [Evidence 1]."
        llm = FakeLLM(
            json.dumps(
                {
                    "has_sufficient_evidence": True,
                    "answer": expected,
                    "citations": [],
                }
            )
        )

        answer = ClinicalResponseGenerator(llm).generate("What now?", self.documents)

        self.assertEqual(answer, expected)

    def test_missing_sufficient_flag_with_marker_is_accepted(self) -> None:
        # Local Ollama/Qwen models frequently omit has_sufficient_evidence;
        # a valid inline marker must still be accepted.
        expected = "Refer to PHC [Evidence 1]."
        llm = FakeLLM(
            json.dumps({"answer": expected, "citations": [1]})
        )

        answer = ClinicalResponseGenerator(llm).generate("What now?", self.documents)

        self.assertEqual(answer, expected)

    def test_fabricated_out_of_range_citation_is_rejected(self) -> None:
        # The one hard failure we must keep: a citation to evidence that was
        # never retrieved (hallucinated source).
        llm = FakeLLM(
            json.dumps(
                {
                    "has_sufficient_evidence": True,
                    "answer": "Take drug X [Evidence 9].",
                    "citations": [9],
                }
            )
        )

        answer = ClinicalResponseGenerator(llm).generate("What now?", self.documents)

        self.assertEqual(answer, INSUFFICIENT_EVIDENCE_RESPONSE)

    def test_multiple_valid_citations_across_chunks_accepted(self) -> None:
        docs = [
            Document(page_content="Measure blood pressure.", metadata={"page": 1}),
            Document(page_content="Refer to PHC if high.", metadata={"page": 2}),
            Document(page_content="Hydrate and rest.", metadata={"page": 3}),
        ]
        expected = "Check BP [Evidence 1] and refer [Evidence 2]."
        llm = FakeLLM(
            json.dumps(
                {
                    "has_sufficient_evidence": True,
                    "answer": expected,
                    "citations": [1, 2, 3],
                }
            )
        )

        answer = ClinicalResponseGenerator(llm).generate("What now?", docs)

        self.assertEqual(answer, expected)

    def test_prompt_contains_query_evidence_and_safety_rules(self) -> None:
        prompt = build_clinical_prompt("What should be checked?", self.documents)

        self.assertIn("What should be checked?", prompt)
        self.assertIn("Measure blood pressure", prompt)
        self.assertIn("Answer only", prompt)
        self.assertIn(INSUFFICIENT_EVIDENCE_RESPONSE, prompt)


if __name__ == "__main__":
    unittest.main()
