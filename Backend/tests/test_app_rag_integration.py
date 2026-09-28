"""Tests for Flask healthcare analysis and RAG response integration."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

with patch("whisper.load_model", return_value=object()):
    import app as flask_app


class SuccessfulPipeline:
    def run(self, query: str):
        self.query = query
        return {
            "answer": "Seek assessment for bleeding. [Evidence 1]",
            "retrieved_sources": [
                {
                    "content": "Bleeding requires assessment.",
                    "metadata": {"source": "WHO.pdf", "page": 2},
                }
            ],
            "metadata": {"retrieved_count": 1},
        }


class FailingPipeline:
    def run(self, _query: str):
        raise RuntimeError("database unavailable")


class FlaskRAGIntegrationTests(unittest.TestCase):
    def test_analysis_uses_recommended_rag_query(self) -> None:
        pipeline = SuccessfulPipeline()

        with patch.object(flask_app, "rag_pipeline", pipeline):
            result = flask_app.analyze_healthcare_text("Bleeding at 7 months")

        self.assertIn("High-risk pregnancy", pipeline.query)
        self.assertIn("bleeding", pipeline.query)
        self.assertEqual(result["risk_level"], "HIGH RISK")
        self.assertEqual(
            result["clinical_recommendation"],
            "Seek assessment for bleeding. [Evidence 1]",
        )
        self.assertEqual(result["sources"][0]["metadata"]["source"], "WHO.pdf")
        self.assertFalse(result["metadata"]["fallback_used"])

    def test_rag_failure_returns_legacy_fallback(self) -> None:
        with patch.object(flask_app, "rag_pipeline", FailingPipeline()):
            result = flask_app.analyze_healthcare_text("No fever but bleeding")

        self.assertIn("Immediate medical attention", result["clinical_recommendation"])
        self.assertEqual(result["sources"], [])
        self.assertFalse(result["metadata"]["rag_available"])
        self.assertTrue(result["metadata"]["fallback_used"])

    def test_analyze_endpoint_returns_requested_contract(self) -> None:
        with patch.object(flask_app, "rag_pipeline", SuccessfulPipeline()):
            response = flask_app.app.test_client().post(
                "/api/analyze",
                json={"text": "Bleeding"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.get_json()),
            {
                "risk_level",
                "symptoms",
                "pregnancy_month",
                "clinical_recommendation",
                "sources",
                "language",
                "metadata",
            },
        )


if __name__ == "__main__":
    unittest.main()
