"""Tests for symptom negation and maternal risk classification."""

from pathlib import Path
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from healthcare_engine import process_healthcare_input


class HealthcareEngineTests(unittest.TestCase):
    def test_supported_english_negations_exclude_symptoms(self) -> None:
        cases = (
            ("No fever", "fever"),
            ("Not bleeding", "bleeding"),
            ("Without dizziness", "dizziness"),
            ("Patient denies vomiting.", "vomiting"),
            ("Patient is negative for weakness.", "weakness"),
        )
        for text, symptom in cases:
            with self.subTest(text=text):
                self.assertNotIn(symptom, process_healthcare_input(text)["symptoms"])

    def test_hindi_negation_before_and_after_symptom(self) -> None:
        self.assertNotIn(
            "fever",
            process_healthcare_input("Nahi bukhar")["symptoms"],
        )
        self.assertNotIn(
            "fever",
            process_healthcare_input("Bukhar nahin hai")["symptoms"],
        )

    def test_positive_statement_after_negated_clause_is_detected(self) -> None:
        result = process_healthcare_input("No fever, but patient reports vomiting.")
        self.assertNotIn("fever", result["symptoms"])
        self.assertIn("vomiting", result["symptoms"])

    def test_high_risk_rules(self) -> None:
        cases = (
            "Headache with swelling and blurred vision",
            "Bleeding",
            "Severe abdominal pain",
            "Reduced fetal movement",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    process_healthcare_input(text)["risk_level"],
                    "HIGH RISK",
                )

    def test_medium_and_low_risk_rules(self) -> None:
        for symptom in ("fever", "vomiting", "weakness"):
            with self.subTest(symptom=symptom):
                self.assertEqual(
                    process_healthcare_input(symptom)["risk_level"],
                    "MEDIUM RISK",
                )
        self.assertEqual(
            process_healthcare_input("Routine antenatal visit")["risk_level"],
            "LOW RISK",
        )

    def test_output_contains_rag_query_instead_of_guidance(self) -> None:
        result = process_healthcare_input(
            "Seven month patient has headache, swelling and blurred vision"
        )
        self.assertNotIn("guidance", result)
        self.assertIn("recommended_rag_query", result)
        self.assertIn("High-risk pregnancy", result["recommended_rag_query"])
        self.assertIn("headache, swelling and blurred vision", result["recommended_rag_query"])


if __name__ == "__main__":
    unittest.main()
