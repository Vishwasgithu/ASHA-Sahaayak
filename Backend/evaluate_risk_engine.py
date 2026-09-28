import csv
from healthcare_engine import process_healthcare_input

# ============================================================
# ASHA-SAHAAYAK RISK ENGINE EVALUATION
# ============================================================
#
# This script evaluates the implemented multilingual,
# rule-based maternal-health symptom extraction and
# risk-screening engine.
#
# Each scenario contains:
# - Input text
# - Expected symptoms
# - Expected risk level
#
# The script:
# 1. Runs the real healthcare engine
# 2. Collects detected symptoms
# 3. Collects predicted risk
# 4. Compares prediction with expected output
# 5. Calculates accuracy
# 6. Saves detailed results to CSV
#
# ============================================================


TEST_CASES = [
    # ========================================================
    # HIGH-RISK CASES
    # ========================================================
    {
        "id": "HR_01",
        "category": "HIGH",
        "language": "English",
        "text": "I am experiencing bleeding during pregnancy.",
        "expected_symptoms": ["bleeding"],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_02",
        "category": "HIGH",
        "language": "English",
        "text": "I have severe abdominal pain.",
        "expected_symptoms": ["severe abdominal pain"],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_03",
        "category": "HIGH",
        "language": "English",
        "text": "My baby is moving less than usual.",
        "expected_symptoms": ["reduced fetal movement"],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_04",
        "category": "HIGH",
        "language": "English",
        "text": "I have headache, swelling, and blurred vision.",
        "expected_symptoms": [
            "headache",
            "swelling",
            "blurred vision",
        ],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_05",
        "category": "HIGH",
        "language": "English",
        "text": "I have bleeding and weakness.",
        "expected_symptoms": [
            "bleeding",
            "weakness",
        ],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_06",
        "category": "HIGH",
        "language": "English",
        "text": "I have reduced fetal movement and dizziness.",
        "expected_symptoms": [
            "reduced fetal movement",
            "dizziness",
        ],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_07",
        "category": "HIGH",
        "language": "English",
        "text": "I have severe abdominal pain and vomiting.",
        "expected_symptoms": [
            "severe abdominal pain",
            "vomiting",
        ],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_08",
        "category": "HIGH",
        "language": "Hindi",
        "text": "मुझे खून आ रहा है।",
        "expected_symptoms": ["bleeding"],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_09",
        "category": "HIGH",
        "language": "Hindi",
        "text": "मुझे तेज पेट दर्द हो रहा है।",
        "expected_symptoms": ["severe abdominal pain"],
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "HR_10",
        "category": "HIGH",
        "language": "English",
        "text": "I have headache and swelling and my vision is blurred.",
        "expected_symptoms": [
            "headache",
            "swelling",
            "blurred vision",
        ],
        "expected_risk": "HIGH RISK",
    },
    # ========================================================
    # MEDIUM-RISK CASES
    # ========================================================
    {
        "id": "MR_01",
        "category": "MEDIUM",
        "language": "English",
        "text": "I have fever.",
        "expected_symptoms": ["fever"],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_02",
        "category": "MEDIUM",
        "language": "English",
        "text": "I have been vomiting repeatedly.",
        "expected_symptoms": ["vomiting"],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_03",
        "category": "MEDIUM",
        "language": "English",
        "text": "I am feeling very weak.",
        "expected_symptoms": ["weakness"],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_04",
        "category": "MEDIUM",
        "language": "English",
        "text": "I have fever and weakness.",
        "expected_symptoms": [
            "fever",
            "weakness",
        ],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_05",
        "category": "MEDIUM",
        "language": "English",
        "text": "I have vomiting and dizziness.",
        "expected_symptoms": [
            "vomiting",
            "dizziness",
        ],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_06",
        "category": "MEDIUM",
        "language": "English",
        "text": "I have fever and headache.",
        "expected_symptoms": [
            "fever",
            "headache",
        ],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_07",
        "category": "MEDIUM",
        "language": "Hindi",
        "text": "मुझे बुखार है।",
        "expected_symptoms": ["fever"],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_08",
        "category": "MEDIUM",
        "language": "Hindi",
        "text": "मुझे उल्टी हो रही है।",
        "expected_symptoms": ["vomiting"],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_09",
        "category": "MEDIUM",
        "language": "Hindi",
        "text": "मुझे कमजोरी महसूस हो रही है।",
        "expected_symptoms": ["weakness"],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "MR_10",
        "category": "MEDIUM",
        "language": "Romanized Hindi",
        "text": "Mujhe bukhar aur kamzori hai.",
        "expected_symptoms": [
            "fever",
            "weakness",
        ],
        "expected_risk": "MEDIUM RISK",
    },
    # ========================================================
    # LOW-RISK CASES
    # ========================================================
    {
        "id": "LR_01",
        "category": "LOW",
        "language": "English",
        "text": "I have a headache.",
        "expected_symptoms": ["headache"],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "LR_02",
        "category": "LOW",
        "language": "English",
        "text": "I have some swelling.",
        "expected_symptoms": ["swelling"],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "LR_03",
        "category": "LOW",
        "language": "English",
        "text": "I feel dizzy.",
        "expected_symptoms": ["dizziness"],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "LR_04",
        "category": "LOW",
        "language": "English",
        "text": "I have diarrhoea.",
        "expected_symptoms": ["diarrhoea"],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "LR_05",
        "category": "LOW",
        "language": "English",
        "text": "My vision is blurry.",
        "expected_symptoms": ["blurred vision"],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "LR_06",
        "category": "LOW",
        "language": "Hindi",
        "text": "मुझे सिरदर्द है।",
        "expected_symptoms": ["headache"],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "LR_07",
        "category": "LOW",
        "language": "Hindi",
        "text": "मुझे चक्कर आ रहे हैं।",
        "expected_symptoms": ["dizziness"],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "LR_08",
        "category": "LOW",
        "language": "Hindi",
        "text": "मुझे दस्त हो रहे हैं।",
        "expected_symptoms": ["diarrhoea"],
        "expected_risk": "LOW RISK",
    },
    # ========================================================
    # NEGATION CASES
    # ========================================================
    {
        "id": "NEG_01",
        "category": "NEGATION",
        "language": "English",
        "text": "I have no fever.",
        "expected_symptoms": [],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "NEG_02",
        "category": "NEGATION",
        "language": "English",
        "text": "I do not have bleeding.",
        "expected_symptoms": [],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "NEG_03",
        "category": "NEGATION",
        "language": "English",
        "text": "I have no fever but I am vomiting.",
        "expected_symptoms": ["vomiting"],
        "expected_risk": "MEDIUM RISK",
    },
    {
        "id": "NEG_04",
        "category": "NEGATION",
        "language": "Romanized Hindi",
        "text": "Mujhe bukhar nahi hai.",
        "expected_symptoms": [],
        "expected_risk": "LOW RISK",
    },
    {
        "id": "NEG_05",
        "category": "NEGATION",
        "language": "Romanized Hindi",
        "text": "Mujhe khoon nahi aa raha hai lekin kamzori hai.",
        "expected_symptoms": ["weakness"],
        "expected_risk": "MEDIUM RISK",
    },
]


def normalize_symptoms(symptoms):
    """Convert symptom list into a comparable sorted representation."""
    return sorted(symptoms)


def main():

    results = []

    total_cases = len(TEST_CASES)
    correct_risk_predictions = 0
    correct_symptom_predictions = 0

    print("\n" + "=" * 75)
    print("ASHA-SAHAAYAK MULTILINGUAL RISK ENGINE EVALUATION")
    print("=" * 75)

    for case in TEST_CASES:

        result = process_healthcare_input(case["text"])

        predicted_symptoms = normalize_symptoms(result["symptoms"])
        expected_symptoms = normalize_symptoms(case["expected_symptoms"])

        predicted_risk = result["risk_level"]
        expected_risk = case["expected_risk"]

        symptom_correct = predicted_symptoms == expected_symptoms
        risk_correct = predicted_risk == expected_risk

        if symptom_correct:
            correct_symptom_predictions += 1

        if risk_correct:
            correct_risk_predictions += 1

        status = "PASS" if risk_correct and symptom_correct else "FAIL"

        print("\n" + "-" * 75)
        print(f"ID: {case['id']}")
        print(f"Language: {case['language']}")
        print(f"Input: {case['text']}")
        print(f"Expected Symptoms: {expected_symptoms}")
        print(f"Predicted Symptoms: {predicted_symptoms}")
        print(f"Expected Risk: {expected_risk}")
        print(f"Predicted Risk: {predicted_risk}")
        print(f"Status: {status}")

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "language": case["language"],
                "input_text": case["text"],
                "expected_symptoms": ", ".join(expected_symptoms),
                "predicted_symptoms": ", ".join(predicted_symptoms),
                "symptom_correct": symptom_correct,
                "expected_risk": expected_risk,
                "predicted_risk": predicted_risk,
                "risk_correct": risk_correct,
                "status": status,
            }
        )

    risk_accuracy = (correct_risk_predictions / total_cases) * 100

    symptom_accuracy = (correct_symptom_predictions / total_cases) * 100

    print("\n" + "=" * 75)
    print("FINAL RESULTS")
    print("=" * 75)

    print(f"Total Test Cases: {total_cases}")
    print(f"Correct Risk Predictions: " f"{correct_risk_predictions}/{total_cases}")
    print(f"Risk Classification Accuracy: {risk_accuracy:.2f}%")

    print(
        f"Correct Symptom Predictions: " f"{correct_symptom_predictions}/{total_cases}"
    )
    print(f"Exact Symptom Extraction Accuracy: {symptom_accuracy:.2f}%")

    # Save detailed results

    output_file = "risk_engine_evaluation_results.csv"

    with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:

        fieldnames = [
            "id",
            "category",
            "language",
            "input_text",
            "expected_symptoms",
            "predicted_symptoms",
            "symptom_correct",
            "expected_risk",
            "predicted_risk",
            "risk_correct",
            "status",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print("\nDetailed results saved to:")
    print(output_file)

    print("=" * 75)


if __name__ == "__main__":
    main()
