"""Rule-based symptom extraction and maternal risk classification.

This module identifies risk signals only. Clinical recommendations are left to
the evidence-grounded RAG pipeline.
"""

from __future__ import annotations

import re
from typing import Final


_SYMPTOM_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "fever": ("fever", "bukhar"),
    "headache": ("headache", "head ache", "sar dard"),
    "dizziness": ("dizziness", "dizzy", "chakkar"),
    "swelling": ("swelling", "sujan"),
    "vomiting": ("vomiting", "vomit", "ulti"),
    "bleeding": ("bleeding", "khoon"),
    "weakness": ("weakness", "weak", "kamzori"),
    "diarrhoea": ("diarrhoea", "diarrhea", "dast"),
    "blurred vision": ("blurred vision", "blurry vision", "dhundhla dikhna"),
    "severe abdominal pain": (
        "severe abdominal pain",
        "severe stomach pain",
        "tez pet dard",
    ),
    "reduced fetal movement": (
        "reduced fetal movement",
        "decreased fetal movement",
        "less fetal movement",
        "baby moving less",
        "bacche ki harkat kam",
    ),
}

_NEGATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:negative\s+for|no|not|without|denies|nahi|nahin)\b",
    re.IGNORECASE,
)
_POST_NEGATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:(?:hai|he)\s+)?(?:nahi|nahin)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[.!?;:\n]|\b(?:but|however|although|yet|lekin|par)\b",
    re.IGNORECASE,
)
_POSITIVE_CUE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:has|have|having|with|reports?|reported|complains?\s+of|positive\s+for)\b",
    re.IGNORECASE,
)
_PREGNANCY_MONTH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(\d{1,2})\s*(?:months?|mahina|mahine)\b",
    re.IGNORECASE,
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = [re.escape(word) for word in phrase.split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE)


_ALIAS_PATTERNS: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    symptom: tuple(_phrase_pattern(alias) for alias in aliases)
    for symptom, aliases in _SYMPTOM_ALIASES.items()
}


def _current_clause_prefix(text: str, mention_start: int) -> str:
    prefix = text[:mention_start]
    boundaries = list(_CLAUSE_BOUNDARY_PATTERN.finditer(prefix))
    return prefix[boundaries[-1].end() :] if boundaries else prefix


def _current_clause_suffix(text: str, mention_end: int) -> str:
    suffix = text[mention_end:]
    boundary = _CLAUSE_BOUNDARY_PATTERN.search(suffix)
    return suffix[: boundary.start()] if boundary else suffix


def _is_negated(text: str, mention_start: int, mention_end: int) -> bool:
    """Return whether one symptom mention falls within a negated clause."""
    prefix = _current_clause_prefix(text, mention_start)
    negations = list(_NEGATION_PATTERN.finditer(prefix))
    if negations:
        last_negation = negations[-1]
        text_after_negation = prefix[last_negation.end() :]
        # A later positive assertion starts a new semantic scope even when the
        # sentence has no punctuation, e.g. "no fever and reports vomiting".
        if not _POSITIVE_CUE_PATTERN.search(text_after_negation):
            return True

    # Hindi commonly places negation after the symptom: "bukhar nahi hai".
    return _POST_NEGATION_PATTERN.search(
        _current_clause_suffix(text, mention_end)
    ) is not None


def _detect_symptoms(text: str) -> list[str]:
    detected: list[str] = []
    for symptom, patterns in _ALIAS_PATTERNS.items():
        mentions = [match for pattern in patterns for match in pattern.finditer(text)]
        if any(not _is_negated(text, match.start(), match.end()) for match in mentions):
            detected.append(symptom)
    return detected


def _classify_risk(symptoms: set[str]) -> str:
    has_preeclampsia_signals = {
        "headache",
        "swelling",
        "blurred vision",
    }.issubset(symptoms)
    has_other_high_risk_signal = bool(
        symptoms.intersection(
            {"bleeding", "severe abdominal pain", "reduced fetal movement"}
        )
    )

    if has_preeclampsia_signals or has_other_high_risk_signal:
        return "HIGH RISK"
    if symptoms.intersection({"fever", "vomiting", "weakness"}):
        return "MEDIUM RISK"
    return "LOW RISK"


def _natural_language_list(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} and {values[-1]}"


def _build_rag_query(
    risk_level: str,
    symptoms: list[str],
    pregnancy_month: str | None,
) -> str:
    risk_label = risk_level.lower().replace(" ", "-")
    month_context = f" in month {pregnancy_month}" if pregnancy_month else ""
    symptom_context = (
        f" with {_natural_language_list(symptoms)}" if symptoms else ""
    )
    return (
        f"{risk_label.capitalize()} pregnancy{month_context}{symptom_context} "
        "according to WHO maternal healthcare guidelines."
    )


def process_healthcare_input(text: str) -> dict[str, object]:
    """Extract non-negated symptoms and prepare a maternal-health RAG query."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    symptoms = _detect_symptoms(text)
    pregnancy_match = _PREGNANCY_MONTH_PATTERN.search(text)
    pregnancy_month = pregnancy_match.group(1) if pregnancy_match else None
    risk_level = _classify_risk(set(symptoms))

    return {
        "input_text": text,
        "symptoms": symptoms,
        "pregnancy_month": pregnancy_month,
        "risk_level": risk_level,
        "recommended_rag_query": _build_rag_query(
            risk_level,
            symptoms,
            pregnancy_month,
        ),
    }


__all__ = ["process_healthcare_input"]
