"""Centralized multilingual language configuration for ASHA Sahaayak."""

from __future__ import annotations

from typing import Final

SUPPORTED_LANGUAGES: Final[dict[str, dict[str, str]]] = {
    "en": {
        "name": "English",
        "native_name": "English",
        "script": "Latin",
        "whisper_code": "en",
    },
    "hi": {
        "name": "Hindi",
        "native_name": "हिन्दी",
        "script": "Devanagari",
        "whisper_code": "hi",
    },
    "mr": {
        "name": "Marathi",
        "native_name": "मराठी",
        "script": "Devanagari",
        "whisper_code": "mr",
    },
    "te": {
        "name": "Telugu",
        "native_name": "తెలుగు",
        "script": "Telugu",
        "whisper_code": "te",
    },
    "bn": {
        "name": "Bengali",
        "native_name": "বাংলা",
        "script": "Bengali",
        "whisper_code": "bn",
    },
    "ta": {
        "name": "Tamil",
        "native_name": "தமிழ்",
        "script": "Tamil",
        "whisper_code": "ta",
    },
    "gu": {
        "name": "Gujarati",
        "native_name": "ગુજરાતી",
        "script": "Gujarati",
        "whisper_code": "gu",
    },
    "kn": {
        "name": "Kannada",
        "native_name": "ಕನ್ನಡ",
        "script": "Kannada",
        "whisper_code": "kn",
    },
    "ml": {
        "name": "Malayalam",
        "native_name": "മലയാളം",
        "script": "Malayalam",
        "whisper_code": "ml",
    },
    "or": {
        "name": "Odia",
        "native_name": "ଓଡ଼ିଆ",
        "script": "Odia",
        "whisper_code": "or",
    },
    "pa": {
        "name": "Punjabi",
        "native_name": "ਪੰਜਾਬੀ",
        "script": "Gurmukhi",
        "whisper_code": "pa",
    },
    "as": {
        "name": "Assamese",
        "native_name": "অসমীয়া",
        "script": "Assamese",
        "whisper_code": "as",
    },
}

LANGUAGE_NAMES: Final[dict[str, str]] = {
    code: info["name"] for code, info in SUPPORTED_LANGUAGES.items()
}


def validate_language(code: str) -> str:
    """Validate an ISO-639-1 language code and return the normalized code.

    Raises ``ValueError`` when the code is not in ``SUPPORTED_LANGUAGES``.
    """
    if not isinstance(code, str):
        raise ValueError(
            f"Language code must be a string, got {type(code).__name__}."
        )
    normalized = code.strip().lower()
    if normalized not in SUPPORTED_LANGUAGES:
        available = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(
            f"Unsupported language code '{code}'. Supported codes: {available}."
        )
    return normalized


__all__ = ["SUPPORTED_LANGUAGES", "LANGUAGE_NAMES", "validate_language"]
