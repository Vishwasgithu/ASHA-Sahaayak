"""Rule-based symptom extraction and maternal risk classification.

This module identifies risk signals only. Clinical recommendations are left to
the evidence-grounded RAG pipeline.

Multilingual symptom support
----------------------------
Symptom detection is driven entirely by ``_SYMPTOM_ALIASES`` below. Every
symptom maps to a tuple of surface forms grouped by language.

The alias table is the ONLY place where language-specific symptom matching
happens. Detection, negation handling, pregnancy-month extraction, risk
classification, and RAG-query construction are all language-agnostic: they
operate on the canonical English symptom keys, never on the raw surface
forms.
"""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Multilingual symptom alias table.
# ---------------------------------------------------------------------------

_SYMPTOM_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "fever": (
        "fever",
        "bukhar",
        "ताप",
        "बुखार",
        "ताप आला",
        "ताप येत आहे",
        "జ్వరం",
        "జ్వరము",
        "জ্বর",
        "জ্বর গula",
        "காய்ச்சல்",
        "காய்ச்சல் உள்ளது",
        "તાપ",
        "જુરમ",
        "ಜ್ವರ",
        "ಜ್ವರ ಬಂದಿದೆ",
        "ജ്വരം",
        "താപം",
        "ଜ୍ୱର",
        "ତାପ",
        "ਬੁਖਾਰ",
        "ਤਾਪ",
        "জ্বৰ",
        "তাপ",
    ),
    "headache": (
        "headache",
        "head ache",
        "sar dard",
        "सिरदर्द",
        "डोकेदुखी",
        "डोकेदुकी",
        "తలనొప్పి",
        "తలనొప్పులు",
        "মাথা ব্যথা",
        "মাথা বাঁচছে",
        "தலைவலி",
        "தலை வலி",
        "માથું દુખાવો",
        "સર દુખાવો",
        "ತಲೆನೋವು",
        "ತಲೆ ನೋವು",
        "ತಲೆ ಬಯಲು",
        "തലവേദന",
        "തലവലി",
        "ମୁଣ୍ଡ ଦୁଃଖ",
        "ମୁଣ୍ଡ ବଥା",
        "ਸਰ ਦਰਦ",
        "ਸਿਰ ਦਰਦ",
        "মূথ আঠা",
        "মূথৰ দুক",
    ),
    "dizziness": (
        "dizziness",
        "dizzy",
        "chakkar",
        "चक्कर",
        "गरगरणे",
        "चक्कर येते",
        "తల తిరగడం",
        "తల తిరుగుట",
        "চক্কার ধারণ",
        "মাথা ঘোরা",
        "ঘোরা",
        "தலை சுழற்சி",
        "சக்கரம்",
        "ચક્કર",
        "માથું ફેરવવું",
        "ತಲ ತಿರುಗು",
        "ಚಕ್ರ",
        "തല തിരിയൽ",
        "തല സഞ്ചരിക്കൽ",
        "തല സഞ്ചരിക്കുന്നു",
        "ମୁଣ୍ଡ ଘୁର୍ମା",
        "ଚକ୍ର",
        "ଘୁର୍ମା",
        "ਚੱਕਰ",
        "ਸਿਰ ਘੁਮਣਾ",
        "ਘੁਮਦਾ",
        "চক্কাৰ",
        "মূথ ঘূৰা",
        "ঘূৰা",
    ),
    "swelling": (
        "swelling",
        "sujan",
        "सूजन",
        "सूज",
        "सूजणे",
        "వాపు",
        "వాపు వచ్చింది",
        "স্ফীতি",
        "ফোলা",
        "வீக்கம்",
        "வீக்கம் உள்ளது",
        "સૂજ",
        "પોથો",
        "ಊರು",
        "ಉಬ್ಬು",
        "വീക്ക്",
        "പൊട്ടിപ്പുക",
        "വീക്ഷണം",
        "ସୂଜ",
        "ଫୁଲ",
        "ਸੂਜ",
        "ਫੁੱਲਣਾ",
        "সূজ",
        "ফুলা",
    ),
    "vomiting": (
        "vomiting",
        "vomit",
        "ulti",
        "उल्टी",
        "ओकाऱ्या",
        "उलटी",
        "వాంతులు",
        "వాంతి",
        "বমি",
        "উলটা",
        "வாந்தி",
        "வெளியேக்கம்",
        "ઉલટી",
        "વાંતી",
        "ವಾಂತಿ",
        "ಕೆಮ್ಮು",
        "വാന്തി",
        "ഉണ്ടാക്കൽ",
        "ଉଲ୍ଟି",
        "ବାନ୍ତି",
        "ਉਲਟੀ",
        "ਵੰਤੀ",
        "উলট",
    ),
    "bleeding": (
        "bleeding",
        "khoon",
        "खून",
        "रक्तस्त्राव",
        "रक्तસ્રાવ",
        "రక్తస్రావం",
        "రక్త స్రావం",
        "রক্তপাত",
        "রক্ত",
        "இரத்தம்",
        "இரத்தம் விழுதல்",
        "રક્તસ્રાવ",
        "ખોન",
        "લોબો વટવું",
        "ರಕ್ತ ಸ್ರಾವ",
        "ರಕ್ತ ಹರಿಯುವುದು",
        "ರಕ್ತ ಹರಿಯುತ್ತಿದೆ",
        "രക്തസ്രാവം",
        "രക്തപാതം",
        "ରକ୍ତସ୍ରାବ",
        "ଲୋହ",
        "ਰਕਤਸਰਾਵ",
        "ਖੂਨ",
        "ৰক্তপাত",
        "খন",
    ),
    "weakness": (
        "weakness",
        "weak",
        "kamzori",
        "कमजोरी",
        "अशक्तपणा",
        "दुबळेपणा",
        "బలహీనత",
        "నిస్సత్తువ",
        "দুর্বলতা",
        "শক্তিহীনতা",
        "দুর্বলতাই",
        "பலவின்மை",
        "குறைபாடு",
        "કમજોરી",
        "દુર્બળતા",
        "ದುರ್ಬಲತೆ",
        "ಬಲಹೀನತೆ",
        "ദുർബലത",
        "ബലഹീനത",
        "ଦୁର୍ବଳତା",
        "ଶକ୍ତିହୀନତା",
        "ਕਮਜ਼ੋਰੀ",
        "ਦੁਰਬਲতা",
        "দুৰ্বলতা",
    ),
    "diarrhoea": (
        "diarrhoea",
        "diarrhea",
        "dast",
        "दस्त",
        "जुलाब",
        "पातळ विष्ठा",
        "విరేచనాలు",
        "విరేచనం",
        "ডায়রিয়া",
        "ডায়ারিয়া",
        "வயிற்றுப்புணர்",
        "வியர்த்தல்",
        "வயிற்றுப்பயிற்சி",
        "ડાયરિયા",
        "જુલાબ",
        "ಅತಿಸಾರ",
        "ಜುಲಾಬ",
        "അതിസാരം",
        "ജുലാബ്",
        "ଅତିସାର",
        "ଦସ୍ତ",
        "ਦਸਤ",
        "ਡਾਇਰੀਆ",
        "দস্ত",
        "কলা",
    ),
    "blurred vision": (
        "blurred vision",
        "blurry vision",
        # Added English variations
        "vision is blurred",
        "vision is blurry",
        "my vision is blurred",
        "my vision is blurry",
        "vision blurred",
        "vision blurry",
        "blurred eyesight",
        "blurry eyesight",
        "dhundhla dikhna",
        "धुंधला दिखाई देना",
        "धूसर दिसणे",
        "धुंदर दिसणे",
        "మసక చూపు",
        "మబ్బు చూపు",
        "ঘোরা দৃষ্টি",
        "দৃষ্টিভঙ্গি",
        "দৃষ্টি ঝula",
        "மங்கிய காட்சி",
        "தெரியாத கண்",
        "ધૂમ્રેલું દેખાવ",
        "ધૂમ્મસ દૃષ્ટિ",
        "દૃષ્ટિ ધૂમ્રી",
        "ಅಸ್ಪಷ್ಟ ದೃಷ್ಟಿ",
        "ಮಸಕಾದ ಕಾಣು",
        "ದೃಷ್ಟಿ ಮಸಕು",
        "മങ്ങിയ കാഴ്ച",
        "അസ്പഷ്ടമായ കാഴ്ച",
        "കാഴ്ച മങ്ങിയത്",
        "ଧୂମ୍ର ଦୃଷ୍ଟି",
        "ଭୁଲ୍ ଦୃଷ୍ଟି",
        "ଦୃଷ୍ଟି ଧୂମ୍ର",
        "ਧੂਮ੍ਰਾ ਦਿੱਖ",
        "ਧੂਮ੍ਰੀ ਦ੍ਰਿਸ਼ਟੀ",
        "ਦਿੱਖ ਧੂਮ੍ਰੀ",
        "অস্পষ্ট দৃষ্টি",
        "ধুম্ৰে দৃষ্টি",
        "দৃষ্টি অস্পষ্ট",
    ),
    "severe abdominal pain": (
        "severe abdominal pain",
        "severe stomach pain",
        "tez pet dard",
        "तेज पेट दर्द",
        "तीव्र पोटदुखी",
        "पोटाचा तीव्र ત્રાસ",
        "తీవ్రమైన కడుపు నొప్పి",
        "కడుపు నొప్పి తీవ్రంగా",
        "তীব্র পেট ব্যথা",
        "পেটে ব্যথা",
        "கடுமையான வயிற்று வலி",
        "அடisent பகுதி வலி",
        "તીવ્ર પેટ દુખાવો",
        "પેટની મોટી દુઃખ",
        "ಕಠಿಣ ಹೊಟ್ಟೆ ನೋವು",
        "ಹೊಟ್ಟೆ ಬೆಚ್ಚಗಿನ ದುಃಖ",
        "കടുത്ത വയ നോവു",
        "വയിൽ കടുത്ത വേദനാ",
        "ତୀବ୍ର ପେଟ ଦୁଃଖ",
        "ପେଟର ବଡ ଦୁଃଖ",
        "ਤੀਬਰ ਪੇਟ ਦਰਦ",
        "ਪੇਟ ਵਿੱਚ ਗੁੱਟ",
        "তীব্ৰ পেটৰ দুঃখ",
        "পেটত বঢ় ব্যথা",
    ),
    "reduced fetal movement": (
        "reduced fetal movement",
        "decreased fetal movement",
        "less fetal movement",
        # English variations
        "baby moving less",
        "baby is moving less",
        "baby moving less than usual",
        "baby is moving less than usual",
        "my baby is moving less",
        "my baby is moving less than usual",
        "baby movements are less",
        "movement of baby is less",
        "baby not moving as much",
        "baby is not moving as much",
        "baby movements reduced",
        "bacche ki harkat kam",
        "बाळाची हालचाल कमी",
        "गर्भाची हालचाल कमी",
        "शिशువు కదలికలు తగ్గాయి",
        "పిండం కదలికలు తగ్గాయి",
        "শিশুর গতি কম",
        "শৈশবে চলন কম",
        "শিশু কম চলছে",
        "கருத்தில் குழந்தை நகர்வு குறைவு",
        "குழந்தை நகர்வு குறைந்தது",
        "குறைந்த நகர்வு",
        "જન્મ પડતા બાળકની ગતિ ઓછી",
        "બાળકની હાલત ઓછી",
        "ಮಗುವಿನ ಚಲನೆ ಕಡಿಮೆ",
        "ಜನ್ಮ ಮಗುವಿನ ಚಲನೆ ಕಡಿಮೆ",
        "ശിശുവിന്റെ ചലനം കുറഞ്ഞു",
        "കുഞ്ഞു ചലിക്കുന്നില്ല",
        "ଶିଶୁ ଚଳନ କମ୍",
        "ଗର୍ଭର ପିଲା ଚଳନ କମ",
        "ପିଲା ଅଳ୍ପ ଚଳନ",
        "ਬਚਚੇ ਦੀ ਗਤਿ ਕਮ",
        "ਬੱਚਾ ਘੱਟ ਹਿਲ ਰਿਹਾ",
        "ਬੱਚੇ ਦੀ ਗਤਿ ਘੱਟ",
        "শিশুৰ গতি কম",
        "বাচ্চা অল্প চলছে",
        "শিশু কম চলছে",
    ),
}


# ---------------------------------------------------------------------------
# Negation handling
# ---------------------------------------------------------------------------

_NEGATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)(?:"
    r"negative\s+for|no|not|without|denies|"
    r"nahi|nahin|na|nai|nathi|illa|illai|thilee|nhi|"
    r"naahi|nei|nije|naai|noi|nou|nohoi|"
    r"না|নয়|নহি|નથી|ના|இல்லை|இல்|"
    r"ಇಲ್ಲ|ಇಲ್ಲೈ|ഇല്ല|ഇല്ലാ|"
    r"ନାହିଁ|ନା|ਨਹੀਂ|নাই|নহও"
    r")(?![\w])",
    re.IGNORECASE,
)

_POST_NEGATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:"
    r"(?:hai|he|hain|che|aahe|ah|ahe|asti|chhe)\s+"
    r")?"
    r"(?:"
    r"nahi|nahin|na|nai|nathi|illa|thilee|nhi|"
    r"naahi|nei|nije|noi|nou|nohoi|"
    r"না|নয়|নহি|નથી|ના|இல்லை|இல்|"
    r"ಇಲ್ಲ|ಇಲ್ಲೈ|ഇല്ല|ഇല്ലാ|"
    r"ନାହିଁ|ନା|ਨਹੀਂ|নাই|নহও"
    r")(?![\w])",
    re.IGNORECASE,
)

_CLAUSE_BOUNDARY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[.!?;:\n]|\b(?:but|however|although|yet|lekin|par)\b",
    re.IGNORECASE,
)

# Important: "have" is intentionally NOT included here.
# Example:
# "I do not have bleeding"
# The word "have" must not cancel the negation.
#
# These cues indicate a genuinely new positive assertion.
_POSITIVE_CUE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:reports?|reported|complains?\s+of|positive\s+for)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pregnancy month extraction
# ---------------------------------------------------------------------------

_PREGNANCY_MONTH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<!\w)"
    r"(?:months?|mahina|mahine|મહિનો|માસ|ମାସ|மாதம்|తెలలు|"
    r"ತಿಂಗಳು|മാസം|ਮਹੀਨਾ|માહ|বৎসর|મહિના)"
    r"\s*(\d{1,2})(?![\w])"
    r"|"
    r"(?<!\w)(\d{1,2})\s*"
    r"(?:months?|mahina|mahine|મહિનો|માસ|ମାସ|ମାସ|"
    r"மாதம்|తెలలు|ತಿಂಗಳು|മാസം|ਮਹੀਨਾ|માહ|বৎসর|મહિના)"
    r"(?![\w])",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Alias pattern creation
# ---------------------------------------------------------------------------


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Build a whole-word matcher for one alias surface form.

    Avoids Python's standard ``\\b`` boundary because Unicode combining marks
    in Indic scripts can cause valid words to fail boundary matching.
    """

    words = [re.escape(word) for word in phrase.split()]

    return re.compile(
        r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)",
        re.IGNORECASE,
    )


_ALIAS_PATTERNS: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    symptom: tuple(_phrase_pattern(alias) for alias in aliases)
    for symptom, aliases in _SYMPTOM_ALIASES.items()
}


# ---------------------------------------------------------------------------
# Clause handling
# ---------------------------------------------------------------------------


def _current_clause_prefix(text: str, mention_start: int) -> str:
    """Return text before a symptom mention within the current clause."""

    prefix = text[:mention_start]
    boundaries = list(_CLAUSE_BOUNDARY_PATTERN.finditer(prefix))

    return prefix[boundaries[-1].end() :] if boundaries else prefix


def _current_clause_suffix(text: str, mention_end: int) -> str:
    """Return text after a symptom mention within the current clause."""

    suffix = text[mention_end:]
    boundary = _CLAUSE_BOUNDARY_PATTERN.search(suffix)

    return suffix[: boundary.start()] if boundary else suffix


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------


def _is_negated(
    text: str,
    mention_start: int,
    mention_end: int,
) -> bool:
    """Return whether one symptom mention falls within a negated context.

    Examples:

    Positive:
        "I have bleeding"

    Negated:
        "I do not have bleeding"
        "No fever"
        "I have no fever"

    Mixed:
        "I have no fever but I am vomiting"

    The function prevents a negation word from incorrectly cancelling a
    later positive assertion while also ensuring helper verbs such as
    "have" do not cancel the negation in sentences like
    "I do not have bleeding".
    """

    prefix = _current_clause_prefix(text, mention_start)

    negations = list(_NEGATION_PATTERN.finditer(prefix))

    if negations:

        last_negation = negations[-1]

        text_after_negation = prefix[last_negation.end() :]

        # Look for a genuine later positive assertion.
        #
        # We intentionally do NOT use "have" as a positive cue.
        # Otherwise:
        #
        # "I do not have bleeding"
        #
        # would incorrectly become positive.
        positive_match = _POSITIVE_CUE_PATTERN.search(text_after_negation)

        if positive_match is None:
            return True

        # If a positive cue occurs later, determine whether it starts a
        # new semantic assertion.
        #
        # Example:
        # "no fever and reports vomiting"
        #
        # For vomiting, "reports" starts a new positive assertion.
        if positive_match.start() > 0:
            return False

        return True

    # Handles post-symptom negation:
    #
    # "bukhar nahi hai"
    #
    suffix = _current_clause_suffix(
        text,
        mention_end,
    )

    return _POST_NEGATION_PATTERN.search(suffix) is not None


# ---------------------------------------------------------------------------
# Symptom detection
# ---------------------------------------------------------------------------


def _detect_symptoms(text: str) -> list[str]:
    """Detect all non-negated symptoms from the input text."""

    detected: list[str] = []

    for symptom, patterns in _ALIAS_PATTERNS.items():

        mentions = [match for pattern in patterns for match in pattern.finditer(text)]

        # Add the canonical symptom only when at least one mention is
        # present outside a negated context.
        if any(
            not _is_negated(
                text,
                match.start(),
                match.end(),
            )
            for match in mentions
        ):
            detected.append(symptom)

    return detected


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


def _classify_risk(symptoms: set[str]) -> str:
    """Classify maternal health risk from detected canonical symptoms."""

    # Combination associated with a pre-eclampsia warning pattern.
    has_preeclampsia_signals = {
        "headache",
        "swelling",
        "blurred vision",
    }.issubset(symptoms)

    # Individual high-priority danger signals.
    has_other_high_risk_signal = bool(
        symptoms.intersection(
            {
                "bleeding",
                "severe abdominal pain",
                "reduced fetal movement",
            }
        )
    )

    if has_preeclampsia_signals or has_other_high_risk_signal:
        return "HIGH RISK"

    if symptoms.intersection(
        {
            "fever",
            "vomiting",
            "weakness",
        }
    ):
        return "MEDIUM RISK"

    return "LOW RISK"


# ---------------------------------------------------------------------------
# RAG query construction
# ---------------------------------------------------------------------------


def _natural_language_list(values: list[str]) -> str:
    """Convert a list into a readable natural-language phrase."""

    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    return f"{', '.join(values[:-1])} " f"and {values[-1]}"


def _build_rag_query(
    risk_level: str,
    symptoms: list[str],
    pregnancy_month: str | None,
) -> str:
    """Build a structured clinical query for the RAG pipeline."""

    risk_label = risk_level.lower().replace(
        " ",
        "-",
    )

    month_context = f" in month {pregnancy_month}" if pregnancy_month else ""

    symptom_context = f" with {_natural_language_list(symptoms)}" if symptoms else ""

    return (
        f"{risk_label.capitalize()} pregnancy"
        f"{month_context}"
        f"{symptom_context} "
        "according to WHO maternal healthcare guidelines."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_healthcare_input(text: str) -> dict[str, object]:
    """Extract non-negated symptoms and prepare a maternal-health RAG query.

    Pipeline:

        Raw user input
              ↓
        Multilingual symptom detection
              ↓
        Negation filtering
              ↓
        Canonical symptom extraction
              ↓
        Risk classification
              ↓
        Structured RAG query generation
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    symptoms = _detect_symptoms(text)

    pregnancy_match = _PREGNANCY_MONTH_PATTERN.search(text)

    pregnancy_month = (
        pregnancy_match.group(1) or pregnancy_match.group(2)
        if pregnancy_match
        else None
    )

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


__all__ = [
    "process_healthcare_input",
]
