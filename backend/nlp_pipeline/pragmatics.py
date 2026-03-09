"""Pragmatic analysis: intent, question vs statement, and safety (educational only; no diagnosis/treatment advice)."""

import re
from typing import Any

# Patterns that suggest diagnosis or treatment advice (responses should avoid these)
DIAGNOSIS_PATTERNS = [
    re.compile(r"\byou\s+have\s+(?:a\s+)?(?:the\s+)?\b", re.I),
    re.compile(r"\bi\s+diagnose\b", re.I),
    re.compile(r"\bdiagnosis\s*:\s*\w+", re.I),
    re.compile(r"\byou\s+(?:have|are)\s+suffering\s+from\b", re.I),
    re.compile(r"\b(?:this\s+is|that\s+is)\s+(?:likely\s+)?\w+\s+disease\b", re.I),
]
TREATMENT_ADVICE_PATTERNS = [
    re.compile(r"\byou\s+should\s+(?:take|use|start|stop)\b", re.I),
    re.compile(r"\bi\s+(?:prescribe|recommend)\s+(?:you\s+)?(?:to\s+)?(?:take|use)\b", re.I),
    re.compile(r"\btake\s+(?:this|these)\s+medication", re.I),
    re.compile(r"\b(?:your\s+)?(?:dosage|dose)\s+(?:is|should\s+be)\b", re.I),
    re.compile(r"\bstop\s+(?:taking|using)\s+(?:your\s+)?\b", re.I),
    re.compile(r"\bstart\s+(?:taking|using)\s+\b", re.I),
    re.compile(r"\byou\s+need\s+(?:to\s+)?(?:take|see\s+a\s+doctor)\b", re.I),
]
# Educational / acceptable hedging and disclaimers
EDUCATIONAL_MARKERS = [
    re.compile(r"\bin\s+general\b", re.I),
    re.compile(r"\btypically\b", re.I),
    re.compile(r"\bgenerally\b", re.I),
    re.compile(r"\bthis\s+is\s+for\s+educational\s+purposes\b", re.I),
    re.compile(r"\bconsult\s+(?:your\s+)?(?:a\s+)?(?:healthcare\s+)?(?:provider|doctor|physician)\b", re.I),
    re.compile(r"\bnot\s+(?:a\s+)?(?:medical\s+)?(?:advice|diagnosis)\b", re.I),
]


def analyze_pragmatics(text: str) -> dict[str, Any]:
    """
    Pragmatic analysis: question vs statement, negation, hedging, and safety checks.
    Ensures responses can be flagged if they contain diagnosis or treatment advice;
    educational tone is encouraged.
    """
    stripped = (text or "").strip()
    is_question = stripped.endswith("?") or stripped.lower().startswith(
        ("what", "how", "why", "when", "where", "is ", "are ", "do ", "does ", "can ", "could ")
    )
    negation_pattern = re.compile(r"\b(not|no|never|none|neither|n't)\b", re.I)
    hedging = re.compile(r"\b(may|might|could|possibly|perhaps|sometimes|often|generally|usually)\b", re.I)

    # Safety: detect diagnosis/treatment language
    diagnosis_detected = any(p.search(stripped) for p in DIAGNOSIS_PATTERNS)
    treatment_advice_detected = any(p.search(stripped) for p in TREATMENT_ADVICE_PATTERNS)
    educational_markers_present = any(p.search(stripped) for p in EDUCATIONAL_MARKERS)

    return {
        "is_question": is_question,
        "has_negation": bool(negation_pattern.search(stripped)),
        "hedging_detected": bool(hedging.search(stripped)),
        "length_category": "short" if len(stripped) < 50 else "medium" if len(stripped) < 200 else "long",
        "diagnosis_language_detected": diagnosis_detected,
        "treatment_advice_detected": treatment_advice_detected,
        "educational_markers_present": educational_markers_present,
        "is_appropriate_for_education": not (diagnosis_detected or treatment_advice_detected) or educational_markers_present,
        "suggest_add_disclaimer": (diagnosis_detected or treatment_advice_detected) and not educational_markers_present,
    }


def ensure_educational_response(text: str) -> tuple[str, dict[str, Any]]:
    """
    Ensure response remains educational: if diagnosis or treatment advice is detected,
    append a standard disclaimer. Returns (possibly modified text, analysis dict).
    """
    analysis = analyze_pragmatics(text)
    disclaimer = (
        " This is for educational purposes only and is not medical advice, diagnosis, or treatment. "
        "Please consult a healthcare provider for personal health decisions."
    )
    if analysis.get("suggest_add_disclaimer") and disclaimer.strip() not in (text or ""):
        return (text or "").rstrip() + disclaimer, analysis
    return text or "", analysis


def filter_unsafe_content(text: str) -> str:
    """
    Do not modify content; return as-is. Callers can use analyze_pragmatics to decide
    whether to show or to run ensure_educational_response. Provided for API consistency.
    """
    return text or ""
