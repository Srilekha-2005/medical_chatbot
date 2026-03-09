"""Morphological analysis: tokenization and normalization of medical abbreviations."""

import re
from typing import Any

# Medical abbreviation → full form (case-insensitive match)
MEDICAL_ABBREVIATIONS = {
    "bp": "blood pressure",
    "hr": "heart rate",
    "bpm": "beats per minute",
    "temp": "temperature",
    "spo2": "oxygen saturation",
    "o2": "oxygen",
    "rr": "respiratory rate",
    "bmi": "body mass index",
    "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "mi": "myocardial infarction",
    "cvd": "cardiovascular disease",
    "rx": "prescription",
    "dx": "diagnosis",
    "tx": "treatment",
    "pt": "patient",
    "qd": "once daily",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "prn": "as needed",
    "po": "by mouth",
    "iv": "intravenous",
    "im": "intramuscular",
    "sc": "subcutaneous",
    "ng": "nasogastric",
    "pr": "by rectum",
    "wbc": "white blood cell",
    "rbc": "red blood cell",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "glu": "glucose",
    "ldl": "low-density lipoprotein",
    "hdl": "high-density lipoprotein",
    "tg": "triglycerides",
    "k": "potassium",
    "na": "sodium",
    "cr": "creatinine",
    "bun": "blood urea nitrogen",
    "egfr": "estimated glomerular filtration rate",
}


def tokenize(text: str) -> list[str]:
    """Split text into words (alphanumeric tokens)."""
    return re.findall(r"\b[\w']+\b", text.lower()) if text else []


def normalize_medical_abbreviations(text: str) -> str:
    """
    Replace medical abbreviations with full forms.
    E.g. "BP is 120/80" → "blood pressure is 120/80"
    """
    if not text or not text.strip():
        return text
    result = text
    # Sort by length descending so longer abbreviations (e.g. "spo2") match before "o2"
    for abbr, full in sorted(MEDICAL_ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
        result = pattern.sub(full, result)
    return result


def analyze_morphology(text: str) -> dict[str, Any]:
    """
    Morphological analysis: tokens, counts, and normalized text with medical
    abbreviations expanded (e.g. BP → blood pressure).
    """
    normalized = normalize_medical_abbreviations(text or "")
    tokens = tokenize(normalized)
    return {
        "tokens": tokens,
        "word_count": len(tokens),
        "character_count": len(text or ""),
        "unique_tokens": len(set(tokens)),
        "normalized_text": normalized,
        "abbreviations_expanded": normalized != (text or ""),
    }
