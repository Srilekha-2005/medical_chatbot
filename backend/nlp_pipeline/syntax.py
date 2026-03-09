"""Syntactic analysis: sentence segmentation and extraction of metric-value pairs."""

import re
from dataclasses import dataclass
from typing import Any, Optional

from backend.nlp_pipeline.morphology import tokenize, normalize_medical_abbreviations


# Metric names (normalized) and their canonical key and value regex
METRIC_PATTERNS = [
    (r"\bblood\s+pressure\b", "blood_pressure", r"(\d{2,3})\s*/\s*(\d{2,3})"),  # 120/80
    (r"\bbp\b", "blood_pressure", r"(\d{2,3})\s*/\s*(\d{2,3})"),
    (r"\bheart\s+rate\b", "heart_rate", r"(\d{2,3})\s*(?:bpm)?"),
    (r"\b(?:hr|pulse)\b", "heart_rate", r"(\d{2,3})\s*(?:bpm)?"),
    (r"\btemperature\b", "temperature", r"(\d{2,3}\.?\d*)\s*°?[fc]?"),
    (r"\btemp(?:erature)?\b", "temperature", r"(\d{2,3}\.?\d*)\s*°?[fc]?"),
    (r"\b(?:oxygen\s+saturation|spo2|sp\s*o2|o2\s*sat)\b", "oxygen_saturation", r"(\d{2,3})\s*%?"),
    (r"\b(?:weight|body\s+mass)\b", "weight", r"(\d+\.?\d*)\s*(?:kg|lb|pounds?)?"),
    (r"\bbmi\b", "bmi", r"(\d+\.?\d*)"),
    (r"\bglucose\b", "glucose", r"(\d{2,3})"),
    (r"\bblood\s+sugar\b", "glucose", r"(\d{2,3})"),
    (r"\b(?:respiratory\s+rate|respiratory\s+rate|breathing\s+rate)\b", "respiratory_rate", r"(\d{1,2})"),
    (r"\bsystolic\b", "systolic_bp", r"(\d{2,3})"),
    (r"\bdiastolic\b", "diastolic_bp", r"(\d{2,3})"),
]


def sentence_split(text: str) -> list[str]:
    """Split text into sentences."""
    if not text or not text.strip():
        return []
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class MetricValuePair:
    """A single metric-value pair extracted from text."""
    metric: str
    value: str
    raw_span: Optional[str] = None


def extract_metric_value_pairs(text: str) -> list[MetricValuePair]:
    """
    Extract metric-value pairs from sentences.
    Example: "My blood pressure is 150/95" → [MetricValuePair(metric="blood_pressure", value="150/95")]
    """
    if not text or not text.strip():
        return []
    normalized = normalize_medical_abbreviations(text)
    pairs: list[MetricValuePair] = []
    seen: set[tuple[str, str]] = set()

    for metric_re, canonical_metric, value_re in METRIC_PATTERNS:
        for m in re.finditer(metric_re, normalized, re.IGNORECASE):
            # Look for value after the metric (within same sentence or next 50 chars)
            start = m.end()
            window = normalized[start : start + 80]
            val_match = re.search(value_re, window)
            if val_match:
                if canonical_metric == "blood_pressure" and len(val_match.groups()) >= 2:
                    value = f"{val_match.group(1)}/{val_match.group(2)}"
                else:
                    value = val_match.group(1)
                key = (canonical_metric, value)
                if key not in seen:
                    seen.add(key)
                    pairs.append(
                        MetricValuePair(
                            metric=canonical_metric,
                            value=value,
                            raw_span=f"{m.group(0)} {val_match.group(0)}".strip(),
                        )
                    )

    # Also try generic "X is Y" pattern for known metric names
    is_pattern = re.compile(
        r"\b(blood\s+pressure|heart\s+rate|temperature|temp|spo2|oxygen\s+saturation|bmi|glucose|weight)\s+is\s+([\d./]+)\s*%?",
        re.IGNORECASE,
    )
    canonical_lower = {
        "blood pressure": "blood_pressure",
        "heart rate": "heart_rate",
        "temperature": "temperature",
        "temp": "temperature",
        "spo2": "oxygen_saturation",
        "oxygen saturation": "oxygen_saturation",
        "bmi": "bmi",
        "glucose": "glucose",
        "weight": "weight",
    }
    for m in is_pattern.finditer(normalized):
        name = m.group(1).lower().strip()
        value = m.group(2).strip()
        canonical = canonical_lower.get(name, name.replace(" ", "_"))
        key = (canonical, value)
        if key not in seen:
            seen.add(key)
            pairs.append(MetricValuePair(metric=canonical, value=value, raw_span=m.group(0)))

    return pairs


def analyze_syntax(text: str) -> dict[str, Any]:
    """
    Syntactic analysis: sentences, sentence count, and extracted metric-value pairs.
    Example: "My blood pressure is 150/95" yields metric_value_pairs with blood_pressure: 150/95.
    """
    sentences = sentence_split(text or "")
    word_counts = [len(tokenize(s)) for s in sentences]
    pairs = extract_metric_value_pairs(text or "")
    return {
        "sentences": sentences,
        "sentence_count": len(sentences),
        "avg_sentence_length": sum(word_counts) / len(word_counts) if word_counts else 0,
        "max_sentence_length": max(word_counts) if word_counts else 0,
        "metric_value_pairs": [
            {"metric": p.metric, "value": p.value, "raw_span": p.raw_span} for p in pairs
        ],
    }
