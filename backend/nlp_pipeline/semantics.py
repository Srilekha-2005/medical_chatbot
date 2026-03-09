"""Semantic analysis: key terms, medical keyword detection, and mapping values to medical ranges."""

import re
from typing import Any, Optional

from backend.nlp_pipeline.morphology import tokenize

# Medical/health keyword sets
MEDICAL_TERMS = {
    "symptom", "treatment", "diagnosis", "patient", "disease", "condition",
    "medication", "therapy", "blood", "heart", "pressure", "sugar", "pain",
    "infection", "doctor", "health", "medical", "clinical", "drug", "dose",
    "surgery", "test", "result", "risk", "cause", "inherited", "gene", "genetic",
    "syndrome", "disorder", "sign", "seizure", "epilepsy", "inheritance",
    # Common conditions, symptoms, and lab / body terms to improve semantic detection
    "asthma", "pneumonia", "anemia", "hypertension", "diabetes",
    "headache", "migraine", "cold", "flu", "cough", "fever",
    "cramp", "cramps", "menstrual", "period", "fatigue", "nausea",
    "vomiting", "diarrhea", "inflammation",
    "vitamin", "deficiency", "cholesterol", "kidney", "liver", "lung",
    "thrombosis", "cardiomyopathy", "bilirubin", "creatinine", "hemoglobin",
    "platelet", "platelets", "wbc", "rbc", "glucose",
}
HEALTH_INDICATORS = {
    "sleep", "bmi", "stress", "exercise", "diet", "vital",
    "oxygen", "heart", "rate",
}


def _normalize_medical_plurals(text: str) -> str:
    """
    Normalize common plural symptom forms so semantic detection works for both
    singular and plural (e.g. "headaches" → "headache").
    """
    if not text:
        return ""
    normalized = text.lower()
    replacements = {
        "headaches": "headache",
        "colds": "cold",
        "cramps": "cramp",
        "migraines": "migraine",
        "fevers": "fever",
    }
    for src, dest in replacements.items():
        normalized = normalized.replace(src, dest)
    return normalized

# --- Medical range definitions (value → label) ---
# Blood pressure: systolic/diastolic in mmHg
def _classify_blood_pressure(systolic: float, diastolic: float) -> str:
    if systolic >= 180 or diastolic >= 120:
        return "hypertensive crisis"
    if systolic >= 140 or diastolic >= 90:
        return "elevated blood pressure (high, stage 2)"
    if systolic >= 130 or diastolic >= 80:
        return "elevated blood pressure (high, stage 1)"
    if 120 <= systolic < 130 and diastolic < 80:
        return "elevated blood pressure"
    if systolic < 120 and diastolic < 80:
        return "normal blood pressure"
    return "blood pressure (unclassified)"


def _classify_heart_rate(bpm: float) -> str:
    if bpm < 60:
        return "low heart rate (bradycardia)"
    if bpm > 100:
        return "elevated heart rate (tachycardia)"
    return "normal heart rate"


def _classify_temperature_f(fahrenheit: float) -> str:
    if fahrenheit >= 100.4:
        return "elevated temperature (fever)"
    if fahrenheit < 95.0:
        return "low temperature (hypothermia)"
    return "normal temperature"


def _classify_oxygen_saturation(percent: float) -> str:
    if percent >= 95:
        return "normal oxygen saturation"
    if percent >= 90:
        return "mildly low oxygen saturation"
    return "low oxygen saturation (hypoxemia)"


def _classify_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal weight"
    if bmi < 30:
        return "overweight"
    return "obese"


def _classify_glucose_fasting(mg_dl: float) -> str:
    if mg_dl < 100:
        return "normal fasting glucose"
    if mg_dl < 126:
        return "elevated fasting glucose (prediabetes range)"
    return "elevated fasting glucose (diabetes range)"


def map_value_to_medical_range(metric: str, value: str) -> Optional[str]:
    """
    Map a metric value to a medical range label.
    Example: ("blood_pressure", "150/95") → "elevated blood pressure (high, stage 2)"
    """
    metric = (metric or "").strip().lower()
    value = (value or "").strip()

    if metric == "blood_pressure":
        m = re.match(r"(\d{2,3})\s*/\s*(\d{2,3})", value)
        if m:
            s, d = float(m.group(1)), float(m.group(2))
            return _classify_blood_pressure(s, d)
    if metric in ("heart_rate", "pulse"):
        m = re.match(r"(\d{2,3})", value)
        if m:
            return _classify_heart_rate(float(m.group(1)))
    if metric == "temperature":
        m = re.match(r"(\d{2,3}\.?\d*)", value.replace(",", "."))
        if m:
            t = float(m.group(1))
            if t <= 45:  # assume Celsius (e.g. 37)
                t_f = t * 9 / 5 + 32
                return _classify_temperature_f(t_f)
            return _classify_temperature_f(t)  # assume Fahrenheit (e.g. 98.6)
    if metric == "oxygen_saturation":
        m = re.match(r"(\d{2,3})", value)
        if m:
            return _classify_oxygen_saturation(float(m.group(1)))
    if metric == "bmi":
        m = re.match(r"(\d+\.?\d*)", value)
        if m:
            return _classify_bmi(float(m.group(1)))
    if metric == "glucose":
        m = re.match(r"(\d{2,3})", value)
        if m:
            return _classify_glucose_fasting(float(m.group(1)))
    return None


def extract_key_terms(text: str, max_terms: int = 20) -> list[str]:
    """Extract key terms (longer, non-stopwords)."""
    tokens = tokenize(text)
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during"}
    candidates = [t for t in set(tokens) if len(t) > 2 and t not in stop]
    return sorted(candidates, key=lambda w: -len(w))[:max_terms]


def analyze_semantics(text: str, metric_value_pairs: Optional[list[dict]] = None) -> dict[str, Any]:
    """
    Semantic analysis: key terms, medical terms, and mapping of metric-value pairs
    to medical ranges (e.g. 150/95 → elevated blood pressure).
    If metric_value_pairs is not provided, syntax extraction is not run; pass pairs from syntax step for range mapping.
    """
    from backend.nlp_pipeline.syntax import extract_metric_value_pairs

    normalized_text = _normalize_medical_plurals(text or "")
    tokens = set(tokenize(normalized_text))
    medical_found = list(tokens & MEDICAL_TERMS)
    health_found = list(tokens & HEALTH_INDICATORS)
    pairs = metric_value_pairs
    if pairs is None:
        pairs = [{"metric": p.metric, "value": p.value} for p in extract_metric_value_pairs(normalized_text)]
    mapped = []
    for p in pairs:
        metric = p.get("metric") or p.get("metric_key")
        value = p.get("value")
        if metric and value is not None:
            label = map_value_to_medical_range(metric, str(value))
            if label:
                mapped.append({"metric": metric, "value": value, "medical_range": label})

    return {
        "key_terms": extract_key_terms(normalized_text or ""),
        "medical_terms_found": medical_found,
        "health_indicators_found": health_found,
        "has_medical_content": len(medical_found) > 0 or len(health_found) > 0,
        "value_to_range": mapped,
    }
