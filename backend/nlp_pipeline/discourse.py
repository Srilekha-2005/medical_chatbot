"""Discourse analysis: cohesion, structure, and trend detection in time-series health data."""

from typing import Any, Optional

from backend.nlp_pipeline.syntax import sentence_split
from backend.nlp_pipeline.morphology import tokenize


def _numeric_values(series: list[Any]) -> list[float]:
    """Extract numeric values from a list (e.g. [120, "130", 125] or list of (t, v) pairs)."""
    out = []
    for x in series:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            out.append(float(x[1]) if x[1] is not None else float("nan"))
        elif isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, str) and x.replace(".", "").replace("-", "").isdigit():
            out.append(float(x))
        else:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                pass
    return out


def _trend_from_slope(values: list[float], threshold_relative: float = 0.05) -> str:
    """
    Classify trend as improving, worsening, or stable using simple linear tendency.
    For health metrics: "improving" when values decrease (e.g. BP), unless we invert by metric.
    This uses sign of (last - first) normalized by range; threshold_relative ignores small changes.
    """
    clean = [v for v in values if v == v]  # drop nan
    if len(clean) < 2:
        return "insufficient_data"
    first, last = clean[0], clean[-1]
    span = max(clean) - min(clean) or 1.0
    change = (last - first) / span
    if abs(change) < threshold_relative:
        return "stable"
    return "improving" if change < 0 else "worsening"


def detect_trend_timeseries(
    values: list[Any],
    *,
    invert_improvement: bool = False,
) -> dict[str, Any]:
    """
    Detect trend across a time-series of values (e.g. repeated blood pressure readings).
    values: list of numbers or list of (timestamp, value) pairs.
    invert_improvement: if True, "improving" means values going up (e.g. SpO2, weight gain desired).
    Returns: trend (stable | improving | worsening), summary stats, and direction.
    """
    nums = _numeric_values(values)
    if not nums:
        return {"trend": "no_data", "summary": {}, "direction": None}
    trend = _trend_from_slope(nums)
    if invert_improvement and trend != "stable":
        trend = "worsening" if trend == "improving" else "improving"
    n = len(nums)
    valid = [x for x in nums if x == x]
    return {
        "trend": trend,
        "summary": {
            "count": n,
            "first": valid[0] if valid else None,
            "last": valid[-1] if valid else None,
            "min": min(valid) if valid else None,
            "max": max(valid) if valid else None,
            "mean": sum(valid) / len(valid) if valid else None,
        },
        "direction": "increasing" if valid and valid[-1] > valid[0] else "decreasing" if valid and valid[-1] < valid[0] else "stable",
    }


def detect_trends_multi_metric(
    data: dict[str, list[Any]],
    *,
    invert_for_metrics: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Detect trends for multiple metrics (time-series health data).
    data: {"blood_pressure_systolic": [120, 125, 118], "heart_rate": [72, 75, 70], ...}
    invert_for_metrics: metrics where higher is better (e.g. oxygen_saturation, so "improving" = values up).
    """
    invert_for_metrics = invert_for_metrics or {"oxygen_saturation", "spo2"}
    invert_for_metrics = {m for m in (invert_for_metrics or set()) if m}
    results = {}
    for metric, series in data.items():
        inv = metric.lower() in invert_for_metrics
        results[metric] = detect_trend_timeseries(series, invert_improvement=inv)
    return results


def analyze_discourse(text: str) -> dict[str, Any]:
    """
    Discourse analysis: sentence/word stats and lexical diversity.
    For trend detection, use detect_trend_timeseries or detect_trends_multi_metric on time-series data.
    """
    sentences = sentence_split(text or "")
    all_tokens = []
    for s in sentences:
        all_tokens.extend(tokenize(s))
    n_tokens = len(all_tokens)
    n_unique = len(set(all_tokens))
    return {
        "num_sentences": len(sentences),
        "total_words": n_tokens,
        "lexical_diversity": n_unique / n_tokens if n_tokens else 0.0,
        "avg_words_per_sentence": n_tokens / len(sentences) if sentences else 0,
    }

