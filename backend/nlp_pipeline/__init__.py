"""NLP pipeline: morphology, syntax, semantics, discourse, pragmatics."""

from backend.nlp_pipeline.morphology import (
    analyze_morphology,
    tokenize,
    normalize_medical_abbreviations,
)
from backend.nlp_pipeline.syntax import (
    analyze_syntax,
    sentence_split,
    extract_metric_value_pairs,
    MetricValuePair,
)
from backend.nlp_pipeline.semantics import (
    analyze_semantics,
    extract_key_terms,
    map_value_to_medical_range,
)
from backend.nlp_pipeline.discourse import (
    analyze_discourse,
    detect_trend_timeseries,
    detect_trends_multi_metric,
)
from backend.nlp_pipeline.pragmatics import (
    analyze_pragmatics,
    ensure_educational_response,
)


def run_pipeline(text: str) -> dict:
    """
    Run full NLP pipeline: morphology → syntax → semantics (with metric-value and range mapping)
    → discourse → pragmatics.
    """
    morphology_out = analyze_morphology(text)
    syntax_out = analyze_syntax(text)
    semantics_out = analyze_semantics(
        text,
        metric_value_pairs=syntax_out.get("metric_value_pairs"),
    )
    discourse_out = analyze_discourse(text)
    pragmatics_out = analyze_pragmatics(text)
    return {
        "morphology": morphology_out,
        "syntax": syntax_out,
        "semantics": semantics_out,
        "discourse": discourse_out,
        "pragmatics": pragmatics_out,
    }


__all__ = [
    "analyze_morphology",
    "analyze_syntax",
    "analyze_semantics",
    "analyze_discourse",
    "analyze_pragmatics",
    "run_pipeline",
    "tokenize",
    "normalize_medical_abbreviations",
    "sentence_split",
    "extract_metric_value_pairs",
    "MetricValuePair",
    "extract_key_terms",
    "map_value_to_medical_range",
    "detect_trend_timeseries",
    "detect_trends_multi_metric",
    "ensure_educational_response",
]
