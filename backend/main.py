"""FastAPI application for the AI Health Education Assistant (multi-agent backend)."""

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import re

from backend.config import API_DESCRIPTION, API_TITLE, API_VERSION, MEDICAL_DISCLAIMER, RETRIEVAL_TOP_K, RAG_TOP_K
from backend.agents import DataRetrievalAgent, HealthInsightAgent
from backend.data_loader import load_all_datasets
from backend.nlp_pipeline import analyze_syntax, run_pipeline
from backend.nlp_pipeline.discourse import detect_trend_timeseries
from backend.rag import RAGPipeline
from backend.report_processing import parse_uploaded_report

# Metric keys used by the insight agent for value-to-range explanation
CHAT_METRIC_KEYS = ("blood_pressure", "heart_rate", "glucose", "oxygen_saturation", "temperature")

# In-memory conversation context: last detected metric(s) for follow-up questions
last_metric_context: dict[str, Any] = {}
NON_MEDICAL_RESPONSE = (
    "I can provide educational information about health metrics and medical topics, "
    "but I cannot answer non-health questions."
)
FOLLOW_UP_PATTERNS = [
    r"is\s+that\s+(dangerous|bad|concerning|serious|normal|ok|high|low)",
    r"is\s+it\s+(dangerous|normal|bad|concerning|serious|ok|high|low)",
    r"what\s+does\s+(it|that)\s+mean",
    r"should\s+i\s+(worry|be\s+concerned|see\s+a\s+doctor)",
    r"what\s+about\s+that",
    r"is\s+that\s+(safe|dangerous)",
    r"how\s+(bad|serious)\s+is\s+(it|that)",
    r"what\s+(should|can)\s+i\s+do",
    r"when\s+should\s+i\s+see\s+(a\s+)?doctor",
    r"what\s+should\s+i\s+do",
]

# Keyword-based fallback: treat as medical if any of these appear (even if semantic detection fails)
MEDICAL_KEYWORDS = [
    # Metrics and vital signs
    "blood pressure",
    "bp",
    "heart rate",
    "hr",
    "oxygen",
    "spo2",
    "temperature",
    "glucose",
    "sugar",
    "pulse",
    "bmi",
    # Conditions and diseases
    "diabetes",
    "hypertension",
    "cholesterol",
    "heart",
    "lung",
    "kidney",
    "liver",
    "asthma",
    "allergy",
    "virus",
    "bacteria",
    "immune",
    "disease",
    "infection",
    "inflammation",
    # Symptoms and complaints
    "symptom",
    "pain",
    "fever",
    "cold",
    "flu",
    "cough",
    "headache",
    "migraine",
    "stomach",
    "cramps",
    "period",
    "menstrual",
    "nausea",
    "vomiting",
    "diarrhea",
    "fatigue",
    "weakness",
    "breathing",
    "rash",
]


def _classify_medical_question(query: str) -> str:
    """
    Classify medical question type for retrieval and response strategy.
    Returns: metric_question | definition_question | cause_question | lifestyle_question |
             risk_question | doctor_advice_question | disease_specific_question
    """
    q = (query or "").strip().lower()
    if len(q) > 200:
        return "definition_question"
    # doctor_advice_question: "When should I see a doctor?"
    if re.search(r"\bwhen\s+should\s+i\s+see\s+(a\s+)?doctor\b", q):
        return "doctor_advice_question"
    if re.search(r"\bshould\s+i\s+see\s+(a\s+)?(doctor|physician)\b", q):
        return "doctor_advice_question"
    # risk_question: "Is that dangerous?", "Is blood pressure 150/95 dangerous?"
    if re.search(r"\bis\s+(it|that|this)\s+(dangerous|safe|serious|concerning)\b", q):
        return "risk_question"
    if re.search(r"\b(dangerous|safe|risky|harmful)\b", q):
        return "risk_question"
    if re.search(r"\bshould\s+i\s+worry\b", q):
        return "risk_question"
    # lifestyle_question: "How can I lower blood pressure naturally?", "How can I improve lung health?"
    if re.search(r"\b(how\s+can\s+i|how\s+to|ways\s+to|tips\s+to)\b", q) and re.search(
        r"\b(improve|reduce|lower|prevent|manage|control)\b", q
    ):
        return "lifestyle_question"
    if re.search(r"\bhow\s+can\s+i\s+improve\b", q):
        return "lifestyle_question"
    if re.search(r"\bhow\s+to\s+improve\b", q) or re.search(r"\bimprove\s+(my\s+)?(heart|health|lung|lungs)\b", q):
        return "lifestyle_question"
    if re.search(r"\bhow\s+can\s+i\s+reduce\b", q) or re.search(r"\bhow\s+to\s+reduce\b", q):
        return "lifestyle_question"
    if re.search(r"\breduce\s+(pain|headache|cramps|symptoms|blood pressure|bp|blood sugar|glucose)\b", q):
        return "lifestyle_question"
    # cause_question: "What causes high heart rate?", "What causes period cramps?", "What causes common cold?"
    if re.search(r"\bwhat\s+causes\b", q) or re.search(r"\bwhat\s+causes?\b", q) or re.search(r"\bwhy\s+(does|do)\s+", q):
        return "cause_question"
    if re.search(r"\bcauses?\s+of\b", q) or re.search(r"\brisk\s+factors\s+for\b", q):
        return "cause_question"
    # disease_specific_question: "What is Liddle syndrome?" (must be before generic "what is")
    if re.search(r"\bwhat\s+is\s+\w+\s+syndrome\b", q):
        return "disease_specific_question"
    if re.search(r"\bwhat\s+is\s+\w+\s+disease\b", q):
        return "disease_specific_question"
    if re.search(r"\w+\s+syndrome\s*\??\s*$", q):
        return "disease_specific_question"
    # metric_question: normal range / meaning of common metrics
    if re.search(r"\bwhat\s+does\s+.*\s+mean\b", q) and re.search(r"\b(blood pressure|bp|heart rate|glucose|blood sugar|oxygen|spo2|temp|temperature)\b", q):
        return "metric_question"
    if re.search(r"\bwhat\s+is\s+normal\s+(blood pressure|bp|heart rate|pulse|glucose|blood sugar|oxygen saturation|spo2|temperature|temp)\b", q):
        return "metric_question"
    if re.search(r"\bnormal\s+(blood pressure|bp|heart rate|pulse|glucose|blood sugar|oxygen saturation|spo2|temperature|temp)\b", q):
        return "metric_question"
    if re.search(r"\bnormal\s+range\b", q) and re.search(r"\b(blood pressure|bp|heart rate|pulse|glucose|blood sugar|oxygen saturation|spo2|temperature|temp)\b", q):
        return "metric_question"
    if re.search(r"\bwhat\s+is\s+normal\b", q) or re.search(r"\bnormal\s+(range|level|blood pressure)\b", q):
        return "metric_question"
    if re.search(r"\b(good|healthy|ideal)\s+(blood pressure|bp|heart rate|glucose|blood sugar)\b", q):
        return "metric_question"
    # definition_question: "What is asthma?", "What are allergies?", "Define X", "Explain pneumonia"
    if re.search(r"^what\s+is\s+[a-z0-9\s\-]+", q):
        return "definition_question"
    if re.search(r"^what\s+are\s+[a-z0-9\s\-]+", q):
        return "definition_question"
    if re.search(r"^define\s+[a-z0-9\s\-]+", q):
        return "definition_question"
    if re.search(r"^explain\s+[a-z0-9\s\-]+", q):
        return "definition_question"
    if re.search(r"\bwhat\s+is\s+", q):
        return "definition_question"
    if re.search(r"\bwhat\s+are\s+", q):
        return "definition_question"
    if re.search(r"\bdefine\s+", q) or re.search(r"\bexplain\s+", q):
        return "definition_question"
    return "definition_question"


def _extract_condition_name(query: str) -> str:
    """Extract probable condition/disease name from a definition-style query."""
    if not query:
        return ""
    q = (query or "").strip().lower()
    for prefix in ("what is ", "what are ", "define ", "explain "):
        if q.startswith(prefix):
            q = q[len(prefix) :].strip()
            break
    return q.rstrip("?.").strip()


def _is_definition_style_query(query: str) -> bool:
    """True if the query is asking for a definition (what is X, what are X, define X, explain X). Used to avoid rejecting these as non-medical."""
    q = (query or "").strip().lower()
    if len(q) < 4:
        return False
    return (
        bool(re.search(r"^what\s+is\s+[a-z0-9\s\-]+", q))
        or bool(re.search(r"^what\s+are\s+[a-z0-9\s\-]+", q))
        or bool(re.search(r"^define\s+[a-z0-9\s\-]+", q))
        or bool(re.search(r"^explain\s+[a-z0-9\s\-]+", q))
    )


def _classify_followup(query: str) -> str:
    """
    Classify follow-up intent for context-aware responses.
    Returns: risk_question | normal_range_question | cause_question | improvement_question | doctor_advice_question | general_followup
    """
    q = (query or "").strip().lower()
    if len(q) > 120:
        return "general_followup"
    # doctor_advice_question: "When should I see a doctor?", "What should I do?"
    if re.search(r"\bwhen\s+should\s+i\s+see\s+(a\s+)?doctor\b", q):
        return "doctor_advice_question"
    if re.search(r"\bwhat\s+should\s+i\s+do\b", q):
        return "doctor_advice_question"
    # risk_question
    if re.search(r"\b(is\s+that|is\s+it)\s+(dangerous|serious|bad|concerning)\b", q):
        return "risk_question"
    if re.search(r"\bshould\s+i\s+worry\b", q):
        return "risk_question"
    if re.search(r"\bhow\s+(bad|serious)\s+is\b", q):
        return "risk_question"
    # normal_range_question
    if re.search(r"\b(is\s+that|is\s+it)\s+(normal|ok|okay)\b", q):
        return "normal_range_question"
    if re.search(r"\bwhat\s+is\s+normal\b", q) or re.search(r"\bnormal\s+(range|blood\s+pressure)\b", q):
        return "normal_range_question"
    # cause_question
    if re.search(r"\bwhat\s+causes?\s+(it|this|that|high|low)\b", q):
        return "cause_question"
    if re.search(r"\bwhy\s+does\s+(this|it|that)\s+happen\b", q):
        return "cause_question"
    # improvement_question
    if re.search(r"\bhow\s+can\s+i\s+(reduce|lower|improve|bring\s+down)\b", q):
        return "improvement_question"
    if re.search(r"\bhow\s+to\s+(reduce|improve)\b", q):
        return "improvement_question"
    # generic follow-up patterns
    for pat in FOLLOW_UP_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "general_followup"
    return "general_followup"


def _followup_response_fallback(intent: str, metric: str, value: str) -> str:
    """Intent-specific short response when LLM is unavailable for follow-up questions."""
    metric_label = metric.replace("_", " ").title()
    base = f"A reading of {value} for {metric_label}. "
    if intent == "risk_question":
        return (
            base
            + "If this level stays high over time it may increase the risk of heart disease or stroke. "
            "However a single reading does not always indicate a medical condition. "
            "Consult a healthcare provider for personal advice."
        )
    if intent == "normal_range_question":
        return (
            base
            + "Normal blood pressure is typically around 120/80 mmHg. "
            "Consult a healthcare provider to interpret your reading in context."
        )
    if intent == "cause_question":
        return (
            base
            + "Common factors include diet, activity, stress, and genetics. "
            "A healthcare provider can help identify causes relevant to you."
        )
    if intent == "improvement_question":
        return (
            base
            + "Lifestyle changes like diet, exercise, and stress management often help. "
            "Consult a healthcare provider for personalized advice."
        )
    return base + "Consult a healthcare provider for personal advice."


# Structured response for "When should I see a doctor?" (generic; use _doctor_advice_for_metric when context exists)
DOCTOR_ADVICE_RESPONSE = (
    "You should consider seeing a doctor if:\n"
    "• symptoms persist for several days\n"
    "• readings are significantly abnormal\n"
    "• severe symptoms appear (difficulty breathing, chest pain, confusion).\n\n"
    "This is general guidance only; when in doubt, contact a healthcare provider."
)


def _doctor_advice_for_metric(metric: str, value: str) -> str:
    """Context-aware doctor advice when the user asked 'When should I see a doctor?' after sharing a metric."""
    m = (metric or "").strip().lower()
    v = (value or "").strip()
    if m == "oxygen_saturation":
        return (
            "If oxygen saturation remains around 90% or below 94% for a prolonged time, medical evaluation is recommended, "
            "especially if symptoms such as shortness of breath, chest pain, confusion, or bluish lips appear."
        )
    if m == "blood_pressure":
        return (
            "If blood pressure readings remain above 140/90 consistently or you experience symptoms like chest pain, "
            "severe headaches, or vision problems, a doctor should evaluate it."
        )
    if m == "temperature":
        return (
            "If fever stays above 38°C for several days or rises above 39°C, medical evaluation is recommended."
        )
    if m == "heart_rate":
        return (
            "If resting heart rate stays above 100 bpm for a prolonged period or causes dizziness or chest discomfort, consult a doctor."
        )
    if m == "glucose":
        return (
            "If blood sugar readings stay very high or very low, or you have symptoms like confusion or feeling unwell, "
            "seek medical attention."
        )
    return DOCTOR_ADVICE_RESPONSE


# --- Pydantic models ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User health education question")
    top_k: Optional[int] = Field(default=RETRIEVAL_TOP_K, ge=1, le=20, description="Number of RAG results")


class QueryResponse(BaseModel):
    query: str
    insight: dict[str, Any]
    retrieval_summary: dict[str, Any]


class InsightFromMetricsRequest(BaseModel):
    """Structured health metrics (e.g. from report parser) and optional query."""
    metrics: dict[str, Any] = Field(..., description="Structured metrics: blood_pressure, heart_rate, glucose, oxygen_saturation, temperature")
    query: Optional[str] = Field(default=None, description="Optional user question")


class ChatRequest(BaseModel):
    """User text question for /chat."""
    message: str = Field(..., min_length=1, description="User question")


class AnalyzeMetricsRequest(BaseModel):
    """Structured health metrics for /analyze-metrics."""
    metrics: dict[str, Any] = Field(..., description="Structured metrics: blood_pressure, heart_rate, glucose, oxygen_saturation, temperature")
    query: Optional[str] = Field(default=None, description="Optional user question")


# --- Constants ---
ALLOWED_REPORT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}


def _pairs_to_structured_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert NLP metric_value_pairs into the structured format expected by the insight agent."""
    out: dict[str, Any] = {
        "blood_pressure": [],
        "heart_rate": [],
        "glucose": [],
        "oxygen_saturation": [],
        "temperature": [],
    }
    for p in pairs:
        metric = (p.get("metric") or "").strip()
        value = p.get("value")
        if metric not in CHAT_METRIC_KEYS or value is None:
            continue
        if metric == "blood_pressure":
            m = re.match(r"(\d{2,3})\s*/\s*(\d{2,3})", str(value))
            if m:
                out["blood_pressure"].append({"systolic": int(m.group(1)), "diastolic": int(m.group(2))})
        elif metric == "heart_rate":
            try:
                out["heart_rate"].append(int(float(str(value))))
            except (ValueError, TypeError):
                pass
        elif metric == "glucose":
            try:
                out["glucose"].append(float(str(value)))
            except (ValueError, TypeError):
                pass
        elif metric == "oxygen_saturation":
            try:
                out["oxygen_saturation"].append(int(float(str(value))))
            except (ValueError, TypeError):
                pass
        elif metric == "temperature":
            try:
                out["temperature"].append(float(str(value)))
            except (ValueError, TypeError):
                pass
    return out


def _has_detected_metrics(structured: dict[str, Any]) -> bool:
    """True if any of the five health metrics have at least one value."""
    for key in CHAT_METRIC_KEYS:
        if structured.get(key) and len(structured[key]) > 0:
            return True
    return False


def _is_follow_up_question(query: str) -> bool:
    """True if the query looks like a follow-up (e.g. 'is that dangerous?')."""
    q = (query or "").strip().lower()
    if len(q) > 120:
        return False
    for pat in FOLLOW_UP_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return True
    return False


def _extract_bp_sequence(text: str) -> list[tuple[int, int]]:
    """Extract all blood pressure pairs (systolic/diastolic) from text. Returns list of (s, d)."""
    matches = re.findall(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", text)
    out = []
    for s, d in matches:
        si, di = int(s), int(d)
        if 70 <= si <= 250 and 40 <= di <= 150:
            out.append((si, di))
    return out


def _build_trend_structured_metrics(bp_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Build structured_metrics from list of (systolic, diastolic) for trend."""
    return {
        "blood_pressure": [{"systolic": s, "diastolic": d} for s, d in bp_pairs],
        "heart_rate": [],
        "glucose": [],
        "oxygen_saturation": [],
        "temperature": [],
    }


def _explain_bp_trend(bp_pairs: list[tuple[int, int]]) -> str:
    """Use detect_trend_timeseries on systolic values and return a short explanation."""
    if len(bp_pairs) < 2:
        return ""
    systolic_list = [p[0] for p in bp_pairs]
    result = detect_trend_timeseries(systolic_list, invert_improvement=False)
    trend = result.get("trend", "stable")
    summary = result.get("summary", {})
    first_s = summary.get("first")
    last_s = summary.get("last")
    if trend == "worsening":
        return (
            f"Your blood pressure readings show an increasing trend. The systolic pressure increased from {first_s} to {last_s} mmHg, "
            "which may indicate worsening blood pressure control. This is educational information only; please consult a healthcare provider for personal advice."
        )
    if trend == "improving":
        return (
            f"Your blood pressure readings show a decreasing trend. The systolic pressure decreased from {first_s} to {last_s} mmHg, "
            "which may indicate improving blood pressure control. This is educational information only; please consult a healthcare provider for personal advice."
        )
    return (
        f"Your blood pressure readings (systolic {first_s} to {last_s} mmHg) appear relatively stable. "
        "This is educational information only; please consult a healthcare provider for personal advice."
    )


def _suggest_follow_ups(insight: dict[str, Any], structured_metrics: dict[str, Any]) -> list[str]:
    """Suggest follow-up questions based on metrics or RAG topic. Limit to 3."""
    suggestions = []
    seen = set()
    value_to_range = insight.get("value_to_range") or []
    if value_to_range:
        for r in value_to_range[:3]:
            metric_name = (r.get("metric") or "").replace("_", " ").title()
            if "blood" in metric_name.lower() and "pressure" in metric_name.lower():
                for s in ("What causes high blood pressure?", "How can I reduce blood pressure naturally?", "What is normal blood pressure?"):
                    if s not in seen:
                        seen.add(s)
                        suggestions.append(s)
                break
        if not suggestions:
            for s in ("What is a normal range for this metric?", "When should I see a doctor?"):
                if s not in seen:
                    seen.add(s)
                    suggestions.append(s)
    if len(suggestions) < 3:
        for s in ("What is normal blood pressure?", "How can I improve my heart health?", "When should I see a doctor about my vitals?"):
            if s not in seen and len(suggestions) < 3:
                seen.add(s)
                suggestions.append(s)
    return suggestions[:3]


def _build_focused_followup_query(intent: str, metric: str, value: str) -> str:
    """Build a focused query so the agent answers the follow-up intent, not the generic explanation."""
    metric_label = metric.replace("_", " ")
    if intent == "risk_question":
        return f"What are the risks of {metric_label} {value}? Is it dangerous? Answer only about risk in a few sentences."
    if intent == "normal_range_question":
        return f"What is the normal range for {metric_label}? Is {value} normal? Answer only about normal range."
    if intent == "cause_question":
        return f"What causes {metric_label} to be at {value}? Answer only about causes."
    if intent == "improvement_question":
        return f"How can I reduce or improve {metric_label}? Answer only about lifestyle and improvement."
    return f"Follow-up: {metric_label} {value}. Provide a brief relevant explanation."


# --- Agent / pipeline instances (initialized on startup) ---
retrieval_agent: Optional[DataRetrievalAgent] = None
insight_agent: Optional[HealthInsightAgent] = None
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents and RAG pipeline on startup."""
    global retrieval_agent, insight_agent, rag_pipeline
    rag_pipeline = RAGPipeline(top_k=RAG_TOP_K)
    retrieval_agent = DataRetrievalAgent(top_k=RETRIEVAL_TOP_K)
    insight_agent = HealthInsightAgent(rag_pipeline=rag_pipeline, use_simplification=True)
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check for the API."""
    return {"status": "ok", "service": "health-education-assistant"}


# -----------------------------------------------------------------------------
# Unified endpoints: retrieval agent + insight agent
# -----------------------------------------------------------------------------


def _query_has_medical_keyword(text: str) -> bool:
    """True if any MEDICAL_KEYWORDS appears in text (case-insensitive)."""
    t = (text or "").lower()
    return any(kw in t for kw in MEDICAL_KEYWORDS)


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Accept user text questions. Uses conversation context for follow-ups, detects
    metric trends, metric analysis when values present, and RAG with LLM summarization otherwise.
    Order: 1) metric sequences (BP trend), 2) metric-value pairs, 3) follow-up, 4) non-medical guard, 5) RAG.
    """
    global last_metric_context
    if retrieval_agent is None or insight_agent is None:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    query = request.message.strip()

    # Step 1: Detect metric sequences (e.g. "120/80 130/85 140/90 150/95") before other checks
    bp_sequence = _extract_bp_sequence(query)
    if len(bp_sequence) >= 3:
        trend_explanation = _explain_bp_trend(bp_sequence)
        trend_metrics = _build_trend_structured_metrics(bp_sequence)
        last_metric_context = {
            "metric": "blood_pressure",
            "value": f"{bp_sequence[-1][0]}/{bp_sequence[-1][1]}",
            "structured_metrics": trend_metrics,
        }
        return {
            "response": trend_explanation,
            "query": query,
            "insight": {"value_to_range": [], "trend": True, "suggested_follow_ups": _suggest_follow_ups({}, trend_metrics)},
            "retrieval_summary": {"used_metric_analysis": True, "rag_results_count": 0},
        }

    # Step 2: Run NLP pipeline for metric-value pairs and semantics
    nlp_result = run_pipeline(query)
    syntax_result = nlp_result.get("syntax", {})
    semantics_result = nlp_result.get("semantics", {})
    pairs = syntax_result.get("metric_value_pairs") or []
    structured_metrics = _pairs_to_structured_metrics(pairs)

    # Step 3: If metric_value_pairs exist, update context and use metric pathway
    if _has_detected_metrics(structured_metrics):
        last_metric_context = _last_context_from_structured(structured_metrics)
        insight = insight_agent.generate_insight_from_metrics(structured_metrics, query=query)
        response_text = insight.get("explanation") or ""
        suggested = _suggest_follow_ups(insight, structured_metrics)
        insight["suggested_follow_ups"] = suggested
        return {
            "response": response_text,
            "query": query,
            "insight": insight,
            "retrieval_summary": {"rag_results_count": 0, "used_metric_analysis": True},
        }

    # Step 4: No metrics but last_metric_context exists and query is follow-up → context-aware response
    if last_metric_context and _is_follow_up_question(query):
        ctx_metric = last_metric_context.get("metric", "")
        ctx_value = last_metric_context.get("value", "")
        ctx_structured = last_metric_context.get("structured_metrics")
        if ctx_metric and ctx_value and ctx_structured:
            intent = _classify_followup(query)
            if intent == "doctor_advice_question":
                response_text = _doctor_advice_for_metric(ctx_metric, ctx_value).rstrip() + "\n\n" + MEDICAL_DISCLAIMER
                return {
                    "response": response_text,
                    "query": query,
                    "insight": {"value_to_range": [], "suggested_follow_ups": _suggest_follow_ups({}, ctx_structured)},
                    "retrieval_summary": {"rag_results_count": 0, "used_metric_analysis": True},
                }
            focused_query = _build_focused_followup_query(intent, ctx_metric, ctx_value)
            insight = insight_agent.generate_insight_from_metrics(ctx_structured, query=focused_query, follow_up_intent=intent)
            response_text = insight.get("explanation") or ""
            if not insight.get("llm_used") and intent != "general_followup":
                response_text = _followup_response_fallback(intent, ctx_metric, ctx_value)
            if "medical disclaimer" not in response_text.lower() and "consult a healthcare" not in response_text.lower():
                response_text = response_text.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
            insight["suggested_follow_ups"] = _suggest_follow_ups(insight, ctx_structured)
            return {
                "response": response_text,
                "query": query,
                "insight": insight,
                "retrieval_summary": {"rag_results_count": 0, "used_metric_analysis": True},
            }

    # Step 5: Non-medical guard — rely on semantic detection
    # If the NLP pipeline sees no medical content, treat as non-medical (even if it looks like a definition).
    if not semantics_result.get("has_medical_content", True):
        return {
            "response": NON_MEDICAL_RESPONSE,
            "query": query,
            "insight": {"value_to_range": [], "suggested_follow_ups": []},
            "retrieval_summary": {"used_metric_analysis": False, "rag_results_count": 0},
        }

    # Step 6: General medical questions — use RAG only for disease_specific_question; else LLM-only
    question_type = _classify_medical_question(query)

    if question_type == "doctor_advice_question":
        response_text = DOCTOR_ADVICE_RESPONSE.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
        return {
            "response": response_text,
            "query": query,
            "insight": {"value_to_range": [], "suggested_follow_ups": _suggest_follow_ups({}, {})},
            "retrieval_summary": {"used_metric_analysis": False, "rag_results_count": 0},
        }

    if question_type == "disease_specific_question":
        retrieval_result = retrieval_agent.retrieve(query, question_type=question_type)
        rag_results = retrieval_result.get("rag_results", [])
        combined_text = " ".join(r.get("text", "") for r in rag_results)
        condition_name = _extract_condition_name(query)
        # If retrieved context does not mention the condition name, ignore RAG and fall back to general LLM explanation.
        if condition_name and condition_name.lower() not in combined_text.lower():
            insight = insight_agent.generate_insight_general(query, question_type=question_type)
            response_text = insight.get("summary") or insight.get("explanation") or ""
            retrieval_summary = {
                "rag_results_count": 0,
                "sleep_data_included": False,
                "vital_signs_included": False,
                "physio_signals_included": False,
                "used_metric_analysis": False,
            }
        else:
            insight = insight_agent.generate_insight(query, retrieval_result)
            response_text = insight.get("summary") or insight.get("explanation") or ""
            rag_count = len(rag_results)
            retrieval_summary = {
                "rag_results_count": rag_count,
                "sleep_data_included": retrieval_result.get("sleep_data") is not None,
                "vital_signs_included": retrieval_result.get("vital_signs") is not None,
                "physio_signals_included": retrieval_result.get("physio_signals") is not None,
                "used_metric_analysis": False,
            }
    else:
        insight = insight_agent.generate_insight_general(query, question_type=question_type)
        response_text = insight.get("summary") or insight.get("explanation") or ""
        retrieval_summary = {
            "rag_results_count": 0,
            "sleep_data_included": False,
            "vital_signs_included": False,
            "physio_signals_included": False,
            "used_metric_analysis": False,
        }

    insight["suggested_follow_ups"] = _suggest_follow_ups(insight, {})
    return {
        "response": response_text,
        "query": query,
        "insight": insight,
        "retrieval_summary": retrieval_summary,
    }


def _last_context_from_structured(structured: dict[str, Any]) -> dict[str, Any]:
    """Build last_metric_context from structured_metrics for follow-up use."""
    for key in CHAT_METRIC_KEYS:
        vals = structured.get(key)
        if not vals:
            continue
        if key == "blood_pressure" and vals and isinstance(vals[0], dict):
            v = vals[0]
            return {"metric": key, "value": f"{v.get('systolic')}/{v.get('diastolic')}", "structured_metrics": structured}
        if vals:
            return {"metric": key, "value": str(vals[0]), "structured_metrics": structured}
    return {}


@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...)):
    """
    Accept uploaded medical reports (PDF or image). Extracts text/metrics, then
    calls retrieval agent and insight agent to generate an educational response.
    """
    if retrieval_agent is None or insight_agent is None:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_REPORT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_REPORT_EXTENSIONS))}",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        parse_result = parse_uploaded_report(content, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Report parsing failed: {str(e)}")
    metrics = parse_result.get("metrics", {}) or {}

    # If no metrics or lab values were detected, do not fabricate vitals.
    if not metrics:
        message = (
            "I couldn't extract clear health metrics from the uploaded report. "
            "Please upload a clearer image or PDF if possible."
        )
        return {
            "response": message,
            "report": {
                "filename": parse_result.get("filename"),
                "source": parse_result.get("source"),
                "metrics": metrics,
                "raw_text_length": parse_result.get("raw_text_length"),
            },
            "insight": {
                "explanation": message + "\n\n" + MEDICAL_DISCLAIMER,
                "metrics_summary": "",
                "value_to_range": [],
            },
            "retrieval_summary": {
                "rag_results_count": 0,
                "sleep_data_included": False,
                "vital_signs_included": False,
                "physio_signals_included": False,
            },
        }

    query = "Explain the health metrics from my uploaded report."
    retrieval_result = retrieval_agent.retrieve(query)
    retrieval_result["metrics"] = metrics
    insight = insight_agent.generate_insight(query, retrieval_result)
    response_text = insight.get("explanation") or insight.get("summary") or ""
    # If CBC metrics are present, add a brief CBC interpretation instead of inventing vitals.
    cbc_metrics = metrics.get("cbc") or {}
    if cbc_metrics:
        lines: list[str] = []
        hgb = cbc_metrics.get("hemoglobin")
        if hgb is not None:
            if 12.0 <= hgb <= 17.5:
                status = "within the typical adult range"
            elif hgb < 12.0:
                status = "slightly below the typical adult range"
            else:
                status = "slightly above the typical adult range"
            lines.append(f"Hemoglobin: {hgb} g/dL – {status}.")
        wbc = cbc_metrics.get("wbc_count")
        if wbc is not None:
            if 4000 <= wbc <= 11000:
                status = "within the usual range"
            elif wbc < 4000:
                status = "below the usual range"
            else:
                status = "slightly above the typical range"
            lines.append(f"WBC count: {int(wbc)} – {status}.")
        plt = cbc_metrics.get("platelet_count")
        if plt is not None:
            if 150000 <= plt <= 450000:
                status = "within the normal range"
            elif plt < 150000:
                status = "below the normal range"
            else:
                status = "above the normal range"
            lines.append(f"Platelet count: {int(plt)} – {status}.")
        esr = cbc_metrics.get("esr")
        if esr is not None:
            if esr <= 20:
                status = "within the typical reference range for many adults"
            else:
                status = "elevated, which can be a sign of inflammation"
            lines.append(f"ESR: {int(esr)} mm/hr – {status}.")
        if lines:
            cbc_text = "CBC interpretation:\n" + "\n".join(lines)
            if response_text:
                response_text = response_text.rstrip() + "\n\n" + cbc_text
            else:
                response_text = cbc_text + "\n\n" + MEDICAL_DISCLAIMER

    insight["suggested_follow_ups"] = _suggest_follow_ups(insight, metrics)
    retrieval_summary = {
        "rag_results_count": len(retrieval_result.get("rag_results", [])),
        "sleep_data_included": retrieval_result.get("sleep_data") is not None,
        "vital_signs_included": retrieval_result.get("vital_signs") is not None,
        "physio_signals_included": retrieval_result.get("physio_signals") is not None,
    }
    return {
        "response": response_text,
        "report": {
            "filename": parse_result.get("filename"),
            "source": parse_result.get("source"),
            "metrics": parse_result.get("metrics"),
            "raw_text_length": parse_result.get("raw_text_length"),
        },
        "insight": insight,
        "retrieval_summary": retrieval_summary,
    }


@app.post("/analyze-metrics")
def analyze_metrics(request: AnalyzeMetricsRequest):
    """
    Accept structured health metrics. Calls retrieval agent then insight agent
    to generate an educational explanation of the metrics.
    """
    if retrieval_agent is None or insight_agent is None:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    metrics = request.metrics
    query = request.query or "Explain these health metrics."
    retrieval_result = retrieval_agent.retrieve(query)
    retrieval_result["metrics"] = metrics
    insight = insight_agent.generate_insight(query, retrieval_result)
    response_text = insight.get("explanation") or insight.get("summary") or ""
    insight["suggested_follow_ups"] = _suggest_follow_ups(insight, metrics)
    retrieval_summary = {
        "rag_results_count": len(retrieval_result.get("rag_results", [])),
        "sleep_data_included": retrieval_result.get("sleep_data") is not None,
        "vital_signs_included": retrieval_result.get("vital_signs") is not None,
        "physio_signals_included": retrieval_result.get("physio_signals") is not None,
    }
    return {
        "response": response_text,
        "metrics": metrics,
        "insight": insight,
        "retrieval_summary": retrieval_summary,
    }


# -----------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main endpoint: run the multi-agent pipeline.
    1. Data Retrieval Agent fetches relevant medical knowledge and optional structured data.
    2. Health Insight Agent produces an educational insight and analysis.
    """
    if retrieval_agent is None or insight_agent is None:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    retrieval_result = retrieval_agent.retrieve(request.query, top_k=request.top_k)
    insight = insight_agent.generate_insight(request.query, retrieval_result)
    retrieval_summary = {
        "rag_results_count": len(retrieval_result.get("rag_results", [])),
        "sleep_data_included": retrieval_result.get("sleep_data") is not None,
        "vital_signs_included": retrieval_result.get("vital_signs") is not None,
        "physio_signals_included": retrieval_result.get("physio_signals") is not None,
    }
    return QueryResponse(
        query=request.query,
        insight=insight,
        retrieval_summary=retrieval_summary,
    )


@app.get("/retrieve")
def retrieve_only(q: str, top_k: int = RETRIEVAL_TOP_K):
    """Retrieve only (Data Retrieval Agent) - medical context + optional structured data."""
    if retrieval_agent is None:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    result = retrieval_agent.retrieve(q, top_k=top_k)
    return {"query": q, "retrieval": result}


@app.get("/retrieve/medical-context")
def retrieve_medical_context_only(q: str, top_k: int = RETRIEVAL_TOP_K):
    """Retrieve only medical context (FAISS + sentence-transformers) - no structured data."""
    if retrieval_agent is None:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    results = retrieval_agent.retrieve_medical_context(q, top_k=top_k)
    return {"query": q, "medical_context": results}


@app.get("/rag/context")
def rag_context(q: str, top_k: int = RAG_TOP_K):
    """
    RAG pipeline: embed query, retrieve top-k similar Q&A (by question), return answers as context.
    Uses sentence-transformers for embeddings and FAISS for vector search.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    context = rag_pipeline.get_context(q, top_k=top_k)
    entries = rag_pipeline.get_context_entries(q, top_k=top_k)
    return {"query": q, "context": context, "entries": entries}


@app.post("/insight")
def insight_from_metrics(request: InsightFromMetricsRequest):
    """
    Generate a patient-friendly, educational explanation of health metrics.
    Runs NLP pipeline, retrieves relevant medical knowledge (RAG), and generates
    explanation with LLaMA-3-8B-Instruct. Response is educational, avoids diagnosis,
    and includes a medical disclaimer.
    """
    if insight_agent is None:
        raise HTTPException(status_code=503, detail="Insight agent not initialized")
    return insight_agent.generate_insight_from_metrics(request.metrics, query=request.query)


@app.post("/report/parse")
async def report_parse(file: UploadFile = File(...)):
    """
    Upload a medical report (PDF or image). Text is extracted via OCR, then medical
    metrics (blood pressure, heart rate, glucose, oxygen saturation, temperature)
    are detected with regex. Returns structured JSON.
    """
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_REPORT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_REPORT_EXTENSIONS))}",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        result = parse_uploaded_report(content, filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Report parsing failed: {str(e)}")


@app.get("/datasets")
def list_datasets():
    """Return summary of loaded datasets (from processed folder)."""
    data = load_all_datasets()
    return {
        "medical_knowledge": {"type": "list", "count": len(data["medical_knowledge"])},
        "physio_signals": {"type": "dataframe", "rows": len(data["physio_signals"])},
        "sleep_data": {"type": "dataframe", "rows": len(data["sleep_data"])},
        "vital_signs": {"type": "dataframe", "rows": len(data["vital_signs"])},
        "simplification_pairs": {"type": "dataframe", "rows": len(data["simplification_pairs"])},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
