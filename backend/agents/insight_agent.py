"""Health Insight Agent: structured metrics → NLP → RAG → LLaMA-3-8B patient-friendly explanations."""

import json
from typing import Any, Optional

from backend.config import (
    INSIGHT_LLM_MODEL,
    MEDICAL_DISCLAIMER,
    OLLAMA_BASE_URL,
    RAG_TOP_K,
)
from backend.nlp_pipeline import (
    run_pipeline,
    map_value_to_medical_range,
    ensure_educational_response,
)

# Patient-facing metric guidelines: normal ranges and short explanations (3–5 sentences max)
METRIC_GUIDELINES = {
    "blood_pressure": {
        "normal": "around 120/80 mmHg",
        "ranges": "Normal: ~120/80. Elevated: 120–129 systolic. Hypertension: ≥130/80.",
        "cause_hint": "Can be influenced by diet, stress, activity, and genetics.",
    },
    "heart_rate": {
        "normal": "60–100 bpm at rest",
        "ranges": "Normal resting: 60–100 bpm.",
        "cause_hint": "High heart rate may be due to exercise, stress, fever, dehydration, caffeine, anxiety, or heart rhythm conditions.",
    },
    "oxygen_saturation": {
        "normal": "95–100%",
        "ranges": "Normal: 95–100%. Below 94% may indicate breathing problems or lung disease.",
        "cause_hint": "Low levels may need evaluation by a healthcare professional.",
    },
    "temperature": {
        "normal": "36.5–37.5°C (97.7–99.5°F)",
        "ranges": "Normal: 36.5–37.5°C. Fever: ≥38°C, often due to infection or inflammation.",
        "cause_hint": "Fever is usually caused by infection or inflammation.",
    },
    "glucose": {
        "normal": "fasting <100 mg/dL",
        "ranges": "Normal fasting: <100 mg/dL. Prediabetes: 100–125. Diabetes: ≥126.",
        "cause_hint": "Affected by diet, activity, and how the body uses insulin.",
    },
}


# Structured metrics format (e.g. from report parser): same keys as detect_medical_metrics
STRUCTURED_METRICS_KEYS = (
    "blood_pressure",
    "heart_rate",
    "glucose",
    "oxygen_saturation",
    "temperature",
)


def _metrics_to_description(metrics: dict[str, Any]) -> str:
    """Turn structured metrics into a short text for NLP and RAG queries."""
    parts = []
    for key in STRUCTURED_METRICS_KEYS:
        vals = metrics.get(key)
        if not vals:
            continue
        if key == "blood_pressure":
            for v in vals[:3]:
                if isinstance(v, dict):
                    parts.append(f"blood pressure {v.get('systolic')}/{v.get('diastolic')}")
        else:
            for v in (vals if isinstance(vals, list) else [vals])[:3]:
                parts.append(f"{key.replace('_', ' ')} {v}")
    return ". ".join(parts) if parts else ""


def _metrics_to_range_labels(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Map metric values to medical range labels (e.g. 150/95 → elevated blood pressure)."""
    out = []
    for key in STRUCTURED_METRICS_KEYS:
        vals = metrics.get(key)
        if not vals:
            continue
        for v in (vals if isinstance(vals, list) else [vals]):
            if key == "blood_pressure" and isinstance(v, dict):
                s, d = v.get("systolic"), v.get("diastolic")
                if s is None or d is None:
                    continue
                value_str = f"{s}/{d}"
            else:
                value_str = str(v)
            label = map_value_to_medical_range(key, value_str)
            if label:
                out.append({"metric": key, "value": value_str, "medical_range": label})
    return out


def _generate_health_summary(range_labels: list[dict[str, Any]]) -> Optional[str]:
    """
    Generate a short summary when multiple metrics appear, highlighting that some
    readings are outside the typical range.
    """
    if not range_labels:
        return None
    abnormal = [r for r in range_labels if "normal" not in str(r.get("medical_range", "")).lower()]
    if len(abnormal) >= 2:
        return (
            "Some of your health readings are outside the typical range, which may indicate "
            "that your body is under stress or dealing with a health issue."
        )
    return None


def _metric_explanation_sentence(metric: str, value: str, medical_range: str) -> str:
    """One conversational sentence for this metric value (no 'categorized as')."""
    guidelines = METRIC_GUIDELINES.get(metric, {})
    normal = guidelines.get("normal", "typical range")
    name = metric.replace("_", " ").title()
    if "blood pressure" in name.lower():
        return f"A blood pressure reading of {value} is {medical_range.lower()}. Normal is typically {normal}."
    if "heart rate" in name.lower():
        return f"A heart rate of {value} bpm is {medical_range.lower()}. Normal resting is usually {normal}."
    if "oxygen" in name.lower():
        return f"An oxygen saturation of {value}% is {medical_range.lower()}. Normal levels are usually {normal}."
    if "temperature" in name.lower():
        return f"A temperature of {value}°C is {medical_range.lower()}. Normal is typically {normal}."
    if "glucose" in name.lower():
        return f"A glucose reading of {value} mg/dL is {medical_range.lower()}. Normal fasting is {normal}."
    return f"Your {name} reading of {value} is {medical_range.lower()}. Consult a healthcare provider for context."


def _build_rag_queries_from_metrics(metrics: dict[str, Any]) -> list[str]:
    """Build search queries for RAG from present metrics."""
    q = []
    if metrics.get("blood_pressure"):
        q.append("What is blood pressure and what do the numbers mean?")
    if metrics.get("heart_rate"):
        q.append("What is heart rate and normal range?")
    if metrics.get("glucose"):
        q.append("What is blood glucose and normal levels?")
    if metrics.get("oxygen_saturation"):
        q.append("What is oxygen saturation and normal range?")
    if metrics.get("temperature"):
        q.append("What is body temperature and normal range?")
    return q if q else ["general health metrics"]


def _call_ollama(prompt: str, base_url: str, model: str, timeout: int = 120) -> Optional[str]:
    """Generate text using Ollama (e.g. LLaMA-3-8B-Instruct). Returns None if unavailable."""
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return (data.get("response") or "").strip()
    except Exception:
        return None


def _build_insight_prompt(
    metrics_description: str,
    range_labels: list[dict],
    rag_context: str,
    user_query: Optional[str] = None,
) -> str:
    """Build system + user prompt for educational, patient-friendly explanation."""
    instructions = (
        "You are a health education assistant. Explain health metrics in simple, conversational language. "
        "Do NOT diagnose or recommend specific treatments. Do NOT use rigid phrases like 'Here is a brief educational overview'. "
        "Write as if talking to a patient: clear, supportive, and natural. "
        "Stick to what each metric means and typical ranges. End with a reminder to consult a healthcare provider."
    )
    parts = [
        "[Instruction]",
        instructions,
        "",
        "[Health metrics to explain]",
        metrics_description,
        "",
    ]
    if range_labels:
        parts.append("[How these values are often categorized (for context only)]")
        for r in range_labels:
            parts.append(f"- {r['metric']}: {r['value']} → {r['medical_range']}")
        parts.append("")
    if rag_context:
        parts.append("[Relevant educational context from our knowledge base]")
        parts.append(rag_context[:3000])
        parts.append("")
    parts.append(
        "Using the above, write a short, educational explanation (3–5 sentences). Use conversational language; "
        "avoid phrases like 'Metric of X is categorized as'. Explain each metric that appears. "
        "Do not diagnose or prescribe. End with one sentence: this is for education only; consult a healthcare provider."
    )
    if user_query:
        parts.append("")
        parts.append(f"[User question: {user_query}]")
    return "\n".join(parts)


def _build_general_question_prompt(query: str, question_type: str) -> str:
    """Prompt for general medical questions (no RAG). Assistant MUST answer with a short educational explanation."""
    type_guidance = {
        "definition_question": (
            "Provide a simple educational explanation in 2–4 sentences. "
            "Explain what the condition or term means and how it affects the body. "
            "Use clear, patient-friendly language. Do NOT refuse—definition questions must always be answered. "
            "Avoid rare diseases unless the user explicitly asked about one. "
            "Do not add a disclaimer; it will be added automatically."
        ),
        "cause_question": "Explain common causes in 3–5 sentences. Include lifestyle and general factors (e.g. exercise, stress, diet, caffeine). Do not focus on rare diseases unless asked.",
        "lifestyle_question": "Give practical lifestyle advice in 3–5 sentences. Supportive and clear.",
        "risk_question": "Explain when to be concerned in 3–5 sentences. Do not diagnose. Use patient-friendly language.",
        "metric_question": "Explain normal ranges and what the numbers mean in 3–5 sentences.",
    }
    guidance = type_guidance.get(question_type, "Answer in 3–5 sentences. Use simple, patient-friendly language.")
    if question_type == "lifestyle_question":
        q_lower = (query or "").lower()
        if any(k in q_lower for k in ("blood pressure", "bp", "heart", "cholesterol")):
            guidance += (
                " For questions about heart health or blood pressure, mention reducing salt intake, maintaining a healthy weight, "
                "regular aerobic exercise, limiting alcohol, and avoiding smoking."
            )
        elif any(k in q_lower for k in ("lung", "lungs", "breathing")):
            guidance += (
                " For lung health, mention avoiding smoking, doing breathing exercises, staying physically active, "
                "reducing exposure to air pollution, and preventing respiratory infections (e.g. vaccines, hand hygiene)."
            )
        elif any(k in q_lower for k in ("blood sugar", "glucose", "sugar")):
            guidance += (
                " For blood sugar, mention a balanced diet with controlled carbohydrates, regular physical activity, "
                "maintaining a healthy weight, and regular blood sugar monitoring as advised by a healthcare provider."
            )
    if question_type == "definition_question":
        return (
            "You are a health education assistant. The user asked for a definition or explanation of a medical term or condition.\n\n"
            "You MUST answer with a clear educational explanation. Do NOT refuse. Do NOT say to ask a healthcare provider or to look elsewhere.\n\n"
            "RULES:\n"
            "1. Always provide a clear educational explanation in 2–4 sentences.\n"
            "2. Use simple, patient-friendly language. Explain what the condition or term is and how it affects the body.\n"
            "3. Do NOT refuse to answer. Definition questions must always receive an explanation.\n"
            "4. Do NOT redirect to a healthcare provider in your answer. The disclaimer will be added automatically at the end.\n"
            "5. Do NOT add a disclaimer or 'consult a doctor' in your text—it will be appended after.\n"
            "6. Only refuse if the user explicitly asks for a diagnosis or a treatment decision for a specific person. Questions like 'What is X?', 'What are Y?', 'Explain Z' MUST be answered with an explanation.\n\n"
            f"User question: {query}\n\n"
            "Write your answer now (2–4 sentences only; clear explanation; no refusal, no redirect, no disclaimer):"
        )
    return (
        "You are a health education assistant. Your job is to ANSWER general health education questions with a short, clear explanation.\n\n"
        "RULES:\n"
        "1. You MUST answer the question. Do NOT refuse or say to ask a healthcare provider instead. Provide the educational explanation.\n"
        "2. Write 3–5 sentences only. Short, clear, and patient-friendly.\n"
        "3. Use simple language. No jargon. Write as if talking to a patient.\n"
        "4. Do NOT add a disclaimer or 'consult a doctor' in your answer. A disclaimer will be added automatically after your text.\n"
        "5. Only refuse to answer if the question explicitly asks for a diagnosis or a specific treatment decision for a real person. General education questions (what is X, what causes Y, how to improve Z) MUST be answered.\n\n"
        f"Question type: {question_type}\n"
        f"Guidance for this type: {guidance}\n\n"
        f"User question: {query}\n\n"
        "Write your answer now (3–5 sentences, educational and direct. No refusal, no disclaimer):"
    )


def _extract_definition_term(query: str) -> str:
    """Extract the medical term from a definition-style query (e.g. 'What is pneumonia?' -> 'Pneumonia')."""
    if not query or not query.strip():
        return "This condition"
    q = query.strip().lower()
    for prefix in ("what is ", "what are ", "define ", "explain "):
        if q.startswith(prefix):
            q = q[len(prefix) :].strip()
            break
    q = q.rstrip("?.").strip()
    if not q:
        return "This condition"
    return q.title()


def _fallback_general_answer(query: str, question_type: str) -> str:
    """Short educational fallback when LLM is unavailable. Never refuse; always give a useful answer."""
    q = (query or "").strip().lower()
    is_definition = question_type == "definition_question" or "what is" in q or "what are" in q or "define" in q or "explain" in q
    # Definition-style fallbacks (for "What is X?", "Explain X", etc.)
    if is_definition and "bilirubin" in q:
        return (
            "Bilirubin is a yellow substance produced when red blood cells break down. "
            "The liver processes bilirubin and removes it through bile. "
            "Blood tests measure bilirubin levels to help evaluate liver function."
        )
    if is_definition and ("wbc" in q or "white blood cell" in q):
        return (
            "WBC stands for white blood cells. "
            "These cells are part of the immune system and help the body fight infections."
        )
    if is_definition and ("rbc" in q or "red blood cell" in q):
        return (
            "RBC stands for red blood cells. "
            "These cells carry oxygen from the lungs to the rest of the body."
        )
    if is_definition and "hemoglobin" in q:
        return (
            "Hemoglobin is a protein inside red blood cells that carries oxygen throughout the body. "
            "It helps deliver oxygen to tissues and organs so they can function properly."
        )
    if is_definition and ("platelet" in q or "platelets" in q):
        return (
            "Platelets are small blood cell fragments that help the blood clot and stop bleeding. "
            "They gather at sites of injury to form a plug and support healing."
        )
    if is_definition and "esr" in q:
        return (
            "ESR, or erythrocyte sedimentation rate, is a blood test that measures how quickly red blood cells settle in a tube. "
            "It is used as a general marker of inflammation in the body."
        )
    if is_definition and "creatinine" in q:
        return (
            "Creatinine is a waste product produced by muscles and removed from the body by the kidneys. "
            "Blood creatinine levels are often measured to help evaluate kidney function."
        )
    if is_definition and "cholesterol" in q:
        return (
            "Cholesterol is a type of fat found in the blood that the body uses to build cells and make certain hormones. "
            "High levels of cholesterol may increase the risk of heart disease."
        )
    if is_definition and "asthma" in q:
        return (
            "Asthma is a chronic condition that affects the airways in the lungs. "
            "In people with asthma, the airways become inflamed and narrow, making it harder to breathe. "
            "This can cause symptoms such as wheezing, coughing, shortness of breath, and chest tightness."
        )
    if is_definition and ("hypertension" in q or ("high" in q and "blood pressure" in q and ("what is" in q or "explain" in q or "define" in q))):
        return (
            "Hypertension, or high blood pressure, is a condition where the force of blood against the artery walls is consistently too high. "
            "Over time this can increase the risk of heart disease and stroke."
        )
    if is_definition and ("allerg" in q or "allergies" in q):
        return (
            "Allergies occur when the immune system overreacts to a normally harmless substance, such as pollen, pet dander, or certain foods. "
            "This can cause symptoms like sneezing, itching, runny nose, or in some cases a more serious reaction. "
            "Avoiding the trigger and sometimes medication can help manage allergies."
        )
    if is_definition and "pneumonia" in q:
        return (
            "Pneumonia is an infection that inflames the air sacs in one or both lungs. "
            "The air sacs may fill with fluid or pus, causing cough, fever, chills, and difficulty breathing. "
            "It can be caused by bacteria, viruses, or other germs."
        )
    if is_definition and "diabetes" in q:
        return (
            "Diabetes is a condition where the body has difficulty controlling blood sugar levels. "
            "This happens when the body does not produce enough insulin or cannot use insulin effectively. "
            "Over time, high blood sugar can affect the heart, kidneys, nerves, and eyes. "
            "Healthy eating, physical activity, and medication can help manage diabetes."
        )
    if is_definition and ("hypertrophic" in q and "cardiomyopathy" in q):
        return (
            "Hypertrophic cardiomyopathy is a condition where the heart muscle becomes abnormally thick. "
            "This thickening can make it harder for the heart to pump blood effectively. "
            "Some people may experience symptoms such as chest pain, shortness of breath, dizziness, or fainting."
        )
    if is_definition and ("deep vein" in q and "thrombosis" in q):
        return (
            "Deep vein thrombosis is a condition where a blood clot forms in a deep vein, usually in the legs. "
            "The clot can block blood flow and may cause swelling or pain. "
            "It is important to seek medical attention if you suspect a blood clot."
        )
    if "diabetes" in q:
        return (
            "Diabetes is a condition where the body has difficulty controlling blood sugar levels. "
            "This happens when the body does not produce enough insulin or cannot use insulin effectively. "
            "Over time, high blood sugar can affect the heart, kidneys, nerves, and eyes. "
            "Healthy eating, physical activity, and medication can help manage diabetes."
        )
    if "heart rate" in q and ("cause" in q or "high" in q):
        return (
            "A high heart rate can occur for several reasons, including exercise, stress, dehydration, fever, or caffeine. "
            "It can also happen with anxiety or certain medications. "
            "In some cases, heart rhythm conditions may cause a faster heart rate."
        )
    if "heart health" in q or "improve" in q and "heart" in q:
        return (
            "You can improve heart health by exercising regularly, eating a balanced diet, reducing salt and processed foods, "
            "avoiding smoking, and managing stress. "
            "Regular checkups and monitoring blood pressure and cholesterol are also important."
        )
    if "blood pressure" in q and ("cause" in q or "high" in q):
        return (
            "High blood pressure can be caused by high salt intake, lack of exercise, obesity, stress, smoking, and family history. "
            "In some cases it may be related to underlying medical conditions. "
            "Lifestyle changes and medication can help manage it."
        )
    if ("period" in q or "menstrual" in q or "cramps" in q) and ("cause" in q or "cramp" in q):
        return (
            "Period cramps occur when the uterus contracts to shed its lining during menstruation. "
            "These contractions are triggered by hormone-like substances called prostaglandins. "
            "Higher prostaglandin levels can cause stronger cramps."
        )
    if ("common cold" in q or "cold" in q) and "cause" in q:
        return (
            "The common cold is caused by viruses that infect the upper respiratory tract, most commonly rhinoviruses. "
            "It spreads through respiratory droplets or contact with contaminated surfaces."
        )
    if "headache" in q and "cause" in q:
        return (
            "Headaches can have many causes, including tension, dehydration, lack of sleep, stress, or eye strain. "
            "Migraines may be triggered by certain foods, hormones, or environmental factors. "
            "If headaches are severe or frequent, a healthcare provider can help identify the cause."
        )
    if ("stomach" in q or "abdominal" in q) and "pain" in q and "cause" in q:
        return (
            "Stomach pain can be caused by many things, such as indigestion, gas, constipation, or viral infections. "
            "Food choices, stress, and eating habits often play a role. "
            "Persistent or severe pain should be evaluated by a healthcare provider."
        )
    if question_type == "cause_question":
        return "Common causes include lifestyle factors like diet, exercise, stress, and sleep. A healthcare provider can help identify causes relevant to you."
    if question_type == "lifestyle_question":
        return (
            "Helpful lifestyle steps often include regular physical activity, a balanced diet with limited salt and processed foods, "
            "getting enough sleep, managing stress, avoiding smoking, and attending routine medical checkups."
        )
    if is_definition:
        term = _extract_definition_term(query or "")
        # Pattern-based generic explanations for common medical categories
        if "deficiency" in q:
            return (
                f"{term} occurs when the body does not have enough of an important nutrient or substance. "
                "Deficiencies can affect normal body functions and may lead to symptoms such as fatigue, weakness, or other health problems "
                "depending on which nutrient is involved."
            )
        if "syndrome" in q:
            return (
                f"{term} is a medical syndrome that involves a group of symptoms or physical changes that occur together. "
                "Syndromes often affect multiple parts of the body and may be related to genetic or developmental factors."
            )
        if "infection" in q:
            return (
                f"{term} is an infection caused by microorganisms such as bacteria, viruses, or fungi. "
                "These infections can trigger inflammation and symptoms that depend on the part of the body that is affected."
            )
        if "disease" in q or "condition" in q:
            return (
                f"{term} is a health condition that affects how certain parts of the body function. "
                "Symptoms and severity can vary from person to person depending on which organs or systems are involved."
            )
        # Generic definition fallback when we don't have more specific pattern matches
        return (
            f"{term} is a medical condition or health-related concept that affects the body. "
            "It can influence how certain organs or systems function and may cause different symptoms from person to person. "
            "A healthcare professional can provide proper diagnosis and guidance on management."
        )
    return (
        "This is a general health education question. For the most accurate information tailored to you, "
        "consider discussing with a healthcare provider."
    )


class HealthInsightAgent:
    """
    Health Insight Agent responsibilities:
    1. Accept structured health metrics (e.g. from report parser).
    2. Run NLP pipeline stages on the metrics description and query.
    3. Retrieve relevant medical knowledge using RAG.
    4. Generate patient-friendly explanations using LLaMA-3-8B-Instruct (via Ollama).

    Responses are educational, avoid diagnosis, and include a medical disclaimer.
    """

    def __init__(
        self,
        rag_pipeline: Optional[Any] = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        llm_model: str = INSIGHT_LLM_MODEL,
        use_simplification: bool = True,
    ):
        self.rag_pipeline = rag_pipeline
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.use_simplification = use_simplification

    def set_rag_pipeline(self, rag_pipeline: Any) -> None:
        self.rag_pipeline = rag_pipeline

    def generate_insight_general(
        self,
        query: str,
        question_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate an educational answer for general medical questions without RAG.
        Used for definition, cause, lifestyle, risk, and metric questions so we do not
        pull rare-disease dataset entries. Answer is LLM-only, 3–5 sentences.
        """
        prompt = _build_general_question_prompt(query or "", question_type or "definition_question")
        out = _call_ollama(prompt, self.ollama_base_url, self.llm_model)
        if out:
            out, _ = ensure_educational_response(out)
            # If the model clearly refused to answer, use fallback instead
            out_lower = out.lower()
            if (
                not out
                or "i cannot answer" in out_lower
                or "i can't answer" in out_lower
                or "not able to answer" in out_lower
                or "outside my scope" in out_lower
            ):
                out = _fallback_general_answer(query or "", question_type or "definition_question")
            if MEDICAL_DISCLAIMER not in out:
                out = out.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
        else:
            out = _fallback_general_answer(query or "", question_type or "definition_question")
            out = out.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
        return {
            "summary": out,
            "explanation": out,
            "value_to_range": [],
            "sources": [],
        }

    def generate_insight_from_metrics(
        self,
        structured_metrics: dict[str, Any],
        query: Optional[str] = None,
        rag_top_k: int = RAG_TOP_K,
        follow_up_intent: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Accept structured health metrics, run NLP, retrieve RAG context, generate
        patient-friendly explanation with LLaMA-3-8B-Instruct. Response is educational,
        avoids diagnosis, and includes a medical disclaimer.
        """
        metrics = structured_metrics if isinstance(structured_metrics, dict) else {}
        if not metrics:
            return self._empty_insight("No health metrics provided.")

        # 1) Text description of metrics for NLP
        metrics_desc = _metrics_to_description(metrics)
        # 2) Run NLP pipeline
        nlp_input = (query or "") + " " + metrics_desc
        nlp_input = nlp_input.strip() or "health metrics"
        nlp_result = run_pipeline(nlp_input)

        # 3) Map values to medical ranges (for prompt and semantics)
        range_labels = _metrics_to_range_labels(metrics)
        summary_line = _generate_health_summary(range_labels)

        # 4) RAG: retrieve relevant medical knowledge
        rag_context_parts = []
        if self.rag_pipeline:
            for q in _build_rag_queries_from_metrics(metrics):
                ctx = self.rag_pipeline.get_context(q, top_k=rag_top_k)
                if ctx:
                    rag_context_parts.append(ctx)
        rag_context = "\n\n---\n\n".join(rag_context_parts) if rag_context_parts else ""

        # 5) Generate explanation with LLaMA via Ollama
        prompt = _build_insight_prompt(metrics_desc, range_labels, rag_context, query)
        llm_response = _call_ollama(prompt, self.ollama_base_url, self.llm_model)

        if llm_response:
            final_text, pragmatics = ensure_educational_response(llm_response)
            if pragmatics.get("suggest_add_disclaimer"):
                final_text = final_text.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
            else:
                if MEDICAL_DISCLAIMER not in final_text:
                    final_text = final_text.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
            explanation = final_text
            disclaimer_appended = True
        else:
            explanation = self._fallback_explanation(metrics, range_labels, follow_up_intent=follow_up_intent)
            explanation = explanation.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
            disclaimer_appended = True

        if summary_line:
            # Prepend a short health summary ahead of the detailed explanation.
            explanation = summary_line + "\n\n" + explanation

        return {
            "explanation": explanation,
            "nlp_analysis": nlp_result,
            "metrics_summary": _metrics_to_description(metrics),
            "value_to_range": range_labels,
            "rag_context_used": bool(rag_context),
            "disclaimer_appended": disclaimer_appended,
            "llm_used": bool(llm_response),
        }

    def _fallback_explanation(
        self,
        metrics: dict,
        range_labels: list,
        follow_up_intent: Optional[str] = None,
    ) -> str:
        """Conversational educational explanation when LLM is unavailable. Patient-friendly, 3–5 sentences."""
        if follow_up_intent and follow_up_intent != "general_followup":
            return self._fallback_followup_by_intent(follow_up_intent, range_labels)
        parts = []
        for r in range_labels:
            sent = _metric_explanation_sentence(
                r["metric"], r["value"], r.get("medical_range", "")
            )
            parts.append(sent + " ")
        if not parts:
            return "I don't have enough information to explain these metrics. Please consult a healthcare provider for personal advice."
        guidelines = METRIC_GUIDELINES.get(range_labels[0]["metric"], {})
        if guidelines.get("cause_hint"):
            parts.append(guidelines["cause_hint"] + " ")
        return ("".join(parts).strip() + " Consult a healthcare provider for personal advice.").strip()

    def _fallback_followup_by_intent(self, intent: str, range_labels: list) -> str:
        """Short intent-specific fallback for follow-up when LLM is unavailable. Metric-aware."""
        if not range_labels:
            return "Consult a healthcare provider for personal advice."
        r = range_labels[0]
        metric = r["metric"]
        metric_name = metric.replace("_", " ").title()
        value = r["value"]
        if intent == "risk_question":
            if metric == "oxygen_saturation":
                return (
                    f"An oxygen saturation of {value}% is below the normal range. "
                    "Normal oxygen levels are typically between 95% and 100%. "
                    "If oxygen levels stay low or symptoms such as shortness of breath appear, medical evaluation is recommended."
                )
            if metric == "blood_pressure":
                return (
                    f"A blood pressure reading of {value} may increase the risk of heart disease or stroke "
                    "if it stays high over time. A single reading does not always indicate a medical condition. "
                    "Consult a healthcare provider for personal advice."
                )
            return (
                f"A {metric_name} reading of {value} may need attention if it stays outside the normal range. "
                "Consult a healthcare provider for personal advice."
            )
        if intent == "normal_range_question":
            guidelines = METRIC_GUIDELINES.get(metric, {})
            normal = guidelines.get("normal", "see a healthcare provider for normal range")
            return (
                f"Normal for {metric_name} is typically {normal}. "
                f"Your reading of {value} can be interpreted in context by a healthcare provider."
            )
        if intent == "cause_question":
            guidelines = METRIC_GUIDELINES.get(metric, {})
            hint = guidelines.get("cause_hint", "A healthcare provider can help identify causes relevant to you.")
            return f"Common factors that can affect {metric_name} include diet, activity, stress, and genetics. {hint}"
        if intent == "improvement_question":
            return (
                f"Lifestyle changes like diet, exercise, and stress management often help with {metric_name}. "
                "Consult a healthcare provider for personalized advice."
            )
        return f"Your {metric_name} reading is {value}. Consult a healthcare provider for personal advice."

    def _empty_insight(self, reason: str) -> dict[str, Any]:
        return {
            "explanation": reason + " " + MEDICAL_DISCLAIMER,
            "nlp_analysis": {},
            "metrics_summary": "",
            "value_to_range": [],
            "rag_context_used": False,
            "disclaimer_appended": True,
            "llm_used": False,
        }

    def generate_insight(self, query: str, retrieval_result: dict[str, Any]) -> dict[str, Any]:
        """
        Generate insight from a text query and pre-retrieved RAG result (existing API).
        Optionally uses structured_metrics if present in retrieval_result.
        When RAG context exists, uses LLM to produce a natural explanation instead of raw Q&A blocks.
        """
        structured = retrieval_result.get("metrics") or retrieval_result.get("structured_metrics")
        if structured:
            return self.generate_insight_from_metrics(structured, query=query)

        rag_results = retrieval_result.get("rag_results", [])
        context_parts = [r["text"] for r in rag_results[:3]]
        context = "\n\n".join(context_parts) if context_parts else ""
        nlp = run_pipeline(query)

        if context:
            summary = self._summarize_rag_with_llm(query, context)
        else:
            summary = "I couldn't find specific information in the knowledge base for your question. Consider rephrasing or asking about conditions or general health topics."

        summary, _ = ensure_educational_response(summary)
        if MEDICAL_DISCLAIMER not in summary:
            summary = summary.rstrip() + "\n\n" + MEDICAL_DISCLAIMER
        return {
            "summary": summary,
            "explanation": summary,
            "sources": [r.get("metadata", {}) for r in rag_results[:5]],
            "structured_data_summary": self._summarize_structured(retrieval_result),
            "nlp_analysis": nlp,
            "query_analysis": {
                "is_question": nlp["pragmatics"]["is_question"],
                "has_medical_content": nlp["semantics"]["has_medical_content"],
            },
            "value_to_range": [],
        }

    def _summarize_rag_with_llm(self, query: str, context: str) -> str:
        """Use LLM to turn RAG context into a simple, natural explanation. Never return raw Q&A."""
        prompt = (
            "You are a medical education assistant. "
            "Using the provided medical information, answer the user's question in simple language. "
            "Do not list Q/A pairs. Do not show dataset entries. "
            "Provide a concise explanation suitable for a patient.\n\n"
            f"User question: {query}\n\n"
            "Medical information:\n"
            f"{context[:3500]}\n\n"
            "Your answer (concise, patient-friendly paragraph):"
        )
        out = _call_ollama(prompt, self.ollama_base_url, self.llm_model)
        if out:
            return out.strip()
        return self._summarize_context_fallback(context)

    def _summarize_context_fallback(self, context: str) -> str:
        """
        Fallback when LLM unavailable: remove lines starting with Q:/A:, return only
        explanatory text, limit to a few sentences. Never return raw dataset Q&A.
        """
        if not context or not context.strip():
            return "I couldn't find specific information in the knowledge base for your question. Consider rephrasing or asking about conditions or general health topics."
        import re
        lines = context.split("\n")
        kept = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^Q:\s*", stripped, re.IGNORECASE):
                continue
            if re.match(r"^A:\s*", stripped, re.IGNORECASE):
                stripped = re.sub(r"^A:\s*", "", stripped, count=1, flags=re.IGNORECASE)
            if stripped and not stripped.startswith("Q:"):
                kept.append(stripped)
        text = " ".join(kept)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "I couldn't find specific information in the knowledge base for your question. Consider rephrasing or asking about conditions or general health topics."
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = " ".join(sentences[:4]).strip()
        if len(result) > 800:
            result = result[:797] + "..."
        return result

    def _summarize_context(self, query: str, context: str) -> str:
        if not context:
            return "I couldn't find specific information in the knowledge base for your question. Consider rephrasing or asking about conditions or general health topics."
        return f"Based on the available health education content:\n\n{context[:1500]}" + ("..." if len(context) > 1500 else "")

    def _summarize_structured(self, retrieval_result: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key in ("sleep_data", "vital_signs", "physio_signals"):
            data = retrieval_result.get(key)
            out[f"{key}_available"] = bool(data and isinstance(data, list) and len(data) > 0)
            if out[f"{key}_available"]:
                out[f"{key}_count"] = len(data)
        return out
