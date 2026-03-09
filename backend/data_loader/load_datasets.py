"""Dataset loader: load from datasets/processed and return structured data for the retrieval agent."""

import json
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from backend.config import (
    MEDICAL_KNOWLEDGE_PATH,
    PHYSIO_SIGNALS_PATH,
    SLEEP_DATA_PATH,
    VITAL_SIGNS_PATH,
    SIMPLIFICATION_PAIRS_PATH,
)


# --- Structured data types for retrieval agent ---


@dataclass(frozen=True)
class MedicalKnowledgeEntry:
    """Single Q&A entry from the medical knowledge base."""
    question: str
    answer: str


@dataclass
class VitalSignsRecord:
    """Single vital signs record (patient_id, time, heart_rate, blood_pressure, temp, spo2)."""
    patient_id: Any
    time: Optional[float] = None
    heart_rate: Optional[float] = None
    blood_pressure: Optional[str] = None
    temp: Optional[float] = None
    spo2: Optional[float] = None


@dataclass
class SleepDataRecord:
    """Single sleep record (patient_id, sleep_duration, oxygen_saturation)."""
    patient_id: Any
    sleep_duration: Optional[float] = None
    oxygen_saturation: Optional[float] = None


@dataclass
class PhysiologicalSignalsRecord:
    """Single physiological/signals record (age, sex, bmi, hypertension, diabetes, hemoglobin, glucose)."""
    age: Optional[float] = None
    sex: Optional[str] = None
    bmi: Optional[float] = None
    hypertension: Optional[int] = None
    diabetes: Optional[int] = None
    hemoglobin: Optional[float] = None
    glucose: Optional[float] = None


@dataclass
class SimplificationPair:
    """Expert-to-simple text pair for simplification."""
    expert: str
    simple: Optional[str] = None


# --- Loaders ---


def load_medical_knowledge() -> list[MedicalKnowledgeEntry]:
    """Load medical Q&A knowledge base from medical_knowledge.json. Returns structured entries."""
    if not MEDICAL_KNOWLEDGE_PATH.exists():
        return []
    with open(MEDICAL_KNOWLEDGE_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        MedicalKnowledgeEntry(question=item.get("question", ""), answer=item.get("answer", ""))
        for item in (raw if isinstance(raw, list) else [raw])
    ]


def load_vital_signs() -> list[VitalSignsRecord]:
    """Load vital signs from vital_signs.csv with pandas. Returns list of VitalSignsRecord."""
    if not VITAL_SIGNS_PATH.exists():
        return []
    df = pd.read_csv(VITAL_SIGNS_PATH)
    records: list[VitalSignsRecord] = []
    for _, row in df.iterrows():
        records.append(
            VitalSignsRecord(
                patient_id=row.get("patient_id"),
                time=_safe_float(row.get("time")),
                heart_rate=_safe_float(row.get("heart_rate")),
                blood_pressure=_safe_str(row.get("blood_pressure")),
                temp=_safe_float(row.get("temp")),
                spo2=_safe_float(row.get("spo2")),
            )
        )
    return records


def load_sleep_data() -> list[SleepDataRecord]:
    """Load sleep data from sleep_data.csv with pandas. Returns list of SleepDataRecord."""
    if not SLEEP_DATA_PATH.exists():
        return []
    df = pd.read_csv(SLEEP_DATA_PATH)
    return [
        SleepDataRecord(
            patient_id=row.get("patient_id"),
            sleep_duration=_safe_float(row.get("sleep_duration")),
            oxygen_saturation=_safe_float(row.get("oxygen_saturation")),
        )
        for _, row in df.iterrows()
    ]


def load_physiological_signals() -> list[PhysiologicalSignalsRecord]:
    """Load physiological signals from physio_signals.csv with pandas. Returns list of PhysiologicalSignalsRecord."""
    if not PHYSIO_SIGNALS_PATH.exists():
        return []
    df = pd.read_csv(PHYSIO_SIGNALS_PATH)
    # Use only the main known columns (CSV may have trailing empty columns)
    cols = [c for c in ["age", "sex", "bmi", "hypertension", "diabetes", "hemoglobin", "glucose"] if c in df.columns]
    records: list[PhysiologicalSignalsRecord] = []
    for _, row in df.iterrows():
        records.append(
            PhysiologicalSignalsRecord(
                age=_safe_float(row.get("age")),
                sex=_safe_str(row.get("sex")),
                bmi=_safe_float(row.get("bmi")),
                hypertension=_safe_int(row.get("hypertension")),
                diabetes=_safe_int(row.get("diabetes")),
                hemoglobin=_safe_float(row.get("hemoglobin")),
                glucose=_safe_float(row.get("glucose")),
            )
        )
    return records


def load_simplification_pairs() -> list[SimplificationPair]:
    """Load expert/simple text pairs from simplification_pairs.csv with pandas. Returns list of SimplificationPair."""
    if not SIMPLIFICATION_PAIRS_PATH.exists():
        return []
    df = pd.read_csv(SIMPLIFICATION_PAIRS_PATH)
    # Column names are 'Expert' and 'Simple' (with possible trailing comma column)
    expert_col = "Expert" if "Expert" in df.columns else df.columns[0]
    simple_col = "Simple" if "Simple" in df.columns else (df.columns[1] if len(df.columns) > 1 else "")
    return [
        SimplificationPair(expert=_safe_str(row.get(expert_col)) or "", simple=_safe_str(row.get(simple_col)))
        for _, row in df.iterrows()
    ]


def _safe_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _safe_str(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v).strip() or None


# --- Convenience: load all + backward compatibility ---


def load_all_datasets() -> dict[str, Any]:
    """Load all processed datasets. Returns dict with structured lists keyed by dataset name."""
    return {
        "medical_knowledge": load_medical_knowledge(),
        "physio_signals": load_physiological_signals(),
        "physiological_signals": load_physiological_signals(),
        "sleep_data": load_sleep_data(),
        "vital_signs": load_vital_signs(),
        "simplification_pairs": load_simplification_pairs(),
    }


def load_physio_signals() -> list[PhysiologicalSignalsRecord]:
    """Alias for load_physiological_signals(). Load physio_signals.csv. Returns list of PhysiologicalSignalsRecord."""
    return load_physiological_signals()
