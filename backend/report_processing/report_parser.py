"""Parse medical reports (PDF or image): OCR, metric detection, structured JSON output."""

import io
import re
from typing import Any, Optional

# -----------------------------------------------------------------------------
# 1. Text extraction (OCR and PDF)
# -----------------------------------------------------------------------------
# OCR uses Tesseract (pytesseract). Install Tesseract: https://github.com/tesseract-ocr/tesseract
# PDF text uses PyMuPDF (fitz); scanned PDFs are rendered to images then OCR'd.


def _ocr_image(image_bytes: bytes) -> str:
    """Extract text from image bytes using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF (text-based). Returns empty if scanned/no text."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _pdf_pages_to_images(pdf_bytes: bytes) -> list[bytes]:
    """Render PDF pages to image bytes (for OCR of scanned PDFs)."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError:
        return []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=150, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(buf.getvalue())
        doc.close()
        return images
    except Exception:
        return []


def extract_text_from_file(content: bytes, filename: str) -> tuple[str, str]:
    """
    Extract text from an uploaded file (PDF or image).
    Returns (extracted_text, source) where source is "pdf" or "image".
    Uses OCR for images and for PDFs with little/no extractable text.
    """
    filename_lower = (filename or "").lower()
    source = "pdf" if filename_lower.endswith(".pdf") else "image"

    if source == "pdf":
        text = _pdf_to_text(content)
        if not text or len(text.strip()) < 50:
            images = _pdf_pages_to_images(content)
            text_parts = []
            for img_bytes in images:
                text_parts.append(_ocr_image(img_bytes))
            text = "\n".join(text_parts)
        return text.strip(), source

    return _ocr_image(content).strip(), source


# -----------------------------------------------------------------------------
# 2. Medical metric detection (regex)
# -----------------------------------------------------------------------------

# Patterns return all matches for each metric (lists)
_BP_PATTERN = re.compile(r"\b(?:blood\s*pressure|bp|sbp/dbp)[:\s]*(\d{2,3})\s*/\s*(\d{2,3})", re.I)
_BP_PATTERN_ALT = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*Hg|mmHg)?", re.I)
_HR_PATTERN = re.compile(r"\b(?:heart\s*rate|hr|pulse|bpm)[:\s]*(\d{2,3})\s*(?:bpm)?", re.I)
_GLUCOSE_PATTERN = re.compile(
    r"\b(?:glucose|blood\s*sugar|glu|bs)[:\s]*(\d{2,3}\.?\d*)\s*(?:mg/dl|mg/dL|mmol/L)?", re.I
)
_SPO2_PATTERN = re.compile(
    r"\b(?:spo2|sp\s*o2|oxygen\s*saturation|o2\s*sat)[:\s]*(\d{2,3})\s*%?", re.I
)
_TEMP_PATTERN = re.compile(
    r"\b(?:temp(?:erature)?|temp\.?)[:\s]*(\d{2,3}\.?\d*)\s*°?\s*[fc]?\b", re.I
)

# CBC-style lab patterns
_HEMOGLOBIN_PATTERN = re.compile(r"\bhemoglobin\b[:\s]*([0-9]+\.?[0-9]*)", re.I)
_RBC_PATTERN = re.compile(r"\b(?:rbc|red\s*blood\s*cells?)\b[:\s]*([0-9]+\.?[0-9]*)", re.I)
_WBC_PATTERN = re.compile(r"\b(?:wbc|white\s*blood\s*cells?)\b[:\s]*([0-9]+\.?[0-9]*)", re.I)
_HEMATOCRIT_PATTERN = re.compile(r"\bhematocrit\b[:\s]*([0-9]+\.?[0-9]*)", re.I)
_MCV_PATTERN = re.compile(r"\bmcv\b[:\s]*([0-9]+\.?[0-9]*)", re.I)
_MCH_PATTERN = re.compile(r"\bmch\b[:\s]*([0-9]+\.?[0-9]*)", re.I)
_MCHC_PATTERN = re.compile(r"\bmchc\b[:\s]*([0-9]+\.?[0-9]*)", re.I)
_RDW_PATTERN = re.compile(r"\brdw\b[:\s]*([0-9]+\.?[0-9]*)", re.I)
_PLATELET_PATTERN = re.compile(r"\b(?:platelets?|platelet\s+count)\b[:\s]*([0-9]+)", re.I)
_ESR_PATTERN = re.compile(r"\b(?:esr|erythrocyte\s+sedimentation\s+rate)\b[:\s]*([0-9]+)", re.I)


def detect_medical_metrics(text: str) -> dict[str, Any]:
    """
    Detect medical metrics in text using regex. Returns structured dict with
    blood_pressure, heart_rate, glucose, oxygen_saturation, temperature.
    Only metrics actually detected are returned (no default vitals).
    """
    text = text or ""
    result: dict[str, Any] = {
        "blood_pressure": [],
        "heart_rate": [],
        "glucose": [],
        "oxygen_saturation": [],
        "temperature": [],
    }

    # Blood pressure: labeled first, then standalone X/Y (avoid duplicates)
    seen_bp: set[tuple[int, int]] = set()
    for m in _BP_PATTERN.finditer(text):
        s, d = int(m.group(1)), int(m.group(2))
        if (s, d) not in seen_bp and 70 <= s <= 250 and 40 <= d <= 150:
            seen_bp.add((s, d))
            result["blood_pressure"].append({"systolic": s, "diastolic": d})
    for m in _BP_PATTERN_ALT.finditer(text):
        s, d = int(m.group(1)), int(m.group(2))
        if (s, d) not in seen_bp and 70 <= s <= 250 and 40 <= d <= 150:
            seen_bp.add((s, d))
            result["blood_pressure"].append({"systolic": s, "diastolic": d})

    # Heart rate
    for m in _HR_PATTERN.finditer(text):
        v = int(m.group(1))
        if 30 <= v <= 250:
            result["heart_rate"].append(v)

    # Glucose
    for m in _GLUCOSE_PATTERN.finditer(text):
        try:
            v = float(m.group(1))
            if 20 <= v <= 600:
                result["glucose"].append(v)
        except ValueError:
            pass

    # Oxygen saturation
    for m in _SPO2_PATTERN.finditer(text):
        v = int(m.group(1))
        if 50 <= v <= 100:
            result["oxygen_saturation"].append(v)

    # Temperature
    for m in _TEMP_PATTERN.finditer(text):
        try:
            v = float(m.group(1))
            if 35 <= v <= 43 or 90 <= v <= 110:
                result["temperature"].append(v)
        except ValueError:
            pass

    # Drop metrics with no detected values so we don't return default vitals.
    return {k: v for k, v in result.items() if v}


def detect_cbc_metrics(text: str) -> dict[str, Any]:
    """Detect common CBC blood test metrics in free text."""
    text = text or ""
    out: dict[str, Any] = {}

    def _first_number(pattern: re.Pattern[str]) -> Optional[float]:
        m = pattern.search(text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            return None

    hgb = _first_number(_HEMOGLOBIN_PATTERN)
    if hgb is not None:
        out["hemoglobin"] = hgb
    rbc = _first_number(_RBC_PATTERN)
    if rbc is not None:
        out["rbc_count"] = rbc
    wbc = _first_number(_WBC_PATTERN)
    if wbc is not None:
        out["wbc_count"] = wbc
    hct = _first_number(_HEMATOCRIT_PATTERN)
    if hct is not None:
        out["hematocrit"] = hct
    mcv = _first_number(_MCV_PATTERN)
    if mcv is not None:
        out["mcv"] = mcv
    mch = _first_number(_MCH_PATTERN)
    if mch is not None:
        out["mch"] = mch
    mchc = _first_number(_MCHC_PATTERN)
    if mchc is not None:
        out["mchc"] = mchc
    rdw = _first_number(_RDW_PATTERN)
    if rdw is not None:
        out["rdw"] = rdw
    plt = _first_number(_PLATELET_PATTERN)
    if plt is not None:
        out["platelet_count"] = plt
    esr = _first_number(_ESR_PATTERN)
    if esr is not None:
        out["esr"] = esr

    return out


def parse_uploaded_report(content: bytes, filename: str) -> dict[str, Any]:
    """
    Full pipeline: extract text from uploaded PDF or image (OCR), detect medical
    metrics with regex, return structured JSON.
    """
    raw_text, source = extract_text_from_file(content, filename)
    metrics = detect_medical_metrics(raw_text)
    cbc_metrics = detect_cbc_metrics(raw_text)
    if cbc_metrics:
        metrics["cbc"] = cbc_metrics
    return {
        "source": source,
        "filename": filename,
        "metrics": metrics,
        "raw_text_preview": (raw_text[:2000] + "…") if len(raw_text) > 2000 else raw_text,
        "raw_text_length": len(raw_text),
    }


# -----------------------------------------------------------------------------
# Legacy API (unchanged)
# -----------------------------------------------------------------------------


def parse_blood_pressure(text: str) -> Optional[tuple[int, int]]:
    """Extract systolic/diastolic from strings like '120/80' or '126/83'."""
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def extract_vitals_from_text(text: str) -> dict[str, Any]:
    """
    Extract vital-like numbers from free text: BP, heart rate, temp, SpO2, etc.
    Single values per metric (first match).
    """
    result: dict[str, Any] = {}
    bp = parse_blood_pressure(text)
    if bp:
        result["blood_pressure"] = {"systolic": bp[0], "diastolic": bp[1]}
    hr = re.search(r"\b(?:heart rate|hr|pulse)[:\s]*(\d{2,3})\b", text, re.I)
    if hr:
        result["heart_rate"] = int(hr.group(1))
    temp = re.search(r"\b(?:temp(?:erature)?|temp\.?)[:\s]*(\d+\.?\d*)\s*°?[cf]?\b", text, re.I)
    if temp:
        result["temperature"] = float(temp.group(1))
    spo2 = re.search(r"\b(?:spo2|sp o2|oxygen saturation|o2 sat)[:\s]*(\d{2,3})\b", text, re.I)
    if spo2:
        result["spo2"] = int(spo2.group(1))
    return result


def parse_report(text: str) -> dict[str, Any]:
    """
    Parse a medical report: extract vitals and segment into sections by common headers.
    """
    sections: dict[str, str] = {}
    current = "preamble"
    current_text: list[str] = []
    headers = re.compile(
        r"^\s*(?:history|assessment|plan|vitals|medications|diagnosis|findings|results)\s*[:\-]?\s*$",
        re.I,
    )
    lines = text.split("\n") if text else []
    for line in lines:
        if headers.match(line.strip()):
            if current_text:
                sections[current] = "\n".join(current_text).strip()
            current = line.strip().rstrip(":-").strip().lower()
            current_text = []
        else:
            current_text.append(line)
    if current_text:
        sections[current] = "\n".join(current_text).strip()
    vitals = extract_vitals_from_text(text)
    return {
        "sections": sections,
        "vitals": vitals,
        "raw_length": len(text),
    }
