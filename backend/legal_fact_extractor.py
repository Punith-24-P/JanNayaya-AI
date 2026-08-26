"""
JanNyaya AI - Structured Legal Fact Extractor

Responsibilities:
1. Extract structured facts from USER DOCUMENTS with provenance:
   - Amounts with semantic typing (loan amount, outstanding amount, etc.)
   - Dates with semantic classification (document date, loan date, deadline, etc.)
   - Interest rates, instalment schedules, notice periods/deadlines
   - Parties (borrower, lender, complainant, accused, landlord, tenant, etc.)
   - Legal references (Acts, Sections, Orders, Rules, Case Numbers)
   - Strict isolation: USER DOCUMENT FACTS are NEVER mixed with statutory law.
2. Group and extract structured legal provisions from RETRIEVED CHUNKS for RAG QA.
3. Multilingual support: English, Hindi, Kannada.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FACT_GROUPS = 3
MAX_FACT_LINES_PER_GROUP = 8
MAX_CONTEXT_CHARS = 8000
MIN_EVIDENCE_LENGTH = 20

PUNISHMENT_PATTERNS = (
    "shall be punished",
    "shall be punishable",
    "punished with",
    "liable to fine",
    "liable to punishment",
    "imprisonment",
    "community service",
    "fine",
    "सजा",
    "कारावास",
    "जुर्माना",
    "ಶಿಕ್ಷೆ",
    "ದಂಡ",
)

DEFINITION_PATTERNS = (
    "is said to commit",
    "means",
    "is said to be",
    "definition",
    "whoever",
    "परिभाषा",
    "अर्थ",
    "ವ್ಯಾಖ್ಯಾನ",
    "ಎಂದರೇನು",
)

CONDITION_PATTERNS = (
    "provided that",
    "if ",
    "where ",
    "in case",
    "second or subsequent",
    "first time",
    "first conviction",
    "upon return",
    "restoration",
    "subject to",
)

SPECIALIZED_PATTERNS = (
    "theft in a dwelling",
    "theft by clerk",
    "theft after preparation",
    "snatching",
    "gang of robbers",
    "house-trespass",
    "house trespass",
)


# ============================================================
# MULTILINGUAL LEGAL TERM MAP
# ============================================================

LEGAL_TERM_MAP = {
    "theft": "theft",
    "bns section 303": "theft",
    "section 303": "theft",
    "stolen": "theft",
    "stole": "theft",
    "dishonestly took": "theft",
    "without consent": "theft",
    "murder": "murder",
    "bns section 101": "murder",
    "bns section 103": "murder",
    "section 101": "murder",
    "section 103": "murder",
    "culpable homicide": "murder",
    "rape": "rape",
    "cheating": "cheating",
    "bns section 318": "cheating",
    "section 318": "cheating",
    "robbery": "robbery",
    "extortion": "extortion",
    "kidnapping": "kidnapping",
    "abduction": "kidnapping",
    "defamation": "defamation",
    "assault": "assault",
    "cheque bounce": "cheque_bounce",
    "dishonour of cheque": "cheque_bounce",
    "dishonor of cheque": "cheque_bounce",
    "section 138": "cheque_bounce",
    "138": "cheque_bounce",
    "loan recovery": "loan_recovery",
    "loan repayment": "loan_recovery",
    "outstanding loan": "loan_recovery",
    "outstanding dues": "loan_recovery",
    "money recovery": "loan_recovery",
    "debt recovery": "loan_recovery",
    "consumer": "consumer_complaint",
    "defective goods": "consumer_complaint",
    "deficiency in service": "consumer_complaint",
    "domestic violence": "domestic_violence",
    "divorce": "marriage_divorce",
    "marriage": "marriage_divorce",
    "cyber crime": "cyber_crime",
    "gratuity": "employment_dispute",
    "unpaid salary": "employment_dispute",
    "legal aid": "legal_aid",
    "lok adalat": "legal_aid",
    "traffic": "traffic_motor_vehicle",
    "traffic light": "traffic_motor_vehicle",
    "red light": "traffic_motor_vehicle",
    "jump red light": "traffic_motor_vehicle",
    "motor vehicles act": "traffic_motor_vehicle",
    "property dispute": "property_dispute",
    "immovable property": "property_dispute",
    "sale deed": "property_dispute",
    "property possession": "property_dispute",
    "land dispute": "property_dispute",
    "encroachment": "property_dispute",
    "rti": "rti",
    "right to information": "rti",
    "pocso": "pocso",
    "rera": "rera_real_estate",
    "senior citizen": "senior_citizens",

    # Hindi
    "चोरी": "theft",
    "हत्या": "murder",
    "बलात्कार": "rape",
    "धोखाधड़ी": "cheating",
    "लूट": "robbery",
    "डकैती": "robbery",
    "जबरन वसूली": "extortion",
    "अपहरण": "kidnapping",
    "मानहानि": "defamation",
    "धमकी": "criminal intimidation",
    "चेक बाउंस": "cheque_bounce",
    "चेक अनादर": "cheque_bounce",
    "ऋण वसूली": "loan_recovery",
    "कर्ज वसूली": "loan_recovery",
    "बकाया राशि": "loan_recovery",
    "उपभोक्ता": "consumer_complaint",
    "खराब सामान": "consumer_complaint",
    "दोषपूर्ण सेवा": "consumer_complaint",
    "घरेलू हिंसा": "domestic_violence",
    "तलाक": "marriage_divorce",
    "विवाह": "marriage_divorce",
    "साइबर अपराध": "cyber_crime",
    "ग्रेच्युटी": "employment_dispute",
    "मुफ्त कानूनी सहायता": "legal_aid",
    "लोक अदालत": "legal_aid",
    "यातायात": "traffic_motor_vehicle",
    "ट्रैफिक लाइट": "traffic_motor_vehicle",
    "चालान": "traffic_motor_vehicle",
    "संपत्ति विवाद": "property_dispute",
    "कब्जा": "property_dispute",
    "सूचना का अधिकार": "rti",
    "पॉक्सो": "pocso",
    "रेरा": "rera_real_estate",
    "वरिष्ठ नागरिक": "senior_citizens",

    # Kannada
    "ಕಳ್ಳತನ": "theft",
    "ಹತ್ಯೆ": "murder",
    "ಕೊಲೆ": "murder",
    "ಅತ್ಯಾಚಾರ": "rape",
    "ಮೋಸ": "cheating",
    "ವಂಚನೆ": "cheating",
    "ದರೋಡೆ": "robbery",
    "ಸುಲಿಗೆ": "extortion",
    "ಅಪಹರಣ": "kidnapping",
    "ಮಾನನಷ್ಟ": "defamation",
    "ಬೆದರಿಕೆ": "criminal intimidation",
    "ಚೆಕ್ ಬೌನ್ಸ್": "cheque_bounce",
    "ಚೆಕ್ ಅಮಾನ್ಯ": "cheque_bounce",
    "ಸಾಲ ವಸೂಲಾತಿ": "loan_recovery",
    "ಬಾಕಿ ಹಣ": "loan_recovery",
    "ಗ್ರಾಹಕ": "consumer_complaint",
    "ದೋಷಪೂರಿತ": "consumer_complaint",
    "ಗೃಹ ಹಿಂಸಾಚಾರ": "domestic_violence",
    "ವಿಚ್ಛೇದನ": "marriage_divorce",
    "ಮದುವೆ": "marriage_divorce",
    "ಸೈಬರ್ ಅಪರಾಧ": "cyber_crime",
    "ಗ್ರಾಚ್ಯುಟಿ": "employment_dispute",
    "ಉಚಿತ ಕಾನೂನು ನೆರವು": "legal_aid",
    "ಲೋಕ ಅದಾಲತ್": "legal_aid",
    "ಸಂಚಾರ ನಿಯಮ": "traffic_motor_vehicle",
    "ಸಿಗ್ನಲ್": "traffic_motor_vehicle",
    "ಕೆಂಪು ದೀಪ": "traffic_motor_vehicle",
    "ಆಸ್ತಿ ವಿವಾದ": "property_dispute",
    "ಸ್ವಾಧೀನ": "property_dispute",
    "ಮಾಹಿತಿ ಹಕ್ಕು": "rti",
    "ಪೋಕ್ಸೋ": "pocso",
    "ರೇರಾ": "rera_real_estate",
    "ಹಿರಿಯ ನಾಗರಿಕರು": "senior_citizens",
}


# ============================================================
# TEXT NORMALIZATION HELPERS
# ============================================================

def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_for_match(text: Any) -> str:
    return _normalize_text(text).lower()


# ============================================================
# USER DOCUMENT FACT EXTRACTION (DOCUMENT EVIDENCE ONLY)
# ============================================================

def extract_document_amounts(
    text: str,
    document_name: str = "Document",
    page_number: int = 1,
) -> List[Dict[str, Any]]:
    """
    Extract monetary amounts exclusively from user document text with
    semantic classification and provenance.
    """
    if not text:
        return []

    val = str(text)
    lower_val = val.lower()
    extracted: List[Dict[str, Any]] = []
    seen_amounts = set()

    # Match currency prefixed amounts
    patterns = [
        (
            r"(?:₹|Rs\.?|INR)\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{2})?)\s*(?:/-)?",
            "currency_prefixed",
        ),
        (
            r"\b([0-9]{1,3}(?:,[0-9]{2,3})+)\s*(?:/-)",
            "comma_slash",
        ),
        (
            r"(?:Rupees|Rs\.)\s+([A-Za-z\s]+(?:Lakh|Lakhs|Crore|Crores|Thousand|Hundred)[A-Za-z\s]*)",
            "words",
        ),
    ]

    for pattern, p_type in patterns:
        for m in re.finditer(pattern, val, flags=re.IGNORECASE):
            raw_match = m.group(0).strip()
            num_str = m.group(1).strip() if m.groups() else raw_match

            # Clean and normalize amount string
            clean_amt = re.sub(r"\s+", " ", raw_match)
            if clean_amt.lower().startswith("t") and re.search(r"[0-9]", clean_amt):
                clean_amt = "₹" + clean_amt[1:].lstrip()

            key = re.sub(r"[^\w]", "", clean_amt).lower()
            if key in seen_amounts or len(clean_amt) < 3:
                continue
            seen_amounts.add(key)

            # Determine semantic amount type from surrounding text
            start_ctx = max(0, m.start() - 80)
            end_ctx = min(len(val), m.end() + 80)
            ctx = lower_val[start_ctx:end_ctx]

            amt_type = "monetary_amount"
            if any(k in ctx for k in ["loan amount", "advanced a loan", "sanctioned loan", "principal loan"]):
                amt_type = "loan_amount"
            elif any(k in ctx for k in ["outstanding", "dues", "balance amount", "due and payable", "arrears"]):
                amt_type = "outstanding_amount"
            elif any(k in ctx for k in ["monthly instalment", "monthly installment", "emi", "per month"]):
                amt_type = "instalment_amount"
            elif any(k in ctx for k in ["claim", "demanded", "called upon to pay", "seek recovery"]):
                amt_type = "claimed_amount"
            elif any(k in ctx for k in ["interest", "further interest"]):
                amt_type = "interest_amount"

            extracted.append({
                "value": clean_amt,
                "type": amt_type,
                "document": document_name,
                "page": page_number,
                "confidence": "high" if "₹" in clean_amt or "rs." in clean_amt.lower() else "medium",
            })

    return extracted


def extract_document_dates(
    text: str,
    document_name: str = "Document",
    page_number: int = 1,
) -> List[Dict[str, Any]]:
    """
    Extract dates exclusively from user document text with semantic
    classification and provenance.
    """
    if not text:
        return []

    val = str(text)
    lower_val = val.lower()
    extracted: List[Dict[str, Any]] = []
    seen_dates = set()

    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
        r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b",
    ]

    for pattern in date_patterns:
        for m in re.finditer(pattern, val, flags=re.IGNORECASE):
            date_str = m.group(0).strip()
            key = date_str.lower()
            if key in seen_dates:
                continue
            seen_dates.add(key)

            # Classify date using context
            start_ctx = max(0, m.start() - 80)
            end_ctx = min(len(val), m.end() + 80)
            ctx = lower_val[start_ctx:end_ctx]

            date_type = "general_date"
            if any(k in ctx for k in ["dated", "date:", "notice date", "legal notice", "dated this"]):
                date_type = "document_date"
            elif any(k in ctx for k in ["loan on", "advanced", "availed on", "borrowed on", "disbursed"]):
                date_type = "loan_date"
            elif any(k in ctx for k in ["agreement date", "contract dated", "deed dated", "executed on"]):
                date_type = "agreement_date"
            elif any(k in ctx for k in ["hearing", "next date of hearing", "listed on", "appear on"]):
                date_type = "hearing_date"
            elif any(k in ctx for k in ["incident", "occurred on", "alleged on"]):
                date_type = "incident_date"
            elif any(k in ctx for k in ["deadline", "due date", "within", "pay before"]):
                date_type = "deadline"

            extracted.append({
                "value": date_str,
                "type": date_type,
                "document": document_name,
                "page": page_number,
                "confidence": "high",
            })

    return extracted


def extract_structured_facts_with_provenance(
    text: str,
    document_name: str = "Document",
    page_number: int = 1,
) -> Dict[str, Any]:
    """
    Extract a comprehensive, structured fact dictionary with provenance
    directly from user document text.
    """
    if not text:
        return {
            "amounts": [],
            "dates": [],
            "interest_rates": [],
            "instalments": [],
            "deadlines": [],
            "legal_references": [],
            "parties": [],
            "case_numbers": [],
            "facts": [],
        }

    val = str(text)
    lower_val = val.lower()

    # 1. Amounts
    amounts = extract_document_amounts(val, document_name, page_number)

    # 2. Dates
    dates = extract_document_dates(val, document_name, page_number)

    # 3. Interest Rates
    interest_rates = []
    for m in re.finditer(r"\b(\d+(?:\.\d+)?\s*%\s*(?:p\.?a\.?|per\s+annum)?)\b", val, re.IGNORECASE):
        rate_val = m.group(0).strip()
        if rate_val not in [i["value"] for i in interest_rates]:
            interest_rates.append({
                "value": rate_val,
                "type": "interest_rate",
                "document": document_name,
                "page": page_number,
                "confidence": "high",
            })

    # 4. Instalments / Repayment schedule
    instalments = []
    for m in re.finditer(r"\b(\d+\s+monthly\s+instalments|\d+\s+monthly\s+installments|\d+\s+emis|\d+\s+equal\s+monthly\s+installments)\b", val, re.IGNORECASE):
        inst_val = m.group(0).strip()
        if inst_val not in [i["value"] for i in instalments]:
            instalments.append({
                "value": inst_val,
                "type": "instalment_schedule",
                "document": document_name,
                "page": page_number,
                "confidence": "high",
            })

    # 5. Deadlines & Notice Periods
    deadlines = []
    for m in re.finditer(r"\b((?:within\s+)?(?:\d+|\b(?:fifteen|thirty|seven|fourteen|twenty|sixty)\b)\s+days(?:\s+from\s+receipt|\s+from\s+the\s+date)?)\b", val, re.IGNORECASE):
        dl_val = m.group(0).strip()
        if dl_val not in [d["value"] for d in deadlines]:
            deadlines.append({
                "value": dl_val,
                "type": "notice_deadline",
                "document": document_name,
                "page": page_number,
                "confidence": "high",
            })

    # 6. Legal References (Acts, Sections, Rules, Orders)
    legal_refs = []
    ref_patterns = [
        r"Order\s+[0-9IVXLCDM]+\s+Rule\s+\d+[^,\n)]*(?:Code of Civil Procedure|CPC)?",
        r"Section\s+\d+[A-Za-z]?(?:\s*\([^)]+\))?(?:\s+of\s+[A-Za-z\s,0-9]+Act)?",
        r"IPC\s+Section\s+\d+[A-Za-z]?",
        r"BNS\s+Section\s+\d+[A-Za-z]?",
        r"Section\s+138(?:\s+Negotiable\s+Instruments\s+Act)?",
        r"Code of Civil Procedure,\s*1908",
        r"Indian Contract Act,\s*1872",
        r"Bharatiya Nyaya Sanhita,\s*2023",
        r"Limitation Act,\s*1963",
        r"Consumer Protection Act,\s*2019",
        r"Negotiable Instruments Act,\s*1881",
    ]
    for pat in ref_patterns:
        for m in re.finditer(pat, val, re.IGNORECASE):
            ref_val = m.group(0).strip()
            if ref_val not in [r["value"] for r in legal_refs]:
                legal_refs.append({
                    "value": ref_val,
                    "type": "statutory_reference",
                    "document": document_name,
                    "page": page_number,
                    "confidence": "high",
                })

    # 7. Case numbers & Court names
    case_numbers = []
    for m in re.finditer(r"\b(?:Case\s+No|O\.?S\.?|C\.?C\.?|Crime\s+No|FIR\s+No|Petition\s+No|Complaint\s+No)\.?\s*[:\s]?\s*([A-Za-z0-9/\-]+)\b", val, re.IGNORECASE):
        cn_val = m.group(0).strip()
        if cn_val not in [c["value"] for c in case_numbers]:
            case_numbers.append({
                "value": cn_val,
                "type": "case_number",
                "document": document_name,
                "page": page_number,
                "confidence": "high",
            })

    # 8. High-level document fact sentences
    fact_lines = []
    for line in val.splitlines():
        line_s = line.strip()
        if 20 <= len(line_s) <= 300:
            line_lower = line_s.lower()
            if any(k in line_lower for k in [
                "loan amount", "advanced a loan", "outstanding amount", "repay",
                "monthly instalments", "interest at", "called upon to pay",
                "notice is hereby given", "failed to pay", "dishonestly took",
                "caused the death", "stolen", "breached", "employment", "deficiency",
            ]):
                fact_lines.append({
                    "value": line_s,
                    "type": "document_evidence",
                    "document": document_name,
                    "page": page_number,
                    "confidence": "high",
                })

    return {
        "amounts": amounts,
        "dates": dates,
        "interest_rates": interest_rates,
        "instalments": instalments,
        "deadlines": deadlines,
        "legal_references": legal_refs,
        "case_numbers": case_numbers,
        "facts": fact_lines[:15],
    }


# ============================================================
# QUERY INTENT & LEGAL TERM (FOR QA PIPELINE)
# ============================================================

def detect_fact_intent(query: str) -> str:
    """Detect intent for Q&A query: punishment, definition, procedure, remedy, general."""
    if not query:
        return "general"

    text = _normalize_for_match(query)

    for p in PUNISHMENT_PATTERNS:
        if p in text:
            return "punishment"

    for d in DEFINITION_PATTERNS:
        if d in text:
            return "definition"

    if any(r in text for r in ["remedy", "relief", "compensation", "damages", "refund", "ಬೆಂಬಲ", "ಪರಿಹಾರ", "उपचार", "मुआवजा"]):
        return "remedy"

    if any(pr in text for pr in ["procedure", "how to file", "process", "steps", "ಕ್ರಮ", "ಪ್ರಕ್ರಿಯೆ", "प्रक्रिया"]):
        return "procedure"

    return "general"


def detect_legal_term(query: str) -> str:
    """Convert multilingual query terms into a canonical legal term."""
    if not query:
        return ""

    text = str(query).strip()

    for phrase in sorted(LEGAL_TERM_MAP.keys(), key=len, reverse=True):
        if phrase in text:
            return LEGAL_TERM_MAP[phrase]

    normalized = _normalize_for_match(text)
    for phrase in sorted(LEGAL_TERM_MAP.keys(), key=len, reverse=True):
        if _normalize_for_match(phrase) in normalized:
            return LEGAL_TERM_MAP[phrase]

    return ""


# ============================================================
# RETRIEVED CHUNK FACT EXTRACTION (FOR QA PIPELINE)
# ============================================================

def _get_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    value = result.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _get_section(result: Dict[str, Any]) -> str:
    metadata = _get_metadata(result)
    value = metadata.get("section_number", metadata.get("section", ""))
    sec = str(value).strip() if value is not None else ""
    if sec and sec.lower() not in ("none", "unknown", ""):
        return sec

    title = str(metadata.get("section_title") or metadata.get("title") or "")
    m = re.match(r"^(?:Section\s*)?(\d+[A-Za-z]?)\b", title, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _get_section_title(result: Dict[str, Any]) -> str:
    metadata = _get_metadata(result)
    value = metadata.get("section_title", "")
    return str(value).strip() if value is not None else ""


def _get_source(result: Dict[str, Any]) -> str:
    metadata = _get_metadata(result)
    value = metadata.get("source", metadata.get("act_name", "Unknown"))
    return str(value).strip() if value is not None else "Unknown"


def _get_document(result: Dict[str, Any]) -> str:
    value = result.get("document", "")
    return _normalize_text(value)


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_EVIDENCE_LENGTH]


def extract_legal_facts(
    results: List[Dict[str, Any]],
    query: str = "",
    legal_term: str = "",
    max_groups: int = MAX_FACT_GROUPS,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Extract clean, structured statutory provisions from retrieved knowledge base chunks.
    Filters out amendment footnotes and metadata noise.
    """
    if not results:
        return []

    facts = []
    seen_sections = set()

    for r in results:
        if not isinstance(r, dict):
            continue

        sec = _get_section(r)
        title = _get_section_title(r)
        source = _get_source(r)
        doc = _get_document(r)
        score = float(r.get("hybrid_score", r.get("score", 0.0)))

        # Reject obvious amendment footnote fragments
        lower_title = title.lower()
        if any(bad in lower_title for bad in ["subs. by", "ins. by", "repealed by", "a.o. 1950", "act 24 of 1917"]):
            continue

        sec_key = (source.lower(), sec)
        if sec and sec_key in seen_sections:
            continue
        if sec:
            seen_sections.add(sec_key)

        sentences = _split_sentences(doc)
        definitions = [s for s in sentences if any(d in s.lower() for d in DEFINITION_PATTERNS)]
        punishments = [s for s in sentences if any(p in s.lower() for p in PUNISHMENT_PATTERNS)]
        conditions = [s for s in sentences if any(c in s.lower() for c in CONDITION_PATTERNS)]

        facts.append({
            "section": sec,
            "section_title": title,
            "source": source,
            "title": _get_metadata(r).get("title", source),
            "document_type": _get_metadata(r).get("document_type", "Act"),
            "best_chunk": _get_metadata(r).get("chunk_index", 0),
            "score": score,
            "definitions": definitions[:3],
            "punishments": punishments[:3],
            "conditions": conditions[:3],
            "full_text": doc[:1200],
        })

        if len(facts) >= max_groups:
            break

    return facts


def build_fact_context(
    facts: List[Dict[str, Any]],
    max_characters: int = MAX_CONTEXT_CHARS,
    *args,
    **kwargs,
) -> str:
    """Format extracted legal facts into clean context for LLM generation."""
    if not facts:
        return "No specific statutory provisions available."

    blocks = []
    for idx, f in enumerate(facts, start=1):
        sec = f.get("section", "")
        title = f.get("section_title", "")
        source = f.get("source", "")
        header = f"[Statutory Provision {idx}]"
        if sec:
            header += f"\nSection: {sec}"
        if title:
            header += f"\nTitle: {title}"
        if source:
            header += f"\nAct: {source}"

        body_parts = []
        if f.get("definitions"):
            body_parts.append("Definition / Elements:\n" + "\n".join(f"- {d}" for d in f["definitions"]))
        if f.get("punishments"):
            body_parts.append("Punishment / Consequences:\n" + "\n".join(f"- {p}" for p in f["punishments"]))
        if f.get("conditions"):
            body_parts.append("Exceptions / Conditions:\n" + "\n".join(f"- {c}" for c in f["conditions"]))

        if not body_parts and f.get("full_text"):
            body_parts.append("Summary:\n" + f["full_text"][:500])

        blocks.append(header + "\n" + "\n\n".join(body_parts))

    return "\n\n" + ("=" * 40) + "\n\n".join(blocks)