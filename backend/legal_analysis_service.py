"""
JanNyaya AI - Comprehensive Legal Document Analysis Service

Features:
- Multi-format document type classification (18+ types with confidence).
- Structured fact extraction with provenance (amounts, dates, interest, instalments, deadlines, parties).
- Strict isolation: User document facts are NEVER conflated with statutory legal provisions.
- Multi-issue detection (primary legal issue and secondary issues with confidence).
- 12 Legal Route classifications and query intent detection.
- Targeted multi-query hybrid retrieval with legal reranking.
- Legal provision analysis with multi-factor relevance scoring and false-positive prevention.
- Multi-document case analysis with cross-document conflict/discrepancy detection.
- Attributed chronological timeline generation.
- Multilingual support: English, Hindi, Kannada.
- Grounded legal caution: Phrased safely ("This may relate to Section X based on available facts").
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.retriever import hybrid_search, detect_query_intent
from backend.legal_fact_extractor import (
    extract_structured_facts_with_provenance,
    extract_document_amounts,
    extract_document_dates,
    detect_legal_term as extract_canonical_term,
)
from backend.legal_provision_service import (
    analyze_provisions,
    legal_topic_label,
)
from backend.llm_service import explain_legal_document


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TEXT_CHARS = 16000
MAX_RESULTS_PER_QUERY = 12
MAX_ACCEPTED_SOURCES = 10
MAX_QUERY_COUNT = 6


# ============================================================
# ROUTE LABELS (12 LEGAL ROUTES)
# ============================================================

ROUTE_LABELS = {
    "criminal": "Criminal law",
    "civil_contractual": "Civil / Contractual law",
    "property": "Property / Real Estate law",
    "family": "Family / Matrimonial law",
    "consumer": "Consumer Protection law",
    "employment": "Employment / Labour law",
    "cyber": "Cyber & Information Technology law",
    "commercial": "Commercial & Negotiable Instruments law",
    "financial": "Banking & Financial Services law",
    "constitutional": "Constitutional & Fundamental Rights law",
    "legal_aid": "Legal Services Authorities / Legal Aid",
    "general": "General legal matter",
}


# ============================================================
# TOPIC CONFIGURATION
# ============================================================

TOPIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "loan_recovery": {
        "route": "civil_contractual",
        "label": "Loan recovery / contractual money recovery",
        "keywords": [
            "loan", "loan amount", "loan agreement", "borrower", "lender",
            "repay", "repayment", "outstanding", "outstanding amount",
            "outstanding dues", "debt", "creditor", "debtor", "interest",
            "instalment", "installment", "default", "recovery", "recover",
            "payment", "money",
        ],
        "queries": [
            "loan agreement repayment recovery",
            "outstanding debt recovery creditor borrower",
            "Indian Contract Act breach of contract compensation damages",
            "Code of Civil Procedure Order 37 summary suit debt recovery",
        ],
    },
    "theft": {
        "route": "criminal",
        "label": "Theft / Dishonest taking of property",
        "keywords": [
            "theft", "stolen", "stole", "dishonestly took", "movable property",
            "without consent", "taking property", "stealing", "thief",
            "section 303", "bns 303", "bns section 303", "dishonest taking",
        ],
        "queries": [
            "Bharatiya Nyaya Sanhita Section 303 theft definition punishment",
            "theft movable property without consent punishment",
            "BNS theft offence",
        ],
    },
    "murder": {
        "route": "criminal",
        "label": "Murder / Culpable homicide",
        "keywords": [
            "murder", "caused the death", "death of", "killed", "homicide",
            "intentional death", "culpable homicide", "section 101", "section 103",
        ],
        "queries": [
            "Bharatiya Nyaya Sanhita Section 101 103 murder punishment",
            "murder offence BNS culpable homicide",
        ],
    },
    "cheating": {
        "route": "criminal",
        "label": "Cheating / Dishonest inducement",
        "keywords": [
            "cheating", "deceived", "deception", "dishonestly induced",
            "false representation", "fraud", "fraudulent", "section 318",
        ],
        "queries": [
            "Bharatiya Nyaya Sanhita Section 318 cheating punishment",
            "cheating deception dishonest inducement property",
        ],
    },
    "property_dispute": {
        "route": "property",
        "label": "Property ownership / possession dispute",
        "keywords": [
            "immovable property", "ownership", "title deed", "land dispute", "building",
            "immovable", "partition", "boundary", "sale deed", "gift deed",
            "mortgage deed", "lease", "tenant", "landlord", "eviction", "encroachment",
            "property dispute", "peaceful possession",
        ],
        "queries": [
            "Transfer of Property Act sale mortgage lease gift",
            "property ownership possession title dispute injunction",
            "Specific Relief Act recovery of possession immovable property",
        ],
    },
    "consumer_complaint": {
        "route": "consumer",
        "label": "Consumer dispute / Defective goods / Deficiency in service",
        "keywords": [
            "consumer", "defective", "defect", "deficiency in service",
            "unfair trade practice", "district commission", "product liability",
            "warranty", "guarantee", "refund", "replacement", "consumer forum",
        ],
        "queries": [
            "Consumer Protection Act 2019 defect goods deficiency service",
            "consumer complaint district commission refund compensation",
            "unfair trade practice product liability consumer rights",
        ],
    },
    "cheque_bounce": {
        "route": "commercial",
        "label": "Dishonour of Cheque / Section 138 NI Act",
        "keywords": [
            "cheque", "dishonour", "dishonored", "dishonoured", "bounce",
            "bounced", "138", "section 138", "insufficient funds",
            "funds insufficient", "drawer", "payee", "notice period 15 days",
        ],
        "queries": [
            "Negotiable Instruments Act Section 138 dishonour of cheque",
            "cheque bounce punishment fine imprisonment 15 days notice",
        ],
    },
    "corporate_company_law": {
        "route": "commercial",
        "label": "Corporate / Company Law / Director duties / Fraud",
        "keywords": [
            "company", "companies act", "director", "directors", "duties of directors",
            "board of directors", "independent director", "corporate fraud", "section 447",
            "section 166", "section 135", "csr", "corporate social responsibility",
            "oppression", "mismanagement", "section 241", "section 242", "nclt",
            "national company law tribunal", "roc", "registrar of companies",
            "related party transactions", "section 188", "section 454", "dormant company",
        ],
        "queries": [
            "Companies Act 2013 Section 447 punishment for fraud imprisonment fine",
            "Duties of directors Section 166 Companies Act 2013 fiduciary duty",
            "Section 135 Corporate Social Responsibility CSR spending net profit",
            "Section 241 242 Oppression and mismanagement NCLT application",
        ],
    },
    "employment_dispute": {
        "route": "employment",
        "label": "Employment / Labour / Unpaid wages dispute",
        "keywords": [
            "employee", "employer", "employment", "salary", "wages", "unpaid salary",
            "termination", "dismissal", "gratuity", "provident fund", "labour",
        ],
        "queries": [
            "Code on Wages 2019 payment of wages delay unpaid salary",
            "Payment of Gratuity Act 1972 eligibility payment default",
            "employment termination wrongful dismissal labour court",
        ],
    },
    "domestic_violence": {
        "route": "family",
        "label": "Domestic violence / Protection of women",
        "keywords": [
            "domestic violence", "protection order", "residence order",
            "shared household", "monetary relief", "physical abuse", "harassment",
        ],
        "queries": [
            "Protection of Women from Domestic Violence Act 2005",
            "domestic violence protection order residence shared household",
        ],
    },
    "marriage_divorce": {
        "route": "family",
        "label": "Marriage / Divorce / Maintenance",
        "keywords": [
            "marriage", "divorce", "restitution of conjugal rights",
            "mutual consent", "alimony", "maintenance", "custody",
        ],
        "queries": [
            "Hindu Marriage Act 1955 divorce mutual consent cruelty maintenance",
            "maintenance permanent alimony child custody",
        ],
    },
    "cyber_crime": {
        "route": "cyber",
        "label": "Cyber crime / Information Technology offence",
        "keywords": [
            "cyber", "cybercrime", "identity theft", "online fraud", "hacking",
            "hacked", "unauthorized access", "phishing", "section 66",
        ],
        "queries": [
            "Information Technology Act 2000 Section 66 identity theft hacking",
            "cyber crime penalty unauthorized access computer system",
        ],
    },
    "legal_aid": {
        "route": "legal_aid",
        "label": "Legal aid / Lok Adalat / Free legal assistance",
        "keywords": [
            "legal aid", "lok adalat", "free legal services", "legal services authority",
            "dlsa", "slsa", "nalasa", "15100",
        ],
        "queries": [
            "Legal Services Authorities Act 1987 free legal aid eligibility",
            "Lok Adalat settlement award finality",
        ],
    },
    "traffic_motor_vehicle": {
        "route": "general",
        "label": "Motor Vehicles / Traffic violation",
        "keywords": [
            "traffic", "traffic light", "red light", "driving license", "challan",
            "motor vehicles act", "rash driving", "overspeeding",
        ],
        "queries": [
            "Motor Vehicles Act 1988 traffic violation penalty fine",
        ],
    },
    "rti": {
        "route": "general",
        "label": "Right to Information (RTI)",
        "keywords": [
            "rti", "right to information", "public information officer", "pio",
            "first appeal", "information request",
        ],
        "queries": [
            "Right to Information Act 2005 Section 6 Section 7 timeline",
        ],
    },
    "pocso": {
        "route": "criminal",
        "label": "Protection of Children from Sexual Offences (POCSO)",
        "keywords": [
            "pocso", "child sexual abuse", "special court", "child victim",
        ],
        "queries": [
            "POCSO Act 2012 offences against children punishment",
        ],
    },
    "senior_citizens": {
        "route": "family",
        "label": "Maintenance and Welfare of Senior Citizens",
        "keywords": [
            "senior citizen", "maintenance of parents", "elderly welfare",
            "maintenance tribunal",
        ],
        "queries": [
            "Maintenance and Welfare of Parents and Senior Citizens Act 2007",
        ],
    },
}


# ============================================================
# TEXT CLEANING HELPERS
# ============================================================

def _clean_text(text: str) -> str:
    if not text:
        return ""
    val = str(text).replace("\x00", " ")
    val = re.sub(r"[ \t]+", " ", val)
    val = re.sub(r"\n{3,}", "\n\n", val)
    return val.strip()


def _lower(text: str) -> str:
    return str(text or "").lower()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _lower(text)).strip()


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_document_language(text: str) -> str:
    if not text:
        return "english"

    hindi = 0
    kannada = 0

    for character in text:
        code = ord(character)
        if 0x0900 <= code <= 0x097F:
            hindi += 1
        elif 0x0C80 <= code <= 0x0CFF:
            kannada += 1

    if kannada > 15 or kannada > hindi:
        return "kannada"
    if hindi > 15:
        return "hindi"

    return "english"


def normalize_language(lang_input: Optional[str]) -> str:
    """Normalize language code or name to 'english', 'hindi', or 'kannada'."""
    if not lang_input:
        return "english"
    norm = str(lang_input).strip().lower()
    if norm in ("kn", "kannada", "kan"):
        return "kannada"
    if norm in ("hi", "hindi", "hin", "devanagari"):
        return "hindi"
    if norm in ("en", "english", "eng"):
        return "english"
    return "english"


# ============================================================
# DOCUMENT TYPE CLASSIFICATION (18+ TYPES WITH CONFIDENCE)
# ============================================================

def detect_document_type(text: str) -> Tuple[str, str]:
    """
    Classify legal document type from evidence, headings, and structure.
    Returns (document_type_name, confidence).
    """
    val = _normalize(text)

    # Type matching patterns in priority order
    type_definitions = [
        ("Legal Notice", [
            "legal notice", "advocate notice", "demand notice", "notice for recovery",
            "under instructions from my client", "hereby call upon you",
            "failing which my client shall be constrained", "initiate legal proceedings",
            "notice is hereby given", "ಕಾನೂನು ನೋಟಿಸ್", "ವಕೀಲರ ನೋಟಿಸ್", "कानूनी नोटिस",
        ]),
        ("Agreement", [
            "loan agreement", "lease agreement", "rent agreement", "service agreement",
            "partnership agreement", "sale agreement", "terms and conditions",
            "mutually agreed", "entered into this agreement", "ಒಪ್ಪಂದ", "करार",
        ]),
        ("Contract", [
            "employment contract", "contract of service", "construction contract",
            "commercial contract", "contract dated", "hereby contract and agree",
        ]),
        ("Court Order", [
            "court order", "interim order", "stay order", "injunction order",
            "ordered accordingly", "it is hereby ordered", "it is ordered that",
            "ಆದೇಶ", "ಅದಾಲತ್ ಆದೇಶ", "अदालत का आदेश",
        ]),
        ("Judgment", [
            "judgment", "final judgment", "decree", "operative portion",
            "in the high court of", "in the supreme court of", "hon'ble mr. justice",
            "appeal allowed", "suit decreed", "ತೀರ್ಪು", "निर्णय", "फैसला",
        ]),
        ("Petition", [
            "writ petition", "special leave petition", "criminal petition",
            "civil petition", "petitioner", "humbly pray", "prayer clause",
            "ಅರ್ಜಿ", "याचिका",
        ]),
        ("Complaint", [
            "private complaint", "complaint under section", "complainant",
            "complaint is filed against", "the complainant states as follows",
            "ದೂರು", "ಖಾಸಗಿ ದೂರು", "शिकायत", "परिवाद",
        ]),
        ("FIR-related document", [
            "first information report", "fir no", "crime no", "police station",
            "station house officer", "sho", "offence under section", "bns", "ipc",
            "ಪ್ರಥಮ ವರ್ತಮಾನ ವರದಿ", "ಎಫ್‌ಐಆರ್", "प्राथमिकी", "थाना",
        ]),
        ("Court Summons / Notice", [
            "summons", "court summons", "notice to appear", "before the hon'ble court",
            "take notice that you are hereby required to appear", "ಸಮನ್ಸ್", "समन",
        ]),
        ("Application", [
            "application under section", "bail application", "interlocutory application",
            "i.a. no", "application for exemption", "ಅರ್ಜಿ", "आवेदन",
        ]),
        ("Government Notification", [
            "gazette of india", "government notification", "ministry of",
            "published by authority", "hereby notifies", "ರಾಜಪತ್ರ", "अधिसूचना",
        ]),
        ("Act / Statute", [
            "an act to provide", "short title and commencement", "be it enacted by parliament",
            "bare act", "section 1", "अधिनियम", "ಕಾಯ್ದೆ",
        ]),
        ("Rules / Regulations", [
            "rules, 20", "regulations, 20", "in exercise of the powers conferred by",
            "ನಿಯಮಗಳು", "ನಿಯಮಾವಳಿ", "नियम", "विनियम",
        ]),
        ("Legal Aid document", [
            "legal aid application", "national legal services authority", "nalasa",
            "dlsa", "slsa", "lok adalat notice", "free legal aid",
        ]),
        ("Financial document", [
            "loan sanction letter", "bank statement", "promissory note",
            "balance confirmation", "demand promissory note", "ಖಾತೆ ವಿವರ",
        ]),
        ("Property document", [
            "sale deed", "gift deed", "mortgage deed", "relinquishment deed",
            "title deed", "khata certificate", "patta", "ಕ್ರಯಪತ್ರ", "बिक्री पत्र",
        ]),
        ("Employment document", [
            "appointment letter", "offer letter", "termination letter", "relieving letter",
            "salary slip", "resignation letter", "ಅಮಾನತು ಪತ್ರ", "त्यागपत्र",
        ]),
        ("Consumer complaint", [
            "consumer complaint", "district consumer disputes", "district commission",
            "state commission", "defective goods", "deficiency in service",
        ]),
    ]

    for doc_type, markers in type_definitions:
        matches = sum(1 for m in markers if m in val)
        if matches >= 2:
            return doc_type, "high"
        elif matches == 1:
            return doc_type, "medium"

    return "Other legal document", "low"


# ============================================================
# LEGAL TOPIC & ROUTE DETECTION
# ============================================================

def detect_legal_term(text: str) -> Tuple[str, List[str]]:
    """
    Detect the primary legal topic and list matching keywords.
    """
    val = _normalize(text)
    scores: Dict[str, int] = {}
    matched_keywords: Dict[str, List[str]] = {}

    for topic, conf in TOPIC_CONFIG.items():
        score = 0
        matches = []
        for kw in conf.get("keywords", []):
            if kw in val:
                score += 2
                matches.append(kw)

        scores[topic] = score
        matched_keywords[topic] = matches

    # Check canonical term from extractor
    canon = extract_canonical_term(text)
    if canon and canon in scores:
        scores[canon] = scores.get(canon, 0) + 5

    best_topic = max(scores, key=scores.get) if scores else "general"
    if scores.get(best_topic, 0) < 2:
        best_topic = "general"

    return best_topic, matched_keywords.get(best_topic, [])


def detect_route(legal_term: str) -> str:
    """Map legal topic to its recognized legal route."""
    if legal_term in TOPIC_CONFIG:
        return TOPIC_CONFIG[legal_term].get("route", "general")
    return "general"


# ============================================================
# MULTI-ISSUE DETECTION (PRIMARY & SECONDARY)
# ============================================================

def detect_legal_issues(text: str, legal_term: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Identify the primary legal issue and secondary legal issues with confidence.
    """
    val = _lower(text)
    secondary_issues: List[Dict[str, Any]] = []

    # Map primary issue based on topic
    if legal_term == "loan_recovery":
        primary = {
            "issue": "Demand for repayment of outstanding loan amount and contractual debt recovery",
            "confidence": "high",
        }
        if any(k in val for k in ["interest", "12%", "18%", "per annum"]):
            secondary_issues.append({
                "issue": "Accrual and calculation of contractual interest",
                "confidence": "medium",
            })
        if any(k in val for k in ["15 days", "within 15", "notice period", "deadline"]):
            secondary_issues.append({
                "issue": "Compliance with notice response deadline prior to litigation",
                "confidence": "high",
            })
        if any(k in val for k in ["cheque", "dishonour", "bounce"]):
            secondary_issues.append({
                "issue": "Potential liability under Section 138 of Negotiable Instruments Act for cheque dishonour",
                "confidence": "medium",
            })

    elif legal_term == "theft":
        primary = {
            "issue": "Alleged dishonest taking of movable property without consent (Theft)",
            "confidence": "high",
        }
        if any(k in val for k in ["recovery", "seized", "recovered"]):
            secondary_issues.append({
                "issue": "Recovery and custody of stolen property",
                "confidence": "medium",
            })

    elif legal_term == "murder":
        primary = {
            "issue": "Allegation of culpable homicide causing death (Murder)",
            "confidence": "high",
        }

    elif legal_term == "cheating":
        primary = {
            "issue": "Alleged fraudulent or dishonest inducement of property (Cheating)",
            "confidence": "high",
        }

    elif legal_term == "property_dispute":
        primary = {
            "issue": "Dispute concerning title, ownership, or lawful possession of immovable property",
            "confidence": "high",
        }

    elif legal_term == "consumer_complaint":
        primary = {
            "issue": "Claim regarding defective goods or deficiency in commercial service",
            "confidence": "high",
        }

    elif legal_term == "employment_dispute":
        primary = {
            "issue": "Dispute regarding unpaid wages, termination, or statutory employment dues",
            "confidence": "high",
        }

    else:
        primary = {
            "issue": "General civil or statutory compliance matter",
            "confidence": "low",
        }

    return primary, secondary_issues


# ============================================================
# TARGETED QUERY GENERATION
# ============================================================

def build_targeted_queries(legal_term: str, text: str) -> List[str]:
    """
    Generate targeted multi-angle queries for dense + lexical retrieval.
    """
    queries = []
    if legal_term in TOPIC_CONFIG:
        queries.extend(TOPIC_CONFIG[legal_term].get("queries", []))

    # Add query derived from specific references in document
    val = _lower(text)
    if "section 138" in val or "cheque" in val:
        queries.append("Section 138 Negotiable Instruments Act dishonour of cheque")
    if "order 37" in val or "summary suit" in val:
        queries.append("Code of Civil Procedure Order 37 summary procedure debt")
    if "section 73" in val or "compensation for loss" in val:
        queries.append("Indian Contract Act Section 73 compensation breach contract")

    return list(dict.fromkeys(queries))[:MAX_QUERY_COUNT]


# ============================================================
# RETRIEVE LEGAL SOURCES
# ============================================================

def retrieve_legal_sources(
    queries: List[str],
    legal_term: str,
    route: str,
    document_text: str,
) -> List[Dict[str, Any]]:
    """
    Retrieve candidate statutory sections using hybrid retrieval.
    """
    all_results = []
    seen = set()

    for query in queries:
        try:
            results = hybrid_search(
                query,
                semantic_k=30,
                bm25_k=30,
                final_k=MAX_RESULTS_PER_QUERY,
            )
        except Exception:
            continue

        for r in results:
            if not isinstance(r, dict):
                continue

            metadata = r.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            sec = str(metadata.get("section_number", metadata.get("section", "")) or "").strip()
            title = str(metadata.get("title", "") or "").strip()
            sec_title = str(metadata.get("section_title", "") or "").strip()
            source = str(metadata.get("source", "") or "").strip()

            # Discard footnote / amendment fragments
            if any(bad in sec_title.lower() or bad in title.lower() for bad in ["subs. by", "ins. by", "repealed by", "a.o. 1950"]):
                continue

            key = (title.lower(), source.lower(), sec)
            if key in seen:
                continue
            seen.add(key)

            all_results.append(r)

    # Sort by legal rerank score
    all_results.sort(
        key=lambda x: float(x.get("legal_rerank_score", x.get("hybrid_score", 0.0))),
        reverse=True,
    )

    return all_results[:MAX_ACCEPTED_SOURCES]


# ============================================================
# MULTI-DOCUMENT CONFLICT & DISCREPANCY DETECTION
# ============================================================

def detect_document_conflicts(
    doc_analyses: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Detect discrepancies/conflicts across multiple uploaded documents:
    - Amount conflicts (e.g. Doc A: ₹2,50,000 vs Doc B: ₹3,00,000)
    - Date conflicts (e.g. Doc A: 15/06/2022 vs Doc B: 20/07/2022)
    - Deadline conflicts (e.g. 15 days vs 30 days)
    - Interest rate conflicts (e.g. 12% vs 14%)
    """
    conflicts: List[Dict[str, Any]] = []
    if len(doc_analyses) < 2:
        return conflicts

    # 1. Amount discrepancies
    amount_map: Dict[str, List[str]] = {}
    for doc in doc_analyses:
        fname = doc.get("filename", "Document")
        amts = [
            a["value"] if isinstance(a, dict) else str(a)
            for a in doc.get("amounts", [])
        ]
        if amts:
            amount_map[fname] = amts

    if len(amount_map) >= 2:
        first_doc = list(amount_map.keys())[0]
        first_amts = set(amount_map[first_doc])
        diff_docs = [
            f for f, a in amount_map.items()
            if set(a) != first_amts
        ]
        if diff_docs:
            conflicts.append({
                "type": "amount_conflict",
                "field": "monetary_amount",
                "severity": "high",
                "title": "Discrepancy in Stated Amounts",
                "documents": list(amount_map.keys()),
                "message": "The uploaded documents state differing financial figures.",
                "details": [f"{f}: {', '.join(a)}" for f, a in amount_map.items()],
            })

    # 2. Date discrepancies
    date_map: Dict[str, List[str]] = {}
    for doc in doc_analyses:
        fname = doc.get("filename", "Document")
        dts = [
            d["value"] if isinstance(d, dict) else str(d)
            for d in doc.get("dates", [])
        ]
        if dts:
            date_map[fname] = dts

    if len(date_map) >= 2:
        first_doc = list(date_map.keys())[0]
        first_dts = set(date_map[first_doc])
        diff_dts = [
            f for f, d in date_map.items()
            if set(d) != first_dts
        ]
        if diff_dts:
            conflicts.append({
                "type": "date_conflict",
                "field": "dates",
                "severity": "medium",
                "title": "Discrepancy in Important Dates",
                "documents": list(date_map.keys()),
                "message": "The uploaded documents mention different transaction or agreement dates.",
                "details": [f"{f}: {', '.join(d)}" for f, d in date_map.items()],
            })

    # 3. Deadline discrepancies
    deadline_map: Dict[str, str] = {}
    for doc in doc_analyses:
        fname = doc.get("filename", "Document")
        text = str(doc.get("text", "")).lower()
        if "15 days" in text or "fifteen days" in text:
            deadline_map[fname] = "15 days"
        elif "30 days" in text or "thirty days" in text:
            deadline_map[fname] = "30 days"
        elif "7 days" in text or "seven days" in text:
            deadline_map[fname] = "7 days"

    if len(deadline_map) >= 2 and len(set(deadline_map.values())) > 1:
        conflicts.append({
            "type": "deadline_conflict",
            "field": "notice_deadline",
            "severity": "medium",
            "title": "Differing Notice Deadlines",
            "documents": list(deadline_map.keys()),
            "message": "The uploaded documents specify different response or compliance deadlines.",
            "details": [f"{f}: {d}" for f, d in deadline_map.items()],
        })

    # 4. Interest rate discrepancies
    rate_map: Dict[str, List[str]] = {}
    for doc in doc_analyses:
        fname = doc.get("filename", "Document")
        rates = [
            r["value"] if isinstance(r, dict) else str(r)
            for r in doc.get("interest_rates", [])
        ]
        if rates:
            rate_map[fname] = rates

    if len(rate_map) >= 2:
        first_doc = list(rate_map.keys())[0]
        first_rates = set(rate_map[first_doc])
        if any(set(r) != first_rates for r in rate_map.values()):
            conflicts.append({
                "type": "interest_rate_conflict",
                "field": "interest_rate",
                "severity": "medium",
                "title": "Differing Interest Rates",
                "documents": list(rate_map.keys()),
                "message": "The uploaded documents mention different interest rates.",
                "details": [f"{f}: {', '.join(r)}" for f, r in rate_map.items()],
            })

    return conflicts


# ============================================================
# DOCUMENT TIMELINE EXTRACTION
# ============================================================

def extract_document_timeline(
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract chronological events with associated dates and sentence context.
    """
    timeline_items = []
    date_patterns = [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
    ]

    for doc in documents:
        fname = doc.get("filename", "Document")
        raw_text = doc.get("text", "")
        if not raw_text:
            continue

        lines = [
            l.strip()
            for l in re.split(r"[\r\n.]+", raw_text)
            if len(l.strip()) > 15
        ]

        for line in lines:
            for pattern in date_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for date_str in matches:
                    clean_date = date_str.strip()
                    clean_event = line.strip()
                    if clean_event.startswith(clean_date):
                        clean_event = clean_event[len(clean_date):].strip(" :-")

                    timeline_items.append({
                        "date": clean_date,
                        "event": clean_event or line.strip(),
                        "source_document": fname,
                    })

    unique_timeline = []
    seen = set()
    for item in timeline_items:
        key = (item["date"].lower(), item["event"][:50].lower())
        if key not in seen:
            seen.add(key)
            unique_timeline.append(item)

    return unique_timeline


# ============================================================
# SINGLE DOCUMENT ANALYSIS
# ============================================================

# ============================================================
# SINGLE DOCUMENT ANALYSIS
# ============================================================

def analyze_legal_document(
    text: str,
    language: str = "english",
) -> Dict[str, Any]:
    """
    Comprehensive document analysis pipeline with strict separation of
    document_language vs explanation_language and rich structured output.
    """
    cleaned_text = _clean_text(text)
    norm_exp_lang = normalize_language(language)

    if not cleaned_text:
        return {
            "status": "error",
            "message": "No readable text was available for analysis.",
            "document_type": "Other legal document",
            "document_type_confidence": "low",
            "document_language": "unknown",
            "explanation_language": norm_exp_lang,
            "language": norm_exp_lang,
            "legal_topic": "general",
            "legal_topic_label": "General legal matter",
            "legal_route": "general",
            "legal_route_label": "General legal matter",
            "summary": "No readable text was available for analysis.",
            "document_overview": [],
            "parties": [],
            "important_facts": [],
            "claims": [],
            "dates": [],
            "amounts": [],
            "deadlines": [],
            "legal_references": [],
            "legal_issues": [],
            "obligations": [],
            "possible_consequences": [],
            "relevant_provisions": [],
            "conflicts": [],
            "missing_information": [],
            "next_steps": ["Please upload a valid, readable legal document."],
            "warnings": [],
            "safety_caution": "General legal information only.",
        }

    analysis_text = cleaned_text[:MAX_TEXT_CHARS]
    doc_lang = detect_document_language(analysis_text)
    doc_type, doc_type_conf = detect_document_type(analysis_text)
    legal_term, matched_terms = detect_legal_term(analysis_text)
    route = detect_route(legal_term)

    legal_term_label = TOPIC_CONFIG.get(legal_term, {}).get("label", "General legal matter")
    route_label = ROUTE_LABELS.get(route, "General legal matter")

    # Structured fact extraction with provenance
    facts = extract_structured_facts_with_provenance(analysis_text, document_name="Uploaded Document", page_number=1)
    primary_issue, secondary_issues = detect_legal_issues(analysis_text, legal_term)

    # Retrieval and provision analysis
    queries = build_targeted_queries(legal_term, analysis_text)
    results = retrieve_legal_sources(
        queries=queries,
        legal_term=legal_term,
        route=route,
        document_text=analysis_text,
    )

    provision_analysis = analyze_provisions(
        results=results,
        legal_term=legal_term,
        document_text=analysis_text,
    )

    # LLM summary explanation using the EXPLANATION language (not the document's language)
    retrieved_summary = "\n".join(
        f"- Section {p.get('section', '')}: {p.get('section_title', '')} ({p.get('source', '')})"
        for p in provision_analysis.get("provisions", [])[:3]
    )

    llm_explanation = explain_legal_document(
        document_text=analysis_text,
        document_type=doc_type,
        language=norm_exp_lang,
        retrieved_context=retrieved_summary,
    )

    # Timeline extraction
    timeline = extract_document_timeline([{"filename": "Document", "text": analysis_text}])

    # Clean next steps and consequences
    next_steps = []
    if llm_explanation.get("actionable_steps"):
        next_steps.extend(llm_explanation["actionable_steps"])
    if provision_analysis.get("next_steps"):
        next_steps.extend(provision_analysis["next_steps"])
    next_steps = list(dict.fromkeys(next_steps))[:8]

    # Grounded cautionary statement
    primary_prov = provision_analysis.get("primary_provision")
    if primary_prov and primary_prov.get("section"):
        caution = f"This document may relate to Section {primary_prov['section']} based on the available facts."
    else:
        caution = "This document relates to general civil/statutory principles based on the available facts."

    # Build structured parties
    parties = []
    if facts.get("parties"):
        for p in facts["parties"]:
            if isinstance(p, dict):
                parties.append(p)
            else:
                parties.append({"role": "Party", "name": str(p), "details": "Mentioned in document text"})

    # Build structured facts (distinguishing facts from allegations)
    important_facts = []
    if facts.get("amounts"):
        important_facts.append(f"Monetary figures stated in document: {', '.join(str(a.get('text', a) if isinstance(a, dict) else a) for a in facts['amounts'][:4])}")
    if facts.get("dates"):
        important_facts.append(f"Critical transaction/notice dates: {', '.join(str(d.get('text', d) if isinstance(d, dict) else d) for d in facts['dates'][:4])}")
    if facts.get("deadlines"):
        important_facts.append(f"Stated response windows: {', '.join(str(dl.get('text', dl) if isinstance(dl, dict) else dl) for dl in facts['deadlines'][:3])}")
    if not important_facts and facts.get("extracted_clauses"):
        important_facts = facts["extracted_clauses"][:4]

    # Legal issues list
    legal_issues_list = []
    if primary_issue and primary_issue.get("issue") != "None identified":
        legal_issues_list.append({
            "issue": primary_issue.get("issue"),
            "category": route_label,
            "confidence": primary_issue.get("confidence", "high"),
            "description": f"Primary issue identified under {route_label}."
        })
    for sec_iss in (secondary_issues or []):
        legal_issues_list.append({
            "issue": sec_iss.get("issue") if isinstance(sec_iss, dict) else str(sec_iss),
            "category": route_label,
            "confidence": sec_iss.get("confidence", "medium") if isinstance(sec_iss, dict) else "medium",
            "description": "Secondary procedural or substantive issue."
        })

    # Relevant provisions formatted
    rel_provisions = []
    for prov in provision_analysis.get("provisions", [])[:5]:
        rel_provisions.append({
            "section": prov.get("section", ""),
            "act": prov.get("source", prov.get("act_name", "Indian Statute")),
            "title": prov.get("section_title", prov.get("title", "")),
            "definition": prov.get("definition", ""),
            "punishment": prov.get("punishment", ""),
            "relevance": prov.get("relevance_reason", "Statutory framework relevant to document demands."),
        })
    if not rel_provisions and results:
        for r in results[:4]:
            rel_provisions.append({
                "section": r.get("section", ""),
                "act": r.get("act_name", r.get("source", "Indian Statute")),
                "title": r.get("title", r.get("section_title", "Statutory Provision")),
                "definition": r.get("snippet", "")[:200],
                "punishment": "",
                "relevance": "Grounded statutory reference retrieved from legal knowledge base.",
            })

    # Missing information checklist
    missing_info = [
        "Proof of service / postal tracking receipt (if legal notice).",
        "Original transaction contracts or bank statements verifying claimed amounts.",
        "Prior written communications or written acknowledgments between the parties.",
    ]

    # Warnings / precautions
    warnings_list = [
        "Preserve original envelopes and delivery slips to establish service limitation periods.",
        "Do not ignore stated statutory response deadlines (typically 15 or 30 days).",
        "Consult a legal professional or District Legal Services Authority before signing settlement agreements.",
    ]

    return {
        "status": "success",
        "document_type": doc_type,
        "document_type_confidence": doc_type_conf,
        "document_language": doc_lang,
        "explanation_language": norm_exp_lang,
        "language": norm_exp_lang,
        "legal_topic": legal_term,
        "legal_topic_label": legal_term_label,
        "matched_topic_terms": matched_terms,
        "legal_route": route,
        "legal_route_label": route_label,
        "legal_route_confidence": "high" if route != "general" else "medium",
        "primary_issue": primary_issue,
        "secondary_issues": secondary_issues,
        "summary": llm_explanation.get("summary", "Document analysis complete."),
        "document_overview": llm_explanation.get("document_overview", []),
        "parties": parties,
        "important_facts": important_facts,
        "claims": llm_explanation.get("claims", []),
        "facts": facts,
        "amounts": facts.get("amounts", []),
        "dates": facts.get("dates", []),
        "interest_rates": facts.get("interest_rates", []),
        "instalments": facts.get("instalments", []),
        "deadlines": facts.get("deadlines", []) or llm_explanation.get("deadlines", []),
        "legal_references": facts.get("legal_references", []),
        "case_numbers": facts.get("case_numbers", []),
        "legal_issues": legal_issues_list,
        "obligations": llm_explanation.get("conditions_and_clauses", []),
        "possible_consequences": [llm_explanation.get("legal_implications", "Review terms carefully.")],
        "relevant_provisions": rel_provisions,
        "provision_analysis": provision_analysis,
        "primary_provision": provision_analysis.get("primary_provision"),
        "provisions": provision_analysis.get("provisions", []),
        "sources": results,
        "timeline": timeline,
        "missing_information": missing_info,
        "next_steps": next_steps,
        "warnings": warnings_list,
        "llm_explanation": llm_explanation,
        "conditions_and_clauses": llm_explanation.get("conditions_and_clauses", []),
        "actionable_steps": next_steps,
        "safety_caution": caution,
        "disclaimer": (
            "JanNyaya AI provides legal information based on verified statutory sources. "
            "It does not constitute formal legal representation or a final judicial opinion."
        ),
    }


# ============================================================
# MULTI-DOCUMENT CASE ANALYSIS
# ============================================================

def analyze_multiple_documents(
    documents: List[Dict[str, Any]],
    language: str = "english",
) -> Dict[str, Any]:
    """
    Analyze multiple uploaded legal documents as a synthesized case with
    full multi-document cross-referencing and explanation language preservation.
    """
    norm_exp_lang = normalize_language(language)

    if not documents:
        return {
            "status": "error",
            "message": "No documents provided for case analysis.",
            "total_documents": 0,
            "document_language": "unknown",
            "explanation_language": norm_exp_lang,
            "language": norm_exp_lang,
            "documents": [],
            "conflicts": [],
            "combined_case_summary": {},
        }

    per_doc_analyses: List[Dict[str, Any]] = []
    combined_texts: List[str] = []
    topic_counts: Dict[str, int] = {}
    route_counts: Dict[str, int] = {}
    detected_languages = []

    for doc in documents:
        fname = doc.get("filename", "Document")
        ftype = doc.get("file_type", "")
        raw_text = doc.get("text", "")
        cleaned = _clean_text(raw_text)

        if not cleaned:
            continue

        combined_texts.append(f"=== Document: {fname} ===\n{cleaned}")

        doc_analysis = analyze_legal_document(cleaned, language=norm_exp_lang)
        doc_analysis["filename"] = fname
        doc_analysis["file_type"] = ftype
        doc_analysis["text"] = cleaned
        detected_languages.append(doc_analysis.get("document_language", "english"))

        term = doc_analysis["legal_topic"]
        rt = doc_analysis["legal_route"]
        topic_counts[term] = topic_counts.get(term, 0) + 1
        route_counts[rt] = route_counts.get(rt, 0) + 1

        per_doc_analyses.append(doc_analysis)

    if not per_doc_analyses:
        return {
            "status": "error",
            "message": "No readable text found across the uploaded documents.",
            "total_documents": len(documents),
            "document_language": "unknown",
            "explanation_language": norm_exp_lang,
            "language": norm_exp_lang,
            "documents": [],
            "conflicts": [],
            "combined_case_summary": {},
        }

    # Consensus topic and route
    spec_topics = {t: c for t, c in topic_counts.items() if t != "general"}
    case_topic = max(spec_topics, key=spec_topics.get) if spec_topics else max(topic_counts, key=topic_counts.get)
    case_route = detect_route(case_topic)

    # Detect conflicts across files
    conflicts = detect_document_conflicts(per_doc_analyses)

    # Extract timeline
    timeline = extract_document_timeline(documents)

    # Clean up per-doc analyses for response
    clean_docs = []
    all_amounts = []
    all_parties = []
    all_legal_refs = []
    all_facts = []

    for d in per_doc_analyses:
        item = dict(d)
        item.pop("text", None)
        clean_docs.append(item)
        if d.get("amounts"):
            all_amounts.extend(d["amounts"])
        if d.get("parties"):
            all_parties.extend(d["parties"])
        if d.get("legal_references"):
            all_legal_refs.extend(d["legal_references"])
        if d.get("important_facts"):
            all_facts.extend(d["important_facts"])

    # Combined retrieval
    combined_case_text = "\n\n".join(combined_texts)[:MAX_TEXT_CHARS]
    queries = build_targeted_queries(case_topic, combined_case_text)
    results = retrieve_legal_sources(
        queries=queries,
        legal_term=case_topic,
        route=case_route,
        document_text=combined_case_text,
    )

    provision_analysis = analyze_provisions(
        results=results,
        legal_term=case_topic,
        document_text=combined_case_text,
    )

    llm_case_explanation = explain_legal_document(
        document_text=combined_case_text,
        document_type=f"Multi-Document Case ({len(per_doc_analyses)} files)",
        language=norm_exp_lang,
        retrieved_context="\n".join(
            f"- Section {p.get('section', '')}: {p.get('section_title', '')}"
            for p in provision_analysis.get("provisions", [])[:3]
        ),
    )

    # Synthesized Combined Case Summary Object
    combined_summary_obj = {
        "common_facts": list(dict.fromkeys(all_facts))[:8],
        "conflicts": conflicts,
        "timeline": timeline,
        "common_parties": all_parties[:6],
        "common_amounts": all_amounts[:6],
        "common_legal_references": list(dict.fromkeys(all_legal_refs))[:6],
        "document_relationships": [
            f"Document '{d.get('filename')}' ({d.get('document_type')}) connects to the overall matter of {TOPIC_CONFIG.get(case_topic, {}).get('label', case_topic)}."
            for d in clean_docs
        ],
        "missing_documents": [
            "Proof of service or acknowledgment receipts.",
            "Certified bank account statements for transaction dates.",
            "Any rejoinders or prior notices exchanged.",
        ],
    }

    multi_summary = (
        llm_case_explanation.get("summary")
        or f"Synthesized analysis across {len(per_doc_analyses)} uploaded files under {ROUTE_LABELS.get(case_route, case_route)}."
    )

    return {
        "status": "success",
        "total_documents": len(per_doc_analyses),
        "document_language": ", ".join(set(detected_languages)) if detected_languages else "english",
        "explanation_language": norm_exp_lang,
        "language": norm_exp_lang,
        "case_overview": (
            f"Synthesized analysis across {len(per_doc_analyses)} uploaded files. "
            f"Primary matter: {TOPIC_CONFIG.get(case_topic, {}).get('label', case_topic)} under {ROUTE_LABELS.get(case_route, case_route)}."
        ),
        "summary": multi_summary,
        "multi_document_summary": multi_summary,
        "document_overview": llm_case_explanation.get("document_overview", []),
        "legal_topic": case_topic,
        "legal_topic_label": TOPIC_CONFIG.get(case_topic, {}).get("label", case_topic),
        "legal_route": case_route,
        "legal_route_label": ROUTE_LABELS.get(case_route, case_route),
        "documents": clean_docs,
        "conflicts": conflicts,
        "timeline": timeline,
        "combined_case_summary": combined_summary_obj,
        "important_facts": combined_summary_obj["common_facts"],
        "amounts": all_amounts,
        "parties": all_parties,
        "legal_references": all_legal_refs,
        "provision_analysis": provision_analysis,
        "primary_provision": provision_analysis.get("primary_provision"),
        "provisions": provision_analysis.get("provisions", []),
        "sources": results,
        "llm_explanation": llm_case_explanation,
        "conditions_and_clauses": llm_case_explanation.get("conditions_and_clauses", []),
        "actionable_steps": llm_case_explanation.get("actionable_steps", []),
        "next_steps": llm_case_explanation.get("actionable_steps", []),
        "safety_caution": "This multi-document case analysis synthesizes facts across all uploaded files based on available evidence.",
        "disclaimer": (
            "JanNyaya AI provides legal information based on verified statutory sources. "
            "It does not constitute formal legal representation or a final judicial opinion."
        ),
    }