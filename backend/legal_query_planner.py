"""
JanNyaya AI - Advanced Legal Query Planner

Responsibilities:
1. Parse user question / case query for legal intent and structure.
2. Classify into 14 standardized Query Types:
   - DIRECT_FACT, DEFINITION, PUNISHMENT, LEGAL_PROCEDURE, RIGHTS,
     OBLIGATIONS, DOCUMENT_QUESTION, CASE_QUESTION, COMPARISON,
     CONTRADICTION, MULTI_HOP_LEGAL_QUERY, SOURCE_LOOKUP,
     SECTION_LOOKUP, GENERAL_LEGAL_INFORMATION
3. Multi-domain and legal route detection:
   - criminal, civil_contractual, property, family, consumer,
     employment, cyber, financial, motor_vehicle, legal_aid, governance
4. Entity, section, and temporal/version extraction (e.g. 2020 vs 2024 for IPC vs BNS).
5. Build structured retrieval plans with sub-queries and required evidence categories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================
# QUERY TYPES
# ============================================================

QUERY_TYPES = {
    "DIRECT_FACT": "Direct factual legal inquiry",
    "DEFINITION": "Statutory definition of an offence, term, or concept",
    "PUNISHMENT": "Penal consequences, imprisonment, fine, or bailability",
    "LEGAL_PROCEDURE": "Step-by-step court, police, or administrative procedure (e.g. FIR, notice, trial)",
    "RIGHTS": "Statutory citizen rights, legal remedies, or constitutional protections",
    "OBLIGATIONS": "Statutory duties, contractual liabilities, or compliance requirements",
    "DOCUMENT_QUESTION": "Question specifically grounded on an uploaded legal document",
    "CASE_QUESTION": "Case-level question combining document facts with legal provisions",
    "COMPARISON": "Comparative evaluation of two laws, sections, or documents",
    "CONTRADICTION": "Identifying conflict or discrepancy between claims, amounts, or dates",
    "MULTI_HOP_LEGAL_QUERY": "Complex multi-step query requiring substantive law + procedure + limitation",
    "SOURCE_LOOKUP": "Locating the exact official Act, Gazette, or court source",
    "SECTION_LOOKUP": "Direct lookup of a specific section number (e.g. BNS 303, NIA 138)",
    "GENERAL_LEGAL_INFORMATION": "General informational question about Indian law",
}


# ============================================================
# DOMAIN & ROUTE DICTIONARY
# ============================================================

DOMAIN_CONFIG: Dict[str, Dict[str, Any]] = {
    "criminal": {
        "route": "criminal",
        "acts": ["Bharatiya Nyaya Sanhita, 2023", "Bharatiya Nagarik Suraksha Sanhita, 2023", "Bharatiya Sakshya Adhiniyam, 2023", "POCSO Act, 2012", "Juvenile Justice Act, 2015"],
        "keywords": [
            "theft", "murder", "kill", "cheating", "fraud", "fir", "arrest", "bail", "bns", "bnss", "bsa",
            "ipc", "crpc", "assault", "kidnap", "extortion", "cybercrime", "rape", "pocso", "police",
            "ಕಳ್ಳತನ", "ಕೊಲೆ", "ವಂಚನೆ", "ಎಫ್‌ಐಆರ್", "ಜಾಮೀನು", "ಬಂಧನ", "ಚೋರಿ", "हत्या", "धोखाधड़ी", "गिरफ्तारी", "जमानत"
        ],
    },
    "civil_contractual": {
        "route": "civil_contractual",
        "acts": ["Indian Contract Act, 1872", "Specific Relief Act, 1963", "Code of Civil Procedure, 1908", "Limitation Act, 1963"],
        "keywords": [
            "contract", "agreement", "breach", "loan", "debt", "repay", "borrower", "lender", "recovery",
            "damages", "specific performance", "injunction", "cpc", "limitation", "plaint", "written statement",
            "ಒಪ್ಪಂದ", "ಸಾಲ", "ವಸೂಲಾತಿ", "ನಷ್ಟಪರಿಹಾರ", "अनुबंध", "ऋण", "लोन", "उधारी", "हर्जाना"
        ],
    },
    "commercial": {
        "route": "commercial",
        "acts": ["Negotiable Instruments Act, 1881", "Companies Act, 2013", "Arbitration and Conciliation Act, 1996"],
        "keywords": [
            "cheque", "bounce", "dishonour", "section 138", "company", "director", "shareholder", "board",
            "arbitration", "arbitrator", "award", "conciliation", "insolvency", "commercial",
            "ಚೆಕ್", "ಬೌನ್ಸ್", "ಕಂಪನಿ", "ಮಧ್ಯಸ್ಥಿಕೆ", "चेक", "बाउंस", "कंपनी", "मध्यस्थता"
        ],
    },
    "property": {
        "route": "property",
        "acts": ["Transfer of Property Act, 1882", "Real Estate Regulation and Development Act, 2016", "Specific Relief Act, 1963"],
        "keywords": [
            "property", "land", "flat", "plot", "sale deed", "mortgage", "lease", "rent", "tenant", "landlord",
            "rera", "builder", "possession", "title", "encumbrance", "registration", "eviction",
            "ಆಸ್ತಿ", "ಜಮೀನು", "ಬಾಡಿಗೆ", "ಖರೀದಿ ಪತ್ರ", "ರೆರಾ", "संपत्ति", "जमीन", "किराया", "बैनामा", "मकान मालिक"
        ],
    },
    "family": {
        "route": "family",
        "acts": ["Protection of Women from Domestic Violence Act, 2005", "Hindu Marriage Act, 1955", "Hindu Succession Act, 1956"],
        "keywords": [
            "marriage", "divorce", "maintenance", "alimony", "domestic violence", "custody", "succession",
            "inheritance", "ancestral property", "partition", "wife", "husband", "child custody",
            "ವಿವಾಹ", "ವಿಚ್ಛೇದನ", "ಜೀವನಾಂಶ", "ಕೌಟುಂಬಿಕ ಹಿಂಸೆ", "ಪಾಲು", "विवाह", "तलाक", "गुजारा भत्ता", "घरेलू हिंसा", "पैतृक संपत्ति"
        ],
    },
    "consumer": {
        "route": "consumer",
        "acts": ["Consumer Protection Act, 2019"],
        "keywords": [
            "consumer", "defective", "deficiency in service", "refund", "unfair trade", "e-daakhil", "consumer forum",
            "compensation", "warranty", "guarantee", "product liability",
            "ಗ್ರಾಹಕ", "ದೋಷಪೂರಿತ", "ಮರುಪಾವತಿ", "ನಷ್ಟ ಪರಿಹಾರ", "उपभोक्ता", "खराब माल", "रिफंड", "उपभोक्ता फोरम"
        ],
    },
    "cyber": {
        "route": "cyber",
        "acts": ["Information Technology Act, 2000", "Bharatiya Nyaya Sanhita, 2023"],
        "keywords": [
            "cyber", "hack", "hacking", "otp", "phishing", "data theft", "identity theft", "section 66",
            "electronic evidence", "digital signature", "online fraud", "deepfake",
            "ಸೈಬರ್", "ಹ್ಯಾಕಿಂಗ್", "ಆನ್‌ಲೈನ್ ವಂಚನೆ", "साइबर", "हैकिंग", "ऑनलाइन फ्रॉड", "ओटीपी"
        ],
    },
    "employment": {
        "route": "employment",
        "acts": ["Code on Wages, 2019", "Payment of Gratuity Act, 1972"],
        "keywords": [
            "salary", "wages", "gratuity", "employment", "employee", "employer", "termination", "bonus",
            "unpaid salary", "labour", "workplace", "severance",
            "ವೇತನ", "ಉದ್ಯೋಗ", "ಗ್ರಾಚ್ಯುಟಿ", "ಸಂಬಳ", "वेतन", "नौकरी", "ग्रेच्युटी", "मजदूरी"
        ],
    },
    "traffic": {
        "route": "traffic",
        "acts": ["Motor Vehicles Act, 1988"],
        "keywords": [
            "traffic", "motor vehicle", "license", "accident", "challan", "drunk driving", "insurance",
            "hit and run", "mact", "overspeeding",
            "ಸಂಚಾರ", "ವಾಹನ", "ಅಪಘಾತ", "ಚಾಲನಾ ಪರವಾನಗಿ", "ट्रैफिक", "वाहन", "दुर्घटना", "चालान"
        ],
    },
    "legal_aid": {
        "route": "legal_aid",
        "acts": ["Legal Services Authorities Act, 1987"],
        "keywords": [
            "legal aid", "free lawyer", "nalsa", "dlsa", "slsa", "lok adalat", "15100", "poor citizen",
            "ಉಚಿತ ಕಾನೂನು ನೆರವು", "ಲೋಕ ಅದಾಲತ್", "ಮುಫತ್ ವಕೀಲ", "मुफ्त कानूनी सहायता", "लोक अदालत", "नालसा"
        ],
    },
    "governance": {
        "route": "governance",
        "acts": ["Right to Information Act, 2005"],
        "keywords": [
            "rti", "right to information", "public authority", "pio", "information officer", "30 days",
            "ಮಾಹಿತಿ ಹಕ್ಕು", "ಆರ್‌ಟಿಐ", "सूचना का अधिकार", "आरटीआई"
        ],
    },
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class LegalEntity:
    name: str
    entity_type: str  # Act, Section, Date, Amount, Person, Org, Court
    raw_text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
        }


@dataclass
class QueryPlan:
    raw_query: str
    query_type: str
    primary_domain: str
    secondary_domains: List[str] = field(default_factory=list)
    primary_route: str = "general"
    detected_acts: List[str] = field(default_factory=list)
    detected_sections: List[str] = field(default_factory=list)
    entities: List[LegalEntity] = field(default_factory=list)
    temporal_version: Optional[str] = None  # e.g., "historical_ipc" vs "current_bns"
    is_multi_hop: bool = False
    sub_queries: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    research_plan_summary: List[str] = field(default_factory=list)
    language: str = "english"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "query_type": self.query_type,
            "primary_domain": self.primary_domain,
            "secondary_domains": self.secondary_domains,
            "primary_route": self.primary_route,
            "detected_acts": self.detected_acts,
            "detected_sections": self.detected_sections,
            "entities": [e.to_dict() for e in self.entities],
            "temporal_version": self.temporal_version,
            "is_multi_hop": self.is_multi_hop,
            "sub_queries": self.sub_queries,
            "required_evidence": self.required_evidence,
            "research_plan_summary": self.research_plan_summary,
            "language": self.language,
        }


# ============================================================
# QUERY PLANNER CLASS
# ============================================================

class LegalQueryPlanner:
    """Lightweight deterministic & rules-guided legal query planner."""

    SECTION_REGEX = re.compile(
        r"(?:section|sec|u/s|धारा|ಸೆಕ್ಷನ್)\s*([0-9]+[A-Za-z]*)",
        re.IGNORECASE,
    )

    ACT_MAP = {
        "bns": "Bharatiya Nyaya Sanhita, 2023",
        "bnss": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "bsa": "Bharatiya Sakshya Adhiniyam, 2023",
        "ipc": "Indian Penal Code, 1860 (Historical / Transitional)",
        "crpc": "Code of Criminal Procedure, 1973 (Historical / Transitional)",
        "iea": "Indian Evidence Act, 1872 (Historical / Transitional)",
        "cpc": "Code of Civil Procedure, 1908",
        "contract act": "Indian Contract Act, 1872",
        "ni act": "Negotiable Instruments Act, 1881",
        "negotiable instruments": "Negotiable Instruments Act, 1881",
        "consumer protection": "Consumer Protection Act, 2019",
        "it act": "Information Technology Act, 2000",
        "motor vehicles": "Motor Vehicles Act, 1988",
        "transfer of property": "Transfer of Property Act, 1882",
        "specific relief": "Specific Relief Act, 1963",
        "limitation act": "Limitation Act, 1963",
        "arbitration": "Arbitration and Conciliation Act, 1996",
        "companies act": "Companies Act, 2013",
        "rera": "Real Estate Regulation and Development Act, 2016",
        "domestic violence": "Protection of Women from Domestic Violence Act, 2005",
        "senior citizens": "Maintenance of Senior Citizens Act, 2007",
        "rti": "Right to Information Act, 2005",
        "pocso": "POCSO Act, 2012",
    }

    @classmethod
    def detect_language(cls, text: str) -> str:
        if not text:
            return "english"
        devanagari = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        kannada = sum(1 for c in text if 0x0C80 <= ord(c) <= 0x0CFF)
        if kannada > devanagari and kannada > 0:
            return "kannada"
        if devanagari > 0:
            return "hindi"
        return "english"

    @classmethod
    def extract_sections(cls, text: str) -> List[str]:
        matches = cls.SECTION_REGEX.findall(text)
        return list(dict.fromkeys(m.strip() for m in matches if m.strip()))

    @classmethod
    def extract_acts(cls, text: str) -> List[str]:
        found = []
        lower = text.lower()
        for key, full_name in cls.ACT_MAP.items():
            if key in lower:
                found.append(full_name)
        return list(dict.fromkeys(found))

    @classmethod
    def extract_entities(cls, text: str) -> List[LegalEntity]:
        entities = []
        # Sections
        for sec in cls.extract_sections(text):
            entities.append(LegalEntity(name=f"Section {sec}", entity_type="Section", raw_text=sec))
        # Acts
        for act in cls.extract_acts(text):
            entities.append(LegalEntity(name=act, entity_type="Act", raw_text=act))
        # Monetary amounts (e.g. ₹ 50,000 or Rs. 1,00,000 or 50000 rupees)
        amount_matches = re.findall(r"(?:₹|rs\.?|inr|rupees?)\s*([0-9,]+(?:\.[0-9]{2})?)", text, re.IGNORECASE)
        for amt in amount_matches:
            clean_amt = amt.replace(",", "").strip()
            if clean_amt:
                entities.append(LegalEntity(name=f"₹{clean_amt}", entity_type="Amount", raw_text=amt))
        # Dates / Years
        year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        for yr in year_matches:
            entities.append(LegalEntity(name=yr, entity_type="Year", raw_text=yr))
        return entities

    @classmethod
    def detect_domains(cls, text: str) -> Tuple[str, List[str], str]:
        lower = text.lower()
        scores: Dict[str, int] = {}
        for domain, cfg in DOMAIN_CONFIG.items():
            score = 0
            for kw in cfg["keywords"]:
                if kw in lower:
                    score += 2 if len(kw) > 4 else 1
            if score > 0:
                scores[domain] = score

        if not scores:
            return "general", [], "general"

        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_domains[0][0]
        secondary = [d[0] for d in sorted_domains[1:] if d[1] >= 2]
        route = DOMAIN_CONFIG.get(primary, {}).get("route", "general")
        return primary, secondary, route

    @classmethod
    def detect_temporal_version(cls, text: str) -> Optional[str]:
        lower = text.lower()
        if any(w in lower for w in ["in 2020", "in 2021", "in 2022", "in 2023", "before 2024", "before july 2024", "old law", "ipc", "crpc"]):
            return "historical_pre_july_2024"
        if any(w in lower for w in ["bns", "bnss", "bsa", "2024", "2025", "new criminal law", "current law"]):
            return "current_post_july_2024"
        return None

    @classmethod
    def classify_query_type(cls, text: str, has_doc_context: bool = False) -> str:
        lower = text.lower()
        sections = cls.extract_sections(text)

        if has_doc_context and any(w in lower for w in ["this document", "uploaded file", "notice says", "agreement mentions", "page", "clause"]):
            return "DOCUMENT_QUESTION"

        if any(w in lower for w in ["punishment", "imprisonment", "fine", "jail", "penalty", "bailable", "non-bailable", "ಶಿಕ್ಷೆ", "ಜೈಲು", "ಸಜಾ", "दण्ड", "सजा"]):
            return "PUNISHMENT"

        if any(w in lower for w in ["definition", "what is", "define", "meaning of", "ಎಂದರೇನು", "ಅರ್ಥ", "परिभाषा", "क्या है"]):
            return "DEFINITION"

        if any(w in lower for w in ["procedure", "how to file", "how to register", "process", "steps", "fir process", "ವಿಧಾನ", "ಪ್ರಕ್ರಿಯೆ", "प्रक्रिया", "कैसे दर्ज करें"]):
            return "LEGAL_PROCEDURE"

        if any(w in lower for w in ["rights", "remedy", "can i claim", "entitled", "ಹಕ್ಕು", "ಅಧಿಕಾರ", "अधिकार", "राहत"]):
            return "RIGHTS"

        if any(w in lower for w in ["obligation", "duty", "must i pay", "liable", "ಹೊಣೆಗಾರಿಕೆ", "ಕರ್ತವ್ಯ", "दायित्व", "बाध्य"]):
            return "OBLIGATIONS"

        if any(w in lower for w in ["compare", "difference between", "vs", "versus", "ವ್ಯತ್ಯಾಸ", "ಹೋಲಿಕೆ", "अंतर", "तुलना"]):
            return "COMPARISON"

        if any(w in lower for w in ["conflict", "contradiction", "discrepancy", "differs", "ವಿಸಂಗತಿ", "ವಿರೋಧ", "विरोधाभास"]):
            return "CONTRADICTION"

        if sections and len(text.split()) <= 7:
            return "SECTION_LOOKUP"

        if any(w in lower for w in ["source", "act name", "gazette", "which section", "ಯಾವ ಕಾಯಿದೆ", "कौन सा कानून"]):
            return "SOURCE_LOOKUP"

        # Multi-hop detection: contains multiple distinct legal questions (e.g. debt + limitation + remedy)
        multi_hop_triggers = [
            ("loan" in lower or "debt" in lower or "cheque" in lower) and ("limitation" in lower or "time limit" in lower or "procedure" in lower),
            ("breach" in lower or "default" in lower) and ("remedy" in lower or "court" in lower) and ("period" in lower or "notice" in lower),
            ("property" in lower) and ("registration" in lower or "unregistered" in lower) and ("remedy" in lower or "possession" in lower),
        ]
        if any(multi_hop_triggers):
            return "MULTI_HOP_LEGAL_QUERY"

        if any(w in lower for w in ["facts", "case", "situation"]):
            return "DIRECT_FACT"

        return "GENERAL_LEGAL_INFORMATION"

    @classmethod
    def plan_query(
        cls,
        question: str,
        has_doc_context: bool = False,
    ) -> QueryPlan:
        """Generates a complete QueryPlan from user input."""
        q = (question or "").strip()
        lang = cls.detect_language(q)
        query_type = cls.classify_query_type(q, has_doc_context=has_doc_context)
        primary_domain, secondary_domains, primary_route = cls.detect_domains(q)
        sections = cls.extract_sections(q)
        acts = cls.extract_acts(q)
        entities = cls.extract_entities(q)
        temporal = cls.detect_temporal_version(q)

        is_multi_hop = query_type == "MULTI_HOP_LEGAL_QUERY" or len(secondary_domains) > 0

        # Construct sub-queries if multi-hop
        sub_queries = [q]
        research_plan = []
        required_evidence = ["primary_statutory_text", "punishment_or_remedy_clause"]

        if is_multi_hop:
            sub_queries = []
            lower = q.lower()
            if "loan" in lower or "debt" in lower or "repay" in lower:
                sub_queries.append(f"loan repayment obligation default debt {primary_domain}")
                sub_queries.append("money recovery civil suit summary procedure cpc")
                sub_queries.append("limitation period debt loan recovery 3 years limitation act 1963")
                research_plan = [
                    "1. Identify contractual repayment obligation under Indian Contract Act 1872",
                    "2. Examine debt recovery enforcement procedure under Code of Civil Procedure 1908",
                    "3. Verify statutory 3-year limitation bar under Limitation Act 1963 Article 18/55",
                    "4. Synthesize consolidated grounded legal advice",
                ]
                required_evidence = ["contractual_obligation", "procedural_enforcement", "limitation_period"]
            elif "cheque" in lower or "bounce" in lower:
                sub_queries.append("cheque bounce dishonour Section 138 Negotiable Instruments Act")
                sub_queries.append("30 days statutory demand notice Section 138 NIA")
                sub_queries.append("limitation period filing cheque bounce complaint 142 NIA")
                research_plan = [
                    "1. Check ingredients of Section 138 dishonour of cheque",
                    "2. Verify mandatory 30-day statutory demand notice rule",
                    "3. Check 15-day payment window and 1-month court filing limitation under Section 142",
                ]
                required_evidence = ["section_138_ingredients", "notice_timeline", "section_142_limitation"]
            elif "theft" in lower or "stolen" in lower:
                sub_queries.append("theft offence movable property without consent BNS Section 303")
                sub_queries.append("theft punishment imprisonment fine BNS 303")
                sub_queries.append("zero fir e-fir registration procedure BNSS Section 173")
                research_plan = [
                    "1. Retrieve substantive definition of theft under BNS Section 303(1)",
                    "2. Retrieve statutory punishment under BNS Section 303(2)",
                    "3. Retrieve police procedure for FIR registration under BNSS Section 173",
                ]
                required_evidence = ["offence_definition", "statutory_punishment", "fir_procedure"]
            else:
                sub_queries.append(f"{primary_domain} statutory provisions definition rights")
                sub_queries.append(f"{primary_domain} legal remedies procedure limitation")
                research_plan = [
                    f"1. Retrieve substantive rights under {primary_domain}",
                    f"2. Identify statutory procedural route and applicable limitations",
                ]

        if not research_plan:
            research_plan = [
                f"1. Query understanding: Classify as {query_type} in {primary_domain} domain",
                f"2. Retrieve verified statutory provisions and authority sources",
                f"3. Generate grounded legal explanation in {lang.upper()}",
            ]

        return QueryPlan(
            raw_query=q,
            query_type=query_type,
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            primary_route=primary_route,
            detected_acts=acts,
            detected_sections=sections,
            entities=entities,
            temporal_version=temporal,
            is_multi_hop=is_multi_hop,
            sub_queries=sub_queries if is_multi_hop else [q],
            required_evidence=required_evidence,
            research_plan_summary=research_plan,
            language=lang,
        )


def plan_legal_query(question: str, has_doc_context: bool = False) -> QueryPlan:
    """Convenience helper to plan a legal query."""
    return LegalQueryPlanner.plan_query(question, has_doc_context=has_doc_context)
