"""
JanNyaya AI - Legal Provision Selection & Explanation Service

Purpose
-------
Takes retrieved legal candidates and produces a safer,
frontend-friendly legal provision analysis.

Important:
    - Does not determine guilt.
    - Does not determine liability.
    - Does not invent legal provisions.
    - Does not guarantee applicability.
    - Uses retrieved legal evidence only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

MAX_SOURCE_TEXT_CHARS = 10000
MAX_FIELD_CHARS = 2200
MAX_PROVISIONS = 6

MIN_GENERAL_RELEVANCE = 3.0
MIN_LOAN_RELEVANCE = 8.0


# ============================================================
# LEGAL TOPIC LABELS
# ============================================================

LEGAL_TOPIC_LABELS = {
    "theft": "Theft",
    "murder": "Murder",
    "cheating": "Cheating",
    "fraud": "Fraud",
    "robbery": "Robbery",
    "snatching": "Snatching",
    "kidnapping": "Kidnapping / abduction",
    "assault": "Assault / injury",
    "loan_recovery": "Loan recovery",
    "contract_breach": "Contractual dispute",
    "property_dispute": "Property dispute",
    "family_dispute": "Family / matrimonial matter",
    "consumer_dispute": "Consumer dispute",
    "employment_dispute": "Employment / labour matter",
    "general": "Legal matter",
}


def legal_topic_label(
    legal_term: str,
) -> str:
    return LEGAL_TOPIC_LABELS.get(
        str(legal_term or "general").strip(),
        "Legal matter",
    )


# ============================================================
# TOPIC CONFIGURATION
# ============================================================

TOPIC_CONFIG: Dict[str, Dict[str, Any]] = {

    "theft": {
        "route": "criminal",
        "positive": [
            "theft",
            "dishonestly",
            "movable property",
            "without consent",
            "stolen property",
            "taking property",
            "steals",
            "stole",
        ],
        "strong": [
            "theft",
            "dishonestly",
            "movable property",
            "without consent",
        ],
        "negative": [
            "loan",
            "borrower",
            "repayment",
            "mortgage",
        ],
        "preferred_sections": [
            "303",
            "305",
            "306",
            "307",
        ],
    },

    "murder": {
        "route": "criminal",
        "positive": [
            "murder",
            "culpable homicide",
            "caused death",
            "death",
            "killed",
            "intentional death",
        ],
        "strong": [
            "murder",
            "culpable homicide",
        ],
        "negative": [
            "loan",
            "contract",
            "repayment",
        ],
        "preferred_sections": [
            "101",
            "103",
        ],
    },

    "cheating": {
        "route": "criminal",
        "positive": [
            "cheating",
            "deceiving",
            "deception",
            "dishonestly induces",
            "fraudulent",
            "false representation",
        ],
        "strong": [
            "cheating",
            "deceiving",
            "dishonestly induces",
        ],
        "negative": [],
        "preferred_sections": [
            "318",
            "319",
        ],
    },

    "loan_recovery": {
        "route": "civil_contractual",

        "positive": [
            "loan",
            "loan agreement",
            "loan amount",
            "borrower",
            "lender",
            "repayment",
            "repay",
            "outstanding",
            "outstanding amount",
            "outstanding dues",
            "debt",
            "creditor",
            "debtor",
            "interest",
            "instalment",
            "installment",
            "default",
            "recovery of money",
            "recovery",
            "money payable",
            "payment of debt",
            "payment",
            "principal debtor",
        ],

        "strong": [
            "loan",
            "repayment",
            "borrower",
            "lender",
            "debt",
            "creditor",
            "debtor",
            "principal debtor",
            "interest",
        ],

        "negative": [
            "bailor",
            "bailee",
            "bailment",
            "pawn",
            "pledge",
            "immovable property",
            "mortgage redemption",
            "sale of immovable property",
            "easement",
            "dwelling house",
            "theft",
            "murder",
        ],

        "preferred_title_terms": [
            "loan",
            "debt",
            "debtor",
            "creditor",
            "repayment",
            "payment",
            "money",
            "interest",
            "default",
        ],

        "preferred_sections": [],
    },

    "contract_breach": {
        "route": "civil_contractual",
        "positive": [
            "contract",
            "agreement",
            "breach",
            "promise",
            "obligation",
            "consideration",
            "performance",
            "default",
            "damages",
            "compensation",
        ],
        "strong": [
            "breach",
            "contract",
            "agreement",
            "obligation",
        ],
        "negative": [
            "theft",
            "murder",
        ],
        "preferred_title_terms": [
            "breach",
            "compensation",
            "damages",
            "performance",
            "contract",
        ],
        "preferred_sections": [],
    },

    "property_dispute": {
        "route": "property",
        "positive": [
            "property",
            "ownership",
            "possession",
            "title",
            "land",
            "building",
            "immovable",
            "partition",
            "boundary",
            "sale deed",
        ],
        "strong": [
            "ownership",
            "possession",
            "immovable property",
        ],
        "negative": [
            "loan",
            "borrower",
            "repayment",
        ],
        "preferred_title_terms": [
            "property",
            "possession",
            "ownership",
            "title",
            "immovable",
        ],
        "preferred_sections": [],
    },

    "employment_dispute": {
        "route": "employment",
        "positive": [
            "employee",
            "employer",
            "employment",
            "salary",
            "wages",
            "termination",
            "dismissal",
            "labour",
            "labor",
            "workplace",
        ],
        "strong": [
            "employee",
            "employer",
            "salary",
            "wages",
        ],
        "negative": [
            "theft",
            "stolen",
            "dishonestly took",
        ],
        "preferred_title_terms": [
            "employee",
            "employer",
            "wages",
            "salary",
            "termination",
        ],
        "preferred_sections": [],
    },
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_text(
    text: Any,
) -> str:

    if not text:
        return ""

    value = str(text)

    value = value.replace(
        "\x00",
        " ",
    )

    value = re.sub(
        r"[ \t]+",
        " ",
        value,
    )

    value = re.sub(
        r"\n{3,}",
        "\n\n",
        value,
    )

    return value.strip()


def _lower(
    text: Any,
) -> str:

    return str(
        text or ""
    ).lower()


def _normalize_space(
    text: Any,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(
            text or ""
        ).strip(),
    )


def _unique_strings(
    values: List[str],
) -> List[str]:

    output: List[str] = []
    seen = set()

    for value in values:

        cleaned = _normalize_space(
            value
        )

        if not cleaned:
            continue

        key = cleaned.lower()

        if key in seen:
            continue

        seen.add(key)
        output.append(cleaned)

    return output


# ============================================================
# METADATA
# ============================================================

def _metadata(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    value = result.get(
        "metadata",
        {},
    )

    if isinstance(value, dict):
        return value

    return {}


def get_section_number(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(result)

    return str(
        metadata.get(
            "section_number",
            metadata.get(
                "section",
                "",
            ),
        )
        or ""
    ).strip()


def get_section_title(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(result)

    return _normalize_space(
        metadata.get(
            "section_title",
            "",
        )
    )


def get_source(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(result)

    value = str(
        metadata.get(
            "source",
            "",
        )
        or ""
    ).strip()

    return value or "Unknown source"


def get_document_title(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(result)

    return str(
        metadata.get(
            "title",
            "",
        )
        or ""
    ).strip()


def get_document_type(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(result)

    return str(
        metadata.get(
            "document_type",
            "",
        )
        or ""
    ).strip()


def get_authority(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(result)

    return str(
        metadata.get(
            "authority",
            "",
        )
        or ""
    ).strip()


def get_route(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(result)

    return str(
        metadata.get(
            "route",
            "",
        )
        or ""
    ).strip().lower()


def get_chunk(
    result: Dict[str, Any],
) -> Optional[int]:

    metadata = _metadata(result)

    value = metadata.get(
        "chunk_index",
        None,
    )

    try:

        if value is None:
            return None

        return int(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


def get_score(
    result: Dict[str, Any],
) -> float:

    value = result.get(
        "legal_rerank_score",
        result.get(
            "hybrid_score",
            result.get(
                "score",
                0.0,
            ),
        ),
    )

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0


# ============================================================
# SOURCE TEXT
# ============================================================

def get_source_text(
    result: Dict[str, Any],
) -> str:

    metadata = _metadata(result)

    title = str(
        metadata.get(
            "title",
            "",
        )
    )

    section_title = str(
        metadata.get(
            "section_title",
            "",
        )
    )

    document = str(
        result.get(
            "document",
            "",
        )
    )

    combined = "\n".join(
        [
            title,
            section_title,
            document,
        ]
    )

    return _clean_text(
        combined
    )[:MAX_SOURCE_TEXT_CHARS]


# ============================================================
# SENTENCES
# ============================================================

def _split_sentences(
    text: str,
) -> List[str]:

    if not text:
        return []

    normalized = _normalize_space(
        text
    )

    parts = re.split(
        r"(?<=[.!?])\s+|;\s+",
        normalized,
    )

    return [
        _normalize_space(
            part
        )
        for part in parts
        if _normalize_space(
            part
        )
    ]


# ============================================================
# DEFINITION
# ============================================================

def extract_definition(
    source_text: str,
    section: str,
) -> str:

    text = _clean_text(
        source_text
    )

    if not text:
        return ""

    sentences = _split_sentences(
        text
    )

    punishment_markers = [
        "shall be punished",
        "punishable",
        "imprisonment",
        "liable to fine",
        "fine",
        "community service",
        "rigorous imprisonment",
        "life imprisonment",
        "death",
    ]

    if section == "303":

        collected = []

        for sentence in sentences:

            lower = _lower(
                sentence
            )

            if any(
                marker in lower
                for marker in punishment_markers
            ):
                continue

            if any(
                marker in lower
                for marker in (
                    "theft",
                    "whoever",
                    "movable property",
                    "without that person's consent",
                )
            ):

                collected.append(
                    sentence
                )

        if collected:

            return _normalize_space(
                " ".join(
                    collected
                )
            )[:MAX_FIELD_CHARS]

    if section == "101":

        collected = []

        for sentence in sentences:

            lower = _lower(
                sentence
            )

            if (
                (
                    "murder" in lower
                    or "culpable homicide" in lower
                )
                and not any(
                    marker in lower
                    for marker in punishment_markers
                )
            ):

                collected.append(
                    sentence
                )

        if collected:

            return _normalize_space(
                " ".join(
                    collected
                )
            )[:MAX_FIELD_CHARS]

    if section == "318":

        collected = []

        for sentence in sentences:

            lower = _lower(
                sentence
            )

            if (
                (
                    "cheating" in lower
                    or "deceiving" in lower
                    or "dishonestly induces" in lower
                )
                and not any(
                    marker in lower
                    for marker in punishment_markers
                )
            ):

                collected.append(
                    sentence
                )

        if collected:

            return _normalize_space(
                " ".join(
                    collected
                )
            )[:MAX_FIELD_CHARS]

    collected = []

    for sentence in sentences:

        lower = _lower(
            sentence
        )

        if any(
            marker in lower
            for marker in punishment_markers
        ):
            continue

        if any(
            marker in lower
            for marker in (
                "is said to",
                "means",
                "defined as",
                "whoever",
            )
        ):

            collected.append(
                sentence
            )

        if len(collected) >= 3:
            break

    if not collected:
        return ""

    return _normalize_space(
        " ".join(
            collected
        )
    )[:MAX_FIELD_CHARS]


# ============================================================
# PUNISHMENT / CONSEQUENCE
# ============================================================

def extract_punishment(
    source_text: str,
) -> str:

    text = _clean_text(
        source_text
    )

    if not text:
        return ""

    sentences = _split_sentences(
        text
    )

    markers = [
        "shall be punished",
        "punishable",
        "imprisonment",
        "liable to fine",
        "fine",
        "community service",
        "rigorous imprisonment",
        "life imprisonment",
        "death",
        "damages",
        "compensation",
    ]

    collected = []

    for sentence in sentences:

        lower = _lower(
            sentence
        )

        if any(
            marker in lower
            for marker in markers
        ):

            if len(sentence) >= 20:

                collected.append(
                    sentence
                )

        if len(
            " ".join(collected)
        ) >= MAX_FIELD_CHARS:

            break

    if not collected:
        return ""

    return _normalize_space(
        " ".join(collected)
    )[:MAX_FIELD_CHARS]


# ============================================================
# CONDITIONS
# ============================================================

def extract_conditions(
    source_text: str,
) -> List[str]:

    text = _clean_text(
        source_text
    )

    if not text:
        return []

    sentences = _split_sentences(
        text
    )

    markers = [
        "provided that",
        "in case of",
        "in case",
        "second or subsequent",
        "first time",
        "first conviction",
        "where",
        "when",
        "upon",
        "subject to",
        "if",
        "unless",
    ]

    output = []

    for sentence in sentences:

        lower = _lower(
            sentence
        )

        if any(
            marker in lower
            for marker in markers
        ):

            output.append(
                sentence
            )

        if len(output) >= 8:
            break

    return _unique_strings(
        output
    )


# ============================================================
# DOCUMENT-SPECIFIC CONDITIONS
# ============================================================

def extract_document_conditions(
    legal_term: str,
    document_text: str,
) -> List[str]:

    text = _lower(
        document_text
    )

    conditions = []

    if legal_term == "theft":

        if "second or subsequent conviction" in text:
            conditions.append(
                "The uploaded document mentions a second or subsequent conviction."
            )

        if (
            "first conviction" in text
            or "first time" in text
        ):
            conditions.append(
                "The uploaded document mentions a first conviction."
            )

        if (
            "five thousand" in text
            or "5000" in text
        ):
            conditions.append(
                "The uploaded document mentions property valued below ₹5,000."
            )

    elif legal_term == "loan_recovery":

        if (
            "15 days" in text
            or "fifteen days" in text
        ):
            conditions.append(
                "The uploaded document states a 15-day payment or response deadline."
            )

        if (
            "12% p.a." in text
            or "12 per cent" in text
            or "12%" in text
        ):
            conditions.append(
                "The uploaded document refers to interest at 12% per annum."
            )

        if (
            "24 monthly instalments" in text
            or "24 monthly installments" in text
        ):
            conditions.append(
                "The uploaded document refers to repayment in 24 monthly instalments."
            )

        if "loan agreement" in text:
            conditions.append(
                "The uploaded document refers to a loan agreement."
            )

    return _unique_strings(
        conditions
    )


# ============================================================
# FRIENDLY SUMMARY
# ============================================================

def build_friendly_summary(
    legal_term: str,
    section: str,
    section_title: str,
) -> str:

    if legal_term == "theft":

        if section == "303":
            return (
                "This is the general provision dealing with theft."
            )

        if section == "305":
            return (
                "This is a special theft provision covering specified places or types of property."
            )

        if section == "306":
            return (
                "This is a special theft provision involving a clerk or servant."
            )

        if section == "307":
            return (
                "This is a special theft provision involving preparation to cause death, hurt or restraint."
            )

    if legal_term == "murder":

        if section == "101":
            return (
                "This provision deals with the legal definition of murder."
            )

        if section == "103":
            return (
                "This provision contains punishment provisions for murder."
            )

    if legal_term == "cheating":

        if section == "318":
            return (
                "This provision deals with cheating."
            )

    if legal_term == "loan_recovery":

        return (
            "This civil-law provision was retrieved as potentially relevant to the loan or money-recovery issue described in the document."
        )

    if legal_term == "contract_breach":

        return (
            "This provision was retrieved in connection with the contractual issue described in the document."
        )

    if legal_term == "property_dispute":

        return (
            "This provision was retrieved in connection with the property issue described in the document."
        )

    if section_title:

        return (
            f"This provision is titled '{section_title}'."
        )

    return (
        "This is a potentially relevant legal provision."
    )


# ============================================================
# RELEVANCE EXPLANATION
# ============================================================

def build_relevance_explanation(
    legal_term: str,
    section: str,
    section_title: str,
    document_text: str,
) -> str:

    text = _lower(
        document_text
    )

    if legal_term == "theft":

        facts = []

        if (
            "dishonestly took" in text
            or "dishonestly taking" in text
        ):
            facts.append(
                "dishonest taking"
            )

        if (
            "without consent" in text
            or "without that person's consent" in text
        ):
            facts.append(
                "lack of consent"
            )

        if (
            "mobile phone" in text
            or "movable property" in text
            or "property" in text
        ):
            facts.append(
                "property being taken"
            )

        if facts:

            return (
                "This provision may be relevant because the uploaded document describes "
                + ", ".join(facts)
                + ". Final applicability depends on the complete facts."
            )

    if legal_term == "murder":

        if (
            "caused the death" in text
            or "death of" in text
            or "murder" in text
        ):
            return (
                "This provision may be relevant because the uploaded document describes a death or an allegation of murder. Final applicability depends on the complete facts."
            )

    if legal_term == "cheating":

        if (
            "deceived" in text
            or "dishonestly induced" in text
            or "false representation" in text
        ):
            return (
                "This provision may be relevant because the uploaded document describes deception or dishonest inducement. Final applicability depends on the complete facts."
            )

    if legal_term == "loan_recovery":

        facts = []

        if "loan" in text:
            facts.append("a loan")

        if "loan agreement" in text:
            facts.append("a loan agreement")

        if (
            "repayment" in text
            or "repay" in text
        ):
            facts.append(
                "repayment obligations"
            )

        if "outstanding" in text:
            facts.append(
                "an outstanding amount"
            )

        if "interest" in text:
            facts.append(
                "interest"
            )

        if facts:

            return (
                "This provision may be relevant because the uploaded document describes "
                + ", ".join(
                    _unique_strings(facts)
                )
                + ". Final applicability should be checked against the complete facts and legal procedure."
            )

    if legal_term == "property_dispute":

        if any(
            term in text
            for term in (
                "property",
                "ownership",
                "possession",
                "title",
            )
        ):
            return (
                "This provision may be relevant because the uploaded document concerns property, ownership or possession. Final applicability depends on the complete facts."
            )

    if legal_term == "contract_breach":

        if any(
            term in text
            for term in (
                "contract",
                "agreement",
                "breach",
                "obligation",
            )
        ):
            return (
                "This provision may be relevant because the uploaded document describes a contract, agreement or contractual obligation."
            )

    return (
        "This provision was retrieved as potentially relevant to the identified legal topic. Final applicability should be checked against the complete facts."
    )


# ============================================================
# PROVISION RELEVANCE
# ============================================================

def calculate_provision_relevance(
    result: Dict[str, Any],
    legal_term: str,
    document_text: str,
) -> float:
    """
    Conservative relevance scoring.

    Loan recovery is intentionally title-first so generic words
    inside unrelated retrieved chunks do not make an unrelated
    section look applicable.
    """

    configuration = TOPIC_CONFIG.get(
        legal_term
    )

    if not configuration:
        return 0.0

    section_title = _lower(
        get_section_title(result)
    )

    source_text = _lower(
        get_source_text(result)
    )

    document_text_lower = _lower(
        document_text
    )

    expected_route = str(
        configuration.get(
            "route",
            "",
        )
    ).lower()

    actual_route = get_route(
        result
    )

    score = 0.0

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    if (
        expected_route
        and actual_route == expected_route
    ):
        score += 4.0

    elif (
        expected_route
        and actual_route
        and actual_route != expected_route
    ):
        score -= 20.0

    # --------------------------------------------------------
    # Amendment / Footnote Fragment Rejection
    # --------------------------------------------------------
    if any(
        frag in section_title or frag in source_text[:200]
        for frag in (
            "subs. by",
            "ins. by",
            "repealed by",
            "a.o. 1950",
            "act 24 of 1917",
            "footnote",
        )
    ):
        return -50.0

    # --------------------------------------------------------
    # Loan recovery: STRICT TITLE & FACT ALIGNMENT
    # --------------------------------------------------------

    if legal_term == "loan_recovery":

        strong_title_terms = [
            "loan",
            "repayment of loan",
            "loan agreement",
            "money recovery",
            "recovery of debt",
            "breach of contract",
            "compensation for breach",
        ]

        direct_title_phrases = [
            "loan agreement",
            "compensation for breach of contract",
            "obligation of parties to contract",
            "summary procedure on negotiable instruments",
            "recovery of money",
        ]

        unrelated_title_terms = [
            "bailor",
            "bailee",
            "bailment",
            "surety",
            "guarantee",
            "contract of guarantee",
            "purchaser",
            "purchase-money",
            "re-sale",
            "immovable property",
            "mortgage",
            "redemption",
            "easement",
            "salary",
            "land-revenue",
            "judgment-debtor",
            "dwelling house",
            "debt to be discharged",
            "application of payment",
        ]

        # Hard penalty for unrelated legal domains
        for term in unrelated_title_terms:
            if term in section_title:
                return -35.0

        # Guarantee/surety (e.g. Sec 126 Contract Act) is relevant ONLY if the user document mentions guarantor
        if any(
            term in section_title
            for term in (
                "guarantee",
                "surety",
                "principal debtor",
            )
        ):
            if not any(
                term in document_text_lower
                for term in (
                    "guarantor",
                    "guarantee",
                    "surety",
                    "co-surety",
                )
            ):
                return -35.0

        # Strong title signals
        for term in strong_title_terms:
            if term in section_title:
                score += 10.0

        for phrase in direct_title_phrases:
            if phrase in section_title:
                score += 12.0

        # General Contract Act provisions that genuinely govern loan agreements (Sec 73, Sec 37, Sec 10)
        sec_num = str(result.get("section", get_section_number(result))).strip()
        if sec_num in ("73", "37", "10") and "contract" in actual_route:
            score += 10.0

        # Factual alignment from document
        document_fact_terms = [
            "loan",
            "loan agreement",
            "repayment",
            "outstanding amount",
            "outstanding dues",
            "borrower",
            "interest",
        ]

        fact_matches = sum(
            1
            for term in document_fact_terms
            if term in document_text_lower
        )

        score += min(
            fact_matches,
            5,
        )

        return float(score)

    # --------------------------------------------------------
    # Criminal topics
    # --------------------------------------------------------

    if legal_term == "theft":

        if "theft" in section_title:
            score += 15.0

        for term in configuration.get(
            "strong",
            [],
        ):

            if term in section_title:
                score += 6.0

        for term in configuration.get(
            "negative",
            [],
        ):

            if term in section_title:
                score -= 10.0

        return float(score)

    if legal_term == "murder":

        if "murder" in section_title:
            score += 15.0

        if "culpable homicide" in section_title:
            score += 8.0

        return float(score)

    if legal_term == "cheating":

        if "cheating" in section_title:
            score += 15.0

        if "deceiving" in section_title:
            score += 8.0

        if "dishonestly induces" in section_title:
            score += 8.0

        return float(score)

    # --------------------------------------------------------
    # Other topics
    # --------------------------------------------------------

    for term in configuration.get(
        "positive",
        [],
    ):

        if term in section_title:
            score += 5.0

    for term in configuration.get(
        "negative",
        [],
    ):

        if term in section_title:
            score -= 8.0

    return float(score)


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    relevance_score: float,
    definition: str,
    punishment: str,
    conditions: List[str],
) -> str:

    score = 0

    if relevance_score >= 18:
        score += 3
    elif relevance_score >= 10:
        score += 2
    elif relevance_score >= 5:
        score += 1

    if definition:
        score += 1

    if punishment:
        score += 2

    if conditions:
        score += 1

    if score >= 6:
        return "high"

    if score >= 3:
        return "medium"

    return "low"


# ============================================================
# EXPLAIN ONE PROVISION
# ============================================================

def explain_provision(
    result: Dict[str, Any],
    legal_term: str,
    document_text: str,
) -> Dict[str, Any]:

    section = get_section_number(
        result
    )

    section_title = get_section_title(
        result
    )

    source_text = get_source_text(
        result
    )

    relevance_score = (
        calculate_provision_relevance(
            result,
            legal_term,
            document_text,
        )
    )

    definition = extract_definition(
        source_text,
        section,
    )

    punishment = extract_punishment(
        source_text
    )

    source_conditions = extract_conditions(
        source_text
    )

    document_conditions = (
        extract_document_conditions(
            legal_term,
            document_text,
        )
    )

    conditions = _unique_strings(
        source_conditions
        + document_conditions
    )[:8]

    summary = build_friendly_summary(
        legal_term,
        section,
        section_title,
    )

    relevance = build_relevance_explanation(
        legal_term,
        section,
        section_title,
        document_text,
    )

    confidence = calculate_confidence(
        relevance_score,
        definition,
        punishment,
        conditions,
    )

    return {
        "section":
            section,

        "section_title":
            section_title,

        "legal_term":
            legal_term,

        "legal_topic":
            legal_topic_label(
                legal_term
            ),

        "summary":
            summary,

        "definition":
            definition,

        "punishment":
            punishment,

        "conditions":
            conditions,

        "why_relevant":
            relevance,

        "source":
            get_source(result),

        "document":
            get_document_title(result),

        "document_type":
            get_document_type(result),

        "authority":
            get_authority(result),

        "route":
            get_route(result),

        "chunk":
            get_chunk(result),

        "retrieval_score":
            round(
                get_score(result),
                4,
            ),

        "relevance_score":
            round(
                relevance_score,
                4,
            ),

        "evidence_confidence":
            confidence,
    }


# ============================================================
# FILTER
# ============================================================

def filter_relevant_provisions(
    provisions: List[
        Dict[str, Any]
    ],
    legal_term: str,
) -> List[
    Dict[str, Any]
]:

    if not provisions:
        return []

    filtered = []

    for provision in provisions:

        title = _lower(
            provision.get(
                "section_title",
                "",
            )
        )

        score = float(
            provision.get(
                "relevance_score",
                0.0,
            )
        )

        # ====================================================
        # LOAN RECOVERY
        # ====================================================

        if legal_term == "loan_recovery":

            title_relevant = any(
                term in title
                for term in (
                    "loan",
                    "debt",
                    "debtor",
                    "creditor",
                    "repayment",
                    "payment",
                    "money",
                    "interest",
                    "default",
                )
            )

            clearly_unrelated = any(
                term in title
                for term in (
                    "bailor",
                    "bailee",
                    "bailment",
                    "surety",
                    "guarantee",
                    "purchaser",
                    "purchase-money",
                    "re-sale",
                    "immovable property",
                    "mortgage",
                    "redemption",
                    "easement",
                    "salary",
                    "land-revenue",
                    "judgment-debtor",
                    "dwelling house",
                )
            )

            if clearly_unrelated:
                continue

            if not title_relevant:
                continue

            if score < MIN_LOAN_RELEVANCE:
                continue

        else:

            if score < MIN_GENERAL_RELEVANCE:
                continue

        filtered.append(
            provision
        )

    filtered.sort(
        key=lambda item: (
            float(
                item.get(
                    "relevance_score",
                    0.0,
                )
            ),
            float(
                item.get(
                    "retrieval_score",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )

    return filtered[
        :MAX_PROVISIONS
    ]


# ============================================================
# DEDUPLICATE
# ============================================================

def deduplicate_provisions(
    provisions: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:

    output = []
    seen = set()

    for item in provisions:

        key = (
            str(
                item.get(
                    "section",
                    "",
                )
            ).strip(),
            _lower(
                item.get(
                    "document",
                    "",
                )
            ),
            _lower(
                item.get(
                    "source",
                    "",
                )
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


# ============================================================
# PRIMARY PROVISION
# ============================================================

def select_primary_provision(
    provisions: List[
        Dict[str, Any]
    ],
    legal_term: str,
) -> Optional[
    Dict[str, Any]
]:

    if not provisions:
        return None

    configuration = TOPIC_CONFIG.get(
        legal_term,
        {},
    )

    preferred_sections = configuration.get(
        "preferred_sections",
        [],
    )

    for section in preferred_sections:

        matches = [
            item
            for item in provisions
            if str(
                item.get(
                    "section",
                    "",
                )
            ).strip() == section
        ]

        if matches:

            matches.sort(
                key=lambda item: float(
                    item.get(
                        "relevance_score",
                        0.0,
                    )
                ),
                reverse=True,
            )

            return matches[0]

    ranked = sorted(
        provisions,
        key=lambda item: (
            float(
                item.get(
                    "relevance_score",
                    0.0,
                )
            ),
            float(
                item.get(
                    "retrieval_score",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )

    return ranked[0]


# ============================================================
# NEXT STEPS
# ============================================================

def build_provision_next_steps(
    legal_term: str,
) -> List[str]:

    if legal_term == "theft":

        return [
            "Keep the original document and supporting evidence safely.",
            "Record important dates, names and facts mentioned in the document.",
            "Consult a qualified lawyer when case-specific legal advice is required.",
        ]

    if legal_term == "murder":

        return [
            "Preserve the original document and supporting evidence.",
            "Keep relevant police, incident and case records together.",
            "Consult a qualified lawyer for case-specific advice.",
        ]

    if legal_term == "cheating":

        return [
            "Preserve agreements, payment records, messages and other supporting evidence.",
            "Keep a clear record of the events and dates described in the document.",
            "Consult a qualified lawyer for case-specific advice.",
        ]

    if legal_term == "loan_recovery":

        return [
            "Review the loan agreement and repayment terms.",
            "Verify payment records and the stated outstanding amount.",
            "Check the payment or response deadline stated in the notice.",
            "Keep the original notice, loan agreement, payment statement and related communications safely.",
            "Consult a qualified lawyer if the matter may lead to proceedings.",
        ]

    if legal_term == "property_dispute":

        return [
            "Preserve title, ownership and possession documents.",
            "Keep relevant dates and communications.",
            "Consult a qualified lawyer for case-specific advice.",
        ]

    if legal_term == "employment_dispute":

        return [
            "Preserve employment records, salary records and communications.",
            "Keep important dates and notices together.",
            "Consult a qualified lawyer for case-specific advice.",
        ]

    return [
        "Preserve the original document and supporting evidence.",
        "Keep important dates and records together.",
        "Consult a qualified lawyer when case-specific legal advice is required.",
    ]


# ============================================================
# MAIN SERVICE
# ============================================================

def analyze_provisions(
    results: List[
        Dict[str, Any]
    ],
    legal_term: str,
    document_text: str,
) -> Dict[str, Any]:

    if not isinstance(
        results,
        list,
    ):
        results = []

    legal_term = str(
        legal_term or "general"
    ).strip()

    document_text = _clean_text(
        document_text
    )

    candidates = []

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        section = get_section_number(
            result
        )

        if not section:
            continue

        candidates.append(
            explain_provision(
                result=result,
                legal_term=legal_term,
                document_text=document_text,
            )
        )

    candidates = deduplicate_provisions(
        candidates
    )

    provisions = filter_relevant_provisions(
        candidates,
        legal_term,
    )

    primary = select_primary_provision(
        provisions,
        legal_term,
    )

    next_steps = build_provision_next_steps(
        legal_term
    )

    if primary is None:

        return {
            "status":
                "no_provision",

            "legal_term":
                legal_term,

            "legal_topic":
                legal_topic_label(
                    legal_term
                ),

            "primary_provision":
                None,

            "provisions":
                [],

            "next_steps":
                next_steps,

            "message":
                (
                    "No sufficiently matched legal provision "
                    "was identified in the current legal "
                    "knowledge base."
                ),
        }

    return {
        "status":
            "success",

        "legal_term":
            legal_term,

        "legal_topic":
            legal_topic_label(
                legal_term
            ),

        "primary_provision":
            primary,

        "provisions":
            provisions,

        "next_steps":
            next_steps,

        "message":
            (
                "The provision information is based on "
                "accepted retrieved legal sources."
            ),
    }


# ============================================================
# CONVENIENCE
# ============================================================

def explain_first_provision(
    results: List[
        Dict[str, Any]
    ],
    legal_term: str,
    document_text: str,
) -> Dict[str, Any]:

    return analyze_provisions(
        results=results,
        legal_term=legal_term,
        document_text=document_text,
    )


# ============================================================
# BUILT-IN TEST
# ============================================================

TEST_THEFT_RESULT = {
    "document": """
    Theft.—(1) Whoever, intending to take dishonestly
    any movable property out of the possession of any
    person without that person's consent, moves that
    property in order to such taking, is said to commit theft.

    Whoever commits theft shall be punished with imprisonment
    of either description for a term which may extend to
    three years, or with fine, or with both.

    In case of second or subsequent conviction of any person
    under this section, he shall be punished with rigorous
    imprisonment for a term which shall not be less than one
    year but which may extend to five years and with fine.

    In cases of theft where the value of the stolen property
    is less than five thousand rupees, and a person is convicted
    for the first time, upon return of the value of property
    or restoration of the stolen property, the person shall be
    punished with community service.
    """,

    "metadata": {
        "section_number":
            "303",

        "section_title":
            "Theft.",

        "source":
            "Bharatiya_Nyaya_Sanhita_2023.pdf",

        "title":
            "Bharatiya Nyaya Sanhita, 2023",

        "document_type":
            "Act",

        "authority":
            "Government of India",

        "route":
            "criminal",

        "chunk_index":
            826,
    },

    "hybrid_score":
        17.35,
}


def main() -> None:

    import json

    print()
    print(
        "JanNyaya AI - Legal Provision Explainer Test"
    )

    result = analyze_provisions(
        results=[
            TEST_THEFT_RESULT,
        ],
        legal_term="theft",
        document_text=(
            "The accused dishonestly took a mobile phone "
            "from the complainant's possession without consent."
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()