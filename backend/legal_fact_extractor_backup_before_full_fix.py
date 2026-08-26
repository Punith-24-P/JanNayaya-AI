"""
JanNyaya AI - Structured Legal Fact Extractor

Purpose
-------
Convert retrieved legal chunks into compact, structured legal facts
before sending them to the LLM.

Supported languages
-------------------
English
Hindi
Kannada

Main responsibilities
----------------------
1. Detect question intent.
2. Detect canonical legal offence.
3. Group retrieved chunks by legal section.
4. Prefer the direct/general offence provision.
5. Extract definition, punishment and condition facts.
6. Build compact grounded context for the LLM.
7. Prevent specialized provisions from replacing the main offence
   provision for generic questions.

Example
-------
"What is the punishment for theft?"
    -> Section 303

"What is theft?"
    -> Section 303

"theft in a dwelling house"
    -> Section 305

"theft by clerk or servant"
    -> Section 306

"theft after preparation..."
    -> Section 307
"""

from __future__ import annotations

import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FACT_GROUPS = 5

MAX_FACT_LINES_PER_TYPE = 8

MAX_CONTEXT_CHARS = 12000

DIRECT_SECTION_BONUS = 8.0

SECTION_303_DIRECT_BONUS = 2.0

SPECIALIZED_SECTION_PENALTY = 1.5


# ============================================================
# MULTILINGUAL PUNISHMENT WORDS
# ============================================================

ENGLISH_PUNISHMENT_WORDS = {
    "punishment",
    "punished",
    "punish",
    "penalty",
    "penalties",
    "sentence",
    "sentencing",
    "imprisonment",
    "prison",
    "jail",
    "fine",
    "fines",
    "punishable",
    "community",
}


HINDI_PUNISHMENT_WORDS = (
    "सजा",
    "सज़ा",
    "दंड",
    "दण्ड",
    "जुर्माना",
    "कारावास",
    "कैद",
    "जेल",
    "सजाएं",
    "सज़ाएं",
)


KANNADA_PUNISHMENT_WORDS = (
    "ಶಿಕ್ಷೆ",
    "ದಂಡ",
    "ದಂಡನೆ",
    "ಕಾರಾಗೃಹ",
    "ಜೈಲು",
    "ಜೈಲುವಾಸ",
    "ಸಜೆ",
)


# ============================================================
# MULTILINGUAL DEFINITION WORDS
# ============================================================

ENGLISH_DEFINITION_PHRASES = (
    "what is",
    "what are",
    "what does",
    "meaning of",
    "define",
    "definition",
    "explain",
)


HINDI_DEFINITION_PHRASES = (
    "क्या है",
    "क्या हैं",
    "क्या होता है",
    "परिभाषा",
    "अर्थ",
    "मतलब",
)


KANNADA_DEFINITION_PHRASES = (
    "ಎಂದರೇನು",
    "ಏನು",
    "ಏನಿದು",
    "ಅರ್ಥ ಏನು",
    "ಅರ್ಥ",
    "ಪರಿಭಾಷೆ",
)


# ============================================================
# LEGAL TERMS
# ============================================================

MULTILINGUAL_LEGAL_TERMS = {

    # English
    "criminal breach of trust": "criminal breach of trust",
    "criminal intimidation": "criminal intimidation",
    "cheating by personation": "cheating by personation",
    "petty organised crime": "petty organised crime",
    "petty organized crime": "petty organised crime",
    "organised crime": "organised crime",
    "organized crime": "organised crime",
    "house trespass": "house trespass",
    "house breaking": "house breaking",
    "house-breaking": "house breaking",
    "snatching": "snatching",
    "defamation": "defamation",
    "kidnapping": "kidnapping",
    "abduction": "abduction",
    "robbery": "robbery",
    "extortion": "extortion",
    "assault": "assault",
    "murder": "murder",
    "rape": "rape",
    "cheating": "cheating",
    "theft": "theft",

    # Hindi
    "चोरी": "theft",
    "चोरी का": "theft",
    "चोरी की": "theft",
    "चोरी के": "theft",
    "चोरी में": "theft",
    "चोरी करना": "theft",

    "हत्या": "murder",
    "हत्या की": "murder",
    "हत्या का": "murder",

    "बलात्कार": "rape",
    "बलात्कार की": "rape",

    "धोखाधड़ी": "cheating",
    "धोखा": "cheating",

    "लूट": "robbery",
    "डकैती": "robbery",

    "अपहरण": "kidnapping",

    "जबरन वसूली": "extortion",

    "मानहानि": "defamation",

    "धमकी": "criminal intimidation",

    # Kannada
    "ಕಳ್ಳತನ": "theft",
    "ಕಳ್ಳತನಕ್ಕೆ": "theft",
    "ಕಳ್ಳತನದ": "theft",
    "ಕಳ್ಳತನವನ್ನು": "theft",
    "ಕಳ್ಳತನದಲ್ಲಿ": "theft",

    "ಹತ್ಯೆ": "murder",
    "ಹತ್ಯೆಯ": "murder",
    "ಹತ್ಯೆಗೆ": "murder",

    "ಅತ್ಯಾಚಾರ": "rape",
    "ಅತ್ಯಾಚಾರದ": "rape",

    "ಮೋಸ": "cheating",
    "ಮೋಸದಿಂದ": "cheating",

    "ದರೋಡೆ": "robbery",

    "ಸುಲಿಗೆ": "extortion",

    "ಅಪಹರಣ": "kidnapping",

    "ಮಾನನಷ್ಟ": "defamation",

    "ಬೆದರಿಕೆ": "criminal intimidation",

    "ದೌರ್ಜನ್ಯ": "assault",
}


# ============================================================
# CANONICAL TERM LIST
# ============================================================

LEGAL_TERMS = [
    "criminal breach of trust",
    "criminal intimidation",
    "cheating by personation",
    "petty organised crime",
    "organised crime",
    "house trespass",
    "house breaking",
    "snatching",
    "defamation",
    "kidnapping",
    "abduction",
    "robbery",
    "extortion",
    "assault",
    "murder",
    "rape",
    "cheating",
    "theft",
]


# ============================================================
# SECTION KNOWLEDGE
# ============================================================
#
# These are only used to rank retrieved evidence.
# They do NOT generate legal answers by themselves.
#
# The primary goal is to distinguish the general offence
# provision from specialized provisions.
# ============================================================

GENERAL_OFFENCE_SECTIONS = {
    "theft": {
        "303",
    },
    "murder": {
        "101",
    },
    "cheating": {
        "318",
    },
    "rape": {
        "64",
    },
    "robbery": {
        "309",
    },
    "extortion": {
        "308",
    },
    "kidnapping": {
        "137",
    },
    "abduction": {
        "138",
    },
    "snatching": {
        "304",
    },
    "defamation": {
        "356",
    },
}


SPECIALIZED_SECTION_TERMS = (
    "after preparation",
    "dwelling house",
    "means of transportation",
    "place of worship",
    "clerk or servant",
    "life-convict",
    "house-breaking",
    "house breaking",
    "organized crime",
    "organised crime",
    "petty organized crime",
    "petty organised crime",
    "government or local authority",
)


# ============================================================
# PUNISHMENT PATTERNS
# ============================================================

PUNISHMENT_PATTERNS = (
    "shall be punished",
    "shall be punishable",
    "punished with",
    "liable to fine",
    "liable to punishment",
    "imprisonment",
    "rigorous imprisonment",
    "community service",
    "fine",
    "death",
    "imprisonment for life",
)


# ============================================================
# DEFINITION PATTERNS
# ============================================================

DEFINITION_PATTERNS = (
    "is said to",
    "means",
    "definition",
    "whoever",
    "intending to",
    "is defined",
)


# ============================================================
# CONDITION PATTERNS
# ============================================================

CONDITION_PATTERNS = (
    "provided that",
    "provided",
    "if ",
    "unless",
    "in case",
    "where",
    "second or subsequent",
    "first time",
    "first conviction",
    "subsequent conviction",
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(
    text: str,
) -> str:
    """
    Normalize whitespace while preserving Unicode.
    """

    if not text:
        return ""

    text = str(text)

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# QUERY INTENT
# ============================================================

def detect_fact_intent(
    query: str,
) -> str:
    """
    Detect:

        punishment
        definition
        general

    Supports English, Hindi and Kannada.

    IMPORTANT:
    Punishment is checked before definition so that:

        चोरी की सजा क्या है?
        ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ ಏನು?

    do not become definition questions merely because they
    contain "क्या है" / "ಏನು".
    """

    if not query:
        return "general"

    query = str(
        query
    ).strip()

    if not query:
        return "general"

    normalized = query.lower()

    # --------------------------------------------------------
    # English punishment
    # --------------------------------------------------------

    for word in ENGLISH_PUNISHMENT_WORDS:

        if word in normalized:

            return "punishment"

    # --------------------------------------------------------
    # Hindi punishment
    # --------------------------------------------------------

    for word in HINDI_PUNISHMENT_WORDS:

        if word in query:

            return "punishment"

    # --------------------------------------------------------
    # Kannada punishment
    # --------------------------------------------------------

    for word in KANNADA_PUNISHMENT_WORDS:

        if word in query:

            return "punishment"

    # --------------------------------------------------------
    # English definition
    # --------------------------------------------------------

    for phrase in ENGLISH_DEFINITION_PHRASES:

        if phrase in normalized:

            return "definition"

    # --------------------------------------------------------
    # Hindi definition
    # --------------------------------------------------------

    for phrase in HINDI_DEFINITION_PHRASES:

        if phrase in query:

            return "definition"

    # --------------------------------------------------------
    # Kannada definition
    # --------------------------------------------------------

    for phrase in KANNADA_DEFINITION_PHRASES:

        if phrase in query:

            return "definition"

    return "general"


# ============================================================
# LEGAL TERM EXTRACTION
# ============================================================

def extract_fact_legal_term(
    query: str,
) -> str:
    """
    Convert a multilingual query into a canonical English
    legal term.
    """

    if not query:
        return ""

    query = str(
        query
    ).strip()

    if not query:
        return ""

    # Longest phrase first.
    for phrase in sorted(
        MULTILINGUAL_LEGAL_TERMS.keys(),
        key=len,
        reverse=True,
    ):

        if phrase in query:

            return MULTILINGUAL_LEGAL_TERMS[
                phrase
            ]

    normalized = query.lower()

    for term in sorted(
        LEGAL_TERMS,
        key=len,
        reverse=True,
    ):

        if term in normalized:

            return term

    return ""


# ============================================================
# BACKWARD-COMPATIBILITY ALIASES
# ============================================================

def detect_query_intent(
    query: str,
) -> str:
    return detect_fact_intent(
        query
    )


def extract_query_legal_term(
    query: str,
) -> str:
    return extract_fact_legal_term(
        query
    )


# ============================================================
# METADATA HELPERS
# ============================================================

def _get_metadata(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    metadata = result.get(
        "metadata",
        {},
    )

    if isinstance(
        metadata,
        dict,
    ):
        return metadata

    return {}


def _get_section(
    result: Dict[str, Any],
) -> str:

    metadata = _get_metadata(
        result
    )

    value = metadata.get(
        "section_number",
        metadata.get(
            "section",
            "",
        ),
    )

    if value is None:
        return ""

    return str(
        value
    ).strip()


def _get_section_title(
    result: Dict[str, Any],
) -> str:

    metadata = _get_metadata(
        result
    )

    value = metadata.get(
        "section_title",
        "",
    )

    if value is None:
        return ""

    return str(
        value
    ).strip()


def _get_source(
    result: Dict[str, Any],
) -> str:

    metadata = _get_metadata(
        result
    )

    value = metadata.get(
        "source",
        "Unknown",
    )

    if value is None:
        return "Unknown"

    return str(
        value
    ).strip()


# ============================================================
# RESULT SCORE
# ============================================================

def _get_result_score(
    result: Dict[str, Any],
) -> float:

    for key in (
        "legal_rerank_score",
        "hybrid_score",
        "rrf_score",
        "score",
    ):

        value = result.get(
            key,
            None,
        )

        if value is not None:

            try:

                return float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

    return 0.0


# ============================================================
# SPECIALIZED PROVISION DETECTION
# ============================================================

def _is_specialized_provision(
    title: str,
    query_term: str,
) -> bool:

    if not title:
        return False

    normalized_title = title.lower()

    # Exact general offence title is never specialized.
    if normalized_title.strip(
        " .—-:"
    ) == query_term:

        return False

    for phrase in SPECIALIZED_SECTION_TERMS:

        if phrase in normalized_title:

            return True

    return False


# ============================================================
# DIRECT OFFENCE DETECTION
# ============================================================

def _is_direct_offence(
    query_term: str,
    section: str,
    title: str,
) -> bool:
    """
    Determine whether a section is the general offence provision.

    This is the critical part that prevents:

        theft -> 305/306/307

    from replacing:

        theft -> 303
    """

    if not query_term:
        return False

    normalized_title = (
        title
        .lower()
        .strip()
    )

    # --------------------------------------------------------
    # Known general section
    # --------------------------------------------------------

    general_sections = GENERAL_OFFENCE_SECTIONS.get(
        query_term,
        set(),
    )

    if section in general_sections:

        return True

    # --------------------------------------------------------
    # Exact title match
    # --------------------------------------------------------

    simplified_title = normalized_title

    # Remove section-specific punctuation.
    simplified_title = simplified_title.rstrip(
        " .—-:"
    )

    if simplified_title == query_term:

        return True

    # --------------------------------------------------------
    # Title beginning with the offence itself.
    #
    # This is allowed only if the title is not clearly a
    # specialized provision.
    # --------------------------------------------------------

    starts_with_term = (
        simplified_title.startswith(
            query_term + " "
        )
        or
        simplified_title.startswith(
            query_term + "."
        )
        or
        simplified_title.startswith(
            query_term + "—"
        )
        or
        simplified_title.startswith(
            query_term + ":"
        )
    )

    if (
        starts_with_term
        and not _is_specialized_provision(
            title,
            query_term,
        )
    ):

        return True

    return False


# ============================================================
# FACT LINE EXTRACTION
# ============================================================

def _split_into_sentences(
    text: str,
) -> List[str]:
    """
    Lightweight sentence splitter suitable for legal text.

    We intentionally preserve the original wording.
    """

    if not text:
        return []

    text = _normalize_text(
        text
    )

    if not text:
        return []

    # Split around sentence-ending punctuation.
    pieces = re.split(
        r"(?<=[.!?।])\s+",
        text,
    )

    output = []

    for piece in pieces:

        piece = piece.strip()

        if piece:

            output.append(
                piece
            )

    return output


def _extract_matching_lines(
    document: str,
    patterns: tuple[str, ...],
    max_lines: int = MAX_FACT_LINES_PER_TYPE,
) -> List[str]:
    """
    Extract the smallest useful evidence lines containing one
    of the specified legal patterns.

    No paraphrasing is performed here.
    """

    if not document:
        return []

    document = _normalize_text(
        document
    )

    if not document:
        return []

    sentences = _split_into_sentences(
        document
    )

    matches = []

    for sentence in sentences:

        normalized_sentence = sentence.lower()

        matched = False

        for pattern in patterns:

            if pattern in normalized_sentence:

                matched = True
                break

        if not matched:
            continue

        # Avoid extremely tiny fragments.
        if len(sentence) < 25:
            continue

        matches.append(
            sentence
        )

        if len(matches) >= max_lines:
            break

    return matches


# ============================================================
# PUNISHMENT FACTS
# ============================================================

def extract_punishment_facts(
    document: str,
) -> List[str]:
    """
    Extract punishment-related evidence.
    """

    return _extract_matching_lines(
        document,
        PUNISHMENT_PATTERNS,
        MAX_FACT_LINES_PER_TYPE,
    )


# ============================================================
# DEFINITION FACTS
# ============================================================

def extract_definition_facts(
    document: str,
) -> List[str]:
    """
    Extract definition-related evidence.
    """

    return _extract_matching_lines(
        document,
        DEFINITION_PATTERNS,
        MAX_FACT_LINES_PER_TYPE,
    )


# ============================================================
# CONDITION FACTS
# ============================================================

def extract_condition_facts(
    document: str,
) -> List[str]:
    """
    Extract conditions / exceptions / special cases.
    """

    return _extract_matching_lines(
        document,
        CONDITION_PATTERNS,
        MAX_FACT_LINES_PER_TYPE,
    )


# ============================================================
# GROUP RESULTS BY SECTION
# ============================================================

def group_results_by_section(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group retrieval results by document + section.

    Important:
    Results are expected to be dictionaries.
    Invalid items are ignored safely.
    """

    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    if not results:
        return grouped

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        section = _get_section(
            result
        )

        source = _get_source(
            result
        )

        key = (
            f"{source}:"
            f"{section}"
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            result
        )

    return grouped


# ============================================================
# MERGE SECTION DOCUMENTS
# ============================================================

def _merge_group_documents(
    results: List[Dict[str, Any]],
) -> str:
    """
    Merge all chunks from one section.

    Chunks are ordered by chunk index where available.
    Duplicate exact text is removed.
    """

    ordered = list(
        results
    )

    def chunk_sort_key(
        result: Dict[str, Any],
    ) -> int:

        metadata = _get_metadata(
            result
        )

        value = metadata.get(
            "chunk_index",
            0,
        )

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0

    ordered.sort(
        key=chunk_sort_key
    )

    pieces = []

    seen = set()

    for result in ordered:

        document = result.get(
            "document",
            "",
        )

        if not document:
            continue

        document = _normalize_text(
            document
        )

        if not document:
            continue

        # Exact duplicate prevention.
        if document in seen:
            continue

        seen.add(
            document
        )

        pieces.append(
            document
        )

    return "\n\n".join(
        pieces
    )


# ============================================================
# SCORE FACT GROUP
# ============================================================

def _score_fact_group(
    query: str,
    query_term: str,
    query_intent: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if not results:
        return {}

    first = results[0]

    section = _get_section(
        first
    )

    title = _get_section_title(
        first
    )

    source = _get_source(
        first
    )

    document = _merge_group_documents(
        results
    )

    direct_offence = _is_direct_offence(
        query_term,
        section,
        title,
    )

    specialized = _is_specialized_provision(
        title,
        query_term,
    )

    base_score = max(
        _get_result_score(
            result
        )
        for result in results
    )

    score = base_score

    # --------------------------------------------------------
    # Direct general provision
    # --------------------------------------------------------

    if direct_offence:

        score += DIRECT_SECTION_BONUS

    # --------------------------------------------------------
    # Extra protection for known general offence section
    # --------------------------------------------------------

    known_general_sections = (
        GENERAL_OFFENCE_SECTIONS.get(
            query_term,
            set(),
        )
    )

    if section in known_general_sections:

        score += SECTION_303_DIRECT_BONUS

    # --------------------------------------------------------
    # Specialized provision penalty
    # --------------------------------------------------------

    if (
        specialized
        and not direct_offence
    ):

        score -= SPECIALIZED_SECTION_PENALTY

    # --------------------------------------------------------
    # Intent-specific evidence
    # --------------------------------------------------------

    punishment_facts = extract_punishment_facts(
        document
    )

    definition_facts = extract_definition_facts(
        document
    )

    condition_facts = extract_condition_facts(
        document
    )

    if (
        query_intent == "punishment"
        and punishment_facts
    ):

        score += 1.0

    if (
        query_intent == "definition"
        and definition_facts
    ):

        score += 1.0

    # --------------------------------------------------------
    # For generic offence questions, direct general section
    # is much more important than specialized provisions.
    # --------------------------------------------------------

    if (
        query_term
        and direct_offence
        and section in known_general_sections
    ):

        score += 2.0

    return {
        "section": section,
        "section_title": title,
        "source": source,
        "document_type": _get_metadata(first).get(
            "document_type",
            "",
        ),
        "title": _get_metadata(first).get(
            "title",
            "",
        ),
        "year": _get_metadata(first).get(
            "year",
            None,
        ),
        "act_name": _get_metadata(first).get(
            "act_name",
            "",
        ),
        "direct_offence": direct_offence,
        "specialized_provision": specialized,
        "query_intent": query_intent,
        "legal_term": query_term,
        "score": float(score),
        "punishment_facts": punishment_facts,
        "definition_facts": definition_facts,
        "condition_facts": condition_facts,
        "document": document,
    }


# ============================================================
# SELECT BEST FACT GROUPS
# ============================================================

def select_best_fact_groups(
    query: str,
    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ],
    query_intent: Optional[str] = None,
    query_term: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Select the best legal section groups.

    Critical rule:
    For a generic offence question, the direct/general
    offence section wins over specialized provisions.

    Example:

        theft
            303 -> preferred

        theft in dwelling house
            305 -> preferred only when retrieval/query
                   actually points toward that specialized context
    """

    if not grouped:
        return []

    if query_intent is None:
        query_intent = detect_fact_intent(
            query
        )

    if query_term is None:
        query_term = extract_fact_legal_term(
            query
        )

    fact_groups = []

    for results in grouped.values():

        fact = _score_fact_group(
            query=query,
            query_term=query_term,
            query_intent=query_intent,
            results=results,
        )

        if fact:
            fact_groups.append(
                fact
            )

    # --------------------------------------------------------
    # Sort by score.
    # --------------------------------------------------------

    fact_groups.sort(
        key=lambda fact: (
            bool(
                fact.get(
                    "direct_offence",
                    False,
                )
            ),
            float(
                fact.get(
                    "score",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Strongest direct section.
    # --------------------------------------------------------

    direct_groups = [
        fact
        for fact in fact_groups
        if fact.get(
            "direct_offence",
            False,
        )
    ]

    # --------------------------------------------------------
    # For generic offence questions, prefer only the strongest
    # direct section.
    #
    # This prevents:
    #
    # theft -> 305 / 306 / 307
    #
    # when the user simply asks:
    #
    # "What is theft?"
    # "What is the punishment for theft?"
    # --------------------------------------------------------

    if direct_groups:

        if (
            query_term
            and query_term in GENERAL_OFFENCE_SECTIONS
        ):

            known_sections = (
                GENERAL_OFFENCE_SECTIONS[
                    query_term
                ]
            )

            known_direct = [
                fact
                for fact in direct_groups
                if fact.get(
                    "section",
                    "",
                )
                in known_sections
            ]

            if known_direct:

                known_direct.sort(
                    key=lambda fact: float(
                        fact.get(
                            "score",
                            0.0,
                        )
                    ),
                    reverse=True,
                )

                primary = known_direct[0]

                return [
                    primary
                ]

        # Otherwise return strongest direct provision.
        return direct_groups[
            :1
        ]

    # --------------------------------------------------------
    # If no direct group exists, return strongest candidates.
    # --------------------------------------------------------

    return fact_groups[
        :MAX_FACT_GROUPS
    ]


# ============================================================
# EXTRACT LEGAL FACTS
# ============================================================

def extract_legal_facts(
    query: str,
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Main structured fact extraction pipeline.

    Input:
        user query
        retrieval result dictionaries

    Output:
        compact structured legal fact groups
    """

    if not query:
        return []

    if not results:
        return []

    valid_results = []

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        if not result.get(
            "document",
            "",
        ):

            continue

        valid_results.append(
            result
        )

    if not valid_results:
        return []

    query_intent = detect_fact_intent(
        query
    )

    query_term = extract_fact_legal_term(
        query
    )

    grouped = group_results_by_section(
        valid_results
    )

    selected = select_best_fact_groups(
        query=query,
        grouped=grouped,
        query_intent=query_intent,
        query_term=query_term,
    )

    return selected


# ============================================================
# BUILD FACT CONTEXT
# ============================================================

def build_fact_context(
    fact_groups: List[Dict[str, Any]],
    max_characters: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Build compact context for the LLM.

    The LLM sees structured evidence rather than the complete
    retrieval dump.
    """

    if not fact_groups:
        return ""

    parts = []

    current_length = 0

    for index, fact in enumerate(
        fact_groups,
        start=1,
    ):

        section = fact.get(
            "section",
            "",
        )

        section_title = fact.get(
            "section_title",
            "",
        )

        source = fact.get(
            "source",
            "Unknown",
        )

        direct = fact.get(
            "direct_offence",
            False,
        )

        query_intent = fact.get(
            "query_intent",
            "general",
        )

        header = (
            f"[Legal Fact {index}]\n"
            f"Section: {section}\n"
            f"Section title: {section_title}\n"
            f"Source: {source}\n"
            f"Direct offence provision: "
            f"{'yes' if direct else 'no'}\n"
        )

        # ----------------------------------------------------
        # Definition facts
        # ----------------------------------------------------

        definition_facts = fact.get(
            "definition_facts",
            [],
        )

        if (
            definition_facts
            and query_intent
            in {
                "definition",
                "general",
            }
        ):

            header += (
                "\nDefinition provisions:\n"
            )

            for line in definition_facts:

                header += (
                    f"- {line}\n"
                )

        # ----------------------------------------------------
        # Punishment facts
        # ----------------------------------------------------

        punishment_facts = fact.get(
            "punishment_facts",
            [],
        )

        if (
            punishment_facts
            and query_intent
            == "punishment"
        ):

            header += (
                "\nPunishment provisions:\n"
            )

            for line in punishment_facts:

                header += (
                    f"- {line}\n"
                )

        # ----------------------------------------------------
        # Conditions
        # ----------------------------------------------------

        condition_facts = fact.get(
            "condition_facts",
            [],
        )

        if condition_facts:

            header += (
                "\nConditions / exceptions:\n"
            )

            for line in condition_facts:

                header += (
                    f"- {line}\n"
                )

        # ----------------------------------------------------
        # Respect total context size
        # ----------------------------------------------------

        if (
            current_length
            + len(header)
            + 2
            > max_characters
        ):

            break

        parts.append(
            header.strip()
        )

        current_length += (
            len(header)
            + 2
        )

    return "\n\n".join(
        parts
    )


# ============================================================
# DEBUG PRINTING
# ============================================================

def print_legal_facts(
    fact_groups: List[Dict[str, Any]],
) -> None:
    """
    Print structured facts for debugging.
    """

    print()
    print(
        "=" * 70
    )
    print(
        "STRUCTURED LEGAL FACTS"
    )
    print(
        "=" * 70
    )

    if not fact_groups:

        print(
            "No legal facts found."
        )

        return

    for index, fact in enumerate(
        fact_groups,
        start=1,
    ):

        print()
        print(
            f"[{index}]"
        )

        print(
            "Section:",
            fact.get(
                "section",
                "",
            ),
        )

        print(
            "Section title:",
            fact.get(
                "section_title",
                "",
            ),
        )

        print(
            "Source:",
            fact.get(
                "source",
                "",
            ),
        )

        print(
            "Direct offence:",
            fact.get(
                "direct_offence",
                False,
            ),
        )

        print(
            "Query intent:",
            fact.get(
                "query_intent",
                "",
            ),
        )

        print(
            "Legal term:",
            fact.get(
                "legal_term",
                "",
            ),
        )

        print(
            "Score:",
            round(
                float(
                    fact.get(
                        "score",
                        0.0,
                    )
                ),
                4,
            ),
        )

        punishment = fact.get(
            "punishment_facts",
            [],
        )

        if punishment:

            print()
            print(
                "Punishment facts:"
            )

            for line in punishment:

                print(
                    "-",
                    line,
                )

        definition = fact.get(
            "definition_facts",
            [],
        )

        if definition:

            print()
            print(
                "Definition facts:"
            )

            for line in definition:

                print(
                    "-",
                    line,
                )

        conditions = fact.get(
            "condition_facts",
            [],
        )

        if conditions:

            print()
            print(
                "Condition facts:"
            )

            for line in conditions:

                print(
                    "-",
                    line,
                )

        print(
            "-" * 60
        )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "JanNyaya AI - Legal Fact Extractor Test"
    )

    tests = [
        "What is the punishment for theft?",
        "What is theft?",
        "चोरी की सजा क्या है?",
        "चोरी क्या है?",
        "ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ ಏನು?",
        "ಕಳ್ಳತನ ಎಂದರೇನು?",
    ]

    try:

        from backend.retriever import (
            hybrid_search,
        )

    except Exception as error:

        print()
        print(
            "Could not import retriever:"
        )
        print(
            error
        )

        raise SystemExit(1)

    for question in tests:

        print()
        print(
            "=" * 70
        )

        print(
            "QUESTION:",
            question,
        )

        print(
            "=" * 70
        )

        results = hybrid_search(
            question,
            semantic_k=30,
            bm25_k=30,
            final_k=10,
        )

        facts = extract_legal_facts(
            question,
            results,
        )

        print_legal_facts(
            facts
        )

        context = build_fact_context(
            facts
        )

        print()
        print(
            "=" * 70
        )

        print(
            "COMPACT LLM CONTEXT"
        )

        print(
            "=" * 70
        )

        print(
            context
        )