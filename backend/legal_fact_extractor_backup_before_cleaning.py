"""
JanNyaya AI - Structured Legal Fact Extractor

Converts retrieved legal chunks into structured legal facts.

Supported:
- English
- Hindi
- Kannada
"""

import re
from typing import List, Dict, Any, Optional


# ============================================================
# QUERY INTENT
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
    "community service",
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

ENGLISH_DEFINITION_PHRASES = (
    "what is",
    "what are",
    "what does",
    "meaning of",
    "define",
    "definition of",
    "explain",
)

HINDI_DEFINITION_PHRASES = (
    "क्या है",
    "क्या होता है",
    "क्या हैं",
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
# LEGAL TERM MAP
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
    "बलात्कार": "rape",
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
    "ಅತ್ಯಾಚಾರ": "rape",
    "ಮೋಸ": "cheating",
    "ದರೋಡೆ": "robbery",
    "ಸುಲಿಗೆ": "extortion",
    "ಅಪಹರಣ": "kidnapping",
    "ಮಾನನಷ್ಟ": "defamation",
    "ಬೆದರಿಕೆ": "criminal intimidation",
    "ದೌರ್ಜನ್ಯ": "assault",
}

LEGAL_TERMS = sorted(
    set(MULTILINGUAL_LEGAL_TERMS.values()),
    key=len,
    reverse=True,
)


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _unique(values: List[str]) -> List[str]:
    result = []
    seen = set()

    for value in values:
        value = _normalize_text(value)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


# ============================================================
# INTENT
# ============================================================

def detect_fact_intent(query: str) -> str:
    if not query:
        return "general"

    query = str(query).strip()
    lower = query.lower()

    # Punishment first.
    for word in HINDI_PUNISHMENT_WORDS:
        if word in query:
            return "punishment"

    for word in KANNADA_PUNISHMENT_WORDS:
        if word in query:
            return "punishment"

    if set(lower.split()).intersection(
        ENGLISH_PUNISHMENT_WORDS
    ):
        return "punishment"

    for phrase in HINDI_DEFINITION_PHRASES:
        if phrase in query:
            return "definition"

    for phrase in KANNADA_DEFINITION_PHRASES:
        if phrase in query:
            return "definition"

    for phrase in ENGLISH_DEFINITION_PHRASES:
        if phrase in lower:
            return "definition"

    return "general"


def detect_query_intent(query: str) -> str:
    return detect_fact_intent(query)


# ============================================================
# LEGAL TERM
# ============================================================

def detect_legal_term(query: str) -> str:
    if not query:
        return ""

    query = str(query).strip()
    lower = query.lower()

    for phrase in sorted(
        MULTILINGUAL_LEGAL_TERMS.keys(),
        key=len,
        reverse=True,
    ):
        if phrase in query:
            return MULTILINGUAL_LEGAL_TERMS[phrase]

    for term in LEGAL_TERMS:
        if term in lower:
            return term

    return ""


def extract_query_legal_term(query: str) -> str:
    return detect_legal_term(query)


# ============================================================
# METADATA HELPERS
# ============================================================

def _metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    value = result.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _section(result: Dict[str, Any]) -> str:
    meta = _metadata(result)

    value = meta.get(
        "section_number",
        meta.get("section", ""),
    )

    return str(value).strip() if value is not None else ""


def _section_title(result: Dict[str, Any]) -> str:
    meta = _metadata(result)
    value = meta.get("section_title", "")
    return str(value).strip() if value is not None else ""


def _source(result: Dict[str, Any]) -> str:
    meta = _metadata(result)
    value = meta.get("source", "Unknown")
    return str(value).strip()


def _chunk_index(result: Dict[str, Any]) -> Optional[int]:
    meta = _metadata(result)

    try:
        return int(meta.get("chunk_index"))
    except (TypeError, ValueError):
        return None


# ============================================================
# DIRECT OFFENCE DETECTION
# ============================================================

def _is_direct_offence_section(
    section_title: str,
    legal_term: str,
) -> bool:
    """
    Section 303 title is:

    Theft.—(1) Whoever...

    This MUST return True.

    Specialized sections such as:

    Theft in a dwelling house...
    Theft by clerk or servant...
    Theft after preparation...

    must return False.
    """

    if not section_title or not legal_term:
        return False

    title = _normalize_text(
        section_title
    ).lower()

    term = legal_term.strip().lower()

    # Remove section title punctuation while preserving words.
    normalized = re.sub(
        r"[—–\-:;,.]+",
        " ",
        title,
    )

    normalized = _normalize_text(
        normalized
    )

    # Exact title.
    if normalized == term:
        return True

    # IMPORTANT:
    # "theft ... whoever" is still the direct section,
    # while "theft in a dwelling..." is specialized.
    if normalized.startswith(term):

        remainder = normalized[len(term):].strip()

        if not remainder:
            return True

        specialized_prefixes = (
            "in a dwelling",
            "in any building",
            "by clerk",
            "by servant",
            "after preparation",
            "in a place",
            "of any",
            "by person",
            "in any",
            "is snatching",
        )

        for prefix in specialized_prefixes:
            if remainder.startswith(prefix):
                return False

        # "theft whoever..."
        # "theft (1) whoever..."
        # "theft whoever commits..."
        if (
            remainder.startswith("whoever")
            or remainder.startswith("(1)")
            or remainder.startswith("1")
        ):
            return True

        # PDF title may contain "(1) Whoever..."
        if "whoever" in remainder[:100]:
            return True

    return False


# ============================================================
# TEXT SPLITTING
# ============================================================

def _split_units(text: str) -> List[str]:
    if not text:
        return []

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Keep line boundaries because legal PDF extraction
    # frequently separates subsection text by lines.
    lines = [
        _normalize_text(line)
        for line in text.split("\n")
    ]

    lines = [
        line
        for line in lines
        if line
    ]

    return lines


# ============================================================
# DEFINITION EXTRACTION
# ============================================================

def _extract_definition_facts(
    document: str,
    legal_term: str,
) -> List[str]:

    if not document:
        return []

    text = _normalize_text(
        document
    )

    facts = []

    # Direct definition sentence.
    if legal_term:
        pattern = (
            rf"(whoever.+?"
            rf"(?:is said to commit|commits)"
            rf"\s+{re.escape(legal_term)})"
        )

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        for match in matches:
            if len(match) > 30:
                facts.append(match)

    # Strong direct phrase.
    phrase_patterns = (
        r"intending to take dishonestly.+?"
        r"is said to commit theft",
        r"whoever.+?is said to commit theft",
    )

    for pattern in phrase_patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        for match in matches:

            if len(match) > 30:
                facts.append(match)

    # Fallback: units containing "is said to".
    if not facts:

        for unit in _split_units(document):

            lower = unit.lower()

            if "is said to" in lower:
                facts.append(unit)

    return _unique(facts)


# ============================================================
# PUNISHMENT EXTRACTION
# ============================================================

def _extract_punishment_facts(
    document: str,
) -> List[str]:

    if not document:
        return []

    text = _normalize_text(
        document
    )

    facts = []

    # --------------------------------------------------------
    # IMPORTANT:
    # Capture the entire punishment clause, not only
    # "shall be punished with imprisonment".
    # --------------------------------------------------------

    patterns = (

        r"shall be punished with "
        r".{0,700}?"
        r"(?:fine|community service|imprisonment)"
        r"(?:\.|:)",

        r"shall be punished with "
        r".{0,700}$",

        r"punished with "
        r".{0,700}?"
        r"(?:fine|community service|imprisonment)"
        r"(?:\.|:)",

        r"shall also be liable to fine"
        r".{0,100}",

    )

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        for match in matches:

            cleaned = _normalize_text(
                match
            )

            if len(cleaned) >= 30:
                facts.append(cleaned)

    # Remove tiny/duplicate fragments.
    facts = _unique(facts)

    # --------------------------------------------------------
    # If regex did not capture the clause, inspect the text
    # around "shall be punished".
    # --------------------------------------------------------

    if not facts:

        marker = "shall be punished"

        lower = text.lower()

        start = lower.find(
            marker
        )

        if start >= 0:

            snippet = text[
                start:
                min(
                    len(text),
                    start + 1000,
                )
            ]

            facts.append(
                _normalize_text(
                    snippet
                )
            )

    return _unique(facts)


# ============================================================
# CONDITION EXTRACTION
# ============================================================

def _extract_condition_facts(
    document: str,
) -> List[str]:

    if not document:
        return []

    text = _normalize_text(
        document
    )

    facts = []

    patterns = (

        r"Provided that "
        r".{0,1000}?"
        r"community service",

        r"second or subsequent conviction"
        r".{0,700}?"
        r"(?:fine|imprisonment)",

        r"first time"
        r".{0,700}?"
        r"(?:return|restoration|community service)",

    )

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        for match in matches:

            cleaned = _normalize_text(
                match
            )

            if len(cleaned) >= 30:
                facts.append(cleaned)

    return _unique(facts)


# ============================================================
# GROUP BY SECTION
# ============================================================

def group_results_by_section(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not results:
        return []

    groups = {}

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        meta = _metadata(
            result
        )

        section = _section(
            result
        )

        source = _source(
            result
        )

        document_id = str(
            meta.get(
                "document_id",
                source,
            )
        )

        key = (
            f"{document_id}:{section}"
            if section
            else
            f"{document_id}:chunk:"
            f"{_chunk_index(result)}"
        )

        if key not in groups:

            groups[key] = {
                "section": section,
                "section_title": _section_title(result),
                "source": source,
                "title": meta.get("title", ""),
                "document_type": meta.get(
                    "document_type",
                    "",
                ),
                "year": meta.get("year"),
                "act_name": meta.get("act_name", ""),
                "document_id": document_id,
                "results": [],
                "best_score": 0.0,
            }

        groups[key]["results"].append(
            result
        )

        score = result.get(
            "hybrid_score",
            result.get(
                "legal_rerank_score",
                result.get(
                    "score",
                    0.0,
                ),
            ),
        )

        try:
            score = float(score)
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        groups[key]["best_score"] = max(
            groups[key]["best_score"],
            score,
        )

    grouped = list(
        groups.values()
    )

    for group in grouped:

        group["results"].sort(
            key=lambda item: (
                _chunk_index(item)
                if _chunk_index(item) is not None
                else 999999
            )
        )

    grouped.sort(
        key=lambda item: item["best_score"],
        reverse=True,
    )

    return grouped


# ============================================================
# ANALYZE GROUP
# ============================================================

def _analyze_group(
    group: Dict[str, Any],
    query: str,
) -> Dict[str, Any]:

    legal_term = detect_legal_term(
        query
    )

    intent = detect_fact_intent(
        query
    )

    documents = []

    for result in group.get(
        "results",
        [],
    ):

        if not isinstance(
            result,
            dict,
        ):
            continue

        document = result.get(
            "document",
            "",
        )

        if document:
            documents.append(
                str(document)
            )

    combined_document = "\n".join(
        documents
    )

    title = group.get(
        "section_title",
        "",
    )

    direct_offence = (
        _is_direct_offence_section(
            title,
            legal_term,
        )
    )

    punishment_facts = (
        _extract_punishment_facts(
            combined_document
        )
    )

    definition_facts = (
        _extract_definition_facts(
            combined_document,
            legal_term,
        )
    )

    condition_facts = (
        _extract_condition_facts(
            combined_document
        )
    )

    score = float(
        group.get(
            "best_score",
            0.0,
        )
    )

    if direct_offence:
        score += 10.0

    if intent == "punishment":

        if punishment_facts:
            score += 5.0

        if condition_facts:
            score += 1.0

    elif intent == "definition":

        if definition_facts:
            score += 5.0

    return {
        "section": group.get("section", ""),
        "section_title": title,
        "source": group.get("source", "Unknown"),
        "title": group.get("title", ""),
        "document_type": group.get(
            "document_type",
            "",
        ),
        "year": group.get("year"),
        "act_name": group.get("act_name", ""),
        "document_id": group.get(
            "document_id",
            "",
        ),
        "best_chunk": (
            _chunk_index(
                group["results"][0]
            )
            if group.get("results")
            else None
        ),
        "direct_offence": direct_offence,
        "query_intent": intent,
        "legal_term": legal_term,
        "score": score,
        "punishment_facts": punishment_facts,
        "definition_facts": definition_facts,
        "condition_facts": condition_facts,
    }


# ============================================================
# SELECT BEST
# ============================================================

def _select_best_fact_groups(
    groups: List[Dict[str, Any]],
    query: str,
    max_groups: int = 5,
) -> List[Dict[str, Any]]:

    if not groups:
        return []

    intent = detect_fact_intent(
        query
    )

    # First prioritize groups containing the requested type.
    relevant = []

    for group in groups:

        if intent == "punishment":

            if group.get(
                "punishment_facts"
            ):
                relevant.append(group)

        elif intent == "definition":

            if group.get(
                "definition_facts"
            ):
                relevant.append(group)

        else:
            relevant.append(group)

    if not relevant:
        relevant = groups

    # Sort.
    relevant.sort(
        key=lambda item: float(
            item.get(
                "score",
                0.0,
            )
        ),
        reverse=True,
    )

    # Strongly prefer DIRECT offence section.
    direct = [
        group
        for group in relevant
        if group.get(
            "direct_offence",
            False,
        )
    ]

    if direct:

        direct.sort(
            key=lambda item: float(
                item.get(
                    "score",
                    0.0,
                )
            ),
            reverse=True,
        )

        # For ordinary direct legal questions,
        # Section 303 alone is what we want.
        return direct[:max_groups]

    return relevant[:max_groups]


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_legal_facts(
    query: str,
    results: List[Dict[str, Any]],
    max_groups: int = 5,
) -> List[Dict[str, Any]]:

    if not query:
        return []

    if not isinstance(
        results,
        list,
    ):
        return []

    # Important: results must be the list returned by
    # hybrid_search().
    valid_results = [
        item
        for item in results
        if isinstance(item, dict)
    ]

    if not valid_results:
        return []

    groups = group_results_by_section(
        valid_results
    )

    if not groups:
        return []

    analyzed = []

    for group in groups:

        analyzed.append(
            _analyze_group(
                group,
                query,
            )
        )

    return _select_best_fact_groups(
        analyzed,
        query,
        max_groups=max_groups,
    )


# ============================================================
# BUILD LLM FACT CONTEXT
# ============================================================

def build_fact_context(
    facts: List[Dict[str, Any]],
    max_characters: int = 8000,
) -> str:

    if not facts:
        return ""

    blocks = []

    for index, fact in enumerate(
        facts,
        start=1,
    ):

        block = (
            f"[Legal Fact {index}]\n"
            f"Section: {fact.get('section', '')}\n"
            f"Section title: {fact.get('section_title', '')}\n"
            f"Source: {fact.get('source', '')}\n"
            f"Direct offence provision: "
            f"{'yes' if fact.get('direct_offence') else 'no'}\n"
        )

        definition_facts = fact.get(
            "definition_facts",
            [],
        )

        if definition_facts:

            block += (
                "\nDefinition provisions:\n"
            )

            for item in definition_facts:

                block += (
                    f"- {item}\n"
                )

        punishment_facts = fact.get(
            "punishment_facts",
            [],
        )

        if punishment_facts:

            block += (
                "\nPunishment provisions:\n"
            )

            for item in punishment_facts:

                block += (
                    f"- {item}\n"
                )

        condition_facts = fact.get(
            "condition_facts",
            [],
        )

        if condition_facts:

            block += (
                "\nConditions / exceptions:\n"
            )

            for item in condition_facts:

                block += (
                    f"- {item}\n"
                )

        blocks.append(
            block.strip()
        )

    context = "\n\n".join(
        blocks
    )

    if len(context) > max_characters:

        context = (
            context[:max_characters]
            + "\n[Fact context truncated]"
        )

    return context.strip()


# ============================================================
# DEBUG PRINT
# ============================================================

def print_legal_facts(
    facts: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 70)
    print("STRUCTURED LEGAL FACTS")
    print("=" * 70)

    for index, fact in enumerate(
        facts,
        start=1,
    ):

        print()
        print(f"[{index}]")
        print(
            "Section:",
            fact.get("section", ""),
        )
        print(
            "Section title:",
            fact.get("section_title", ""),
        )
        print(
            "Source:",
            fact.get("source", ""),
        )
        print(
            "Direct offence:",
            fact.get("direct_offence", False),
        )
        print(
            "Query intent:",
            fact.get("query_intent", ""),
        )
        print(
            "Legal term:",
            fact.get("legal_term", ""),
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

        if fact.get("definition_facts"):
            print("\nDefinition facts:")
            for item in fact["definition_facts"]:
                print("-", item)

        if fact.get("punishment_facts"):
            print("\nPunishment facts:")
            for item in fact["punishment_facts"]:
                print("-", item)

        if fact.get("condition_facts"):
            print("\nCondition facts:")
            for item in fact["condition_facts"]:
                print("-", item)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from backend.retriever import hybrid_search

    questions = [
        "What is the punishment for theft?",
        "What is theft?",
        "चोरी की सजा क्या है?",
        "चोरी क्या है?",
        "ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ ಏನು?",
        "ಕಳ್ಳತನ ಎಂದರೇನು?",
    ]

    print(
        "JanNyaya AI - Legal Fact Extractor Test"
    )

    for question in questions:

        print("\n")
        print("=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        results = hybrid_search(
            question,
            semantic_k=30,
            bm25_k=30,
            final_k=10,
        )

        print(
            "Retrieved results:",
            len(results),
        )

        facts = extract_legal_facts(
            question,
            results,
            max_groups=5,
        )

        print_legal_facts(
            facts
        )

        print()
        print("=" * 70)
        print("COMPACT LLM CONTEXT")
        print("=" * 70)

        print(
            build_fact_context(
                facts
            )
        )