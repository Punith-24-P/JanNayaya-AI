"""
JanNyaya AI - Legal Fact Extractor

Purpose
-------
Convert retrieved legal chunks into a compact, structured,
grounded representation for the LLM.

Main goals
----------
1. Group chunks belonging to the same legal section.
2. Prefer the direct offence section.
3. Extract punishment statements and conditions.
4. Preserve legal wording such as:
       may extend to
       shall not be less than
       shall be punished
       community service
       fine
5. Avoid mixing unrelated specialized provisions.
6. Work with English legal source documents while accepting
   English / Hindi / Kannada user questions.
7. Produce clean context for the final LLM.

This module does NOT generate legal conclusions.
It only extracts information already present in retrieved text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

MAX_SECTIONS = 3

MAX_CHUNKS_PER_SECTION = 8

MAX_FACTS_PER_SECTION = 12

MAX_CONTEXT_CHARS = 12000


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _normalize_space(text: str) -> str:
    """
    Normalize whitespace without destroying legal punctuation.
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


def _normalize_for_matching(text: str) -> str:
    """
    Normalize text for case-insensitive matching.
    """

    if not text:
        return ""

    text = str(text).lower()

    text = text.replace(
        "–",
        "-",
    )

    text = text.replace(
        "—",
        "-",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# METADATA HELPERS
# ============================================================

def _get_section(
    metadata: Dict[str, Any],
) -> str:
    """
    Safely obtain section number.
    """

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
    metadata: Dict[str, Any],
) -> str:
    """
    Safely obtain section title.
    """

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
    metadata: Dict[str, Any],
) -> str:
    """
    Safely obtain source document name.
    """

    value = metadata.get(
        "source",
        "Unknown",
    )

    if value is None:
        return "Unknown"

    return str(
        value
    ).strip()


def _get_chunk_index(
    metadata: Dict[str, Any],
) -> int:
    """
    Safely obtain chunk index.
    """

    value = metadata.get(
        "chunk_index",
        0,
    )

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0


# ============================================================
# QUERY INTENT
# ============================================================

def detect_fact_intent(
    query: str,
) -> str:
    """
    Detect whether the user is asking for:

        definition
        punishment
        general
    """

    if not query:
        return "general"

    query = str(
        query
    ).strip().lower()

    punishment_terms = (
        # English
        "punishment",
        "punished",
        "penalty",
        "sentence",
        "imprisonment",
        "jail",
        "prison",
        "fine",
        "punishable",

        # Hindi
        "सजा",
        "सज़ा",
        "दंड",
        "दण्ड",
        "जुर्माना",
        "कारावास",
        "कैद",
        "जेल",

        # Kannada
        "ಶಿಕ್ಷೆ",
        "ದಂಡ",
        "ದಂಡನೆ",
        "ಕಾರಾಗೃಹ",
        "ಜೈಲು",
        "ಸಜೆ",
    )

    for term in punishment_terms:
        if term in query:
            return "punishment"

    definition_terms = (
        # English
        "what is",
        "what are",
        "meaning of",
        "define",
        "definition",
        "what does",

        # Hindi
        "क्या है",
        "क्या होता है",
        "परिभाषा",
        "अर्थ",
        "मतलब",

        # Kannada
        "ಎಂದರೇನು",
        "ಏನು",
        "ಅರ್ಥ",
        "ಪರಿಭಾಷೆ",
    )

    for term in definition_terms:
        if term in query:
            return "definition"

    return "general"


# ============================================================
# LEGAL TERM DETECTION
# ============================================================

def detect_legal_term(
    query: str,
) -> str:
    """
    Detect the canonical offence term from the question.

    Supports:
        English
        Hindi
        Kannada
    """

    if not query:
        return ""

    query = str(
        query
    ).strip().lower()

    terms = {

        # English
        "criminal breach of trust":
            "criminal breach of trust",

        "criminal intimidation":
            "criminal intimidation",

        "cheating by personation":
            "cheating by personation",

        "snatching":
            "snatching",

        "defamation":
            "defamation",

        "kidnapping":
            "kidnapping",

        "abduction":
            "abduction",

        "robbery":
            "robbery",

        "extortion":
            "extortion",

        "assault":
            "assault",

        "murder":
            "murder",

        "rape":
            "rape",

        "cheating":
            "cheating",

        "theft":
            "theft",

        # Hindi
        "चोरी":
            "theft",

        "हत्या":
            "murder",

        "बलात्कार":
            "rape",

        "धोखाधड़ी":
            "cheating",

        "धोखा":
            "cheating",

        "लूट":
            "robbery",

        "डकैती":
            "robbery",

        "अपहरण":
            "kidnapping",

        "जबरन वसूली":
            "extortion",

        "मानहानि":
            "defamation",

        "धमकी":
            "criminal intimidation",

        # Kannada
        "ಕಳ್ಳತನ":
            "theft",

        "ಹತ್ಯೆ":
            "murder",

        "ಅತ್ಯಾಚಾರ":
            "rape",

        "ಮೋಸ":
            "cheating",

        "ದರೋಡೆ":
            "robbery",

        "ಸುಲಿಗೆ":
            "extortion",

        "ಅಪಹರಣ":
            "kidnapping",

        "ಮಾನನಷ್ಟ":
            "defamation",

        "ಬೆದರಿಕೆ":
            "criminal intimidation",

        "ದೌರ್ಜನ್ಯ":
            "assault",
    }

    for term in sorted(
        terms.keys(),
        key=len,
        reverse=True,
    ):

        if term in query:
            return terms[term]

    return ""


# ============================================================
# SECTION GROUPING
# ============================================================

def group_results_by_section(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group retrieved chunks by document + section.

    Example:

        Bharatiya_Nyaya_Sanhita_2023.pdf + 303

    becomes one group containing:

        chunk 825
        chunk 826
        ...
        chunk 833
    """

    groups: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        section = _get_section(
            metadata
        )

        source = _get_source(
            metadata
        )

        if not section:
            # Keep unsectioned evidence separately.
            key = (
                f"{source}:__NO_SECTION__:"
                f"{_get_chunk_index(metadata)}"
            )
        else:
            key = (
                f"{source}:"
                f"{section}"
            )

        groups.setdefault(
            key,
            [],
        ).append(
            result
        )

    # Preserve useful ordering.
    for group in groups.values():

        group.sort(
            key=lambda result: (
                _get_chunk_index(
                    result.get(
                        "metadata",
                        {},
                    )
                )
            )
        )

    return groups


# ============================================================
# CHUNK MERGING
# ============================================================

def merge_section_chunks(
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Merge chunk text in document order.

    Duplicate overlapping text is not aggressively removed because
    legal text can contain meaningful repeated subsection context.
    """

    if not chunks:
        return ""

    selected = chunks[
        :MAX_CHUNKS_PER_SECTION
    ]

    parts = []

    for chunk in selected:

        document = chunk.get(
            "document",
            "",
        )

        if not document:
            continue

        document = _normalize_space(
            str(document)
        )

        if not document:
            continue

        parts.append(
            document
        )

    return "\n\n".join(
        parts
    )


# ============================================================
# SECTION TITLE MATCH
# ============================================================

def is_direct_offence_section(
    section_title: str,
    legal_term: str,
) -> bool:
    """
    Determine whether a section title is the direct offence
    provision.

    Examples:

        Theft.                         -> True
        Theft by clerk or servant.    -> False
        Theft after preparation...    -> False
        Murder.                       -> True
        Punishment for murder.        -> True
    """

    if not section_title or not legal_term:
        return False

    title = _normalize_for_matching(
        section_title
    )

    term = _normalize_for_matching(
        legal_term
    )

    if title == term:
        return True

    direct_prefixes = (
        f"{term}.",
        f"{term}-",
        f"{term} ",
    )

    if title.startswith(
        direct_prefixes
    ):

        specialized_terms = (
            "by clerk",
            "after preparation",
            "in a dwelling",
            "in any building",
            "by personation",
            "by life-convict",
            "after making preparation",
            "in a dwelling house",
            "in place of worship",
        )

        for specialized in specialized_terms:

            if specialized in title:
                return False

        return True

    return False


# ============================================================
# PUNISHMENT SENTENCE EXTRACTION
# ============================================================

def extract_punishment_sentences(
    text: str,
) -> List[str]:
    """
    Extract sentences / paragraphs containing punishment language.

    This is intentionally conservative.

    We do not invent a punishment.
    We only return text already present in the evidence.
    """

    if not text:
        return []

    text = _normalize_space(
        text
    )

    # Split around obvious legal sentence boundaries.
    # We retain the actual source wording.
    pieces = re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    )

    punishment_markers = (
        "shall be punished",
        "shall be punishable",
        "punished with",
        "liable to fine",
        "imprisonment",
        "community service",
        "rigorous imprisonment",
        "imprisonment for life",
        "death",
        "fine",
        "liable to punishment",
    )

    results = []

    current_buffer = ""

    for piece in pieces:

        piece = piece.strip()

        if not piece:
            continue

        normalized = _normalize_for_matching(
            piece
        )

        has_punishment = any(
            marker in normalized
            for marker in punishment_markers
        )

        if has_punishment:

            # Some legal punishment statements are broken
            # across PDF line/chunk boundaries. Include a little
            # surrounding text where useful.
            if current_buffer:
                combined = (
                    current_buffer
                    + " "
                    + piece
                )

                results.append(
                    combined.strip()
                )

                current_buffer = ""

            else:
                results.append(
                    piece
                )

        else:

            # Keep short preceding clause available because
            # provisos/conditions can span multiple lines.
            if len(piece) < 500:

                current_buffer = piece

            else:

                current_buffer = ""

    # Remove duplicates while preserving order.
    unique = []

    seen = set()

    for item in results:

        normalized = _normalize_for_matching(
            item
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        unique.append(
            item
        )

    return unique[
        :MAX_FACTS_PER_SECTION
    ]


# ============================================================
# DEFINITION SENTENCE EXTRACTION
# ============================================================

def extract_definition_sentences(
    text: str,
) -> List[str]:
    """
    Extract source text useful for definition questions.
    """

    if not text:
        return []

    text = _normalize_space(
        text
    )

    pieces = re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    )

    markers = (
        "is said to",
        "whoever",
        "means",
        "defined as",
        "is called",
        "shall be deemed",
        "intending to",
    )

    results = []

    for piece in pieces:

        piece = piece.strip()

        if not piece:
            continue

        normalized = _normalize_for_matching(
            piece
        )

        if any(
            marker in normalized
            for marker in markers
        ):

            results.append(
                piece
            )

    unique = []

    seen = set()

    for item in results:

        key = _normalize_for_matching(
            item
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            item
        )

    return unique[
        :MAX_FACTS_PER_SECTION
    ]


# ============================================================
# SPECIAL CONDITION EXTRACTION
# ============================================================

def extract_special_conditions(
    text: str,
) -> List[str]:
    """
    Extract provisions introduced by conditions/provisos.

    Examples:

        Provided that...
        Provided...
        if...
        in case...
        first time...
        second or subsequent...
    """

    if not text:
        return []

    normalized_text = _normalize_space(
        text
    )

    patterns = (
        r"(?:Provided that|Provided|Proviso)\b.*?(?=\n\n|$)",
        r"(?:if the value|if any|if a person|in case of|in cases of).*?(?=\n\n|$)",
        r"(?:second or subsequent conviction).*?(?=\n\n|$)",
        r"(?:first time|first conviction).*?(?=\n\n|$)",
    )

    results = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            normalized_text,
            flags=re.IGNORECASE,
        )

        for match in matches:

            match = match.strip()

            if match:
                results.append(
                    match
                )

    # Remove duplicates.
    unique = []

    seen = set()

    for item in results:

        key = _normalize_for_matching(
            item
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            item
        )

    return unique[
        :MAX_FACTS_PER_SECTION
    ]


# ============================================================
# BUILD SECTION FACT
# ============================================================

def build_section_fact(
    section: str,
    section_title: str,
    source: str,
    chunks: List[Dict[str, Any]],
    query_intent: str,
    legal_term: str,
) -> Dict[str, Any]:
    """
    Build one structured fact object for one legal section.
    """

    merged_text = merge_section_chunks(
        chunks
    )

    direct_section = is_direct_offence_section(
        section_title,
        legal_term,
    )

    fact: Dict[str, Any] = {

        "section":
            section,

        "section_title":
            section_title,

        "source":
            source,

        "direct_offence_section":
            direct_section,

        "query_intent":
            query_intent,

        "legal_term":
            legal_term,

        "text":
            merged_text,

        "punishment_facts":
            [],

        "definition_facts":
            [],

        "conditions":
            [],
    }

    if query_intent == "punishment":

        fact[
            "punishment_facts"
        ] = extract_punishment_sentences(
            merged_text
        )

        fact[
            "conditions"
        ] = extract_special_conditions(
            merged_text
        )

    elif query_intent == "definition":

        fact[
            "definition_facts"
        ] = extract_definition_sentences(
            merged_text
        )

        fact[
            "conditions"
        ] = extract_special_conditions(
            merged_text
        )

    else:

        fact[
            "punishment_facts"
        ] = extract_punishment_sentences(
            merged_text
        )

        fact[
            "definition_facts"
        ] = extract_definition_sentences(
            merged_text
        )

        fact[
            "conditions"
        ] = extract_special_conditions(
            merged_text
        )

    return fact


# ============================================================
# SCORE SECTION
# ============================================================

def score_section_fact(
    fact: Dict[str, Any],
) -> float:
    """
    Rank structured section facts.

    Direct sections are strongly preferred for simple offence
    questions.

    Specialized provisions are not automatically treated as
    the main answer.
    """

    score = 0.0

    if fact.get(
        "direct_offence_section",
        False,
    ):
        score += 10.0

    query_intent = fact.get(
        "query_intent",
        "general",
    )

    if query_intent == "punishment":

        if fact.get(
            "punishment_facts"
        ):
            score += 5.0

        if fact.get(
            "conditions"
        ):
            score += 1.0

    elif query_intent == "definition":

        if fact.get(
            "definition_facts"
        ):
            score += 5.0

    # Prefer shorter direct offence sections over irrelevant
    # unrelated sections when both have legal language.
    section = str(
        fact.get(
            "section",
            "",
        )
    )

    if section:
        try:
            numeric = int(section)
            score += max(
                0.0,
                1.0 - (numeric / 10000.0),
            )
        except ValueError:
            pass

    return score


# ============================================================
# EXTRACT LEGAL FACTS
# ============================================================

def extract_legal_facts(
    results: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """
    Main entry point.

    Converts raw retrieval results into structured legal facts.

    Returns:
        List of section-level fact dictionaries.
    """

    if not results:
        return []

    query_intent = detect_fact_intent(
        query
    )

    legal_term = detect_legal_term(
        query
    )

    grouped = group_results_by_section(
        results
    )

    facts = []

    for group in grouped.values():

        if not group:
            continue

        first = group[0]

        metadata = first.get(
            "metadata",
            {},
        )

        section = _get_section(
            metadata
        )

        section_title = _get_section_title(
            metadata
        )

        source = _get_source(
            metadata
        )

        fact = build_section_fact(
            section=section,
            section_title=section_title,
            source=source,
            chunks=group,
            query_intent=query_intent,
            legal_term=legal_term,
        )

        fact[
            "score"
        ] = score_section_fact(
            fact
        )

        facts.append(
            fact
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    facts.sort(
        key=lambda item: float(
            item.get(
                "score",
                0.0,
            )
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Prefer direct section
    # --------------------------------------------------------

    direct_facts = [
        fact
        for fact in facts
        if fact.get(
            "direct_offence_section",
            False,
        )
    ]

    other_facts = [
        fact
        for fact in facts
        if not fact.get(
            "direct_offence_section",
            False,
        )
    ]

    ordered = (
        direct_facts
        + other_facts
    )

    # --------------------------------------------------------
    # Limit sections
    # --------------------------------------------------------

    ordered = ordered[
        :MAX_SECTIONS
    ]

    return ordered


# ============================================================
# COMPACT FACT CONTEXT
# ============================================================

def build_fact_context(
    facts: List[Dict[str, Any]],
    query: str,
    max_characters: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Convert structured facts into compact LLM context.

    This is deliberately much smaller and cleaner than sending
    all retrieved chunks to the model.
    """

    if not facts:
        return ""

    query_intent = detect_fact_intent(
        query
    )

    parts = []

    current_length = 0

    for index, fact in enumerate(
        facts,
        start=1,
    ):

        section = fact.get(
            "section",
            "",
        )

        title = fact.get(
            "section_title",
            "",
        )

        source = fact.get(
            "source",
            "Unknown",
        )

        direct = fact.get(
            "direct_offence_section",
            False,
        )

        block = (
            f"[Legal Fact {index}]\n"
            f"Section: {section}\n"
            f"Section title: {title}\n"
            f"Source: {source}\n"
            f"Direct offence provision: "
            f"{'yes' if direct else 'no'}\n"
        )

        # ----------------------------------------------------
        # Punishment facts
        # ----------------------------------------------------

        if query_intent == "punishment":

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

            conditions = fact.get(
                "conditions",
                [],
            )

            if conditions:

                block += (
                    "\nSpecial conditions:\n"
                )

                for item in conditions:

                    block += (
                        f"- {item}\n"
                    )

        # ----------------------------------------------------
        # Definition facts
        # ----------------------------------------------------

        elif query_intent == "definition":

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

        # ----------------------------------------------------
        # General
        # ----------------------------------------------------

        else:

            definition_facts = fact.get(
                "definition_facts",
                [],
            )

            punishment_facts = fact.get(
                "punishment_facts",
                [],
            )

            conditions = fact.get(
                "conditions",
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

            if punishment_facts:

                block += (
                    "\nPunishment provisions:\n"
                )

                for item in punishment_facts:

                    block += (
                        f"- {item}\n"
                    )

            if conditions:

                block += (
                    "\nConditions:\n"
                )

                for item in conditions:

                    block += (
                        f"- {item}\n"
                    )

        block = _normalize_space(
            block
        )

        # ----------------------------------------------------
        # Context limit
        # ----------------------------------------------------

        extra = (
            2
            if parts
            else 0
        )

        if (
            current_length
            + extra
            + len(block)
            > max_characters
        ):

            remaining = (
                max_characters
                - current_length
                - extra
            )

            if remaining > 300:

                parts.append(
                    block[
                        :remaining
                    ]
                    + "\n[Fact context truncated]"
                )

            break

        parts.append(
            block
        )

        current_length += (
            extra
            + len(block)
        )

    return "\n\n".join(
        parts
    )


# ============================================================
# DEBUG PRINT
# ============================================================

def print_legal_facts(
    facts: List[Dict[str, Any]],
) -> None:
    """
    Display extracted legal facts for debugging.
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

    if not facts:

        print(
            "No legal facts extracted."
        )

        return

    for index, fact in enumerate(
        facts,
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
                "direct_offence_section",
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

        punishment_facts = fact.get(
            "punishment_facts",
            [],
        )

        if punishment_facts:

            print(
                "\nPunishment facts:"
            )

            for item in punishment_facts:
                print(
                    "-",
                    item,
                )

        definition_facts = fact.get(
            "definition_facts",
            [],
        )

        if definition_facts:

            print(
                "\nDefinition facts:"
            )

            for item in definition_facts:
                print(
                    "-",
                    item,
                )

        conditions = fact.get(
            "conditions",
            [],
        )

        if conditions:

            print(
                "\nConditions:"
            )

            for item in conditions:
                print(
                    "-",
                    item,
                )

        print(
            "-" * 60
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "JanNyaya AI - Legal Fact Extractor Test"
    )
    print()

    # --------------------------------------------------------
    # Minimal realistic Section 303 test
    # --------------------------------------------------------

    sample_results = [

        {
            "document": """
303. Theft.—(1) Whoever, intending to take dishonestly any
movable property out of the possession of any person without
that person's consent, moves that property in order to such
taking, is said to commit theft.

(2) Whoever commits theft shall be punished with imprisonment
of either description for a term which may extend to three
years, or with fine, or with both and in case of second or
subsequent conviction of any person under this section, he shall
be punished with rigorous imprisonment for a term which shall
not be less than one year but which may extend to five years
and with fine:

Provided that in cases of theft where the value of the stolen
property is less than five thousand rupees, and a person is
convicted for the first time, upon return of the value of
property or restoration of the stolen property, the person
shall be punished with community service.
""",
            "metadata": {
                "document_id":
                    "act_bns_2023",

                "document_type":
                    "Act",

                "title":
                    "Bharatiya Nyaya Sanhita, 2023",

                "source":
                    "Bharatiya_Nyaya_Sanhita_2023.pdf",

                "chunk_index":
                    833,

                "section_number":
                    "303",

                "section_title":
                    "Theft.—(1) Whoever, intending to take dishonestly any movable property out of the possession",
            },
        },

        {
            "document": """
305. Theft in a dwelling house, or means of transportation
or place of worship, etc.—Whoever commits theft in any
building, tent or vessel used as a human dwelling shall be
punished with imprisonment which may extend to seven years,
and shall also be liable to fine.
""",
            "metadata": {
                "document_id":
                    "act_bns_2023",

                "document_type":
                    "Act",

                "title":
                    "Bharatiya Nyaya Sanhita, 2023",

                "source":
                    "Bharatiya_Nyaya_Sanhita_2023.pdf",

                "chunk_index":
                    835,

                "section_number":
                    "305",

                "section_title":
                    "Theft in a dwelling house, or means of transportation or place of worship, etc.",
            },
        },

    ]

    test_questions = [
        "What is the punishment for theft?",
        "What is theft?",
        "चोरी की सजा क्या है?",
        "चोरी क्या है?",
        "ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ ಏನು?",
        "ಕಳ್ಳತನ ಎಂದರೇನು?",
    ]

    for question in test_questions:

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

        facts = extract_legal_facts(
            sample_results,
            question,
        )

        print_legal_facts(
            facts
        )

        print()
        print(
            "COMPACT LLM CONTEXT:"
        )

        print(
            build_fact_context(
                facts,
                question,
            )
        )

        print()