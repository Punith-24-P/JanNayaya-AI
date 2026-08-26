"""
JanNyaya AI - Hybrid Legal Retriever

Retrieval pipeline:

    Semantic Search
           +
        BM25 Search
           ↓
    Reciprocal Rank Fusion
           ↓
    Legal Intent Detection
           ↓
    Legal Evidence Reranking
           ↓
    Duplicate Control
           ↓
      Final Top-K

Designed for Indian legal documents where several sections
may mention the same offence but only one section may contain
the direct definition or direct punishment provision.
"""

import os
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from backend.embedding_service import create_embedding
from backend.vector_store import (
    get_all_documents,
    search_by_embedding,
)

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

RAG_TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "5",
    )
)

# ============================================================
# LEGAL INTENT
# ============================================================

PUNISHMENT_WORDS = {
    "punishment",
    "punished",
    "penalty",
    "penalties",
    "sentence",
    "sentenced",
    "imprisonment",
    "fine",
    "punishable",
    "community",
    "jail",
    "prison",
}

DEFINITION_WORDS = {
    "define",
    "definition",
    "meaning",
    "means",
    "what",
}

DEFINITION_PHRASES = (
    "what is",
    "what are",
    "what does",
    "meaning of",
    "define",
    "definition of",
    "explain",
)

# ============================================================
# KNOWN LEGAL TERMS
# ============================================================

LEGAL_TERMS = [
    "criminal breach of trust",
    "criminal intimidation",
    "cheating by personation",
    "petty organised crime",
    "petty organized crime",
    "organized crime",
    "organised crime",
    "house trespass",
    "house-trespass",
    "house breaking",
    "house-breaking",
    "kidnapping",
    "abduction",
    "snatching",
    "defamation",
    "extortion",
    "robbery",
    "assault",
    "murder",
    "rape",
    "cheating",
    "theft",
]

# ============================================================
# TOKENIZER
# ============================================================

def _tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 search.
    """

    if not text:
        return []

    text = text.lower()

    return re.findall(
        r"[a-zA-Z0-9₹]+",
        text,
    )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Normalize text for legal matching.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Normalize common legal punctuation.
    text = text.replace("—", " ")
    text = text.replace("–", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"[^a-z0-9₹\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# DETECT QUERY INTENT
# ============================================================

def detect_query_intent(query: str) -> str:
    """
    Detect the main legal question intent.

    Returns:
        punishment
        definition
        general
    """

    normalized = _normalize_text(query)

    words = set(
        normalized.split()
    )

    # Punishment intent gets priority because questions such as
    # "What is the punishment for murder?" contain "what is".
    if words.intersection(
        PUNISHMENT_WORDS
    ):
        return "punishment"

    if any(
        phrase in normalized
        for phrase in DEFINITION_PHRASES
    ):
        return "definition"

    if words.intersection(
        DEFINITION_WORDS
    ):
        return "definition"

    return "general"


# ============================================================
# EXTRACT MAIN LEGAL TERM
# ============================================================

def extract_query_legal_term(query: str) -> str:
    """
    Extract the main offence/legal term from the question.

    Longest terms are checked first.
    """

    normalized = _normalize_text(
        query
    )

    for term in sorted(
        LEGAL_TERMS,
        key=len,
        reverse=True,
    ):
        normalized_term = _normalize_text(
            term
        )

        if normalized_term in normalized:
            return normalized_term

    return ""


# ============================================================
# SECTION NUMBER
# ============================================================

def _get_section(
    metadata: Dict[str, Any],
) -> str:
    """
    Return section number as a clean string.
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


# ============================================================
# SECTION TITLE
# ============================================================

def _get_section_title(
    metadata: Dict[str, Any],
) -> str:
    """
    Return section title as a clean string.
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


# ============================================================
# RESULT KEY
# ============================================================

def _result_key(
    result: Dict[str, Any],
) -> str:
    """
    Create a stable key for the same legal chunk.

    document_id + chunk_index is preferred.
    """

    metadata = result.get(
        "metadata",
        {},
    )

    document_id = metadata.get(
        "document_id",
        "",
    )

    chunk_index = metadata.get(
        "chunk_index",
        "",
    )

    document = result.get(
        "document",
        "",
    )

    if document_id:
        return (
            f"{document_id}:"
            f"{chunk_index}"
        )

    return document[:500]


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query: str,
    documents: List[str],
    metadatas: List[dict],
    top_k: int = 30,
) -> List[Dict[str, Any]]:
    """
    Perform lexical BM25 search.
    """

    if not query or not query.strip():
        return []

    if not documents:
        return []

    tokenized_documents = [
        _tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = _tokenize(
        query
    )

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    ranked_indexes = ranked_indexes[
        :top_k
    ]

    results = []

    for rank, index in enumerate(
        ranked_indexes,
        start=1,
    ):
        results.append(
            {
                "document": documents[index],
                "metadata": metadatas[index],
                "score": float(
                    scores[index]
                ),
                "bm25_rank": rank,
                "method": "bm25",
            }
        )

    return results


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    top_k: int = 30,
) -> List[Dict[str, Any]]:
    """
    Semantic vector search using SentenceTransformer
    and ChromaDB.
    """

    if not query or not query.strip():
        return []

    embedding = create_embedding(
        query
    )

    if not embedding:
        return []

    results = search_by_embedding(
        embedding,
        n_results=top_k,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    output = []

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        index = rank - 1

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else 0.0
        )

        # ChromaDB distance:
        # lower = more similar.
        #
        # Convert to similarity-like score.
        semantic_score = (
            1.0
            / (
                1.0
                + float(distance)
            )
        )

        output.append(
            {
                "document": document,
                "metadata": metadata,
                "score": float(
                    semantic_score
                ),
                "semantic_rank": rank,
                "method": "semantic",
            }
        )

    return output


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def _min_max_normalize(
    values: List[float],
) -> List[float]:
    """
    Normalize numeric values into [0, 1].
    """

    if not values:
        return []

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    if maximum == minimum:
        return [
            1.0
            for _ in values
        ]

    return [
        (
            value - minimum
        )
        / (
            maximum - minimum
        )
        for value in values
    ]


# ============================================================
# LEGAL EVIDENCE FEATURES
# ============================================================

def _calculate_legal_features(
    query: str,
    result: Dict[str, Any],
) -> Dict[str, float]:
    """
    Calculate transparent legal-specific ranking features.
    """

    document = str(
        result.get(
            "document",
            "",
        )
    )

    metadata = result.get(
        "metadata",
        {},
    )

    normalized_document = _normalize_text(
        document
    )

    normalized_title = _normalize_text(
        _get_section_title(
            metadata
        )
    )

    normalized_query = _normalize_text(
        query
    )

    query_term = extract_query_legal_term(
        query
    )

    query_intent = detect_query_intent(
        query
    )

    # ========================================================
    # BASIC TERM MATCH
    # ========================================================

    query_term_match = 0.0

    if query_term:

        if query_term in normalized_title:
            query_term_match = 1.0

        elif query_term in normalized_document:
            query_term_match = 0.7

    # ========================================================
    # EXACT TITLE MATCH
    # ========================================================

    exact_title_match = 0.0

    if query_term:

        title = normalized_title

        if title == query_term:
            exact_title_match = 1.0

        elif title.startswith(
            query_term + " "
        ):
            exact_title_match = 0.9

    # ========================================================
    # PUNISHMENT TITLE MATCH
    # ========================================================

    punishment_title_match = 0.0

    if (
        query_intent == "punishment"
        and query_term
    ):

        title = normalized_title

        direct_patterns = [
            f"punishment for {query_term}",
            f"punishment of {query_term}",
            f"punishable for {query_term}",
        ]

        for pattern in direct_patterns:

            if title.startswith(
                pattern
            ):

                punishment_title_match = 1.0
                break

        # Example:
        # "Theft"
        # "Theft by clerk or servant..."
        #
        # Basic direct offence provisions still deserve
        # a strong score when they contain actual
        # punishment language.

        if (
            punishment_title_match == 0.0
            and title.startswith(
                query_term
            )
        ):
            punishment_title_match = 0.85

    # ========================================================
    # PUNISHMENT LANGUAGE
    # ========================================================

    punishment_matches = 0

    punishment_patterns = [
        "shall be punished",
        "shall be punishable",
        "punished with",
        "liable to fine",
        "liable to punishment",
        "imprisonment",
        "community service",
        "fine",
    ]

    for pattern in punishment_patterns:

        if pattern in normalized_document:
            punishment_matches += 1

    punishment_score = min(
        punishment_matches / 4.0,
        1.0,
    )

    # ========================================================
    # DEFINITION LANGUAGE
    # ========================================================

    definition_matches = 0

    definition_patterns = [
        "means",
        "is said to",
        "is called",
        "definition",
        "whoever",
        "intending to",
        "is said to cheat",
        "is said to commit",
    ]

    for pattern in definition_patterns:

        if pattern in normalized_document:
            definition_matches += 1

    definition_score = min(
        definition_matches / 4.0,
        1.0,
    )

    # ========================================================
    # DIRECT DEFINITION TITLE
    # ========================================================

    definition_title_match = 0.0

    if (
        query_intent == "definition"
        and query_term
    ):

        title = normalized_title

        if title == query_term:
            definition_title_match = 1.0

        elif title.startswith(
            query_term + " "
        ):
            definition_title_match = 0.9

    # ========================================================
    # SPECIALIZED PROVISION
    # ========================================================

    specialized_penalty = 0.0

    specialized_terms = [
        "after preparation",
        "dwelling house",
        "means of transportation",
        "place of worship",
        "government",
        "local authority",
        "assault or criminal force",
        "clerk or servant",
        "house trespass",
        "house breaking",
        "house breaking",
        "organised crime",
        "organized crime",
        "petty organised crime",
        "petty organized crime",
        "kidnapping",
        "abducting",
        "attempt",
        "attempt to murder",
        "attempt to commit",
        "receiving stolen property",
        "stolen property",
        "personation",
    ]

    for term in specialized_terms:

        if term in normalized_title:

            specialized_penalty = max(
                specialized_penalty,
                1.0,
            )

    # ========================================================
    # VERY SHORT HEADING DETECTION
    # ========================================================

    short_heading_penalty = 0.0

    if len(
        normalized_document
    ) < 100:

        short_heading_penalty = 1.0

    # ========================================================
    # DIRECT PUNISHMENT LANGUAGE
    # ========================================================

    direct_punishment_language = 0.0

    direct_punishment_phrases = [
        "shall be punished with",
        "shall be punished",
        "punishment for",
        "punishment of",
    ]

    for phrase in direct_punishment_phrases:

        if phrase in normalized_document:

            direct_punishment_language = max(
                direct_punishment_language,
                1.0,
            )

    # ========================================================
    # MAIN OFFENCE TITLE SIGNAL
    # ========================================================

    direct_offence_title = 0.0

    if query_term:

        title = normalized_title

        # Example:
        # "theft"
        # "murder"
        # "cheating"
        # "rape"

        if title == query_term:

            direct_offence_title = 1.0

        elif title.startswith(
            query_term + " "
        ):

            direct_offence_title = 0.9

    # ========================================================
    # RETURN FEATURES
    # ========================================================

    return {
        "query_term_match":
            query_term_match,

        "exact_title_match":
            exact_title_match,

        "punishment_title_match":
            punishment_title_match,

        "punishment_score":
            punishment_score,

        "definition_score":
            definition_score,

        "definition_title_match":
            definition_title_match,

        "specialized_penalty":
            specialized_penalty,

        "short_heading_penalty":
            short_heading_penalty,

        "direct_punishment_language":
            direct_punishment_language,

        "direct_offence_title":
            direct_offence_title,

        "query_intent":
            1.0
            if query_intent == "punishment"
            else (
                0.5
                if query_intent == "definition"
                else 0.0
            ),

        "query_term_present":
            1.0
            if query_term
            and query_term in normalized_document
            else 0.0,

        "query_present":
            1.0
            if normalized_query
            and normalized_query in normalized_document
            else 0.0,
    }


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    semantic_k: int = 30,
    bm25_k: int = 30,
    final_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid legal retrieval.

    Steps:

        1. Semantic search
        2. BM25
        3. RRF
        4. Preserve both search signals
        5. Legal reranking
        6. Duplicate control
        7. Final results
    """

    if not query or not query.strip():
        return []

    query = query.strip()

    if final_k is None:
        final_k = RAG_TOP_K

    final_k = max(
        1,
        int(final_k),
    )

    semantic_k = max(
        10,
        int(semantic_k),
    )

    bm25_k = max(
        10,
        int(bm25_k),
    )

    # ========================================================
    # QUERY INTENT
    # ========================================================

    query_intent = detect_query_intent(
        query
    )

    query_term = extract_query_legal_term(
        query
    )

    print(
        f"Detected intent: {query_intent}"
    )

    print(
        f"Detected legal term: "
        f"{query_term or 'None'}"
    )

    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    print(
        "Running semantic search..."
    )

    semantic_results = semantic_search(
        query,
        semantic_k,
    )

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    print(
        "Running BM25 search..."
    )

    documents, metadatas = (
        get_all_documents()
    )

    bm25_results = bm25_search(
        query,
        documents,
        metadatas,
        bm25_k,
    )

    # ========================================================
    # COMBINE BY UNIQUE CHUNK
    # ========================================================

    candidates: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------------
    # Semantic results
    # --------------------------------------------------------

    for result in semantic_results:

        key = _result_key(
            result
        )

        if key not in candidates:

            candidates[key] = {
                "document":
                    result.get(
                        "document",
                        "",
                    ),

                "metadata":
                    result.get(
                        "metadata",
                        {},
                    ),

                "semantic_score":
                    float(
                        result.get(
                            "score",
                            0.0,
                        )
                    ),

                "semantic_rank":
                    int(
                        result.get(
                            "semantic_rank",
                            999999,
                        )
                    ),

                "bm25_score":
                    0.0,

                "bm25_rank":
                    None,

                "semantic_found":
                    True,

                "bm25_found":
                    False,

            }

        else:

            candidates[key][
                "semantic_score"
            ] = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            candidates[key][
                "semantic_rank"
            ] = int(
                result.get(
                    "semantic_rank",
                    999999,
                )
            )

            candidates[key][
                "semantic_found"
            ] = True

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for result in bm25_results:

        key = _result_key(
            result
        )

        if key not in candidates:

            candidates[key] = {
                "document":
                    result.get(
                        "document",
                        "",
                    ),

                "metadata":
                    result.get(
                        "metadata",
                        {},
                    ),

                "semantic_score":
                    0.0,

                "semantic_rank":
                    None,

                "bm25_score":
                    float(
                        result.get(
                            "score",
                            0.0,
                        )
                    ),

                "bm25_rank":
                    int(
                        result.get(
                            "bm25_rank",
                            999999,
                        )
                    ),

                "semantic_found":
                    False,

                "bm25_found":
                    True,

            }

        else:

            candidates[key][
                "bm25_score"
            ] = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            candidates[key][
                "bm25_rank"
            ] = int(
                result.get(
                    "bm25_rank",
                    999999,
                )
            )

            candidates[key][
                "bm25_found"
            ] = True

    if not candidates:
        return []

    # ========================================================
    # NORMALIZE SEARCH SCORES
    # ========================================================

    all_bm25_scores = [
        float(
            item.get(
                "bm25_score",
                0.0,
            )
        )
        for item in candidates.values()
        if item.get(
            "bm25_found",
            False,
        )
    ]

    all_semantic_scores = [
        float(
            item.get(
                "semantic_score",
                0.0,
            )
        )
        for item in candidates.values()
        if item.get(
            "semantic_found",
            False,
        )
    ]

    normalized_bm25_values = (
        _min_max_normalize(
            all_bm25_scores
        )
    )

    normalized_semantic_values = (
        _min_max_normalize(
            all_semantic_scores
        )
    )

    bm25_normalized_map = {}
    semantic_normalized_map = {}

    bm25_cursor = 0
    semantic_cursor = 0

    for key, item in candidates.items():

        if item.get(
            "bm25_found",
            False,
        ):

            bm25_normalized_map[key] = (
                normalized_bm25_values[
                    bm25_cursor
                ]
            )

            bm25_cursor += 1

        else:

            bm25_normalized_map[key] = 0.0

        if item.get(
            "semantic_found",
            False,
        ):

            semantic_normalized_map[key] = (
                normalized_semantic_values[
                    semantic_cursor
                ]
            )

            semantic_cursor += 1

        else:

            semantic_normalized_map[key] = 0.0

    # ========================================================
    # SCORE EACH CANDIDATE
    # ========================================================

    ranked_candidates = []

    for key, item in candidates.items():

        metadata = item.get(
            "metadata",
            {},
        )

        document = str(
            item.get(
                "document",
                "",
            )
        )

        # ----------------------------------------------------
        # RRF SCORE
        # ----------------------------------------------------

        semantic_rank = item.get(
            "semantic_rank"
        )

        bm25_rank = item.get(
            "bm25_rank"
        )

        rrf_score = 0.0

        if (
            semantic_rank is not None
            and semantic_rank < 999999
        ):

            rrf_score += (
                1.0
                / (
                    60.0
                    + float(
                        semantic_rank
                    )
                )
            )

        if (
            bm25_rank is not None
            and bm25_rank < 999999
        ):

            rrf_score += (
                1.0
                / (
                    60.0
                    + float(
                        bm25_rank
                    )
                )
            )

        # ----------------------------------------------------
        # BUILD TEMP RESULT
        # ----------------------------------------------------

        temp_result = {
            "document":
                document,

            "metadata":
                metadata,

            "rrf_score":
                rrf_score,

            "bm25_score":
                item.get(
                    "bm25_score",
                    0.0,
                ),

            "semantic_score":
                item.get(
                    "semantic_score",
                    0.0,
                ),

            "bm25_rank":
                bm25_rank,

            "semantic_rank":
                semantic_rank,

            "method":
                "hybrid",
        }

        # ----------------------------------------------------
        # LEGAL FEATURES
        # ----------------------------------------------------

        features = (
            _calculate_legal_features(
                query,
                temp_result,
            )
        )

        # ----------------------------------------------------
        # SEARCH SIGNALS
        # ----------------------------------------------------

        bm25_normalized = (
            bm25_normalized_map.get(
                key,
                0.0,
            )
        )

        semantic_normalized = (
            semantic_normalized_map.get(
                key,
                0.0,
            )
        )

        # ----------------------------------------------------
        # START WITH RRF
        # ----------------------------------------------------

        final_score = (
            rrf_score
            * 1.00
        )

        # ----------------------------------------------------
        # SEARCH SCORE SUPPORT
        # ----------------------------------------------------

        final_score += (
            bm25_normalized
            * 0.12
        )

        final_score += (
            semantic_normalized
            * 0.12
        )

        # ----------------------------------------------------
        # QUERY TERM MATCH
        # ----------------------------------------------------

        final_score += (
            features[
                "query_term_match"
            ]
            * 0.25
        )

        # ----------------------------------------------------
        # PUNISHMENT QUESTIONS
        # ----------------------------------------------------

        if query_intent == "punishment":

            # Direct title such as:
            # "Punishment for murder"
            # gets the strongest boost.

            final_score += (
                features[
                    "punishment_title_match"
                ]
                * 2.50
            )

            # A direct basic offence provision such as:
            # "303. Theft"
            # should also rank highly if it contains
            # the actual punishment clause.

            final_score += (
                features[
                    "direct_offence_title"
                ]
                * 1.00
            )

            final_score += (
                features[
                    "punishment_score"
                ]
                * 1.35
            )

            final_score += (
                features[
                    "direct_punishment_language"
                ]
                * 0.75
            )

        # ----------------------------------------------------
        # DEFINITION QUESTIONS
        # ----------------------------------------------------

        elif query_intent == "definition":

            final_score += (
                features[
                    "definition_title_match"
                ]
                * 2.00
            )

            final_score += (
                features[
                    "direct_offence_title"
                ]
                * 1.00
            )

            final_score += (
                features[
                    "definition_score"
                ]
                * 1.25
            )

        # ----------------------------------------------------
        # GENERAL QUESTIONS
        # ----------------------------------------------------

        else:

            final_score += (
                features[
                    "direct_offence_title"
                ]
                * 1.20
            )

            final_score += (
                features[
                    "definition_score"
                ]
                * 0.50
            )

            final_score += (
                features[
                    "punishment_score"
                ]
                * 0.40
            )

        # ----------------------------------------------------
        # MULTI-SIGNAL BONUS
        #
        # A chunk appearing in BOTH BM25 and semantic
        # search is more trustworthy than one appearing
        # in only one.
        # ----------------------------------------------------

        if (
            item.get(
                "semantic_found",
                False,
            )
            and item.get(
                "bm25_found",
                False,
            )
        ):

            final_score += 0.30

        # ----------------------------------------------------
        # SPECIALIZED PROVISION PENALTY
        #
        # We only penalize these when the user asks about
        # the offence generally.
        #
        # Example:
        # "What is the punishment for theft?"
        #
        # Section 305/306/307 are specialized theft
        # provisions and should appear below Section 303.
        # ----------------------------------------------------

        if (
            query_term
            and features[
                "specialized_penalty"
            ] > 0
        ):

            final_score -= (
                features[
                    "specialized_penalty"
                ]
                * 0.65
            )

        # ----------------------------------------------------
        # VERY SHORT HEADING PENALTY
        # ----------------------------------------------------

        final_score -= (
            features[
                "short_heading_penalty"
            ]
            * 0.75
        )

        # ----------------------------------------------------
        # STRONG PROTECTION FOR DIRECT PROVISION
        #
        # This is the critical part.
        #
        # Punishment question:
        # "What is the punishment for murder?"
        #
        # Section 103:
        # "Punishment for murder..."
        #
        # Definition question:
        # "What is cheating?"
        #
        # Section 318:
        # "Cheating..."
        # ----------------------------------------------------

        if query_term:

            if query_intent == "punishment":

                if (
                    features[
                        "punishment_title_match"
                    ] >= 0.9
                ):

                    final_score += 1.50

                elif (
                    features[
                        "direct_offence_title"
                    ] >= 0.9
                    and features[
                        "punishment_score"
                    ] >= 0.5
                ):

                    final_score += 0.95

            elif query_intent == "definition":

                if (
                    features[
                        "definition_title_match"
                    ] >= 0.9
                ):

                    final_score += 1.20

        # ----------------------------------------------------
        # STORE RANKING DATA
        # ----------------------------------------------------

        temp_result[
            "bm25_normalized"
        ] = bm25_normalized

        temp_result[
            "semantic_normalized"
        ] = semantic_normalized

        temp_result[
            "legal_rerank_score"
        ] = float(
            final_score
        )

        temp_result[
            "hybrid_score"
        ] = float(
            final_score
        )

        temp_result[
            "query_intent"
        ] = query_intent

        temp_result[
            "query_legal_term"
        ] = query_term

        temp_result[
            "section_number"
        ] = _get_section(
            metadata
        )

        temp_result[
            "section_title"
        ] = _get_section_title(
            metadata
        )

        ranked_candidates.append(
            temp_result
        )

    # ========================================================
    # SORT
    # ========================================================

    ranked_candidates.sort(
        key=lambda item:
            float(
                item.get(
                    "legal_rerank_score",
                    0.0,
                )
            ),
        reverse=True,
    )

    # ========================================================
    # FINAL DUPLICATE CONTROL
    # ========================================================

    final_results = []

    seen_keys = set()

    for result in ranked_candidates:

        key = _result_key(
            result
        )

        if key in seen_keys:
            continue

        seen_keys.add(
            key
        )

        final_results.append(
            result
        )

        if len(final_results) >= final_k:
            break

    return final_results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results: List[Dict[str, Any]],
) -> None:
    """
    Print retrieved results for debugging.
    """

    print()
    print("=" * 70)
    print("TOP RETRIEVED RESULTS")
    print("=" * 70)

    if not results:

        print(
            "\nNo results found."
        )

        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print()
        print(
            f"[{index}]"
        )

        print(
            "Section:",
            metadata.get(
                "section_number",
                metadata.get(
                    "section",
                    "None",
                ),
            ),
        )

        print(
            "Section title:",
            metadata.get(
                "section_title",
                "",
            ),
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Chunk:",
            metadata.get(
                "chunk_index",
                "None",
            ),
        )

        print(
            "Query intent:",
            result.get(
                "query_intent",
                "",
            ),
        )

        print(
            "Query legal term:",
            result.get(
                "query_legal_term",
                "",
            ),
        )

        print(
            "Semantic rank:",
            result.get(
                "semantic_rank",
                "",
            ),
        )

        print(
            "BM25 rank:",
            result.get(
                "bm25_rank",
                "",
            ),
        )

        print(
            "RRF score:",
            round(
                float(
                    result.get(
                        "rrf_score",
                        0.0,
                    )
                ),
                6,
            ),
        )

        print(
            "Hybrid/legal score:",
            round(
                float(
                    result.get(
                        "hybrid_score",
                        0.0,
                    )
                ),
                6,
            ),
        )

        print()
        print(
            "Text:"
        )

        print(
            result.get(
                "document",
                "",
            )
        )

        print(
            "-" * 60
        )


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "JanNyaya AI - Retriever Test"
    )

    print(
        "Type a legal question."
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        question = input(
            "\nEnter legal question: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nExiting."
            )

            break

        if not question:

            continue

        try:

            results = hybrid_search(
                question,
                semantic_k=30,
                bm25_k=30,
                final_k=RAG_TOP_K,
            )

            print_results(
                results
            )

        except Exception as error:

            print()
            print(
                "ERROR:"
            )
            print(
                str(error)
            )