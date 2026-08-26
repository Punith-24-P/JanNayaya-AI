"""
JanNyaya AI - Multilingual Hybrid Legal Retriever

Pipeline:

User Query
    ↓
Intent Detection
    ↓
Multilingual Semantic Search
    ↓
BM25 Lexical Search
    ↓
Reciprocal Rank Fusion
    ↓
Legal-aware Reranking
    ↓
Duplicate Section Control
    ↓
Top-K Evidence

Designed for:
    English
    Hindi
    Kannada

The retriever works with:
    intfloat/multilingual-e5-small
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple

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
# RRF
# ============================================================

RRF_K = 60.0

# ============================================================
# SCORING WEIGHTS
# ============================================================

WEIGHT_RRF = 1.0

WEIGHT_BM25 = 0.10
WEIGHT_SEMANTIC = 0.10

WEIGHT_QUERY_TERM = 0.30
WEIGHT_DIRECT_SECTION = 0.90

WEIGHT_PUNISHMENT = 1.20
BONUS_PUNISHMENT_LANGUAGE = 0.50

WEIGHT_DEFINITION = 1.10
BONUS_DEFINITION_LANGUAGE = 0.30

PENALTY_SPECIALIZED_SECTION = 0.35

# ============================================================
# LEGAL INTENT
# ============================================================

PUNISHMENT_WORDS = {
    "punishment",
    "punished",
    "penalty",
    "penalties",
    "sentence",
    "imprisonment",
    "fine",
    "punishable",
    "punishable",
    "community",
    "सजा",
    "दंड",
    "जुर्माना",
    "कारावास",
    "ಶಿಕ್ಷೆ",
    "ದಂಡ",
    "ಜೈಲು",
}

DEFINITION_PHRASES = (
    "what is",
    "what are",
    "what does",
    "meaning of",
    "define",
    "definition of",
    "explain",

    "क्या है",
    "क्या होता है",
    "अर्थ",
    "परिभाषा",

    "ಏನು",
    "ಎಂದರೇನು",
    "ಅರ್ಥವೇನು",
)

# ============================================================
# LEGAL TERMS
# ============================================================

LEGAL_TERMS = [
    # English
    "criminal breach of trust",
    "criminal intimidation",
    "cheating by personation",
    "petty organised crime",
    "petty organized crime",
    "organized crime",
    "organised crime",
    "house trespass",
    "house breaking",
    "house-breaking",
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

    # Hindi
    "आपराधिक विश्वासघात",
    "आपराधिक धमकी",
    "धोखाधड़ी",
    "चोरी",
    "डकैती",
    "बलात्कार",
    "हत्या",
    "अपहरण",
    "मानहानि",

    # Kannada
    "ವಿಶ್ವಾಸದ್ರೋಹ",
    "ಕ್ರಿಮಿನಲ್ ಬೆದರಿಕೆ",
    "ಮೋಸ",
    "ಕಳ್ಳತನ",
    "ದರೋಡೆ",
    "ಅತ್ಯಾಚಾರ",
    "ಕೊಲೆ",
    "ಅಪಹರಣ",
    "ಮಾನಹಾನಿ",
]

# ============================================================
# SPECIALIZED SECTION TERMS
# ============================================================

SPECIALIZED_TITLE_TERMS = [
    "after preparation",
    "dwelling house",
    "means of transportation",
    "place of worship",
    "government",
    "local authority",
    "assault or criminal force",
    "clerk or servant",
    "house trespass",
    "house-breaking",
    "organized crime",
    "organised crime",
    "petty organized crime",
    "petty organised crime",

    "तैयारी",
    "गृह",
    "परिवहन",
    "सरकार",
    "स्थानीय प्राधिकरण",

    "ಮನೆ",
    "ಸಾರಿಗೆ",
    "ಸರ್ಕಾರ",
]

# ============================================================
# PUNISHMENT PATTERNS
# ============================================================

PUNISHMENT_PATTERNS = [
    "shall be punished",
    "shall be punishable",
    "punished with",
    "liable to fine",
    "liable to punishment",
    "imprisonment",
    "community service",
    "fine",

    "सजा",
    "दंड",
    "जुर्माना",
    "कारावास",

    "ಶಿಕ್ಷೆ",
    "ದಂಡ",
    "ಜೈಲು",
]

# ============================================================
# DEFINITION PATTERNS
# ============================================================

DEFINITION_PATTERNS = [
    "means",
    "is said to",
    "definition",
    "whoever",
    "intending to",

    "अर्थ",
    "परिभाषा",
    "जो कोई",

    "ಎಂದರೆ",
    "ವ್ಯಾಖ್ಯಾನ",
]

# ============================================================
# TOKENIZER
# ============================================================

def _tokenize(text: str) -> List[str]:
    """
    Unicode-aware tokenizer.

    Keeps:
        English
        Hindi
        Kannada
        Numbers
        ₹
    """

    if not text:
        return []

    text = str(text).lower()

    return re.findall(
        r"[^\W_]+",
        text,
        flags=re.UNICODE,
    )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Unicode-friendly normalization.
    """

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# QUERY INTENT
# ============================================================

def detect_query_intent(
    query: str,
) -> str:
    """
    Returns:

        punishment
        definition
        general
    """

    normalized = _normalize_text(
        query
    )

    words = set(
        _tokenize(normalized)
    )

    if words.intersection(
        PUNISHMENT_WORDS
    ):
        return "punishment"

    for phrase in DEFINITION_PHRASES:

        if phrase in normalized:
            return "definition"

    return "general"


# ============================================================
# QUERY LEGAL TERM
# ============================================================

def extract_query_legal_term(
    query: str,
) -> str:
    """
    Detect the main legal offence.

    Longest matching term wins.
    """

    normalized = _normalize_text(
        query
    )

    for term in sorted(
        LEGAL_TERMS,
        key=len,
        reverse=True,
    ):

        if term.lower() in normalized:

            return term.lower()

    return ""


# ============================================================
# BM25 CACHE
# ============================================================

_bm25_cache: Dict[str, Any] = {
    "bm25": None,
    "documents": None,
    "metadatas": None,
    "doc_count": -1,
}


# ============================================================
# GET BM25 INDEX
# ============================================================

def _get_bm25_index(
) -> Tuple[
    Optional[BM25Okapi],
    List[str],
    List[dict],
]:

    documents, metadatas = (
        get_all_documents()
    )

    document_count = len(
        documents
    )

    if (
        _bm25_cache["doc_count"]
        != document_count
    ):

        tokenized_documents = [
            _tokenize(document)
            for document in documents
        ]

        if tokenized_documents:

            _bm25_cache[
                "bm25"
            ] = BM25Okapi(
                tokenized_documents
            )

        else:

            _bm25_cache[
                "bm25"
            ] = None

        _bm25_cache[
            "documents"
        ] = documents

        _bm25_cache[
            "metadatas"
        ] = metadatas

        _bm25_cache[
            "doc_count"
        ] = document_count

    return (
        _bm25_cache["bm25"],
        _bm25_cache["documents"]
        or [],
        _bm25_cache["metadatas"]
        or [],
    )


# ============================================================
# INVALIDATE CACHE
# ============================================================

def invalidate_bm25_cache() -> None:
    """
    Force BM25 index rebuild.
    """

    _bm25_cache[
        "doc_count"
    ] = -1


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query: str,
    documents: List[str],
    metadatas: List[dict],
    top_k: int = 30,
    bm25_index: Optional[BM25Okapi] = None,
) -> List[Dict[str, Any]]:

    if (
        not query
        or not query.strip()
        or not documents
    ):
        return []

    query_tokens = _tokenize(
        query
    )

    if not query_tokens:
        return []

    bm25 = (
        bm25_index
        if bm25_index is not None
        else BM25Okapi(
            [
                _tokenize(doc)
                for doc in documents
            ]
        )
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for rank, index in enumerate(
        ranked_indexes,
        start=1,
    ):

        results.append(
            {
                "document":
                    documents[index],

                "metadata":
                    metadatas[index],

                "score":
                    float(
                        scores[index]
                    ),

                "bm25_rank":
                    rank,

                "method":
                    "bm25",
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

    if not query or not query.strip():
        return []

    # E5 models work best when queries are marked as query.
    embedding = create_embedding(
        f"query: {query}"
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

        idx = rank - 1

        metadata = (
            metadatas[idx]
            if idx < len(metadatas)
            else {}
        )

        distance = (
            distances[idx]
            if idx < len(distances)
            else 0.0
        )

        semantic_score = (
            1.0
            / (
                1.0
                + float(distance)
            )
        )

        output.append(
            {
                "document":
                    document,

                "metadata":
                    metadata,

                "score":
                    float(
                        semantic_score
                    ),

                "semantic_rank":
                    rank,

                "method":
                    "semantic",
            }
        )

    return output


# ============================================================
# RESULT KEY
# ============================================================

def _result_key(
    result: Dict[str, Any],
) -> str:

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
# METADATA HELPERS
# ============================================================

def _get_section(
    metadata: Dict[str, Any],
) -> str:

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
# SCORE NORMALIZATION
# ============================================================

def _min_max_normalize(
    values: List[float],
) -> List[float]:

    if not values:
        return []

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    if minimum == maximum:

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
# LEGAL FEATURES
# ============================================================

def _calculate_legal_features(
    query: str,
    result: Dict[str, Any],
) -> Dict[str, float]:

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

    normalized_document = (
        _normalize_text(
            document
        )
    )

    normalized_title = (
        _normalize_text(
            _get_section_title(
                metadata
            )
        )
    )

    query_term = (
        extract_query_legal_term(
            query
        )
    )

    intent = (
        detect_query_intent(
            query
        )
    )

    # --------------------------------------------------------
    # Legal term match
    # --------------------------------------------------------

    query_term_match = 0.0

    if query_term:

        if (
            query_term
            in normalized_document
        ):

            query_term_match = 1.0

        elif (
            query_term
            in normalized_title
        ):

            query_term_match = 0.8

    # --------------------------------------------------------
    # Punishment language
    # --------------------------------------------------------

    punishment_matches = sum(
        1
        for pattern
        in PUNISHMENT_PATTERNS
        if pattern
        in normalized_document
    )

    punishment_score = min(
        punishment_matches / 4.0,
        1.0,
    )

    # --------------------------------------------------------
    # Definition language
    # --------------------------------------------------------

    definition_matches = sum(
        1
        for pattern
        in DEFINITION_PATTERNS
        if pattern
        in normalized_document
    )

    definition_score = min(
        definition_matches / 3.0,
        1.0,
    )

    # --------------------------------------------------------
    # Specialized title
    # --------------------------------------------------------

    specialized_penalty = 0.0

    if any(
        term in normalized_title
        for term
        in SPECIALIZED_TITLE_TERMS
    ):

        specialized_penalty = 1.0

    # --------------------------------------------------------
    # Direct offence section
    # --------------------------------------------------------

    general_offence_title = 0.0

    if query_term:

        title = normalized_title

        if title == query_term:

            general_offence_title = 1.0

        elif title.startswith(
            query_term + " "
        ):

            general_offence_title = 0.8

        elif title.startswith(
            query_term + "."
        ):

            general_offence_title = 1.0

        elif title.startswith(
            query_term + "—"
        ):

            general_offence_title = 1.0

    # --------------------------------------------------------
    # Intent relevance
    # --------------------------------------------------------

    punishment_relevance = (
        punishment_score
        if intent == "punishment"
        else 0.0
    )

    definition_relevance = (
        definition_score
        if intent == "definition"
        else 0.0
    )

    return {
        "query_term_match":
            query_term_match,

        "punishment_score":
            punishment_score,

        "definition_score":
            definition_score,

        "specialized_penalty":
            specialized_penalty,

        "general_offence_title":
            general_offence_title,

        "punishment_relevance":
            punishment_relevance,

        "definition_relevance":
            definition_relevance,
    }


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    semantic_k: int = 30,
    bm25_k: int = 30,
    final_k: Optional[int] = None,
) -> List[Dict[str, Any]]:

    if not query or not query.strip():
        return []

    query = query.strip()

    if final_k is None:
        final_k = RAG_TOP_K

    final_k = max(
        1,
        int(final_k),
    )

    print(
        "Running legal retrieval..."
    )

    query_intent = (
        detect_query_intent(
            query
        )
    )

    query_term = (
        extract_query_legal_term(
            query
        )
    )

    print(
        f"Detected intent: "
        f"{query_intent}"
    )

    print(
        f"Detected legal term: "
        f"{query_term or 'none'}"
    )

    # --------------------------------------------------------
    # Semantic
    # --------------------------------------------------------

    print(
        "Running semantic search..."
    )

    semantic_results = semantic_search(
        query,
        top_k=max(
            semantic_k,
            30,
        ),
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "Running BM25 search..."
    )

    bm25_index, documents, metadatas = (
        _get_bm25_index()
    )

    bm25_results = bm25_search(
        query,
        documents,
        metadatas,
        top_k=max(
            bm25_k,
            30,
        ),
        bm25_index=bm25_index,
    )

    # --------------------------------------------------------
    # Candidate fusion
    # --------------------------------------------------------

    fusion_scores: Dict[
        str,
        float
    ] = {}

    result_data: Dict[
        str,
        Dict[str, Any]
    ] = {}

    # ========================================================
    # SEMANTIC RRF
    # ========================================================

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):

        key = _result_key(
            result
        )

        rrf_score = (
            1.0
            / (
                RRF_K
                + rank
            )
        )

        fusion_scores[key] = (
            fusion_scores.get(
                key,
                0.0,
            )
            + rrf_score
        )

        entry = result_data.setdefault(
            key,
            {
                "document":
                    result[
                        "document"
                    ],

                "metadata":
                    result[
                        "metadata"
                    ],

                "methods":
                    set(),
            },
        )

        entry[
            "methods"
        ].add(
            "semantic"
        )

        entry[
            "semantic_score"
        ] = float(
            result[
                "score"
            ]
        )

        entry[
            "semantic_rank"
        ] = rank

    # ========================================================
    # BM25 RRF
    # ========================================================

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        key = _result_key(
            result
        )

        rrf_score = (
            1.0
            / (
                RRF_K
                + rank
            )
        )

        fusion_scores[key] = (
            fusion_scores.get(
                key,
                0.0,
            )
            + rrf_score
        )

        entry = result_data.setdefault(
            key,
            {
                "document":
                    result[
                        "document"
                    ],

                "metadata":
                    result[
                        "metadata"
                    ],

                "methods":
                    set(),
            },
        )

        entry[
            "methods"
        ].add(
            "bm25"
        )

        entry[
            "bm25_score"
        ] = float(
            result[
                "score"
            ]
        )

        entry[
            "bm25_rank"
        ] = rank

    if not result_data:
        return []

    # ========================================================
    # CREATE CANDIDATES
    # ========================================================

    candidates = []

    for key, data in (
        result_data.items()
    ):

        item = dict(
            data
        )

        item[
            "rrf_score"
        ] = fusion_scores[
            key
        ]

        candidates.append(
            item
        )

    # ========================================================
    # NORMALIZE BM25
    # ========================================================

    bm25_values = [
        float(
            item[
                "bm25_score"
            ]
        )
        for item in candidates
        if "bm25_score" in item
    ]

    semantic_values = [
        float(
            item[
                "semantic_score"
            ]
        )
        for item in candidates
        if "semantic_score" in item
    ]

    normalized_bm25 = (
        _min_max_normalize(
            bm25_values
        )
    )

    normalized_semantic = (
        _min_max_normalize(
            semantic_values
        )
    )

    bm25_index_counter = 0
    semantic_index_counter = 0

    for item in candidates:

        if "bm25_score" in item:

            item[
                "bm25_normalized"
            ] = normalized_bm25[
                bm25_index_counter
            ]

            bm25_index_counter += 1

        else:

            item[
                "bm25_normalized"
            ] = 0.0

        if "semantic_score" in item:

            item[
                "semantic_normalized"
            ] = normalized_semantic[
                semantic_index_counter
            ]

            semantic_index_counter += 1

        else:

            item[
                "semantic_normalized"
            ] = 0.0

    # ========================================================
    # LEGAL RERANK
    # ========================================================

    for item in candidates:

        features = (
            _calculate_legal_features(
                query,
                item,
            )
        )

        item.update(
            features
        )

        final_score = (
            float(
                item.get(
                    "rrf_score",
                    0.0,
                )
            )
        )

        final_score += (
            float(
                item.get(
                    "bm25_normalized",
                    0.0,
                )
            )
            * WEIGHT_BM25
        )

        final_score += (
            float(
                item.get(
                    "semantic_normalized",
                    0.0,
                )
            )
            * WEIGHT_SEMANTIC
        )

        final_score += (
            features[
                "query_term_match"
            ]
            * WEIGHT_QUERY_TERM
        )

        final_score += (
            features[
                "general_offence_title"
            ]
            * WEIGHT_DIRECT_SECTION
        )

        # ----------------------------------------------------
        # Punishment
        # ----------------------------------------------------

        if query_intent == "punishment":

            final_score += (
                features[
                    "punishment_relevance"
                ]
                * WEIGHT_PUNISHMENT
            )

            normalized_document = (
                _normalize_text(
                    item[
                        "document"
                    ]
                )
            )

            if (
                "shall be punished"
                in normalized_document
            ):

                final_score += (
                    BONUS_PUNISHMENT_LANGUAGE
                )

            elif (
                "punished with"
                in normalized_document
            ):

                final_score += (
                    BONUS_PUNISHMENT_LANGUAGE
                    / 2.0
                )

        # ----------------------------------------------------
        # Definition
        # ----------------------------------------------------

        if query_intent == "definition":

            final_score += (
                features[
                    "definition_relevance"
                ]
                * WEIGHT_DEFINITION
            )

            normalized_document = (
                _normalize_text(
                    item[
                        "document"
                    ]
                )
            )

            if (
                "means"
                in normalized_document
                or
                "is said to"
                in normalized_document
            ):

                final_score += (
                    BONUS_DEFINITION_LANGUAGE
                )

        # ----------------------------------------------------
        # Specialized section penalty
        # ----------------------------------------------------

        if query_term:

            final_score -= (
                features[
                    "specialized_penalty"
                ]
                * PENALTY_SPECIALIZED_SECTION
            )

        item[
            "legal_rerank_score"
        ] = float(
            final_score
        )

        item[
            "hybrid_score"
        ] = float(
            final_score
        )

        item[
            "query_intent"
        ] = query_intent

        item[
            "query_legal_term"
        ] = query_term

        item[
            "method"
        ] = "+".join(
            sorted(
                item.get(
                    "methods",
                    set(),
                )
            )
        )

    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda item:
            item.get(
                "legal_rerank_score",
                0.0,
            ),
        reverse=True,
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_results = []

    # Avoid duplicate sections when several chunks
    # of the same section rank together.

    seen_sections = set()

    # First pass: strongest result from each section
    # until top-k is reached.

    for item in candidates:

        metadata = item.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "",
        )

        section = _get_section(
            metadata
        )

        section_key = (
            f"{source}:"
            f"{section}"
        )

        if section_key in seen_sections:
            continue

        seen_sections.add(
            section_key
        )

        final_results.append(
            item
        )

        if len(final_results) >= final_k:
            break

    return final_results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    results: List[Dict[str, Any]],
) -> None:

    print()
    print(
        "=" * 70
    )

    print(
        "TOP RETRIEVED RESULTS"
    )

    print(
        "=" * 70
    )

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
            "Method:",
            result.get(
                "method",
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
            "Legal score:",
            round(
                float(
                    result.get(
                        "legal_rerank_score",
                        0.0,
                    )
                ),
                6,
            ),
        )

        print(
            "Intent:",
            result.get(
                "query_intent",
                "",
            ),
        )

        print(
            "Legal term:",
            result.get(
                "query_legal_term",
                "",
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
        "JanNyaya AI - Multilingual Retriever"
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

        results = hybrid_search(
            question,
            semantic_k=30,
            bm25_k=30,
            final_k=RAG_TOP_K,
        )

        print_results(
            results
        )