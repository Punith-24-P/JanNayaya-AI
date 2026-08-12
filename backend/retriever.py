from backend.embedding_service import create_embeddings
from backend.vector_store import collection

from rank_bm25 import BM25Okapi
import re


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for lexical/BM25 search.
    """

    if not text:
        return ""

    text = text.lower()

    # Keep letters, numbers, rupee symbol and spaces
    text = re.sub(r"[^a-z0-9₹\s]", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> list[str]:
    """
    Convert text into tokens for BM25.
    """

    return normalize_text(text).split()


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

def get_all_documents():
    """
    Load all legal documents and metadata from ChromaDB.
    """

    data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    return documents, metadatas


# ============================================================
# QUERY INTENT
# ============================================================

def detect_query_intent(query: str) -> str:
    """
    Detect the user's legal question intent.

    Returns:

        punishment
        definition
        general
    """

    normalized = normalize_text(query)

    words = set(normalized.split())

    # --------------------------------------------------------
    # Punishment intent
    # --------------------------------------------------------

    punishment_words = {
        "punishment",
        "punished",
        "penalty",
        "sentence",
        "imprisonment",
        "fine",
        "punishable",
    }

    if words.intersection(punishment_words):
        return "punishment"

    # --------------------------------------------------------
    # Definition intent
    # --------------------------------------------------------

    definition_phrases = [
        "what is",
        "what are",
        "what does",
        "meaning of",
        "define",
        "definition of",
        "explain",
    ]

    for phrase in definition_phrases:

        if phrase in normalized:
            return "definition"

    return "general"


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    top_k: int = 50
) -> list[dict]:
    """
    Search ChromaDB using embedding similarity.
    """

    if collection.count() == 0:
        return []

    query_embedding = create_embeddings([query])[0]

    top_k = min(
        top_k,
        collection.count()
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    results_list = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        semantic_score = (
            1.0 / (1.0 + float(distance))
        )

        results_list.append({
            "text": document,
            "metadata": metadata,
            "distance": float(distance),
            "semantic_score": float(
                semantic_score
            )
        })

    return results_list


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query: str,
    top_k: int = 50
) -> list[dict]:
    """
    Search the entire legal knowledge base using BM25.
    """

    documents, metadatas = get_all_documents()

    if not documents:
        return []

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indexes[:top_k]:

        results.append({
            "text": documents[index],
            "metadata": metadatas[index],
            "bm25_score": float(
                scores[index]
            )
        })

    return results


# ============================================================
# QUERY TERM EXTRACTION
# ============================================================

def extract_query_terms(
    query: str
) -> list[str]:
    """
    Extract meaningful terms from the user's question.
    """

    text = normalize_text(query)

    stop_words = {
        "what",
        "is",
        "the",
        "for",
        "under",
        "a",
        "an",
        "of",
        "and",
        "to",
        "in",
        "on",
        "with",
        "this",
        "that",
        "are",
        "was",
        "were",
        "be",
        "can",
        "may",
        "does",
        "do",
        "which",
        "who",
        "how",
        "tell",
        "me",
        "about",
        "explain",
        "give",
        "please",
        "punishment",
        "punished",
        "penalty",
        "sentence",
        "imprisonment",
        "fine",
    }

    words = text.split()

    return [
        word
        for word in words
        if word not in stop_words
    ]


# ============================================================
# QUERY SPECIFICITY
# ============================================================

def calculate_query_specificity(
    query: str,
    text: str
) -> float:
    """
    Measures how many important query terms occur
    in the retrieved document.
    """

    query_terms = extract_query_terms(
        query
    )

    if not query_terms:
        return 0.0

    normalized_text = normalize_text(
        text
    )

    matched_terms = 0

    for term in query_terms:

        if term in normalized_text:
            matched_terms += 1

    return min(
        matched_terms / len(query_terms),
        1.0
    )


# ============================================================
# LEGAL EVIDENCE SCORE
# ============================================================

def calculate_legal_evidence_score(
    query: str,
    text: str
) -> float:
    """
    Detect whether a chunk looks like an actual
    legal provision instead of general/background text.
    """

    normalized = normalize_text(
        text
    )

    patterns = [
        r"\bwhoever\b",
        r"\bshall be punished\b",
        r"\bpunished with\b",
        r"\bimprisonment\b",
        r"\bfine\b",
        r"\bprovided that\b",
        r"\bsection\b",
        r"\bmeans\b",
        r"\bis said to\b",
    ]

    matches = 0

    for pattern in patterns:

        if re.search(
            pattern,
            normalized
        ):
            matches += 1

    if matches >= 5:
        return 0.9

    if matches >= 4:
        return 0.8

    if matches >= 3:
        return 0.6

    if matches >= 2:
        return 0.4

    if matches >= 1:
        return 0.2

    return 0.0


# ============================================================
# SPECIFIC LEGAL PROVISION SCORE
# ============================================================

def calculate_provision_score(
    query: str,
    text: str
) -> float:
    """
    Detect whether the chunk contains the legal
    provision most relevant to the user's question.

    Definition questions prefer definitional language.

    Punishment questions prefer punishment language.
    """

    normalized = normalize_text(
        text
    )

    query_normalized = normalize_text(
        query
    )

    intent = detect_query_intent(
        query
    )

    score = 0.0

    # ========================================================
    # PUNISHMENT INTENT
    # ========================================================

    if intent == "punishment":

        punishment_patterns = [
            "shall be punished",
            "punished with",
            "liable to fine",
            "imprisonment",
            "community service",
        ]

        punishment_matches = sum(
            1
            for pattern in punishment_patterns
            if pattern in normalized
        )

        if punishment_matches >= 4:
            score = max(score, 1.0)

        elif punishment_matches >= 3:
            score = max(score, 0.9)

        elif punishment_matches >= 2:
            score = max(score, 0.75)

        elif punishment_matches >= 1:
            score = max(score, 0.5)

    # ========================================================
    # DEFINITION INTENT
    # ========================================================

    elif intent == "definition":

        definition_patterns = [
            "means",
            "is said to",
            "definition",
            "shall be deemed",
        ]

        definition_matches = sum(
            1
            for pattern in definition_patterns
            if pattern in normalized
        )

        if definition_matches >= 3:
            score = max(score, 1.0)

        elif definition_matches >= 2:
            score = max(score, 0.8)

        elif definition_matches >= 1:
            score = max(score, 0.6)

        # ----------------------------------------------------
        # Strong penalty for pure punishment provisions.
        #
        # This prevents:
        #
        # "What is murder?"
        #
        # from selecting:
        #
        # "Punishment for murder"
        # ----------------------------------------------------

        if (
            "shall be punished" in normalized
            and "means" not in normalized
            and "is said to" not in normalized
        ):
            score -= 0.45

        score = max(
            score,
            0.0
        )

    # ========================================================
    # GENERAL INTENT
    # ========================================================

    else:

        if "means" in normalized:
            score = max(
                score,
                0.6
            )

        if "shall be punished" in normalized:
            score = max(
                score,
                0.5
            )

    # ========================================================
    # LEGAL TERM MATCH
    # ========================================================

    legal_terms = [
        "theft",
        "snatching",
        "murder",
        "rape",
        "cheating",
        "robbery",
        "assault",
        "defamation",
        "criminal intimidation",
        "criminal breach of trust",
        "extortion",
        "kidnapping",
        "abduction",
    ]

    for term in legal_terms:

        if (
            term in query_normalized
            and term in normalized
        ):
            score += 0.15
            break

    return min(
        max(score, 0.0),
        1.0
    )


# ============================================================
# BM25 NORMALIZATION
# ============================================================

def normalize_bm25_scores(
    combined: dict
) -> None:
    """
    Normalize BM25 scores between 0 and 1.
    """

    max_bm25 = max(
        [
            item.get(
                "bm25_score",
                0.0
            )
            for item in combined.values()
        ],
        default=0.0
    )

    for item in combined.values():

        if max_bm25 > 0:

            item["bm25_normalized"] = (
                item.get(
                    "bm25_score",
                    0.0
                )
                / max_bm25
            )

        else:

            item["bm25_normalized"] = 0.0


# ============================================================
# HYBRID LEGAL RETRIEVAL
# ============================================================

def search_documents(
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Hybrid legal retrieval.

    Uses:

    1. Semantic search
    2. BM25 lexical search
    3. Query specificity
    4. Legal evidence detection
    5. Query intent
    6. Provision relevance
    7. Punishment relevance
    """

    if not query or not query.strip():
        return []

    # ========================================================
    # STEP 1: DETECT INTENT
    # ========================================================

    query_intent = detect_query_intent(
        query
    )

    # ========================================================
    # STEP 2: SEMANTIC SEARCH
    # ========================================================

    semantic_results = semantic_search(
        query,
        top_k=50
    )

    # ========================================================
    # STEP 3: BM25 SEARCH
    # ========================================================

    bm25_results = bm25_search(
        query,
        top_k=50
    )

    # ========================================================
    # STEP 4: COMBINE RESULTS
    # ========================================================

    combined = {}

    # --------------------------------------------------------
    # Semantic results
    # --------------------------------------------------------

    for rank, result in enumerate(
        semantic_results
    ):

        metadata = (
            result["metadata"] or {}
        )

        key = (
            metadata.get("source"),
            metadata.get("chunk_index")
        )

        combined[key] = {
            "text": result["text"],
            "metadata": metadata,
            "distance": result["distance"],
            "semantic_score": result[
                "semantic_score"
            ],
            "bm25_score": 0.0,
            "semantic_rank": rank + 1,
            "bm25_rank": None,
        }

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results
    ):

        metadata = (
            result["metadata"] or {}
        )

        key = (
            metadata.get("source"),
            metadata.get("chunk_index")
        )

        if key not in combined:

            combined[key] = {
                "text": result["text"],
                "metadata": metadata,
                "distance": 999.0,
                "semantic_score": 0.0,
                "bm25_score": result[
                    "bm25_score"
                ],
                "semantic_rank": None,
                "bm25_rank": rank + 1,
            }

        else:

            combined[key][
                "bm25_score"
            ] = result[
                "bm25_score"
            ]

            combined[key][
                "bm25_rank"
            ] = rank + 1

    # ========================================================
    # STEP 5: NORMALIZE BM25
    # ========================================================

    normalize_bm25_scores(
        combined
    )

    # ========================================================
    # STEP 6: PUNISHMENT INTENT
    # ========================================================

    punishment_query = (
        query_intent == "punishment"
    )

    # ========================================================
    # STEP 7: CALCULATE FEATURES
    # ========================================================

    for result in combined.values():

        text = result["text"]

        # ----------------------------------------------------
        # Query specificity
        # ----------------------------------------------------

        specificity = (
            calculate_query_specificity(
                query,
                text
            )
        )

        # ----------------------------------------------------
        # Legal evidence
        # ----------------------------------------------------

        evidence = (
            calculate_legal_evidence_score(
                query,
                text
            )
        )

        # ----------------------------------------------------
        # Provision score
        # ----------------------------------------------------

        provision = (
            calculate_provision_score(
                query,
                text
            )
        )

        result[
            "query_specificity_score"
        ] = specificity

        result[
            "legal_evidence_score"
        ] = evidence

        result[
            "provision_score"
        ] = provision

        result[
            "query_intent"
        ] = query_intent

        # ====================================================
        # PROVISION BOOST
        # ====================================================

        provision_boost = 0.0

        if provision >= 1.0:
            provision_boost = 1.25

        elif provision >= 0.9:
            provision_boost = 0.90

        elif provision >= 0.75:
            provision_boost = 0.50

        elif provision >= 0.50:
            provision_boost = 0.20

        result[
            "provision_boost"
        ] = provision_boost

        # ====================================================
        # PUNISHMENT BOOST
        # ====================================================

        punishment_boost = 0.0

        if punishment_query:

            normalized_text = (
                normalize_text(text)
            )

            punishment_matches = 0

            if (
                "shall be punished"
                in normalized_text
            ):
                punishment_matches += 1

            if (
                "punished with"
                in normalized_text
            ):
                punishment_matches += 1

            if (
                "imprisonment"
                in normalized_text
            ):
                punishment_matches += 1

            if (
                "fine"
                in normalized_text
            ):
                punishment_matches += 1

            if (
                "community service"
                in normalized_text
            ):
                punishment_matches += 1

            if punishment_matches >= 4:
                punishment_boost = 0.45

            elif punishment_matches >= 3:
                punishment_boost = 0.35

            elif punishment_matches >= 2:
                punishment_boost = 0.20

            elif punishment_matches >= 1:
                punishment_boost = 0.08

        result[
            "punishment_boost"
        ] = punishment_boost

    # ========================================================
    # STEP 8: FINAL SCORE
    # ========================================================

    for result in combined.values():

        semantic = result[
            "semantic_score"
        ]

        bm25 = result[
            "bm25_normalized"
        ]

        specificity = result[
            "query_specificity_score"
        ]

        evidence = result[
            "legal_evidence_score"
        ]

        provision = result[
            "provision_score"
        ]

        provision_boost = result[
            "provision_boost"
        ]

        punishment_boost = result[
            "punishment_boost"
        ]

        # ----------------------------------------------------
        # Base retrieval score
        # ----------------------------------------------------

        base_score = (
            semantic * 0.25
            +
            bm25 * 0.20
        )

        # ----------------------------------------------------
        # Legal relevance
        # ----------------------------------------------------

        legal_score = (
            specificity * 0.15
            +
            evidence * 0.10
            +
            provision * 0.20
        )

        # ----------------------------------------------------
        # Definition intent protection
        # ----------------------------------------------------

        definition_penalty = 0.0

        if query_intent == "definition":

            normalized_text = (
                normalize_text(text)
            )

            # A chunk that is clearly a punishment
            # provision should be less preferred for
            # a "What is X?" question.

            if (
                "punishment for"
                in normalized_text
                and "shall be punished"
                in normalized_text
            ):
                definition_penalty = 0.30

            elif (
                "shall be punished"
                in normalized_text
                and "means"
                not in normalized_text
            ):
                definition_penalty = 0.15

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (
            base_score
            + legal_score
            + provision_boost
            + punishment_boost
            - definition_penalty
        )

        result[
            "score"
        ] = final_score

    # ========================================================
    # STEP 9: SORT
    # ========================================================

    ranked_results = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # ========================================================
    # STEP 10: RETURN TOP RESULTS
    # ========================================================

    return ranked_results[:top_k]