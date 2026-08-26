import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from backend.embedding_service import (
    create_embedding
)

from backend.vector_store import (
    get_all_documents,
    search_by_embedding
)


# ============================================================
# TOKENIZER
# ============================================================

def _tokenize(
    text: str
) -> List[str]:
    """
    Simple tokenizer for BM25.
    """

    if not text:
        return []

    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9₹]+\b",
        text
    )

    return tokens


# ============================================================
# NORMALIZE SCORE
# ============================================================

def _normalize_scores(
    scores: List[float]
) -> List[float]:

    if not scores:
        return []

    maximum = max(
        scores
    )

    minimum = min(
        scores
    )

    if maximum == minimum:

        return [
            1.0
            for _ in scores
        ]

    return [
        (
            score - minimum
        )
        / (
            maximum - minimum
        )
        for score in scores
    ]


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query: str,
    documents: List[str],
    metadatas: List[dict],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    BM25 lexical search.
    """

    if not query.strip():
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

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    ranked_indexes = ranked_indexes[
        :top_k
    ]

    results = []

    for index in ranked_indexes:

        results.append(
            {
                "document":
                    documents[index],

                "metadata":
                    metadatas[index],

                "score":
                    float(scores[index]),

                "method":
                    "bm25"
            }
        )

    return results


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Semantic vector search using SentenceTransformer
    + ChromaDB.
    """

    embedding = create_embedding(
        query
    )

    if not embedding:
        return []

    results = search_by_embedding(
        embedding,
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

    output = []

    for i, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        distance = (
            distances[i]
            if i < len(distances)
            else 0
        )

        # Chroma distance:
        # lower = more similar
        #
        # Convert into a similarity-like score.

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
                    semantic_score,

                "method":
                    "semantic"
            }
        )

    return output


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    semantic_k: int = 10,
    bm25_k: int = 10,
    final_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:

        Semantic search
              +
            BM25
              ↓
        Reciprocal Rank Fusion
              ↓
        Deduplication
              ↓
        Final results
    """

    print(
        "Running semantic search..."
    )

    semantic_results = semantic_search(
        query,
        semantic_k
    )

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
        bm25_k
    )

    # ========================================================
    # RRF
    # ========================================================

    fusion_scores = {}

    result_data = {}

    # --------------------------------------------------------
    # Semantic rankings
    # --------------------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        document = result[
            "document"
        ]

        key = _result_key(
            result
        )

        score = 1.0 / (
            60 + rank
        )

        fusion_scores[key] = (
            fusion_scores.get(
                key,
                0.0
            )
            + score
        )

        result_data[key] = result

    # --------------------------------------------------------
    # BM25 rankings
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        key = _result_key(
            result
        )

        score = 1.0 / (
            60 + rank
        )

        fusion_scores[key] = (
            fusion_scores.get(
                key,
                0.0
            )
            + score
        )

        if key not in result_data:

            result_data[key] = result

    # ========================================================
    # SORT
    # ========================================================

    ranked = sorted(
        fusion_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    final_results = []

    for key, fusion_score in ranked:

        result = dict(
            result_data[key]
        )

        result[
            "hybrid_score"
        ] = fusion_score

        final_results.append(
            result
        )

        if len(final_results) >= final_k:
            break

    return final_results


# ============================================================
# RESULT KEY
# ============================================================

def _result_key(
    result: Dict[str, Any]
) -> str:
    """
    Generate a stable key for deduplication.
    """

    metadata = result.get(
        "metadata",
        {}
    )

    document_id = metadata.get(
        "document_id",
        ""
    )

    chunk_index = metadata.get(
        "chunk_index",
        ""
    )

    document = result.get(
        "document",
        ""
    )

    if document_id:

        return (
            f"{document_id}:"
            f"{chunk_index}"
        )

    return document[:500]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    results: List[Dict[str, Any]]
) -> None:
    """
    Print retrieved results for debugging.
    """

    print(
        "\n========== TOP RESULTS ==========\n"
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        metadata = result.get(
            "metadata",
            {}
        )

        print(
            f"[{index}]"
        )

        print(
            "Section:",
            metadata.get(
                "section_number",
                metadata.get(
                    "section",
                    "None"
                )
            )
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "Unknown"
            )
        )

        print(
            "Chunk:",
            metadata.get(
                "chunk_index",
                "None"
            )
        )

        print(
            "Score:",
            round(
                result.get(
                    "hybrid_score",
                    result.get(
                        "score",
                        0
                    )
                ),
                4
            )
        )

        print(
            "Text:"
        )

        print(
            result.get(
                "document",
                ""
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
        "# JanNyaya AI - Retriever Test"
    )

    question = input(
        "\nEnter legal question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

        raise SystemExit(1)

    results = hybrid_search(
        question,
        semantic_k=10,
        bm25_k=10,
        final_k=5
    )

    print_results(
        results
    )