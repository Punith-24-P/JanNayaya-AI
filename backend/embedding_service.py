from typing import List

from sentence_transformers import SentenceTransformer


# ============================================================
# GLOBAL MODEL
# ============================================================

_model = None


# ============================================================
# MODEL NAME
# ============================================================

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def get_embedding_model() -> SentenceTransformer:
    """
    Load the multilingual SentenceTransformer model.

    The model is loaded only once and reused for all
    subsequent embedding requests.
    """

    global _model

    if _model is None:

        print("=" * 60)
        print("Loading multilingual embedding model...")
        print(f"Model: {EMBEDDING_MODEL_NAME}")

        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print(
            "Multilingual embedding model loaded successfully."
        )

        print("=" * 60)

    return _model


# ============================================================
# PREPARE PASSAGE TEXT
# ============================================================

def _prepare_passage(text: str) -> str:
    """
    Prepare a legal document chunk for E5 passage embedding.
    """

    if text is None:
        text = ""

    text = str(text).strip()

    if not text:
        return "passage:"

    return f"passage: {text}"


# ============================================================
# PREPARE QUERY TEXT
# ============================================================

def _prepare_query(text: str) -> str:
    """
    Prepare a user question for E5 query embedding.
    """

    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    return f"query: {text}"


# ============================================================
# CREATE PASSAGE EMBEDDINGS
# ============================================================

def create_embeddings(
    texts: List[str]
) -> List[List[float]]:
    """
    Create multilingual passage embeddings.

    These embeddings are intended for stored legal documents,
    sections, judgments, reports, and other knowledge-base text.
    """

    if not texts:
        return []

    prepared_texts = [
        _prepare_passage(text)
        for text in texts
    ]

    model = get_embedding_model()

    embeddings = model.encode(
        prepared_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


# ============================================================
# CREATE SINGLE QUERY EMBEDDING
# ============================================================

def create_embedding(
    text: str
) -> List[float]:
    """
    Create a multilingual query embedding.

    This is used for user questions such as English,
    Hindi, Kannada, etc.
    """

    prepared_text = _prepare_query(text)

    if not prepared_text:
        return []

    model = get_embedding_model()

    embedding = model.encode(
        [prepared_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    if embedding is None or len(embedding) == 0:
        return []

    return embedding[0].tolist()


def test_embedding_model() -> bool:
    """
    Quick sanity check that embedding model is loaded and functional.
    """
    try:
        emb = create_embedding("test query")
        return len(emb) > 0
    except Exception:
        return False


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("Testing multilingual embedding service...")

    test_texts = [
        "What is the punishment for theft?",
        "चोरी की सजा क्या है?",
        "ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ ಏನು?",
    ]

    for text in test_texts:

        embedding = create_embedding(text)

        print()
        print("Question:", text)
        print("Embedding dimensions:", len(embedding))

        if embedding:
            print(
                "First 10 values:",
                embedding[:10]
            )

    print()
    print(
        "Multilingual embedding test completed."
    )