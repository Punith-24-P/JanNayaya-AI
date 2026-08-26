from typing import List

from sentence_transformers import SentenceTransformer


# ============================================================
# GLOBAL MODEL
# ============================================================

_model = None


# ============================================================
# MODEL NAME
# ============================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def get_embedding_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer embedding model.

    The model is loaded only once and reused for all
    subsequent embedding requests.
    """

    global _model

    if _model is None:

        print("=" * 60)
        print("Loading embedding model...")
        print(f"Model: {EMBEDDING_MODEL_NAME}")

        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print("Embedding model loaded successfully.")
        print("=" * 60)

    return _model


# ============================================================
# CREATE MULTIPLE EMBEDDINGS
# ============================================================

def create_embeddings(
    texts: List[str]
) -> List[List[float]]:
    """
    Create embeddings for a list of text chunks.

    Parameters
    ----------
    texts:
        List of strings.

    Returns
    -------
    List[List[float]]
        Embedding vectors.
    """

    if not texts:
        return []

    # Remove completely empty values while preserving
    # the number/order of usable texts.
    cleaned_texts = []

    for text in texts:

        if text is None:
            cleaned_texts.append("")

        else:
            cleaned_texts.append(
                str(text)
            )

    model = get_embedding_model()

    embeddings = model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


# ============================================================
# CREATE SINGLE EMBEDDING
# ============================================================

def create_embedding(
    text: str
) -> List[float]:
    """
    Create an embedding for a single text string.
    """

    if not text or not text.strip():
        return []

    embeddings = create_embeddings(
        [text]
    )

    if not embeddings:
        return []

    return embeddings[0]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Testing embedding service...")

    test_text = (
        "What is the punishment for theft?"
    )

    embedding = create_embedding(
        test_text
    )

    print(
        "Embedding dimensions:",
        len(embedding)
    )

    print(
        "First 10 values:",
        embedding[:10]
    )