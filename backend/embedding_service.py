from sentence_transformers import SentenceTransformer

_model = None


def get_embedding_model():
    global _model

    if _model is None:
        print("Loading embedding model...")

        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully.")

    return _model


def create_embeddings(texts: list[str]):
    if not texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings.tolist()
