import os
import gc
from typing import List, Optional

# Configure CPU threads dynamically (allow 4 threads locally for 10x faster query embedding, 1 on Render)
if os.getenv("RENDER"):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = "1"
else:
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["TORCH_NUM_THREADS"] = "4"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"

# ============================================================
# GLOBAL MODEL
# ============================================================

_model = None
_torch_loaded = False


# ============================================================
# MODEL NAME
# ============================================================

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    """
    Load the multilingual SentenceTransformer model in optimized mode.
    The model is loaded only once and reused for all subsequent requests.
    If memory is constrained (Render 512MB), gracefully returns None so
    BM25 and statutory retrieval take over with 0MB memory overhead.
    """
    global _model, _torch_loaded

    # Check if low-memory mode is explicitly requested
    if os.getenv("LOW_MEMORY_MODE", "").lower() in ["1", "true", "yes"]:
        return None

    if _model is None:
        try:
            print("=" * 60)
            print("Loading lightweight multilingual embedding model...")
            print(f"Model: {EMBEDDING_MODEL_NAME}")
            import torch
            torch.set_grad_enabled(False)
            num_threads = 1 if os.getenv("RENDER") else 4
            torch.set_num_threads(num_threads)

            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device="cpu",
            )
            _model.eval()
            _torch_loaded = True
            print(f"Multilingual embedding model loaded successfully with {num_threads} CPU threads.")
            print("=" * 60)
        except Exception as e:
            print(f"Notice: Neural embedding fallback to high-speed BM25 statutory engine: {e}")
            _model = None

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
    if model is None:
        return []

    try:
        import torch
        with torch.inference_mode():
            embeddings = model.encode(
                prepared_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings.tolist()
    except Exception as e:
        print(f"Embedding generation error: {e}")
        return []


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
    if model is None:
        return []

    try:
        import torch
        with torch.inference_mode():
            embedding = model.encode(
                [prepared_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        if embedding is None or len(embedding) == 0:
            return []

        return embedding[0].tolist()
    except Exception as e:
        print(f"Query embedding generation error: {e}")
        return []


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