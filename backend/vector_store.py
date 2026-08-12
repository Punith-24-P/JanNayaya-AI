import hashlib
from pathlib import Path

import chromadb


# =========================================================
# CHROMADB CONFIGURATION
# =========================================================

CHROMA_PATH = Path("chroma_db")

COLLECTION_NAME = "jan_nyaya_documents"


# =========================================================
# PERSISTENT CHROMADB CLIENT
# =========================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# =========================================================
# LEGAL DOCUMENT COLLECTION
# =========================================================

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================================================
# ADD CHUNKS
# =========================================================

def add_chunks(
    chunks: list[str],
    metadatas: list[dict] | None = None,
    embeddings: list[list[float]] | None = None,
    source: str = "unknown",
) -> int:
    """
    Add legal document chunks to ChromaDB.

    Each chunk receives a deterministic ID based on:
        source + chunk index + document metadata

    Parameters
    ----------
    chunks:
        Text chunks to store.

    metadatas:
        Metadata corresponding to each chunk.

    embeddings:
        Pre-computed embeddings corresponding to each chunk.

    source:
        Fallback source name.
    """

    if not chunks:
        return 0

    # -----------------------------------------------------
    # Validate metadata
    # -----------------------------------------------------

    if metadatas is None:

        metadatas = [
            {
                "source": source,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

    if len(metadatas) != len(chunks):
        raise ValueError(
            "Number of metadata entries must match "
            "number of chunks."
        )

    # -----------------------------------------------------
    # Validate embeddings
    # -----------------------------------------------------

    if embeddings is not None:

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match "
                "number of chunks."
            )

    # -----------------------------------------------------
    # Create IDs
    # -----------------------------------------------------

    ids = []

    for index, metadata in enumerate(metadatas):

        document_id = metadata.get(
            "document_id",
            source
        )

        unique_string = (
            f"{document_id}_"
            f"{metadata.get('chunk_index', index)}"
        )

        chunk_id = hashlib.sha256(
            unique_string.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    # -----------------------------------------------------
    # Prepare metadata
    # -----------------------------------------------------

    cleaned_metadatas = []

    for index, metadata in enumerate(metadatas):

        metadata_copy = dict(metadata)

        # Chroma metadata values must be primitive values.
        # Convert unsupported values to strings.

        for key, value in list(
            metadata_copy.items()
        ):

            if value is None:
                metadata_copy[key] = ""

            elif not isinstance(
                value,
                (str, int, float, bool)
            ):
                metadata_copy[key] = str(value)

        # Ensure chunk index exists

        if "chunk_index" not in metadata_copy:
            metadata_copy["chunk_index"] = index

        cleaned_metadatas.append(
            metadata_copy
        )

    # -----------------------------------------------------
    # Store in ChromaDB
    # -----------------------------------------------------

    add_arguments = {
        "ids": ids,
        "documents": chunks,
        "metadatas": cleaned_metadatas,
    }

    if embeddings is not None:
        add_arguments["embeddings"] = embeddings

    collection.add(
        **add_arguments
    )

    return len(chunks)


# =========================================================
# SEARCH CHUNKS
# =========================================================

def search_chunks(
    query: str,
    n_results: int = 5,
) -> dict:
    """
    Search the legal document collection.

    This function is mainly a basic ChromaDB search.
    The main hybrid retrieval is handled by retriever.py.
    """

    if not query or not query.strip():

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    collection_count = collection.count()

    if collection_count == 0:

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    # Never request more chunks than available.

    n_results = max(
        1,
        min(
            n_results,
            collection_count
        )
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return results


# =========================================================
# SEARCH USING EMBEDDINGS
# =========================================================

def search_by_embedding(
    embedding: list[float],
    n_results: int = 5,
) -> dict:
    """
    Search ChromaDB using a pre-computed embedding.
    """

    if not embedding:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    collection_count = collection.count()

    if collection_count == 0:

        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    n_results = max(
        1,
        min(
            n_results,
            collection_count
        )
    )

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return results


# =========================================================
# GET ALL DOCUMENTS
# =========================================================

def get_all_documents() -> tuple[list, list]:
    """
    Return all documents and metadata from ChromaDB.
    """

    if collection.count() == 0:
        return [], []

    data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    documents = data.get(
        "documents",
        []
    )

    metadatas = data.get(
        "metadatas",
        []
    )

    return documents, metadatas


# =========================================================
# COLLECTION COUNT
# =========================================================

def get_collection_count() -> int:
    """
    Return the number of chunks currently stored.
    """

    return collection.count()


# =========================================================
# DELETE ALL DOCUMENTS
# =========================================================

def clear_collection() -> None:
    """
    Delete every chunk from the current collection.

    IMPORTANT:
    Use this only when intentionally rebuilding the
    knowledge base.
    """

    global collection

    client.delete_collection(
        name=COLLECTION_NAME
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )


# =========================================================
# DELETE DOCUMENT
# =========================================================

def delete_document(
    document_id: str,
) -> int:
    """
    Delete all chunks belonging to a document.

    Returns the number of chunks deleted.
    """

    if not document_id:
        return 0

    results = collection.get(
        where={
            "document_id": document_id
        },
        include=[
            "metadatas",
        ],
    )

    ids = results.get(
        "ids",
        []
    )

    if not ids:
        return 0

    collection.delete(
        ids=ids
    )

    return len(ids)