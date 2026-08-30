import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb


# ============================================================
# CHROMADB CONFIGURATION
# ============================================================

CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", "chroma_db"))

COLLECTION_NAME = "jan_nyaya_documents"


# ============================================================
# PERSISTENT CHROMADB CLIENT
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# ============================================================
# LEGAL DOCUMENT COLLECTION
# ============================================================

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# EMPTY SEARCH RESULT
# ============================================================

def _empty_result() -> dict:
    """
    Return a consistent empty ChromaDB result.
    """

    return {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


# ============================================================
# CREATE DETERMINISTIC CHUNK ID
# ============================================================

def _create_chunk_id(
    document_id: str,
    chunk_index: int
) -> str:
    """
    Create a deterministic ID for a document chunk.
    """

    unique_string = (
        f"{document_id}_{chunk_index}"
    )

    return hashlib.sha256(
        unique_string.encode("utf-8")
    ).hexdigest()


# ============================================================
# ADD / UPSERT CHUNKS
# ============================================================

def add_chunks(
    chunks: List[str],
    metadatas: Optional[List[dict]] = None,
    embeddings: Optional[List[List[float]]] = None,
    source: str = "unknown",
) -> int:
    """
    Add or update legal document chunks in ChromaDB.

    Uses upsert so re-ingesting the same document does not
    create duplicate records or fail because an ID already
    exists.
    """

    if not chunks:
        return 0

    # --------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------

    if metadatas is None:

        metadatas = [
            {
                "source": source,
                "chunk_index": index,
                "document_id": source,
            }
            for index in range(len(chunks))
        ]

    if len(metadatas) != len(chunks):

        raise ValueError(
            "Number of metadata entries must match "
            "number of chunks."
        )

    # --------------------------------------------------------
    # Validate embeddings
    # --------------------------------------------------------

    if embeddings is not None:

        if len(embeddings) != len(chunks):

            raise ValueError(
                "Number of embeddings must match "
                "number of chunks."
            )

    # --------------------------------------------------------
    # Create IDs
    # --------------------------------------------------------

    ids = []

    for index, metadata in enumerate(metadatas):

        document_id = metadata.get(
            "document_id",
            source
        )

        chunk_index = metadata.get(
            "chunk_index",
            index
        )

        chunk_id = _create_chunk_id(
            str(document_id),
            int(chunk_index)
        )

        ids.append(chunk_id)

    # --------------------------------------------------------
    # Clean metadata
    # --------------------------------------------------------

    cleaned_metadatas = []

    for index, metadata in enumerate(metadatas):

        metadata_copy = dict(metadata)

        if "source" not in metadata_copy:
            metadata_copy["source"] = source

        if "chunk_index" not in metadata_copy:
            metadata_copy["chunk_index"] = index

        if "document_id" not in metadata_copy:
            metadata_copy["document_id"] = source

        # Chroma metadata supports primitive values.
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

        cleaned_metadatas.append(
            metadata_copy
        )

    # --------------------------------------------------------
    # Prepare Chroma arguments
    # --------------------------------------------------------

    upsert_arguments = {
        "ids": ids,
        "documents": chunks,
        "metadatas": cleaned_metadatas,
    }

    if embeddings is not None:

        upsert_arguments[
            "embeddings"
        ] = embeddings

    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    collection.upsert(
        **upsert_arguments
    )

    return len(chunks)


# ============================================================
# SEARCH USING CHROMADB'S EMBEDDING FUNCTION
# ============================================================

def search_chunks(
    query: str,
    n_results: int = 5,
) -> dict:
    """
    Basic ChromaDB text search.

    Main hybrid retrieval is handled by retriever.py.
    """

    if not query or not query.strip():

        return _empty_result()

    collection_count = collection.count()

    if collection_count == 0:

        return _empty_result()

    n_results = max(
        1,
        min(
            int(n_results),
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


# ============================================================
# SEARCH USING PRE-COMPUTED EMBEDDING
# ============================================================

def search_by_embedding(
    embedding: List[float],
    n_results: int = 5,
) -> dict:
    """
    Search ChromaDB using a pre-computed embedding.
    """

    if not embedding:

        return _empty_result()

    collection_count = collection.count()

    if collection_count == 0:

        return _empty_result()

    n_results = max(
        1,
        min(
            int(n_results),
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


# ============================================================
# GET ALL DOCUMENTS
# ============================================================

def get_all_documents() -> Tuple[List[str], List[dict]]:
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


# ============================================================
# COLLECTION COUNT
# ============================================================

def get_collection_count() -> int:
    """
    Return the number of chunks currently stored.
    """

    return collection.count()


# ============================================================
# CLEAR COLLECTION
# ============================================================

def clear_collection() -> None:
    """
    Delete every chunk from the current collection.

    WARNING:
    This deletes the complete legal knowledge base.
    """

    global collection

    client.delete_collection(
        name=COLLECTION_NAME
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        "ChromaDB collection cleared successfully."
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

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


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "JanNyaya AI Vector Store"
    )

    print(
        "Collection:",
        COLLECTION_NAME
    )

    print(
        "Stored chunks:",
        get_collection_count()
    )