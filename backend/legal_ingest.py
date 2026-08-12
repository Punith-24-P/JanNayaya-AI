from pathlib import Path

from backend.pdf_service import extract_text_from_pdf
from backend.text_cleaner import clean_text
from backend.chunker import chunk_text
from backend.embedding_service import create_embeddings
from backend.vector_store import add_chunks


# =========================================================
# INGEST LEGAL DOCUMENT
# =========================================================

def ingest_legal_document(
    file_path: str,
    document_type: str,
    title: str,
    year: int | None = None,
    authority: str = "Government of India",
    act_name: str | None = None,
) -> dict:
    """
    Ingest a legal PDF into the JanNyaya AI knowledge base.

    Supported document types can include:

        Act
        Judgment
        Legal_Aid
        Report
        Notification
        Regulation
        Other
    """

    path = Path(file_path)

    # -----------------------------------------------------
    # 1. Validate file
    # -----------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF documents are currently supported."
        )

    if not document_type.strip():
        raise ValueError(
            "document_type cannot be empty."
        )

    if not title.strip():
        raise ValueError(
            "title cannot be empty."
        )

    print("=" * 60)
    print("JAN NYAYA AI - LEGAL DOCUMENT INGESTION")
    print("=" * 60)

    print(f"\nFile           : {path.name}")
    print(f"Document type  : {document_type}")
    print(f"Title          : {title}")
    print(f"Year           : {year}")
    print(f"Authority      : {authority}")

    if act_name:
        print(f"Act name       : {act_name}")

    # -----------------------------------------------------
    # 2. Extract PDF text
    # -----------------------------------------------------

    print("\n[1/5] Extracting PDF text...")

    text = extract_text_from_pdf(
        str(path)
    )

    print(
        f"Extracted characters: {len(text):,}"
    )

    if not text.strip():
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    # -----------------------------------------------------
    # 3. Clean text
    # -----------------------------------------------------

    print("\n[2/5] Cleaning text...")

    cleaned_text = clean_text(
        text
    )

    print(
        f"Cleaned characters: {len(cleaned_text):,}"
    )

    if not cleaned_text.strip():
        raise ValueError(
            "Document became empty after cleaning."
        )

    # -----------------------------------------------------
    # 4. Create chunks
    # -----------------------------------------------------

    print("\n[3/5] Creating legal chunks...")

    chunks = chunk_text(
        cleaned_text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(
        f"Created chunks: {len(chunks):,}"
    )

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    # -----------------------------------------------------
    # 5. Create embeddings
    # -----------------------------------------------------

    print("\n[4/5] Creating embeddings...")

    embeddings = create_embeddings(
        chunks
    )

    print(
        f"Created embeddings: {len(embeddings):,}"
    )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    # -----------------------------------------------------
    # 6. Create document ID
    # -----------------------------------------------------

    document_id = (
        f"{document_type.lower()}_"
        f"{path.stem}"
    )

    # -----------------------------------------------------
    # 7. Create metadata
    # -----------------------------------------------------

    metadatas = []

    for index, chunk in enumerate(chunks):

        metadata = {
            "document_id": document_id,
            "document_type": document_type,
            "title": title,
            "source": path.name,
            "authority": authority,
            "chunk_index": index,
        }

        # ---------------------------------------------
        # Optional fields
        # ---------------------------------------------

        if year is not None:
            metadata["year"] = str(year)

        if act_name:
            metadata["act_name"] = act_name

        metadatas.append(
            metadata
        )

    # -----------------------------------------------------
    # 8. Store in ChromaDB
    # -----------------------------------------------------

    print("\n[5/5] Storing chunks in ChromaDB...")

    stored_count = add_chunks(
        chunks=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(
        f"Stored chunks: {stored_count:,}"
    )

    # -----------------------------------------------------
    # 9. Final result
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("LEGAL DOCUMENT INGESTION COMPLETED")
    print("=" * 60)

    print(f"Document ID    : {document_id}")
    print(f"Title          : {title}")
    print(f"Type           : {document_type}")
    print(f"Source         : {path.name}")
    print(f"Chunks         : {stored_count:,}")
    print(f"Characters     : {len(cleaned_text):,}")

    print("=" * 60)

    return {
        "document_id": document_id,
        "title": title,
        "document_type": document_type,
        "source": path.name,
        "authority": authority,
        "year": year,
        "act_name": act_name,
        "chunks": stored_count,
        "characters": len(cleaned_text),
    }


# =========================================================
# BACKWARD-COMPATIBLE ACT INGESTION
# =========================================================

def ingest_legal_act(
    file_path: str,
    act_name: str,
    year: int,
) -> dict:
    """
    Backward-compatible wrapper for ingesting an Act.
    """

    return ingest_legal_document(
        file_path=file_path,
        document_type="Act",
        title=act_name,
        year=year,
        authority="Government of India",
        act_name=act_name,
    )