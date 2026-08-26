import re
from pathlib import Path
from typing import Optional

from backend.pdf_service import (
    extract_text_from_pdf
)

from backend.text_cleaner import (
    clean_text
)

from backend.chunker import (
    chunk_text
)

from backend.embedding_service import (
    create_embeddings
)

from backend.vector_store import (
    add_chunks
)


# ============================================================
# CREATE DOCUMENT ID
# ============================================================

def _create_document_id(
    document_type: str,
    file_path: Path
) -> str:
    """
    Create a stable document ID.
    """

    safe_type = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        document_type.lower()
    )

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        file_path.stem.lower()
    )

    return (
        f"{safe_type}_{safe_name}"
    )


# ============================================================
# INGEST LEGAL DOCUMENT
# ============================================================

def ingest_legal_document(
    file_path: str,
    document_type: str,
    title: str,
    year: Optional[int] = None,
    authority: str = "Government of India",
    act_name: Optional[str] = None,
) -> dict:
    """
    Complete legal document ingestion pipeline.

    Pipeline:

        PDF
          ↓
        Text extraction
          ↓
        Text cleaning
          ↓
        Chunking
          ↓
        Metadata
          ↓
        Embeddings
          ↓
        ChromaDB
    """

    # ========================================================
    # 1. VALIDATE INPUT
    # ========================================================

    path = Path(
        file_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Provided path is not a file: {file_path}"
        )

    if path.suffix.lower() != ".pdf":

        raise ValueError(
            "Only PDF documents are currently supported."
        )

    if not document_type or not document_type.strip():

        raise ValueError(
            "document_type cannot be empty."
        )

    if not title or not title.strip():

        raise ValueError(
            "title cannot be empty."
        )

    # ========================================================
    # 2. DOCUMENT INFORMATION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "JAN NYAYA AI - LEGAL DOCUMENT INGESTION"
    )

    print("=" * 70)

    print(
        f"\nFile           : {path.name}"
    )

    print(
        f"Full path      : {path}"
    )

    print(
        f"Document type  : {document_type}"
    )

    print(
        f"Title          : {title}"
    )

    print(
        f"Year           : {year}"
    )

    print(
        f"Authority      : {authority}"
    )

    if act_name:

        print(
            f"Act name       : {act_name}"
        )

    # ========================================================
    # 3. EXTRACT PDF TEXT
    # ========================================================

    print(
        "\n[1/5] Extracting PDF text..."
    )

    text = extract_text_from_pdf(
        str(path)
    )

    if text is None:

        raise ValueError(
            "PDF text extraction returned None."
        )

    print(
        f"Extracted characters: "
        f"{len(text):,}"
    )

    if not text.strip():

        raise ValueError(
            "No text could be extracted from the PDF."
        )

    # ========================================================
    # 4. CLEAN TEXT
    # ========================================================

    print(
        "\n[2/5] Cleaning text..."
    )

    cleaned_text = clean_text(
        text
    )

    if not cleaned_text.strip():

        raise ValueError(
            "Document became empty after cleaning."
        )

    print(
        f"Cleaned characters: "
        f"{len(cleaned_text):,}"
    )

    # ========================================================
    # 5. CREATE CHUNKS
    # ========================================================

    print(
        "\n[3/5] Creating legal chunks..."
    )

    chunks = chunk_text(
        cleaned_text,
        chunk_size=1000,
        chunk_overlap=200
    )

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )

    print(
        f"Created chunks: "
        f"{len(chunks):,}"
    )

    # ========================================================
    # 6. CREATE DOCUMENT ID
    # ========================================================

    document_id = _create_document_id(
        document_type,
        path
    )

    print(
        f"Document ID: {document_id}"
    )

    # ========================================================
    # 7. CREATE METADATA
    # ========================================================

    print(
        "\nCreating chunk metadata..."
    )

    metadatas = []

    for index, chunk in enumerate(
        chunks
    ):

        # ----------------------------------------------------
        # Detect legal section.
        #
        # Example:
        #
        # 303. Theft.—
        #
        # ----------------------------------------------------

        section_match = re.search(
            r"^\s*(\d{1,4})\.\s+([^\n]+)",
            chunk
        )

        if section_match:

            section_number = (
                section_match
                .group(1)
                .strip()
            )

            section_title = (
                section_match
                .group(2)
                .strip()
            )

        else:

            section_number = ""

            section_title = ""

        metadata = {

            "document_id":
                document_id,

            "document_type":
                document_type,

            "title":
                title,

            "source":
                path.name,

            "authority":
                authority,

            "chunk_index":
                index,

            "section_number":
                section_number,

            "section_title":
                section_title,
        }

        if year is not None:

            metadata["year"] = int(
                year
            )

        if act_name:

            metadata["act_name"] = (
                act_name
            )

        metadatas.append(
            metadata
        )

    # ========================================================
    # 8. EMBEDDINGS
    # ========================================================

    print(
        "\n[4/5] Creating embeddings..."
    )

    embeddings = create_embeddings(
        chunks
    )

    if not embeddings:

        raise ValueError(
            "Embedding generation failed."
        )

    if len(embeddings) != len(chunks):

        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    print(
        f"Created embeddings: "
        f"{len(embeddings):,}"
    )

    # ========================================================
    # 9. SECTION STATISTICS
    # ========================================================

    sections_found = sum(
        1
        for metadata in metadatas
        if metadata.get(
            "section_number"
        )
    )

    print(
        f"Sections detected: "
        f"{sections_found:,}"
    )

    # ========================================================
    # 10. STORE IN CHROMADB
    # ========================================================

    print(
        "\n[5/5] Storing chunks in ChromaDB..."
    )

    stored_count = add_chunks(
        chunks=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
        source=path.name,
    )

    print(
        f"Stored chunks: "
        f"{stored_count:,}"
    )

    if stored_count != len(chunks):

        raise ValueError(
            "Number of stored chunks does not match "
            "number of generated chunks."
        )

    # ========================================================
    # 11. FINAL RESULT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LEGAL DOCUMENT INGESTION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Document ID    : {document_id}"
    )

    print(
        f"Title          : {title}"
    )

    print(
        f"Type           : {document_type}"
    )

    print(
        f"Source         : {path.name}"
    )

    print(
        f"Chunks         : {stored_count:,}"
    )

    print(
        f"Characters     : {len(cleaned_text):,}"
    )

    print(
        f"Sections found : {sections_found:,}"
    )

    print(
        "=" * 70
    )

    return {

        "document_id":
            document_id,

        "title":
            title,

        "document_type":
            document_type,

        "source":
            path.name,

        "authority":
            authority,

        "year":
            year,

        "act_name":
            act_name,

        "chunks":
            stored_count,

        "characters":
            len(cleaned_text),

        "sections_found":
            sections_found,
    }


# ============================================================
# BACKWARD-COMPATIBLE ACT INGESTION
# ============================================================

def ingest_legal_act(
    file_path: str,
    act_name: str,
    year: int,
) -> dict:
    """
    Backward-compatible helper for ingesting an Act.
    """

    return ingest_legal_document(

        file_path=file_path,

        document_type="Act",

        title=act_name,

        year=year,

        authority="Government of India",

        act_name=act_name,
    )