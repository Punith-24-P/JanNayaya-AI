"""
JanNyaya AI - Multilingual Legal Document Ingestion

This module ingests legal PDFs into the JanNyaya AI knowledge base.

Pipeline:

    PDF
      ↓
    Text extraction
      ↓
    Text cleaning
      ↓
    Legal-aware chunking
      ↓
    Metadata extraction
      ↓
    Multilingual E5 embeddings
      ↓
    ChromaDB

Important:
The embedding model is multilingual-e5-small.

E5 embedding convention:
    Documents -> "passage: ..."
    Queries   -> "query: ..."

The actual prefixing for query embeddings is handled by
backend.embedding_service.py.
This module prefixes document chunks as "passage: ...".
"""

import re
from pathlib import Path
from typing import Optional, List, Dict

from backend.pdf_service import extract_text_from_pdf
from backend.text_cleaner import clean_text
from backend.chunker import chunk_text
from backend.embedding_service import create_embeddings
from backend.vector_store import add_chunks


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
    ).strip("_")

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        file_path.stem.lower()
    ).strip("_")

    if not safe_type:
        safe_type = "document"

    if not safe_name:
        safe_name = "unknown"

    return f"{safe_type}_{safe_name}"


# ============================================================
# DETECT SECTION METADATA
# ============================================================

def _extract_section_metadata(
    chunk: str
) -> tuple[str, str]:
    """
    Extract the legal section number and section title
    from the beginning of a chunk.

    Example:

        303. Theft.—

    becomes:

        section_number = "303"
        section_title  = "Theft.—"
    """

    if not chunk:
        return "", ""

    section_match = re.search(
        r"^\s*(\d{1,4})\.\s+([^\n]+)",
        chunk
    )

    if not section_match:
        return "", ""

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

    return section_number, section_title


# ============================================================
# BUILD METADATA
# ============================================================

def _build_metadata(
    chunks: List[str],
    document_id: str,
    document_type: str,
    title: str,
    source: str,
    authority: str,
    year: Optional[int],
    act_name: Optional[str],
) -> List[Dict]:
    """
    Create metadata for every chunk.
    """

    metadatas = []

    for index, chunk in enumerate(chunks):

        section_number, section_title = (
            _extract_section_metadata(chunk)
        )

        metadata = {
            "document_id": document_id,
            "document_type": document_type,
            "title": title,
            "source": source,
            "authority": authority,
            "chunk_index": index,
            "section_number": section_number,
            "section_title": section_title,
        }

        if year is not None:
            metadata["year"] = int(year)

        if act_name:
            metadata["act_name"] = act_name

        metadatas.append(metadata)

    return metadatas


# ============================================================
# PREPARE DOCUMENT TEXT FOR E5
# ============================================================

def _prepare_passages(
    chunks: List[str]
) -> List[str]:
    """
    Prepare legal chunks for multilingual-e5 embeddings.

    E5 expects document text in the form:

        passage: <document text>

    """

    passages = []

    for chunk in chunks:

        if not chunk:
            passages.append(
                "passage:"
            )

        else:
            passages.append(
                f"passage: {chunk}"
            )

    return passages


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

    Parameters
    ----------
    file_path:
        Path to the PDF.

    document_type:
        Examples:
            Act
            Judgment
            Legal_Aid
            Report
            Notification
            Regulation
            Other

    title:
        Human-readable document title.

    year:
        Optional publication/year value.

    authority:
        Issuing authority.

    act_name:
        Optional Act name.

    Returns
    -------
    dict
        Information about ingestion.
    """

    # ========================================================
    # 1. VALIDATE INPUT
    # ========================================================

    path = Path(file_path)

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

    print()
    print("=" * 70)
    print("JAN NYAYA AI - MULTILINGUAL LEGAL INGESTION")
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
        "\n[1/6] Extracting PDF text..."
    )

    text = extract_text_from_pdf(
        str(path)
    )

    if text is None:
        raise ValueError(
            "PDF text extraction returned None."
        )

    print(
        f"Extracted characters: {len(text):,}"
    )

    if not text.strip():
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    # ========================================================
    # 4. CLEAN TEXT
    # ========================================================

    print(
        "\n[2/6] Cleaning text..."
    )

    cleaned_text = clean_text(
        text
    )

    if cleaned_text is None:
        raise ValueError(
            "Text cleaning returned None."
        )

    print(
        f"Cleaned characters: {len(cleaned_text):,}"
    )

    if not cleaned_text.strip():
        raise ValueError(
            "Document became empty after cleaning."
        )

    # ========================================================
    # 5. CREATE LEGAL CHUNKS
    # ========================================================

    print(
        "\n[3/6] Creating legal chunks..."
    )

    chunks = chunk_text(
        cleaned_text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    if chunks is None:
        raise ValueError(
            "Chunking returned None."
        )

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    print(
        f"Created chunks: {len(chunks):,}"
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
        "\n[4/6] Creating chunk metadata..."
    )

    metadatas = _build_metadata(
        chunks=chunks,
        document_id=document_id,
        document_type=document_type,
        title=title,
        source=path.name,
        authority=authority,
        year=year,
        act_name=act_name,
    )

    if len(metadatas) != len(chunks):
        raise ValueError(
            "Number of metadata entries does not match "
            "number of chunks."
        )

    # ========================================================
    # 8. SECTION STATISTICS
    # ========================================================

    sections_found = sum(
        1
        for metadata in metadatas
        if metadata.get("section_number")
    )

    print(
        f"Sections detected: {sections_found:,}"
    )

    print(
        "\nSample section metadata:"
    )

    shown = 0

    for metadata in metadatas:

        section_number = metadata.get(
            "section_number",
            ""
        )

        section_title = metadata.get(
            "section_title",
            ""
        )

        if section_number:

            print(
                f"  Section {section_number}: "
                f"{section_title}"
            )

            shown += 1

        if shown >= 5:
            break

    # ========================================================
    # 9. PREPARE E5 PASSAGES
    # ========================================================

    print(
        "\nPreparing multilingual E5 passages..."
    )

    passage_chunks = _prepare_passages(
        chunks
    )

    if len(passage_chunks) != len(chunks):
        raise ValueError(
            "Prepared passage count does not match "
            "original chunk count."
        )

    # ========================================================
    # 10. CREATE EMBEDDINGS
    # ========================================================

    print(
        "\n[5/6] Creating multilingual embeddings..."
    )

    embeddings = create_embeddings(
        passage_chunks
    )

    if embeddings is None:
        raise ValueError(
            "Embedding generation returned None."
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    print(
        f"Created embeddings: {len(embeddings):,}"
    )

    # ========================================================
    # 11. STORE IN CHROMADB
    # ========================================================

    print(
        "\n[6/6] Storing chunks in ChromaDB..."
    )

    stored_count = add_chunks(
        chunks=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
        source=path.name,
    )

    print(
        f"Stored chunks: {stored_count:,}"
    )

    if stored_count != len(chunks):
        raise ValueError(
            "Number of stored chunks does not match "
            "number of generated chunks."
        )

    # ========================================================
    # 12. FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("LEGAL DOCUMENT INGESTION COMPLETED")
    print("=" * 70)

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
        "Embedding model: intfloat/multilingual-e5-small"
    )

    print(
        "Embedding mode : passage"
    )

    print("=" * 70)

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
        "sections_found": sections_found,
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


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "JanNyaya AI - Legal Ingestion Module"
    )

    print(
        "Import successful."
    )