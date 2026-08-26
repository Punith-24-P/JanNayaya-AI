"""
JanNyaya AI - PDF Ingestion

Backward-compatible PDF ingestion used by FastAPI.

For legal documents, processing is handled by:
    backend.legal_ingest.ingest_legal_document
"""

import sys
from pathlib import Path

from backend.legal_ingest import ingest_legal_document


# ============================================================
# BACKWARD-COMPATIBLE PDF INGESTION
# ============================================================

def ingest_pdf(file_path: str) -> dict:
    """
    Ingest a PDF document into the JanNyaya AI knowledge base.

    Generic uploaded PDFs use:
        document_type = "Other"
        title = filename
        authority = "Unknown"
    """

    if not file_path:
        raise ValueError(
            "file_path cannot be empty."
        )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Provided path is not a file: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    return ingest_legal_document(
        file_path=str(path),
        document_type="Other",
        title=path.name,
        authority="Unknown",
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main() -> None:
    """
    Command-line test for PDF ingestion.

    Usage:
        python -m backend.ingest path/to/file.pdf
    """

    if len(sys.argv) < 2:
        print()
        print("Usage:")
        print(
            "python -m backend.ingest path/to/file.pdf"
        )
        print()
        sys.exit(1)

    file_path = sys.argv[1]

    print()
    print("=" * 70)
    print("JAN NYAYA AI - PDF INGESTION")
    print("=" * 70)
    print()
    print(f"File: {file_path}")
    print()

    try:
        result = ingest_pdf(
            file_path
        )

        print()
        print("=" * 70)
        print("INGESTION RESULT")
        print("=" * 70)
        print()

        for key, value in result.items():
            print(
                f"{key}: {value}"
            )

        print()
        print("=" * 70)

    except Exception as error:
        print()
        print("=" * 70)
        print("INGESTION FAILED")
        print("=" * 70)
        print()
        print(
            str(error)
        )
        print()

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()