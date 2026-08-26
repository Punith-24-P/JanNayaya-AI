"""
JanNyaya AI - Clean Legal Database Rebuild

WARNING
-------
This script CLEARs the current Chroma legal collection and rebuilds it
ONLY from files inside:

    legal_data/

Use this after organizing the legal corpus.

Current expected structure:

legal_data/
├── criminal/
│   └── BNS/
├── civil/
│   ├── Contract_Act/
│   ├── CPC/
│   └── Limitation_Act/
├── property/
├── family/
├── consumer/
└── employment/

The rebuild is intentionally conservative:
- old Chroma records are removed
- only legal_data PDFs are ingested
- all records get route-aware metadata
"""

from __future__ import annotations

from backend.vector_store import (
    clear_collection,
    get_collection_count,
)

from backend.legal_data_ingest import (
    ingest_all_legal_data,
)


def main() -> None:

    print()
    print("=" * 70)
    print("JAN NYAYA AI - CLEAN LEGAL DATABASE REBUILD")
    print("=" * 70)

    print()
    print(
        "Current Chroma chunks:",
        get_collection_count(),
    )

    print()
    print(
        "WARNING: The existing legal knowledge base will be cleared."
    )

    confirmation = input(
        "Type REBUILD to continue: "
    ).strip()

    if confirmation != "REBUILD":

        print()
        print(
            "Rebuild cancelled."
        )

        return

    # ========================================================
    # CLEAR OLD DATABASE
    # ========================================================

    print()
    print(
        "Clearing old Chroma collection..."
    )

    clear_collection()

    print(
        "Chunks after clearing:",
        get_collection_count(),
    )

    # ========================================================
    # INGEST ORGANIZED LEGAL DATA
    # ========================================================

    print()
    print(
        "Ingesting organized legal data..."
    )

    result = ingest_all_legal_data()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("REBUILD COMPLETE")
    print("=" * 70)

    print(
        "Result:",
        result,
    )

    print()
    print(
        "Final Chroma chunks:",
        get_collection_count(),
    )

    print(
        "=" * 70,
    )


if __name__ == "__main__":
    main()