"""
JanNyaya AI - Organized Legal Data Ingestion

Reads PDFs from:

legal_data/
├── criminal/BNS/
├── civil/Contract_Act/
├── civil/CPC/
├── civil/Limitation_Act/
├── property/
├── family/
├── consumer/
└── employment/

Uses the existing JanNyaya services:

    backend.pdf_service
    backend.text_cleaner
    backend.chunker
    backend.embedding_service
    backend.vector_store

It does NOT replace the existing Chroma database.

Metadata added:
    route
    category
    act_name
    year
    source
    title
    authority
    document_type
    section_number
    section_title
    document_id
    chunk_index
"""


from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import Any, Dict, List


from backend.pdf_service import (
    extract_text_from_pdf,
    is_valid_pdf_file,
)

from backend.text_cleaner import (
    clean_text,
)

from backend.chunker import (
    chunk_text,
)

from backend.embedding_service import (
    create_embeddings,
)

from backend.vector_store import (
    add_chunks,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

LEGAL_DATA_DIR = (
    PROJECT_ROOT
    / "legal_data"
)


# ============================================================
# ROUTE MAPPING
# ============================================================

ROUTE_MAP = {
    "criminal": "criminal",
    "civil": "civil_contractual",
    "property": "property",
    "family": "family",
    "consumer": "consumer",
    "employment": "employment",
    "commercial": "commercial",
    "cyber": "cyber",
    "legal_aid": "legal_aid",
    "traffic": "traffic",
    "governance": "governance_rti",
    "children": "children_pocso",
    "real_estate": "real_estate",
    "senior_citizens": "senior_citizens",
}


# ============================================================
# CATEGORY METADATA
# ============================================================

CATEGORY_INFO = {
    "BNS": {
        "act_name": "Bharatiya Nyaya Sanhita",
        "year": 2023,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Contract_Act": {
        "act_name": "Indian Contract Act",
        "year": 1872,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "CPC": {
        "act_name": "Code of Civil Procedure",
        "year": 1908,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Limitation_Act": {
        "act_name": "Limitation Act",
        "year": 1963,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Transfer_of_Property_Act": {
        "act_name": "Transfer of Property Act",
        "year": 1882,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Consumer_Protection_Act": {
        "act_name": "Consumer Protection Act",
        "year": 2019,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Negotiable_Instruments_Act": {
        "act_name": "Negotiable Instruments Act",
        "year": 1881,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Domestic_Violence_Act": {
        "act_name": "Protection of Women from Domestic Violence Act",
        "year": 2005,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Hindu_Marriage_Act": {
        "act_name": "Hindu Marriage Act",
        "year": 1955,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Information_Technology_Act": {
        "act_name": "Information Technology Act",
        "year": 2000,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Payment_of_Gratuity_Act": {
        "act_name": "Payment of Gratuity Act",
        "year": 1972,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Legal_Services_Authorities_Act": {
        "act_name": "Legal Services Authorities Act",
        "year": 1987,
        "authority": "Government of India",
        "document_type": "Act",
    },
    "Motor_Vehicles_Act": {
        "act_name": "Motor Vehicles Act",
        "year": 1988,
        "authority": "Ministry of Road Transport and Highways",
        "document_type": "Act",
    },
    "RTI_Act": {
        "act_name": "Right to Information Act",
        "year": 2005,
        "authority": "Ministry of Personnel and Public Grievances",
        "document_type": "Act",
    },
    "POCSO_Act": {
        "act_name": "Protection of Children from Sexual Offences Act",
        "year": 2012,
        "authority": "Ministry of Women and Child Development",
        "document_type": "Act",
    },
    "RERA_Act": {
        "act_name": "Real Estate (Regulation and Development) Act",
        "year": 2016,
        "authority": "Ministry of Housing and Urban Affairs",
        "document_type": "Act",
    },
    "Senior_Citizens_Act": {
        "act_name": "Maintenance and Welfare of Parents and Senior Citizens Act",
        "year": 2007,
        "authority": "Ministry of Social Justice and Empowerment",
        "document_type": "Act",
    },
    "Code_on_Wages": {
        "act_name": "Code on Wages",
        "year": 2019,
        "authority": "Ministry of Labour and Employment",
        "document_type": "Act",
    },
    "Specific_Relief_Act": {
        "act_name": "Specific Relief Act",
        "year": 1963,
        "authority": "Ministry of Law and Justice",
        "document_type": "Act",
    },
    "Hindu_Succession_Act": {
        "act_name": "Hindu Succession Act",
        "year": 1956,
        "authority": "Ministry of Law and Justice",
        "document_type": "Act",
    },
    "Banking_Regulation_Act": {
        "act_name": "Banking Regulation Act",
        "year": 1949,
        "authority": "Reserve Bank of India / Ministry of Finance",
        "document_type": "Act",
    },
    "SARFAESI_Act": {
        "act_name": "SARFAESI Act (Securitisation & Asset Recovery)",
        "year": 2002,
        "authority": "Ministry of Finance",
        "document_type": "Act",
    },
    "Insolvency_Bankruptcy_Code": {
        "act_name": "Insolvency and Bankruptcy Code (IBC)",
        "year": 2016,
        "authority": "Insolvency and Bankruptcy Board of India",
        "document_type": "Act",
    },
    "RBI_Act": {
        "act_name": "Reserve Bank of India Act & Banking Ombudsman Rules",
        "year": 1934,
        "authority": "Reserve Bank of India",
        "document_type": "Act / Regulations",
    },
    "BNSS": {
        "act_name": "Bharatiya Nagarik Suraksha Sanhita (BNSS)",
        "year": 2023,
        "authority": "Ministry of Home Affairs",
        "document_type": "Act",
    },
    "BSA": {
        "act_name": "Bharatiya Sakshya Adhiniyam (BSA - Evidence)",
        "year": 2023,
        "authority": "Ministry of Home Affairs",
        "document_type": "Act",
    },
    "DPDP_Act": {
        "act_name": "Digital Personal Data Protection Act (DPDP)",
        "year": 2023,
        "authority": "Ministry of Electronics and Information Technology",
        "document_type": "Act",
    },
    "Arbitration_Act": {
        "act_name": "Arbitration and Conciliation Act",
        "year": 1996,
        "authority": "Ministry of Law and Justice",
        "document_type": "Act",
    },
    "Juvenile_Justice_Act": {
        "act_name": "Juvenile Justice (Care and Protection of Children) Act",
        "year": 2015,
        "authority": "Ministry of Women and Child Development",
        "document_type": "Act",
    },
    "Companies_Act": {
        "act_name": "Companies Act",
        "year": 2013,
        "authority": "Ministry of Corporate Affairs, Government of India",
        "document_type": "Act",
    },
}


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(
    path: Path,
) -> str:
    """
    SHA256 hash for identifying duplicate PDF files.
    """

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================
# DOCUMENT ID
# ============================================================

def build_document_id(
    act_name: str,
    year: Any,
) -> str:

    safe_name = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        act_name,
    ).strip("_")

    if year:
        return (
            f"{safe_name}_{year}"
        )

    return safe_name


# ============================================================
# SECTION METADATA
# ============================================================

def extract_section_metadata(
    text: str,
) -> Dict[str, str]:
    """
    Best-effort section number/title extraction.

    We preserve the actual source text and only add metadata
    when a section heading can be detected.
    """

    if not text:
        return {
            "section_number":
                "",

            "section_title":
                "",
        }

    section_number = ""

    section_title = ""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # --------------------------------------------------------
    # Find section number
    # --------------------------------------------------------

    patterns = [
        r"^Section\s+(\d+[A-Za-z]?)",
        r"^(\d+[A-Za-z]?)\.\s+",
        r"^(\d+[A-Za-z]?)\s*[–—:-]\s*",
    ]

    matched_index = None

    for index, line in enumerate(
        lines
    ):

        for pattern in patterns:

            match = re.search(
                pattern,
                line,
                flags=re.IGNORECASE,
            )

            if match:

                section_number = (
                    match.group(
                        1
                    ).strip()
                )

                matched_index = index

                break

        if section_number:
            break

    if not section_number:
        return {
            "section_number":
                "",

            "section_title":
                "",
        }

    # --------------------------------------------------------
    # Section title
    # --------------------------------------------------------

    if matched_index is not None:

        source_line = lines[
            matched_index
        ]

        cleaned_title = re.sub(
            r"^Section\s+\d+[A-Za-z]?\s*"
            r"[\.\)\-–—:]?\s*",
            "",
            source_line,
            flags=re.IGNORECASE,
        )

        cleaned_title = re.sub(
            r"^\d+[A-Za-z]?\s*"
            r"[\.\)\-–—:]?\s*",
            "",
            cleaned_title,
        ).strip()

        if cleaned_title:
            section_title = (
                cleaned_title[:300]
            )

    # --------------------------------------------------------
    # Sometimes heading is on next line
    # --------------------------------------------------------

    if (
        not section_title
        and matched_index is not None
        and matched_index + 1
        < len(lines)
    ):

        candidate = lines[
            matched_index + 1
        ].strip()

        if (
            candidate
            and len(candidate) <= 300
        ):

            section_title = candidate

    return {
        "section_number":
            section_number,

        "section_title":
            section_title,
    }


# ============================================================
# CHUNK SECTION CONTEXT
# ============================================================

def detect_best_section(
    chunk: str,
    previous_section: Dict[str, str],
) -> Dict[str, str]:
    """
    Use the chunk's own section when possible.
    Otherwise carry forward the previous detected section.
    """

    current = (
        extract_section_metadata(
            chunk
        )
    )

    if current[
        "section_number"
    ]:

        return current

    return previous_section.copy()


# ============================================================
# DISCOVER LEGAL PDFs
# ============================================================

def discover_legal_pdfs() -> List[Path]:

    if not LEGAL_DATA_DIR.exists():
        return []

    return sorted(
        LEGAL_DATA_DIR.rglob(
            "*.pdf"
        )
    )


# ============================================================
# READ FOLDER METADATA
# ============================================================

def get_folder_metadata(
    path: Path,
) -> Dict[str, Any]:

    relative = path.relative_to(
        LEGAL_DATA_DIR
    )

    parts = relative.parts

    route_folder = (
        parts[0]
        if len(parts) >= 1
        else "general"
    )

    category_folder = (
        parts[1]
        if len(parts) >= 2
        else path.parent.name
    )

    route = ROUTE_MAP.get(
        route_folder,
        "general",
    )

    category_info = CATEGORY_INFO.get(
        category_folder
    )

    if category_info:

        act_name = category_info[
            "act_name"
        ]

        year = category_info[
            "year"
        ]

        authority = category_info[
            "authority"
        ]

        document_type = category_info[
            "document_type"
        ]

    else:

        act_name = (
            category_folder
            .replace(
                "_",
                " ",
            )
        )

        year = None

        authority = (
            "Unknown"
        )

        document_type = (
            "Legal document"
        )

        # Try year from filename.
        match = re.search(
            r"\b(18|19|20)\d{2}\b",
            path.name,
        )

        if match:

            year = int(
                match.group(
                    0
                )
            )

    return {
        "route":
            route,

        "category":
            route_folder,

        "category_folder":
            category_folder,

        "act_name":
            act_name,

        "year":
            year,

        "authority":
            authority,

        "document_type":
            document_type,

        "relative_path":
            str(relative),
    }


# ============================================================
# PREPARE PDF
# ============================================================

def prepare_pdf(
    path: Path,
) -> Dict[str, Any]:

    if not is_valid_pdf_file(path):
        raise ValueError(
            f"File '{path.name}' does not have a valid %PDF signature header."
        )

    metadata = get_folder_metadata(
        path
    )

    file_hash = calculate_file_hash(
        path
    )

    document_id = build_document_id(
        metadata[
            "act_name"
        ],
        metadata[
            "year"
        ],
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FILE:",
        path,
    )

    print(
        "Route:",
        metadata[
            "route"
        ],
    )

    print(
        "Act:",
        metadata[
            "act_name"
        ],
    )

    print(
        "Year:",
        metadata[
            "year"
        ],
    )

    print(
        "Document ID:",
        document_id,
    )

    print(
        "File hash:",
        file_hash[
            :16
        ],
        "...",
    )

    # --------------------------------------------------------
    # Extract PDF
    # --------------------------------------------------------

    raw_text = (
        extract_text_from_pdf(
            str(path)
        )
    )

    text = clean_text(
        raw_text
    )

    if not text:

        raise ValueError(
            "No readable text extracted."
        )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    chunks = chunk_text(
        text
    )

    if not chunks:

        raise ValueError(
            "No chunks were generated."
        )

    documents = []

    metadatas = []

    previous_section = {
        "section_number":
            "",

        "section_title":
            "",
    }

    # --------------------------------------------------------
    # Create records
    # --------------------------------------------------------

    for index, chunk in enumerate(
        chunks
    ):

        chunk = str(
            chunk or ""
        ).strip()

        if not chunk:
            continue

        section_info = (
            detect_best_section(
                chunk,
                previous_section,
            )
        )

        if section_info[
            "section_number"
        ]:

            previous_section = (
                section_info.copy()
            )

        record_metadata = {
            "route":
                metadata[
                    "route"
                ],

            "category":
                metadata[
                    "category"
                ],

            "category_folder":
                metadata[
                    "category_folder"
                ],

            "act_name":
                metadata[
                    "act_name"
                ],

            "year":
                (
                    metadata[
                        "year"
                    ]
                    if metadata[
                        "year"
                    ] is not None
                    else ""
                ),

            "source":
                path.name,

            "title":
                metadata[
                    "act_name"
                ],

            "authority":
                metadata[
                    "authority"
                ],

            "document_type":
                metadata[
                    "document_type"
                ],

            "section_number":
                section_info[
                    "section_number"
                ],

            "section_title":
                section_info[
                    "section_title"
                ],

            "document_id":
                document_id,

            "chunk_index":
                index,

            "relative_path":
                metadata[
                    "relative_path"
                ],

            "file_hash":
                file_hash,
        }

        documents.append(
            chunk
        )

        metadatas.append(
            record_metadata
        )

    return {
        "documents":
            documents,

        "metadatas":
            metadatas,

        "document_id":
            document_id,

        "source":
            path.name,

        "act_name":
            metadata[
                "act_name"
            ],

        "year":
            metadata[
                "year"
            ],

        "route":
            metadata[
                "route"
            ],

        "file_hash":
            file_hash,

        "characters":
            len(text),

        "chunks":
            len(documents),
    }


# ============================================================
# DUPLICATE RECORD HANDLING
# ============================================================

def remove_duplicate_records(
    documents: List[str],
    metadatas: List[dict],
) -> tuple[
    List[str],
    List[dict],
]:
    """
    Remove exact duplicate legal records.

    This is content-based, not merely filename-based.
    """

    unique_documents = []

    unique_metadatas = []

    seen = set()

    for index, document in enumerate(
        documents
    ):

        normalized = re.sub(
            r"\s+",
            " ",
            str(document).strip().lower(),
        )

        metadata = metadatas[
            index
        ]

        key = (
            metadata.get(
                "act_name",
                "",
            ),

            metadata.get(
                "year",
                "",
            ),

            metadata.get(
                "section_number",
                "",
            ),

            hashlib.sha1(
                normalized.encode(
                    "utf-8",
                    errors="ignore",
                )
            ).hexdigest(),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_documents.append(
            document
        )

        unique_metadatas.append(
            metadata
        )

    return (
        unique_documents,
        unique_metadatas,
    )


# ============================================================
# INGEST ALL LEGAL DATA
# ============================================================

def ingest_all_legal_data() -> Dict[str, Any]:

    pdfs = discover_legal_pdfs()

    if not pdfs:

        return {
            "status":
                "error",

            "message":
                (
                    "No PDF files found inside "
                    "legal_data/"
                ),

            "documents":
                0,

            "chunks":
                0,
        }

    all_documents = []

    all_metadatas = []

    processed_files = 0

    failed_files = []

    # --------------------------------------------------------
    # Prepare PDFs
    # --------------------------------------------------------

    for pdf in pdfs:

        try:

            result = prepare_pdf(
                pdf
            )

            all_documents.extend(
                result[
                    "documents"
                ]
            )

            all_metadatas.extend(
                result[
                    "metadatas"
                ]
            )

            processed_files += 1

            print(
                "Prepared:",
                result[
                    "chunks"
                ],
                "chunks"
            )

        except Exception as error:

            failed_files.append(
                {
                    "file":
                        str(pdf),

                    "error":
                        str(error),
                }
            )

            print()
            print(
                "FAILED:",
                pdf,
            )

            print(
                type(error).__name__,
                str(error),
            )

    if not all_documents:

        return {
            "status":
                "error",

            "message":
                "No legal chunks prepared.",

            "documents":
                processed_files,

            "chunks":
                0,

            "failed":
                failed_files,
        }

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before = len(
        all_documents
    )

    (
        unique_documents,
        unique_metadatas,
    ) = remove_duplicate_records(
        all_documents,
        all_metadatas,
    )

    duplicates_removed = (
        before
        - len(
            unique_documents
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Records before deduplication:",
        before,
    )

    print(
        "Records after deduplication:",
        len(
            unique_documents
        ),
    )

    print(
        "Duplicates removed:",
        duplicates_removed,
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print()
    print(
        "Creating embeddings..."
    )

    embeddings = create_embeddings(
        unique_documents
    )

    print(
        "Embeddings created:",
        len(
            embeddings
        ),
    )

    if len(
        embeddings
    ) != len(
        unique_documents
    ):

        raise RuntimeError(
            "Embedding count does not match "
            "document count."
        )

    # --------------------------------------------------------
    # Add to Chroma
    # --------------------------------------------------------

    print()
    print(
        "Adding records to ChromaDB..."
    )

    added = add_chunks(
        chunks=unique_documents,

        metadatas=unique_metadatas,

        embeddings=embeddings,

        source="legal_data",
    )

    print()
    print(
        "=" * 70
    )

    print(
        "LEGAL DATA INGESTION COMPLETE"
    )

    print(
        "Files processed:",
        processed_files,
    )

    print(
        "Chunks added/updated:",
        added,
    )

    print(
        "Duplicates removed:",
        duplicates_removed,
    )

    print(
        "Failed files:",
        len(
            failed_files
        ),
    )

    print(
        "=" * 70
    )

    return {
        "status":
            "success",

        "documents":
            processed_files,

        "chunks":
            added,

        "duplicates_removed":
            duplicates_removed,

        "failed":
            failed_files,
    }


# ============================================================
# TEST ONLY CURRENT LEGAL DATA
# ============================================================

def main() -> None:

    print()
    print(
        "JanNyaya AI - Organized Legal Data Ingestion"
    )

    print(
        "Directory:",
        LEGAL_DATA_DIR,
    )

    print()

    result = (
        ingest_all_legal_data()
    )

    print()
    print(
        "RESULT:"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()