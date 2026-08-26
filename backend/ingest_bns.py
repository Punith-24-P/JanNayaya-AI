from backend.legal_ingest import (
    ingest_legal_document
)


# ============================================================
# BNS DOCUMENT
# ============================================================

PDF_PATH = (
    "data/acts/"
    "Bharatiya_Nyaya_Sanhita_2023.pdf"
)


# ============================================================
# INGEST
# ============================================================

if __name__ == "__main__":

    result = ingest_legal_document(

        file_path=PDF_PATH,

        document_type="Act",

        title=(
            "Bharatiya Nyaya Sanhita, 2023"
        ),

        year=2023,

        authority=(
            "Government of India"
        ),

        act_name=(
            "Bharatiya Nyaya Sanhita, 2023"
        ),
    )

    print(
        "\nFINAL RESULT"
    )

    print(
        result
    )