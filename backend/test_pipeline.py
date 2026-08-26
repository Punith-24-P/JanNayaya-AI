from backend.pdf_service import (
    extract_text_from_pdf
)

from backend.text_cleaner import (
    clean_text
)

from backend.chunker import (
    chunk_text
)


# ============================================================
# PDF PATH
# ============================================================

PDF_PATH = (
    "uploads/SYNOPSIS.vtu0.pdf"
)


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    print(
        "\n1. Extracting text from PDF..."
    )

    raw_text = extract_text_from_pdf(
        PDF_PATH
    )

    print(
        "Extracted characters:",
        len(raw_text)
    )

    print(
        "\n2. Cleaning text..."
    )

    cleaned_text = clean_text(
        raw_text
    )

    print(
        "Cleaned characters:",
        len(cleaned_text)
    )

    print(
        "\n3. Creating chunks..."
    )

    chunks = chunk_text(

        cleaned_text,

        chunk_size=1000,

        chunk_overlap=200
    )

    print(
        "Number of chunks:",
        len(chunks)
    )

    print(
        "\n4. First chunk:"
    )

    print(
        "-" * 50
    )

    if chunks:

        print(
            chunks[0]
        )

    else:

        print(
            "NO CHUNKS CREATED"
        )

    print(
        "-" * 50
    )

    print(
        "\nPipeline completed successfully!"
    )