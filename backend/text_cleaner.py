import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF/OCR text before sending it
    to the RAG pipeline.
    """

    if not text:
        return ""

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text