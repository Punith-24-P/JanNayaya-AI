"""
JanNyaya AI - Text Cleaner & OCR Normalization Service

Repairs OCR artifacts safely:
- Currency fixes: t2,50,000 / r2,50,000 -> ₹2,50,000
- Comma number reconstruction: 2, 50, 000 -> 2,50,000
- Date reconstruction: 15 / 06 / 2022 -> 15/06/2022
- Percentage & interest: 12 % p . a . -> 12% p.a.
- Whitespace and line formatting
"""

import re
from typing import Tuple


def normalize_ocr_artifacts(text: str) -> str:
    """
    Repair common Indian OCR artifacts in legal documents safely without
    altering core textual evidence.
    """
    if not text:
        return ""

    val = str(text)

    # 1. PaddleOCR often transcribes Rupee symbol '₹' as lowercase 't', 'r', or 'z'
    # before Indian numbers (e.g., t2,50,000/- or t1,87,560)
    val = re.sub(
        r"(?<![A-Za-z0-9_])(?:t|r|z)(?=\s*[0-9]{1,3}(?:,[0-9]{2,3})+)",
        "₹",
        val,
        flags=re.IGNORECASE,
    )

    # Also for simple digits: t250000 or t 2,50,000
    val = re.sub(
        r"(?<![A-Za-z0-9_])t\s*(?=[0-9]{3,})",
        "₹",
        val,
        flags=re.IGNORECASE,
    )

    # 2. Fix broken currency abbreviations (e.g. Rs . 2,50,000 or Rs , -> Rs. )
    val = re.sub(r"\bRs\s*[\.,]\s*", "Rs. ", val, flags=re.IGNORECASE)
    val = re.sub(r"\bINR\s*[\.:]?\s*", "INR ", val, flags=re.IGNORECASE)
    val = re.sub(r"₹\s+", "₹", val)

    # 3. Repair fragmented Indian comma numbers (e.g. 2, 50, 000 -> 2,50,000)
    val = re.sub(
        r"(\b[0-9]{1,3})\s*,\s*([0-9]{2,3})\s*,\s*([0-9]{3}\b)",
        r"\1,\2,\3",
        val,
    )
    val = re.sub(
        r"(\b[0-9]{1,3})\s*,\s*([0-9]{2,3}\b)",
        r"\1,\2",
        val,
    )

    # 4. Repair broken dates (e.g. 15 / 06 / 2022 or 12 - 04 - 2024 -> 15/06/2022)
    val = re.sub(
        r"\b(\d{1,2})\s*[/]\s*(\d{1,2})\s*[/]\s*(\d{2,4})\b",
        r"\1/\2/\3",
        val,
    )
    val = re.sub(
        r"\b(\d{1,2})\s*[-]\s*(\d{1,2})\s*[-]\s*(\d{2,4})\b",
        r"\1-\2-\3",
        val,
    )
    val = re.sub(
        r"\b(\d{1,2})\s*[\.]\s*(\d{1,2})\s*[\.]\s*(\d{2,4})\b",
        r"\1.\2.\3",
        val,
    )

    # 5. Repair interest rates and percentages (e.g. 12 % p . a . -> 12% p.a.)
    val = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:p\s*[\.]?\s*a\s*[\.]?|per\s+annum)",
        r"\1% p.a.",
        val,
        flags=re.IGNORECASE,
    )
    val = re.sub(r"(\d+)\s*%", r"\1%", val)

    # 6. Repair standard /- amount endings
    val = re.sub(r"\s*/\s*-\s*", "/- ", val)

    return val


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted PDF/OCR text.
    Preserves document structure and paragraphing while eliminating noise.
    """
    if not text:
        return ""

    text = str(text)

    # Normalize line breaks and remove null bytes
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")

    # Apply OCR repairs
    text = normalize_ocr_artifacts(text)

    # Normalize spaces within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Trim spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Normalize spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Prevent excessive consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text_with_raw(text: str) -> Tuple[str, str]:
    """
    Returns a tuple of (original_raw_text, cleaned_normalized_text)
    to guarantee evidence preservation without silent overwrites.
    """
    raw = str(text or "")
    normalized = clean_text(raw)
    return raw, normalized