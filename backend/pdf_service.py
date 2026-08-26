"""
JanNyaya AI - PDF & Document Text Extraction Service

Features:
- Validates file signatures / magic bytes (PDF, JPG, PNG, WEBP).
- Rejects HTML error pages, XML, corrupted, or zero-byte files.
- Extracts selectable text page-by-page using PyMuPDF.
- Falls back to PaddleOCR for scanned or image-only pages.
- Preserves page number provenance and line ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union
import fitz
import numpy as np

from backend.ocr_service import get_ocr, _extract_text_from_result


# ============================================================
# FILE SIGNATURE & CONTENT VALIDATION
# ============================================================

def validate_document_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate that a file exists, is non-empty, has valid magic bytes,
    and is not an HTML error page or corrupted content.

    Supported formats: PDF, JPG, JPEG, PNG, WEBP.
    """
    p = Path(path)
    if not p.exists():
        return {
            "is_valid": False,
            "file_type": "unknown",
            "file_size": 0,
            "error": f"File does not exist: {p.name}",
        }

    if not p.is_file():
        return {
            "is_valid": False,
            "file_type": "unknown",
            "file_size": 0,
            "error": f"Path is not a regular file: {p.name}",
        }

    file_size = p.stat().st_size
    if file_size < 16:
        return {
            "is_valid": False,
            "file_type": "unknown",
            "file_size": file_size,
            "error": f"File is empty or too small ({file_size} bytes): {p.name}",
        }

    # Read header bytes for signature check
    try:
        with p.open("rb") as f:
            header = f.read(2048)
    except Exception as read_err:
        return {
            "is_valid": False,
            "file_type": "unknown",
            "file_size": file_size,
            "error": f"Failed to read file header: {str(read_err)}",
        }

    # Explicitly check and reject HTML / HTTP error pages
    lower_header = header.lower()
    if (
        b"<html" in lower_header
        or b"<!doctype html" in lower_header
        or b"<head" in lower_header
        or b"404 not found" in lower_header
        or b"500 internal server" in lower_header
        or b"access denied" in lower_header
    ):
        return {
            "is_valid": False,
            "file_type": "html_error",
            "file_size": file_size,
            "error": f"Uploaded file '{p.name}' is an HTML document or HTTP error page, not a valid legal document.",
        }

    # Check magic signatures
    detected_type = None

    if b"%PDF-" in header[:1024]:
        detected_type = "pdf"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_type = "png"
    elif header.startswith(b"\xff\xd8\xff"):
        detected_type = "jpg"
    elif header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        detected_type = "webp"
    elif p.suffix.lower() in (".txt", ".text"):
        try:
            header.decode("utf-8")
            detected_type = "txt"
        except UnicodeDecodeError:
            detected_type = None

    if not detected_type:
        ext = p.suffix.lower().lstrip(".")
        return {
            "is_valid": False,
            "file_type": "unsupported",
            "file_size": file_size,
            "error": (
                f"File '{p.name}' has invalid or unrecognizable signature. "
                "Supported formats: PDF (%PDF-), PNG, JPG/JPEG, WEBP, TXT."
            ),
        }

    return {
        "is_valid": True,
        "file_type": detected_type,
        "file_size": file_size,
        "error": None,
    }


def is_valid_pdf_file(path: Union[str, Path]) -> bool:
    """Verify that the file is a valid PDF document with %PDF- header."""
    res = validate_document_file(path)
    return res["is_valid"] and res["file_type"] == "pdf"


# ============================================================
# EXTRACT PAGES FROM PDF
# ============================================================

def extract_pages_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF document.

    Returns:
        List of dicts with keys:
            - page_number (1-indexed int)
            - text (str)
            - extraction_method ("selectable" or "ocr")
            - character_count (int)
    """
    path = Path(file_path)
    validation = validate_document_file(path)

    if not validation["is_valid"]:
        raise ValueError(validation["error"])

    if validation["file_type"] != "pdf":
        raise ValueError(f"Expected PDF document, but detected '{validation['file_type']}'.")

    try:
        document = fitz.open(str(path))
    except Exception as err:
        raise ValueError(f"Unable to open PDF document '{path.name}': {str(err)}")

    pages: List[Dict[str, Any]] = []
    ocr = None

    try:
        total_pages = len(document)
        if total_pages == 0:
            raise ValueError(f"PDF document '{path.name}' contains 0 pages.")

        for page_number, page in enumerate(document):
            p_num = page_number + 1

            # 1. Try normal selectable PDF text
            text = (page.get_text("text") or "").strip()

            if text and len(text) >= 15:
                pages.append({
                    "page_number": p_num,
                    "text": text,
                    "extraction_method": "selectable",
                    "character_count": len(text),
                })
                continue

            # 2. No or minimal selectable text -> OCR fallback on rendered page
            if ocr is None:
                ocr = get_ocr()

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            )

            image_bytes = pixmap.tobytes("png")
            image = np.frombuffer(image_bytes, dtype=np.uint8)

            try:
                result = ocr.predict(image)
                page_text_lines = _extract_text_from_result(result)
                ocr_text = "\n".join(page_text_lines).strip()
            except Exception as ocr_error:
                ocr_text = ""

            pages.append({
                "page_number": p_num,
                "text": ocr_text,
                "extraction_method": "ocr",
                "character_count": len(ocr_text),
            })

    finally:
        document.close()

    return pages


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(
    file_path: str
) -> str:
    """
    Extract full aggregated text from a PDF, preserving clear page boundaries.
    """
    pages = extract_pages_from_pdf(file_path)

    all_text = []
    for page_info in pages:
        text = page_info.get("text", "").strip()
        p_num = page_info.get("page_number", 1)
        if text:
            all_text.append(f"--- Page {p_num} ---\n{text}")

    final_text = "\n\n".join(all_text)
    return final_text