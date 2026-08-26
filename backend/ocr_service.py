from pathlib import Path
from typing import Any
import json

from paddleocr import PaddleOCR


# ============================================================
# GLOBAL OCR MODEL
# ============================================================

_ocr = None


# ============================================================
# GET OCR MODEL
# ============================================================

def get_ocr() -> PaddleOCR:
    """
    Initialize PaddleOCR only once.

    The OCR model is reused for subsequent requests.
    """

    global _ocr

    if _ocr is None:

        print("=" * 60)
        print("Initializing PaddleOCR...")

        _ocr = PaddleOCR(
            lang="en"
        )

        print(
            "PaddleOCR initialized successfully."
        )

        print("=" * 60)

    return _ocr


# ============================================================
# EXTRACT TEXT FROM OCR RESULT
# ============================================================

def _extract_text_from_result(
    result: Any
) -> list[str]:
    """
    Extract recognized text from PaddleOCR result.

    Supports the current PaddleOCR result format where
    result pages expose JSON through the .json property.
    """

    extracted_text = []

    if result is None:
        return extracted_text

    try:

        for page in result:

            data = None

            # ------------------------------------------------
            # PaddleOCR result object
            # ------------------------------------------------

            if hasattr(
                page,
                "json"
            ):

                data = page.json

            # ------------------------------------------------
            # Dictionary result
            # ------------------------------------------------

            elif isinstance(
                page,
                dict
            ):

                data = page

            # ------------------------------------------------
            # JSON string result
            # ------------------------------------------------

            elif isinstance(
                page,
                str
            ):

                data = page

            # ------------------------------------------------
            # Convert JSON string
            # ------------------------------------------------

            if isinstance(
                data,
                str
            ):

                try:

                    data = json.loads(
                        data
                    )

                except json.JSONDecodeError:

                    continue

            if not isinstance(
                data,
                dict
            ):

                continue

            page_data = data.get(
                "res",
                data
            )

            if not isinstance(
                page_data,
                dict
            ):

                continue

            texts = page_data.get(
                "rec_texts",
                []
            )

            if isinstance(
                texts,
                list
            ):

                for text in texts:

                    if text is None:
                        continue

                    text = str(
                        text
                    ).strip()

                    if text:

                        extracted_text.append(
                            text
                        )

    except Exception as error:

        print(
            "Warning while reading OCR result:",
            str(error)
        )

    return extracted_text


# ============================================================
# IMAGE OCR
# ============================================================

def extract_text_from_image(
    file_path: str
) -> str:
    """
    Extract text from an image using PaddleOCR.
    """

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

    ocr = get_ocr()

    print(
        f"Running OCR on: {path.name}"
    )

    try:

        result = ocr.predict(
            str(path)
        )

    except Exception as error:

        raise RuntimeError(
            f"PaddleOCR failed: {str(error)}"
        ) from error

    texts = _extract_text_from_result(
        result
    )

    extracted_text = "\n".join(
        texts
    )

    print(
        f"OCR extracted {len(extracted_text):,} characters."
    )

    return extracted_text


# Alias for multi-engine orchestrator
ocr_image = extract_text_from_image


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "PaddleOCR service test."
    )

    print(
        "Use extract_text_from_image() "
        "with an actual image path."
    )