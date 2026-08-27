from pathlib import Path
from typing import Any
import json

# ============================================================
# GLOBAL OCR MODEL (LAZY LOADED)
# ============================================================

_ocr = None


# ============================================================
# GET OCR MODEL
# ============================================================

def get_ocr() -> Any:
    """
    Initialize PaddleOCR lazily only when requested.
    Reuses instance for subsequent requests.
    """
    global _ocr

    if _ocr is None:
        try:
            print("=" * 60)
            print("Initializing PaddleOCR...")
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(lang="en")
            print("PaddleOCR initialized successfully.")
        except Exception as e:
            print(f"PaddleOCR fallback notice: {e}")
            _ocr = None

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


def _extract_text_with_groq_vision(image_path: Path) -> str:
    """
    Extract legal text from image using Groq Vision API.
    Zero-RAM footprint, supports English, Hindi, and Kannada.
    """
    import os
    import base64
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Image uploaded: Unable to extract text (GROQ_API_KEY is not set)."

    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        ext = image_path.suffix.lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        media_type = f"image/{ext}" if ext in ["jpeg", "png", "webp", "gif"] else "image/jpeg"

        client = Groq(api_key=api_key, timeout=45.0)

        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an expert Indian Legal OCR engine. Extract ALL text verbatim from this legal document/notice/case image. "
                                "Preserve section numbers, dates, party names, amounts, and legal terms accurately. "
                                "Extract text in its original language/script (English, Kannada, or Hindi). "
                                "Do NOT add conversational commentary, output only the extracted text."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=2500,
        )

        extracted = response.choices[0].message.content.strip()
        print(f"Groq Vision OCR extracted {len(extracted):,} characters from {image_path.name}.")
        return extracted
    except Exception as e:
        print(f"Groq Vision OCR notice: {e}")
        return ""


# ============================================================
# IMAGE OCR
# ============================================================

def extract_text_from_image(
    file_path: str
) -> str:
    """
    Extract text from an image using PaddleOCR or high-accuracy Groq Vision fallback.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Provided path is not a file: {file_path}")

    print(f"Running OCR on: {path.name}")

    ocr = get_ocr()

    if ocr is not None:
        try:
            result = ocr.predict(str(path))
            texts = _extract_text_from_result(result)
            extracted_text = "\n".join(texts).strip()
            if extracted_text:
                print(f"PaddleOCR extracted {len(extracted_text):,} characters.")
                return extracted_text
        except Exception as error:
            print(f"PaddleOCR warning: {error}. Falling back to Vision OCR.")

    # High-accuracy zero-RAM Cloud Vision OCR Fallback
    vision_text = _extract_text_with_groq_vision(path)
    if vision_text:
        return vision_text

    return f"[Document {path.name}: Image attached for legal analysis]"


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