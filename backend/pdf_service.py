import fitz
import numpy as np

from backend.ocr_service import get_ocr


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF.

    For normal PDFs:
        Uses PyMuPDF text extraction.

    For scanned/image-only pages:
        Converts the page to an image and sends it to PaddleOCR.
    """

    document = fitz.open(file_path)

    all_text = []

    # Initialize OCR only once
    ocr = get_ocr()

    try:

        for page_number, page in enumerate(document):

            # -------------------------------------------------
            # 1. Try normal selectable PDF text first
            # -------------------------------------------------

            text = page.get_text("text").strip()

            if text:
                all_text.append(
                    f"--- Page {page_number + 1} ---\n{text}"
                )

                continue

            # -------------------------------------------------
            # 2. No selectable text -> run OCR
            # -------------------------------------------------

            print(
                f"Page {page_number + 1}: "
                "No selectable text found. Running OCR..."
            )

            # Render PDF page as an image
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            # -------------------------------------------------
            # 3. Convert PNG bytes to NumPy image
            # -------------------------------------------------

            image = np.frombuffer(
                pixmap.tobytes("png"),
                dtype=np.uint8
            )

            # -------------------------------------------------
            # 4. Run PaddleOCR
            # -------------------------------------------------

            result = ocr.predict(image)

            page_text = []

            # -------------------------------------------------
            # 5. Extract recognized text
            # -------------------------------------------------

            for result_page in result:

                data = result_page.json

                if isinstance(data, str):
                    import json
                    data = json.loads(data)

                page_data = data.get("res", data)

                texts = page_data.get(
                    "rec_texts",
                    []
                )

                page_text.extend(texts)

            # -------------------------------------------------
            # 6. Save OCR result
            # -------------------------------------------------

            if page_text:

                all_text.append(
                    f"--- Page {page_number + 1} ---\n"
                    + "\n".join(page_text)
                )

            else:

                print(
                    f"Page {page_number + 1}: "
                    "OCR completed but no text was detected."
                )

    finally:

        document.close()

    # ---------------------------------------------------------
    # Return complete PDF text
    # ---------------------------------------------------------

    return "\n\n".join(all_text)
