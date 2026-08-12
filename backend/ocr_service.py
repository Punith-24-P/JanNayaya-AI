from pathlib import Path
from paddleocr import PaddleOCR

_ocr = None


def get_ocr():
    global _ocr

    if _ocr is None:
        print("Initializing PaddleOCR...")

        _ocr = PaddleOCR(
            lang="en"
        )

        print("PaddleOCR initialized successfully.")

    return _ocr


def extract_text_from_image(file_path: str) -> str:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    ocr = get_ocr()

    result = ocr.predict(str(path))

    extracted_text = []

    for page in result:

        data = page.json

        if isinstance(data, str):
            import json
            data = json.loads(data)

        page_data = data.get("res", data)

        texts = page_data.get("rec_texts", [])

        extracted_text.extend(texts)

    return "\n".join(extracted_text)