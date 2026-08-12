from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from pydantic import BaseModel

from backend.ocr_service import extract_text_from_image
from backend.pdf_service import extract_text_from_pdf
from backend.text_cleaner import clean_text
from backend.rag_service import answer_question
from backend.ingest import ingest_pdf
from backend.vector_store import get_collection_count


app = FastAPI(
    title="JanNyaya AI",
    description="Multimodal Intelligent Legal Assistance System for Citizens",
    version="0.1.0",
)


class QuestionRequest(BaseModel):
    question: str


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "project": "JanNyaya AI",
        "status": "running",
        "message": "Welcome to JanNyaya AI"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "knowledge_base_chunks": get_collection_count()
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    extension = Path(file.filename).suffix.lower()

    allowed_extensions = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Supported files: PDF, JPG, JPEG, PNG and WEBP."
        )

    file_path = UPLOAD_DIR / file.filename

    try:

        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # PDF processing
        if extension == ".pdf":

            result = ingest_pdf(str(file_path))

            return {
                "status": "success",
                "filename": file.filename,
                "message": "Document uploaded and added to the legal knowledge base.",
                "chunks": result["chunks"],
                "characters": result["characters"]
            }

        # Image processing
        else:

            raw_text = extract_text_from_image(str(file_path))

            text = clean_text(raw_text)

            return {
                "status": "success",
                "filename": file.filename,
                "message": "Image uploaded and OCR text extracted.",
                "characters": len(text),
                "text": text
            }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(error)}"
        )


@app.post("/ocr")
async def ocr_document(file: UploadFile = File(...)):

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Currently OCR supports JPG, JPEG, PNG and WEBP images."
        )

    file_path = UPLOAD_DIR / file.filename

    try:

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_text_from_image(str(file_path))

        text = clean_text(raw_text)

        return {
            "status": "success",
            "filename": file.filename,
            "text": text
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(error)}"
        )


@app.post("/ocr-pdf")
async def ocr_pdf_document(file: UploadFile = File(...)):

    extension = Path(file.filename).suffix.lower()

    if extension != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    file_path = UPLOAD_DIR / file.filename

    try:

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_text = extract_text_from_pdf(str(file_path))

        text = clean_text(raw_text)

        return {
            "status": "success",
            "filename": file.filename,
            "text": text
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"PDF OCR processing failed: {str(error)}"
        )


@app.post("/ask")
async def ask_question(request: QuestionRequest):

    try:

        result = answer_question(request.question)

        return {
            "status": "success",
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(error)}"
        )