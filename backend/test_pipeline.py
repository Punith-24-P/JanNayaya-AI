from backend.pdf_service import extract_text_from_pdf
from backend.text_cleaner import clean_text
from backend.chunker import chunk_text

PDF_PATH = "uploads/SYNOPSIS.vtu0.pdf"

print("1. Extracting text from PDF...")

raw_text = extract_text_from_pdf(PDF_PATH)

print("Extracted characters:", len(raw_text))

print("\n2. Cleaning text...")

cleaned_text = clean_text(raw_text)

print("Cleaned characters:", len(cleaned_text))

print("\n3. Creating chunks...")

chunks = chunk_text(
    cleaned_text,
    chunk_size=1000,
    chunk_overlap=200
)

print("Number of chunks:", len(chunks))

print("\n4. First chunk:")
print("--------------------------------")
print(chunks[0] if chunks else "NO CHUNKS CREATED")
print("--------------------------------")

print("\nPipeline completed successfully!")
