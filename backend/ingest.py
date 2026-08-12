from backend.pdf_service import extract_text_from_pdf
from backend.text_cleaner import clean_text
from backend.chunker import chunk_text
from backend.embedding_service import create_embeddings
from backend.vector_store import collection


def ingest_pdf(file_path: str):
    print("1. Extracting text...")

    text = extract_text_from_pdf(file_path)

    print(f"Extracted characters: {len(text)}")

    print("2. Cleaning text...")

    cleaned_text = clean_text(text)

    print(f"Cleaned characters: {len(cleaned_text)}")

    print("3. Creating chunks...")

    chunks = chunk_text(
        cleaned_text,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Number of chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created.")

    print("4. Creating embeddings...")

    embeddings = create_embeddings(chunks)

    print(f"Created embeddings: {len(embeddings)}")

    print("5. Storing in ChromaDB...")

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    print("6. Ingestion completed successfully!")

    return {
        "chunks": len(chunks),
        "characters": len(cleaned_text)
    }