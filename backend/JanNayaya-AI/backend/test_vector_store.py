from backend.vector_store import get_collection_count


count = get_collection_count()

print()
print("=" * 50)
print("CHROMADB TEST")
print("=" * 50)

print(f"Total chunks: {count}")

print("=" * 50)