# JanNyaya AI — Production Docker Container
FROM python:3.11-slim

# System dependencies for PyMuPDF, OCR & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/ ./backend/
COPY legal_data/ ./legal_data/
COPY chroma_db/ ./chroma_db/
COPY data/ ./data/

# Environment variables for memory & CPU efficiency
ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    MALLOC_TRIM_THRESHOLD_=100000 \
    LOW_MEMORY_MODE=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "python -m backend.main"]
