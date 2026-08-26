# JanNyaya AI

**Multimodal Intelligent Legal Assistance System for Citizens**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com/)
[![Groq Whisper](https://img.shields.io/badge/Groq-Whisper_Large_v3-purple.svg)](https://groq.com/)
[![Groq LLM](https://img.shields.io/badge/Groq-GPT_OSS_120B-red.svg)](https://groq.com/)

---

## 1. Project Overview

**JanNyaya AI** is a multimodal, multilingual Indian legal assistance and information system designed to bridge the accessibility gap between complex Indian statutory laws and ordinary citizens.

Citizens can interact with the system via:
1. **Text Queries** in English, Hindi (हिन्दी), and Kannada (ಕನ್ನಡ).
2. **Microphone Voice Input** using browser `MediaRecorder` and cloud Groq Whisper (`whisper-large-v3`).
3. **Single & Multi-Document Uploads** supporting PDFs, scanned documents, and images (`.pdf`, `.jpg`, `.jpeg`, `.png`, `.webp`).
4. **Automated Document Intelligence**: Optical Character Recognition (PaddleOCR) + direct PyMuPDF text extraction with `%PDF-` signature verification.
5. **Cross-Document Conflict Detection**: Compares amounts, dates, and deadlines across multiple files to identify discrepancies.
6. **Route-Aware Hybrid Legal RAG**: Multilingual E5 embeddings (`intfloat/multilingual-e5-small`) + BM25 lexical search with Reciprocal Rank Fusion (RRF) and route-specific reranking.
7. **Statutory Grounding & Explanations**: Extracts legal provisions into simple definitions, punishments/consequences, conditions, and practical next steps without hallucinating sections or procedures.

---

## 2. System Architecture

```text
                                 JAN NYAYA AI
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
        TEXT Q&A                  VOICE Q&A              DOCUMENT UPLOAD
   (EN / HI / KN)             (MediaRecorder)         (Single / Multi-File)
            │                         │                         │
            │                         ▼                         ▼
            │                   Groq Whisper             PyMuPDF / PaddleOCR
            │                 (whisper-large-v3)         (with %PDF validation)
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
                                Text Cleaner
                                      │
                                      ▼
                           Language & Route Analysis
                      (Criminal, Civil, Property, etc.)
                                      │
                                      ▼
                          Hybrid Legal Retriever
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
               Semantic Search                       BM25
             (multilingual-e5)                  (Lexical RRF)
                      └───────────────┬───────────────┘
                                      ▼
                                Legal Reranker
                                      │
                                      ▼
                            Legal Fact Extractor
                                      │
                                      ▼
                           Legal Provision Explainer
                                      │
                                      ▼
                           Precision LLM Generator
                            (openai/gpt-oss-120b)
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
              Citizen Explanation            Statutory Sources
            (Simple, Natural & Safe)      (Act, Section, Confidence)
```

---

## 3. Technology Stack

### Backend
- **Framework**: FastAPI, Uvicorn, Pydantic
- **Vector Database**: ChromaDB (Persistent local storage)
- **Embeddings**: `intfloat/multilingual-e5-small` (HuggingFace / Sentence Transformers)
- **Lexical Search**: Custom BM25 implementation with Reciprocal Rank Fusion (RRF)
- **OCR Engine**: PaddleOCR
- **PDF Extraction**: PyMuPDF (`fitz`) with header signature validation
- **ASR (Speech-to-Text)**: Groq Whisper (`whisper-large-v3`)
- **LLM**: Groq Cloud (`openai/gpt-oss-120b`)
- **Language Detection**: Unicode-range script detection (English, Devanagari, Kannada)

### Frontend
- **Framework**: React 18, Vite
- **Styling**: Modern, responsive CSS design with Indian legal identity
- **Voice Recording**: Web Audio API / `MediaRecorder`
- **Text-to-Speech**: Web Speech API (`SpeechSynthesisUtterance`) for `en-IN`, `hi-IN`, and `kn-IN`

---

## 4. Project Directory Structure

```text
JanNayaya-AI/
├── backend/
│   ├── main.py                     # FastAPI application endpoints
│   ├── retriever.py                # Hybrid RAG (Semantic E5 + BM25 + RRF)
│   ├── legal_analysis_service.py   # Single & multi-document case analysis
│   ├── legal_provision_service.py  # Structured statutory provision explainer
│   ├── legal_fact_extractor.py     # Deduplicated legal fact extraction
│   ├── llm_service.py              # Precision multilingual LLM generation
│   ├── speech_service.py           # Groq Whisper speech transcription
│   ├── pdf_service.py              # PyMuPDF extraction + OCR fallback
│   ├── ocr_service.py              # PaddleOCR image text extraction
│   ├── chunker.py                  # Section-aware legal text chunking
│   ├── embedding_service.py        # SentenceTransformer multilingual E5
│   ├── vector_store.py             # ChromaDB interface & persistence
│   ├── text_cleaner.py             # OCR & text cleaning / normalization
│   ├── legal_data_ingest.py        # Ingestion pipeline for legal corpus
│   └── rebuild_legal_database.py   # Clean database rebuild utility
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main application UI & state
│   │   ├── App.css                 # Responsive design & component styling
│   │   └── main.jsx                # React entrypoint
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── legal_data/                     # Authoritative statutory PDF corpus
│   ├── criminal/BNS/               # Bharatiya Nyaya Sanhita, 2023
│   ├── civil/
│   │   ├── Contract_Act/           # Indian Contract Act, 1872
│   │   ├── CPC/                    # Code of Civil Procedure, 1908
│   │   └── Limitation_Act/         # Limitation Act, 1963
│   ├── property/
│   ├── family/
│   ├── consumer/
│   └── employment/
│
├── uploads/                        # Upload storage directory
├── chroma_db/                      # Persistent ChromaDB vector database
├── .env.example                    # Environment configuration template
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive project documentation
```

---

## 5. Quick Start & Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ and npm
- Groq API Key ([console.groq.com](https://console.groq.com))

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/JanNyaya-AI.git
cd JanNayaya-AI
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure your keys:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=openai/gpt-oss-120b
SPEECH_MODEL=whisper-large-v3
LLM_MAX_TOKENS=650
```

### 3. Backend Setup
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run build
cd ..
```

### 5. Running the Application
**Start Backend Server:**
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
- API Root: `http://127.0.0.1:8000/`
- Interactive API Docs: `http://127.0.0.1:8000/docs`

**Start Frontend Development Server:**
```bash
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service root and feature summary |
| `GET` | `/health` | Health check and knowledge base chunk counter |
| `GET` | `/endpoints` | Overview of all available endpoints |
| `GET` | `/conversations` | List previous consultation sessions with metadata & search |
| `POST` | `/conversations` | Create a new consultation session |
| `GET` | `/conversations/{id}` | Retrieve full conversation with question-answer turns & sources |
| `POST` | `/conversations/{id}/messages` | Multi-turn conversational question with grounded statutory RAG |
| `DELETE` | `/conversations/{id}` | Delete consultation session and message history |
| `GET` | `/knowledge-base/stats` | Live ChromaDB statistics (5,115 chunks, 24 acts, 14 domains) |
| `GET` | `/knowledge-base/search` | Search bare acts by Act, Section, Legal topic, Keyword |
| `GET` | `/library/acts` | Catalog of verified Indian Bare Acts |
| `POST` | `/ask` | Text legal Q&A in English, Hindi, or Kannada |
| `POST` | `/speech-to-text` | Audio transcription via Groq Whisper (`.webm`, `.wav`, `.mp3`) |
| `POST` | `/upload` | Single legal document upload & analysis |
| `POST` | `/upload-multiple` | Multi-document upload, conflict detection & case analysis |
| `POST` | `/document/chat` | Interactive Q&A on uploaded document text |
| `POST` | `/analyze-timeline` | Case chronology and date milestones extractor |
| `POST` | `/analyze-text` | Direct text legal analysis and route classification |
| `POST` | `/ocr` | Raw OCR extraction from image files |
| `POST` | `/ocr-pdf` | Text extraction from PDF documents |

---

## 7. Testing & Verification

### Test Retrieval Pipeline
```bash
./venv/bin/python -c "
from backend.retriever import hybrid_search
import json
results = hybrid_search('What is the punishment for theft?', final_k=5)
print(json.dumps([{'section': r['metadata']['section_number'], 'title': r['metadata']['section_title'], 'score': r.get('legal_rerank_score', 0)} for r in results], indent=2))
"
```

### Test Multi-Document Case Analysis & Conflict Detection
```bash
./venv/bin/python -c "
from backend.legal_analysis_service import analyze_multiple_documents
docs = [
    {'filename': 'notice1.pdf', 'file_type': 'pdf', 'text': 'Loan of Rs. 2,50,000/- with 15 days deadline.'},
    {'filename': 'notice2.jpg', 'file_type': 'jpeg', 'text': 'Claim of Rs. 2,10,000/- with 30 days deadline.'}
]
res = analyze_multiple_documents(docs)
print('Consensus Route:', res['legal_route'])
print('Conflicts Detected:', len(res['conflicts']))
"
```

### Build Frontend
```bash
cd frontend && npm run build
```

---

## 8. Non-Negotiable Product Principles

- **No Legal Hallucinations**: JanNyaya strictly grounds explanations in verified retrieved statutory provisions.
- **Precision Nuances**: Distinguishes statutory mandates (`shall` vs `may`, `minimum` vs `maximum`, `and` vs `or`).
- **No Meta Jargon**: Answers do not expose internal retrieval mechanics (no "Based on BM25..." or "Detected intent...").
- **Language Parity**: Respects official statutory names while explaining natural legal outcomes in English, Hindi, and Kannada.
- **Clear Information vs Advice Distinction**: The system assists citizens with legal literacy and information without replacing professional legal representation.

---

## 9. Legal Disclaimer

> **Disclaimer**: *JanNyaya AI is an artificial intelligence research platform designed to provide accessible general legal information based on Indian statutory sources. It does not provide personalized legal advice, create an attorney-client relationship, or determine the final legal outcome of any matter. Citizens are advised to consult a qualified advocate for case-specific legal representation.*
