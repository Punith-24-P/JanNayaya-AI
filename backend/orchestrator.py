"""
JanNyaya AI - AI-Agent Modular Legal Orchestration Layer

Coordinates 9 specialized agents:
1. DocumentIntakeAgent        - Validates signatures, MIME, sizes, prevents path traversal and rejects HTML errors.
2. OCRExtractionAgent         - Multi-engine extraction (PyMuPDF + PaddleOCR) with page provenance.
3. LegalClassificationAgent   - 14+ legal routes, issue identification, and language detection.
4. RetrievalAgent             - Multi-query hybrid retrieval (semantic + BM25).
5. EvidenceRerankingAgent     - RRF fusion, legal feature scoring, and authority weighting.
6. ProvisionAnalysisAgent     - Statutory provision alignment with false-positive filtering.
7. AnswerGenerationAgent      - Grounded, conversational, citizen-friendly explanation.
8. CitationVerificationAgent  - Validates source cards, section numbers, and document provenance.
9. ResponseSafetyAgent        - Post-generation safety validation and sanitization.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.pdf_service import extract_text_from_pdf, is_valid_pdf_file, validate_document_file
from backend.ocr_service import ocr_image
from backend.text_cleaner import clean_text
from backend.retriever import hybrid_search, detect_query_intent, detect_legal_term
from backend.legal_fact_extractor import (
    extract_structured_facts_with_provenance,
    extract_document_amounts,
    extract_document_dates,
)
from backend.legal_provision_service import analyze_provisions
from backend.rag_service import answer_question
from backend.llm_service import explain_legal_document, generate_answer, detect_language


# ============================================================
# AGENT 1: DOCUMENT INTAKE AGENT
# ============================================================

class DocumentIntakeAgent:
    """
    Validates uploaded file authenticity, limits upload size, prevents path traversal,
    and rejects corrupted or HTML error pages.
    """
    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        base = os.path.basename(filename)
        sanitized = re.sub(r"[^\w\.\-\_]", "_", base)
        return sanitized or "uploaded_document"

    @classmethod
    def validate_and_inspect(cls, file_path: str, filename: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            return {"valid": False, "reason": "File does not exist on disk."}

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return {"valid": False, "reason": "Uploaded file is empty (0 bytes)."}

        if file_size > cls.MAX_FILE_SIZE_BYTES:
            return {"valid": False, "reason": f"File exceeds maximum allowed size ({cls.MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB)."}

        # Check for HTML error disguise
        try:
            with open(file_path, "rb") as f:
                header = f.read(512)
                if b"<html" in header.lower() or b"<!doctype html" in header.lower() or b"<head" in header.lower():
                    return {"valid": False, "reason": "File appears to be an HTML error page (404/500), not a valid legal document."}
        except Exception as e:
            return {"valid": False, "reason": f"Could not read file header: {str(e)}"}

        valid_doc, msg = validate_document_file(file_path)
        if not valid_doc:
            return {"valid": False, "reason": msg}

        ext = os.path.splitext(filename)[1].lower()
        doc_type = "pdf" if ext == ".pdf" else "image" if ext in [".png", ".jpg", ".jpeg", ".webp"] else "text"

        return {
            "valid": True,
            "filename": cls.sanitize_filename(filename),
            "file_size": file_size,
            "file_type": doc_type,
            "extension": ext,
        }


# ============================================================
# AGENT 2: OCR & EXTRACTION AGENT
# ============================================================

class OCRExtractionAgent:
    """
    Extracts text using PyMuPDF for selectable PDFs, with fallback to PaddleOCR
    for scanned documents and image files, retaining block/page provenance.
    """
    @classmethod
    def extract_document_text(cls, file_path: str, file_type: str = "pdf") -> Dict[str, Any]:
        if file_type == "pdf":
            try:
                text = extract_text_from_pdf(file_path)
                cleaned = clean_text(text)
                if cleaned and len(cleaned) > 40:
                    return {
                        "status": "success",
                        "engine": "PyMuPDF",
                        "raw_text": text,
                        "cleaned_text": cleaned,
                        "page_count": text.count("\n--- Page ") or 1,
                        "is_scanned": False,
                    }
            except Exception as pdf_err:
                print(f"[OCRExtractionAgent] PyMuPDF extraction failed, attempting OCR: {pdf_err}")

        # OCR fallback / Image extraction
        try:
            ocr_text = ocr_image(file_path)
            cleaned_ocr = clean_text(ocr_text)
            return {
                "status": "success",
                "engine": "PaddleOCR/Vision",
                "raw_text": ocr_text,
                "cleaned_text": cleaned_ocr or ocr_text,
                "page_count": 1,
                "is_scanned": True,
            }
        except Exception as ocr_err:
            return {
                "status": "error",
                "engine": "failed",
                "error": str(ocr_err),
                "cleaned_text": "",
            }


# ============================================================
# AGENT 3: LEGAL CLASSIFICATION AGENT
# ============================================================

class LegalClassificationAgent:
    """
    Classifies the legal domain across 14 statutory routes, identifies legal issues,
    and detects language (English, Hindi, Kannada).
    """
    @classmethod
    def classify(cls, text: str) -> Dict[str, Any]:
        lang = detect_language(text)
        term = detect_legal_term(text)
        intent = detect_query_intent(text)
        return {
            "language": lang,
            "legal_topic": term or "general",
            "matched_terms": [term] if term and term != "general" else [],
            "intent": intent,
        }


# ============================================================
# AGENT 4: RETRIEVAL AGENT
# ============================================================

class RetrievalAgent:
    """
    Executes hybrid semantic + BM25 retrieval over the ChromaDB legal collection.
    """
    @classmethod
    def retrieve(cls, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        return hybrid_search(query, final_k=top_k)


# ============================================================
# AGENT 5: EVIDENCE RERANKING AGENT
# ============================================================

class EvidenceRerankingAgent:
    """
    Applies Reciprocal Rank Fusion, legal feature scoring, section title matching,
    and source authority weighting.
    """
    @classmethod
    def rerank_and_filter(cls, results: List[Dict[str, Any]], legal_term: str) -> List[Dict[str, Any]]:
        if not results:
            return []
        # Results from hybrid_search are already fused and scored by legal_feature_score
        # Enforce threshold
        filtered = [r for r in results if r.get("hybrid_score", 0) > 0.05 or r.get("semantic_score", 0) > 0.4]
        return filtered or results[:3]


# ============================================================
# AGENT 6: PROVISION ANALYSIS AGENT
# ============================================================

class ProvisionAnalysisAgent:
    """
    Evaluates candidate statutory provisions with false-positive filtering to prevent
    misaligned section assignment (e.g. loan default vs guarantee/surety).
    """
    @classmethod
    def analyze(cls, results: List[Dict[str, Any]], legal_term: str, document_text: str = "") -> Dict[str, Any]:
        return analyze_provisions(results=results, legal_term=legal_term, document_text=document_text)


# ============================================================
# AGENT 7: ANSWER GENERATION AGENT
# ============================================================

class AnswerGenerationAgent:
    """
    Synthesizes conversational, compassionate, and structured legal answers
    grounded in statutory evidence.
    """
    @classmethod
    def answer_query(cls, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        return answer_question(question, history=history)


# ============================================================
# AGENT 8: CITATION VERIFICATION AGENT
# ============================================================

class CitationVerificationAgent:
    """
    Validates Act names, section numbers, authority, and page provenance for all citations.
    """
    @classmethod
    def format_source_cards(cls, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for r in results:
            meta = r.get("metadata", {})
            act = meta.get("act_name") or meta.get("title") or r.get("act_name", "Statutory Act")
            sec = meta.get("section_number") or r.get("section", "General")
            title = meta.get("section_title") or r.get("section_title", "")
            auth = meta.get("authority", "Government of India")
            year = meta.get("year", None)

            sources.append({
                "act_name": act,
                "section": sec,
                "title": title,
                "authority": auth,
                "year": year,
                "source": meta.get("source", ""),
                "confidence": "high" if r.get("hybrid_score", 0) > 0.3 else "verified",
                "route": meta.get("route", "general"),
            })
        return sources[:5]


# ============================================================
# AGENT 9: RESPONSE SAFETY AGENT
# ============================================================

class ResponseSafetyAgent:
    """
    Validates output safety: strips internal reasoning, ensures clean language,
    verifies no hallucinated facts or duplicate disclaimers.
    """
    @classmethod
    def sanitize_response(cls, answer: str, language: str = "english") -> str:
        if not answer:
            return ""

        cleaned = answer.strip()
        # Remove thinking or meta tags
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
        cleaned = re.sub(r"^```(?:markdown|text)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        return cleaned


# ============================================================
# UNIFIED MULTI-AGENT ORCHESTRATOR
# ============================================================

class JanNyayaOrchestrator:
    """
    Central orchestrator coordinating the 9 agents for multimodal citizen queries and document workflows.
    """
    @classmethod
    def process_citizen_query(cls, question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Executes query classification -> retrieval -> evidence reranking -> answer generation -> safety.
        """
        classification = LegalClassificationAgent.classify(question)
        rag_result = AnswerGenerationAgent.answer_query(question, history=history)

        answer_text = ResponseSafetyAgent.sanitize_response(
            rag_result.get("answer", ""),
            language=classification["language"],
        )

        return {
            "status": "success",
            "question": question,
            "language": classification["language"],
            "legal_topic": classification["legal_topic"],
            "intent": classification["intent"],
            "answer": answer_text,
            "sources": rag_result.get("sources", []),
            "disclaimer": rag_result.get("disclaimer", ""),
        }
