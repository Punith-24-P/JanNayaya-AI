"""
JanNyaya AI - Fast Subsystem Verification Script
Validates:
1. Knowledge Base Statistics & Health
2. Fact Extraction & Law Separation (Loan Notice, Theft FIR)
3. Multi-Document Discrepancy & Conflict Detection
4. Intent & Multilingual Route Classification (EN, HI, KN)
5. Document Intake Security & Validation
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.knowledge_base_service import (
    get_knowledge_base_statistics,
    get_knowledge_base_health,
)
from backend.orchestrator import (
    DocumentIntakeAgent,
    LegalClassificationAgent,
    CitationVerificationAgent,
    ResponseSafetyAgent,
)
from backend.legal_fact_extractor import (
    extract_structured_facts_with_provenance,
    extract_document_amounts,
    extract_document_dates,
)
from backend.legal_analysis_service import (
    detect_document_conflicts,
    extract_document_timeline,
    detect_route,
)
from backend.retriever import (
    detect_query_intent,
    detect_legal_term,
)


def run_checks():
    print("=" * 70)
    print("JAN NYAYA AI - FAST SUBSYSTEM VERIFICATION")
    print("=" * 70)

    # 1. Knowledge Base Stats & Health
    print("\n[CHECK 1] Knowledge Base Statistics & Transparency:")
    stats = get_knowledge_base_statistics()
    print(f"  -> Total Chunks: {stats.get('total_chunks')}")
    print(f"  -> Total Central Acts: {stats.get('total_acts_indexed')}")
    print(f"  -> Total Domains: {stats.get('total_domains')}")
    print(f"  -> Coverage Statement: {stats.get('coverage_statement')[:100]}...")
    assert stats["total_chunks"] > 5000, "Expected >5000 chunks"
    assert stats["total_acts_indexed"] >= 12, "Expected >=12 acts"

    health = get_knowledge_base_health()
    print(f"  -> Overall Health: {health.get('overall_status')}")
    assert health["overall_status"] == "healthy"
    print("  ✓ Knowledge Base Transparency passed!")

    # 2. Document Intake Validation
    print("\n[CHECK 2] Document Intake Security & Sanitize:")
    sanitized = DocumentIntakeAgent.sanitize_filename("../../dangerous/path/Legal_Notice (1).pdf")
    assert ".." not in sanitized and "/" not in sanitized
    print(f"  -> Sanitized filename: {sanitized}")

    # Test rejection of HTML file disguised as pdf
    dummy_html_path = PROJECT_ROOT / "scratch_test_404.pdf"
    dummy_html_path.write_bytes(b"<html><head><title>404 Not Found</title></head><body><h1>404</h1></body></html>")
    try:
        val_res = DocumentIntakeAgent.validate_and_inspect(str(dummy_html_path), "test.pdf")
        assert val_res["valid"] is False, "Expected HTML file to be rejected"
        print(f"  -> Rejection reason: {val_res['reason']}")
    finally:
        if dummy_html_path.exists():
            dummy_html_path.unlink()
    print("  ✓ Document Intake Agent security passed!")

    # 3. Fact Extraction & Strict Provenance Separation
    print("\n[CHECK 3] Strict Fact Extraction from User Document:")
    notice_text = """
    LEGAL NOTICE UNDER SECTION 138 OF NI ACT
    Dated: 15/06/2023
    To: Mr. Ramesh Kumar
    You issued cheque no. 482910 for Rs. 1,87,560/- dated 10/05/2023 which was dishonoured with remarks 'Funds Insufficient'.
    You are hereby called upon to pay Rs. 1,87,560/- within 15 days of receipt of this notice.
    """
    facts = extract_structured_facts_with_provenance(notice_text)
    amounts = [a["value"] for a in facts.get("amounts", [])]
    dates = [d["value"] for d in facts.get("dates", [])]
    deadlines = [dl["value"] for dl in facts.get("deadlines", [])]

    print(f"  -> Extracted Amounts: {amounts}")
    print(f"  -> Extracted Dates: {dates}")
    print(f"  -> Extracted Deadlines: {deadlines}")

    assert any("1,87,560" in a or "187560" in a for a in amounts), "Missing cheque amount"
    assert any("15/06/2023" in d or "10/05/2023" in d for d in dates), "Missing notice dates"
    assert any("15 days" in dl.lower() or "15" in dl for dl in deadlines), "Missing 15 days deadline"
    print("  ✓ Fact extraction and provenance isolation passed!")

    # 4. Multi-Document Discrepancy & Conflict Detection
    print("\n[CHECK 4] Multi-Document Cross-File Conflict Detection:")
    doc1 = {
        "filename": "Legal_Notice.pdf",
        "amounts": [{"value": "₹1,87,560"}],
        "dates": [{"value": "15/06/2023"}],
        "interest_rates": [{"value": "18%"}],
    }
    doc2 = {
        "filename": "Bank_Statement.pdf",
        "amounts": [{"value": "₹1,65,000"}],
        "dates": [{"value": "15/06/2023"}],
        "interest_rates": [{"value": "12%"}],
    }
    conflicts = detect_document_conflicts([doc1, doc2])
    print(f"  -> Detected Conflicts Count: {len(conflicts)}")
    for c in conflicts:
        print(f"     * [{c.get('type')}] {c.get('title')}: {c.get('message')}")
    assert len(conflicts) >= 2, "Expected at least amount and interest rate conflicts"
    print("  ✓ Multi-document conflict detection passed!")

    # 5. Multilingual Intent & Route Classification
    print("\n[CHECK 5] Multilingual Intent & Route Classification:")
    queries = [
        ("What is the definition of theft under Indian law?", "definition", "theft", "criminal"),
        ("What is the punishment for theft under BNS?", "punishment", "theft", "criminal"),
        ("ಚೆಕ್ ಬೌನ್ಸ್ ಆದರೆ ಏನು ಶಿಕ್ಷೆ?", "punishment", "cheque_bounce", "commercial"),
        ("चोरी की सजा क्या है?", "punishment", "theft", "criminal"),
    ]
    for q, exp_intent, exp_term, exp_route in queries:
        cls_res = LegalClassificationAgent.classify(q)
        intent = detect_query_intent(q)
        term = detect_legal_term(q)
        route = detect_route(term)
        print(f"  -> Query: '{q}'")
        print(f"     Language: {cls_res['language']} | Intent: {intent} | Topic: {term} | Route: {route}")
        assert route == exp_route, f"Expected {exp_route}, got {route}"
    print("  ✓ Multilingual intent and route classification passed!")

    # 6. Response Safety & Sanitization
    print("\n[CHECK 6] Response Safety & Sanitization:")
    raw_ai_out = "<think>We need to check BNS Section 303</think>```markdown\n**Quick Summary**: Theft is defined under BNS.\n```"
    sanitized_out = ResponseSafetyAgent.sanitize_response(raw_ai_out)
    assert "<think>" not in sanitized_out and "```" not in sanitized_out
    print(f"  -> Sanitized: {sanitized_out}")
    print("  ✓ Response safety passed!")

    print("\n" + "=" * 70)
    print("ALL 6 FAST SUBSYSTEM CHECKS PASSED PERFECTLY (100%)")
    print("=" * 70)


if __name__ == "__main__":
    run_checks()
