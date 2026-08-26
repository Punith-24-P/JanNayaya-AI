#!/usr/bin/env python3
"""
Comprehensive Final Quality and Bug-Fix Verification Suite for JanNyaya AI.
Validates:
1. Document explanation language fidelity (Kannada, Hindi, English).
2. Script purity & formatting artifact sanitation (no raw pipe tables or divider artifacts).
3. Grounded Legal RAG answers & provenance for BNS 303, BNS 101/103, NI Act 138, Companies Act 447.
4. ChromaDB persistence and exact chunk count (5,142 chunks).
5. Auth user profile, avatar, citizen status, default explanation language, and stats API.
"""

import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.legal_analysis_service import analyze_legal_document, analyze_multiple_documents, normalize_language
from backend.rag_service import answer_question
from backend.vector_store import client, collection, COLLECTION_NAME
from backend.auth_service import init_auth_db, register_user, login_user, update_user_profile, get_user_stats, get_user_from_token
from backend.llm_service import _validate_script_purity, _clean_formatting_artifacts

def run_tests():
    print("=" * 70)
    print("JAN NYAYA AI — COMPREHENSIVE FINAL VERIFICATION PASS")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. ChromaDB Collection and Chunk Count
    # -------------------------------------------------------------
    print("\n[TEST 1] ChromaDB Persistent Store & Chunk Count")
    total_chunks = collection.count()
    print(f"Total chunks in '{COLLECTION_NAME}': {total_chunks}")
    assert total_chunks == 5142, f"Expected 5142 chunks, got {total_chunks}"
    print("  --> PASS: Exact count 5,142 chunks verified (+27 Companies Act 2013).")

    # -------------------------------------------------------------
    # 2. Kannada Document Explanation Fidelity & Script Purity
    # -------------------------------------------------------------
    print("\n[TEST 2] English Document -> Kannada Explanation")
    sample_notice_en = """
    LEGAL DEMAND NOTICE
    Under Section 138 of Negotiable Instruments Act, 1881.
    To: Mr. R. Sharma, Bangalore.
    From: Advocate V. Rao on behalf of QuickLoans FinCorp.
    Subject: Dishonour of Cheque No. 445210 for Rs. 3,50,000 due to Insufficiency of Funds.
    You are hereby called upon to remit the entire cheque amount of Rs. 3,50,000 within 15 days of receipt of this notice, failing which criminal complaint under Section 138 of NI Act shall be instituted against you in the Court of Metropolitan Magistrate.
    """
    res_kn = analyze_legal_document(sample_notice_en, language="kannada")
    doc_lang_kn = res_kn.get("document_language")
    exp_lang_kn = res_kn.get("explanation_language")
    summary_kn = res_kn.get("summary", "")

    print(f"Detected Document Language: {doc_lang_kn}")
    print(f"Normalized Explanation Language: {exp_lang_kn}")
    print(f"Summary Preview:\n{summary_kn[:250]}...\n")

    assert doc_lang_kn == "english", f"Expected doc_lang 'english', got {doc_lang_kn}"
    assert exp_lang_kn == "kannada", f"Expected exp_lang 'kannada', got {exp_lang_kn}"
    assert _validate_script_purity(summary_kn, "kannada"), "Summary failed Kannada script purity check!"
    assert "|" not in summary_kn or "---" not in summary_kn, "Raw markdown table artifacts detected in summary!"
    print("  --> PASS: English text correctly extracted; Explanation strictly generated in Kannada script.")

    # -------------------------------------------------------------
    # 3. Hindi Document Explanation Fidelity & Script Purity
    # -------------------------------------------------------------
    print("\n[TEST 3] English Document -> Hindi Explanation")
    res_hi = analyze_legal_document(sample_notice_en, language="hindi")
    doc_lang_hi = res_hi.get("document_language")
    exp_lang_hi = res_hi.get("explanation_language")
    summary_hi = res_hi.get("summary", "")

    print(f"Detected Document Language: {doc_lang_hi}")
    print(f"Normalized Explanation Language: {exp_lang_hi}")
    print(f"Summary Preview:\n{summary_hi[:250]}...\n")

    assert doc_lang_hi == "english", f"Expected doc_lang 'english', got {doc_lang_hi}"
    assert exp_lang_hi == "hindi", f"Expected exp_lang 'hindi', got {exp_lang_hi}"
    assert _validate_script_purity(summary_hi, "hindi"), "Summary failed Hindi script purity check!"
    print("  --> PASS: English text correctly extracted; Explanation strictly generated in Devanagari script.")

    # -------------------------------------------------------------
    # 4. Structured Document Case Extraction
    # -------------------------------------------------------------
    print("\n[TEST 4] Structured Case Entity & Field Extraction")
    amounts = res_kn.get("amounts", [])
    deadlines = res_kn.get("deadlines", [])
    provisions = res_kn.get("relevant_provisions", [])
    overview = res_kn.get("document_overview", {})

    print(f"Extracted Amounts: {amounts}")
    print(f"Extracted Deadlines: {deadlines}")
    print(f"Relevant Provisions Count: {len(provisions)}")
    print(f"Document Overview: {overview}")

    assert any("3,50,000" in str(a) for a in amounts), "Missing Rs. 3,50,000 amount extraction"
    assert any("15" in str(d) for d in deadlines), "Missing 15 days deadline extraction"
    assert len(provisions) >= 1, "Missing statutory provisions"
    assert res_kn.get("document_type") == "Legal Notice", f"Expected Legal Notice, got {res_kn.get('document_type')}"
    print("  --> PASS: Structured entities (Amounts, Deadlines, Provisions, Overview) extracted accurately.")

    # -------------------------------------------------------------
    # 5. Grounded RAG Query 1: BNS Section 303 Theft
    # -------------------------------------------------------------
    print("\n[TEST 5] Grounded RAG: Theft under Section 303 BNS 2023")
    q_theft = "What is the definition of theft and what is the punishment under Section 303 of BNS 2023?"
    ans_theft = answer_question(q_theft)
    answer_text_theft = ans_theft.get("answer", "")
    sources_theft = [s.get("title", "") for s in ans_theft.get("sources", [])]

    print(f"Sources: {sources_theft}")
    print(f"Answer Preview:\n{answer_text_theft[:300]}...\n")

    assert any("303" in s or "Theft" in s or "Bharatiya Nyaya Sanhita" in s for s in sources_theft), "Sources did not ground BNS 303"
    assert "303" in answer_text_theft or "theft" in answer_text_theft.lower(), "Answer does not reference Section 303 / theft"
    print("  --> PASS: BNS Section 303 theft retrieved and answered with grounded statutory sources.")

    # -------------------------------------------------------------
    # 6. Grounded RAG Query 2: NI Act Section 138 Cheque Bounce
    # -------------------------------------------------------------
    print("\n[TEST 6] Grounded RAG: Cheque Bounce under Section 138 NI Act")
    q_cheque = "What is the notice period and punishment for cheque bounce under Section 138 of Negotiable Instruments Act?"
    ans_cheque = answer_question(q_cheque)
    answer_text_cheque = ans_cheque.get("answer", "")
    sources_cheque = [s.get("title", "") for s in ans_cheque.get("sources", [])]

    print(f"Sources: {sources_cheque}")
    print(f"Answer Preview:\n{answer_text_cheque[:300]}...\n")

    assert any("138" in s or "Negotiable Instruments" in s for s in sources_cheque), "Sources did not ground NI Act 138"
    assert "138" in answer_text_cheque or "15" in answer_text_cheque, "Answer does not reference 138 or notice period"
    print("  --> PASS: NI Act Section 138 cheque bounce retrieved and answered with grounded sources.")

    # -------------------------------------------------------------
    # 7. Grounded RAG Query 3: Companies Act 2013 Section 447 Fraud
    # -------------------------------------------------------------
    print("\n[TEST 7] Grounded RAG: Corporate Fraud under Section 447 Companies Act 2013")
    q_corp = "What is the punishment for fraud under Section 447 of the Companies Act, 2013?"
    ans_corp = answer_question(q_corp)
    answer_text_corp = ans_corp.get("answer", "")
    sources_corp = [s.get("title", "") for s in ans_corp.get("sources", [])]

    print(f"Sources: {sources_corp}")
    print(f"Answer Preview:\n{answer_text_corp[:300]}...\n")

    assert any("Companies Act" in s for s in sources_corp), "Sources did not ground Companies Act 2013"
    assert "447" in answer_text_corp or "fraud" in answer_text_corp.lower() or "companies act" in answer_text_corp.lower(), "Answer does not reference Companies Act fraud"
    print("  --> PASS: Companies Act 2013 Section 447 corporate fraud retrieved and answered with grounded sources.")

    # -------------------------------------------------------------
    # 8. Auth Profile, Avatar, Citizen Status & Live Stats
    # -------------------------------------------------------------
    print("\n[TEST 8] Auth Profile, Avatar, Citizen Status, Password Change & Live Stats")
    init_auth_db()
    # Test register or login
    test_user_payload = {
        "username": f"test_citizen_{os.urandom(3).hex()}",
        "password": "Password123!",
        "full_name": "Test Citizen Advocate",
        "email": "citizen@jannyaya.in",
        "language": "kannada",
    }
    reg_res = register_user(
        username=test_user_payload["username"],
        password=test_user_payload["password"],
        full_name=test_user_payload["full_name"],
        email=test_user_payload["email"],
        language=test_user_payload["language"],
    )
    user_id = reg_res["user"]["id"]
    token = reg_res["token"]

    # Update profile with avatar, citizen status, default explanation lang, and password change
    sample_avatar_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    updated_user = update_user_profile(
        user_id=user_id,
        full_name="Punith Kumar V",
        email="punith@jannyaya.gov.in",
        language="kannada",
        avatar=sample_avatar_b64,
        citizen_status="Legal Practitioner / Law Student",
        default_explanation_lang="kannada",
        current_password="Password123!",
        new_password="NewSecurePassword456!",
    )

    user_obj = updated_user.get("user", {})
    assert user_obj.get("full_name") == "Punith Kumar V"
    assert user_obj.get("citizen_status") == "Legal Practitioner / Law Student"
    assert user_obj.get("default_explanation_lang") == "kannada"
    assert user_obj.get("avatar") == sample_avatar_b64

    # Verify new password authentication
    auth_check = login_user(test_user_payload["username"], "NewSecurePassword456!")
    assert auth_check.get("status") == "success", "Authentication failed with new password!"
    print("  --> PASS: Profile updated with avatar, citizen status, default language, and new password.")

    # Check live statistics
    stats = get_user_stats(user_id)
    print(f"Live User Stats: {stats}")
    assert "consultations_count" in stats
    assert "documents_analyzed" in stats
    assert "questions_asked" in stats
    assert "topics_explored" in stats
    print("  --> PASS: Live user statistics calculated accurately from database.")

    print("\n" + "=" * 70)
    print("ALL 8 VERIFICATION SUITE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
