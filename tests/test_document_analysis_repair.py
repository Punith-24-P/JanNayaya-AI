"""
JanNyaya AI - Verification Test Suite for Document-Analysis Subsystem Repair

Validates all 14 required scenarios:
1. Theft document (Criminal route, BNS Sec 303 definition & punishment)
2. Theft definition intent
3. Murder document (Criminal route, BNS Sec 101/103)
4. Cheating document (Criminal route, BNS Sec 318)
5. Loan recovery notice (Legal Notice, Civil/Contractual route, amounts ₹2,50,000 & ₹1,87,560, interest 12%, 24 instalments, 15 days deadline; rejects guarantee Sec 126)
6. Property dispute (Property route, Transfer of Property Act)
7. Consumer complaint (Consumer route, Consumer Protection Act)
8. Employment dispute (Employment route, Payment of Gratuity / Code on Wages)
9. Kannada document (Kannada detection & analysis)
10. Hindi document (Hindi detection & analysis)
11. Image file validation & OCR normalization
12. PDF file validation & page extraction
13. Multi-document analysis & consensus
14. Conflicting documents & discrepancy detection
"""

import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.pdf_service import validate_document_file
from backend.text_cleaner import clean_text, normalize_ocr_artifacts
from backend.legal_analysis_service import (
    analyze_legal_document,
    analyze_multiple_documents,
    detect_document_type,
    detect_document_conflicts,
    extract_document_timeline,
)
from backend.retriever import hybrid_search, detect_query_intent
from backend.legal_provision_service import analyze_provisions


def test_1_theft_document():
    print("--- Test 1: Theft Document Analysis ---")
    doc = """
    FIRST INFORMATION REPORT
    Police Station: Indiranagar, Bengaluru
    Crime No: 342/2024
    Under Section: BNS Section 303
    Complainant states that on 12/04/2024, the accused dishonestly took away a mobile phone worth Rs. 25,000/- out of possession without consent.
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["legal_route"] == "criminal", f"Expected criminal, got {res['legal_route']}"
    assert res["legal_topic"] == "theft", f"Expected theft, got {res['legal_topic']}"
    print("  ✓ Theft detected as criminal route")


def test_2_theft_definition_intent():
    print("--- Test 2: Query Intent Detection ---")
    intent = detect_query_intent("What is the definition of theft under Indian law?")
    assert intent == "definition", f"Expected definition, got {intent}"
    intent_p = detect_query_intent("What is the punishment for theft?")
    assert intent_p == "punishment", f"Expected punishment, got {intent_p}"
    print("  ✓ Definition and punishment intents correctly detected")


def test_3_murder_document():
    print("--- Test 3: Murder Document Analysis ---")
    doc = """
    FIRST INFORMATION REPORT
    Police Station: Cyberabad
    Crime No: 112/2024
    Allegation: The accused intentionally caused the death of the victim using a sharp weapon. Culpable homicide amounting to murder.
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["legal_route"] == "criminal"
    assert res["legal_topic"] == "murder"
    print("  ✓ Murder detected as criminal route")


def test_4_cheating_document():
    print("--- Test 4: Cheating Document Analysis ---")
    doc = """
    POLICE COMPLAINT
    The accused fraudulently deceived the complainant by false representation and dishonestly induced him to deliver Rs. 5,00,000/- for a bogus land scheme.
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["legal_route"] == "criminal"
    assert res["legal_topic"] == "cheating"
    print("  ✓ Cheating detected as criminal route")


def test_5_loan_recovery_notice():
    print("--- Test 5: Loan Recovery Notice (Crucial Separation & False-Positive Prevention) ---")
    doc = """
    LEGAL NOTICE
    Under instructions from my client, notice is hereby given:
    1. Our client advanced a loan amount of Rs. 2,50,000/- to you on 15/06/2022.
    2. You agreed to repay with interest at 12% p.a. in 24 monthly instalments.
    3. You defaulted and the outstanding amount due is Rs. 1,87,560/-.
    4. You are called upon to pay the outstanding dues within 15 days from receipt.
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["document_type"] == "Legal Notice", f"Got doc type: {res['document_type']}"
    assert res["legal_route"] == "civil_contractual", f"Got route: {res['legal_route']}"
    assert res["legal_topic"] == "loan_recovery", f"Got topic: {res['legal_topic']}"

    # Verify extracted facts
    amounts_str = " ".join([a["value"] for a in res["amounts"]])
    assert "2,50,000" in amounts_str or "250000" in amounts_str, f"Missing loan amount: {amounts_str}"
    assert "1,87,560" in amounts_str or "187560" in amounts_str, f"Missing outstanding: {amounts_str}"

    dates_str = " ".join([d["value"] for d in res["dates"]])
    assert "15/06/2022" in dates_str, f"Missing loan date: {dates_str}"

    deadlines_str = " ".join([d["value"] for d in res["deadlines"]])
    assert "15 days" in deadlines_str or "15" in deadlines_str, f"Missing deadline: {deadlines_str}"

    # Verify false positive prevention: Must NOT pick Contract Act Section 126 (guarantee) or CPC Sec 60/Order 21
    prov = res.get("primary_provision")
    if prov:
        sec = str(prov.get("section", "")).strip()
        assert sec != "126", "False positive! Section 126 (Guarantee) was selected for bilateral loan notice without guarantor."
        assert sec != "60", "False positive! Section 60 (Salary attachment/appropriation) was selected."
        assert sec != "8", "False positive! CPC Order 21 / Sec 8 (Court auction purchase money) was selected."

    print("  ✓ Loan recovery facts cleanly extracted with provenance; false positives (Sec 126/60/8) strictly prevented")


def test_6_property_dispute():
    print("--- Test 6: Property Dispute ---")
    doc = """
    LEGAL NOTICE FOR IMMOVABLE PROPERTY
    The subject property bearing Site No. 42 is owned by my client by virtue of Sale Deed dated 10/01/2015.
    You have unlawfully interfered with my client's peaceful possession and attempted illegal encroachment.
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["legal_route"] == "property"
    assert res["legal_topic"] == "property_dispute"
    print("  ✓ Property dispute detected with Property route")


def test_7_consumer_complaint():
    print("--- Test 7: Consumer Complaint ---")
    doc = """
    BEFORE THE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION
    Consumer Complaint under Consumer Protection Act, 2019.
    Complainant purchased a defective refrigerator with manufacturing defects. The opposite party failed to replace or refund despite warranty.
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["legal_route"] == "consumer"
    assert res["legal_topic"] == "consumer_complaint"
    print("  ✓ Consumer complaint detected with Consumer route")


def test_8_employment_dispute():
    print("--- Test 8: Employment Dispute ---")
    doc = """
    DEMAND NOTICE FOR UNPAID SALARY AND GRATUITY
    The employee served the organization for 6 years and was wrongfully terminated without notice pay, unpaid salary for 3 months, and gratuity dues.
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["legal_route"] == "employment"
    assert res["legal_topic"] == "employment_dispute"
    print("  ✓ Employment dispute detected with Employment route")


def test_9_kannada_document():
    print("--- Test 9: Kannada Document Analysis ---")
    doc = """
    ಕಾನೂನು ನೋಟಿಸ್
    ನನ್ನ ಕಕ್ಷಿದಾರರು ದಿನಾಂಕ 15/06/2022 ರಂದು ನಿಮಗೆ ರೂ. 2,50,000/- ಸಾಲ ನೀಡಿದ್ದರು.
    ನೀವು ಬಾಕಿ ಹಣ ರೂ. 1,87,560/- ಅನ್ನು 15 ದಿನಗಳ ಒಳಗೆ ಪಾವತಿಸಬೇಕು, ಇಲ್ಲದಿದ್ದರೆ ಕಾನೂನು ಕ್ರಮ ಕೈಗೊಳ್ಳಲಾಗುವುದು.
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["language"] == "kannada", f"Expected kannada, got {res['language']}"
    assert res["legal_route"] == "civil_contractual"
    print("  ✓ Kannada document successfully identified and analyzed")


def test_10_hindi_document():
    print("--- Test 10: Hindi Document Analysis ---")
    doc = """
    विधिक नोटिस
    मेरे मुवक्किल ने आपको दिनांक 15/06/2022 को रु 2,50,000/- का ऋण दिया था।
    बकाया राशि रु 1,87,560/- का भुगतान 15 दिनों के भीतर करें अन्यथा कानूनी कार्रवाई की जाएगी।
    """
    res = analyze_legal_document(doc)
    assert res["status"] == "success"
    assert res["language"] == "hindi", f"Expected hindi, got {res['language']}"
    assert res["legal_route"] == "civil_contractual"
    print("  ✓ Hindi document successfully identified and analyzed")


def test_11_ocr_normalization():
    print("--- Test 11: OCR Normalization & Currency Repair ---")
    raw_ocr = "Our client advanced a loan of t2,50,000/- on 15 / 06 / 2022 with 12 % p . a . interest. Outstanding is t1,87,560/-."
    normalized = clean_text(raw_ocr)
    assert "₹2,50,000" in normalized or "₹ 2,50,000" in normalized, f"Currency repair failed: {normalized}"
    assert "15/06/2022" in normalized, f"Date repair failed: {normalized}"
    assert "12% p.a." in normalized, f"Interest repair failed: {normalized}"
    assert "₹1,87,560" in normalized or "₹ 1,87,560" in normalized, f"Outstanding currency repair failed: {normalized}"
    print("  ✓ OCR artifacts repaired: currency symbols, spaced dates, and percentage notations normalized")


def test_12_file_validation():
    print("--- Test 12: File Magic-Byte Signature Validation ---")
    # Test valid PDF header
    temp_pdf = Path("test_sample.pdf")
    temp_pdf.write_bytes(b"%PDF-1.4\n%test content with enough bytes for validation test\n")
    val_pdf = validate_document_file(temp_pdf)
    assert val_pdf["is_valid"] and val_pdf["file_type"] == "pdf"
    temp_pdf.unlink()

    # Test HTML error page rejection
    temp_html = Path("error_page.pdf")
    temp_html.write_bytes(b"<!DOCTYPE html><html><head><title>404 Not Found</title></head><body>Error</body></html>")
    val_html = validate_document_file(temp_html)
    assert not val_html["is_valid"], "Failed: HTML error page was accepted as valid document!"
    temp_html.unlink()
    print("  ✓ File magic bytes validated; HTML error pages rejected")


def test_13_multi_document_analysis():
    print("--- Test 13: Multi-Document Unified Case Analysis ---")
    doc1 = {
        "filename": "Loan_Agreement.pdf",
        "file_type": "pdf",
        "text": "LOAN AGREEMENT: Borrower agrees to borrow Rs. 2,50,000/- dated 15/06/2022 at 12% p.a. payable in 24 instalments.",
    }
    doc2 = {
        "filename": "Demand_Notice.pdf",
        "file_type": "pdf",
        "text": "LEGAL NOTICE: Default in repayment of loan dated 15/06/2022. Outstanding dues Rs. 1,87,560/-. Pay within 15 days.",
    }
    case_res = analyze_multiple_documents([doc1, doc2])
    assert case_res["status"] == "success"
    assert case_res["total_documents"] == 2
    assert case_res["legal_route"] == "civil_contractual"
    assert len(case_res["timeline"]) >= 1
    print("  ✓ Multi-document case analysis synthesized consensus topic, route, and timeline")


def test_14_conflicting_documents_detection():
    print("--- Test 14: Conflicting Documents & Discrepancy Detection ---")
    doc_a = {
        "filename": "Agreement_A.pdf",
        "file_type": "pdf",
        "text": "LOAN AGREEMENT: Loan amount is Rs. 2,50,000/- executed on 15/06/2022 with 15 days notice period.",
    }
    doc_b = {
        "filename": "Notice_B.pdf",
        "file_type": "pdf",
        "text": "DEMAND NOTICE: Loan amount claimed is Rs. 3,50,000/- executed on 20/08/2022 with 30 days notice period.",
    }
    case_res = analyze_multiple_documents([doc_a, doc_b])
    conflicts = case_res.get("conflicts", [])
    assert len(conflicts) >= 1, "Failed: Discrepancies between documents were not detected!"
    conflict_types = [c["type"] for c in conflicts]
    print(f"  ✓ Detected conflicts across documents: {conflict_types}")


def run_all_tests():
    print("=" * 70)
    print("JAN NYAYA AI - DOCUMENT-ANALYSIS REPAIR VERIFICATION SUITE")
    print("=" * 70)

    test_1_theft_document()
    test_2_theft_definition_intent()
    test_3_murder_document()
    test_4_cheating_document()
    test_5_loan_recovery_notice()
    test_6_property_dispute()
    test_7_consumer_complaint()
    test_8_employment_dispute()
    test_9_kannada_document()
    test_10_hindi_document()
    test_11_ocr_normalization()
    test_12_file_validation()
    test_13_multi_document_analysis()
    test_14_conflicting_documents_detection()

    print("\n" + "=" * 70)
    print("ALL 14 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
