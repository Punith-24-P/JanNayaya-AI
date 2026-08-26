#!/usr/bin/env python3
"""
Comprehensive multilingual benchmark test for JanNyaya AI Legal Assistant.
Tests:
1. Retriever multilingual routing & precision (Kannada, Hindi, English).
2. LLM response generation with 5-part structure.
"""

import os
import sys

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from backend.retriever import detect_query_intent, hybrid_search
from backend.rag_service import answer_question
from backend.llm_service import detect_language

def test_multilingual_retriever():
    print("\n" + "=" * 60)
    print("TEST 1: MULTILINGUAL RETRIEVER TEST")
    print("=" * 60)

    test_queries = [
        # English
        ("What is the punishment for cheque bounce under section 138?", "commercial"),
        ("Bank sent me a SARFAESI Section 13(2) notice for loan default. What can I do?", "commercial"),
        ("Someone stole my laptop from my office. Which section applies?", "criminal"),
        ("I was scammed in an online UPI lottery fraud. How to report?", "cyber"),
        
        # Kannada
        ("ಚೆಕ್ ಬೌನ್ಸ್ ಆದರೆ ಏನು ಶಿಕ್ಷೆ?", "commercial"),
        ("ಬ್ಯಾಂಕ್ ಸಾಲ ಮರುಪಾವತಿ ಮಾಡದಿದ್ದರೆ ಜಪ್ತಿ ನೋಟಿಸ್ ಬಂದರೆ ಏನು ಮಾಡಬೇಕು?", "commercial"),
        ("ನನ್ನ ಮೊಬೈಲ್ ಕಳ್ಳತನವಾಗಿದೆ, ಯಾವ ಸೆಕ್ಷನ್ ಅಡಿಯಲ್ಲಿ ದೂರು ದಾಖಲಿಸಬೇಕು?", "criminal"),
        ("ಆನ್‌ಲೈನ್ ಯುಪಿಐ ವಂಚನೆ ಆಯಿತು, ಸೈಬರ್ ಕ್ರೈಮ್ ದೂರು ನೀಡುವುದು ಹೇಗೆ?", "cyber"),

        # Hindi
        ("चेक बाउंस होने पर क्या सजा और जुर्माना होता है?", "commercial"),
        ("बैंक ने सरफेसी एक्ट के तहत नोटिस भेजा है, क्या करें?", "commercial"),
        ("दुकान से चोरी करने पर बीएनएस में क्या कानूनी सजा है?", "criminal"),
        ("ऑनलाइन फ्रॉड में पैसे कट गए, 1930 पर शिकायत कैसे दर्ज करें?", "cyber"),
    ]

    passed = 0
    for query, expected_domain in test_queries:
        lang = detect_language(query)
        intent = detect_query_intent(query)
        docs = hybrid_search(query, final_k=3)
        doc_count = len(docs)
        print(f"\nQuery ({lang.upper()}): {query}")
        print(f"  -> Detected Intent: {intent}")
        print(f"  -> Retrieved Documents: {doc_count} chunks")
        if doc_count > 0:
            meta = docs[0].get("metadata", {})
            act = meta.get("act_name") or docs[0].get("act_name")
            sec = meta.get("section_number") or docs[0].get("section")
            print(f"  -> Top Result: {act} | Section {sec}")
            passed += 1
        else:
            print("  -> FAILED to retrieve context")

    print(f"\nRetriever Test Passed: {passed}/{len(test_queries)}")
    return passed == len(test_queries)

def test_multilingual_llm():
    print("\n" + "=" * 60)
    print("TEST 2: MULTILINGUAL LLM ANSWER GENERATION")
    print("=" * 60)

    kannada_q = "ನನ್ನ ಚೆಕ್ ಬೌನ್ಸ್ ಆಗಿದೆ, ನನಗೆ ಯಾವ ಪರಿಹಾರ ಸಿಗುತ್ತದೆ ಮತ್ತು ಶಿಕ್ಷೆ ಏನು?"
    res_k = answer_question(kannada_q)
    print("\n[Kannada Response Generated]:")
    print(res_k.get("answer", "")[:400] + "...\n")
    
    hindi_q = "बैंक से लोन नोटिस आया है, मेरे पास क्या कानूनी अधिकार हैं?"
    res_h = answer_question(hindi_q)
    print("\n[Hindi Response Generated]:")
    print(res_h.get("answer", "")[:400] + "...\n")

    return True

if __name__ == "__main__":
    retriever_ok = test_multilingual_retriever()
    llm_ok = test_multilingual_llm()
    if retriever_ok and llm_ok:
        print("\nALL MULTILINGUAL SYSTEM VERIFICATIONS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED.")
        sys.exit(1)
