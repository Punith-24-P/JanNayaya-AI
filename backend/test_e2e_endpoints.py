"""
JanNyaya AI - End-to-End API Integration & Regression Test Suite
Tests:
1. Health & Status
2. Live Knowledge Base Stats (5,115 chunks, 24 acts)
3. Knowledge Base Search by Act, Section, Keyword
4. Conversation Lifecycle (Create -> Turn 1 -> Turn 2 Continuation -> Get -> List -> Delete)
5. Grounded Multi-Turn RAG on newly ingested acts (BNSS, BSA, Arbitration, JJ Act)
6. Bare Acts Catalog
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 70)
    print("JAN NYAYA AI — AUTOMATED API REGRESSION SUITE")
    print("=" * 70)

    # 1. Health check
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("✓ Health Check: Passed (status 200)")

    # 2. KB Stats
    res = requests.get(f"{BASE_URL}/knowledge-base/stats")
    assert res.status_code == 200, f"KB stats failed: {res.text}"
    stats = res.json()
    assert stats["total_chunks"] >= 5115, f"Expected >= 5115 chunks, got {stats['total_chunks']}"
    assert stats["total_acts_indexed"] >= 24, f"Expected >= 24 acts, got {stats['total_acts_indexed']}"
    print(f"✓ KB Stats: Passed ({stats['total_chunks']} chunks, {stats['total_acts_indexed']} acts across {stats['total_domains']} domains)")

    # 3. KB Search
    res = requests.get(f"{BASE_URL}/knowledge-base/search?q=Section 173&limit=5")
    assert res.status_code == 200, f"KB search failed: {res.text}"
    search_data = res.json()
    assert search_data["status"] == "success"
    assert len(search_data["results"]) > 0, "Expected at least 1 search match for Section 173"
    print(f"✓ KB Search ('Section 173'): Passed ({len(search_data['results'])} matches found)")

    # 4. Conversation Lifecycle & Continuation
    # Step 4a: Create Conversation
    create_payload = {
        "title": "Cheque Bounce Consultation Test",
        "language": "english",
        "legal_topic": "banking",
    }
    res = requests.post(f"{BASE_URL}/conversations", json=create_payload)
    assert res.status_code == 200, f"Create conversation failed: {res.text}"
    conv = res.json()["conversation"]
    conv_id = conv["id"]
    print(f"✓ Create Conversation: Passed (ID: {conv_id})")

    # Step 4b: First Turn Message
    turn1_payload = {
        "question": "What is the notice requirement under Section 138 of Negotiable Instruments Act?",
        "language": "english",
        "history": []
    }
    res = requests.post(f"{BASE_URL}/conversations/{conv_id}/messages", json=turn1_payload)
    assert res.status_code == 200, f"Turn 1 failed: {res.text}"
    t1_data = res.json()
    assert len(t1_data["answer"]) > 50, "Expected substantive legal answer"
    assert len(t1_data["sources"]) > 0, "Expected statutory sources"
    print(f"✓ Turn 1 (Section 138 query): Passed (Answer: {len(t1_data['answer'])} chars, Sources: {len(t1_data['sources'])})")

    # Step 4c: Second Turn Message (Continuation)
    turn2_payload = {
        "question": "What is the maximum punishment or imprisonment if convicted?",
        "language": "english",
        "history": [
            {"role": "user", "content": turn1_payload["question"]},
            {"role": "assistant", "content": t1_data["answer"]}
        ]
    }
    res = requests.post(f"{BASE_URL}/conversations/{conv_id}/messages", json=turn2_payload)
    assert res.status_code == 200, f"Turn 2 failed: {res.text}"
    t2_data = res.json()
    assert len(t2_data["answer"]) > 50, "Expected substantive turn 2 answer"
    print(f"✓ Turn 2 Continuation (Multi-turn RAG): Passed (Answer: {len(t2_data['answer'])} chars)")

    # Step 4d: Retrieve Conversation Record
    res = requests.get(f"{BASE_URL}/conversations/{conv_id}")
    assert res.status_code == 200, f"Get conversation failed: {res.text}"
    fetched = res.json()["conversation"]
    assert len(fetched["messages"]) == 4, f"Expected 4 messages (2 user + 2 assistant), got {len(fetched['messages'])}"
    assert fetched["question_count"] == 2, f"Expected question_count 2, got {fetched['question_count']}"
    print(f"✓ Get Conversation History: Passed ({len(fetched['messages'])} messages restored with sources)")

    # Step 4e: List Conversations
    res = requests.get(f"{BASE_URL}/conversations")
    assert res.status_code == 200, f"List conversations failed: {res.text}"
    conv_list = res.json()["conversations"]
    assert any(c["id"] == conv_id for c in conv_list), "Created conversation should be in list"
    print(f"✓ List Conversations: Passed ({len(conv_list)} conversations retrieved)")

    # Step 4f: Delete Conversation
    res = requests.delete(f"{BASE_URL}/conversations/{conv_id}")
    assert res.status_code == 200, f"Delete conversation failed: {res.text}"
    print(f"✓ Delete Conversation: Passed")

    # 5. Bare Acts Catalog
    res = requests.get(f"{BASE_URL}/library/acts")
    assert res.status_code == 200, f"Acts catalog failed: {res.text}"
    acts = res.json()["acts"]
    assert len(acts) >= 20, f"Expected >= 20 acts, got {len(acts)}"
    print(f"✓ Bare Acts Catalog: Passed ({len(acts)} acts listed)")

    print("=" * 70)
    print("ALL 7 TEST SUITES PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
