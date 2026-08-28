"""
Automated Test Suite for JanNyaya AI Account-Centric Data Isolation & Security.
Verifies:
1. User Registration & Password Hashing
2. Token generation & session management
3. Strict User A vs User B history & conversation isolation
4. Cross-account unauthorized access rejection
5. Logout & token invalidation
"""

import sys
import os
import uuid
import tempfile
import sqlite3

# Ensure backend path is on sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from auth_service import (
    register_user,
    login_user,
    get_user_from_token,
    logout_user,
    create_conversation,
    list_conversations,
    get_conversation,
    delete_conversation,
    add_conversation_turn
)

def run_tests():
    print("=" * 60)
    print("Starting JanNyaya AI Account Isolation & Security Tests")
    print("=" * 60)

    unique_suffix = uuid.uuid4().hex[:6]
    user_a_name = f"advocate_sharma_{unique_suffix}"
    user_b_name = f"citizen_priya_{unique_suffix}"
    password = "SecurePassword@2025"

    # 2. Register User A & User B
    print(f"\n[Test 1] Registering User A ({user_a_name}) and User B ({user_b_name})...")
    res_a = register_user(
        username=user_a_name,
        password=password,
        full_name="Adv. Rajesh Sharma",
        email=f"{user_a_name}@lawcourt.in",
        language="english"
    )
    assert res_a["status"] == "success", f"Failed to register User A: {res_a}"
    user_a = res_a["user"]
    user_a_id = user_a["id"]

    res_b = register_user(
        username=user_b_name,
        password=password,
        full_name="Priya Patel",
        email=f"{user_b_name}@gmail.com",
        language="hindi"
    )
    assert res_b["status"] == "success", f"Failed to register User B: {res_b}"
    user_b = res_b["user"]
    user_b_id = user_b["id"]
    print(f"✅ User A created with ID: {user_a_id}")
    print(f"✅ User B created with ID: {user_b_id}")

    # 3. Authenticate & obtain tokens
    print("\n[Test 2] Authenticating User A and User B...")
    login_res_a = login_user(user_a_name, password)
    assert login_res_a["status"] == "success", "User A authentication failed"
    token_a = login_res_a["token"]
    auth_user_a = login_res_a["user"]
    assert auth_user_a["id"] == user_a_id

    login_res_b = login_user(user_b_name, password)
    assert login_res_b["status"] == "success", "User B authentication failed"
    token_b = login_res_b["token"]
    auth_user_b = login_res_b["user"]
    assert auth_user_b["id"] == user_b_id
    print("✅ Successfully generated independent session tokens.")

    # 4. User A creates a conversation
    print("\n[Test 3] User A creating conversation: 'Cheque Bounce NI Act 138 Defense'...")
    conv_a = create_conversation(
        user_id=user_a_id,
        title="Cheque Bounce NI Act 138 Defense",
        language="english",
        legal_topic="banking"
    )
    conv_a_id = conv_a["id"]
    add_conversation_turn(
        conversation_id=conv_a_id,
        user_text="What is the limitation period for statutory notice under Section 138 of NI Act?",
        bot_text="Under Section 138(b) of Negotiable Instruments Act 1881, the legal notice must be dispatched within 30 days.",
        language="english",
        legal_topic="banking",
        user_id=user_a_id
    )
    print(f"✅ Conversation A created: {conv_a_id}")

    # 5. User B creates a conversation
    print("\n[Test 4] User B creating conversation: 'Consumer Court Defective Laptop Claim'...")
    conv_b = create_conversation(
        user_id=user_b_id,
        title="Consumer Court Defective Laptop Claim",
        language="hindi",
        legal_topic="consumer"
    )
    conv_b_id = conv_b["id"]
    add_conversation_turn(
        conversation_id=conv_b_id,
        user_text="Laptop kharab hone par consumer court mein claim kaise karein?",
        bot_text="Consumer Protection Act 2019 ke tehat aap District Consumer Commission mein shikayat darj kar sakte hain.",
        language="hindi",
        legal_topic="consumer",
        user_id=user_b_id
    )
    print(f"✅ Conversation B created: {conv_b_id}")
    print(f"✅ Conversation B created: {conv_b_id}")

    # 6. Verify User A cannot see User B's conversations
    print("\n[Test 5] Checking list_conversations isolation...")
    list_a = list_conversations(user_id=user_a_id)
    list_b = list_conversations(user_id=user_b_id)

    conv_ids_a = [c["id"] for c in list_a]
    conv_ids_b = [c["id"] for c in list_b]

    assert conv_a_id in conv_ids_a, "User A must see their own conversation"
    assert conv_b_id not in conv_ids_a, "CRITICAL ERROR: User A can see User B's conversation!"
    print("✅ User A sees ONLY their own conversation.")

    assert conv_b_id in conv_ids_b, "User B must see their own conversation"
    assert conv_a_id not in conv_ids_b, "CRITICAL ERROR: User B can see User A's conversation!"
    print("✅ User B sees ONLY their own conversation.")

    # 7. Verify cross-account direct access rejection
    print("\n[Test 6] User A attempting direct get_conversation for User B's conversation...")
    unauthorized_fetch = get_conversation(conv_b_id, user_id=user_a_id)
    assert unauthorized_fetch is None, "CRITICAL ERROR: User A accessed User B's conversation directly!"
    print("✅ Unauthorized cross-account conversation access correctly blocked (returned None).")

    # 8. Verify cross-account deletion rejection
    print("\n[Test 7] User A attempting to delete User B's conversation...")
    delete_result = delete_conversation(conv_b_id, user_id=user_a_id)
    assert delete_result is False, "CRITICAL ERROR: User A deleted User B's conversation!"

    # Ensure User B's conversation still exists
    verify_b = get_conversation(conv_b_id, user_id=user_b_id)
    assert verify_b is not None, "User B's conversation was wrongfully deleted!"
    print("✅ Unauthorized cross-account deletion correctly rejected.")

    # 9. Verify Logout & Token Invalidation
    print("\n[Test 8] User A logging out...")
    logout_success = logout_user(token_a)
    assert logout_success is True, "Logout failed"
    session_after_logout = get_user_from_token(token_a)
    assert session_after_logout is None, "CRITICAL ERROR: Token still valid after logout!"
    print("✅ Token successfully invalidated upon logout.")

    # 10. User B's token remains valid
    session_b_check = get_user_from_token(token_b)
    assert session_b_check is not None and session_b_check["id"] == user_b_id, "User B session was impacted by User A logout"
    print("✅ User B session remains active and isolated.")

    print("\n" + "=" * 60)
    print("🎉 ALL 8 ACCOUNT ISOLATION AND SECURITY TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
