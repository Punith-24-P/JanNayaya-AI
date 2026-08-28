"""
JanNyaya AI - Authentication & Case History Service

Features:
- Secure user registration and login with salt-hashed passwords.
- Lightweight SQLite persistence (`data/jannyaya_users.db`).
- Session token generation and validation.
- User legal consultation history tracking (questions, answers, analyses).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "jannyaya_users.db"


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    """Initialize SQLite tables for users, sessions, history, chat sessions, and messages."""
    conn = _get_db()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT DEFAULT '',
                language TEXT DEFAULT 'english',
                theme TEXT DEFAULT 'light',
                created_at REAL NOT NULL
            )
        """)
        # Ensure email, theme, avatar, citizen_status, and default_explanation_lang columns exist
        try:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN theme TEXT DEFAULT 'light'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN avatar TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN citizen_status TEXT DEFAULT 'Verified Citizen'")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN default_explanation_lang TEXT DEFAULT 'english'")
        except sqlite3.OperationalError:
            pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS case_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_type TEXT NOT NULL,
                title TEXT NOT NULL,
                data_json TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                sources_json TEXT DEFAULT '[]',
                timestamp TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER DEFAULT 0,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                language TEXT DEFAULT 'English',
                question_count INTEGER DEFAULT 0,
                last_question TEXT DEFAULT '',
                last_answer TEXT DEFAULT '',
                documents_count INTEGER DEFAULT 0,
                legal_topic TEXT DEFAULT 'general',
                status TEXT DEFAULT 'active',
                analysis_json TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_id INTEGER DEFAULT 0,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                sources_json TEXT DEFAULT '[]',
                timestamp TEXT NOT NULL,
                created_at REAL NOT NULL,
                legal_topic TEXT DEFAULT '',
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
    conn.close()


init_auth_db()


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def register_user(
    username: str,
    password: str,
    full_name: str,
    email: str = "",
    language: str = "english",
) -> Dict[str, Any]:
    username = username.strip().lower()
    full_name = full_name.strip()
    email = email.strip()
    if not username or len(username) < 3:
        return {"status": "error", "message": "Username must be at least 3 characters long."}
    if not password or len(password) < 4:
        return {"status": "error", "message": "Password must be at least 4 characters long."}

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    now = time.time()

    conn = _get_db()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, full_name, email, language, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, pwd_hash, salt, full_name or username, email, language, now),
            )
            user_id = cursor.lastrowid
            token = secrets.token_hex(32)
            conn.execute(
                "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
                (token, user_id, now),
            )
        return {
            "status": "success",
            "token": token,
            "user": {
                "id": user_id,
                "username": username,
                "full_name": full_name or username,
                "email": email,
                "language": language,
            },
        }
    except sqlite3.IntegrityError:
        return {"status": "error", "message": "Username already exists. Please choose another."}
    finally:
        conn.close()


def login_user(username: str, password: str) -> Dict[str, Any]:
    username = username.strip().lower()
    conn = _get_db()
    try:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if not user:
            return {"status": "error", "message": "Invalid username or password."}

        pwd_hash = _hash_password(password, user["salt"])
        if pwd_hash != user["password_hash"]:
            return {"status": "error", "message": "Invalid username or password."}

        token = secrets.token_hex(32)
        now = time.time()
        with conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
                (token, user["id"], now),
            )

        return {
            "status": "success",
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "full_name": user["full_name"],
                "email": user["email"] if "email" in user.keys() else "",
                "language": user["language"],
            },
        }
    finally:
        conn.close()


def logout_user(token: str) -> bool:
    """Invalidate active session token on logout."""
    if not token:
        return False
    conn = _get_db()
    try:
        with conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        return True
    except Exception as e:
        print(f"Logout error: {e}")
        return False
    finally:
        conn.close()


def get_user_from_auth_header(auth_header: Optional[str]) -> Optional[Dict[str, Any]]:
    """Helper to extract token from Authorization header and return authenticated user dict."""
    if not auth_header:
        return None
    parts = auth_header.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1]
        return get_user_from_token(token)
    elif len(parts) == 1 and not parts[0].lower().startswith("bearer"):
        return get_user_from_token(parts[0])
    return None


def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    conn = _get_db()
    try:
        row = conn.execute(
            """
            SELECT u.id, u.username, u.full_name, u.email, u.language, u.avatar, u.citizen_status, u.default_explanation_lang, u.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if row:
            keys = row.keys()
            return {
                "id": row["id"],
                "username": row["username"],
                "full_name": row["full_name"],
                "email": row["email"] if "email" in keys else "",
                "language": row["language"],
                "avatar": row["avatar"] if "avatar" in keys else "",
                "citizen_status": row["citizen_status"] if "citizen_status" in keys else "Verified Citizen",
                "default_explanation_lang": row["default_explanation_lang"] if "default_explanation_lang" in keys else "english",
                "created_at": row["created_at"],
            }
        return None
    finally:
        conn.close()


def update_user_profile(
    user_id: int,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    language: Optional[str] = None,
    avatar: Optional[str] = None,
    citizen_status: Optional[str] = None,
    default_explanation_lang: Optional[str] = None,
    current_password: Optional[str] = None,
    new_password: Optional[str] = None,
) -> Dict[str, Any]:
    conn = _get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return {"status": "error", "message": "User not found."}

        keys = user.keys()

        # Password change verification
        if new_password and str(new_password).strip():
            if not current_password:
                return {"status": "error", "message": "Current password is required to set a new password."}
            curr_hash = _hash_password(current_password, user["salt"])
            if curr_hash != user["password_hash"]:
                return {"status": "error", "message": "Incorrect current password."}
            if len(new_password) < 4:
                return {"status": "error", "message": "New password must be at least 4 characters."}
            new_salt = secrets.token_hex(16)
            new_hash = _hash_password(new_password, new_salt)
            with conn:
                conn.execute(
                    "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                    (new_hash, new_salt, user_id),
                )

        updated_name = full_name.strip() if full_name is not None and full_name.strip() else user["full_name"]
        updated_email = email.strip() if email is not None else (user["email"] if "email" in keys else "")
        updated_lang = language.strip() if language is not None and language.strip() else user["language"]
        updated_avatar = avatar if avatar is not None else (user["avatar"] if "avatar" in keys else "")
        updated_status = citizen_status.strip() if citizen_status is not None and citizen_status.strip() else (user["citizen_status"] if "citizen_status" in keys else "Verified Citizen")
        updated_def_exp = default_explanation_lang.strip() if default_explanation_lang is not None and default_explanation_lang.strip() else (user["default_explanation_lang"] if "default_explanation_lang" in keys else "english")

        with conn:
            conn.execute(
                """
                UPDATE users
                SET full_name = ?, email = ?, language = ?, avatar = ?, citizen_status = ?, default_explanation_lang = ?
                WHERE id = ?
                """,
                (updated_name, updated_email, updated_lang, updated_avatar, updated_status, updated_def_exp, user_id),
            )

        return {
            "status": "success",
            "message": "Profile updated successfully.",
            "user": {
                "id": user_id,
                "username": user["username"],
                "full_name": updated_name,
                "email": updated_email,
                "language": updated_lang,
                "avatar": updated_avatar,
                "citizen_status": updated_status,
                "default_explanation_lang": updated_def_exp,
                "created_at": user["created_at"],
            },
        }
    finally:
        conn.close()


def get_user_stats(user_id: int) -> Dict[str, Any]:
    """Retrieve live computed activity statistics for the user profile."""
    conn = _get_db()
    try:
        # Consultations: distinct sessions or conversations
        conv_row = conn.execute("SELECT COUNT(*) as cnt, SUM(question_count) as q_sum, SUM(documents_count) as d_sum FROM conversations WHERE user_id = ?", (user_id,)).fetchone()
        conv_cnt = conv_row["cnt"] if conv_row and conv_row["cnt"] else 0
        conv_q = conv_row["q_sum"] if conv_row and conv_row["q_sum"] else 0
        conv_d = conv_row["d_sum"] if conv_row and conv_row["d_sum"] else 0

        # Chat sessions
        chat_sess_row = conn.execute("SELECT COUNT(*) as cnt FROM chat_sessions WHERE user_id = ?", (user_id,)).fetchone()
        chat_cnt = chat_sess_row["cnt"] if chat_sess_row and chat_sess_row["cnt"] else 0

        # Case history items
        doc_hist_row = conn.execute("SELECT COUNT(*) as cnt FROM case_history WHERE user_id = ? AND item_type = 'document'", (user_id,)).fetchone()
        doc_hist_cnt = doc_hist_row["cnt"] if doc_hist_row and doc_hist_row["cnt"] else 0

        # User messages
        user_msgs_row = conn.execute("SELECT COUNT(*) as cnt FROM chat_messages WHERE user_id = ? AND role = 'user'", (user_id,)).fetchone()
        user_msgs_cnt = user_msgs_row["cnt"] if user_msgs_row and user_msgs_row["cnt"] else 0

        # Distinct legal topics explored
        topics_rows = conn.execute("SELECT DISTINCT legal_topic FROM conversations WHERE user_id = ? AND legal_topic != '' AND legal_topic != 'general'", (user_id,)).fetchall()
        topics_set = {r["legal_topic"] for r in topics_rows if r["legal_topic"]}

        total_consultations = max(1, conv_cnt + chat_cnt)
        total_docs = max(conv_d + doc_hist_cnt, 0)
        total_questions = max(conv_q + user_msgs_cnt, 0)
        total_topics = max(len(topics_set), 1 if (total_consultations > 0 or total_docs > 0) else 0)

        return {
            "status": "success",
            "consultations_count": total_consultations,
            "documents_analyzed": total_docs,
            "questions_asked": total_questions,
            "topics_explored": total_topics,
        }
    except Exception as e:
        print("Error computing user stats:", e)
        return {
            "status": "success",
            "consultations_count": 1,
            "documents_analyzed": 0,
            "questions_asked": 0,
            "topics_explored": 1,
        }
    finally:
        conn.close()


# ============================================================
# CHAT SESSIONS & CONVERSATION HISTORY
# ============================================================

def create_chat_session(user_id: int, title: str = "New Legal Consultation") -> Dict[str, Any]:
    session_id = secrets.token_hex(12)
    now = time.time()
    conn = _get_db()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, title[:100], now, now),
            )
        return {
            "id": session_id,
            "title": title[:100],
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
    finally:
        conn.close()


def get_user_chat_sessions(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(m.id) as message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON s.id = m.session_id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

        sessions = []
        for r in rows:
            sessions.append({
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["message_count"],
            })
        return sessions
    finally:
        conn.close()


def get_chat_session_messages(session_id: str, user_id: int) -> List[Dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, role, text, sources_json, timestamp, created_at
            FROM chat_messages
            WHERE session_id = ? AND user_id = ?
            ORDER BY id ASC
            """,
            (session_id, user_id),
        ).fetchall()

        messages = []
        for r in rows:
            try:
                sources = json.loads(r["sources_json"]) if r["sources_json"] else []
            except Exception:
                sources = []
            messages.append({
                "id": r["id"],
                "role": r["role"],
                "text": r["text"],
                "sources": sources,
                "timestamp": r["timestamp"],
            })
        return messages
    finally:
        conn.close()


def save_chat_message(
    session_id: str,
    user_id: int,
    role: str,
    text: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    timestamp: Optional[str] = None,
) -> bool:
    if not session_id or not text:
        return False
    now = time.time()
    ts = timestamp or time.strftime("%I:%M %p")
    sources_str = json.dumps(sources or [], ensure_ascii=False)

    conn = _get_db()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, role, text, sources_json, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, text, sources_str, ts, now),
            )
            # Update session's updated_at timestamp and title if first user message
            cursor = conn.execute("SELECT COUNT(*) as c FROM chat_messages WHERE session_id = ?", (session_id,))
            count = cursor.fetchone()["c"]
            if role == "user" and count <= 2:
                # Update title based on user's first query
                clean_title = text.strip()[:60]
                if len(text.strip()) > 60:
                    clean_title += "..."
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ?, title = ? WHERE id = ?",
                    (now, clean_title, session_id),
                )
            else:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                    (now, session_id),
                )
        return True
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return False
    finally:
        conn.close()


def rename_chat_session(session_id: str, user_id: int, new_title: str) -> bool:
    new_title = new_title.strip()[:100]
    if not new_title:
        return False
    conn = _get_db()
    try:
        with conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (new_title, time.time(), session_id, user_id),
            )
        return True
    finally:
        conn.close()


def delete_chat_session(session_id: str, user_id: int) -> bool:
    conn = _get_db()
    try:
        with conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ? AND user_id = ?", (session_id, user_id))
            conn.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
        return True
    finally:
        conn.close()


# ============================================================
# LEGACY CASE HISTORY
# ============================================================

def save_history_item(
    user_id: int,
    item_type: str,
    title: str,
    data: Dict[str, Any],
) -> bool:
    conn = _get_db()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO case_history (user_id, item_type, title, data_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, item_type, title[:100], json.dumps(data, ensure_ascii=False), time.time()),
            )
        return True
    except Exception as e:
        print(f"Error saving history item: {e}")
        return False
    finally:
        conn.close()


def get_user_history(user_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, item_type, title, data_json, timestamp
            FROM case_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

        history = []
        for r in rows:
            try:
                parsed_data = json.loads(r["data_json"])
            except Exception:
                parsed_data = {}
            history.append({
                "id": r["id"],
                "item_type": r["item_type"],
                "title": r["title"],
                "data": parsed_data,
                "timestamp": r["timestamp"],
            })
        return history
    finally:
        conn.close()


# ============================================================
# COMPREHENSIVE CONVERSATION HISTORY & PERSISTENCE
# ============================================================

def create_conversation(
    title: str = "New Legal Consultation",
    language: str = "English",
    legal_topic: str = "general",
    user_id: int = 0,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new consultation session matching the history data model."""
    conv_id = conversation_id or secrets.token_hex(16)
    now = time.time()
    clean_title = title.strip()[:120] or "New Legal Consultation"

    conn = _get_db()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    id, user_id, title, created_at, updated_at, language,
                    question_count, last_question, last_answer, documents_count,
                    legal_topic, status, analysis_json
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, '', '', 0, ?, 'active', '{}')
                """,
                (conv_id, user_id, clean_title, now, now, language, legal_topic),
            )
        return {
            "conversation_id": conv_id,
            "id": conv_id,
            "title": clean_title,
            "created_at": now,
            "updated_at": now,
            "language": language,
            "question_count": 0,
            "last_question": "",
            "last_answer": "",
            "documents_count": 0,
            "legal_topic": legal_topic,
            "status": "active",
            "messages": [],
            "analysis": {},
        }
    finally:
        conn.close()


def list_conversations(
    user_id: Optional[int] = None,
    search: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List recent consultations strictly isolated by user_id."""
    if user_id is None:
        return []

    conn = _get_db()
    try:
        query = "SELECT * FROM conversations WHERE user_id = ?"
        params: List[Any] = [user_id]

        if search and search.strip():
            query += " AND (title LIKE ? OR last_question LIKE ? OR legal_topic LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term, term])

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conversations = []
        for r in rows:
            conversations.append({
                "conversation_id": r["id"],
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "language": r["language"],
                "question_count": r["question_count"],
                "last_question": r["last_question"],
                "last_answer": r["last_answer"],
                "documents_count": r["documents_count"],
                "legal_topic": r["legal_topic"],
                "status": r["status"],
            })
        return conversations
    finally:
        conn.close()


def get_conversation(
    conversation_id: str,
    user_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Load full previous conversation with all questions, answers, sources, timestamps, and analysis strictly isolated by user_id."""
    if user_id is None or not conversation_id:
        return None

    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()

        if not row:
            # Fallback check if it was in legacy chat_sessions for this exact user
            legacy_session = conn.execute("SELECT * FROM chat_sessions WHERE id = ? AND user_id = ?", (conversation_id, user_id)).fetchone()
            if legacy_session:
                legacy_msgs = conn.execute(
                    "SELECT * FROM chat_messages WHERE session_id = ? AND user_id = ? ORDER BY id ASC",
                    (conversation_id, user_id),
                ).fetchall()
                formatted_msgs = []
                for m in legacy_msgs:
                    try:
                        srcs = json.loads(m["sources_json"]) if m["sources_json"] else []
                    except Exception:
                        srcs = []
                    formatted_msgs.append({
                        "id": m["id"],
                        "role": m["role"],
                        "text": m["text"],
                        "sources": srcs,
                        "timestamp": m["timestamp"],
                        "created_at": m["created_at"],
                        "legal_topic": "",
                    })
                return {
                    "conversation_id": legacy_session["id"],
                    "id": legacy_session["id"],
                    "title": legacy_session["title"],
                    "created_at": legacy_session["created_at"],
                    "updated_at": legacy_session["updated_at"],
                    "language": "English",
                    "question_count": len([m for m in formatted_msgs if m["role"] == "user"]),
                    "last_question": formatted_msgs[-1]["text"] if formatted_msgs else "",
                    "last_answer": "",
                    "documents_count": 0,
                    "legal_topic": "general",
                    "status": "active",
                    "messages": formatted_msgs,
                    "analysis": {},
                }
            return None

        # Fetch all messages
        msg_rows = conn.execute(
            """
            SELECT id, role, text, sources_json, timestamp, created_at, legal_topic
            FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

        messages = []
        for mr in msg_rows:
            try:
                sources = json.loads(mr["sources_json"]) if mr["sources_json"] else []
            except Exception:
                sources = []
            messages.append({
                "id": mr["id"],
                "role": mr["role"],
                "text": mr["text"],
                "sources": sources,
                "timestamp": mr["timestamp"],
                "created_at": mr["created_at"],
                "legal_topic": mr["legal_topic"] or "",
            })

        try:
            analysis = json.loads(row["analysis_json"]) if row["analysis_json"] else {}
        except Exception:
            analysis = {}

        return {
            "conversation_id": row["id"],
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "language": row["language"],
            "question_count": row["question_count"],
            "last_question": row["last_question"],
            "last_answer": row["last_answer"],
            "documents_count": row["documents_count"],
            "legal_topic": row["legal_topic"],
            "status": row["status"],
            "messages": messages,
            "analysis": analysis,
        }
    finally:
        conn.close()


def add_conversation_turn(
    conversation_id: str,
    user_text: str,
    bot_text: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    language: str = "English",
    legal_topic: Optional[str] = None,
    user_id: int = 0,
    attached_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a complete user-assistant turn to the conversation and update metadata."""
    now = time.time()
    ts = time.strftime("%I:%M %p")
    sources_list = sources or []
    sources_json = json.dumps(sources_list, ensure_ascii=False)
    topic = legal_topic or "general"

    conn = _get_db()
    try:
        with conn:
            # Check if conversation exists, otherwise create it
            existing = conn.execute(
                "SELECT id, question_count, title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if not existing:
                clean_title = user_text.strip()[:60]
                if len(user_text.strip()) > 60:
                    clean_title += "..."
                conn.execute(
                    """
                    INSERT INTO conversations (
                        id, user_id, title, created_at, updated_at, language,
                        question_count, last_question, last_answer, documents_count,
                        legal_topic, status, analysis_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 0, '', '', 0, ?, 'active', ?)
                    """,
                    (
                        conversation_id,
                        user_id,
                        clean_title or "Legal Consultation",
                        now,
                        now,
                        language,
                        topic,
                        json.dumps(attached_analysis or {}, ensure_ascii=False),
                    ),
                )
                q_count = 0
            else:
                q_count = existing["question_count"]

            # Insert User Message
            conn.execute(
                """
                INSERT INTO conversation_messages (
                    conversation_id, user_id, role, text, sources_json, timestamp, created_at, legal_topic
                )
                VALUES (?, ?, 'user', ?, '[]', ?, ?, ?)
                """,
                (conversation_id, user_id, user_text, ts, now, topic),
            )

            # Insert Assistant Message
            conn.execute(
                """
                INSERT INTO conversation_messages (
                    conversation_id, user_id, role, text, sources_json, timestamp, created_at, legal_topic
                )
                VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?)
                """,
                (conversation_id, user_id, bot_text, sources_json, ts, now, topic),
            )

            # Update conversation header
            new_q_count = q_count + 1
            if new_q_count == 1:
                title_update = user_text.strip()[:60]
                if len(user_text.strip()) > 60:
                    title_update += "..."
                conn.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?, question_count = ?, last_question = ?, last_answer = ?,
                        title = ?, legal_topic = ?, language = ?
                    WHERE id = ?
                    """,
                    (now, new_q_count, user_text, bot_text, title_update, topic, language, conversation_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?, question_count = ?, last_question = ?, last_answer = ?,
                        legal_topic = ?, language = ?
                    WHERE id = ?
                    """,
                    (now, new_q_count, user_text, bot_text, topic, language, conversation_id),
                )

        return get_conversation(conversation_id, user_id=user_id) or {}
    finally:
        conn.close()


def update_conversation_analysis(
    conversation_id: str,
    analysis_data: Dict[str, Any],
    documents_count: Optional[int] = None,
) -> bool:
    """Store or update associated document analysis in the conversation record."""
    conn = _get_db()
    try:
        analysis_str = json.dumps(analysis_data, ensure_ascii=False)
        with conn:
            if documents_count is not None:
                conn.execute(
                    "UPDATE conversations SET analysis_json = ?, documents_count = ?, updated_at = ? WHERE id = ?",
                    (analysis_str, documents_count, time.time(), conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET analysis_json = ?, updated_at = ? WHERE id = ?",
                    (analysis_str, time.time(), conversation_id),
                )
        return True
    except Exception as e:
        print(f"Error updating conversation analysis: {e}")
        return False
    finally:
        conn.close()


def delete_conversation(conversation_id: str, user_id: Optional[int] = None) -> bool:
    """Safely delete a conversation and its messages strictly scoped to user_id."""
    if not conversation_id or user_id is None or user_id <= 0:
        return False
    conn = _get_db()
    try:
        with conn:
            # Check ownership first
            owner_check = conn.execute(
                "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()

            legacy_check = None
            if not owner_check:
                legacy_check = conn.execute(
                    "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                    (conversation_id, user_id),
                ).fetchone()

            if not owner_check and not legacy_check:
                # User does not own this conversation
                return False

            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            return True
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return False
    finally:
        conn.close()

