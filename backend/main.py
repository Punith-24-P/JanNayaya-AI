"""
JanNyaya AI - FastAPI Backend
"""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form,
    Header,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ============================================================
# PROJECT IMPORTS
# ============================================================

from backend.ocr_service import (
    extract_text_from_image,
)

from backend.pdf_service import (
    extract_text_from_pdf,
    validate_document_file,
)

from backend.text_cleaner import (
    clean_text,
)

from backend.rag_service import (
    answer_question,
)

from backend.llm_service import (
    chat_with_document,
)

from backend.vector_store import (
    get_collection_count,
)

from backend.legal_analysis_service import (
    analyze_legal_document,
    analyze_multiple_documents,
    extract_document_timeline,
)

from backend.legal_provision_service import (
    analyze_provisions,
)

from backend.auth_service import (
    register_user,
    login_user,
    get_user_from_token,
    update_user_profile,
    get_user_stats,
    create_chat_session,
    get_user_chat_sessions,
    get_chat_session_messages,
    save_chat_message,
    rename_chat_session,
    delete_chat_session,
    save_history_item,
    get_user_history,
    create_conversation,
    list_conversations,
    get_conversation,
    add_conversation_turn,
    update_conversation_analysis,
    delete_conversation,
)

from backend.legal_data_ingest import (
    CATEGORY_INFO,
    ROUTE_MAP,
)

from backend.knowledge_base_service import (
    get_knowledge_base_statistics,
    get_knowledge_base_health,
    delete_document_by_id,
    search_knowledge_base,
)

from backend.orchestrator import (
    JanNyayaOrchestrator,
    DocumentIntakeAgent,
)


# ============================================================
# SPEECH SERVICE
# ============================================================

try:

    from backend.speech_service import (
        transcribe_audio,
    )

    SPEECH_AVAILABLE = True

except Exception as error:

    print(
        "Speech service unavailable:",
        type(error).__name__,
        str(error),
    )

    transcribe_audio = None
    SPEECH_AVAILABLE = False


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="JanNyaya AI",
    description=(
        "Multimodal Intelligent Legal "
        "Assistance System for Citizens"
    ),
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class QuestionRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    language: Optional[str] = "english"


class AnalyzeTextRequest(BaseModel):
    text: str
    language: Optional[str] = "english"
    explanation_language: Optional[str] = None


class DocumentChatRequest(BaseModel):
    document_text: str
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    language: Optional[str] = "english"
    explanation_language: Optional[str] = None


class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = ""
    email: Optional[str] = ""
    language: Optional[str] = "english"


class LoginRequest(BaseModel):
    username: str
    password: str


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    language: Optional[str] = None
    avatar: Optional[str] = None
    citizen_status: Optional[str] = None
    default_explanation_lang: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Legal Consultation"


class RenameSessionRequest(BaseModel):
    title: str


class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    language: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None


class SaveHistoryRequest(BaseModel):
    item_type: str
    title: str
    data: Dict[str, Any]


class TimelineRequest(BaseModel):
    text: Optional[str] = ""
    documents: Optional[List[Dict[str, Any]]] = None


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Legal Consultation"
    language: Optional[str] = "English"
    legal_topic: Optional[str] = "general"
    conversation_id: Optional[str] = None


class PostMessageRequest(BaseModel):
    question: str
    language: Optional[str] = "English"
    history: Optional[List[Dict[str, Any]]] = None


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print()
    print("=" * 70)
    print("JAN NYAYA AI BACKEND")
    print("=" * 70)
    print("API: http://127.0.0.1:8000")
    print("Docs: http://127.0.0.1:8000/docs")
    print("Speech: /speech-to-text")
    print("Legal Q&A: /ask")
    print("Document analysis: /upload")
    print("=" * 70)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "JanNyaya AI",
        "status": "running",
        "message": "Welcome to JanNyaya AI - Multimodal Legal Assistance System for Citizens",
        "version": "0.1.0",
        "features": {
            "legal_qa": True,
            "document_analysis": True,
            "multi_document_analysis": True,
            "provision_explanation": True,
            "ocr": True,
            "speech_to_text": SPEECH_AVAILABLE,
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():

    try:
        chunk_count = get_collection_count()
    except Exception:
        chunk_count = 0

    return {
        "status": "healthy",
        "knowledge_base_chunks": chunk_count,
        "speech_available": SPEECH_AVAILABLE,
    }


# ============================================================
# AUTHENTICATION & SESSIONS
# ============================================================

def _get_auth_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    parts = authorization.split()
    token = parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else authorization
    return get_user_from_token(token)


@app.post("/auth/register")
def api_register(req: RegisterRequest):
    res = register_user(
        username=req.username,
        password=req.password,
        full_name=req.full_name or req.username,
        email=req.email or "",
        language=req.language or "english",
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Registration failed"))
    return res


@app.post("/auth/login")
def api_login(req: LoginRequest):
    res = login_user(username=req.username, password=req.password)
    if res.get("status") == "error":
        raise HTTPException(status_code=401, detail=res.get("message", "Invalid credentials"))
    return res


@app.get("/auth/me")
def api_me(authorization: Optional[str] = Header(None)):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized session.")
    return {"status": "success", "user": user}


@app.put("/auth/profile")
def api_update_profile(req: UpdateProfileRequest, authorization: Optional[str] = Header(None)):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to update your profile.")
    res = update_user_profile(
        user_id=user["id"],
        full_name=req.full_name,
        email=req.email,
        language=req.language,
        avatar=req.avatar,
        citizen_status=req.citizen_status,
        default_explanation_lang=req.default_explanation_lang,
        current_password=req.current_password,
        new_password=req.new_password,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Profile update failed"))
    return res


@app.get("/auth/stats")
def api_get_user_stats(authorization: Optional[str] = Header(None)):
    user = _get_auth_user(authorization)
    user_id = user["id"] if user else 0
    stats = get_user_stats(user_id)
    return stats


# ============================================================
# CHAT SESSIONS & CONVERSATION WORKFLOW (CHATGPT STYLE)
# ============================================================

@app.post("/chat/sessions")
@app.post("/auth/sessions")
def api_create_session(req: CreateSessionRequest, authorization: Optional[str] = Header(None)):
    user = _get_auth_user(authorization)
    user_id = user["id"] if user else 0
    session = create_chat_session(user_id=user_id, title=req.title or "New Legal Consultation")
    return {"status": "success", "session": session}


@app.get("/chat/sessions")
@app.get("/auth/sessions")
def api_list_sessions(authorization: Optional[str] = Header(None), limit: int = 50):
    user = _get_auth_user(authorization)
    if not user:
        return {"status": "success", "sessions": []}
    sessions = get_user_chat_sessions(user_id=user["id"], limit=limit)
    return {"status": "success", "sessions": sessions}


@app.get("/chat/sessions/{session_id}")
def api_get_session(session_id: str, authorization: Optional[str] = Header(None)):
    user = _get_auth_user(authorization)
    user_id = user["id"] if user else 0
    messages = get_chat_session_messages(session_id=session_id, user_id=user_id)
    return {"status": "success", "session_id": session_id, "messages": messages}


@app.put("/chat/sessions/{session_id}/title")
def api_rename_session(session_id: str, req: RenameSessionRequest, authorization: Optional[str] = Header(None)):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    renamed = rename_chat_session(session_id=session_id, user_id=user["id"], new_title=req.title)
    return {"status": "success" if renamed else "error"}


@app.delete("/chat/sessions/{session_id}")
def api_delete_session(session_id: str, authorization: Optional[str] = Header(None)):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    deleted = delete_chat_session(session_id=session_id, user_id=user["id"])
    return {"status": "success" if deleted else "error"}


@app.post("/chat/message")
def api_chat_message(req: ChatMessageRequest, authorization: Optional[str] = Header(None)):
    """
    ChatGPT-style conversational message processing:
    1. Resolves/creates session.
    2. Runs multi-turn grounded RAG with conversation history.
    3. Persists user and assistant messages into session.
    """
    user = _get_auth_user(authorization)
    user_id = user["id"] if user else 0
    session_id = req.session_id

    # Create new session if none provided and user is authenticated
    if not session_id and user:
        clean_title = req.question.strip()[:60] or "New Legal Consultation"
        session = create_chat_session(user_id=user_id, title=clean_title)
        session_id = session["id"]

    # Fetch conversation history from session if available
    history = req.history
    if not history and session_id and user:
        past_msgs = get_chat_session_messages(session_id=session_id, user_id=user_id)
        history = [{"role": m["role"], "content": m["text"]} for m in past_msgs[-8:]]

    # Execute Grounded RAG with history
    result = answer_question(
        question=req.question,
        history=history,
    )

    answer_text = result.get("answer", "")
    sources = result.get("sources", [])

    # Persist messages if user is logged in and session exists
    if user and session_id:
        import time
        ts = time.strftime("%I:%M %p")
        save_chat_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            text=req.question,
            sources=[],
            timestamp=ts,
        )
        save_chat_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            text=answer_text,
            sources=sources,
            timestamp=ts,
        )

    return {
        "status": "success",
        "session_id": session_id,
        "question": req.question,
        "answer": answer_text,
        "sources": sources,
    }


@app.get("/auth/history")
def api_get_history(authorization: Optional[str] = Header(None), limit: int = 30):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to view case history.")
    history = get_user_history(user_id=user["id"], limit=limit)
    return {"status": "success", "history": history}


@app.post("/auth/history/save")
def api_save_history(req: SaveHistoryRequest, authorization: Optional[str] = Header(None)):
    user = _get_auth_user(authorization)
    if not user:
        return {"status": "skipped", "message": "Guest mode - history not persisted to database."}
    saved = save_history_item(
        user_id=user["id"],
        item_type=req.item_type,
        title=req.title,
        data=req.data,
    )
    return {"status": "success" if saved else "error"}


# ============================================================
# CONVERSATIONS & CONSULTATION HISTORY REST API
# ============================================================

@app.get("/conversations")
def api_list_conversations(
    authorization: Optional[str] = Header(None),
    search: Optional[str] = None,
    limit: int = 50,
):
    """
    List all previous consultations with real persistent metadata:
    conversation_id, title, created_at, updated_at, language,
    question_count, last_question, last_answer, documents_count, legal_topic, status.
    """
    user = _get_auth_user(authorization)
    user_id = user["id"] if user else None
    convs = list_conversations(user_id=user_id, search=search, limit=limit)
    return {
        "status": "success",
        "total": len(convs),
        "conversations": convs,
    }


@app.get("/conversations/{conversation_id}")
def api_get_conversation(
    conversation_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    Retrieve full conversation with complete question-answer turns,
    sources, timestamps, attached document facts, and legal topic.
    """
    user = _get_auth_user(authorization)
    user_id = user["id"] if user else None
    conv = get_conversation(conversation_id=conversation_id, user_id=user_id)
    if not conv:
        raise HTTPException(
            status_code=404,
            detail=f"Consultation '{conversation_id}' was not found.",
        )
    return {
        "status": "success",
        "conversation": conv,
    }


@app.post("/conversations")
def api_create_conversation(
    req: CreateConversationRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Create a new consultation session.
    """
    user = _get_auth_user(authorization)
    user_id = user["id"] if user else 0
    conv = create_conversation(
        title=req.title or "New Legal Consultation",
        language=req.language or "English",
        legal_topic=req.legal_topic or "general",
        user_id=user_id,
        conversation_id=req.conversation_id,
    )
    return {
        "status": "success",
        "conversation": conv,
    }


@app.post("/conversations/{conversation_id}/messages")
def api_post_conversation_message(
    conversation_id: str,
    req: PostMessageRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Append a new question to an existing or active conversation.
    Executes grounded RAG with multi-turn history context, saves both turns,
    and updates conversation metadata and title.
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    user = _get_auth_user(authorization)
    user_id = user["id"] if user else 0

    # Build history context from previous messages if not explicitly supplied
    history = req.history
    if not history:
        conv = get_conversation(conversation_id=conversation_id, user_id=user_id)
        if conv and conv.get("messages"):
            past_msgs = conv["messages"]
            history = [{"role": m["role"], "content": m["text"]} for m in past_msgs[-8:]]

    try:
        # Run grounded RAG
        rag_result = answer_question(
            question=q,
            history=history,
        )

        if not isinstance(rag_result, dict):
            rag_result = {"answer": str(rag_result), "sources": []}

        answer_text = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])
        legal_topic = rag_result.get("legal_topic") or rag_result.get("route") or "general"

        updated_conv = add_conversation_turn(
            conversation_id=conversation_id,
            user_text=q,
            bot_text=answer_text,
            sources=sources,
            language=req.language or "English",
            legal_topic=legal_topic,
            user_id=user_id,
        )

        return {
            "status": "success",
            "conversation_id": conversation_id,
            "question": q,
            "answer": answer_text,
            "sources": sources,
            "conversation": updated_conv,
        }
    except Exception as error:
        print(f"Error in conversation message RAG: {error}")
        raise HTTPException(
            status_code=500,
            detail=f"Legal reasoning failed for this consultation turn: {str(error)}",
        )


@app.delete("/conversations/{conversation_id}")
def api_delete_conversation(
    conversation_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    Delete a consultation and all its messages.
    """
    user = _get_auth_user(authorization)
    user_id = user["id"] if user else None
    deleted = delete_conversation(conversation_id=conversation_id, user_id=user_id)
    return {
        "status": "success" if deleted else "error",
        "conversation_id": conversation_id,
    }


# ============================================================
# KNOWLEDGE BASE SEARCH
# ============================================================

@app.get("/knowledge-base/search")
def api_knowledge_base_search(
    q: Optional[str] = "",
    domain: Optional[str] = "all",
    section: Optional[str] = None,
    authority: Optional[str] = None,
    limit: int = 30,
):
    """
    Search indexed legal bare acts by Act, Section, Legal topic, Keyword, Authority, Domain.
    """
    return search_knowledge_base(
        query=q,
        domain=domain,
        section=section,
        authority=authority,
        limit=limit,
    )



# ============================================================
# LEGAL LIBRARY & STATUTE CATALOG
# ============================================================

@app.get("/library/acts")
def api_library_acts():
    """Return statutory bare acts with metadata."""
    acts_list = []
    for cat_id, info in CATEGORY_INFO.items():
        acts_list.append({
            "category_id": cat_id,
            "act_name": info.get("act_name", cat_id),
            "year": info.get("year", ""),
            "authority": info.get("authority", "Government of India"),
            "document_type": info.get("document_type", "Act"),
            "route": info.get("route", "general"),
        })
    return {
        "status": "success",
        "total_acts": len(acts_list),
        "acts": sorted(acts_list, key=lambda x: x["act_name"]),
    }


# ============================================================
# KNOWLEDGE BASE STATISTICS & TRANSPARENCY
# ============================================================

@app.get("/knowledge-base/stats")
def api_knowledge_base_stats():
    """
    Returns verified, honest coverage and statistics for the Indian legal knowledge base.
    Never claims 'all Indian laws are stored'.
    """
    return get_knowledge_base_statistics()


@app.get("/knowledge-base/health")
def api_knowledge_base_health():
    """
    Health check for ChromaDB, embedding model, BM25, and retrieval readiness.
    """
    return get_knowledge_base_health()


@app.delete("/knowledge-base/document/{document_id}")
def api_knowledge_base_delete_document(document_id: str):
    """
    Safely delete an indexed document and its chunks from ChromaDB.
    """
    return delete_document_by_id(document_id)


# ============================================================
# CASE TIMELINE & SESSION ANALYSIS
# ============================================================

@app.post("/analyze-session")
def api_analyze_session(req: MultiDocumentRequest):
    """
    Synthesize multi-document analysis across uploaded files with cross-document conflict detection.
    """
    docs = req.documents or []
    if not docs:
        raise HTTPException(status_code=400, detail="No documents provided for session analysis.")
    res = analyze_multiple_documents(docs, language=req.language or "english")
    return res


# ============================================================
# CASE TIMELINE EXTRACTION
# ============================================================

@app.post("/analyze-timeline")
def api_analyze_timeline(req: TimelineRequest):
    docs = req.documents or []
    if req.text and not docs:
        docs = [{"filename": "Document", "text": req.text}]
    timeline = extract_document_timeline(docs)
    return {
        "status": "success",
        "timeline_events": len(timeline),
        "timeline": timeline,
    }


# ============================================================
# LEGAL Q&A
# ============================================================

@app.post("/ask")
@app.post("/chat")
async def ask_question(
    request: QuestionRequest,
):

    question = str(
        request.question or request.query or ""
    ).strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        result = answer_question(
            question,
            history=request.conversation_history,
        )

        if not isinstance(
            result,
            dict,
        ):
            result = {
                "answer": str(result),
                "sources": [],
            }

        answer_text = result.get(
            "answer",
            "",
        )

        return {
            "status": "success",
            "question": question,
            "query": question,
            "answer": answer_text,
            "response": answer_text,
            "sources": result.get(
                "sources",
                [],
            ),
        }

    except Exception as error:

        print(
            "Question answering error:",
            type(error).__name__,
            str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Question answering failed: "
                f"{str(error)}"
            ),
        )


# ============================================================
# SAVE UPLOAD
# ============================================================

async def _save_upload(
    file: UploadFile,
) -> Path:

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    original_path = Path(
        file.filename
    )

    extension = (
        original_path
        .suffix
        .lower()
    )

    safe_stem = "".join(
        character
        for character in original_path.stem
        if character.isalnum()
        or character in ("_", "-")
    )

    if not safe_stem:
        safe_stem = "uploaded_file"

    filename = (
        f"{safe_stem}_"
        f"{uuid.uuid4().hex[:8]}"
        f"{extension}"
    )

    path = UPLOAD_DIR / filename

    with path.open("wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer,
        )

    return path


# ============================================================
# TEXT ANALYSIS PIPELINE
# ============================================================

def _analyze_text_pipeline(
    text: str,
    language: str = "english",
) -> dict:

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("No readable text was extracted.")

    analysis = analyze_legal_document(
        cleaned,
        language=language,
    )

    if not isinstance(analysis, dict):
        raise ValueError("Legal analysis returned an invalid result.")

    raw_results = analysis.get("_retrieval_results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    legal_term = str(analysis.get("legal_topic", analysis.get("legal_term", "general")) or "general").strip()

    provision_analysis = analyze_provisions(
        results=raw_results,
        legal_term=legal_term,
        document_text=cleaned,
    )

    public_analysis = dict(analysis)
    public_analysis.pop("_retrieval_results", None)

    return {
        "cleaned_text": cleaned,
        "analysis": public_analysis,
        "provision_analysis": provision_analysis,
    }


# ============================================================
# ANALYZE TEXT
# ============================================================

@app.post("/analyze-text")
async def analyze_text(
    request: AnalyzeTextRequest,
):
    text = str(request.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    requested_lang = request.explanation_language or request.language or "english"

    try:
        result = _analyze_text_pipeline(
            text,
            language=requested_lang,
        )

        analysis_data = result["analysis"]

        return {
            "status": "success",
            "text": result["cleaned_text"],
            "characters": len(result["cleaned_text"]),
            "document_language": analysis_data.get("document_language", "english"),
            "explanation_language": analysis_data.get("explanation_language", requested_lang),
            "language": analysis_data.get("explanation_language", requested_lang),
            "summary": analysis_data.get("summary", ""),
            "document_overview": analysis_data.get("document_overview", []),
            "parties": analysis_data.get("parties", []),
            "important_facts": analysis_data.get("important_facts", []),
            "claims": analysis_data.get("claims", []),
            "amounts": analysis_data.get("amounts", []),
            "dates": analysis_data.get("dates", []),
            "deadlines": analysis_data.get("deadlines", []),
            "legal_references": analysis_data.get("legal_references", []),
            "legal_issues": analysis_data.get("legal_issues", []),
            "obligations": analysis_data.get("obligations", []),
            "possible_consequences": analysis_data.get("possible_consequences", []),
            "relevant_provisions": analysis_data.get("relevant_provisions", []),
            "conflicts": analysis_data.get("conflicts", []),
            "missing_information": analysis_data.get("missing_information", []),
            "next_steps": analysis_data.get("next_steps", []),
            "warnings": analysis_data.get("warnings", []),
            "actionable_steps": analysis_data.get("actionable_steps", []),
            "conditions_and_clauses": analysis_data.get("conditions_and_clauses", []),
            "provisions": analysis_data.get("provisions", []),
            "primary_provision": analysis_data.get("primary_provision"),
            "timeline": analysis_data.get("timeline", []),
            "safety_caution": analysis_data.get("safety_caution", "General legal information only."),
            "disclaimer": analysis_data.get("disclaimer", "JanNyaya AI provides verified statutory legal information."),
            "analysis": analysis_data,
            "provision_analysis": result["provision_analysis"],
        }

    except Exception as error:
        print("Legal analysis error:", type(error).__name__, str(error))
        raise HTTPException(
            status_code=500,
            detail=f"Legal analysis failed: {str(error)}",
        )


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    language: Optional[str] = Form("english"),
    explanation_language: Optional[str] = Form(None),
):
    allowed_extensions = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".txt",
    }

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Supported files: PDF, JPG, JPEG, PNG, WEBP and TXT.",
        )

    requested_lang = explanation_language or language or "english"

    try:
        file_path = await _save_upload(file)

        validation = validate_document_file(file_path)
        if not validation["is_valid"]:
            try:
                file_path.unlink()
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail=validation["error"],
            )

        if extension == ".pdf":
            raw_text = extract_text_from_pdf(str(file_path))
        elif extension in (".txt", ".text"):
            try:
                raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                raw_text = ""
        else:
            raw_text = extract_text_from_image(str(file_path))

        text = clean_text(raw_text)

        if not text:
            raise HTTPException(
                status_code=422,
                detail="No readable text could be extracted from the document.",
            )

        pipeline = _analyze_text_pipeline(
            text,
            language=requested_lang,
        )

        analysis_data = pipeline["analysis"]
        llm_exp = analysis_data.get("llm_explanation") or {}
        summary_val = llm_exp.get("summary") or analysis_data.get("summary") or "Document analysis complete."
        clauses_val = llm_exp.get("conditions_and_clauses") or analysis_data.get("conditions_and_clauses") or []
        steps_val = llm_exp.get("actionable_steps") or analysis_data.get("next_steps") or []
        facts_val = analysis_data.get("facts") or {}
        provisions_val = analysis_data.get("provisions") or pipeline["provision_analysis"].get("provisions") or []
        primary_prov = analysis_data.get("primary_provision") or pipeline["provision_analysis"].get("primary_provision")
        safety_caution = analysis_data.get("safety_caution") or "General legal information only."
        disclaimer_val = analysis_data.get("disclaimer") or "JanNyaya AI provides verified statutory legal information."

        return {
            "status": "success",
            "filename": file.filename,
            "message": "Document uploaded, text extracted and legally analyzed.",
            "file_type": extension.lstrip("."),
            "characters": len(text),
            "text": text,
            "extracted_text": text,
            "document_language": analysis_data.get("document_language", "english"),
            "explanation_language": analysis_data.get("explanation_language", requested_lang),
            "language": analysis_data.get("explanation_language", requested_lang),
            "summary": summary_val,
            "document_overview": analysis_data.get("document_overview", []),
            "parties": analysis_data.get("parties", []),
            "important_facts": analysis_data.get("important_facts", []),
            "claims": analysis_data.get("claims", []),
            "conditions_and_clauses": clauses_val,
            "actionable_steps": steps_val,
            "facts": facts_val,
            "amounts": analysis_data.get("amounts", []),
            "dates": analysis_data.get("dates", []),
            "interest_rates": analysis_data.get("interest_rates", []),
            "instalments": analysis_data.get("instalments", []),
            "deadlines": analysis_data.get("deadlines", []),
            "legal_references": analysis_data.get("legal_references", []),
            "legal_issues": analysis_data.get("legal_issues", []),
            "obligations": analysis_data.get("obligations", []),
            "possible_consequences": analysis_data.get("possible_consequences", []),
            "relevant_provisions": analysis_data.get("relevant_provisions", []),
            "conflicts": analysis_data.get("conflicts", []),
            "missing_information": analysis_data.get("missing_information", []),
            "next_steps": steps_val,
            "warnings": analysis_data.get("warnings", []),
            "provisions": provisions_val,
            "primary_provision": primary_prov,
            "timeline": analysis_data.get("timeline", []),
            "safety_caution": safety_caution,
            "disclaimer": disclaimer_val,
            "analysis": analysis_data,
            "provision_analysis": pipeline["provision_analysis"],
        }

    except HTTPException:
        raise
    except Exception as error:
        print("Document processing error:", type(error).__name__, str(error))
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(error)}",
        )


# ============================================================
# MULTIPLE DOCUMENT UPLOAD & CASE ANALYSIS
# ============================================================

@app.post("/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    language: Optional[str] = Form("english"),
    explanation_language: Optional[str] = Form(None),
):
    """
    Upload and analyze multiple legal documents (PDFs, JPGs, PNGs, WEBPs)
    in a unified case workflow with multilingual LLM synthesis.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were provided in the request.",
        )

    allowed_extensions = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".txt",
    }

    processed_docs = []
    failed_files = []
    requested_lang = explanation_language or language or "english"

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()

        if ext not in allowed_extensions:
            failed_files.append({
                "filename": file.filename,
                "error": f"Unsupported format '{ext}'. Allowed: PDF, JPG, PNG, WEBP, TXT."
            })
            continue

        try:
            file_path = await _save_upload(file)

            validation = validate_document_file(file_path)
            if not validation["is_valid"]:
                try:
                    file_path.unlink()
                except Exception:
                    pass
                failed_files.append({
                    "filename": file.filename,
                    "error": validation["error"],
                })
                continue

            if ext == ".pdf":
                raw_text = extract_text_from_pdf(str(file_path))
            elif ext in (".txt", ".text"):
                try:
                    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    raw_text = ""
            else:
                raw_text = extract_text_from_image(str(file_path))

            text = clean_text(raw_text)

            if not text:
                failed_files.append({
                    "filename": file.filename,
                    "error": "No readable text could be extracted."
                })
                continue

            processed_docs.append({
                "filename": file.filename,
                "file_type": ext.lstrip("."),
                "text": text,
                "characters": len(text),
            })

        except Exception as file_err:
            failed_files.append({
                "filename": file.filename,
                "error": str(file_err),
            })

    if not processed_docs:
        raise HTTPException(
            status_code=422,
            detail=(
                f"None of the {len(files)} uploaded document(s) could be extracted. "
                f"Failures: {failed_files}"
            ),
        )

    try:
        case_result = analyze_multiple_documents(
            processed_docs,
            language=requested_lang,
        )

        raw_results = case_result.get("_retrieval_results", [])
        if not isinstance(raw_results, list):
            raw_results = []

        case_legal_term = str(case_result.get("legal_topic", "general") or "general").strip()
        combined_text = "\n\n".join(d["text"] for d in processed_docs)

        provision_analysis = analyze_provisions(
            results=raw_results,
            legal_term=case_legal_term,
            document_text=combined_text,
        )

        public_case_result = dict(case_result)
        public_case_result.pop("_retrieval_results", None)

        multi_summary = (
            public_case_result.get("llm_explanation", {}).get("summary")
            or public_case_result.get("case_overview")
            or f"Synthesized analysis across {len(processed_docs)} uploaded files."
        )
        multi_clauses = public_case_result.get("llm_explanation", {}).get("conditions_and_clauses", [])
        multi_steps = public_case_result.get("llm_explanation", {}).get("actionable_steps", [])

        return {
            "status": "success",
            "message": f"Successfully processed and analyzed {len(processed_docs)} document(s).",
            "total_documents": len(processed_docs),
            "document_language": public_case_result.get("document_language", "english"),
            "explanation_language": public_case_result.get("explanation_language", requested_lang),
            "language": public_case_result.get("explanation_language", requested_lang),
            "text": combined_text,
            "extracted_text": combined_text,
            "summary": multi_summary,
            "multi_document_summary": multi_summary,
            "case_overview": public_case_result.get("case_overview", ""),
            "document_overview": public_case_result.get("document_overview", []),
            "combined_case_summary": public_case_result.get("combined_case_summary", {}),
            "important_facts": public_case_result.get("important_facts", []),
            "amounts": public_case_result.get("amounts", []),
            "parties": public_case_result.get("parties", []),
            "legal_references": public_case_result.get("legal_references", []),
            "conditions_and_clauses": multi_clauses,
            "actionable_steps": multi_steps,
            "next_steps": multi_steps,
            "conflicts": public_case_result.get("conflicts", []),
            "timeline": public_case_result.get("timeline", []),
            "provisions": public_case_result.get("provisions", []),
            "primary_provision": public_case_result.get("primary_provision"),
            "documents": public_case_result.get("documents", []),
            "failed_files": failed_files,
            "analysis": public_case_result,
            "provision_analysis": provision_analysis,
        }

    except Exception as error:
        print(
            "Multi-document case processing error:",
            type(error).__name__,
            str(error),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Multi-document case analysis failed: {str(error)}"
        )


# ============================================================
# INTERACTIVE DOCUMENT CHAT ASSISTANT
# ============================================================

@app.post("/document/chat")
@app.post("/chat-document")
def api_document_chat(request: DocumentChatRequest):
    """
    Interactive Q&A assistant specifically grounded in an uploaded document.
    Allows citizens to ask follow-up questions in English, Hindi, or Kannada.
    Supports both /document/chat and /chat-document endpoints.
    """
    question = str(request.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    doc_text = str(request.document_text or "").strip()
    requested_lang = request.explanation_language or request.language or "english"

    try:
        answer = chat_with_document(
            document_text=doc_text,
            question=question,
            history=request.conversation_history,
            language=requested_lang,
        )

        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "explanation_language": requested_lang,
            "language": requested_lang,
        }

    except Exception as error:
        print(
            "Document chat error:",
            type(error).__name__,
            str(error),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Document chat failed: {str(error)}"
        )



# ============================================================
# IMAGE OCR
# ============================================================

@app.post("/ocr")
async def ocr_document(
    file: UploadFile = File(...),
):

    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension not in allowed:

        raise HTTPException(
            status_code=400,
            detail=(
                "OCR endpoint accepts JPG, "
                "JPEG, PNG and WEBP."
            ),
        )

    try:

        path = await _save_upload(
            file
        )

        text = clean_text(
            extract_text_from_image(
                str(path)
            )
        )

        return {
            "status": "success",
            "filename": file.filename,
            "text": text,
            "characters": len(text),
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Image OCR failed: "
                f"{str(error)}"
            ),
        )


# ============================================================
# PDF OCR / TEXT EXTRACTION
# ============================================================

@app.post("/ocr-pdf")
async def ocr_pdf_document(
    file: UploadFile = File(...),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension != ".pdf":

        raise HTTPException(
            status_code=400,
            detail="This endpoint accepts PDF files only.",
        )

    try:

        path = await _save_upload(
            file
        )

        text = clean_text(
            extract_text_from_pdf(
                str(path)
            )
        )

        if not text:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text could be extracted."
                ),
            )

        return {
            "status": "success",
            "filename": file.filename,
            "text": text,
            "characters": len(text),
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF extraction failed: "
                f"{str(error)}"
            ),
        )


# ============================================================
# SPEECH TO TEXT
# ============================================================

@app.post("/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = Form("auto"),
):

    if not SPEECH_AVAILABLE:

        raise HTTPException(
            status_code=503,
            detail=(
                "Speech-to-text service is unavailable."
            ),
        )

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No audio file provided.",
        )

    try:

        audio_bytes = await file.read()

        if not audio_bytes:

            raise HTTPException(
                status_code=400,
                detail="Audio file is empty.",
            )

        print()
        print("=" * 70)
        print("SPEECH REQUEST")
        print(
            "Filename:",
            file.filename,
        )
        print(
            "Content-Type:",
            file.content_type,
        )
        print(
            "Language:",
            language,
        )
        print(
            "Bytes:",
            len(audio_bytes),
        )
        print("=" * 70)

        result = transcribe_audio(
            audio_bytes=audio_bytes,
            filename=file.filename,
            content_type=file.content_type,
            language=language,
        )

        if not isinstance(
            result,
            dict,
        ):
            result = {
                "status": "success",
                "text": str(result or ""),
                "language": language,
            }

        return result

    except HTTPException:
        raise

    except Exception as error:

        print(
            "Speech-to-text error:",
            type(error).__name__,
            str(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Speech-to-text failed: "
                f"{str(error)}"
            ),
        )


# ============================================================
# ENDPOINT SUMMARY
# ============================================================

@app.get("/endpoints")
def endpoint_summary():

    return {
        "status": "success",

        "endpoints": {
            "root": "GET /",
            "health": "GET /health",
            "ask": "POST /ask",
            "upload": "POST /upload",
            "upload_multiple": "POST /upload-multiple",
            "analyze_text": "POST /analyze-text",
            "ocr": "POST /ocr",
            "ocr_pdf": "POST /ocr-pdf",
            "speech_to_text":
                "POST /speech-to-text",
        },
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, log_level="info")