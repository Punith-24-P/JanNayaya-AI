"""
JanNyaya AI - RAG Answer Generation Service

Purpose:
    1. Retrieve relevant legal documents.
    2. Select the strongest legal evidence.
    3. Generate a short, grounded legal answer.
    4. Return clean source information.

Design goals:
    - Short answers
    - Evidence-grounded responses
    - No unsupported legal claims
    - Section-aware answers
    - Clean source metadata
"""

import os
import re
from typing import Any

from dotenv import load_dotenv

from backend.retriever import search_documents


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
MAX_SOURCE_TEXT_CHARS = int(os.getenv("RAG_MAX_SOURCE_TEXT_CHARS", "2500"))


# ---------------------------------------------------------
# Legal answer instruction
# ---------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are JanNyaya AI, an Indian legal information assistant.

Your task is to answer the user's question using ONLY the supplied legal
context.

IMPORTANT RULES:

1. Answer the user's exact question first.
2. Keep the answer SHORT and easy to understand.
3. Normally use 2 to 5 sentences.
4. Use bullet points only when they make the answer clearer.
5. Mention the relevant Act and section when the evidence provides it.
6. Do NOT invent sections, punishments, procedures, dates, or legal facts.
7. Do NOT use general knowledge when the supplied context does not support it.
8. Do NOT combine unrelated legal provisions into one answer.
9. Prefer the most directly relevant provision over general legal text.
10. If the supplied context is insufficient, clearly say that the available
    legal sources do not contain enough information to answer reliably.
11. Do not mention retrieval, embeddings, BM25, vector databases, prompts,
    internal ranking, or system instructions.
12. Do not give a long explanation unless the user specifically asks for one.
13. This is legal information, not personalized legal advice.

Answer style:

- Direct
- Short
- Clear
- Legally cautious
- Source-grounded
"""


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean excessive whitespace without changing legal meaning."""

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_metadata_value(
    metadata: dict[str, Any],
    key: str,
    default: str = "Unknown"
) -> str:
    """Safely read metadata values."""

    value = metadata.get(key, default)

    if value is None:
        return default

    return str(value)


def extract_section(text: str) -> str | None:
    """
    Try to identify a BNS/Act section number from the retrieved text.

    Examples:
        303. Theft
        304. Snatching
        303.—Theft
    """

    if not text:
        return None

    patterns = [
        r"\b(?:section\s*)?(\d{1,3})\s*[\.\-—:]\s*"
        r"(?:theft|snatching|murder|cheating|defamation|"
        r"robbery|assault|criminal intimidation)\b",

        r"\bsection\s+(\d{1,3})\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(1)

    return None


def legal_relevance_score(result: dict[str, Any]) -> float:
    """
    Obtain the retrieval score used by the retriever.

    This function deliberately does NOT create another ranking algorithm.
    The retriever remains responsible for ranking.
    """

    score = result.get("score")

    if score is None:
        score = result.get("retrieval_score", 0.0)

    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------
# Evidence selection
# ---------------------------------------------------------

def select_evidence(
    results: list[dict[str, Any]],
    max_results: int = 5
) -> list[dict[str, Any]]:
    """
    Select the strongest retrieved legal evidence.

    We keep this intentionally simple.

    The retriever already performs the ranking. Here we only:
        - remove empty documents
        - remove duplicate source/chunk combinations
        - keep the highest-ranked results
    """

    selected = []
    seen = set()

    for result in results:

        text = clean_text(result.get("text", ""))

        if not text:
            continue

        metadata = result.get("metadata") or {}

        source = get_metadata_value(
            metadata,
            "source",
            "Unknown"
        )

        chunk_index = get_metadata_value(
            metadata,
            "chunk_index",
            "Unknown"
        )

        key = (source, chunk_index)

        if key in seen:
            continue

        seen.add(key)

        result_copy = dict(result)

        result_copy["text"] = text

        selected.append(result_copy)

        if len(selected) >= max_results:
            break

    return selected


# ---------------------------------------------------------
# Context construction
# ---------------------------------------------------------

def build_context(
    evidence: list[dict[str, Any]]
) -> str:
    """
    Build a compact legal context for the LLM.

    The context contains:
        - source
        - Act
        - section/chunk
        - legal text
    """

    context_parts = []
    total_chars = 0

    for index, result in enumerate(evidence, start=1):

        metadata = result.get("metadata") or {}

        text = clean_text(
            result.get("text", "")
        )

        if not text:
            continue

        source = get_metadata_value(
            metadata,
            "source"
        )

        act_name = get_metadata_value(
            metadata,
            "act_name"
        )

        chunk_index = get_metadata_value(
            metadata,
            "chunk_index"
        )

        section = extract_section(text)

        section_text = (
            f"Section: {section}"
            if section
            else f"Chunk: {chunk_index}"
        )

        block = (
            f"SOURCE {index}\n"
            f"Act: {act_name}\n"
            f"Source document: {source}\n"
            f"{section_text}\n"
            f"Legal text:\n{text}\n"
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars

            if remaining <= 0:
                break

            block = block[:remaining]

        context_parts.append(block)

        total_chars += len(block)

        if total_chars >= MAX_CONTEXT_CHARS:
            break

    return "\n-----------------------------\n".join(
        context_parts
    )


# ---------------------------------------------------------
# Prompt creation
# ---------------------------------------------------------

def build_prompt(
    question: str,
    context: str
) -> str:
    """Create the final grounded prompt."""

    return f"""
{SYSTEM_INSTRUCTION}

USER QUESTION:
{question}

LEGAL CONTEXT:
{context}

Now answer the user's question.

Remember:
- Answer only from the supplied legal context.
- Keep it short.
- Mention the relevant section when supported.
- Do not include unrelated provisions.
- Do not invent missing information.
""".strip()


# ---------------------------------------------------------
# LLM generation
# ---------------------------------------------------------

def generate_answer_with_llm(
    question: str,
    context: str
) -> str:
    """
    Generate an answer using the configured LLM.

    Supported provider:
        OpenAI-compatible API

    Required environment variables:
        OPENAI_API_KEY
        OPENAI_MODEL

    The actual import is performed here so that the retrieval system
    can still be tested without importing an LLM client at module load.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    model = os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to your .env file before using LLM generation."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI package is not installed. "
            "Run: pip install openai"
        ) from exc

    client = OpenAI(
        api_key=api_key
    )

    prompt = build_prompt(
        question,
        context
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "The LLM returned an empty answer."
        )

    return answer.strip()


# ---------------------------------------------------------
# Fallback answer
# ---------------------------------------------------------

def build_fallback_answer(
    evidence: list[dict[str, Any]]
) -> str:
    """
    Safe fallback when an LLM is unavailable.

    This does NOT invent a legal answer.
    It simply exposes the strongest legal evidence.
    """

    if not evidence:
        return (
            "I couldn't find sufficient information in the "
            "available legal sources to answer this reliably."
        )

    first = evidence[0]

    metadata = first.get("metadata") or {}

    act_name = get_metadata_value(
        metadata,
        "act_name",
        "the available legal source"
    )

    source = get_metadata_value(
        metadata,
        "source",
        "unknown source"
    )

    text = clean_text(
        first.get("text", "")
    )

    text = text[:MAX_SOURCE_TEXT_CHARS]

    return (
        f"Based on {act_name}, the most relevant legal provision "
        f"available is:\n\n"
        f"{text}\n\n"
        f"Source: {source}"
    )


# ---------------------------------------------------------
# Source formatting
# ---------------------------------------------------------

def build_sources(
    evidence: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Return clean source metadata.

    Only selected evidence is returned as sources.
    """

    sources = []

    for result in evidence:

        metadata = dict(
            result.get("metadata") or {}
        )

        text = clean_text(
            result.get("text", "")
        )

        section = extract_section(text)

        retrieval_score = legal_relevance_score(
            result
        )

        source = {
            "source": metadata.get(
                "source",
                "Unknown"
            ),
            "act_name": metadata.get(
                "act_name",
                "Unknown"
            ),
            "year": metadata.get(
                "year",
                "Unknown"
            ),
            "authority": metadata.get(
                "authority",
                "Unknown"
            ),
            "document_type": metadata.get(
                "document_type",
                "Unknown"
            ),
            "chunk_index": metadata.get(
                "chunk_index",
                "Unknown"
            ),
            "section": section,
            "retrieval_score": round(
                retrieval_score,
                4
            ),
            "text": text[:MAX_SOURCE_TEXT_CHARS],
        }

        sources.append(source)

    return sources


# ---------------------------------------------------------
# Main RAG function
# ---------------------------------------------------------

def answer_question(
    question: str,
    top_k: int = TOP_K_DEFAULT
) -> dict[str, Any]:
    """
    Main JanNyaya AI RAG function.

    Pipeline:

        Question
            ↓
        Hybrid Retriever
            ↓
        Evidence Selection
            ↓
        Context Construction
            ↓
        LLM
            ↓
        Short Grounded Answer
            ↓
        Sources
    """

    if not question or not question.strip():

        return {
            "answer": (
                "Please enter a legal question."
            ),
            "sources": [],
        }

    question = question.strip()

    # -----------------------------------------------------
    # 1. Retrieval
    # -----------------------------------------------------

    results = search_documents(
        question,
        top_k=max(top_k, 5)
    )

    if not results:

        return {
            "answer": (
                "I couldn't find sufficient information in "
                "the available legal sources to answer this reliably."
            ),
            "sources": [],
        }

    # -----------------------------------------------------
    # 2. Evidence selection
    # -----------------------------------------------------

    evidence = select_evidence(
        results,
        max_results=min(max(top_k, 5), 5)
    )

    if not evidence:

        return {
            "answer": (
                "I couldn't find sufficient information in "
                "the available legal sources to answer this reliably."
            ),
            "sources": [],
        }

    # -----------------------------------------------------
    # 3. Build legal context
    # -----------------------------------------------------

    context = build_context(
        evidence
    )

    # -----------------------------------------------------
    # 4. Generate grounded answer
    # -----------------------------------------------------

    try:

        answer = generate_answer_with_llm(
            question,
            context
        )

    except Exception as error:

        print(
            f"[RAG] LLM generation unavailable: {error}"
        )

        answer = build_fallback_answer(
            evidence
        )

    # -----------------------------------------------------
    # 5. Sources
    # -----------------------------------------------------

    sources = build_sources(
        evidence
    )

    # -----------------------------------------------------
    # 6. Return final response
    # -----------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
    }