"""
JanNyaya AI - Grounded RAG Service

Pipeline:

    User Question
         ↓
    Hybrid Retriever
         ↓
    Structured Legal Fact Extractor
         ↓
    Compact Legal Context
         ↓
    Multilingual LLM
         ↓
    Grounded Answer
         ↓
    Sources

Supported:
    English
    Hindi
    Kannada
"""

import os
import time
from typing import List, Dict, Any, Optional

try:
    from backend.retriever import hybrid_search
    from backend.legal_fact_extractor import (
        detect_fact_intent,
        detect_legal_term,
        extract_legal_facts,
        build_fact_context,
    )
    from backend.llm_service import generate_answer
    from backend.legal_query_planner import plan_legal_query
    from backend.multi_hop_rag import execute_multi_hop_retrieval
    from backend.source_verifier import verify_sources
    from backend.evidence_graph import generate_evidence_graph
    from backend.grounding_guard import verify_grounding
except ImportError:
    from retriever import hybrid_search
    from legal_fact_extractor import (
        detect_fact_intent,
        detect_legal_term,
        extract_legal_facts,
        build_fact_context,
    )
    from llm_service import generate_answer
    from legal_query_planner import plan_legal_query
    from multi_hop_rag import execute_multi_hop_retrieval
    from source_verifier import verify_sources
    from evidence_graph import generate_evidence_graph
    from grounding_guard import verify_grounding


# ============================================================
# CONFIGURATION
# ============================================================

RAG_TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "10",
    )
)

RAG_MAX_FACT_GROUPS = int(
    os.getenv(
        "RAG_MAX_FACT_GROUPS",
        "3",
    )
)

RAG_MAX_CONTEXT_CHARS = int(
    os.getenv(
        "RAG_MAX_CONTEXT_CHARS",
        "9000",
    )
)


# ============================================================
# LANGUAGE
# ============================================================

def detect_language(
    text: str,
) -> str:

    if not text:
        return "english"

    devanagari = 0
    kannada = 0

    for char in text:

        code = ord(char)

        if 0x0900 <= code <= 0x097F:
            devanagari += 1

        elif 0x0C80 <= code <= 0x0CFF:
            kannada += 1

    if devanagari:
        return "hindi"

    if kannada:
        return "kannada"

    return "english"


# ============================================================
# DISCLAIMER
# ============================================================

def _disclaimer(
    language: str,
) -> str:

    if language == "hindi":

        return (
            "अस्वीकरण: यह जानकारी केवल उपलब्ध कानूनी "
            "दस्तावेजों पर आधारित है और व्यक्तिगत कानूनी "
            "सलाह नहीं है।"
        )

    if language == "kannada":

        return (
            "ಹಕ್ಕುತ್ಯಾಗ: ಈ ಮಾಹಿತಿಯು ಒದಗಿಸಲಾದ ಕಾನೂನು "
            "ದಾಖಲೆಗಳ ಆಧಾರದ ಮೇಲೆ ಮಾತ್ರ ನೀಡಲಾಗಿದೆ ಮತ್ತು "
            "ವೈಯಕ್ತಿಕ ಕಾನೂನು ಸಲಹೆಯಲ್ಲ."
        )

    return (
        "Disclaimer: This information is based solely on "
        "the supplied legal documents and is not personalized "
        "legal advice."
    )


# ============================================================
# SOURCES
# ============================================================

def build_sources(
    results: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build clean frontend source objects.

    Prefer structured legal facts because those represent
    the final evidence selected for the answer.
    """

    sources = []

    # --------------------------------------------------------
    # Structured fact sources
    # --------------------------------------------------------

    if facts:

        for fact in facts:

            sources.append(
                {
                    "section":
                        fact.get(
                            "section",
                            "",
                        ),

                    "section_title":
                        fact.get(
                            "section_title",
                            "",
                        ),

                    "source":
                        fact.get(
                            "source",
                            "Unknown",
                        ),

                    "title":
                        fact.get(
                            "title",
                            "",
                        ),

                    "document_type":
                        fact.get(
                            "document_type",
                            "",
                        ),

                    "chunk":
                        fact.get(
                            "best_chunk",
                            None,
                        ),

                    "score":
                        round(
                            float(
                                fact.get(
                                    "score",
                                    0.0,
                                )
                            ),
                            4,
                        ),
                }
            )

        return sources

    # --------------------------------------------------------
    # Retrieval fallback
    # --------------------------------------------------------

    seen = set()

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        metadata = result.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        section = metadata.get(
            "section_number",
            metadata.get(
                "section",
                "",
            ),
        )

        source = metadata.get(
            "source",
            "Unknown",
        )

        key = (
            f"{source}:"
            f"{section}"
        )

        if key in seen:
            continue

        seen.add(key)

        score = result.get(
            "hybrid_score",
            result.get(
                "score",
                0.0,
            ),
        )

        try:
            score = round(
                float(score),
                4,
            )
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        sources.append(
            {
                "section":
                    section,

                "section_title":
                    metadata.get(
                        "section_title",
                        "",
                    ),

                "source":
                    source,

                "title":
                    metadata.get(
                        "title",
                        "",
                    ),

                "document_type":
                    metadata.get(
                        "document_type",
                        "",
                    ),

                "chunk":
                    metadata.get(
                        "chunk_index",
                        None,
                    ),

                "score":
                    score,
            }
        )

        if len(sources) >= 5:
            break

    return sources


# ============================================================
# SIMPLE FACT FALLBACK
# ============================================================

def _build_fallback_from_facts(
    question: str,
    facts: List[Dict[str, Any]],
) -> str:
    """
    Deterministic fallback.

    This is used when Groq is temporarily unavailable.

    It does NOT invent legal information.
    """

    language = detect_language(
        question
    )

    if not facts:

        return (
            "The available legal documents do not contain "
            "enough information to answer this question."
            "\n\n"
            + _disclaimer(language)
        )

    fact = facts[0]

    section = str(
        fact.get(
            "section",
            "",
        )
    ).strip()

    intent = fact.get(
        "query_intent",
        "general",
    )

    punishment_facts = fact.get(
        "punishment_facts",
        [],
    )

    definition_facts = fact.get(
        "definition_facts",
        [],
    )

    condition_facts = fact.get(
        "condition_facts",
        [],
    )

    # ========================================================
    # ENGLISH
    # ========================================================

    if language == "english":

        if intent == "punishment":

            answer = (
                f"Section {section} of the "
                f"Bharatiya Nyaya Sanhita, 2023 "
                f"contains the punishment provision."
            )

            if punishment_facts:

                answer += (
                    "\n\n"
                    + punishment_facts[0]
                )

            if len(punishment_facts) > 1:

                answer += (
                    "\n\n"
                    + punishment_facts[1]
                )

            if condition_facts:

                answer += (
                    "\n\n"
                    + condition_facts[0]
                )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        if intent == "definition":

            if definition_facts:

                answer = (
                    f"Section {section} defines the offence "
                    f"as follows:\n\n"
                    f"{definition_facts[0]}"
                )

            else:

                answer = (
                    f"Section {section} is the relevant "
                    f"provision for this offence."
                )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

    # ========================================================
    # HINDI
    # ========================================================

    if language == "hindi":

        if intent == "punishment":

            answer = (
                f"धारा {section} में इस अपराध के "
                f"दंड का प्रावधान है।"
            )

            if punishment_facts:

                answer += (
                    "\n\n"
                    + punishment_facts[0]
                )

            if len(punishment_facts) > 1:

                answer += (
                    "\n\n"
                    + punishment_facts[1]
                )

            if condition_facts:

                answer += (
                    "\n\n"
                    + condition_facts[0]
                )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        if intent == "definition":

            if definition_facts:

                answer = (
                    f"धारा {section} के अनुसार:\n\n"
                    + definition_facts[0]
                )

            else:

                answer = (
                    f"धारा {section} इस अपराध से "
                    f"संबंधित प्रावधान है।"
                )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

    # ========================================================
    # KANNADA
    # ========================================================

    if language == "kannada":

        if intent == "punishment":

            answer = (
                f"ಸೆಕ್ಷನ್ {section} ರಲ್ಲಿ ಈ ಅಪರಾಧದ "
                f"ಶಿಕ್ಷೆಯ ಬಗ್ಗೆ ನಿಬಂಧನೆ ಇದೆ."
            )

            if punishment_facts:

                answer += (
                    "\n\n"
                    + punishment_facts[0]
                )

            if len(punishment_facts) > 1:

                answer += (
                    "\n\n"
                    + punishment_facts[1]
                )

            if condition_facts:

                answer += (
                    "\n\n"
                    + condition_facts[0]
                )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        if intent == "definition":

            if definition_facts:

                answer = (
                    f"ಸೆಕ್ಷನ್ {section} ಪ್ರಕಾರ:\n\n"
                    + definition_facts[0]
                )

            else:

                answer = (
                    f"ಸೆಕ್ಷನ್ {section} ಈ ಅಪರಾಧಕ್ಕೆ "
                    f"ಸಂಬಂಧಿಸಿದ ನಿಬಂಧನೆಯಾಗಿದೆ."
                )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

    return (
        "The available legal evidence has been retrieved, "
        "but a concise answer could not be generated."
        "\n\n"
        + _disclaimer(language)
    )


# ============================================================
# SMART FOLLOW-UP GENERATOR
# ============================================================

def generate_smart_followups(
    question: str,
    facts: List[Dict[str, Any]],
    language: str,
) -> List[str]:
    """Generate 3 contextual follow-up legal queries."""
    lang = (language or "english").lower().strip()
    q_lower = (question or "").lower()

    # Kannada followups
    if lang in ("kannada", "kn"):
        if any(w in q_lower for w in ["ಕಳ್ಳತನ", "ಚೋರಿ", "ಅಪರಾಧ", "ಕೊಲೆ", "ಶಿಕ್ಷೆ", "theft", "murder"]):
            return [
                "ಈ ಅಪರಾಧಕ್ಕೆ ಬಿಎನ್‌ಎಸ್ (BNS 2023) ಅಡಿಯಲ್ಲಿ ನಿಖರ ಶಿಕ್ಷೆ ಮತ್ತು ದಂಡ ಏನು?",
                "ಪೊಲೀಸ್ ಠಾಣೆಯಲ್ಲಿ ಜೀರೋ ಎಫ್‌ಐಆರ್ ಅಥವಾ ಇ-ಎಫ್‌ಐಆರ್ ದಾಖಲಿಸುವ ವಿಧಾನವೇನು?",
                "ನಲ್ಸಾ (NALSA 15100) ಉಚಿತ ಕಾನೂನು ನೆರವು ಪಡೆಯುವುದು ಹೇಗೆ?",
            ]
        elif any(w in q_lower for w in ["ಚೆಕ್", "ಸಾಲ", "ಬ್ಯಾಂಕ್", "ನೋಟಿಸ್", "cheque", "loan"]):
            return [
                "ಸೆಕ್ಷನ್ 138 ರ ಅಡಿಯಲ್ಲಿ 30 ದಿನಗಳ ಶಾಸನಬದ್ಧ ನೋಟಿಸ್ ಪ್ರಕ್ರಿಯೆ ಏನು?",
                "ಸಾಲ ವಸೂಲಾತಿ ಅಥವಾ ಲೋಕ ಅದಾಲತ್‌ನಲ್ಲಿ ರಾಜಿ ಮಾಡಿಕೊಳ್ಳಲು ಅವಕಾಶವಿದೆಯೇ?",
                "ಬ್ಯಾಂಕ್ ನೋಟಿಸ್‌ಗೆ ಲಿಖಿತ ಉತ್ತರ ನೀಡುವ ಕಾನೂನು ಕ್ರಮಗಳೇನು?",
            ]
        return [
            "ಈ ಕಾನೂನು ನಿಯಮ ನನ್ನ ಸನ್ನಿವೇಶಕ್ಕೆ ಹೇಗೆ ಅನ್ವಯವಾಗುತ್ತದೆ?",
            "ನನ್ನ ಪರವಾಗಿ ಯಾವ ದಾಖಲೆಗಳನ್ನು ಮತ್ತು ಪುರಾವೆಗಳನ್ನು ಇಟ್ಟುಕೊಳ್ಳಬೇಕು?",
            "ನಲ್ಸಾ (NALSA 15100) ಉಚಿತ ಕಾನೂನು ಸಹಾಯವಾಣಿಗೆ ಕರೆ ಮಾಡುವುದು ಹೇಗೆ?",
        ]

    # Hindi followups
    if lang in ("hindi", "hi"):
        if any(w in q_lower for w in ["चोरी", "सजा", "अपराध", "हत्या", "theft", "crime"]):
            return [
                "भारतीय न्याय संहिता 2023 (BNS) के तहत इस अपराध में क्या सजा और जुर्माना है?",
                "बीएनएसएस (BNSS 173) के तहत ई-एफआईआर या जीरो एफआईआर कैसे दर्ज करें?",
                "नालसा (NALSA 15100) से मुफ्त कानूनी सहायता कैसे प्राप्त करें?",
            ]
        elif any(w in q_lower for w in ["चेक", "ऋण", "लोन", "बैंक", "नोटिस", "cheque", "loan"]):
            return [
                "एनआई एक्ट की धारा 138 के तहत 30 दिन के कानूनी नोटिस की प्रक्रिया क्या है?",
                "क्या चेक बाउंस मामले में लोक अदालत में समझौता किया जा सकता है?",
                "बैंक या वित्तीय संस्थान के नोटिस का कानूनी जवाब कैसे दें?",
            ]
        return [
            "यह कानूनी प्रावधान मेरी स्थिति में कैसे लागू हो सकता है?",
            "अपने बचाव में कौन से दस्तावेज़ और साक्ष्य सुरक्षित रखने चाहिए?",
            "नालसा (NALSA 15100) राष्ट्रीय कानूनी सेवा हेल्पलाइन से संपर्क कैसे करें?",
        ]

    # English followups
    if any(w in q_lower for w in ["theft", "punishment", "bns", "crime", "fir", "bail", "arrest"]):
        return [
            "What is the statutory punishment and bailability under BNS 2023?",
            "What is the procedure to register a Zero FIR or e-FIR under Section 173 of BNSS?",
            "How can a citizen get free legal aid from NALSA via National Helpline 15100?",
        ]
    elif any(w in q_lower for w in ["cheque", "bounce", "138", "loan", "debt", "recovery", "notice"]):
        return [
            "What is the mandatory 30-day statutory notice procedure under Section 138 NI Act?",
            "Can this financial claim be settled through Lok Adalat or pre-litigation mediation?",
            "What are the legally valid defences against an unsubstantiated demand notice?",
        ]
    elif any(w in q_lower for w in ["consumer", "refund", "defective", "service", "daakhil"]):
        return [
            "How do I file an online consumer grievance on the e-Daakhil portal?",
            "What compensation or punitive damages can be claimed under Consumer Protection Act 2019?",
            "What is the statutory limitation period (2 years) to file a consumer complaint?",
        ]

    return [
        "Does this statutory provision apply to first-time occurrences or specific civil agreements?",
        "What essential documents and electronic evidence should I preserve?",
        "How do I contact the District Legal Services Authority (DLSA) / NALSA (15100) for free aid?",
    ]


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    question: str,
    history: Optional[Any] = None,
    mode: str = "citizen",
) -> Dict[str, Any]:

    t_start = time.time()

    if not isinstance(history, list):
        history = []

    if not question or not question.strip():
        return {
            "status": "error",
            "question": question,
            "answer": "Please enter a legal question.",
            "sources": [],
            "query_plan": None,
            "evidence_graph": None,
            "grounding": None,
            "followups": [],
            "latency": {"total_ms": 0},
        }

    question = question.strip()

    print()
    print("=" * 70)
    print("JAN NYAYA AI - ADVANCED GROUNDED LEGAL RAG PIPELINE")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Consultation Mode: {mode.upper()}")

    # 1. ADVANCED LEGAL QUERY PLANNER
    t_plan_start = time.time()
    plan = plan_legal_query(question)
    t_plan_ms = int((time.time() - t_plan_start) * 1000)

    print(f"[Query Plan] Type: {plan.query_type} | Domain: {plan.primary_domain} | Route: {plan.primary_route} | Multi-Hop: {plan.is_multi_hop}")

    search_query = question
    if history and isinstance(history, list) and len(history) > 0:
        recent_user_queries = [
            str(h.get("content", "")) for h in history if h.get("role") == "user" and str(h.get("content", "")).strip()
        ]
        if recent_user_queries:
            last_query = recent_user_queries[-1]
            if len(question.split()) < 7 or any(w in question.lower() for w in ["what if", "is it", "can they", "how about", "in this case", "punishment", "bail", "arrest", "penalty", "liable"]):
                search_query = f"{last_query} {question}"
                print(f"Contextualized multi-turn search query: {search_query}")

    # 2. RETRIEVAL (MULTI-HOP or SINGLE-HOP HYBRID SEARCH)
    t_ret_start = time.time()
    try:
        if plan.is_multi_hop and len(plan.sub_queries) > 1:
            print(f"[RAG] Executing Multi-Hop RAG across {len(plan.sub_queries)} statutory sub-queries...")
            results = execute_multi_hop_retrieval(
                plan=plan,
                semantic_k=30,
                bm25_k=30,
                final_k=RAG_TOP_K,
            )
        else:
            print("[RAG] Executing Single-Hop Hybrid Retrieval...")
            results = hybrid_search(
                search_query,
                semantic_k=30,
                bm25_k=30,
                final_k=RAG_TOP_K,
            )
    except Exception as error:
        print("RETRIEVAL ERROR:", type(error).__name__, str(error))
        return {
            "status": "error",
            "question": question,
            "answer": "An error occurred while retrieving the legal documents.",
            "sources": [],
            "query_plan": plan.to_dict(),
            "evidence_graph": None,
            "grounding": None,
            "followups": [],
            "latency": {"total_ms": int((time.time() - t_start) * 1000)},
        }

    t_ret_ms = int((time.time() - t_ret_start) * 1000)
    print(f"Retrieved results: {len(results)} (in {t_ret_ms} ms)")

    language = detect_language(question)

    if not results:
        return {
            "status": "success",
            "question": question,
            "answer": (
                "The available legal documents do not contain relevant information."
                "\n\n"
                + _disclaimer(language)
            ),
            "sources": [],
            "query_plan": plan.to_dict(),
            "evidence_graph": None,
            "grounding": {
                "confidence_level": "Insufficient Evidence",
                "grounding_score": 0.0,
                "notes": "No matching statutory documents found.",
            },
            "followups": generate_smart_followups(question, [], language),
            "latency": {"total_ms": int((time.time() - t_start) * 1000)},
        }

    # 3. STRUCTURED LEGAL FACT EXTRACTION
    try:
        facts = extract_legal_facts(
            question,
            results,
            max_groups=RAG_MAX_FACT_GROUPS,
        )
    except Exception as error:
        print("FACT EXTRACTION ERROR:", type(error).__name__, str(error))
        facts = []

    # Fallback to top direct evidence chunks if extractor returns 0
    if not facts and results:
        print("Building direct evidence facts from top retrieved results...")
        for result in results[:3]:
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            sec = str(metadata.get("section_number", "")).strip()
            title = str(metadata.get("section_title", "")).strip()
            doc_text = str(result.get("document", "")).strip()
            if not doc_text:
                continue

            fallback_fact = {
                "section": sec,
                "section_title": title,
                "source": metadata.get("source", "Official Gazette"),
                "title": metadata.get("title", metadata.get("act_name", "Statutory Act")),
                "document_type": metadata.get("document_type", "Act"),
                "year": metadata.get("year", None),
                "best_chunk": metadata.get("chunk_index", None),
                "direct_offence": True,
                "query_intent": plan.query_type.lower(),
                "legal_term": plan.primary_domain,
                "score": float(result.get("legal_rerank_score", result.get("score", 1.0))),
                "punishment_facts": [],
                "definition_facts": [doc_text],
                "condition_facts": [],
            }
            lower_doc = doc_text.lower()
            if "shall be punished" in lower_doc or "punished with" in lower_doc or "penalty" in lower_doc:
                fallback_fact["punishment_facts"] = [doc_text]
            facts.append(fallback_fact)

    # 4. LEGAL SOURCE VERIFICATION & CITATION HIERARCHY
    verified_sources = verify_sources(results, facts)

    context = build_fact_context(
        facts,
        max_characters=RAG_MAX_CONTEXT_CHARS,
    )

    # 5. GROUNDED ANSWER GENERATION
    t_llm_start = time.time()
    if not context:
        answer = _build_fallback_from_facts(question, facts)
    else:
        try:
            answer = generate_answer(
                question,
                context,
                history=history,
            )
        except Exception as error:
            print("LLM ERROR:", type(error).__name__, str(error))
            answer = _build_fallback_from_facts(question, facts)

    if not answer or not str(answer).strip():
        answer = _build_fallback_from_facts(question, facts)

    t_llm_ms = int((time.time() - t_llm_start) * 1000)

    # 6. GROUNDING GUARD & EVIDENCE GRAPH
    grounding_report = verify_grounding(str(answer), verified_sources, facts)
    evidence_graph = generate_evidence_graph(
        question=question,
        sources=verified_sources,
        facts=facts,
        domain=plan.primary_domain,
    )
    followups = generate_smart_followups(question, facts, language)

    t_total_ms = int((time.time() - t_start) * 1000)

    print(f"RAG complete. Latency: Plan={t_plan_ms}ms, Retrieval={t_ret_ms}ms, LLM={t_llm_ms}ms, Total={t_total_ms}ms")
    print(f"Grounding Confidence: {grounding_report.get('confidence_level')} (Score: {grounding_report.get('grounding_score')})")
    print("=" * 70)

    return {
        "status": "success",
        "question": question,
        "answer": str(answer).strip(),
        "sources": verified_sources,
        "query_plan": plan.to_dict(),
        "evidence_graph": evidence_graph,
        "grounding": grounding_report,
        "followups": followups,
        "latency": {
            "planning_ms": t_plan_ms,
            "retrieval_ms": t_ret_ms,
            "llm_ms": t_llm_ms,
            "total_ms": t_total_ms,
        },
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

def main() -> None:

    questions = [
        "What is the punishment for theft?",
        "What is theft?",
        "चोरी की सजा क्या है?",
        "चोरी क्या है?",
        "ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ ಏನು?",
        "ಕಳ್ಳತನ ಎಂದರೇನು?",
    ]

    print()
    print(
        "JanNyaya AI - Grounded RAG Service Test"
    )

    for question in questions:

        try:

            result = answer_question(
                question
            )

            print()
            print("=" * 70)
            print("QUESTION")
            print("=" * 70)
            print(
                result["question"]
            )

            print()
            print("=" * 70)
            print("ANSWER")
            print("=" * 70)
            print(
                result["answer"]
            )

            print()
            print("=" * 70)
            print("SOURCES")
            print("=" * 70)

            for source in result.get(
                "sources",
                [],
            ):

                print(
                    f"Section {source.get('section')} | "
                    f"{source.get('source')} | "
                    f"Score {source.get('score')}"
                )

        except Exception as error:

            print()
            print(
                "TEST ERROR:",
                type(error).__name__,
                str(error),
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()