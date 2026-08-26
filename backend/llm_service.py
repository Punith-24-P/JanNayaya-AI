"""
JanNyaya AI - Friendly Precision Multilingual Legal LLM

Responsibilities
----------------
1. Receive a legal question.
2. Receive verified structured legal facts.
3. Detect English / Hindi / Kannada.
4. Ask Groq to explain only supported legal facts.
5. Produce a natural, citizen-friendly answer.
6. Preserve legal accuracy and conditions.
7. Provide a deterministic fallback if Groq is unavailable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PROJECT ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_FILE
)


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing from .env"
    )

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

FALLBACK_MODELS = [
    GROQ_MODEL,
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]

MAX_TOKENS = int(
    os.getenv(
        "LLM_MAX_TOKENS",
        "1500",
    )
)


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY,
    timeout=60.0,
)


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text: str) -> str:
    if not text:
        return "english"

    hindi_count = 0
    kannada_count = 0

    for character in str(text):

        code = ord(character)

        if 0x0900 <= code <= 0x097F:
            hindi_count += 1

        elif 0x0C80 <= code <= 0x0CFF:
            kannada_count += 1

    if kannada_count > hindi_count:
        return "kannada"

    if hindi_count > 0:
        return "hindi"

    return "english"


def _language_name(language: str) -> str:
    return {
        "english": "English",
        "hindi": "Hindi",
        "kannada": "Kannada",
    }.get(
        language,
        "English",
    )


# ============================================================
# DISCLAIMER
# ============================================================

def _disclaimer(language: str) -> str:

    if language == "hindi":
        return (
            "**अस्वीकरण:** यह सामान्य कानूनी जानकारी है, जो उपलब्ध वैधानिक दस्तावेजों पर आधारित है। "
            "यह व्यक्तिगत कानूनी सलाह अथवा अंतिम न्यायिक निर्णय नहीं है।"
        )

    if language == "kannada":
        return (
            "**ಹಕ್ಕುತ್ಯಾಗ:** ಇದು ಸಾಮಾನ್ಯ ಕಾನೂನು ಮಾಹಿತಿ ಮಾತ್ರ. "
            "ಇದು ವೈಯಕ್ತಿಕ ವಕಾಲತ್ತು ಸಲಹೆ ಅಥವಾ ಅಂತಿಮ ನ್ಯಾಯಾಲಯದ ತೀರ್ಮಾನವಲ್ಲ."
        )

    return (
        "**Disclaimer:** This is general legal information based on retrieved statutory provisions. "
        "It does not constitute personalized legal counsel or a formal judicial opinion."
    )


# ============================================================
# ANSWER CLEANING
# ============================================================

def _clean_answer(
    answer: str,
    language: str,
) -> str:

    if not answer:
        return ""

    answer = str(answer).strip()

    # Strip <think>...</think> reasoning tags completely
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()

    # If response is for Kannada/Hindi, strip any leading English monologue
    if language in ("kannada", "hindi"):
        indic_pattern = r"[\u0C80-\u0CFF]" if language == "kannada" else r"[\u0900-\u097F]"
        indic_match = re.search(indic_pattern, answer)
        if indic_match and indic_match.start() > 20:
            preceding = answer[:indic_match.start()]
            if any(k in preceding.lower() for k in ["we need", "the question", "we have", "the context", "in this response", "means:"]):
                answer = answer[indic_match.start():].strip()

    # Remove code fences accidentally returned by the model.
    answer = re.sub(
        r"^```(?:text|markdown)?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    answer = re.sub(
        r"\s*```$",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    # Strip opening robotic monologue sentences repeatedly until clean
    for _ in range(8):
        clean_prev = answer
        robotic_prefixes = [
            r"^\s*(?:we need to answer|we must answer|to answer the question|the user asks|the user is asking)\s*:\s*(?:\"[^\"]*\"|'[^']*'|[^\n]*)\s*",
            r"^\s*we need to answer\s+(?:based on|the question|using|from)[^\n]*?[\.:\-]\s*",
            r"^\s*use provided legal context\s*[\.:\-]?\s*",
            r"^\s*the context includes\s+[^\n]*?[\.:\-]\s*",
            r"^\s*the context is\s+[^\n]*?[\.:\-]\s*",
            r"^\s*we have legal context\s*[\.:\-]?\s*",
            r"^\s*according to the provided legal context\s*[\.:,-]?\s*",
            r"^\s*based on the verified legal context\s*[\.:,-]?\s*",
            r"^\s*it mentions punishment for\s+[^\n]*?[\.:\-]\s*",
            r"^\s*it seems the provision\s+[^\n]*?[\.:\-]\s*",
            r"^\s*in this response\s*,?\s*(?:we will|i will)[^\n]*?[\.:\-]\s*",
            r"^\s*let(?:'s| us) examine\s+[^\n]*?[\.:\-]\s*",
            r"^\s*so the answer is\s*:\s*",
        ]
        for p in robotic_prefixes:
            answer = re.sub(p, "", answer, flags=re.IGNORECASE).strip()
        if answer == clean_prev:
            break

    # Remove robotic meta openings.
    meta_patterns = [
        r"^\s*verified answer\s*:\s*",
        r"^\s*verified information\s*:\s*",
        r"^\s*verified punishment information\s*:\s*",
        r"^\s*relevant provision identified\s*:\s*",
        r"^\s*the relevant provision is\s*:\s*",
        r"^\s*according to the evidence provided\s*[:,]?\s*",
        r"^\s*based on the retrieved facts\s*[:,]?\s*",
        r"^\s*the available legal evidence(?: indicates)?\s*[:,]?\s*",
        r"^\s*system analysis shows\s*[:,]?\s*",
    ]

    for pattern in meta_patterns:
        answer = re.sub(
            pattern,
            "",
            answer,
            flags=re.IGNORECASE,
        )

    # Strip duplicate disclaimers strictly if they occur as a dedicated ending block
    disclaimer_end_patterns = [
        r"\n+(?:---\s*\n+)?(?:⚖️\s*)?(?:\*\*|\*)?(?:disclaimer|अस्वीकरण|ಹಕ್ಕುತ್ಯಾಗ)(?:\*\*|\*)?\s*:[^\n]*(?:\n+[^\n]*)*$",
    ]

    for pat in disclaimer_end_patterns:
        # Only strip if it contains standard legal disclaimer wording
        match = re.search(pat, answer, flags=re.IGNORECASE)
        if match and any(w in match.group(0).lower() for w in ["general legal", "personalized", "judicial opinion", "ವೈಯಕ್ತಿಕ", "ಕಾನೂನು ಮಾಹಿತಿ", "ನ್ಯಾಯಾಲಯದ", "कानूनी जानकारी", "न्यायिक निर्णय"]):
            answer = answer[:match.start()].strip()

    return answer.strip()


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are JanNyaya AI, a highly knowledgeable, compassionate, and authoritative legal assistant for Indian citizens.
Your mission is to empower citizens by giving clear, friendly, structured, and legally accurate answers to their queries and real-world legal situations.

CORE PRINCIPLES:
1. CITIZEN-FIRST EMPATHY & CLARITY:
   - Speak directly to the citizen with warmth, clarity, and reassurance.
   - Explain legal terms in simple, plain language while retaining precise statutory citations.
   - Never output internal thinking, scratchpad reasoning, or robotic meta-phrases (e.g. "We need to answer", "Based on the context").

2. STRUCTURED 5-PART LEGAL GUIDANCE:
   Structure your answer using the following clear sections:
   - **Quick Summary & Legal Standing**: A 1-2 sentence empathetic summary explaining what the law says about the citizen's situation.
   - **Applicable Legal Provisions**: Exact Indian Act name (e.g., Bharatiya Nyaya Sanhita 2023, Consumer Protection Act 2019, Negotiable Instruments Act 1881, Banking Regulation Act 1949, Information Technology Act 2000) and Section numbers with clear explanation of the legal elements.
   - **Penalties, Fines & Legal Remedies**: Specific imprisonment terms, fine limits, community service, monetary compensation, or civil orders provided by law.
   - **Step-by-Step Action Plan (What to do right now)**: Practical checklist (e.g., preserve receipts/CCTV/bank statements, send statutory notice, file an FIR or online consumer complaint on NCH/e-Daakhil, approach DLSA/banking ombudsman/court).
   - **Citizen Rights & Essential Tips**: Limitation periods, bailable/non-bailable status, fee waivers, or free legal aid helpline (NALSA/DLSA 15100).

3. MULTILINGUAL ACCURACY:
   - English: Polished, warm, structured conversational English.
   - Hindi (हिन्दी): Respond completely in natural, grammatically pure Hindi with Hindi headers (e.g. **त्वरित सारांश एवं कानूनी स्थिति**, **लागू होने वाले कानूनी प्रावधान**, **सजा, जुर्माना एवं कानूनी उपचार**, **चरणबद्ध कार्य योजना (अभी क्या करें)**, **नागरिक अधिकार एवं जरूरी सलाह**).
   - Kannada (ಕನ್ನಡ): Respond completely in natural, grammatically pure Kannada with Kannada headers (e.g. **ತ್ವರಿತ ಸಾರಾಂಶ ಮತ್ತು ಕಾನೂನು ಸ್ಥಿತಿ**, **ಅನ್ವಯವಾಗುವ ಕಾನೂನು ವಿಧಿಗಳು**, **ಶಿಕ್ಷೆ, ದಂಡ ಮತ್ತು ಪರಿಹಾರಗಳು**, **ಹಂತ-ಹಂತದ ಕ್ರಿಯಾ ಯೋಜನೆ (ಈಗ ಏನು ಮಾಡಬೇಕು)**, **ನಾಗರಿಕರ ಹಕ್ಕುಗಳು ಮತ್ತು ಸಲಹೆಗಳು**).

4. GROUNDING & FIDELITY:
   - Base definitions, sections, and punishments strictly on the provided verified statutory context.
   - Always retain distinguishing legal nuances (such as mandatory vs discretionary relief, time limits).
"""


# ============================================================
# USER PROMPT
# ============================================================

def _build_prompt(
    question: str,
    facts_context: str,
) -> str:

    language = detect_language(
        question
    )

    if language == "hindi":
        return f"""नागरिक का प्रश्न:
"{question}"

निर्देश:
इस प्रश्न का उत्तर केवल और केवल शुद्ध, सरल एवं स्पष्ट हिन्दी में दीजिए।
कृपया उत्तर को 5 स्पष्ट भागों में संरचित करें:
1. **त्वरित सारांश एवं कानूनी स्थिति** (नागरिक के लिए सीधा व सरल स्पष्टीकरण)
2. **लागू होने वाले कानूनी प्रावधान** (अधिनियम का नाम व धारा संख्या)
3. **सजा, जुर्माना एवं कानूनी उपचार** (सजा की अवधि, जुर्माना या मुआवजा)
4. **चरणबद्ध कार्य योजना (अभी क्या करें)** (सबूत जुटाना, शिकायत/एफआईआर दर्ज करना आदि)
5. **नागरिक अधिकार एवं जरूरी सलाह** (समय सीमा, कानूनी सहायता 15100 आदि)

प्रामाणिक कानूनी संदर्भ:
---------------- कानूनी संदर्भ ----------------
{facts_context}
----------------- संदर्भ समाप्त -----------------

अब नागरिक के लिए सीधा, स्पष्ट और व्यावहारिक हिन्दी में उत्तर लिखिए:"""

    if language == "kannada":
        return f"""ನಾಗರಿಕರ ಪ್ರಶ್ನೆ:
"{question}"

ಸೂಚನೆ:
ಈ ಪ್ರಶ್ನೆಗೆ ನೇರವಾಗಿ ಶುದ್ಧ, ಸರಳ ಮತ್ತು ನಿಖರವಾದ ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಿ.
ದಯವಿಟ್ಟು ಉತ್ತರವನ್ನು 5 ಸ್ಪಷ್ಟ ವಿಭಾಗಗಳಲ್ಲಿ ರಚಿಸಿ:
1. **ತ್ವರಿತ ಸಾರಾಂಶ ಮತ್ತು ಕಾನೂನು ಸ್ಥಿತಿ** (ನಾಗರಿಕರಿಗೆ ಸರಳ ವಿವರಣೆ)
2. **ಅನ್ವಯವಾಗುವ ಕಾನೂನು ವಿಧಿಗಳು** (ಕಾನೂನಿನ ಹೆಸರು ಮತ್ತು ಸೆಕ್ಷನ್ ಸಂಖ್ಯೆ)
3. **ಶಿಕ್ಷೆ, ದಂಡ ಮತ್ತು ಪರಿಹಾರಗಳು** (ಜೈಲು ಶಿಕ್ಷೆ, ದಂಡ ಅಥವಾ ಪರಿಹಾರ)
4. **ಹಂತ-ಹಂತದ ಕ್ರಿಯಾ ಯೋಜನೆ (ಈಗ ಏನು ಮಾಡಬೇಕು)** (ಸಾಕ್ಷ್ಯ ಸಂಗ್ರಹ, ದೂರು/ಎಫ್‌ಐಆರ್ ದಾಖಲಿಸುವುದು)
5. **ನಾಗರಿಕರ ಹಕ್ಕುಗಳು ಮತ್ತು ಸಲಹೆಗಳು** (ಸಮಯ ಮಿತಿ, ಉಚಿತ ಕಾನೂನು ನೆರವು 15100)

ಪ್ರಾಮಾಣಿಕ ಕಾನೂನು ಮಾಹಿತಿ:
---------------- ಕಾನೂನು ಮಾಹಿತಿ ----------------
{facts_context}
----------------- ಮಾಹಿತಿ ಮುಕ್ತಾಯ -----------------

ಈಗ ನಾಗರಿಕರಿಗಾಗಿ ನೇರ, ಸ್ಪಷ್ಟ ಮತ್ತು ನಿಖರವಾದ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ:"""

    return f"""You are advising an Indian citizen on the following legal question or real-world situation:
\"{question}\"

Please provide a compassionate, practical, and authoritative answer structured into the following 5 parts:
1. **Quick Summary & Legal Standing** (Empathetic, clear breakdown of what the law says)
2. **Applicable Legal Provisions** (Exact Act name and Section numbers with core legal elements)
3. **Penalties, Fines & Legal Remedies** (Specific imprisonment terms, fine limits, community service, or civil remedies)
4. **Step-by-Step Action Plan (What to do right now)** (Actionable checklist: evidence preservation, notice, filing complaint/FIR, authorities to contact)
5. **Citizen Rights & Essential Tips** (Limitation periods, bail status, free legal aid helpline 15100)

---------------- BEGIN STATUTORY CONTEXT ----------------
{facts_context}
----------------- END STATUTORY CONTEXT -----------------

Write the complete, citizen-friendly response directly now:
"""

# ============================================================
# CONTEXT HELPERS
# ============================================================

def _extract_section(
    facts_context: str,
) -> str:

    if not facts_context:
        return ""

    patterns = [
        r"Section:\s*([A-Za-z0-9\-]+)",
        r"section:\s*([A-Za-z0-9\-]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            facts_context,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return ""


def _extract_fact_lines(
    facts_context: str,
    keywords,
) -> List[str]:

    if not facts_context:
        return []

    if isinstance(
        keywords,
        str,
    ):
        keywords = [keywords]

    lines = []

    for raw_line in facts_context.splitlines():

        line = raw_line.strip()

        if not line.startswith("-"):
            continue

        cleaned = line[1:].strip()

        if not cleaned:
            continue

        if any(
            str(keyword).lower()
            in cleaned.lower()
            for keyword in keywords
        ):
            lines.append(
                cleaned
            )

    return lines


def _unique_lines(
    lines: List[str],
) -> List[str]:

    result = []
    seen = set()

    for line in lines:

        normalized = re.sub(
            r"\s+",
            " ",
            str(line).strip().lower(),
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        result.append(
            str(line).strip()
        )

    return result


# ============================================================
# GENERIC FRIENDLY FALLBACK
# ============================================================

def _generic_fallback(
    question: str,
    facts_context: str,
) -> str:

    language = detect_language(
        question
    )

    section = _extract_section(
        facts_context
    )

    punishment_lines = _unique_lines(
        _extract_fact_lines(
            facts_context,
            [
                "punished",
                "punishment",
            ],
        )
    )

    definition_lines = _unique_lines(
        _extract_fact_lines(
            facts_context,
            [
                "means",
                "is said to",
                "defined",
            ],
        )
    )

    # --------------------------------------------------------
    # ENGLISH
    # --------------------------------------------------------

    if language == "english":

        if punishment_lines:

            opening = (
                "Here's the key point."
            )

            if section:

                opening = (
                    f"Sure. The retrieved legal material "
                    f"for Section {section} says:"
                )

            answer = (
                opening
                + "\n\n"
                + "\n".join(
                    f"• {line}"
                    for line in punishment_lines[:3]
                )
            )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        if definition_lines:

            opening = (
                "In simple terms, this provision explains:"
            )

            if section:

                opening = (
                    f"In simple terms, Section {section} "
                    f"explains:"
                )

            answer = (
                opening
                + "\n\n"
                + definition_lines[0]
            )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        return (
            "I found some relevant legal information, "
            "but the available material does not contain "
            "enough detail to give you a reliable answer yet."
            "\n\n"
            + _disclaimer(language)
        )

    # --------------------------------------------------------
    # HINDI
    # --------------------------------------------------------

    if language == "hindi":

        if punishment_lines:

            opening = "मुख्य बात यह है:"

            if section:

                opening = (
                    f"धारा {section} से जुड़ी मुख्य बात यह है:"
                )

            answer = (
                opening
                + "\n\n"
                + "\n".join(
                    f"• {line}"
                    for line in punishment_lines[:3]
                )
            )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        if definition_lines:

            opening = (
                "सरल शब्दों में, इसका मतलब है:"
            )

            if section:

                opening = (
                    f"सरल शब्दों में, धारा {section} "
                    f"का मतलब है:"
                )

            answer = (
                opening
                + "\n\n"
                + definition_lines[0]
            )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        return (
            "मुझे कुछ संबंधित कानूनी जानकारी मिली है, "
            "लेकिन उपलब्ध सामग्री के आधार पर अभी "
            "विश्वसनीय उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"
            "\n\n"
            + _disclaimer(language)
        )

    # --------------------------------------------------------
    # KANNADA
    # --------------------------------------------------------

    if language == "kannada":

        if punishment_lines:

            opening = (
                "ಮುಖ್ಯ ವಿಷಯ ಏನೆಂದರೆ:"
            )

            if section:

                opening = (
                    f"ಸೆಕ್ಷನ್ {section}ಗೆ ಸಂಬಂಧಿಸಿದ "
                    f"ಮುಖ್ಯ ವಿಷಯ ಏನೆಂದರೆ:"
                )

            answer = (
                opening
                + "\n\n"
                + "\n".join(
                    f"• {line}"
                    for line in punishment_lines[:3]
                )
            )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        if definition_lines:

            opening = (
                "ಸರಳವಾಗಿ ಹೇಳುವುದಾದರೆ:"
            )

            if section:

                opening = (
                    f"ಸರಳವಾಗಿ ಹೇಳುವುದಾದರೆ, ಸೆಕ್ಷನ್ "
                    f"{section}ರ ಅರ್ಥ:"
                )

            answer = (
                opening
                + "\n\n"
                + definition_lines[0]
            )

            return (
                answer
                + "\n\n"
                + _disclaimer(language)
            )

        return (
            "ಸಂಬಂಧಿತ ಕಾನೂನು ಮಾಹಿತಿ ಸಿಕ್ಕಿದೆ. ಆದರೆ "
            "ಲಭ್ಯವಿರುವ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಈಗ "
            "ವಿಶ್ವಾಸಾರ್ಹ ಉತ್ತರ ನೀಡಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ."
            "\n\n"
            + _disclaimer(language)
        )

    return (
        "I found relevant legal information, but "
        "there is not enough verified material to "
        "give you a reliable answer."
        "\n\n"
        + _disclaimer(language)
    )


# ============================================================
# FRIENDLY THEFT FALLBACK
# ============================================================

def _fallback_theft(
    question: str,
    facts_context: str,
):

    language = detect_language(
        question
    )

    section = _extract_section(
        facts_context
    )

    if section != "303":
        return None

    lower_context = facts_context.lower()

    is_punishment = any(
        phrase in lower_context
        for phrase in (
            "punishment provisions:",
            "shall be punished",
            "punished with",
        )
    )

    is_definition = (
        "definition provisions:"
        in lower_context
    )

    # --------------------------------------------------------
    # ENGLISH
    # --------------------------------------------------------

    if language == "english":

        if is_punishment:

            parts = []

            if (
                "may extend to three years"
                in lower_context
                or
                "or with fine, or with both"
                in lower_context
            ):

                parts.append(
                    "Sure. Under Section 303 of the "
                    "Bharatiya Nyaya Sanhita, 2023, "
                    "theft can be punished with imprisonment "
                    "of either description for up to three "
                    "years, or with a fine, or with both."
                )

            if (
                "second or subsequent conviction"
                in lower_context
                and
                "five years"
                in lower_context
                and
                "one year"
                in lower_context
            ):

                parts.append(
                    "If it is a second or later conviction "
                    "under this section, the punishment is "
                    "more serious: rigorous imprisonment "
                    "of at least one year and up to five years, "
                    "along with a fine."
                )

            if (
                "less than five thousand rupees"
                in lower_context
                and
                "first time"
                in lower_context
                and
                "community service"
                in lower_context
            ):

                parts.append(
                    "There is also a special rule for a "
                    "first conviction where the stolen "
                    "property is worth less than five "
                    "thousand rupees. If the value is returned "
                    "or the property is restored, the punishment "
                    "is community service."
                )

            if parts:

                return (
                    "\n\n".join(parts)
                    + "\n\n"
                    + _disclaimer(language)
                )

        if is_definition:

            return (
                "Sure. Section 303 of the Bharatiya "
                "Nyaya Sanhita, 2023 deals with theft. "
                "In simple terms, theft involves dishonestly "
                "taking movable property from another person's "
                "possession without that person's consent."
                "\n\n"
                + _disclaimer(language)
            )

    # --------------------------------------------------------
    # HINDI
    # --------------------------------------------------------

    if language == "hindi":

        if is_punishment:

            parts = []

            if (
                "may extend to three years"
                in lower_context
                or
                "or with fine, or with both"
                in lower_context
            ):

                parts.append(
                    "जी हाँ। भारतीय न्याय संहिता, 2023 की "
                    "धारा 303 के तहत चोरी के लिए तीन वर्ष "
                    "तक का कारावास, या जुर्माना, या दोनों "
                    "का प्रावधान है।"
                )

            if (
                "second or subsequent conviction"
                in lower_context
                and
                "five years"
                in lower_context
                and
                "one year"
                in lower_context
            ):

                parts.append(
                    "अगर इस धारा के तहत दूसरी या बाद की "
                    "दोषसिद्धि होती है, तो कठोर कारावास "
                    "एक वर्ष से कम नहीं होगा और पांच वर्ष "
                    "तक हो सकता है, साथ में जुर्माना भी होगा।"
                )

            if (
                "less than five thousand rupees"
                in lower_context
                and
                "first time"
                in lower_context
                and
                "community service"
                in lower_context
            ):

                parts.append(
                    "पहली बार की दोषसिद्धि में, यदि चोरी की "
                    "गई संपत्ति की कीमत पांच हजार रुपये से कम "
                    "है और संपत्ति की कीमत वापस कर दी जाती है "
                    "या संपत्ति बहाल कर दी जाती है, तो "
                    "सामुदायिक सेवा की सजा का प्रावधान है।"
                )

            if parts:

                return (
                    "\n\n".join(parts)
                    + "\n\n"
                    + _disclaimer(language)
                )

        if is_definition:

            return (
                "जी हाँ। भारतीय न्याय संहिता, 2023 की "
                "धारा 303 चोरी से संबंधित है। सरल शब्दों "
                "में, किसी दूसरे व्यक्ति की चल संपत्ति को "
                "उसकी सहमति के बिना बेईमानी से लेना चोरी "
                "हो सकता है।"
                "\n\n"
                + _disclaimer(language)
            )

    # --------------------------------------------------------
    # KANNADA
    # --------------------------------------------------------

    if language == "kannada":

        if is_punishment:

            parts = []

            if (
                "may extend to three years"
                in lower_context
                or
                "or with fine, or with both"
                in lower_context
            ):

                parts.append(
                    "ಹೌದು. ಭಾರತೀಯ ನ್ಯಾಯ ಸಂಹಿತೆ, 2023ರ "
                    "ಸೆಕ್ಷನ್ 303ರ ಪ್ರಕಾರ, ಕಳ್ಳತನಕ್ಕೆ ಮೂರು "
                    "ವರ್ಷಗಳವರೆಗೆ ಕಾರಾಗೃಹ ಶಿಕ್ಷೆ, ಅಥವಾ ದಂಡ, "
                    "ಅಥವಾ ಎರಡನ್ನೂ ವಿಧಿಸಬಹುದು."
                )

            if (
                "second or subsequent conviction"
                in lower_context
                and
                "five years"
                in lower_context
                and
                "one year"
                in lower_context
            ):

                parts.append(
                    "ಇದು ಎರಡನೇ ಅಥವಾ ನಂತರದ ದೋಷಸಿದ್ಧಿಯಾಗಿದ್ದರೆ, "
                    "ಕಠಿಣ ಕಾರಾಗೃಹ ಶಿಕ್ಷೆ ಕನಿಷ್ಠ ಒಂದು ವರ್ಷದಿಂದ "
                    "ಐದು ವರ್ಷಗಳವರೆಗೆ ಇರಬಹುದು ಮತ್ತು ದಂಡವೂ "
                    "ವಿಧಿಸಲಾಗುತ್ತದೆ."
                )

            if (
                "less than five thousand rupees"
                in lower_context
                and
                "first time"
                in lower_context
                and
                "community service"
                in lower_context
            ):

                parts.append(
                    "ಮೊದಲ ಬಾರಿಗೆ ದೋಷಸಿದ್ಧಿಯಾಗಿದ್ದು, ಕದ್ದ "
                    "ಆಸ್ತಿಯ ಮೌಲ್ಯ ಐದು ಸಾವಿರ ರೂಪಾಯಿಗಿಂತ "
                    "ಕಡಿಮೆಯಿದ್ದರೆ ಮತ್ತು ಆಸ್ತಿಯ ಮೌಲ್ಯವನ್ನು "
                    "ಹಿಂದಿರುಗಿಸಿದರೆ ಅಥವಾ ಆಸ್ತಿಯನ್ನು ಮರುಸ್ಥಾಪಿಸಿದರೆ, "
                    "ಸಮುದಾಯ ಸೇವೆಯ ಶಿಕ್ಷೆ ವಿಧಿಸುವ ವಿಶೇಷ ನಿಯಮವೂ ಇದೆ."
                )

            if parts:

                return (
                    "\n\n".join(parts)
                    + "\n\n"
                    + _disclaimer(language)
                )

        if is_definition:

            return (
                "ಹೌದು. ಭಾರತೀಯ ನ್ಯಾಯ ಸಂಹಿತೆ, 2023ರ "
                "ಸೆಕ್ಷನ್ 303 ಕಳ್ಳತನಕ್ಕೆ ಸಂಬಂಧಿಸಿದೆ. ಸರಳವಾಗಿ "
                "ಹೇಳುವುದಾದರೆ, ಇನ್ನೊಬ್ಬರ ಚಲಿಸುವ ಆಸ್ತಿಯನ್ನು "
                "ಅವರ ಒಪ್ಪಿಗೆಯಿಲ್ಲದೆ ಅಪ್ರಾಮಾಣಿಕವಾಗಿ ತೆಗೆದುಕೊಳ್ಳುವುದು "
                "ಕಳ್ಳತನವಾಗಬಹುದು."
                "\n\n"
                + _disclaimer(language)
            )

    return None


# ============================================================
# MASTER FALLBACK
# ============================================================

def _fallback_from_facts(
    question: str,
    facts_context: str,
) -> str:

    specialised = _fallback_theft(
        question,
        facts_context,
    )

    if specialised:
        return specialised

    return _generic_fallback(
        question,
        facts_context,
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question: str,
    facts_context: str,
    history: list | None = None,
) -> str:

    if not question or not question.strip():
        return "Please enter a legal question."

    question = question.strip()

    language = detect_language(
        question
    )

    if not facts_context or not facts_context.strip():

        if language == "hindi":
            return (
                "इस सवाल का जवाब देने के लिए "
                "पर्याप्त सत्यापित कानूनी जानकारी उपलब्ध नहीं है."
                "\n\n"
                + _disclaimer(language)
            )

        if language == "kannada":
            return (
                "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು "
                "ಪರಿಶೀಲಿತ ಕಾನೂನು ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ."
                "\n\n"
                + _disclaimer(language)
            )

        return (
            "There isn't enough verified legal information "
            "available to answer this question reliably."
            "\n\n"
            + _disclaimer(language)
        )

    print(
        "Detected answer language:",
        _language_name(language),
    )

    prompt = _build_prompt(
        question,
        facts_context,
    )

    try:

        print(
            "Calling Groq..."
        )

        response = (
            client
            .chat
            .completions
            .create(
                model=GROQ_MODEL,

                messages=(
                    [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                    ]
                    + [
                        {
                            "role": str(h.get("role", "user")),
                            "content": str(h.get("content", "")),
                        }
                        for h in (history or [])[-6:]
                        if h.get("content", "").strip()
                    ]
                    + [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                ),

                temperature=0.3,
                max_tokens=MAX_TOKENS,
            )
        )

        if not response or not response.choices:
            raise RuntimeError("Groq returned no choices.")

        msg = response.choices[0].message
        answer = msg.content or getattr(msg, "reasoning", "") or ""

        if not answer or not str(answer).strip():
            raise RuntimeError("Groq returned an empty answer.")

        cleaned = _clean_answer(
            answer,
            language,
        )

        if not cleaned:
            raise RuntimeError(
                "Groq answer became empty after cleaning."
            )

        print(
            "Groq answer generated successfully."
        )

        return cleaned

    except Exception as error:

        print()
        print(
            "=" * 70
        )
        print(
            "GROQ ERROR"
        )
        print(
            "Type:",
            type(error).__name__,
        )
        print(
            "Message:",
            str(error),
        )
        print(
            "Using deterministic legal fallback."
        )
        print(
            "=" * 70
        )

        return _fallback_from_facts(
            question,
            facts_context,
        )


# ============================================================
# MULTILINGUAL LEGAL DOCUMENT EXPLAINER
# ============================================================

def _validate_script_purity(text: str, language: str) -> bool:
    """
    Verify that generated output contains the expected script for the language.
    For Kannada: requires presence of Kannada script chars (\\u0C80-\\u0CFF).
    For Hindi: requires presence of Devanagari script chars (\\u0900-\\u097F).
    """
    if not text:
        return False
    lang = (language or "english").lower().strip()
    if lang == "kannada":
        kannada_chars = sum(1 for c in text if 0x0C80 <= ord(c) <= 0x0CFF)
        return kannada_chars >= 20
    elif lang == "hindi":
        hindi_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        return hindi_chars >= 20
    return True


def _clean_formatting_artifacts(text: str) -> str:
    """
    Remove raw markdown table syntax (| Item | Value |), table dividers (|---|---|),
    excessive hashes (###), raw horizontal rules (---, ___), and stray html tags.
    """
    if not text:
        return ""
    val = str(text)
    lines = []
    for line in val.splitlines():
        l_str = line.strip()
        # Skip table divider lines like |---|---| or |:---|:---|
        if re.match(r"^\|?[\s\-:]+(\|[\s\-:]+)+\|?$", l_str):
            continue
        # If line has multiple pipes, convert to clean readable text if it has data
        if l_str.startswith("|") and l_str.endswith("|") and l_str.count("|") >= 2:
            cells = [c.strip() for c in l_str.split("|") if c.strip()]
            if cells:
                lines.append(" • " + " — ".join(cells))
            continue
        # Remove standalone horizontal lines
        if l_str in ("---", "___", "***", "===", "----", "_____"):
            continue
        # Remove HTML breaks
        clean_l = re.sub(r"<br\s*/?>", " ", line, flags=re.IGNORECASE)
        lines.append(clean_l)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


DOCUMENT_EXPLAINER_PROMPT = """You are JanNyaya AI, an expert, compassionate legal assistance system for Indian citizens.
Your job is to analyze the provided legal document text thoroughly and provide an exhaustive, clear, accurate, and citizen-friendly explanation.

CRITICAL FORMATTING & LANGUAGE RULES:
1. Respond ENTIRELY in the requested LANGUAGE (English, Hindi, or Kannada).
2. DO NOT output markdown tables with pipes (|) or dashed lines (|---|). Use clean structured headings, bullet points, and plain text.
3. Keep the tone warm, empowering, objective, and easy for an ordinary citizen without a law degree to understand.
4. Distinguish clearly between stated document facts vs claims/allegations made by the sender.
5. Provide practical, structured advice grounded in the document text and Indian law."""


def explain_legal_document(
    document_text: str,
    document_type: str = "Legal Document",
    language: str = "english",
    retrieved_context: str = "",
) -> Dict[str, Any]:
    """
    Generate an exhaustive, citizen-friendly explanation of an uploaded legal document
    in English, Hindi, or Kannada using Groq LLM with script validation and structured fallback.
    """
    if not document_text or not document_text.strip():
        return _build_fallback_document_explanation("", document_type, language)

    # Normalize explanation language cleanly
    lang = (language or "english").lower().strip()
    if lang in ("kn", "kannada", "kan"):
        lang = "kannada"
    elif lang in ("hi", "hindi", "hin", "devanagari"):
        lang = "hindi"
    else:
        lang = "english"

    lang_name = _language_name(lang)

    # Explicit prompt tailored to the requested explanation language
    if lang == "kannada":
        lang_instruction = "The user selected Kannada for the explanation. Explain the document ENTIRELY in natural, fluent Kannada script (ಕನ್ನಡ). Avoid machine-translated or robotic phrasing."
    elif lang == "hindi":
        lang_instruction = "The user selected Hindi for the explanation. Explain the document ENTIRELY in natural, fluent Hindi script (हिन्दी). Avoid machine-translated or robotic phrasing."
    else:
        lang_instruction = "Explain the document entirely in clear, natural, and citizen-friendly English."

    user_prompt = f"""{lang_instruction}

DOCUMENT TYPE: {document_type}

----------------- DOCUMENT TEXT START -----------------
{document_text[:9000]}
------------------ DOCUMENT TEXT END ------------------

ADDITIONAL STATUTORY CONTEXT (if relevant):
{retrieved_context[:2000]}

Please explain the document in {lang_name} covering these 5 areas:
### 1. Document Overview & Nature
(Explain what this document is, who issued it, and its main purpose)

### 2. Key Terms, Conditions & Demands
(Bullet points of every condition, financial claim, obligation, or allegation)

### 3. Legal Implications & Rights
(What this means for the citizen, applicable laws, potential risks)

### 4. Important Deadlines & Timelines
(Any 15-day, 30-day, or hearing dates mentioned or implied by law)

### 5. Recommended Next Steps (Action Plan)
(Clear step-by-step guide on what the citizen should do right now)
"""

    try:
        print(f"Calling Groq for Document Explanation in {lang_name} (Requested: {lang})...")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": DOCUMENT_EXPLAINER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Empty response from Groq document explainer")

        # Validate script compliance for Kannada and Hindi
        if not _validate_script_purity(content, lang):
            print(f"Script validation failed for {lang}. Retrying with strict language correction prompt...")
            retry_prompt = f"""CRITICAL INSTRUCTION: The user explicitly chose {lang_name} for this document explanation.
Your previous response contained English or incorrect script.
You MUST rewrite the complete document explanation STRICTLY in {lang_name} script now without English sentences:

DOCUMENT TYPE: {document_type}
DOCUMENT TEXT:
{document_text[:6000]}
"""
            retry_response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": DOCUMENT_EXPLAINER_PROMPT},
                    {"role": "user", "content": retry_prompt},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            retry_content = (retry_response.choices[0].message.content or "").strip()
            if retry_content and _validate_script_purity(retry_content, lang):
                content = retry_content
            elif not _validate_script_purity(content, lang):
                print(f"Second attempt also failed script purity for {lang}. Using natural {lang_name} fallback.")
                return _build_fallback_document_explanation(document_text, document_type, lang)

        # Parse key sections out of the structured output
        sections_map = _parse_document_sections(content, lang, document_type, document_text)
        return sections_map

    except Exception as e:
        print("Groq document explanation error:", type(e).__name__, str(e))
        return _build_fallback_document_explanation(document_text, document_type, lang)


def _parse_document_sections(markdown_text: str, language: str, document_type: str = "Legal Document", raw_doc_text: str = "") -> Dict[str, Any]:
    """Parse structured output into frontend-ready fields, clean formatting artifacts, and create structured JSON."""
    cleaned_md = _clean_formatting_artifacts(markdown_text)
    lines = cleaned_md.splitlines()
    overview = []
    conditions = []
    implications = []
    deadlines = []
    next_steps = []
    current_sec = "overview"

    for line in lines:
        l_lower = line.lower()
        if "1. document overview" in l_lower or ("overview" in l_lower and line.startswith("#")):
            current_sec = "overview"
            continue
        elif "2. key terms" in l_lower or ("conditions" in l_lower and line.startswith("#")):
            current_sec = "conditions"
            continue
        elif "3. legal implications" in l_lower or ("implications" in l_lower and line.startswith("#")):
            current_sec = "implications"
            continue
        elif "4. important deadlines" in l_lower or ("deadlines" in l_lower and line.startswith("#")):
            current_sec = "deadlines"
            continue
        elif "5. recommended next steps" in l_lower or "next steps" in l_lower or ("action plan" in l_lower and line.startswith("#")):
            current_sec = "next_steps"
            continue

        clean_l = line.strip()
        if not clean_l or clean_l.startswith("###") or clean_l.startswith("==") or clean_l.startswith("---"):
            continue

        if current_sec == "overview":
            overview.append(clean_l)
        elif current_sec == "conditions":
            if clean_l.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
                clean_item = clean_l.lstrip("-•* 0123456789.").strip()
                if clean_item:
                    conditions.append(clean_item)
            else:
                conditions.append(clean_l)
        elif current_sec == "implications":
            implications.append(clean_l)
        elif current_sec == "deadlines":
            if clean_l.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.")):
                clean_item = clean_l.lstrip("-•* 0123456789.").strip()
                if clean_item:
                    deadlines.append(clean_item)
            else:
                deadlines.append(clean_l)
        elif current_sec == "next_steps":
            if clean_l.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
                clean_item = clean_l.lstrip("-•* 0123456789.").strip()
                if clean_item:
                    next_steps.append(clean_item)
            else:
                next_steps.append(clean_l)

    summary_text = "\n\n".join(overview) if overview else cleaned_md[:600]
    impl_text = "\n\n".join(implications) if implications else "Review the document carefully and verify all terms."

    # Filter empty items & clean artifacts
    conditions = [_clean_formatting_artifacts(c) for c in conditions if len(c) > 3][:8]
    deadlines = [_clean_formatting_artifacts(d) for d in deadlines if len(d) > 3][:5]
    next_steps = [_clean_formatting_artifacts(s) for s in next_steps if len(s) > 3][:8]

    # Structured Document Overview Cards
    if language == "kannada":
        doc_overview = [
            {"label": "ದಾಖಲೆಯ ಪ್ರಕಾರ (Document Type)", "value": document_type},
            {"label": "ವಿವರಣೆ ಭಾಷೆ (Explanation Language)", "value": "ಕನ್ನಡ (Kannada)"},
            {"label": "ಮುಖ್ಯ ಉದ್ದೇಶ (Purpose)", "value": summary_text[:180] + "..." if len(summary_text) > 180 else summary_text},
            {"label": "ಕಾನೂನು ಸ್ಥಿತಿ (Legal Standing)", "value": "ಪರಿಶೀಲನೆ ಮತ್ತು ನಿಯಮಾನುಸಾರ ಕ್ರಮ ಅಗತ್ಯವಿದೆ."},
        ]
    elif language == "hindi":
        doc_overview = [
            {"label": "दस्तावेज़ का प्रकार (Document Type)", "value": document_type},
            {"label": "विवरण भाषा (Explanation Language)", "value": "हिन्दी (Hindi)"},
            {"label": "मुख्य उद्देश्य (Purpose)", "value": summary_text[:180] + "..." if len(summary_text) > 180 else summary_text},
            {"label": "कानूनी स्थिति (Legal Standing)", "value": "समीक्षा एवं यथोचित कानूनी कदम आवश्यक हैं।"},
        ]
    else:
        doc_overview = [
            {"label": "Document Type", "value": document_type},
            {"label": "Explanation Language", "value": "English"},
            {"label": "Primary Purpose", "value": summary_text[:180] + "..." if len(summary_text) > 180 else summary_text},
            {"label": "Legal Status", "value": "Formal review and procedural action required."},
        ]

    # Generate suggested questions based on language
    suggested = {
        "kannada": [
            "ಈ ನೋಟಿಸ್‌ಗೆ ನಾನು ಹೇಗೆ ಉತ್ತರ (Reply) ನೀಡಬೇಕು?",
            "ನೋಟಿಸ್‌ನಲ್ಲಿ ನೀಡಿರುವ ಅವಧಿಯೊಳಗೆ ಉತ್ತರಿಸದಿದ್ದರೆ ಏನಾಗುತ್ತದೆ?",
            "ನನಗೆ ಉಚಿತ ಕಾನೂನು ನೆರವು (Legal Aid) ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ?",
            "ನನ್ನ ಪರವಾಗಿ ವಕೀಲರ ಅಗತ್ಯವಿದೆಯೇ?",
        ],
        "hindi": [
            "मुझे इस नोटिस का कानूनी जवाब कैसे देना चाहिए?",
            "यदि मैं समय सीमा में जवाब नहीं देता तो क्या होगा?",
            "क्या मुझे मुफ्त कानूनी सहायता (Legal Aid) मिल सकती है?",
            "मुझे कौन-कौन से साक्ष्य और दस्तावेज सुरक्षित रखने चाहिए?",
        ],
        "english": [
            "How should I reply to this legal notice / document?",
            "What happens if I miss the stated deadline?",
            "Can I get free legal aid under NALSA for this case?",
            "What specific evidence or receipts should I preserve?",
        ],
    }.get(language, [
        "How should I reply to this legal notice / document?",
        "What happens if I miss the stated deadline?",
        "Can I get free legal aid under NALSA for this case?",
    ])

    return {
        "full_explanation": cleaned_md,
        "summary": summary_text,
        "document_overview": doc_overview,
        "conditions_and_clauses": conditions,
        "claims": conditions,
        "legal_implications": impl_text,
        "urgency": "High - Response Required" if deadlines or ("15" in cleaned_md or "30" in cleaned_md) else "Standard Legal Review",
        "deadlines": deadlines,
        "actionable_steps": next_steps,
        "next_steps": next_steps,
        "suggested_questions": suggested,
        "disclaimer": _disclaimer(language),
    }


def _build_fallback_document_explanation(
    document_text: str,
    document_type: str,
    language: str,
) -> Dict[str, Any]:
    """Deterministic, natural citizen-friendly fallback when LLM is unavailable or for instant verification."""
    doc_lower = (document_text or "").lower()
    deadlines = []
    if "15 days" in doc_lower or "fifteen days" in doc_lower or "15 ದಿನ" in doc_lower or "15 दिन" in doc_lower:
        deadlines.append("15-day statutory/notice deadline detected from document text.")
    if "30 days" in doc_lower or "thirty days" in doc_lower or "30 ದಿನ" in doc_lower or "30 दिन" in doc_lower:
        deadlines.append("30-day response period specified in document.")

    if language == "kannada":
        summary = f"ಈ ದಾಖಲೆ ಒಂದು {document_type} ಆಗಿದೆ. ಇದರಲ್ಲಿ ನಮೂದಿಸಲಾದ ಕಾನೂನು ನಿಯಮಗಳು, ಮೊತ್ತ ಮತ್ತು ಷರತ್ತುಗಳನ್ನು ಪರಿಶೀಲಿಸಲಾಗಿದೆ. ಈ ದಾಖಲೆಯಲ್ಲಿ ನೀಡಿರುವ ಸೂಚನೆಗಳನ್ನು ಗಮನಿಸಿ ಸೂಕ್ತ ಕ್ರಮ ಕೈಗೊಳ್ಳುವುದು ಅಗತ್ಯವಾಗಿದೆ."
        implications = "ದಾಖಲೆಯಲ್ಲಿ ನಮೂದಿಸಲಾದ ಷರತ್ತುಗಳಿಗೆ ನಿಗದಿತ ಸಮಯದಲ್ಲಿ ಸ್ಪಂದಿಸದಿದ್ದರೆ ಕಾನೂನು ಅಥವಾ ನ್ಯಾಯಾಲಯದ ಪ್ರಕ್ರಿಯೆಗಳು ಪ್ರಾರಂಭವಾಗುವ ಸಾಧ್ಯತೆಯಿದೆ."
        next_steps = [
            "ದಾಖಲೆಯ ಮೂಲ ಪ್ರತಿ ಮತ್ತು ಅಂಚೆ ಲಕೋಟೆಯನ್ನು (ದಿನಾಂಕದ ಪುರಾವೆಗಾಗಿ) ಭದ್ರವಾಗಿ ಇಟ್ಟುಕೊಳ್ಳಿ.",
            "ದಾಖಲೆಯಲ್ಲಿ ನಮೂದಿಸಲಾದ ಹಣಕಾಸಿನ ಮೊತ್ತ ಮತ್ತು ದಿನಾಂಕಗಳನ್ನು ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಲೆಕ್ಕದೊಂದಿಗೆ ತಾಳೆ ನೋಡಿ.",
            "ನಿಗದಿತ ಕಾಲಮಿತಿಯೊಳಗೆ ಸೂಕ್ತ ಕಾನೂನು ಉತ್ತರವನ್ನು (Legal Reply) ನೀಡಲು ವಕೀಲರನ್ನು ಅಥವಾ ಕಾನೂನು ಸೇವಾ ಪ್ರಾಧಿಕಾರವನ್ನು ಸಂಪರ್ಕಿಸಿ.",
            "ಸಂಬಂಧಪಟ್ಟ ರಸೀದಿಗಳು, ಬ್ಯಾಂಕ್ ಸ್ಟೇಟ್‌ಮೆಂಟ್‌ಗಳು ಮತ್ತು ಸಂವಹನ ದಾಖಲೆಗಳನ್ನು ಒಟ್ಟುಗೂಡಿಸಿ.",
        ]
        conditions = [
            "ದಾಖಲೆಯಲ್ಲಿ ಉಲ್ಲೇಖಿಸಲಾದ ಹಣಕಾಸಿನ ಬೇಡಿಕೆ ಅಥವಾ ಮರುಪಾವತಿ ಷರತ್ತುಗಳು.",
            "ಸಂಬಂಧಿತ ಪಕ್ಷಕಾರರ ನಡುವಿನ ಕಾನೂನುಬದ್ಧ ಒಪ್ಪಂದ ಮತ್ತು ನಿಯಮಗಳ ಪಾಲನೆ.",
        ]
        doc_overview = [
            {"label": "ದಾಖಲೆಯ ಪ್ರಕಾರ", "value": document_type},
            {"label": "ವಿವರಣೆ ಭಾಷೆ", "value": "ಕನ್ನಡ (Kannada)"},
            {"label": "ಮುಖ್ಯ ಉದ್ದೇಶ", "value": f"{document_type} ಪರಿಶೀಲನೆ ಮತ್ತು ನಾಗರಿಕರಿಗೆ ಸರಳ ವಿವರಣೆ."},
            {"label": "ಕಾನೂನು ಸ್ಥಿತಿ", "value": "ಕಾನೂನುಬದ್ಧ ಪರಿಶೀಲನೆ ಮತ್ತು ನಿಯಮಾನುಸಾರ ಉತ್ತರ ನೀಡುವುದು ಅವಶ್ಯಕ."},
        ]
    elif language == "hindi":
        summary = f"यह दस्तावेज़ एक {document_type} है। इसमें उल्लिखित कानूनी शर्तों, वित्तीय मांगों और नियमों का विश्लेषण किया गया है। दस्तावेज़ में दी गई सूचनाओं के आधार पर समय पर उचित कानूनी कदम उठाना आवश्यक है।"
        implications = "दस्तावेज़ में दी गई शर्तों या मांग पर समय सीमा के भीतर कार्रवाई न करने पर कानूनी या न्यायिक कार्यवाही का जोखिम हो सकता है।"
        next_steps = [
            "दस्तावेज़ की मूल प्रति और डाक लिफाफे (प्राप्ति की तारीख के प्रमाण हेतु) को सुरक्षित रखें।",
            "दस्तावेज़ में उल्लिखित वित्तीय राशि, ब्याज और तारीखों का अपने व्यक्तिगत रिकॉर्ड से मिलान करें।",
            "निर्धारित समय सीमा के भीतर औपचारिक कानूनी जवाब (Reply) तैयार करने के लिए किसी योग्य अधिवक्ता या जिला विधिक सेवा प्राधिकरण से संपर्क करें।",
            "संबंधित भुगतान रसीदें, बैंक विवरण और पिछले पत्र-व्यवहार एकत्र करें।",
        ]
        conditions = [
            "दस्तावेज़ में उल्लिखित औपचारिक मांग अथवा भुगतान की शर्तें।",
            "पक्षकारों के बीच कानूनी दायित्व और विधिक बाध्यताएं।",
        ]
        doc_overview = [
            {"label": "दस्तावेज़ का प्रकार", "value": document_type},
            {"label": "विवरण भाषा", "value": "हिन्दी (Hindi)"},
            {"label": "मुख्य उद्देश्य", "value": f"{document_type} का विधिक विश्लेषण और नागरिक हेतु सरल विवरण।"},
            {"label": "कानूनी स्थिति", "value": "विधिक समीक्षा एवं समय पर उत्तर देना आवश्यक है।"},
        ]
    else:
        summary = f"This appears to be a formal {document_type}. The document sets forth specific legal terms, obligations, financial claims, or statutory demands that require careful attention and timely response."
        implications = "Failure to address the demands or conditions within the stipulated statutory timeline may lead to formal legal proceedings or statutory liability."
        next_steps = [
            "Preserve the original document along with postal envelopes (vital for proof of date of service).",
            "Cross-verify all claimed amounts, dates, and account figures against your personal transaction records.",
            "Draft a formal, legally grounded reply within the prescribed response window with assistance from legal counsel or the District Legal Services Authority (DLSA).",
            "Compile all relevant transaction receipts, bank statements, notices, and communications.",
        ]
        conditions = [
            "Statutory or contractual demands and allegations stated in the document.",
            "Binding obligations, covenants, and repayment conditions applicable to the parties.",
        ]
        doc_overview = [
            {"label": "Document Type", "value": document_type},
            {"label": "Explanation Language", "value": "English"},
            {"label": "Primary Purpose", "value": f"Legal analysis of {document_type} for citizen guidance."},
            {"label": "Legal Status", "value": "Actionable document requiring timely legal response and verification."},
        ]

    return {
        "full_explanation": summary,
        "summary": summary,
        "document_overview": doc_overview,
        "conditions_and_clauses": conditions,
        "claims": conditions,
        "legal_implications": implications,
        "urgency": "High - Action Recommended" if deadlines else "Standard Legal Review",
        "deadlines": deadlines or ["Verify any explicit response timeline mentioned in the original document."],
        "actionable_steps": next_steps,
        "next_steps": next_steps,
        "suggested_questions": [
            "How should I reply to this document?",
            "What if I don't respond within the deadline?",
            "Where can I find free legal assistance?",
        ],
        "disclaimer": _disclaimer(language),
    }


# ============================================================
# INTERACTIVE DOCUMENT CHAT ASSISTANT
# ============================================================

DOCUMENT_CHAT_SYSTEM_PROMPT = (
    "You are JanNyaya AI's Senior Interactive Document & Legal Analysis Assistant.\n"
    "The user has uploaded one or more legal documents (such as legal notices, agreements, court petitions, FIRs, loan contracts, challans, orders) "
    "and is asking questions specifically about them.\n\n"
    "Your responsibilities:\n"
    "1. Answer the question directly, thoroughly, and compassionately using facts extracted from the document text.\n"
    "2. If the user asks 'what should I do next', 'what are my options', or 'explain this document', provide a crystal-clear, step-by-step practical action plan.\n"
    "3. Identify all critical dates, deadlines (e.g. 15-day or 30-day response window), claimed monetary amounts (principal, interest, penalty), parties involved, and referenced statutory provisions (e.g. Order 5 Rule 1 CPC, Section 138 NI Act, Section 73 Contract Act, BNS/IPC sections).\n"
    "4. Explain legal jargon in simple, easily understandable language without losing statutory precision.\n"
    "5. If a specific question cannot be directly answered from the document, clearly explain why and provide standard Indian statutory procedure/remedies for such situations.\n"
    "6. Respond completely in the requested language (English, Hindi, or Kannada).\n"
    "7. Maintain a professional, reassuring, and objective tone. Do not invent non-existent facts.\n"
)

def chat_with_document(
    document_text: str,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    language: str = "english",
) -> str:
    """
    Interactive Q&A assistant specifically grounded in an uploaded legal document.
    Supports English, Hindi, and Kannada conversation with multi-turn history.
    """
    if not question or not question.strip():
        return "Please ask a question about your uploaded document."

    lang = (language or "english").lower().strip()
    if lang not in ("hindi", "kannada", "english"):
        lang = detect_language(question)

    lang_name = _language_name(lang)
    clean_doc = (document_text or "").strip()
    doc_context = clean_doc[:16000] if clean_doc else "No document text was provided."

    user_content = f"""
LANGUAGE: Respond completely and naturally in {lang_name}.

UPLOADED DOCUMENT CONTEXT:
---------------- BEGIN DOCUMENT ----------------
{doc_context}
----------------- END DOCUMENT -----------------

USER QUESTION:
{question}

Provide a structured, helpful, and direct answer based on the document and Indian legal principles.
"""

    messages = [{"role": "system", "content": DOCUMENT_CHAT_SYSTEM_PROMPT}]
    for h in (history or [])[-6:]:
        h_text = str(h.get("content") or h.get("text") or "").strip()
        if h_text:
            role = "user" if h.get("role") in ("user", "human") else "assistant"
            messages.append({"role": role, "content": h_text})
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1200,
        )
        ans = response.choices[0].message.content.strip()
        return _clean_answer(ans, lang)
    except Exception as e:
        print("Document chat error:", type(e).__name__, str(e))
        if lang == "kannada":
            return f"ದಾಖಲೆಯ ವಿಶ್ಲೇಷಣೆ: ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದಂತೆ, ದಾಖಲೆಯಲ್ಲಿರುವ ದಿನಾಂಕಗಳು, ಮೊತ್ತಗಳು ಮತ್ತು ಷರತ್ತುಗಳನ್ನು ಪರಿಶೀಲಿಸಿ. ಕಾನೂನು ಪ್ರಕ್ರಿಯೆಯನ್ನು ಎದುರಿಸಲು ಸಂಬಂಧಿತ ದಾಖಲೆಗಳನ್ನು ಸಂರಕ್ಷಿಸಿ ಮತ್ತು ಅರ್ಹ ವಕೀಲರ ಸಲಹೆ ಪಡೆಯಿರಿ.\n\n{_disclaimer(lang)}"
        elif lang == "hindi":
            return f"दस्तावेज़ का विश्लेषण: इस संबंध में दस्तावेज़ में उल्लिखित समय-सीमा, राशि और शर्तों की जांच करें। अपने अधिकारों की सुरक्षा के लिए सभी साक्ष्य सुरक्षित रखें और आवश्यकतानुसार किसी योग्य अधिवक्ता से परामर्श लें।\n\n{_disclaimer(lang)}"
        else:
            return f"Based on your document analysis: Please carefully check the stated deadlines, demand amounts, and contractual clauses. Preserve all payment proofs/receipts and consult a legal professional for formal legal reply.\n\n{_disclaimer(lang)}"



# ============================================================
# TEST DATA
# ============================================================

TEST_FACTS = """
[Legal Fact 1]
Section: 303
Section title: Theft
Source: Bharatiya_Nyaya_Sanhita_2023.pdf
Direct offence provision: yes

Definition provisions:
- Whoever, intending to take dishonestly any movable property out of the possession of any person without consent, moves that property in order to such taking, is said to commit theft.

Punishment provisions:
- Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.
- In case of second or subsequent conviction of any person under this section, he shall be punished with rigorous imprisonment for a term which shall not be less than one year but which may extend to five years and with fine.
- In cases of theft where the value of the stolen property is less than five thousand rupees, and a person is convicted for the first time, upon return of the value of property or restoration of the stolen property, the person shall be punished with community service.
"""


# ============================================================
# COMMAND LINE TEST
# ============================================================

def main() -> None:

    print()
    print(
        "JanNyaya AI - Friendly Multilingual LLM Test"
    )
    print(
        f"Model: {GROQ_MODEL}"
    )
    print(
        f"Max tokens: {MAX_TOKENS}"
    )

    questions = [
        "What is the punishment for theft?",
        "Can you explain theft in simple words?",
        "Someone took my phone without asking. What law may apply?",
        "चोरी की सजा क्या है?",
        "चोरी को आसान भाषा में समझाइए।",
        "ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ ಏನು?",
        "ಕಳ್ಳತನವನ್ನು ಸರಳವಾಗಿ ವಿವರಿಸಿ.",
        "ಯಾರೋ ನನ್ನ ಫೋನ್ ಅನುಮತಿಯಿಲ್ಲದೆ ತೆಗೆದುಕೊಂಡಿದ್ದಾರೆ. ಏನು ಮಾಡಬಹುದು?",
    ]

    for question in questions:

        print()
        print("=" * 70)
        print("QUESTION:")
        print(question)
        print("=" * 70)

        answer = generate_answer(
            question,
            TEST_FACTS,
        )

        print()
        print("ANSWER:")
        print(answer)


if __name__ == "__main__":
    main()