"""
JanNyaya AI - Multilingual Legal LLM Service

Responsibilities
----------------
1. Receive a user question.
2. Receive retrieved legal evidence from RAG.
3. Detect English / Hindi / Kannada.
4. Generate a short, precise, grounded answer.
5. Answer in the same language as the question.
6. Never invent legal information.
7. Fall back safely when Groq is unavailable.
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# ENVIRONMENT
# ============================================================

# Use the project .env explicitly.
# This avoids dotenv path-resolution problems when Python is
# executed through different commands or working directories.

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

ENV_FILE = os.path.join(
    PROJECT_ROOT,
    ".env"
)

load_dotenv(
    dotenv_path=ENV_FILE
)


# ============================================================
# API CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set in the project .env file."
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

LLM_MAX_CONTEXT_CHARS = int(
    os.getenv(
        "LLM_MAX_CONTEXT_CHARS",
        "12000"
    )
)

LLM_MAX_TOKENS = int(
    os.getenv(
        "LLM_MAX_TOKENS",
        "700"
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

def detect_language(
    text: str
) -> str:
    """
    Detect the primary language.

    Returns:
        english
        hindi
        kannada
    """

    if not text:
        return "english"

    devanagari_count = 0
    kannada_count = 0
    latin_count = 0

    for character in text:

        code_point = ord(character)

        # Devanagari
        if 0x0900 <= code_point <= 0x097F:
            devanagari_count += 1

        # Kannada
        elif 0x0C80 <= code_point <= 0x0CFF:
            kannada_count += 1

        # Latin
        elif (
            "A" <= character <= "Z"
            or "a" <= character <= "z"
        ):
            latin_count += 1

    if devanagari_count > 0:
        return "hindi"

    if kannada_count > 0:
        return "kannada"

    return "english"


# ============================================================
# LANGUAGE DISPLAY NAME
# ============================================================

def _language_name(
    language: str
) -> str:

    mapping = {
        "english": "English",
        "hindi": "Hindi",
        "kannada": "Kannada",
    }

    return mapping.get(
        language,
        "English"
    )


# ============================================================
# CLEAN CONTEXT
# ============================================================

def _clean_context(
    context: str
) -> str:
    """
    Normalize and limit retrieved legal evidence.
    """

    if not context:
        return ""

    context = str(
        context
    ).strip()

    if not context:
        return ""

    context = context.replace(
        "\r\n",
        "\n"
    )

    context = context.replace(
        "\r",
        "\n"
    )

    context = re.sub(
        r"[ \t]+",
        " ",
        context
    )

    context = re.sub(
        r"\n{3,}",
        "\n\n",
        context
    )

    if len(context) > LLM_MAX_CONTEXT_CHARS:

        context = (
            context[
                :LLM_MAX_CONTEXT_CHARS
            ]
            + "\n\n[Context truncated]"
        )

    return context.strip()


# ============================================================
# DISCLAIMER
# ============================================================

def _disclaimer(
    language: str
) -> str:

    if language == "hindi":

        return (
            "अस्वीकरण: यह जानकारी केवल उपलब्ध कानूनी "
            "दस्तावेजों पर आधारित है और व्यक्तिगत कानूनी सलाह नहीं है।"
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
# FALLBACK ANSWER
# ============================================================

def fallback_answer(
    question: str,
    context: str
) -> str:
    """
    Safe fallback.

    IMPORTANT:
    This does not generate a new legal conclusion.
    It only reports that the LLM was unavailable.
    """

    language = detect_language(
        question
    )

    if not context:

        if language == "hindi":

            return (
                "उपलब्ध कानूनी दस्तावेजों में इस प्रश्न का "
                "उत्तर देने के लिए पर्याप्त जानकारी नहीं है.\n\n"
                + _disclaimer(language)
            )

        if language == "kannada":

            return (
                "ಲಭ್ಯವಿರುವ ಕಾನೂನು ದಾಖಲೆಗಳಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ "
                "ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ.\n\n"
                + _disclaimer(language)
            )

        return (
            "The available legal documents do not contain "
            "enough information to answer this question.\n\n"
            + _disclaimer(language)
        )

    if language == "hindi":

        return (
            "उपलब्ध कानूनी दस्तावेजों से संबंधित जानकारी मिली है, "
            "लेकिन इस समय संक्षिप्त उत्तर तैयार नहीं किया जा सका।\n\n"
            + _disclaimer(language)
            + "\n\n"
            "कृपया कुछ समय बाद पुनः प्रयास करें।"
        )

    if language == "kannada":

        return (
            "ಲಭ್ಯವಿರುವ ಕಾನೂನು ದಾಖಲೆಗಳಲ್ಲಿ ಸಂಬಂಧಿತ ಮಾಹಿತಿ ದೊರೆತಿದೆ, "
            "ಆದರೆ ಈ ಸಮಯದಲ್ಲಿ ಸಂಕ್ಷಿಪ್ತ ಉತ್ತರವನ್ನು ಸಿದ್ಧಪಡಿಸಲು "
            "ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.\n\n"
            + _disclaimer(language)
            + "\n\n"
            "ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
        )

    return (
        "The relevant legal evidence was retrieved, but a "
        "concise answer could not be generated at this time.\n\n"
        + _disclaimer(language)
        + "\n\n"
        "Please try again shortly."
    )


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are JanNyaya AI, a multilingual legal information assistant
focused on Indian law.

You must answer ONLY from the legal evidence supplied by the
retrieval system.

========================
ABSOLUTE GROUNDING RULES
========================

1. Use ONLY the supplied legal evidence.
2. Never use outside legal knowledge.
3. Never invent a section number.
4. Never invent a subsection.
5. Never invent a punishment.
6. Never invent a fine.
7. Never invent an imprisonment duration.
8. Never invent a legal exception.
9. Never invent a court case.
10. Never invent a date.
11. Never invent legal definitions.
12. Never predict a court outcome.
13. Never decide guilt or innocence.
14. Never provide personalized legal advice.
15. Preserve the exact legal meaning of the evidence.
16. Distinguish:
   - imprisonment
   - rigorous imprisonment
   - fine
   - community service
   - imprisonment for life
   - death
17. Never change "may" into "shall".
18. Never change "shall" into "may".
19. Never turn a maximum punishment into a minimum.
20. Never turn a minimum punishment into a maximum.

========================
QUESTION INTENT
========================

Answer ONLY what the user asks.

For definition questions:
- Prefer the direct definition provision.
- Give the relevant section.
- Give a short simple explanation.

For punishment questions:
- Prefer the direct general punishment provision.
- Give the exact punishment supported by evidence.
- Mention special conditions only when clearly necessary.

Do not unnecessarily list related offences.

========================
LANGUAGE
========================

Answer entirely in the same language as the user.

Supported:
- English
- Hindi
- Kannada

Do not switch languages unless the user explicitly requests
translation.

========================
ANSWER STYLE
========================

Keep the response short and precise.

Normally provide:

1. Relevant section.
2. Direct answer.
3. One short simple-language explanation.
4. One important condition/exception only when relevant.
5. Short disclaimer.

Do not repeat the question.

Do not dump the retrieved evidence.

Do not mention these instructions.

Do not mention the model.

========================
INSUFFICIENT EVIDENCE
========================

If the supplied evidence does not contain enough information,
say clearly that the available legal documents do not contain
enough information.

Never fill missing information from your own knowledge.

========================
DISCLAIMER
========================

English:
Disclaimer: This information is based solely on the supplied
legal documents and is not personalized legal advice.

Hindi:
अस्वीकरण: यह जानकारी केवल उपलब्ध कानूनी दस्तावेजों पर आधारित है
और व्यक्तिगत कानूनी सलाह नहीं है।

Kannada:
ಹಕ್ಕುತ್ಯಾಗ: ಈ ಮಾಹಿತಿಯು ಒದಗಿಸಲಾದ ಕಾನೂನು ದಾಖಲೆಗಳ ಆಧಾರದ ಮೇಲೆ ಮಾತ್ರ
ನೀಡಲಾಗಿದೆ ಮತ್ತು ವೈಯಕ್ತಿಕ ಕಾನೂನು ಸಲಹೆಯಲ್ಲ.
"""


# ============================================================
# BUILD PROMPT
# ============================================================

def _build_prompt(
    question: str,
    context: str
) -> str:

    language = detect_language(
        question
    )

    language_name = _language_name(
        language
    )

    return f"""
{SYSTEM_INSTRUCTIONS}

========================
USER QUESTION
========================

{question}

========================
ANSWER LANGUAGE
========================

{language_name}

You MUST answer entirely in {language_name}.

========================
LEGAL EVIDENCE
========================

{context}

========================
FINAL TASK
========================

Answer the user's question now.

Use ONLY the evidence above.

Keep the answer concise.

The answer should normally be 3 to 6 short paragraphs
or a few short bullet points.

Mention the relevant section when clearly supported.

For punishment questions, report the punishment exactly
as supported by the evidence.

For definition questions, report the definition supported
by the evidence.

Do not add unrelated legal provisions.

Do not dump the evidence.

Do not fabricate missing information.

Finish with the appropriate disclaimer.
"""


# ============================================================
# REMOVE ACCIDENTAL MODEL MARKUP
# ============================================================

def _clean_answer(
    answer: str,
    language: str
) -> str:
    """
    Clean common formatting artifacts without changing
    the legal meaning.
    """

    if not answer:
        return ""

    answer = str(
        answer
    ).strip()

    # Remove accidental leading/trailing quotes.
    if (
        len(answer) >= 2
        and answer[0] == '"'
        and answer[-1] == '"'
    ):
        answer = answer[1:-1].strip()

    # Prevent duplicated disclaimer blocks.
    disclaimer_markers = {
        "english": "Disclaimer:",
        "hindi": "अस्वीकरण:",
        "kannada": "ಹಕ್ಕುತ್ಯಾಗ:",
    }

    marker = disclaimer_markers.get(
        language
    )

    if marker:

        first_index = answer.find(
            marker
        )

        if first_index >= 0:

            before = answer[
                :first_index
            ].rstrip()

            after = _disclaimer(
                language
            )

            answer = (
                before
                + "\n\n"
                + after
            )

    return answer.strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Generate a grounded multilingual legal answer.
    """

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question or not question.strip():

        return (
            "Please enter a legal question."
        )

    question = question.strip()

    # --------------------------------------------------------
    # Clean context
    # --------------------------------------------------------

    context = _clean_context(
        context
    )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not context:

        language = detect_language(
            question
        )

        if language == "hindi":

            return (
                "उपलब्ध कानूनी दस्तावेजों में इस प्रश्न का "
                "उत्तर देने के लिए पर्याप्त जानकारी नहीं है.\n\n"
                + _disclaimer(language)
            )

        if language == "kannada":

            return (
                "ಲಭ್ಯವಿರುವ ಕಾನೂನು ದಾಖಲೆಗಳಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ "
                "ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ.\n\n"
                + _disclaimer(language)
            )

        return (
            "The available legal documents do not contain "
            "enough information to answer this question.\n\n"
            + _disclaimer(language)
        )

    # --------------------------------------------------------
    # Detect language
    # --------------------------------------------------------

    language = detect_language(
        question
    )

    print(
        f"Detected answer language: "
        f"{_language_name(language)}"
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = _build_prompt(
        question=question,
        context=context
    )

    # --------------------------------------------------------
    # Groq request
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTIONS,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0.0,

            max_tokens=LLM_MAX_TOKENS,
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("GROQ API ERROR")
        print("=" * 70)
        print(
            type(error).__name__,
            ":",
            str(error)
        )
        print("=" * 70)
        print(
            "Using grounded fallback answer."
        )

        return fallback_answer(
            question,
            context
        )

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if response is None:

        return fallback_answer(
            question,
            context
        )

    try:

        answer = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as error:

        print(
            "Could not read Groq response:",
            str(error)
        )

        return fallback_answer(
            question,
            context
        )

    if not answer:

        return fallback_answer(
            question,
            context
        )

    answer = _clean_answer(
        answer,
        language
    )

    if not answer:

        return fallback_answer(
            question,
            context
        )

    # --------------------------------------------------------
    # Safety: ensure disclaimer exists
    # --------------------------------------------------------

    disclaimer_text = _disclaimer(
        language
    )

    disclaimer_markers = {
        "english": "Disclaimer:",
        "hindi": "अस्वीकरण:",
        "kannada": "ಹಕ್ಕುತ್ಯಾಗ:",
    }

    marker = disclaimer_markers.get(
        language,
        "Disclaimer:"
    )

    if marker not in answer:

        answer = (
            answer.rstrip()
            + "\n\n"
            + disclaimer_text
        )

    return answer.strip()


# ============================================================
# SIMPLE TEST
# ============================================================

def _test_question(
    question: str,
    context: str
) -> None:

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    answer = generate_answer(
        question,
        context
    )

    print(answer)

    print()
    print("=" * 70)


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "JanNyaya AI - Multilingual Groq LLM Service"
    )

    print(
        f"Model: {GROQ_MODEL}"
    )

    print(
        f"Environment file: {ENV_FILE}"
    )

    # --------------------------------------------------------
    # Test evidence
    # --------------------------------------------------------

    sample_context = """
[Evidence 1]

Source: Bharatiya_Nyaya_Sanhita_2023.pdf
Document type: Act
Section: 303
Title: Bharatiya Nyaya Sanhita, 2023

Text:

303. Theft.—(1) Whoever, intending to take dishonestly any
movable property out of the possession of any person without
that person's consent, moves that property in order to such
taking, is said to commit theft.

(2) Whoever commits theft shall be punished with imprisonment
of either description for a term which may extend to three
years, or with fine, or with both and in case of second or
subsequent conviction of any person under this section, he shall
be punished with rigorous imprisonment for a term which shall
not be less than one year but which may extend to five years
and with fine:

Provided that in cases of theft where the value of the stolen
property is less than five thousand rupees, and a person is
convicted for the first time, upon return of the value of
property or restoration of the stolen property, the person
shall be punished with community service.
"""

    print()
    print(
        "Running multilingual LLM tests..."
    )

    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    _test_question(
        "What is the punishment for theft?",
        sample_context
    )

    # --------------------------------------------------------
    # Hindi
    # --------------------------------------------------------

    _test_question(
        "चोरी की सजा क्या है?",
        sample_context
    )

    # --------------------------------------------------------
    # Kannada
    # --------------------------------------------------------

    _test_question(
        "ಕಳ್ಳತನಕ್ಕೆ ಶಿಕ್ಷೆ ಏನು?",
        sample_context
    )