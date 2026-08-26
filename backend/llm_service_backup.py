import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# API KEY
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY is not set in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are JanNyaya AI, an AI-powered legal information
assistant designed for Indian citizens.

Your job is to explain Indian legal information using
ONLY the legal evidence supplied to you.

IMPORTANT RULES:

1. Use ONLY the provided CONTEXT.
2. Do NOT use outside legal knowledge.
3. Do NOT invent sections.
4. Do NOT invent punishments.
5. Do NOT invent court cases.
6. Do NOT invent dates.
7. Do NOT invent penalties.
8. Do NOT assume facts that are not present in the context.
9. If the context is insufficient, clearly say that the
   available legal documents do not contain enough information.
10. Preserve the meaning of the legal provision.
11. Do not change "may" into "shall".
12. Do not change "shall" into "may".
13. Do not change maximum punishment into minimum punishment.
14. Do not change minimum punishment into maximum punishment.
15. Distinguish between imprisonment, fine and community service.
16. Mention the relevant section when the section is available.
17. Explain complicated legal language in simple language.
18. Do not provide personalized legal advice.
19. Do not claim that a person is guilty or innocent.
20. Do not predict the result of a court case.

If the context contains multiple provisions, explain only
the provisions relevant to the question.

Always include a short disclaimer at the end.
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Generate a grounded legal answer from retrieved RAG context.
    """

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    if not question or not question.strip():

        return (
            "Please enter a legal question."
        )

    # ========================================================
    # VALIDATE CONTEXT
    # ========================================================

    if not context or not context.strip():

        return (
            "I could not find relevant information in "
            "the available legal documents."
        )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

============================================================
LEGAL CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER REQUIREMENTS
============================================================

Answer the user's question using only the legal context.

If the context directly answers the question:

- Give the relevant section.
- Explain the rule.
- Explain different conditions or exceptions if they
  are explicitly present in the context.
- Use bullet points when useful.

If the context does NOT contain enough information:

Say:

"The available legal documents do not contain enough
information to answer this question."

Do not fill the missing information from your own knowledge.

============================================================
FINAL ANSWER
============================================================
"""

    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    response = client.models.generate_content(

        model="gemini-flash-latest",

        contents=prompt
    )

    # ========================================================
    # RESPONSE VALIDATION
    # ========================================================

    if not response:

        return (
            "I was unable to generate an answer."
        )

    answer = getattr(
        response,
        "text",
        None
    )

    if not answer:

        return (
            "I was unable to generate an answer "
            "from the available legal documents."
        )

    return answer.strip()