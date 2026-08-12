import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(question: str, context: str) -> str:
    """
    Generate a simplified answer using Gemini
    based only on the retrieved RAG context.
    """

    if not context:
        return "I could not find relevant information in the available legal documents."

    prompt = f"""
You are Jan Nyaya AI, an AI-powered legal assistance system for Indian citizens.

Answer the user's question using ONLY the information provided in the context below.

Rules:
- Do not invent facts or laws.
- Do not make up legal sections, cases, dates, or penalties.
- Explain the information in simple language.
- If the context does not contain enough information, clearly say so.
- This is an informational legal assistant, not a replacement for a qualified lawyer.
- Keep the answer clear and useful.

CONTEXT:
{context}

USER QUESTION:
{question}

Provide a clear and simplified answer:
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    return response.text.strip()
