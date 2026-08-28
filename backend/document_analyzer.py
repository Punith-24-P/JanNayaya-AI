"""
JanNyaya AI - Legal Document Analyzer

Analyzes text extracted from:
- PDF
- scanned PDF
- image
- camera photo

Produces:
- document type
- subject
- summary
- important facts
- dates
- amounts
- legal references
- possible legal issues
- possible next steps
- warnings

The analyzer must use only the supplied document text.
"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PROJECT / ENVIRONMENT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

ENV_FILE = os.path.join(
    PROJECT_ROOT,
    ".env",
)

load_dotenv(
    dotenv_path=ENV_FILE,
)


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    print("Notice: GROQ_API_KEY is not set in environment. Document analysis will require API key.")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

DOCUMENT_ANALYSIS_MAX_TOKENS = int(
    os.getenv(
        "DOCUMENT_ANALYSIS_MAX_TOKENS",
        "2500",
    )
)

DOCUMENT_ANALYSIS_MAX_CHARS = int(
    os.getenv(
        "DOCUMENT_ANALYSIS_MAX_CHARS",
        "25000",
    )
)


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY or "gsk_placeholder",
    timeout=60.0,
)


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(
    text: str,
) -> str:

    if not text:
        return "english"

    hindi_count = 0
    kannada_count = 0

    for character in str(text):

        code = ord(
            character
        )

        # Devanagari
        if 0x0900 <= code <= 0x097F:
            hindi_count += 1

        # Kannada
        elif 0x0C80 <= code <= 0x0CFF:
            kannada_count += 1

    if hindi_count > 0:
        return "hindi"

    if kannada_count > 0:
        return "kannada"

    return "english"


def language_name(
    language: str,
) -> str:

    return {
        "english": "English",
        "hindi": "Hindi",
        "kannada": "Kannada",
    }.get(
        language,
        "English",
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are JanNyaya AI's legal document analysis assistant.

You receive text extracted from a legal document.

Your task is to understand ONLY that supplied document
and produce structured information.

RULES:

1. Use ONLY the supplied document text.
2. Never invent facts.
3. Never invent dates.
4. Never invent amounts.
5. Never invent legal sections.
6. Never invent names.
7. Never invent deadlines.
8. Never assume facts that are not written.
9. Do not decide guilt or innocence.
10. Do not provide personalized legal advice.
11. Clearly distinguish document facts from cautious next steps.
12. If information is missing, return an empty value.
13. Preserve important conditions and warnings.
14. Answer in the same language as the document.

Return ONLY valid JSON.

JSON structure:

{
  "document_type": "",
  "title_or_subject": "",
  "summary": "",
  "important_facts": [],
  "dates": [],
  "amounts": [],
  "legal_sections": [],
  "legal_issues": [],
  "possible_next_steps": [],
  "warnings": []
}

document_type examples:
- Legal Notice
- Court Order
- Judgment
- Agreement
- Act
- Complaint
- Application
- Petition
- Affidavit
- Contract
- Other

IMPORTANT:

important_facts:
Only facts explicitly present in the document.

dates:
Only dates explicitly present in the document.

amounts:
Only monetary amounts explicitly present in the document.

legal_sections:
Only legal sections, Acts, Orders, Rules or statutes
explicitly mentioned in the document.

legal_issues:
Describe issues visible from the document text.
Do not invent legal conclusions.

possible_next_steps:
Keep these cautious and generic.
Examples:
- Review the referenced agreement.
- Preserve the original document.
- Check payment records.
- Note any stated deadline.
- Consult a qualified lawyer if necessary.

warnings:
Mention explicit demands, deadlines, threats of proceedings,
missing information, or other important warnings visible
in the document.
"""


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    text: str,
) -> str:

    language = detect_language(
        text
    )

    return f"""
{SYSTEM_PROMPT}

DOCUMENT LANGUAGE:
{language_name(language)}

DOCUMENT TEXT:
============================================================
{text}
============================================================

Analyze the document.

Return ONLY valid JSON.
"""


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(
    content: str,
) -> Dict[str, Any]:

    if not content:
        raise ValueError(
            "Empty response received from document analyzer."
        )

    content = content.strip()

    # Remove markdown code fences if model added them.
    if content.startswith("```"):

        lines = content.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        content = "\n".join(
            lines
        ).strip()

    try:

        data = json.loads(
            content
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Invalid JSON from document analyzer: {error}"
        )

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "Document analyzer response is not a JSON object."
        )

    return data


# ============================================================
# NORMALIZE RESULT
# ============================================================

def normalize_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    normalized = {

        "document_type":
            str(
                result.get(
                    "document_type",
                    "",
                ) or ""
            ).strip(),

        "title_or_subject":
            str(
                result.get(
                    "title_or_subject",
                    "",
                ) or ""
            ).strip(),

        "summary":
            str(
                result.get(
                    "summary",
                    "",
                ) or ""
            ).strip(),
    }

    list_fields = [
        "important_facts",
        "dates",
        "amounts",
        "legal_sections",
        "legal_issues",
        "possible_next_steps",
        "warnings",
    ]

    for field in list_fields:

        value = result.get(
            field,
            [],
        )

        if not isinstance(
            value,
            list,
        ):
            value = [value]

        cleaned = []

        for item in value:

            item_text = str(
                item
            ).strip()

            if item_text:
                cleaned.append(
                    item_text
                )

        normalized[field] = cleaned

    return normalized


# ============================================================
# FALLBACK
# ============================================================

def fallback_analysis(
    text: str,
) -> Dict[str, Any]:

    language = detect_language(
        text
    )

    if language == "hindi":

        summary = (
            "दस्तावेज़ का पाठ सफलतापूर्वक निकाला गया है, "
            "लेकिन विस्तृत AI विश्लेषण इस समय उपलब्ध नहीं है।"
        )

        next_steps = [
            "मूल दस्तावेज़ सुरक्षित रखें।",
            "दस्तावेज़ में दिए गए समय-सीमा या निर्देशों की समीक्षा करें।",
            "आवश्यक होने पर योग्य वकील से सलाह लें।",
        ]

    elif language == "kannada":

        summary = (
            "ದಾಖಲೆಯ ಪಠ್ಯವನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಹೊರತೆಗೆಯಲಾಗಿದೆ, "
            "ಆದರೆ ವಿವರವಾದ AI ವಿಶ್ಲೇಷಣೆ ಪ್ರಸ್ತುತ ಲಭ್ಯವಿಲ್ಲ."
        )

        next_steps = [
            "ಮೂಲ ದಾಖಲೆಯನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಇಟ್ಟುಕೊಳ್ಳಿ.",
            "ದಾಖಲೆಯಲ್ಲಿ ನೀಡಿರುವ ಗಡುವು ಅಥವಾ ಸೂಚನೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
            "ಅಗತ್ಯವಿದ್ದರೆ ಅರ್ಹ ವಕೀಲರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        ]

    else:

        summary = (
            "The document text was extracted successfully, "
            "but detailed AI analysis is temporarily unavailable."
        )

        next_steps = [
            "Preserve the original document.",
            "Review any deadline or instruction stated in the document.",
            "Consult a qualified lawyer when case-specific advice is required.",
        ]

    return {

        "document_type":
            "Unknown",

        "title_or_subject":
            "",

        "summary":
            summary,

        "important_facts":
            [],

        "dates":
            [],

        "amounts":
            [],

        "legal_sections":
            [],

        "legal_issues":
            [],

        "possible_next_steps":
            next_steps,

        "warnings":
            [],
    }


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_document(
    text: str,
) -> Dict[str, Any]:

    if not text or not text.strip():

        return {
            "status": "error",
            "message": (
                "No readable document text was available."
            ),
            "analysis":
                fallback_analysis(""),
        }

    text = text.strip()

    # Avoid sending excessively large OCR output.
    if len(text) > DOCUMENT_ANALYSIS_MAX_CHARS:

        text = text[
            :DOCUMENT_ANALYSIS_MAX_CHARS
        ]

    prompt = build_prompt(
        text
    )

    try:

        print()
        print(
            "=" * 70
        )
        print(
            "JAN NYAYA AI - DOCUMENT ANALYZER"
        )
        print(
            "=" * 70
        )

        print(
            "Document characters:",
            len(text),
        )

        print(
            "Calling Groq document analyzer..."
        )

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },
                {
                    "role":
                        "user",

                    "content":
                        prompt,
                },
            ],

            temperature=0.0,

            max_tokens=DOCUMENT_ANALYSIS_MAX_TOKENS,
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        result = parse_json_response(
            content
        )

        result = normalize_result(
            result
        )

        print(
            "Document analysis successful."
        )

        return {
            "status":
                "success",

            "analysis":
                result,
        }

    except Exception as error:

        print()
        print(
            "=" * 70
        )
        print(
            "DOCUMENT ANALYZER ERROR"
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
            "Using fallback analysis."
        )
        print(
            "=" * 70
        )

        return {
            "status":
                "fallback",

            "analysis":
                fallback_analysis(
                    text
                ),

            "error":
                str(error),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_text = """
    IN THE COURT OF THE CIVIL JUDGE
    BANGALORE, KARNATAKA

    Case No: O.S. 245/2024
    Date: 12/04/2024

    LEGAL NOTICE
    (Order 5 Rule 1, Code of Civil Procedure, 1908)

    Legal Notice for Recovery of Outstanding Amount

    Our client had advanced a loan amount of
    ₹2,50,000/- to you on 15/06/2022.

    As per the agreement, you were liable to repay the
    amount with interest at 12% p.a. in 24 monthly instalments.

    There is an outstanding amount of ₹1,87,560/-.

    You are called upon to pay the outstanding amount of
    ₹1,87,560/- along with further interest at 12% p.a.
    within 15 days from the date of receipt of this notice,
    failing which our client will initiate appropriate legal
    proceedings against you.
    """

    result = analyze_document(
        sample_text
    )

    print()
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )