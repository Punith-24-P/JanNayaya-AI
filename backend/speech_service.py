"""
JanNyaya AI - Multilingual Speech-to-Text Service

Supports:
    English
    Hindi
    Kannada

The frontend records microphone audio and sends it here.
Groq Whisper converts the audio into text.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PROJECT / ENVIRONMENT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

ENV_FILE = (
    PROJECT_ROOT / ".env"
)

load_dotenv(
    dotenv_path=ENV_FILE
)


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY environment variable is not set. Speech calls will require an API key.")


SPEECH_MODEL = os.getenv(
    "SPEECH_MODEL",
    "whisper-large-v3",
)


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY or "gsk_placeholder",
    timeout=120.0,
)


# ============================================================
# LANGUAGE
# ============================================================

def normalize_language(
    language: Optional[str],
) -> Optional[str]:
    """
    Convert frontend language names to
    Whisper ISO-639-1 language codes.

    Returns:
        en
        hi
        kn
        None for auto
    """

    if not language:
        return None

    value = (
        str(language)
        .strip()
        .lower()
    )

    mapping = {
        "auto": None,

        "english": "en",
        "en": "en",
        "en-in": "en",

        "hindi": "hi",
        "hi": "hi",
        "hi-in": "hi",

        "kannada": "kn",
        "kn": "kn",
        "kn-in": "kn",
    }

    return mapping.get(
        value
    )


# ============================================================
# SAFE AUDIO EXTENSION
# ============================================================

def get_audio_suffix(
    filename: Optional[str],
    content_type: Optional[str],
) -> str:

    allowed = {
        ".webm",
        ".wav",
        ".mp3",
        ".mp4",
        ".m4a",
        ".ogg",
        ".mpeg",
        ".mpga",
        ".flac",
    }

    if filename:

        suffix = (
            Path(filename)
            .suffix
            .lower()
        )

        if suffix in allowed:
            return suffix

    content_type = (
        str(content_type or "")
        .lower()
        .split(";")[0]
        .strip()
    )

    content_mapping = {
        "audio/webm": ".webm",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".mp4",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/ogg": ".ogg",
        "audio/flac": ".flac",
        "audio/aac": ".aac",
    }

    return content_mapping.get(
        content_type,
        ".webm",
    )


# ============================================================
# LEGAL PROMPT
# ============================================================

def build_speech_prompt(
    language_code: Optional[str],
) -> str:

    if language_code == "kn":

        return (
            "Indian legal query in Kannada language. "
            "Accurately transcribe spoken Kannada words and legal terms: "
            "ಕಳ್ಳತನ, ಶಿಕ್ಷೆ, ಕಾನೂನು, ಧಾರಾ, ಸೆಕ್ಷನ್, ಭಾರತೀಯ ನ್ಯಾಯ ಸಂಹಿತೆ, ಬಿಎನ್‌ಎಸ್, "
            "ಚೆಕ್ ಬೌನ್ಸ್, ಸಾಲ ವಸೂಲಾತಿ, ಬ್ಯಾಂಕ್ ನೋಟಿಸ್, ಸರ್ಫೇಸಿ, ಗ್ರಾಹಕ ದೂರು, "
            "ಸೈಬರ್ ವಂಚನೆ, ಪೊಲೀಸ್ ಠಾಣೆ, ಎಫ್‌ಐಆರ್, ಜಾಮೀನು, ನ್ಯಾಯಾಲಯ, ಪರಿಹಾರ."
        )

    if language_code == "hi":

        return (
            "Indian legal query in Hindi language. "
            "Accurately transcribe spoken Hindi words and legal terms: "
            "चोरी, सजा, कानून, धारा, भारतीय न्याय संहिता, बीएनएस, "
            "चेक बाउंस, धारा 138, ऋण वसूली, बैंक नोटिस, सरफेसी एक्ट, "
            "उपभोक्ता शिकायत, साइबर अपराध, पुलिस थाना, एफआईआर, जमानत, न्यायालय, मुआवजा."
        )

    return (
        "Indian legal query in English. "
        "Preserve exact statutory sections and Indian legal terms: "
        "BNS 2023, BNSS, BSA, IPC, CrPC, Section 138 Negotiable Instruments Act, "
        "SARFAESI Act, Debt Recovery Tribunal DRT, Consumer Protection Act, "
        "FIR, bail, criminal complaint, legal notice, breach of contract, cyber fraud."
    )


# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe_audio(
    audio_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """
    Convert recorded microphone audio to text.
    """

    if not audio_bytes:

        raise ValueError(
            "Audio file is empty."
        )

    language_code = (
        normalize_language(
            language
        )
    )

    suffix = get_audio_suffix(
        filename,
        content_type,
    )

    temporary_path = None

    try:

        # ----------------------------------------------------
        # Create temporary audio file
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            delete=False,
        ) as temp_file:

            temp_file.write(
                audio_bytes
            )

            temporary_path = (
                temp_file.name
            )

        print()
        print("=" * 70)
        print(
            "JAN NYAYA AI - SPEECH TO TEXT"
        )
        print("=" * 70)

        print(
            "Model:",
            SPEECH_MODEL,
        )

        print(
            "Requested language:",
            language or "auto",
        )

        print(
            "Detected ISO language:",
            language_code or "auto",
        )

        print(
            "Audio format:",
            suffix,
        )

        # ----------------------------------------------------
        # Send to Groq
        # ----------------------------------------------------

        with open(
            temporary_path,
            "rb",
        ) as audio_file:

            request_args = {
                "file": (
                    filename
                    or f"voice{suffix}",
                    audio_file.read(),
                ),
                "model": SPEECH_MODEL,
                "response_format": "json",
                "temperature": 0.0,
                "prompt": build_speech_prompt(
                    language_code
                ),
            }

            # Do not send language for AUTO.
            if language_code:

                request_args[
                    "language"
                ] = language_code

            transcription = (
                client
                .audio
                .transcriptions
                .create(
                    **request_args
                )
            )

        # ----------------------------------------------------
        # Read result
        # ----------------------------------------------------

        transcript = str(
            getattr(
                transcription,
                "text",
                "",
            )
            or ""
        ).strip()

        if not transcript:

            raise RuntimeError(
                "Whisper returned an empty transcript."
            )

        print(
            "Transcript:",
            transcript,
        )

        print(
            "Speech recognition successful."
        )

        print("=" * 70)

        return {
            "status": "success",
            "text": transcript,
            "language": (
                language_code
                or "auto"
            ),
            "model": SPEECH_MODEL,
        }

    finally:

        # ----------------------------------------------------
        # Remove temporary file
        # ----------------------------------------------------

        if temporary_path:

            try:
                os.remove(
                    temporary_path
                )
            except OSError:
                pass


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "JanNyaya AI - Speech Service"
    )

    print(
        "Model:",
        SPEECH_MODEL,
    )

    print(
        "Speech service loaded successfully."
    )