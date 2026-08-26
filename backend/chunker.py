"""
JanNyaya AI - Legal-Aware Chunker

This chunker is designed for Indian legal documents.

Goals:
1. Preserve legal section boundaries.
2. Preserve the section heading in every chunk.
3. Keep overlapping chunks for semantic retrieval.
4. Prevent section metadata from disappearing when a
   legal section is split across multiple chunks.
"""

import re
from typing import List, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


# ============================================================
# NORMALIZE TEXT
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize spaces/tabs.
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Avoid excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# SECTION HEADER DETECTION
# ============================================================

SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(\d{1,4})\.\s+(.+?)\s*$"
)


def _is_section_header(line: str) -> bool:
    """
    Return True when a line looks like a legal section heading.

    Examples:

        303. Theft.—
        318. Cheating.—
        101. Murder.—
    """

    if not line:
        return False

    return bool(
        SECTION_HEADER_PATTERN.match(
            line.strip()
        )
    )


def _extract_section_header(
    line: str
) -> Tuple[str, str]:
    """
    Extract section number and title.

    Returns:
        ("303", "Theft.—")
    """

    match = SECTION_HEADER_PATTERN.match(
        line.strip()
    )

    if not match:
        return "", ""

    section_number = (
        match.group(1).strip()
    )

    section_title = (
        match.group(2).strip()
    )

    return (
        section_number,
        section_title
    )


# ============================================================
# SPLIT INTO LEGAL SECTIONS
# ============================================================

def _split_into_sections(
    text: str
) -> List[Tuple[str, str, str]]:
    """
    Split a legal document into:

        section_number
        section_title
        section_text

    Every section keeps its original heading.

    Content before the first detected section is placed in
    section_number = "".
    """

    lines = text.splitlines()

    sections = []

    current_section_number = ""
    current_section_title = ""
    current_lines = []

    def flush_current() -> None:

        if not current_lines:
            return

        content = "\n".join(
            current_lines
        ).strip()

        if not content:
            return

        sections.append(
            (
                current_section_number,
                current_section_title,
                content
            )
        )

    for line in lines:

        stripped = line.strip()

        if not stripped:
            current_lines.append("")
            continue

        if _is_section_header(stripped):

            # Save previous section.
            flush_current()

            # Start new section.
            (
                current_section_number,
                current_section_title
            ) = _extract_section_header(
                stripped
            )

            current_lines = [
                stripped
            ]

        else:

            current_lines.append(
                stripped
            )

    flush_current()

    return sections


# ============================================================
# SPLIT TEXT BY WORD/CHARACTER WINDOW
# ============================================================

def _split_large_body(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Split a large section body while trying to stop at
    sentence/paragraph boundaries.
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        if end < text_length:

            search_start = max(
                start,
                end - 200
            )

            candidates = [
                text.rfind(
                    ". ",
                    search_start,
                    end
                ),
                text.rfind(
                    "; ",
                    search_start,
                    end
                ),
                text.rfind(
                    ": ",
                    search_start,
                    end
                ),
                text.rfind(
                    "\n",
                    search_start,
                    end
                ),
                text.rfind(
                    "।",
                    search_start,
                    end
                ),
            ]

            boundary = max(
                candidates
            )

            if boundary > start:

                end = boundary + 1

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        if end >= text_length:
            break

        next_start = (
            end - chunk_overlap
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# CHUNK LEGAL SECTION
# ============================================================

def _chunk_section(
    section_number: str,
    section_title: str,
    section_text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Chunk one legal section.

    IMPORTANT:
    Every chunk keeps the legal section heading.

    Example:

        303. Theft.—

    This prevents metadata and legal context from being lost.
    """

    section_text = section_text.strip()

    if not section_text:
        return []

    heading = ""

    if section_number:

        heading = (
            f"{section_number}. "
            f"{section_title}".strip()
        )

    # If the section already contains the heading as its first
    # line, remove it before splitting the body.
    body_lines = section_text.splitlines()

    if (
        body_lines
        and heading
        and body_lines[0].strip() == heading
    ):

        body_lines = body_lines[1:]

    body = "\n".join(
        body_lines
    ).strip()

    # --------------------------------------------------------
    # If the heading itself is almost the entire section.
    # --------------------------------------------------------

    if not body:

        return [
            heading
            if heading
            else section_text
        ]

    # --------------------------------------------------------
    # Reserve space for heading.
    # --------------------------------------------------------

    reserved_heading = (
        len(heading) + 2
        if heading
        else 0
    )

    effective_chunk_size = max(
        200,
        chunk_size - reserved_heading
    )

    effective_overlap = min(
        chunk_overlap,
        max(
            0,
            effective_chunk_size // 3
        )
    )

    body_chunks = _split_large_body(
        body,
        effective_chunk_size,
        effective_overlap
    )

    final_chunks = []

    for body_chunk in body_chunks:

        if heading:

            final_chunk = (
                heading
                + "\n\n"
                + body_chunk
            )

        else:

            final_chunk = body_chunk

        final_chunks.append(
            final_chunk.strip()
        )

    return final_chunks


# ============================================================
# FALLBACK PARAGRAPH CHUNKING
# ============================================================

def _fallback_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Fallback for introductory material before the first
    legal section.
    """

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        if not current:

            if len(paragraph) <= chunk_size:

                current = paragraph

            else:

                large_chunks = _split_large_body(
                    paragraph,
                    chunk_size,
                    chunk_overlap
                )

                chunks.extend(
                    large_chunks
                )

        elif (
            len(current)
            + 2
            + len(paragraph)
            <= chunk_size
        ):

            current = (
                current
                + "\n\n"
                + paragraph
            )

        else:

            chunks.append(
                current.strip()
            )

            if chunk_overlap > 0:

                overlap = current[
                    -chunk_overlap:
                ]

                current = (
                    overlap
                    + "\n\n"
                    + paragraph
                )

                if len(current) > chunk_size:

                    large_chunks = _split_large_body(
                        current,
                        chunk_size,
                        chunk_overlap
                    )

                    chunks.extend(
                        large_chunks[:-1]
                    )

                    current = (
                        large_chunks[-1]
                        if large_chunks
                        else ""
                    )

            else:

                current = paragraph

    if current.strip():

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# MAIN CHUNK FUNCTION
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Legal-aware text chunker.

    The main improvement is that section headings are
    propagated into every chunk belonging to that section.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    normalized_text = _normalize_text(
        text
    )

    if not normalized_text:
        return []

    sections = _split_into_sections(
        normalized_text
    )

    # If no legal sections were detected,
    # use normal fallback chunking.
    if not sections:
        return _fallback_chunk(
            normalized_text,
            chunk_size,
            chunk_overlap
        )

    final_chunks = []

    for (
        section_number,
        section_title,
        section_text
    ) in sections:

        # ----------------------------------------------------
        # Introductory content before first section
        # ----------------------------------------------------

        if not section_number:

            intro_chunks = _fallback_chunk(
                section_text,
                chunk_size,
                chunk_overlap
            )

            final_chunks.extend(
                intro_chunks
            )

            continue

        # ----------------------------------------------------
        # Legal section
        # ----------------------------------------------------

        section_chunks = _chunk_section(
            section_number,
            section_title,
            section_text,
            chunk_size,
            chunk_overlap
        )

        final_chunks.extend(
            section_chunks
        )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    cleaned_chunks = []

    for chunk in final_chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        # Avoid excessive blank lines.
        chunk = re.sub(
            r"\n{3,}",
            "\n\n",
            chunk
        )

        cleaned_chunks.append(
            chunk
        )

    return cleaned_chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample = """
    101. Murder.—
    
    Except in the cases hereinafter excepted, culpable homicide
    is murder.
    
    Explanation one.
    
    102. Punishment.—
    
    Whoever commits the offence shall be punished.
    """

    result = chunk_text(
        sample,
        chunk_size=100,
        chunk_overlap=20
    )

    print(
        "Chunks:",
        len(result)
    )

    for index, chunk in enumerate(
        result,
        start=1
    ):

        print()
        print(
            f"--- CHUNK {index} ---"
        )
        print(chunk)