import re


# ---------------------------------------------------------
# LEGAL SECTION DETECTION
# ---------------------------------------------------------

SECTION_PATTERN = re.compile(
    r"(?m)^(?:\s*)(\d{1,4})\.\s+([^\n]+)"
)


def _split_large_section(
    section_text: str,
    chunk_size: int,
    chunk_overlap: int
) -> list[str]:
    """
    Split a legal section that is larger than chunk_size.

    Tries to split at paragraphs, sub-sections and sentences
    while preserving the beginning of the legal provision.
    """

    if len(section_text) <= chunk_size:
        return [section_text.strip()]

    parts = re.split(
        r"\n\s*\n|(?<=\.)\s+(?=\(\d+\))|(?<=\.)\s+(?=\([a-z]\))",
        section_text
    )

    chunks = []
    current = ""

    for part in parts:
        part = part.strip()

        if not part:
            continue

        if len(current) + len(part) + 2 <= chunk_size:
            if current:
                current += "\n\n" + part
            else:
                current = part

        else:
            if current:
                chunks.append(current.strip())

            if len(part) <= chunk_size:
                current = part
            else:
                # Extremely long paragraph/sentence
                start = 0

                while start < len(part):
                    end = start + chunk_size
                    piece = part[start:end].strip()

                    if piece:
                        chunks.append(piece)

                    start = end

                current = ""

    if current:
        chunks.append(current.strip())

    # Add controlled overlap
    if chunk_overlap > 0 and len(chunks) > 1:

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):

            previous = chunks[i - 1]

            overlap = previous[-chunk_overlap:].strip()

            if overlap:
                combined = (
                    overlap
                    + "\n\n"
                    + chunks[i]
                )
            else:
                combined = chunks[i]

            overlapped.append(combined)

        chunks = overlapped

    return chunks


# ---------------------------------------------------------
# MAIN LEGAL CHUNKER
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200
) -> list[str]:
    """
    Legal-aware text chunker.

    Designed for Indian legal documents such as:

        303. Theft.—
        (1) ...
        (2) ...
        Provided that ...

    The chunker attempts to keep each legal section together
    instead of blindly splitting every N characters.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    # -----------------------------------------------------
    # 1. Normalize whitespace
    # -----------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = text.strip()

    if not text:
        return []

    # -----------------------------------------------------
    # 2. Find legal sections
    # -----------------------------------------------------

    matches = list(
        SECTION_PATTERN.finditer(text)
    )

    # -----------------------------------------------------
    # 3. If no sections are detected,
    #    fall back to normal chunking
    # -----------------------------------------------------

    if not matches:
        return _fallback_chunk(
            text,
            chunk_size,
            chunk_overlap
        )

    sections = []

    # Text before first section
    prefix = text[:matches[0].start()].strip()

    if prefix:
        sections.append({
            "number": None,
            "title": None,
            "text": prefix
        })

    # -----------------------------------------------------
    # 4. Extract each legal section
    # -----------------------------------------------------

    for i, match in enumerate(matches):

        section_number = match.group(1)
        section_title = match.group(2).strip()

        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        sections.append({
            "number": section_number,
            "title": section_title,
            "text": section_text
        })

    # -----------------------------------------------------
    # 5. Create chunks
    # -----------------------------------------------------

    final_chunks = []

    for section in sections:

        section_text = section["text"]

        if not section_text:
            continue

        # Keep complete legal provision together
        if len(section_text) <= chunk_size:

            final_chunks.append(
                section_text
            )

        else:

            # Large section needs splitting
            split_chunks = _split_large_section(
                section_text,
                chunk_size,
                chunk_overlap
            )

            # Make sure every split chunk retains
            # section number and title.
            if section["number"]:

                header_match = re.match(
                    r"^\s*(\d{1,4}\.\s+[^\n]+)",
                    section_text
                )

                if header_match:
                    header = header_match.group(1).strip()

                    updated_chunks = []

                    for chunk in split_chunks:

                        if not chunk.startswith(
                            section["number"] + "."
                        ):
                            chunk = (
                                header
                                + "\n"
                                + chunk
                            )

                        updated_chunks.append(
                            chunk
                        )

                    split_chunks = updated_chunks

            final_chunks.extend(
                split_chunks
            )

    # -----------------------------------------------------
    # 6. Clean chunks
    # -----------------------------------------------------

    cleaned_chunks = []

    for chunk in final_chunks:

        chunk = re.sub(
            r"[ \t]+",
            " ",
            chunk
        )

        chunk = re.sub(
            r"\n{3,}",
            "\n\n",
            chunk
        )

        chunk = chunk.strip()

        if chunk:
            cleaned_chunks.append(
                chunk
            )

    return cleaned_chunks


# ---------------------------------------------------------
# FALLBACK CHUNKER
# ---------------------------------------------------------

def _fallback_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> list[str]:
    """
    Fallback chunker for documents where legal
    section numbers cannot be detected.
    """

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    chunks = []
    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if (
            len(current)
            + len(paragraph)
            + 2
            <= chunk_size
        ):

            if current:
                current += (
                    "\n\n"
                    + paragraph
                )
            else:
                current = paragraph

        else:

            if current:
                chunks.append(
                    current.strip()
                )

            if len(paragraph) <= chunk_size:

                current = paragraph

            else:

                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    paragraph
                )

                current = ""

                for sentence in sentences:

                    sentence = sentence.strip()

                    if not sentence:
                        continue

                    if (
                        len(current)
                        + len(sentence)
                        + 1
                        <= chunk_size
                    ):

                        if current:
                            current += (
                                " "
                                + sentence
                            )
                        else:
                            current = sentence

                    else:

                        if current:
                            chunks.append(
                                current.strip()
                            )

                        current = sentence

    if current:
        chunks.append(
            current.strip()
        )

    # -----------------------------------------------------
    # Controlled overlap
    # -----------------------------------------------------

    if (
        chunk_overlap > 0
        and len(chunks) > 1
    ):

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):

            previous = chunks[i - 1]

            overlap = previous[
                -chunk_overlap:
            ].strip()

            if overlap:

                combined = (
                    overlap
                    + "\n\n"
                    + chunks[i]
                )

            else:

                combined = chunks[i]

            overlapped.append(
                combined
            )

        chunks = overlapped

    return chunks