from typing import List, Dict, Any

from backend.retriever import (
    hybrid_search
)

from backend.llm_service import (
    generate_answer
)


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results: List[Dict[str, Any]],
    max_characters: int = 12000
) -> str:
    """
    Convert retrieved documents into grounded LLM context.
    """

    if not results:

        return ""

    context_parts = []

    current_length = 0

    for index, result in enumerate(
        results,
        start=1
    ):

        document = result.get(
            "document",
            ""
        )

        metadata = result.get(
            "metadata",
            {}
        )

        if not document:
            continue

        section = metadata.get(
            "section_number",
            metadata.get(
                "section",
                ""
            )
        )

        title = metadata.get(
            "title",
            ""
        )

        source = metadata.get(
            "source",
            "Unknown source"
        )

        chunk_index = metadata.get(
            "chunk_index",
            ""
        )

        header = (
            f"[Evidence {index}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Chunk: {chunk_index}\n"
        )

        if title:

            header += (
                f"Title: {title}\n"
            )

        evidence = (
            header
            + "Text:\n"
            + document
        )

        # ----------------------------------------------------
        # Respect context limit
        # ----------------------------------------------------

        if (
            current_length
            + len(evidence)
            > max_characters
        ):

            remaining = (
                max_characters
                - current_length
            )

            if remaining > 300:

                evidence = evidence[
                    :remaining
                ]

                context_parts.append(
                    evidence
                )

            break

        context_parts.append(
            evidence
        )

        current_length += len(
            evidence
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    question: str
) -> Dict[str, Any]:
    """
    Complete JanNyaya RAG pipeline.

    Question
       ↓
    Hybrid retrieval
       ↓
    Context construction
       ↓
    Gemini
       ↓
    Grounded answer
    """

    if not question or not question.strip():

        return {
            "question":
                question,

            "answer":
                "Please enter a legal question.",

            "sources":
                []
        }

    print(
        "\nSearching legal knowledge base..."
    )

    # ========================================================
    # RETRIEVE
    # ========================================================

    results = hybrid_search(

        question,

        semantic_k=10,

        bm25_k=10,

        final_k=5
    )

    if not results:

        return {

            "question":
                question,

            "answer":
                (
                    "I could not find relevant "
                    "information in the available "
                    "legal documents."
                ),

            "sources":
                []
        }

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = build_context(
        results
    )

    if not context:

        return {

            "question":
                question,

            "answer":
                (
                    "I could not find enough relevant "
                    "information in the available "
                    "legal documents."
                ),

            "sources":
                []
        }

    # ========================================================
    # GEMINI
    # ========================================================

    print(
        "\nGenerating grounded answer..."
    )

    answer = generate_answer(
        question,
        context
    )

    # ========================================================
    # SOURCES
    # ========================================================

    sources = []

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        sources.append(
            {
                "section":
                    metadata.get(
                        "section_number",
                        metadata.get(
                            "section",
                            None
                        )
                    ),

                "source":
                    metadata.get(
                        "source",
                        "Unknown"
                    ),

                "chunk":
                    metadata.get(
                        "chunk_index",
                        None
                    ),

                "score":
                    round(
                        result.get(
                            "hybrid_score",
                            0
                        ),
                        4
                    )
            }
        )

    return {

        "question":
            question,

        "answer":
            answer,

        "sources":
            sources
    }


# ============================================================
# INTERACTIVE MODE
# ============================================================

def main():

    print(
        "# JanNyaya AI - RAG Service"
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nExiting JanNyaya AI."
            )

            break

        if not question:

            continue

        try:

            result = answer_question(
                question
            )

            print(
                "\n" + "=" * 70
            )

            print(
                result["answer"]
            )

            print(
                "\n" + "=" * 70
            )

            print(
                "\nSources:"
            )

            for index, source in enumerate(
                result["sources"],
                start=1
            ):

                print(
                    f"[{index}] "
                    f"Section "
                    f"{source['section']} | "
                    f"{source['source']} | "
                    f"Chunk "
                    f"{source['chunk']} | "
                    f"Score: "
                    f"{source['score']}"
                )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                str(error)
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()