from backend.rag_service import answer_question


QUESTIONS = [

    "What is theft?",

    "What is the punishment for theft?",

    "What is murder?",

    "What is the punishment for murder?",

    "What is snatching?",

    "What is the punishment for snatching?",

    "What is rape?",

    "What is the punishment for rape?",

    "What is cheating?",

    "What is the punishment for cheating?",
]


# ============================================================
# TEST RAG
# ============================================================

if __name__ == "__main__":

    for question in QUESTIONS:

        print(
            "\n" + "=" * 80
        )

        print(
            "QUESTION:",
            question
        )

        print(
            "=" * 80
        )

        try:

            result = answer_question(
                question
            )

            print(
                "\nANSWER:\n"
            )

            print(
                result["answer"]
            )

            print(
                "\nSOURCES:\n"
            )

            for source in result[
                "sources"
            ]:

                print(
                    source
                )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                str(error)
            )